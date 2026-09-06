"""Real-scale Flow 1 analysis: LAPE + attribution + patching validation across
many language-specific neurons, on a real multilingual corpus. Runs on
whichever Pythia checkpoint is given (70M reuses Lawson et al.'s released
MLSAE; larger sizes need `train_mlsae.py` run first for that size).

Two passes over the corpus, deliberately, to stay memory-safe at large model
widths:
  1. Stream every example once, accumulating LAPE statistics only
     (`LapeAccumulator`), discarding each example's activations immediately.
     This is what identifies the target neurons.
  2. Re-stream, per language, using `run_with_selective_cache` to cache only
     the handful of (layer, neuron) columns that language's target neurons
     actually need, plus the full residual stream (needed for attribution
     and patching). Caching the *full* per-layer neuron activation for every
     example in the corpus (the original one-pass design) is what silently
     OOM'd the whole machine on BLOOM-1.7B (d_mlp=8192, 24 layers, ~16,500
     examples): the full cache alone needs several hundred GB at that width.

Saves per-neuron results to a JSON file so the run is resumable/inspectable
without re-running the (expensive) analysis.
"""

import argparse
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")

import torch

from langflow.attribution import (
    choose_target_layer,
    gather_top_firing_selective,
    gradient_based_attribution,
    jaccard_overlap,
    selection_hit_rate,
    top_latents,
    weight_based_attribution,
)
from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.lape import LapeAccumulator, select_language_neurons
from langflow.mlsae import TopKSAE, load_mlsae
from langflow.models import run_with_cache, run_with_selective_cache
from langflow.patching import patch_latents, patch_neuron
from langflow.pythia import load_pythia

PYTHIA_LANGUAGES = ["en", "fr", "de", "es", "it", "nl", "pt", "ru", "zh", "ja"]
# Matches the language set used for the standalone BLOOM LAPE runs
# (scripts/bloom_lape_only.py), so BLOOM results here are directly comparable.
BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


def analyze_neuron(model, sae, lang_acts, layer, neuron, target_layer, top_n_firing, n_patch_tokens, device):
    resid, neuron_act, pairs = gather_top_firing_selective(lang_acts, layer, neuron, top_n=top_n_firing)
    resid, neuron_act = resid.to(device), neuron_act.to(device)

    w_result = weight_based_attribution(model, sae, resid[layer], neuron_act, layer, neuron, target_layer)
    _, w_candidates = top_latents(w_result, n=100)
    g_result = gradient_based_attribution(
        model, sae, resid[target_layer], neuron_act, layer, neuron, target_layer, candidate_latents=w_candidates
    )
    _, g_top20 = top_latents(g_result, n=20)
    _, w_top20 = top_latents(w_result, n=20)
    overlap = jaccard_overlap(w_top20, g_top20)

    hit_rate = selection_hit_rate(sae, resid[target_layer], w_candidates)
    encoded_indices = sae.encode(resid[target_layer])[0].indices.flatten()
    patchable_mask = torch.isin(w_candidates, encoded_indices)
    patchable = w_candidates[patchable_mask][:20]

    result = {
        "layer": layer,
        "neuron": neuron,
        "target_layer": target_layer,
        "n_gathered_tokens": len(pairs),
        "weight_grad_jaccard_top20": overlap,
        "hit_rate_top100": hit_rate,
        "n_patchable": int(patchable_mask.sum().item()),
    }

    if len(patchable) == 0:
        result["patching"] = None
        return result

    torch.manual_seed(hash((layer, neuron)) % (2**31))
    random_idx = torch.randperm(sae.n_latents, device=device)[: len(patchable)]

    neuron_shifts, attributed_shifts, random_shifts = [], [], []
    for ex_idx, pos in pairs[:n_patch_tokens]:
        acts = lang_acts[ex_idx]
        tokens = acts.tokens.to(device)
        attention_mask = torch.ones_like(tokens)

        neuron_patch = patch_neuron(model, tokens, attention_mask, layer, neuron, value=0.0)
        neuron_shifts.append(
            (neuron_patch.patched_logits[:, pos] - neuron_patch.baseline_logits[:, pos]).abs().mean().item()
        )

        recon_baseline = patch_latents(
            model, sae, tokens, attention_mask, target_layer,
            latent_indices=torch.tensor([], dtype=torch.long, device=device),
        )
        attributed_patch = patch_latents(model, sae, tokens, attention_mask, target_layer, latent_indices=patchable)
        attributed_shifts.append(
            (attributed_patch.patched_logits[:, pos] - recon_baseline.patched_logits[:, pos]).abs().mean().item()
        )

        random_patch = patch_latents(model, sae, tokens, attention_mask, target_layer, latent_indices=random_idx)
        random_shifts.append(
            (random_patch.patched_logits[:, pos] - recon_baseline.patched_logits[:, pos]).abs().mean().item()
        )

    result["patching"] = {
        "neuron_shift_mean": sum(neuron_shifts) / len(neuron_shifts),
        "attributed_shift_mean": sum(attributed_shifts) / len(attributed_shifts),
        "random_shift_mean": sum(random_shifts) / len(random_shifts),
        "n_tokens_tested": len(neuron_shifts),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pythia-70m", "pythia-410m", "bloom-560m", "bloom-1b7"], default="pythia-70m")
    parser.add_argument("--mlsae-checkpoint", default=None, help="Local path to a trained MLSAE state_dict; if omitted, uses Lawson et al.'s released checkpoint (pythia-70m only)")
    parser.add_argument("--mlsae-config", default=None, help="JSON with n_inputs/n_latents/k, required if --mlsae-checkpoint is a local state_dict")
    parser.add_argument("--n-examples-per-lang", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-neurons", type=int, default=30, help="how many top language-specific neurons to fully analyze")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="results/flow1_pythia70m.json")
    args = parser.parse_args()

    device = args.device
    is_bloom = args.model.startswith("bloom")
    languages = BLOOM_LANGUAGES if is_bloom else PYTHIA_LANGUAGES
    print(f"[{args.model}] Loading model on {device}...")
    if args.model == "pythia-70m":
        model, tokenizer = load_pythia("EleutherAI/pythia-70m-deduped", device=device)
    elif args.model == "pythia-410m":
        model, tokenizer = load_pythia("EleutherAI/pythia-410m-deduped", device=device)
    elif args.model == "bloom-560m":
        model, tokenizer = load_bloom("bigscience/bloom-560m", device=device)
    else:
        model, tokenizer = load_bloom("bigscience/bloom-1b7", device=device)

    if args.mlsae_checkpoint is None:
        sae = load_mlsae("tim-lawson/mlsae-pythia-70m-deduped-x64-k32", device=device)
    else:
        config = json.load(open(args.mlsae_config))
        sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
        sae.load_state_dict(torch.load(args.mlsae_checkpoint, map_location=device))
        sae.to(device)
        sae.eval()
    print(f"MLSAE: n_inputs={sae.n_inputs}, n_latents={sae.n_latents}, k={sae.k}")

    print(f"Streaming probe corpus: {languages}, {args.n_examples_per_lang} examples/language...")
    t0 = time.time()
    corpus = load_probe_corpus(languages=languages, n_examples=args.n_examples_per_lang, min_chars=200)
    for lang, texts in corpus.items():
        print(f"  {lang}: {len(texts)} examples")
    print(f"  corpus streamed in {time.time()-t0:.0f}s")

    print("Pass 1/2: streaming through every example, accumulating LAPE statistics only...")
    t0 = time.time()
    lape_acc = LapeAccumulator(languages)
    n_done, total = 0, sum(len(t) for t in corpus.values())
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            acts = run_with_cache(model, tokens, mask)
            lape_acc.update(lang, acts.neurons)
            del acts
            n_done += 1
            if n_done % 2000 == 0:
                print(f"  {n_done}/{total} examples processed, {time.time()-t0:.0f}s elapsed")
    print(f"  LAPE accumulation done in {time.time()-t0:.0f}s")

    scores = lape_acc.finalize()
    language_neurons = select_language_neurons(scores, entropy_percentile=0.01)
    print(f"  {len(language_neurons)} language-specific neurons found (percentile cutoff)")

    n_layers = model.config.num_hidden_layers
    top_neurons = language_neurons[: args.n_neurons]
    neurons_by_lang: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for n in top_neurons:
        neurons_by_lang[n.language].append((n.layer, n.neuron))

    print(
        f"Pass 2/2: re-streaming each language's examples, this time caching only the residual "
        f"stream plus the {len(top_neurons)} target neurons' own activation columns..."
    )
    t0 = time.time()
    acts_by_language = {}
    for lang, texts in corpus.items():
        target_neurons = neurons_by_lang.get(lang, [])
        if not target_neurons:
            continue
        acts_by_language[lang] = []
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            sel = run_with_selective_cache(model, tokens, mask, target_neurons=target_neurons)
            acts_by_language[lang].append(sel.cpu())
    print(f"  selective caching done in {time.time()-t0:.0f}s")

    results = {
        "model": args.model,
        "languages": languages,
        "n_examples_per_lang": args.n_examples_per_lang,
        "n_language_neurons_found": len(language_neurons),
        "lape_top": [
            {"layer": n.layer, "neuron": n.neuron, "language": n.language, "entropy": n.entropy,
             "firing_rate": n.firing_rate_own_language}
            for n in language_neurons[:200]
        ],
        "neuron_analyses": [],
    }

    print(f"Analyzing top {len(top_neurons)} neurons (attribution + patching)...")
    for i, target in enumerate(top_neurons):
        t0 = time.time()
        target_layer, is_last_layer_forced = choose_target_layer(target.layer, n_layers)
        try:
            analysis = analyze_neuron(
                model, sae, acts_by_language[target.language], target.layer, target.neuron,
                target_layer, top_n_firing=40, n_patch_tokens=15, device=device,
            )
            analysis["language"] = target.language
            analysis["entropy"] = target.entropy
            analysis["is_last_layer_forced"] = is_last_layer_forced
            results["neuron_analyses"].append(analysis)
            dt = time.time() - t0
            patching = analysis.get("patching")
            print(
                f"  [{i+1}/{len(top_neurons)}] L{target.layer}N{target.neuron} ({target.language}): "
                f"overlap={analysis['weight_grad_jaccard_top20']:.2f}, "
                f"n_patchable={analysis['n_patchable']}, "
                + (
                    f"attributed/neuron={patching['attributed_shift_mean']:.3f}/{patching['neuron_shift_mean']:.3f}, "
                    f"random={patching['random_shift_mean']:.3f}"
                    if patching
                    else "no patchable latents"
                )
                + f" ({dt:.0f}s)"
            )
        except Exception as e:
            print(f"  [{i+1}/{len(top_neurons)}] L{target.layer}N{target.neuron}: FAILED: {e}")

        # Save incrementally so a long run is inspectable/resumable if interrupted.
        import os

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {args.out}")


if __name__ == "__main__":
    main()
