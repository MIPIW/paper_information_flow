"""Compare our neuron-to-latent attribution against the two baselines named
in flow_1.md/flow_2.md's Experimental Plan: a correlational co-activation
baseline, and a GradSAE-style output-gradient baseline (Shu et al.), which
attributes latents to the model's output rather than to the neuron.

Reuses an existing real-scale results JSON's `lape_top` field to skip
recomputing LAPE from scratch (already validated in that run), then streams
the corpus once more with selective caching for a modest neuron subset
(this is a baseline comparison, not the full pipeline, so a smaller neuron
budget keeps it fast) before comparing all three attribution methods on the
same gathered tokens per neuron, plus a causal-patching check for the
correlational and GradSAE latent sets to see whether either baseline finds
causally real latents at all.
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
    jaccard_overlap,
    top_latents,
    weight_based_attribution,
)
from langflow.baselines import correlational_attribution, gradsae_output_attribution
from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.mlsae import TopKSAE, load_mlsae
from langflow.models import run_with_selective_cache
from langflow.patching import patch_latents
from langflow.pythia import load_pythia

PYTHIA_LANGUAGES = ["en", "fr", "de", "es", "it", "nl", "pt", "ru", "zh", "ja"]
BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pythia-70m", "pythia-410m", "bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--mlsae-checkpoint", default=None)
    parser.add_argument("--mlsae-config", default=None)
    parser.add_argument("--existing-results", required=True, help="Path to an existing flow1_analysis.py results JSON to reuse its lape_top")
    parser.add_argument("--n-neurons", type=int, default=40)
    parser.add_argument("--n-examples-per-lang", type=int, default=1500)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
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

    existing = json.load(open(args.existing_results))
    top_neurons = existing["lape_top"][: args.n_neurons]
    print(f"Reusing {len(top_neurons)} LAPE-selected neurons from {args.existing_results}")

    print(f"Streaming probe corpus: {languages}, {args.n_examples_per_lang} examples/language...")
    t0 = time.time()
    corpus = load_probe_corpus(languages=languages, n_examples=args.n_examples_per_lang, min_chars=200)
    print(f"  corpus streamed in {time.time()-t0:.0f}s")

    n_layers = model.config.num_hidden_layers
    neurons_by_lang: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for n in top_neurons:
        neurons_by_lang[n["language"]].append((n["layer"], n["neuron"]))

    print("Selectively caching the residual stream and target neuron columns...")
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

    print(f"\nComparing attribution methods on {len(top_neurons)} neurons...")
    results = []
    for i, target in enumerate(top_neurons):
        layer, neuron, language = target["layer"], target["neuron"], target["language"]
        target_layer, is_last_layer_forced = choose_target_layer(layer, n_layers)
        lang_acts = acts_by_language[language]

        resid, neuron_act, pairs = gather_top_firing_selective(lang_acts, layer, neuron, top_n=40)
        resid, neuron_act = resid.to(device), neuron_act.to(device)

        w_result = weight_based_attribution(model, sae, resid[layer], neuron_act, layer, neuron, target_layer)
        _, w_top = top_latents(w_result, n=100)

        c_result = correlational_attribution(sae, resid[target_layer], neuron_act, layer, neuron, target_layer)
        _, c_top = top_latents(c_result, n=100)

        # GradSAE needs a real forward pass with real tokens, on the single
        # highest-firing token found for this neuron.
        ex_idx, pos = pairs[0]
        tokens = lang_acts[ex_idx].tokens.to(device)
        attention_mask = torch.ones_like(tokens)
        g_result = gradsae_output_attribution(
            model, sae, tokens, attention_mask, layer, neuron, target_layer, position=pos
        )
        _, g_top = top_latents(g_result, n=100)

        w_top20, c_top20, g_top20 = w_top[:20], c_top[:20], g_top[:20]
        entry = {
            "layer": layer,
            "neuron": neuron,
            "language": language,
            "target_layer": target_layer,
            "is_last_layer_forced": is_last_layer_forced,
            "weight_vs_correlational_jaccard": jaccard_overlap(w_top20, c_top20),
            "weight_vs_gradsae_jaccard": jaccard_overlap(w_top20, g_top20),
            "correlational_vs_gradsae_jaccard": jaccard_overlap(c_top20, g_top20),
        }

        # Causal check: do the baselines' top latents actually do anything
        # when patched, the same test our own attribution had to pass?
        recon_baseline = patch_latents(
            model, sae, tokens, attention_mask, target_layer,
            latent_indices=torch.tensor([], dtype=torch.long, device=device),
        )
        for method_name, top_set in [("weight", w_top), ("correlational", c_top), ("gradsae", g_top)]:
            encoded_indices = sae.encode(resid[target_layer])[0].indices.flatten()
            patchable_mask = torch.isin(top_set, encoded_indices)
            patchable = top_set[patchable_mask][:20]
            if len(patchable) == 0:
                entry[f"{method_name}_patch_shift"] = None
                continue
            patched = patch_latents(model, sae, tokens, attention_mask, target_layer, latent_indices=patchable)
            shift = (patched.patched_logits[:, pos] - recon_baseline.patched_logits[:, pos]).abs().mean().item()
            entry[f"{method_name}_patch_shift"] = shift
            entry[f"{method_name}_n_patchable"] = int(patchable_mask.sum().item())

        results.append(entry)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(top_neurons)} neurons compared")

    out = {"model": args.model, "n_neurons": len(top_neurons), "comparisons": results}
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. Results saved to {args.out}")


if __name__ == "__main__":
    main()
