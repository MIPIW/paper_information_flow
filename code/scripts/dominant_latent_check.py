"""Robustness/ablation check for RQ3/RQ4/RQ5 (flow_2.md): does the causal
patching validation get inflated by "dominant" SAE latents whose activation
magnitude is large regardless of context (a massive-activation-style
artifact, see NOTES.md's "Root mechanism" note), rather than by genuine
causal relevance to the neuron under study?

For each already-analyzed neuron's top-20 patchable attributed latents, two
checks distinguish "genuinely relevant to this neuron" from "generically
dominant":

  1. Leave-one-out share: patch each attributed latent ALONE (not the whole
     set) on this neuron's own high-firing tokens, and see what fraction of
     the *sum* of all 20 individual shifts that one latent accounts for.
  2. Context-independence: patch that same single latent ALONE on a small
     shared background sample unrelated to this neuron/language, and compare
     to its own-context shift. A latent whose background shift is comparable
     to its own-context shift is behaving the same way regardless of
     context -- its magnitude has nothing to do with this specific neuron.

A latent flagged by BOTH (share > --dominance-share-threshold AND
background/own ratio > --context-independence-threshold) is excluded, and
the attributed/neuron ratio is recomputed on the remaining latents for
direct comparison against the original (uncorrected) number. This is a
correction applied on top of the already-completed flow1_analysis.py run,
not a new research direction -- it reuses that run's `lape_top` neuron list
exactly as `baselines_comparison.py` does.
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
    top_latents,
    weight_based_attribution,
)
from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.mlsae import TopKLatents, TopKSAE
from langflow.models import get_layers, run_with_cache, run_with_selective_cache
from langflow.patching import patch_latents, patch_neuron

BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


@torch.no_grad()
def patched_logits_only(model, sae, tokens, attention_mask, target_layer, latent_indices):
    """Same hook logic as `patch_latents`, but skips the redundant baseline
    forward pass -- callers here always already have a baseline (either the
    real one or the SAE-reconstruction one) computed once and reused across
    many single-latent checks, so recomputing it per call (as `patch_latents`
    does) would double the cost of this script's already-expensive per-latent
    sweep for no benefit."""

    def zero_latents(_module, _args, output):
        real_resid = output[0] if isinstance(output, tuple) else output
        latents, stats = sae.encode(real_resid)
        zero_mask = torch.isin(latents.indices, latent_indices)
        patched_values = latents.values.masked_fill(zero_mask, 0.0)
        patched_resid = sae.decode(TopKLatents(patched_values, latents.indices), stats)
        if isinstance(output, tuple):
            return (patched_resid,) + output[1:]
        return patched_resid

    handle = get_layers(model)[target_layer].register_forward_hook(zero_latents)
    try:
        patched = model(input_ids=tokens, attention_mask=attention_mask).logits
    finally:
        handle.remove()
    return patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mlsae-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--existing-results", required=True, help="flow1_analysis.py output, for lape_top")
    parser.add_argument("--n-neurons", type=int, default=150)
    parser.add_argument("--n-examples-per-lang", type=int, default=1500)
    parser.add_argument("--n-patch-tokens", type=int, default=5)
    parser.add_argument("--n-background-examples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dominance-share-threshold", type=float, default=0.4)
    parser.add_argument("--context-independence-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = args.device
    print(f"[{args.model}] Loading model + trained MLSAE on {device}...")
    model, tokenizer = load_bloom(args.model_name, device=device)
    config = json.load(open(args.mlsae_config))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    sae.load_state_dict(torch.load(args.mlsae_checkpoint, map_location=device))
    sae.to(device)
    sae.eval()
    n_layers = model.config.num_hidden_layers

    existing = json.load(open(args.existing_results))
    top_neurons = existing["lape_top"][: args.n_neurons]
    print(f"Reusing {len(top_neurons)} LAPE-selected neurons from {args.existing_results}")

    print(f"Streaming probe corpus: {BLOOM_LANGUAGES}, {args.n_examples_per_lang} examples/language...")
    t0 = time.time()
    corpus = load_probe_corpus(languages=BLOOM_LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200)
    print(f"  corpus streamed in {time.time()-t0:.0f}s")

    neurons_by_lang = defaultdict(list)
    for n in top_neurons:
        neurons_by_lang[n["language"]].append((n["layer"], n["neuron"]))

    print("Selectively caching residual stream + target neuron columns...")
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

    # A small, shared, context-independent background sample: real tokens
    # from a random mix of languages, unrelated to any specific neuron. Used
    # to test whether a latent's large effect persists outside the context
    # it was attributed in.
    print(f"Building a {args.n_background_examples}-example shared background sample...")
    bg_langs = BLOOM_LANGUAGES[: args.n_background_examples]
    bg_corpus = load_probe_corpus(languages=bg_langs, n_examples=1, min_chars=200)
    background_tokens = []
    for lang in bg_langs:
        text = bg_corpus[lang][0]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        background_tokens.append((enc["input_ids"].to(device), enc["attention_mask"].to(device)))

    def single_latent_shift(tokens, attention_mask, target_layer, latent_id, recon_baseline_logits, pos=-1):
        patched = patched_logits_only(
            model, sae, tokens, attention_mask, target_layer,
            latent_indices=torch.tensor([latent_id], device=device, dtype=torch.long),
        )
        return (patched[:, pos] - recon_baseline_logits[:, pos]).abs().mean().item()

    # Cache each target_layer's background recon-baseline logits once, reused
    # across every latent/neuron that shares that target_layer.
    bg_recon_baseline_cache = {}

    def get_bg_recon_baseline(target_layer):
        if target_layer not in bg_recon_baseline_cache:
            logits_list = []
            for tokens, mask in background_tokens:
                r = patch_latents(
                    model, sae, tokens, mask, target_layer,
                    latent_indices=torch.tensor([], dtype=torch.long, device=device),
                )
                logits_list.append(r.patched_logits)
            bg_recon_baseline_cache[target_layer] = logits_list
        return bg_recon_baseline_cache[target_layer]

    print(f"\nRunning dominant-latent check on {len(top_neurons)} neurons...")
    results = []
    t0 = time.time()
    for i, target in enumerate(top_neurons):
        layer, neuron, language = target["layer"], target["neuron"], target["language"]
        target_layer, is_last_layer_forced = choose_target_layer(layer, n_layers)
        if is_last_layer_forced:
            continue  # already excluded from the main pooled statistics; skip here too

        lang_acts = acts_by_language.get(language)
        if not lang_acts:
            continue
        resid, neuron_act, pairs = gather_top_firing_selective(lang_acts, layer, neuron, top_n=40)
        resid, neuron_act = resid.to(device), neuron_act.to(device)

        w_result = weight_based_attribution(model, sae, resid[layer], neuron_act, layer, neuron, target_layer)
        _, w_candidates = top_latents(w_result, n=100)
        encoded_indices = sae.encode(resid[target_layer])[0].indices.flatten()
        patchable_mask = torch.isin(w_candidates, encoded_indices)
        patchable = w_candidates[patchable_mask][:20]
        if len(patchable) == 0:
            continue

        patch_pairs = pairs[: args.n_patch_tokens]

        # Original (uncorrected) attributed/neuron ratio, recomputed here so
        # it's directly comparable to the corrected version below (same
        # patch_pairs subset, not necessarily identical to flow1_analysis.py's
        # own n_patch_tokens=15 default -- this script trades some token
        # coverage for the extra per-latent cost of the two new checks).
        neuron_shifts_orig, attributed_shifts_orig = [], []
        own_shift_by_latent = {lid: [] for lid in patchable.tolist()}
        for ex_idx, pos in patch_pairs:
            acts = lang_acts[ex_idx]
            tokens = acts.tokens.to(device)
            attention_mask = torch.ones_like(tokens)

            neuron_patch = patch_neuron(model, tokens, attention_mask, layer, neuron, value=0.0)
            neuron_shifts_orig.append(
                (neuron_patch.patched_logits[:, pos] - neuron_patch.baseline_logits[:, pos]).abs().mean().item()
            )

            recon_baseline = patch_latents(
                model, sae, tokens, attention_mask, target_layer,
                latent_indices=torch.tensor([], dtype=torch.long, device=device),
            )
            attributed_patch = patch_latents(model, sae, tokens, attention_mask, target_layer, latent_indices=patchable)
            attributed_shifts_orig.append(
                (attributed_patch.patched_logits[:, pos] - recon_baseline.patched_logits[:, pos]).abs().mean().item()
            )

            for lid in patchable.tolist():
                own_shift_by_latent[lid].append(single_latent_shift(tokens, attention_mask, target_layer, lid, recon_baseline.patched_logits, pos))

        neuron_shift_mean = sum(neuron_shifts_orig) / len(neuron_shifts_orig)
        attributed_shift_mean_orig = sum(attributed_shifts_orig) / len(attributed_shifts_orig)
        own_shift_mean = {lid: sum(v) / len(v) for lid, v in own_shift_by_latent.items()}
        total_individual = sum(own_shift_mean.values()) or 1e-8

        # Background (context-independence) check, reusing the cached
        # per-target_layer recon baseline.
        bg_recon_logits_list = get_bg_recon_baseline(target_layer)
        bg_shift_mean = {}
        for lid in patchable.tolist():
            shifts = []
            for (tokens, mask), recon_logits in zip(background_tokens, bg_recon_logits_list):
                shifts.append(single_latent_shift(tokens, mask, target_layer, lid, recon_logits, pos=-1))
            bg_shift_mean[lid] = sum(shifts) / len(shifts)

        flagged = []
        for lid in patchable.tolist():
            share = own_shift_mean[lid] / total_individual
            ctx_ratio = bg_shift_mean[lid] / (own_shift_mean[lid] + 1e-8)
            if share > args.dominance_share_threshold and ctx_ratio > args.context_independence_threshold:
                flagged.append(lid)

        clean_latents = torch.tensor([lid for lid in patchable.tolist() if lid not in flagged], device=device, dtype=torch.long)
        if len(clean_latents) > 0:
            attributed_shifts_clean = []
            for ex_idx, pos in patch_pairs:
                acts = lang_acts[ex_idx]
                tokens = acts.tokens.to(device)
                attention_mask = torch.ones_like(tokens)
                recon_baseline = patch_latents(
                    model, sae, tokens, attention_mask, target_layer,
                    latent_indices=torch.tensor([], dtype=torch.long, device=device),
                )
                clean_patch = patch_latents(model, sae, tokens, attention_mask, target_layer, latent_indices=clean_latents)
                attributed_shifts_clean.append(
                    (clean_patch.patched_logits[:, pos] - recon_baseline.patched_logits[:, pos]).abs().mean().item()
                )
            attributed_shift_mean_clean = sum(attributed_shifts_clean) / len(attributed_shifts_clean)
        else:
            attributed_shift_mean_clean = None

        entry = {
            "layer": layer, "neuron": neuron, "language": language, "target_layer": target_layer,
            "n_patchable": len(patchable),
            "n_flagged_dominant": len(flagged),
            "flagged_latents": flagged,
            "neuron_shift_mean": neuron_shift_mean,
            "attributed_shift_mean_original": attributed_shift_mean_orig,
            "attributed_shift_mean_corrected": attributed_shift_mean_clean,
            "ratio_original": attributed_shift_mean_orig / neuron_shift_mean if neuron_shift_mean else None,
            "ratio_corrected": (attributed_shift_mean_clean / neuron_shift_mean) if (attributed_shift_mean_clean is not None and neuron_shift_mean) else None,
            "per_latent_own_shift": own_shift_mean,
            "per_latent_bg_shift": bg_shift_mean,
        }
        results.append(entry)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(top_neurons)} neurons checked, {time.time()-t0:.0f}s elapsed, "
                  f"{sum(1 for r in results if r['n_flagged_dominant']>0)} with >=1 flagged latent so far")

    n_with_flags = sum(1 for r in results if r["n_flagged_dominant"] > 0)
    print(f"\n{n_with_flags}/{len(results)} neurons had at least one dominant/context-independent latent flagged.")

    out = {
        "model": args.model,
        "dominance_share_threshold": args.dominance_share_threshold,
        "context_independence_threshold": args.context_independence_threshold,
        "n_patch_tokens": args.n_patch_tokens,
        "n_background_examples": args.n_background_examples,
        "results": results,
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Done. Results saved to {args.out}")


if __name__ == "__main__":
    main()
