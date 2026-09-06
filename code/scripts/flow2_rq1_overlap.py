"""Flow 2's central RQ1 test: do the latents that our weight-based and
gradient-based attribution identify for a language-specific neuron overlap,
above chance, with the latents that two independently reproduced published
methods (Andrylie et al.'s SAE-LAPE, Deng et al.'s monolinguality metric)
flag as language-specific in the SAME language, on the SAME trained BLOOM
MLSAE?

This is a from-scratch analysis, not a reuse of flow1_analysis.py's saved
JSON, because that script only persists summary statistics (counts, ratios)
and not the actual attributed latent indices needed for a set-overlap test.

Two passes over the corpus, for the same reason as flow1_analysis.py: caching
every example's full activations (even just the raw neuron tensor) is what
silently OOM'd the whole machine at BLOOM-1.7B's width. Pass 1 accumulates
LAPE, SAE-LAPE, and monolinguality statistics in streaming fashion (no
activations retained beyond each example's own update). Pass 2 re-streams,
per language, caching only the residual stream and the specific (layer,
neuron) columns the LAPE-selected neurons need for attribution.
"""

import argparse
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")

import torch
from scipy.stats import hypergeom

from langflow.attribution import (
    choose_target_layer,
    gather_top_firing_selective,
    top_latents,
    weight_based_attribution,
)
from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.lape import LapeAccumulator, select_language_neurons
from langflow.mlsae import TopKSAE
from langflow.models import run_with_cache, run_with_selective_cache
from langflow.reference_features import (
    MonolingualityAccumulator,
    SAELapeAccumulator,
    top_monolinguality_latents,
    top_sae_lape_latents,
)

BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def enrichment_pvalue(attributed: set, reference: set, n_latents: int) -> float:
    """One-sided hypergeometric test: is the overlap between `attributed` and
    `reference` larger than expected if `attributed` were a random subset of
    the same size drawn from the full dictionary?"""
    overlap = len(attributed & reference)
    # sf(k-1) = P(X >= k), the standard "at least this much overlap" tail
    return float(hypergeom.sf(overlap - 1, n_latents, len(reference), len(attributed)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="bloom-560m")
    parser.add_argument("--model-name", default="bigscience/bloom-560m")
    parser.add_argument("--mlsae-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--n-examples-per-lang", type=int, default=1500)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-neurons", type=int, default=150)
    parser.add_argument("--top-k-per-language-monolinguality", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = args.device
    print(f"Loading {args.model_key} and its trained MLSAE on {device}...")
    model, tokenizer = load_bloom(args.model_name, device=device)
    config = json.load(open(args.mlsae_config))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    sae.load_state_dict(torch.load(args.mlsae_checkpoint, map_location=device))
    sae.to(device)
    sae.eval()
    print(f"MLSAE: n_inputs={sae.n_inputs}, n_latents={sae.n_latents}, k={sae.k}")

    print(f"Streaming probe corpus: {BLOOM_LANGUAGES}, {args.n_examples_per_lang} examples/language...")
    t0 = time.time()
    corpus = load_probe_corpus(
        languages=BLOOM_LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200
    )
    print(f"  corpus streamed in {time.time()-t0:.0f}s")

    print("Pass 1/2: streaming through the corpus, accumulating LAPE (neuron), SAE-LAPE")
    print("(latent), and monolinguality (latent) statistics only — no activations retained.")
    t0 = time.time()
    lape_acc = LapeAccumulator(BLOOM_LANGUAGES)
    sae_lape_acc = SAELapeAccumulator(sae, BLOOM_LANGUAGES)
    mono_acc = MonolingualityAccumulator(sae, BLOOM_LANGUAGES)

    n_done = 0
    total = sum(len(t) for t in corpus.values())
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            acts = run_with_cache(model, tokens, mask)
            lape_acc.update(lang, acts.neurons)
            sae_lape_acc.update(lang, acts.resid)
            mono_acc.update(lang, acts.resid)
            del acts
            n_done += 1
            if n_done % 2000 == 0:
                print(f"  {n_done}/{total} examples processed, {time.time()-t0:.0f}s elapsed")
    print(f"  accumulation pass done in {time.time()-t0:.0f}s")

    print("Finalizing LAPE (neuron), SAE-LAPE (latent), and monolinguality (latent) scores...")
    lape_scores = lape_acc.finalize()
    language_neurons = select_language_neurons(lape_scores, entropy_percentile=0.01)
    print(f"  {len(language_neurons)} language-specific neurons found")

    sae_lape_scores = sae_lape_acc.finalize()
    sae_lape_latents = top_sae_lape_latents(sae_lape_scores, entropy_percentile=0.01, k=sae.k)
    sae_lape_by_lang: dict[str, set[int]] = defaultdict(set)
    for sl in sae_lape_latents:
        sae_lape_by_lang[sl.language].add(sl.latent)
    print(f"  SAE-LAPE: {len(sae_lape_latents)} language-specific latents found")

    nu = mono_acc.finalize()
    mono_latents = top_monolinguality_latents(nu, top_k_per_language=args.top_k_per_language_monolinguality)
    mono_by_lang: dict[str, set[int]] = defaultdict(set)
    for ml in mono_latents:
        mono_by_lang[ml.language].add(ml.latent)
    print(f"  Monolinguality: {len(mono_latents)} language-specific latents found")

    # Persist the raw per-latent scores so a threshold/parameter change later
    # doesn't require re-streaming the whole corpus and re-running the SAE
    # over it again (the expensive part of this script).
    raw_scores_path = args.out.replace(".json", "_raw_scores.pt")
    torch.save(
        {
            "sae_lape_firing_rate": sae_lape_scores.firing_rate,
            "sae_lape_entropy": sae_lape_scores.entropy,
            "sae_lape_dominant_language": sae_lape_scores.dominant_language,
            "sae_lape_overall_firing_rate": sae_lape_scores.overall_firing_rate,
            "sae_lape_languages": sae_lape_scores.languages,
            "monolinguality_nu": nu,
        },
        raw_scores_path,
    )
    print(f"  raw per-latent scores saved to {raw_scores_path} for future re-tuning")

    n_layers = model.config.num_hidden_layers
    top_neurons = language_neurons[: args.n_neurons]
    neurons_by_lang: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for n in top_neurons:
        neurons_by_lang[n.language].append((n.layer, n.neuron))

    print(
        f"Pass 2/2: re-streaming each language's examples, caching only the residual stream "
        f"plus the {len(top_neurons)} target neurons' own activation columns..."
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

    print(f"\nRunning attribution on top {len(top_neurons)} neurons to get attributed latent sets...")
    attributed_by_lang: dict[str, set[int]] = defaultdict(set)
    per_neuron_results = []
    for i, target in enumerate(top_neurons):
        target_layer, is_last_layer_forced = choose_target_layer(target.layer, n_layers)
        resid, neuron_act, pairs = gather_top_firing_selective(
            acts_by_language[target.language], target.layer, target.neuron, top_n=40
        )
        resid, neuron_act = resid.to(device), neuron_act.to(device)
        w_result = weight_based_attribution(
            model, sae, resid[target.layer], neuron_act, target.layer, target.neuron, target_layer
        )
        _, w_top = top_latents(w_result, n=100)
        # Exclude latents attributed via a forced last-layer target from the
        # main pooled overlap test: that layer's SAE reconstruction is
        # consistently the least reliable of any layer (see choose_target_layer),
        # so mixing those latents in would inflate or deflate the overlap
        # statistics for reasons unrelated to the attribution method itself.
        if not is_last_layer_forced:
            attributed_by_lang[target.language].update(w_top.tolist())
        per_neuron_results.append(
            {
                "layer": target.layer,
                "neuron": target.neuron,
                "language": target.language,
                "target_layer": target_layer,
                "is_last_layer_forced": is_last_layer_forced,
                "attributed_latents": w_top.tolist(),
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(top_neurons)} neurons attributed")

    print("\nComputing RQ1 overlap: attributed latents vs. SAE-LAPE and monolinguality reference sets...")
    overlap_results = []
    for lang in BLOOM_LANGUAGES:
        attributed = attributed_by_lang.get(lang, set())
        if not attributed:
            continue
        sae_lape_ref = sae_lape_by_lang.get(lang, set())
        mono_ref = mono_by_lang.get(lang, set())

        entry = {
            "language": lang,
            "n_attributed": len(attributed),
            "n_sae_lape_reference": len(sae_lape_ref),
            "n_monolinguality_reference": len(mono_ref),
            "sae_lape_jaccard": jaccard(attributed, sae_lape_ref),
            "sae_lape_overlap_count": len(attributed & sae_lape_ref),
            "sae_lape_enrichment_pvalue": enrichment_pvalue(attributed, sae_lape_ref, sae.n_latents)
            if sae_lape_ref
            else None,
            "monolinguality_jaccard": jaccard(attributed, mono_ref),
            "monolinguality_overlap_count": len(attributed & mono_ref),
            "monolinguality_enrichment_pvalue": enrichment_pvalue(attributed, mono_ref, sae.n_latents)
            if mono_ref
            else None,
        }
        overlap_results.append(entry)
        print(
            f"  {lang}: attributed={entry['n_attributed']}, "
            f"SAE-LAPE overlap={entry['sae_lape_overlap_count']} (p={entry['sae_lape_enrichment_pvalue']}), "
            f"monolinguality overlap={entry['monolinguality_overlap_count']} (p={entry['monolinguality_enrichment_pvalue']})"
        )

    # Pooled (language-agnostic) comparison as a secondary check.
    all_attributed = set().union(*attributed_by_lang.values()) if attributed_by_lang else set()
    all_sae_lape = set().union(*sae_lape_by_lang.values()) if sae_lape_by_lang else set()
    all_mono = set().union(*mono_by_lang.values()) if mono_by_lang else set()
    pooled = {
        "n_attributed": len(all_attributed),
        "n_sae_lape_reference": len(all_sae_lape),
        "n_monolinguality_reference": len(all_mono),
        "sae_lape_jaccard": jaccard(all_attributed, all_sae_lape),
        "sae_lape_overlap_count": len(all_attributed & all_sae_lape),
        "sae_lape_enrichment_pvalue": enrichment_pvalue(all_attributed, all_sae_lape, sae.n_latents),
        "monolinguality_jaccard": jaccard(all_attributed, all_mono),
        "monolinguality_overlap_count": len(all_attributed & all_mono),
        "monolinguality_enrichment_pvalue": enrichment_pvalue(all_attributed, all_mono, sae.n_latents),
    }
    print(f"\nPooled (language-agnostic): {pooled}")

    results = {
        "model": args.model_key,
        "languages": BLOOM_LANGUAGES,
        "n_examples_per_lang": args.n_examples_per_lang,
        "n_language_neurons_found": len(language_neurons),
        "n_neurons_attributed": len(top_neurons),
        "per_language": overlap_results,
        "pooled": pooled,
        "per_neuron": per_neuron_results,
    }
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results saved to {args.out}")


if __name__ == "__main__":
    main()
