"""RQ3 (flow_1.md), answered properly: does a language-specific neuron's
attributed SAE signal persist / reappear at layers OTHER than the single
`neuron_layer + 2` heuristic that attribution+patching have used so far?
And in what geometric form -- concentrated in one latent (monosemantic) or
spread across many, mutually-interfering latents (superposed)?

For each neuron, using its weight-attributed candidate latents (computed
once, at its own layer, per the corrected design in attribution.py):

  1. Firing rate & language-specificity (mean activation on this language's
     gathered tokens minus mean activation on a shared cross-language
     background) of the candidate latents, swept across EVERY layer.
  2. Participation Ratio (Gao/Recanatesi-style effective dimensionality,
     PR = (sum lambda)^2 / sum(lambda^2) over the eigenvalues of the
     candidate latents' activation covariance across this language's
     tokens) at every layer -- PR near 1 means the language signal at that
     layer concentrates in one dominant latent (monosemantic); PR large
     means it spreads roughly evenly across the candidate set (superposed).
  3. Pairwise decoder-direction interference (Elhage et al. 2209.10652,
     "Toy Models of Superposition": the dot product between two features'
     directions measures how much they interfere geometrically) among the
     latents empirically active at that layer -- since decoder columns are
     unit-norm (verified directly on this project's trained SAEs), this
     dot product is exactly a cosine similarity. Near 0 across all pairs
     means the active latents are geometrically disentangled; large
     magnitudes mean they overlap/interfere.
  4. Causal injection: add the candidate set's decoder-direction sum at
     that layer (reusing `steer_latents`'s mechanism) on a small prompt
     set, and measure the shift in log-probability toward this language's
     marker tokens -- the direct causal counterpart to (1)-(3)'s
     correlational/geometric picture.

Batching notes (this sweep is n_neurons x n_layers, so per-call overhead
matters): the background contrast is neuron-independent, so its SAE
encoding is precomputed ONCE per layer before the neuron loop rather than
recomputed per neuron. Steering prompts are batched into a single
left-padded tensor (so `position=-1` always indexes each row's real last
token) and steered in one `steer_latents` call per (neuron, layer) instead
of one call per prompt. The steering coefficient's scale reference
(`typical_norm`) reuses this neuron's own already-cached residual instead
of an extra forward pass.
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "src")

import torch

from langflow.attribution import gather_top_firing_selective, top_latents, weight_based_attribution
from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.mlsae import TopKSAE
from langflow.models import get_layers, run_with_selective_cache
from langflow.patching import language_logprob_shift, steer_latents

BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


def participation_ratio(activations: torch.Tensor) -> float:
    """activations: (n_tokens, n_candidates), dense (0 where a candidate
    latent was not selected for that token). PR = (sum lambda)^2 /
    sum(lambda^2) over the eigenvalues of the covariance matrix across the
    candidate dimension."""
    if activations.shape[0] < 2:
        return float("nan")
    centered = activations - activations.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (activations.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 < 1e-12:
        return float("nan")
    return (s1 ** 2 / s2).item()


def get_marker_tokens(tokenizer, texts, n_markers=15, min_len=3):
    counts = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text))
    candidates = [(tid, c) for tid, c in counts.items() if len(tokenizer.decode([tid]).strip()) >= min_len]
    candidates.sort(key=lambda x: -x[1])
    return [tid for tid, _ in candidates[:n_markers]]


def left_pad_batch(token_lists, pad_id, device):
    """Left-pad variable-length token lists into one (batch, maxlen) tensor,
    so `position=-1` always indexes each row's real last token regardless
    of its original length."""
    maxlen = max(len(t) for t in token_lists)
    input_ids = torch.full((len(token_lists), maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(token_lists), maxlen), dtype=torch.long)
    for i, t in enumerate(token_lists):
        input_ids[i, maxlen - len(t):] = torch.tensor(t)
        attention_mask[i, maxlen - len(t):] = 1
    return input_ids.to(device), attention_mask.to(device)


@torch.no_grad()
def resid_at_layer(model, tokens, attention_mask, layer):
    """One real (possibly batched) forward pass's residual-stream output at
    `layer`, via the same forward-hook pattern used throughout this
    codebase."""
    captured = {}

    def hook(_module, _args, output):
        captured["resid"] = output[0] if isinstance(output, tuple) else output

    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        model(input_ids=tokens, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["resid"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mlsae-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--existing-results", required=True, help="flow1_analysis.py output, for lape_top")
    parser.add_argument("--n-neurons-per-lang", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--n-examples-per-lang", type=int, default=1500)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-background-examples", type=int, default=8)
    parser.add_argument("--n-steer-prompts", type=int, default=8)
    parser.add_argument("--steer-alpha", type=float, default=6.0)
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
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )

    existing = json.load(open(args.existing_results))
    by_lang = defaultdict(list)
    for n in existing["lape_top"]:
        by_lang[n["language"]].append(n)
    target_neurons_list = []
    for lang, ns in by_lang.items():
        target_neurons_list.extend(ns[: args.n_neurons_per_lang])
    print(f"Selected {len(target_neurons_list)} neurons across {len(by_lang)} languages "
          f"(up to {args.n_neurons_per_lang} per language)")

    print(f"Streaming probe corpus: {BLOOM_LANGUAGES}, {args.n_examples_per_lang} examples/language...")
    t0 = time.time()
    corpus = load_probe_corpus(languages=BLOOM_LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200)
    print(f"  corpus streamed in {time.time()-t0:.0f}s")

    neurons_by_lang = defaultdict(list)
    for n in target_neurons_list:
        neurons_by_lang[n["language"]].append((n["layer"], n["neuron"]))

    print("Selectively caching FULL residual stream (all layers) + target neuron columns...")
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

    # Shared cross-language background (neuron-independent) and steering
    # prompts, both batched with left-padding.
    bg_langs = BLOOM_LANGUAGES[: args.n_background_examples]
    bg_corpus_full = load_probe_corpus(languages=bg_langs, n_examples=1, min_chars=200)
    bg_token_lists = [tokenizer(bg_corpus_full[lang][0], truncation=True, max_length=args.max_length)["input_ids"] for lang in bg_langs]
    bg_input_ids, bg_attention_mask = left_pad_batch(bg_token_lists, pad_id, device)

    english_corpus = load_probe_corpus(languages=["en"], n_examples=args.n_steer_prompts, min_chars=200)["en"]
    steer_token_lists = [tokenizer.encode(text)[:20] for text in english_corpus]
    steer_input_ids, steer_attention_mask = left_pad_batch(steer_token_lists, pad_id, device)

    marker_corpus = load_probe_corpus(languages=list(by_lang.keys()), n_examples=30, min_chars=200)

    print(f"Precomputing background SAE encoding at all {n_layers} layers (neuron-independent, done once)...")
    t0 = time.time()
    bg_encoded_by_layer = []
    for l in range(n_layers):
        resid_bg = resid_at_layer(model, bg_input_ids, bg_attention_mask, l)[:, -1, :]  # last real token (left-padded)
        latents_bg, _ = sae.encode(resid_bg)
        bg_encoded_by_layer.append((latents_bg.indices, latents_bg.values))  # each (n_bg, k)
    print(f"  done in {time.time()-t0:.0f}s")

    print(f"\nRunning cross-layer persistence sweep on {len(target_neurons_list)} neurons, {n_layers} layers each...")
    results = []
    t0 = time.time()
    for i, target in enumerate(target_neurons_list):
        layer, neuron, language = target["layer"], target["neuron"], target["language"]
        lang_acts = acts_by_language.get(language)
        if not lang_acts:
            continue

        resid, neuron_act, pairs = gather_top_firing_selective(lang_acts, layer, neuron, top_n=40)
        resid, neuron_act = resid.to(device), neuron_act.to(device)

        # Candidate latents from this neuron's OWN layer (corrected design).
        w_result = weight_based_attribution(model, sae, resid[layer], neuron_act, layer, neuron, layer)
        _, candidates = top_latents(w_result, n=args.n_candidates)
        candidates_list = candidates.tolist()

        marker_tokens = get_marker_tokens(tokenizer, marker_corpus[language], n_markers=15)

        layer_profile = []
        for l in range(n_layers):
            own_resid = resid[l]  # (top_n, 1, d_model), this language's gathered tokens at layer l
            own_enc, _ = sae.encode(own_resid)
            idx_flat = own_enc.indices.squeeze(1)  # (n_tokens, k)
            val_flat = own_enc.values.squeeze(1)  # (n_tokens, k)

            own_dense = torch.zeros(own_resid.shape[0], len(candidates_list), device=device)
            per_candidate_fired = torch.zeros(len(candidates_list), dtype=torch.bool, device=device)
            for ci, lid in enumerate(candidates_list):
                hit = idx_flat == lid  # (n_tokens, k)
                own_dense[:, ci] = (val_flat * hit).sum(dim=-1)
                per_candidate_fired[ci] = hit.any()
            firing_rate = (own_dense != 0).any(dim=-1).float().mean().item()

            bidx, bval = bg_encoded_by_layer[l]  # (n_bg, k) each, precomputed
            bg_dense = torch.zeros(bidx.shape[0], len(candidates_list), device=device)
            for ci, lid in enumerate(candidates_list):
                hit = bidx == lid
                bg_dense[:, ci] = (bval * hit).sum(dim=-1)

            language_specificity = (own_dense.mean(dim=0) - bg_dense.mean(dim=0)).abs().mean().item()
            pr = participation_ratio(own_dense)

            n_active = int(per_candidate_fired.sum().item())
            if n_active >= 2:
                active_latents = candidates[per_candidate_fired]
                decoder_cols = sae.decoder.weight[:, active_latents]  # (d_model, n_active), unit-norm columns
                gram = (decoder_cols.T @ decoder_cols).abs()
                off_diag_mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=device)
                mean_interference = gram[off_diag_mask].mean().item()
            else:
                mean_interference = float("nan")

            # Causal injection, batched over every steer prompt in one call;
            # typical_norm reuses this neuron's already-cached residual
            # instead of an extra forward pass.
            typical_norm = own_resid.norm(dim=-1).mean().item()
            coeff = args.steer_alpha * typical_norm / max(len(candidates_list), 1)
            result = steer_latents(model, sae, steer_input_ids, steer_attention_mask, l, candidates, coefficient=coeff)
            steer_shift_mean = language_logprob_shift(result, marker_tokens, position=-1)

            layer_profile.append({
                "layer": l,
                "firing_rate": firing_rate,
                "n_active_candidates": n_active,
                "language_specificity": language_specificity,
                "participation_ratio": pr,
                "mean_pairwise_interference": mean_interference,
                "steer_shift_mean": steer_shift_mean,
            })

        results.append({
            "layer": layer, "neuron": neuron, "language": language,
            "candidates": candidates_list,
            "layer_profile": layer_profile,
        })
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(target_neurons_list)} neurons swept, {time.time()-t0:.0f}s elapsed")

    out = {"model": args.model, "n_layers": n_layers, "n_candidates": args.n_candidates, "results": results}
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. Results saved to {args.out}")


if __name__ == "__main__":
    main()
