"""Step 1 of the outlier-dimension follow-up: does BLOOM's last-layer OD
literally implement Macocco et al.'s "boost frequent tokens" heuristic, and
if so, does the language-dependent OD magnitude found in
outlier_dimension_bridge.py reflect a *different* boosted token set per
language, or the *same* (English/Latin-BPE-skewed) token set being
suppressed in some language contexts because boosting it would be actively
wrong there?

Method (logit-lens, no gradient, no SAE involved):
  1. Take the mean last-layer residual across all languages as a neutral
     baseline point mu.
  2. For the primary OD dimension d, perturb mu by +/- its typical signed
     magnitude in each language (from real per-token residuals, not just
     abs value), pass mu and mu+perturbation through ln_f -> lm_head, and
     read off which tokens' log-probability increases most.
  3. Compare that boosted-token set against each language's own empirical
     token-frequency ranking (from the probe corpus), both as a frequency
     lookup (are the OD-boosted tokens even frequent in this language's
     text at all?) and as an overlap with the boosted set computed at a
     DIFFERENT language's typical magnitude (same tokens, just scaled
     differently, or a genuinely different set?).
"""

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, "src")

import torch

from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.models import get_layers

BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


@torch.no_grad()
def capture_last_layer(model, tokens, mask, last_layer_idx):
    captured = {}

    def hook(_m, _a, output):
        captured["resid"] = output[0] if isinstance(output, tuple) else output
    handle = get_layers(model)[last_layer_idx].register_forward_hook(hook)
    try:
        model(input_ids=tokens, attention_mask=mask)
    finally:
        handle.remove()
    return captured["resid"]


@torch.no_grad()
def logits_from_resid(model, resid_vec):
    """resid_vec: (d_model,). Apply the model's final ln_f + lm_head directly
    (skip the remaining transformer layers -- resid_vec is already the
    output of the LAST transformer layer)."""
    x = resid_vec.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    x = model.transformer.ln_f(x)
    logits = model.lm_head(x)
    return logits[0, 0]  # (vocab,)


def top_tokens(logits, tokenizer, n=20):
    vals, idx = torch.topk(logits, n)
    return [(tokenizer.decode([i.item()]), round(v.item(), 3)) for v, i in zip(vals, idx)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--outlier-bridge-results", required=True)
    parser.add_argument("--n-examples-per-lang", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[{args.model}] loading model on {args.device}...")
    model, tokenizer = load_bloom(args.model_name, device=args.device)
    n_layers = model.config.num_hidden_layers
    last_layer = n_layers - 1

    bridge = json.load(open(args.outlier_bridge_results))
    last_layer_result = [r for r in bridge["results"] if r["layer"] == last_layer][0]
    primary_od = last_layer_result["primary_od"]
    print(f"primary OD dim at last layer: {primary_od}")

    print(f"streaming probe corpus: {BLOOM_LANGUAGES} x {args.n_examples_per_lang}...")
    corpus = load_probe_corpus(languages=BLOOM_LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200)

    resid_by_lang = {lang: [] for lang in BLOOM_LANGUAGES}
    token_freq_by_lang = {lang: Counter() for lang in BLOOM_LANGUAGES}
    n_texts = 0
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens, mask = enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)
            if tokens.shape[1] < 2:
                continue
            resid = capture_last_layer(model, tokens, mask, last_layer)
            m = mask[0].bool()
            r = resid[0][m].cpu()
            resid_by_lang[lang].append(r)
            token_freq_by_lang[lang].update(tokens[0][m].cpu().tolist())
            n_texts += 1
        print(f"  {lang} done ({n_texts} texts so far)")

    all_resid = torch.cat([torch.cat(v, dim=0) for v in resid_by_lang.values() if v], dim=0)
    mu = all_resid.mean(dim=0).to(args.device)  # (d_model,) neutral baseline

    baseline_logits = logits_from_resid(model, mu)
    baseline_logprobs = torch.log_softmax(baseline_logits, dim=-1)

    # signed per-language mean activation of the primary OD dim (not abs --
    # need direction for a logit-lens perturbation)
    signed_mag_by_lang = {}
    for lang, chunks in resid_by_lang.items():
        if not chunks:
            continue
        vals = torch.cat(chunks, dim=0)[:, primary_od]
        signed_mag_by_lang[lang] = vals.mean().item()
    print("signed OD mean activation by language:", {k: round(v, 2) for k, v in signed_mag_by_lang.items()})

    # global top-boosted tokens using the OD's overall mean signed magnitude
    global_signed_mag = all_resid[:, primary_od].mean().item()
    perturbed = mu.clone()
    perturbed[primary_od] = mu[primary_od] + abs(global_signed_mag)
    perturbed_logits = logits_from_resid(model, perturbed)
    perturbed_logprobs = torch.log_softmax(perturbed_logits, dim=-1)
    shift = perturbed_logprobs - baseline_logprobs
    top_boosted_vals, top_boosted_idx = torch.topk(shift, 30)
    top_boosted_tokens = [(tokenizer.decode([i.item()]), round(v.item(), 3)) for v, i in zip(top_boosted_vals, top_boosted_idx)]
    print("\ntop 30 tokens boosted by increasing the OD (global magnitude):")
    for tok, v in top_boosted_tokens:
        print(f"    {tok!r:20s} logprob_shift={v:+.3f}")

    boosted_ids = top_boosted_idx.tolist()

    # for each language: how frequent are these OD-boosted tokens actually
    # in that language's own text, and does perturbing at THAT language's
    # own typical magnitude change which tokens top the list?
    lang_analysis = {}
    for lang in BLOOM_LANGUAGES:
        freq = token_freq_by_lang[lang]
        total = sum(freq.values()) or 1
        boosted_freq_rank = []
        for tid in boosted_ids[:10]:
            count = freq.get(tid, 0)
            boosted_freq_rank.append({"token": tokenizer.decode([tid]), "count_in_lang": count,
                                       "freq_per_10k": round(count / total * 10000, 3)})
        # perturb at this language's own signed magnitude (sign-preserving)
        lang_mag = signed_mag_by_lang.get(lang, 0.0)
        pert_lang = mu.clone()
        pert_lang[primary_od] = mu[primary_od] + lang_mag
        pert_lang_logits = logits_from_resid(model, pert_lang)
        pert_lang_logprobs = torch.log_softmax(pert_lang_logits, dim=-1)
        shift_lang = pert_lang_logprobs - baseline_logprobs
        top_lang_vals, top_lang_idx = torch.topk(shift_lang, 10)
        top_lang_boosted = [tokenizer.decode([i.item()]) for i in top_lang_idx]
        jaccard = len(set(top_lang_idx.tolist()) & set(boosted_ids[:10])) / 10.0

        lang_analysis[lang] = {
            "signed_od_magnitude": lang_mag,
            "boosted_tokens_freq_in_this_lang": boosted_freq_rank,
            "top10_boosted_at_own_magnitude": top_lang_boosted,
            "jaccard_vs_global_boosted_set": jaccard,
        }
        freq_summary = [(b["token"], b["freq_per_10k"]) for b in boosted_freq_rank[:5]]
        print(f"\n{lang}: OD mag={lang_mag:+.2f}  jaccard(own-mag top10, global top10)={jaccard:.2f}")
        print(f"  global-boosted tokens' frequency in {lang} text (per 10k tokens): {freq_summary}")

    result = {
        "model": args.model, "primary_od": primary_od,
        "global_top30_boosted": top_boosted_tokens,
        "signed_mag_by_lang": signed_mag_by_lang,
        "lang_analysis": lang_analysis,
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
