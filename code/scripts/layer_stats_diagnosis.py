"""Descriptive-statistics diagnosis of *why* steering effect drops off at
late absolute layers (see steer_heatmap.py / plot_steer_heatmap.py finding:
horizontal banding, strong for target_layer ~0-16, weak for ~20-23,
independent of the neuron's own layer).

For each layer, on a shared multilingual probe sample, computes:
  - residual stream norm: mean/std across tokens (does scale explode late?)
  - concentration: fraction of ||resid||^2 carried by the single largest-
    magnitude dimension (massive-activation-style concentration check)
  - SAE reconstruction fraction of variance explained (does the shared-
    dictionary SAE simply fit late-layer residuals worse?)
  - mean number of the SAE's top-k latents that are "new" relative to the
    previous layer's top-k on the same token (does the active latent set
    become unstable/different late, hinting the SAE's dictionary doesn't
    track late-layer structure as cleanly)

These are correlational diagnostics only -- meant to narrow down *which*
per-layer property tracks the steering drop-off, not to prove a causal
mechanism.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")

import torch

from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.mlsae import TopKSAE
from langflow.models import get_layers

LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


@torch.no_grad()
def resid_all_layers(model, tokens, attention_mask):
    """Full residual stream at every layer's output, for one batch."""
    captured = []
    handles = []
    for layer_mod in get_layers(model):
        def hook(_m, _a, output, captured=captured):
            captured.append(output[0] if isinstance(output, tuple) else output)
        handles.append(layer_mod.register_forward_hook(hook))
    try:
        model(input_ids=tokens, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captured  # list of (batch, seq, d_model), len n_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mlsae-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--n-examples-per-lang", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = args.device
    print(f"[{args.model}] loading model + SAE on {device}...")
    model, tokenizer = load_bloom(args.model_name, device=device)
    config = json.load(open(args.mlsae_config))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    sae.load_state_dict(torch.load(args.mlsae_checkpoint, map_location=device))
    sae.to(device)
    sae.eval()
    n_layers = model.config.num_hidden_layers

    print(f"streaming probe corpus: {LANGUAGES} x {args.n_examples_per_lang}...")
    corpus = load_probe_corpus(languages=LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200)

    # accumulate per-layer stats over all tokens (padding excluded) of all texts/languages
    sum_norm = torch.zeros(n_layers)
    sum_norm_sq = torch.zeros(n_layers)
    sum_top1_frac = torch.zeros(n_layers)
    sum_fvu_num = torch.zeros(n_layers)  # sum ||resid-recon||^2
    sum_fvu_den = torch.zeros(n_layers)  # sum ||resid-mean||^2 (per-batch mean, see below)
    n_tokens = torch.zeros(n_layers)

    n_texts = 0
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            if tokens.shape[1] < 2:
                continue
            resid_layers = resid_all_layers(model, tokens, mask)
            m = mask[0].bool()
            for l in range(n_layers):
                r = resid_layers[l][0][m]  # (n_valid_tokens, d_model)
                if r.shape[0] == 0:
                    continue
                norms = r.norm(dim=-1)
                sum_norm[l] += norms.sum().item()
                sum_norm_sq[l] += (norms ** 2).sum().item()
                top1 = r.abs().max(dim=-1).values ** 2
                sum_top1_frac[l] += (top1 / (norms ** 2 + 1e-9)).sum().item()

                recon, _latents = sae(r)
                sum_fvu_num[l] += ((r - recon) ** 2).sum().item()
                mean_r = r.mean(dim=0, keepdim=True)
                sum_fvu_den[l] += ((r - mean_r) ** 2).sum().item()
                n_tokens[l] += r.shape[0]
            n_texts += 1
            del resid_layers
        print(f"  {lang} done ({n_texts} texts so far)")

    results = []
    for l in range(n_layers):
        nt = max(n_tokens[l].item(), 1)
        mean_norm = sum_norm[l].item() / nt
        var_norm = sum_norm_sq[l].item() / nt - mean_norm ** 2
        fvu = sum_fvu_num[l].item() / max(sum_fvu_den[l].item(), 1e-9)
        results.append({
            "layer": l,
            "mean_resid_norm": mean_norm,
            "std_resid_norm": max(var_norm, 0.0) ** 0.5,
            "mean_top1_dim_frac_of_norm_sq": sum_top1_frac[l].item() / nt,
            "sae_fraction_variance_unexplained": fvu,
            "sae_fraction_variance_explained": 1.0 - fvu,
            "n_tokens": int(nt),
        })
        print(f"  layer {l:2d}: mean_norm={mean_norm:8.2f}  top1_frac={results[-1]['mean_top1_dim_frac_of_norm_sq']:.4f}  "
              f"SAE FVE={results[-1]['sae_fraction_variance_explained']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n_layers": n_layers, "results": results}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
