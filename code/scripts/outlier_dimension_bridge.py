"""Bridge between last-layer outlier dimensions (Macocco et al. 2025, arXiv
2503.21718: "Not a nuisance but a useful heuristic") and the SAE's dominant
latent artifact (dominant_latent_check.py) found in this project.

Macocco et al.'s OD identification criterion (Sec. 3, "OD identification"):
  1. Pool |activation| over all (token, dim) pairs in a layer.
  2. threshold = the 99th percentile of that pooled distribution.
  3. A dimension d is an Outlier Dimension (OD) if, for >=50% of input
     tokens, |activation[t, d]| exceeds threshold.

We adapt the same "always active regardless of context" logic to the SAE's
latent space to define a "dominant latent": a latent that appears in the
top-k set for >=50% of tokens (this is exactly the operational definition
behind the dominant_latent_check.py robustness correction, now made
explicit and tied to the OD literature).

For each of a few layers (mid-layer control + last 4 layers, where ODs are
reported to concentrate), this script:
  1. identifies OD raw dimensions and dominant SAE latents independently
  2. checks whether the SAE has effectively "absorbed" the OD into one
     latent: correlation between the OD's raw activation and the dominant
     latent's activation, and whether the OD dimension has outsized weight
     in that latent's encoder row / decoder column
  3. breaks the OD's magnitude down by language (does the OD favor a
     particular language's frequent tokens more?)
  4. an exploratory "robust standardization" diagnostic: if the SAE's
     per-token standardization excludes the OD dimensions when computing
     mean/std, does reconstruction fidelity (FVE) improve at the layers
     where it collapsed? NOTE: the frozen encoder was never trained under
     this modified normalization, so this is suggestive, not a rigorous
     causal test -- an improvement supports the hypothesis but a null
     result does not rule it out (the encoder is now out-of-distribution).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "src")

import torch

from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.mlsae import TopKSAE, standardize
from langflow.models import get_layers

BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


@torch.no_grad()
def capture_layers(model, tokens, mask, target_layers):
    captured = {}
    handles = []
    target_set = set(target_layers)
    for l, layer_mod in enumerate(get_layers(model)):
        if l not in target_set:
            continue

        def hook(_m, _a, output, l=l):
            captured[l] = output[0] if isinstance(output, tuple) else output
        handles.append(layer_mod.register_forward_hook(hook))
    try:
        model(input_ids=tokens, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    return captured


def identify_ods(resid, quantile=0.99, frac_threshold=0.5):
    """Macocco et al.'s OD criterion. resid: (n_tokens, n_inputs).

    torch.quantile refuses tensors above ~16M elements, which n_tokens *
    n_inputs regularly exceeds here -- use numpy (no such limit; the sort
    it does under the hood is the same O(n log n) cost either way)."""
    import numpy as np
    abs_resid = resid.abs()
    threshold = float(np.quantile(abs_resid.flatten().numpy(), quantile))
    frac_exceed = (abs_resid > threshold).float().mean(dim=0)  # (n_inputs,)
    od_dims = (frac_exceed >= frac_threshold).nonzero().flatten().tolist()
    return od_dims, frac_exceed, threshold


def identify_dominant_latents(sae, resid, frac_threshold=0.5):
    """Latents that appear in the top-k set for >=frac_threshold of tokens."""
    latents, stats = sae.encode(resid)
    n_tokens = resid.shape[0]
    counts = torch.bincount(latents.indices.flatten(), minlength=sae.n_latents).float()
    frac_active = counts / n_tokens
    dominant = (frac_active >= frac_threshold).nonzero().flatten().tolist()
    return dominant, frac_active, latents, stats


def latent_activation_per_token(latents, latent_id, n_tokens):
    """Dense per-token activation value for one latent id (0 where not in top-k)."""
    dense = torch.zeros(n_tokens)
    hit = (latents.indices == latent_id)
    rows, cols = hit.nonzero(as_tuple=True)
    dense[rows] = latents.values[rows, cols]
    return dense


def robust_standardize_fve(sae, resid, od_dims, eps=1e-5):
    """Diagnostic only (see module docstring caveat): standardize excluding
    OD dims from mean/std, run through the (frozen, unmodified) encoder/decoder,
    invert with the SAME robust stats, and report FVE against the true resid."""
    n_inputs = resid.shape[-1]
    keep_mask = torch.ones(n_inputs, dtype=torch.bool)
    keep_mask[od_dims] = False
    if keep_mask.sum() == 0:
        return None
    sub = resid[:, keep_mask]
    mean = sub.mean(dim=-1, keepdim=True)
    std = sub.std(dim=-1, keepdim=True)
    x = (resid - mean) / (std + eps)
    pre_acts = sae.encoder(x - sae.pre_encoder_bias)
    values, indices = torch.topk(pre_acts, k=sae.k, dim=-1, sorted=False)
    values = torch.relu(values)
    gathered = torch.nn.functional.embedding(indices, sae.decoder.weight.T)
    recon_std_space = (gathered * values.unsqueeze(-1)).sum(dim=-2) + sae.pre_encoder_bias
    recon = recon_std_space * std + mean
    fvu = ((resid - recon) ** 2).sum().item() / ((resid - resid.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - fvu


def standard_fve(sae, resid):
    recon, _ = sae(resid)
    fvu = ((resid - recon) ** 2).sum().item() / ((resid - resid.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - fvu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mlsae-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--n-examples-per-lang", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[{args.model}] loading model + SAE on cpu...")
    model, tokenizer = load_bloom(args.model_name, device="cpu")
    config = json.load(open(args.mlsae_config))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    sae.load_state_dict(torch.load(args.mlsae_checkpoint, map_location="cpu"))
    sae.eval()
    n_layers = model.config.num_hidden_layers

    target_layers = sorted(set([n_layers // 2, n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1]))
    print(f"target layers: {target_layers}")

    print(f"streaming probe corpus: {BLOOM_LANGUAGES} x {args.n_examples_per_lang}...")
    corpus = load_probe_corpus(languages=BLOOM_LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200)

    # per-layer accumulation: resid tokens + language label per token
    resid_by_layer = defaultdict(list)
    lang_by_layer = defaultdict(list)

    n_texts = 0
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens, mask = enc["input_ids"], enc["attention_mask"]
            if tokens.shape[1] < 2:
                continue
            captured = capture_layers(model, tokens, mask, target_layers)
            m = mask[0].bool()
            for l in target_layers:
                r = captured[l][0][m]
                resid_by_layer[l].append(r)
                lang_by_layer[l].extend([lang] * r.shape[0])
            n_texts += 1
        print(f"  {lang} done ({n_texts} texts so far)")

    layer_results = []
    for l in target_layers:
        resid = torch.cat(resid_by_layer[l], dim=0)  # (n_tokens, n_inputs)
        languages_flat = lang_by_layer[l]
        n_tokens = resid.shape[0]
        print(f"\n=== layer {l}: {n_tokens} tokens ===")

        od_dims, frac_exceed, threshold = identify_ods(resid)
        print(f"  OD dims: {od_dims} (global top-1% threshold={threshold:.3f})")

        dominant_latents, frac_active, latents, stats = identify_dominant_latents(sae, resid)
        print(f"  dominant latents (active in >=50% of tokens): {dominant_latents}")

        fve_standard = standard_fve(sae, resid)
        fve_robust = robust_standardize_fve(sae, resid, od_dims) if od_dims else None
        print(f"  SAE FVE: standard={fve_standard:.4f}  robust(OD-excluded std)={fve_robust}")

        # overlap: for each dominant latent, correlate its activation with the
        # strongest OD dim's raw activation, and check encoder/decoder weight
        # concentration on OD dims.
        overlaps = []
        primary_od = None
        if od_dims:
            primary_od = max(od_dims, key=lambda d: frac_exceed[d].item())
            od_activation = resid[:, primary_od]
        for lat in dominant_latents:
            lat_activation = latent_activation_per_token(latents, lat, n_tokens)
            corr = None
            if od_dims and lat_activation.std() > 1e-9 and od_activation.std() > 1e-9:
                corr = torch.corrcoef(torch.stack([od_activation, lat_activation]))[0, 1].item()
            enc_row = sae.encoder.weight[lat].abs()  # (n_inputs,)
            dec_col = sae.decoder.weight[:, lat].abs()  # (n_inputs,)
            enc_rank = None
            dec_rank = None
            if od_dims:
                enc_rank = (enc_row > enc_row[primary_od]).sum().item()  # 0 = OD dim has the single largest weight
                dec_rank = (dec_col > dec_col[primary_od]).sum().item()
            overlaps.append({
                "latent": lat, "frac_active": frac_active[lat].item(),
                "corr_with_primary_od": corr,
                "primary_od_encoder_weight_rank": enc_rank,
                "primary_od_decoder_weight_rank": dec_rank,
                "n_inputs": resid.shape[-1],
            })
            print(f"    latent {lat}: frac_active={frac_active[lat].item():.3f}  "
                  f"corr_with_OD{primary_od}={corr}  enc_rank={enc_rank}/{resid.shape[-1]}  dec_rank={dec_rank}/{resid.shape[-1]}")

        # language breakdown of the primary OD's magnitude
        lang_breakdown = {}
        if primary_od is not None:
            od_vals = resid[:, primary_od]
            by_lang = defaultdict(list)
            for v, lg in zip(od_vals.tolist(), languages_flat):
                by_lang[lg].append(v)
            for lg, vals in by_lang.items():
                t = torch.tensor(vals)
                lang_breakdown[lg] = {
                    "n_tokens": len(vals),
                    "mean_abs": t.abs().mean().item(),
                    "median_abs": t.abs().median().item(),
                    "frac_exceed_threshold": (t.abs() > threshold).float().mean().item(),
                }
            print(f"  primary OD dim {primary_od} magnitude by language:")
            for lg in sorted(lang_breakdown, key=lambda x: -lang_breakdown[x]["mean_abs"]):
                b = lang_breakdown[lg]
                print(f"    {lg:4s}: mean_abs={b['mean_abs']:8.3f}  median_abs={b['median_abs']:8.3f}  "
                      f"frac_exceed={b['frac_exceed_threshold']:.3f}  n={b['n_tokens']}")

        layer_results.append({
            "layer": l, "n_tokens": n_tokens, "n_inputs": resid.shape[-1],
            "od_dims": od_dims, "od_threshold": threshold,
            "dominant_latents": dominant_latents,
            "fve_standard": fve_standard, "fve_robust_od_excluded": fve_robust,
            "overlaps": overlaps,
            "primary_od": primary_od,
            "primary_od_lang_breakdown": lang_breakdown,
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "n_layers": n_layers, "target_layers": target_layers,
                    "results": layer_results}, f, indent=2)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
