"""Evaluate the OD-aware fine-tuned MLSAE checkpoint (finetune_od_aware_sae.py)
against the original frozen checkpoint's outlier_dimension_bridge.py results:

  1. Does per-layer FVE (under OD-aware/robust standardization, i.e. the
     normalization the fine-tune actually trained under) recover at the
     late layers where the original frozen SAE's standard-standardization
     FVE collapsed?
  2. Has the single "dominant latent" that absorbed the OD in the original
     checkpoint split into multiple latents with more distinct per-language
     profiles, or does one latent still dominate across all languages?
"""

import argparse
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")

import torch

from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.mlsae import TopKSAE
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


def robust_standardize(x, keep_mask, eps=1e-5):
    sub = x[:, keep_mask]
    mean = sub.mean(dim=-1, keepdim=True)
    std = sub.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps), mean, std


@torch.no_grad()
def robust_encode_decode(sae, resid, keep_mask):
    x, mean, std = robust_standardize(resid, keep_mask)
    pre_acts = sae.encoder(x - sae.pre_encoder_bias)
    values, indices = torch.topk(pre_acts, k=sae.k, dim=-1, sorted=False)
    values = torch.relu(values)
    gathered = torch.nn.functional.embedding(indices, sae.decoder.weight.T)
    recon_std_space = (gathered * values.unsqueeze(-1)).sum(dim=-2) + sae.pre_encoder_bias
    recon = recon_std_space * std + mean
    from langflow.mlsae import TopKLatents
    return recon, TopKLatents(values, indices)


def fve(resid, recon):
    fvu = ((resid - recon) ** 2).sum().item() / ((resid - resid.mean(dim=0, keepdim=True)) ** 2).sum().item()
    return 1.0 - fvu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--od-meta", required=True, help="the .od_meta.json written by finetune_od_aware_sae.py")
    parser.add_argument("--outlier-bridge-results", required=True, help="original frozen-checkpoint results, for comparison")
    parser.add_argument("--n-examples-per-lang", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"[{args.model}] loading model + fine-tuned SAE on {args.device}...")
    model, tokenizer = load_bloom(args.model_name, device=args.device)
    config = json.load(open(args.mlsae_config))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    sae.load_state_dict(torch.load(args.finetuned_checkpoint, map_location=args.device))
    sae.to(args.device)
    sae.eval()
    n_layers = model.config.num_hidden_layers

    od_meta = json.load(open(args.od_meta))
    od_dims = od_meta["od_dims_excluded"]
    keep_mask = torch.ones(config["n_inputs"], dtype=torch.bool, device=args.device)
    keep_mask[od_dims] = False

    original_bridge = json.load(open(args.outlier_bridge_results))
    target_layers = original_bridge["target_layers"]
    print(f"target layers: {target_layers}, OD dims excluded: {od_dims}")

    print(f"streaming probe corpus: {BLOOM_LANGUAGES} x {args.n_examples_per_lang}...")
    corpus = load_probe_corpus(languages=BLOOM_LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200)

    resid_by_layer = defaultdict(list)
    lang_by_layer = defaultdict(list)
    n_texts = 0
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens, mask = enc["input_ids"].to(args.device), enc["attention_mask"].to(args.device)
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
        resid = torch.cat(resid_by_layer[l], dim=0)
        languages_flat = lang_by_layer[l]
        n_tokens = resid.shape[0]

        recon, latents = robust_encode_decode(sae, resid, keep_mask)
        new_fve = fve(resid, recon)

        counts = torch.bincount(latents.indices.flatten(), minlength=sae.n_latents).float()
        frac_active = counts / n_tokens
        dominant_latents = (frac_active >= 0.5).nonzero().flatten().tolist()

        orig_layer_result = [r for r in original_bridge["results"] if r["layer"] == l][0]
        orig_fve_standard = orig_layer_result["fve_standard"]
        orig_fve_robust = orig_layer_result["fve_robust_od_excluded"]
        orig_dominant = orig_layer_result["dominant_latents"]

        print(f"\n=== layer {l} ===")
        print(f"  FVE: original(standard std)={orig_fve_standard:.4f}  "
              f"original(robust std, frozen weights)={orig_fve_robust}  "
              f"NEW(robust std, fine-tuned weights)={new_fve:.4f}")
        print(f"  dominant latents: original={orig_dominant}  new={dominant_latents}")

        # per-language activation-frequency profile for each new dominant latent
        lat_lang_profiles = {}
        for lat in dominant_latents:
            hit = (latents.indices == lat)
            active_per_token = hit.any(dim=-1)  # (n_tokens,)
            by_lang_active = defaultdict(lambda: [0, 0])
            for is_active, lg in zip(active_per_token.tolist(), languages_flat):
                by_lang_active[lg][1] += 1
                if is_active:
                    by_lang_active[lg][0] += 1
            lat_lang_profiles[lat] = {lg: round(c[0] / c[1], 3) for lg, c in by_lang_active.items()}
            print(f"    latent {lat} (frac_active={frac_active[lat].item():.3f}) per-language active-frac: "
                  f"{lat_lang_profiles[lat]}")

        layer_results.append({
            "layer": l, "n_tokens": n_tokens,
            "fve_original_standard": orig_fve_standard,
            "fve_original_robust_frozen": orig_fve_robust,
            "fve_new_robust_finetuned": new_fve,
            "dominant_latents_original": orig_dominant,
            "dominant_latents_new": dominant_latents,
            "new_dominant_latent_lang_profiles": lat_lang_profiles,
        })

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "od_dims": od_dims, "results": layer_results}, f, indent=2)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
