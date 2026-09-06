"""Step 2 of the outlier-dimension follow-up: warm-start from the existing
trained MLSAE checkpoint and continue training for a modest additional
token budget, but with per-token standardization computed EXCLUDING the
identified late-layer OD dimensions (see outlier_dimension_bridge.py).

This is the actual test of the hypothesis that
`outlier_dimension_bridge.py`'s naive inference-time version could not
test: does letting the SAE *learn* under OD-aware normalization (rather
than just evaluating a frozen, differently-trained SAE under it) recover
reconstruction fidelity in the layers where FVE collapsed, and does it let
the SAE represent the (language-correlated) information in the OD as
something richer than one dominant, near-context-independent latent?

Deliberately NOT a from-scratch retrain (original cost: ~8h for 560M,
~22h for 1.7B, pooling all layers) -- this is a short continued-training
run (default 50M tokens, a small fraction of the original 750M-1B token
budget) meant to answer "is this worth a full retrain" cheaply, warm-started
from the current checkpoint so it only has to adapt, not relearn everything.

The OD dimension SET is fixed per model (the union of OD dims identified at
the last 4 layers in outlier_dimension_bridge.py's results) and applied
uniformly across all layers for simplicity: at layers where these dims
aren't actually OD (e.g. the mid-layer control), excluding a handful of the
~1000-2000 input dims from the mean/std calculation has negligible effect.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

import torch
import torch.nn.functional as F

from langflow.mlsae import TopKSAE
from train_mlsae import MODEL_LOADERS, stream_wikipedia_multilingual_texts, training_batches


def collect_od_dims(outlier_bridge_path, n_layers, last_n=4):
    bridge = json.load(open(outlier_bridge_path))
    late_layers = set(range(n_layers - last_n, n_layers))
    od_dims = set()
    for r in bridge["results"]:
        if r["layer"] in late_layers:
            od_dims.update(r["od_dims"])
    return sorted(od_dims)


def robust_standardize(x, keep_mask, eps=1e-5):
    sub = x[:, keep_mask]
    mean = sub.mean(dim=-1, keepdim=True)
    std = sub.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps), mean, std


def robust_forward(sae, batch, keep_mask):
    x, mean, std = robust_standardize(batch, keep_mask)
    pre_acts = sae.encoder(x - sae.pre_encoder_bias)
    values, indices = torch.topk(pre_acts, k=sae.k, dim=-1, sorted=False)
    values = torch.relu(values)
    gathered = F.embedding(indices, sae.decoder.weight.T)
    recon_std_space = (gathered * values.unsqueeze(-1)).sum(dim=-2) + sae.pre_encoder_bias
    recon = recon_std_space * std + mean
    from langflow.mlsae import TopKLatents
    return recon, TopKLatents(values, indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--warm-start-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--outlier-bridge-results", required=True)
    parser.add_argument("--target-tokens", type=int, default=50_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=200)
    args = parser.parse_args()

    device = args.device
    model, tokenizer = MODEL_LOADERS[args.model_key](device)
    n_layers = model.config.num_hidden_layers
    config = json.load(open(args.mlsae_config))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    sae.load_state_dict(torch.load(args.warm_start_checkpoint, map_location=device))
    sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    od_dims = collect_od_dims(args.outlier_bridge_results, n_layers)
    print(f"[{args.model_key}] OD dims excluded from standardization (fixed set, all layers): {od_dims}")
    keep_mask = torch.ones(config["n_inputs"], dtype=torch.bool, device=device)
    keep_mask[od_dims] = False

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    meta_path = args.save_path + ".meta.json"
    json.dump(config, open(args.save_path + ".config.json", "w"))
    json.dump({"od_dims_excluded": od_dims, "warm_start_from": args.warm_start_checkpoint},
               open(args.save_path + ".od_meta.json", "w"))

    text_iter = stream_wikipedia_multilingual_texts()
    step, tokens_seen = 0, 0
    last_nonzero = torch.zeros(sae.n_latents, dtype=torch.long, device=device)
    t_start = time.time()

    for batch, n_tokens in training_batches(model, tokenizer, text_iter, args.max_length, args.batch_size, device):
        recon, latents = robust_forward(sae, batch, keep_mask)
        loss = (recon - batch).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.unit_norm_decoder()

        fired = torch.zeros(sae.n_latents, dtype=torch.bool, device=device)
        fired[latents.indices[latents.values > 1e-3]] = True
        last_nonzero = torch.where(fired, torch.zeros_like(last_nonzero), last_nonzero + 1)

        tokens_seen += n_tokens
        step += 1

        if step % 20 == 0:
            # The (N, k, n_inputs) gather in robust_forward's decode step is
            # a large, batch-shape-dependent intermediate (N varies with how
            # much of the padded batch was real tokens), which fragments the
            # CUDA caching allocator over many steps until "reserved but
            # unallocated" memory balloons and later steps OOM even though
            # actual live memory is far under capacity -- periodic
            # empty_cache() keeps the reserved pool from growing unbounded.
            torch.cuda.empty_cache()

        if step % args.log_every == 0:
            fvu = (recon - batch).pow(2).sum() / (batch - batch.mean(-1, keepdim=True)).pow(2).sum()
            dead_frac = (last_nonzero > 1000).float().mean().item()
            elapsed = time.time() - t_start
            throughput = tokens_seen / elapsed
            eta_hours = (args.target_tokens - tokens_seen) / throughput / 3600 if throughput > 0 else float("inf")
            print(f"step {step}: tokens={tokens_seen}/{args.target_tokens} "
                  f"({100*tokens_seen/args.target_tokens:.1f}%), loss={loss.item():.4f}, FVE={1-fvu.item():.4f}, "
                  f"dead_frac={dead_frac:.3f}, throughput={throughput:.0f} tok/s, ETA={eta_hours:.2f}h", flush=True)

        if step % args.save_every_steps == 0:
            torch.save(sae.state_dict(), args.save_path)
            json.dump({"step": step, "tokens_seen": tokens_seen, "target_tokens": args.target_tokens,
                       "elapsed_seconds": time.time() - t_start}, open(meta_path, "w"))

        if tokens_seen >= args.target_tokens:
            break

    torch.save(sae.state_dict(), args.save_path)
    json.dump({"step": step, "tokens_seen": tokens_seen, "target_tokens": args.target_tokens,
               "elapsed_seconds": time.time() - t_start, "done": True}, open(meta_path, "w"))
    print(f"Fine-tuning complete: {step} steps, {tokens_seen} tokens. Saved to {args.save_path}")


if __name__ == "__main__":
    main()
