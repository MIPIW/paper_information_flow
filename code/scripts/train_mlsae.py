"""Train a multi-layer SAE from scratch, streaming real training data rather
than materializing a fixed corpus, so the token budget can scale to whatever
compute is available (Lawson et al.'s own MLSAEs were trained on ~1B tokens).

Standard TopK SAE training loop: MSE reconstruction loss, Adam, unit-norm the
decoder after every step. Deliberately does not implement the dead-latent
auxiliary loss from the original mlsae repo (mlsae/model/autoencoders/topk.py) —
a reasonable simplification for a first checkpoint; dead-latent fraction is
logged so it's visible whether this simplification is actually costing
anything at real scale.

Checkpoints periodically (state_dict + a small JSON of run metadata) so a
long run is resumable and inspectable without waiting for it to finish.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

import torch
from datasets import load_dataset

from langflow.bloom import load_bloom
from langflow.mlsae import TopKSAE
from langflow.models import run_with_cache
from langflow.pythia import load_pythia

MODEL_LOADERS = {
    "pythia-70m": lambda device: load_pythia("EleutherAI/pythia-70m-deduped", device=device),
    "pythia-410m": lambda device: load_pythia("EleutherAI/pythia-410m-deduped", device=device),
    "pythia-1.4b": lambda device: load_pythia("EleutherAI/pythia-1.4b-deduped", device=device),
    "bloom-560m": lambda device: load_bloom("bigscience/bloom-560m", device=device),
    "bloom-1b7": lambda device: load_bloom("bigscience/bloom-1b7", device=device),
}

# A real, ungated streaming source per model family. Pythia was trained on the
# Pile, so training its MLSAE on the same distribution matches Lawson et al.
# exactly. BLOOM's own pretraining corpus (ROOTS) is not conveniently
# streamable here, so we substitute a multilingual Wikipedia mix across
# BLOOM's language list as a practical, ungated stand-in (flagged in flow_2.md
# as a scope simplification versus the real ROOTS-derived plan).
PYTHIA_CORPUS = ("monology/pile-uncopyrighted", None)
BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn"]


def stream_pile_texts(min_chars: int = 200):
    ds = load_dataset(PYTHIA_CORPUS[0], split="train", streaming=True)
    for example in ds:
        text = example["text"].strip()
        if len(text) >= min_chars:
            yield text


def stream_wikipedia_multilingual_texts(min_chars: int = 200):
    from itertools import cycle

    streams = [
        iter(load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True))
        for lang in BLOOM_LANGUAGES
    ]
    for stream in cycle(streams):
        try:
            example = next(stream)
        except StopIteration:
            continue
        text = example["text"].strip()
        if len(text) >= min_chars:
            yield text


def batched_texts(text_iter, batch_size: int):
    batch = []
    for text in text_iter:
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []


def training_batches(model, tokenizer, text_iter, max_length: int, batch_size: int, device: str):
    """Yield (n_real_tokens_across_layers, d_model) flattened activation
    batches, pooling every layer together as MLSAE training requires."""

    for batch_texts in batched_texts(text_iter, batch_size):
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )
        tokens, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        acts = run_with_cache(model, tokens, mask)
        real_mask = mask.bool()  # (batch, pos)
        resid = acts.resid[:, real_mask, :]  # (n_layers, n_real_tokens, d_model)
        n_tokens_this_batch = real_mask.sum().item() * resid.shape[0]
        yield resid.reshape(-1, resid.shape[-1]), n_tokens_this_batch


def train(
    model_key: str,
    expansion_factor: int,
    k: int,
    target_tokens: int,
    batch_size: int,
    max_length: int,
    lr: float,
    device: str,
    save_path: str,
    log_every: int,
    save_every_steps: int,
):
    model, tokenizer = MODEL_LOADERS[model_key](device)
    d_model = model.config.hidden_size
    sae = TopKSAE(n_inputs=d_model, n_latents=d_model * expansion_factor, k=k)
    sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    text_iter = (
        stream_pile_texts() if model_key.startswith("pythia") else stream_wikipedia_multilingual_texts()
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    meta_path = save_path + ".meta.json"
    config = {"n_inputs": d_model, "n_latents": d_model * expansion_factor, "k": k}
    json.dump(config, open(save_path + ".config.json", "w"))

    step = 0
    tokens_seen = 0
    last_nonzero = torch.zeros(sae.n_latents, dtype=torch.long, device=device)
    t_start = time.time()

    for batch, n_tokens in training_batches(model, tokenizer, text_iter, max_length, batch_size, device):
        recon, latents = sae(batch)
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

        if step % log_every == 0:
            fvu = (recon - batch).pow(2).sum() / (batch - batch.mean(-1, keepdim=True)).pow(2).sum()
            dead_frac = (last_nonzero > 1000).float().mean().item()
            elapsed = time.time() - t_start
            throughput = tokens_seen / elapsed
            eta_hours = (target_tokens - tokens_seen) / throughput / 3600 if throughput > 0 else float("inf")
            print(
                f"step {step}: tokens={tokens_seen}/{target_tokens} ({100*tokens_seen/target_tokens:.1f}%), "
                f"loss={loss.item():.4f}, FVU={fvu.item():.4f}, dead_frac={dead_frac:.3f}, "
                f"throughput={throughput:.0f} tok/s, ETA={eta_hours:.1f}h",
                flush=True,
            )

        if step % save_every_steps == 0:
            torch.save(sae.state_dict(), save_path)
            json.dump(
                {"step": step, "tokens_seen": tokens_seen, "target_tokens": target_tokens,
                 "elapsed_seconds": time.time() - t_start},
                open(meta_path, "w"),
            )

        if tokens_seen >= target_tokens:
            break

    torch.save(sae.state_dict(), save_path)
    json.dump(
        {"step": step, "tokens_seen": tokens_seen, "target_tokens": target_tokens,
         "elapsed_seconds": time.time() - t_start, "done": True},
        open(meta_path, "w"),
    )
    print(f"Training complete: {step} steps, {tokens_seen} tokens. Saved to {save_path}")
    return sae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=list(MODEL_LOADERS.keys()), required=True)
    parser.add_argument("--expansion-factor", type=int, default=64)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every-steps", type=int, default=500)
    args = parser.parse_args()

    train(
        model_key=args.model_key,
        expansion_factor=args.expansion_factor,
        k=args.k,
        target_tokens=args.target_tokens,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path,
        log_every=args.log_every,
        save_every_steps=args.save_every_steps,
    )
