"""Cross-family check of the layer_stats_diagnosis.py finding (BLOOM: residual
norm blows up and SAE reconstruction quality collapses in the last few
layers): does the residual-norm blow-up alone (no SAE needed here, since
these families have no trained MLSAE) also show up in other model families
--Qwen, Gemma, Llama, Mistral?

Generic over any AutoModelForCausalLM: uses output_hidden_states=True instead
of manual per-architecture hooks (unlike bloom.py/pythia.py's approach),
since we only need the residual stream, not selective neuron activations.

For each layer, on a small multilingual probe sample, reports:
  - mean residual norm across valid tokens
  - mean fraction of ||resid||^2 carried by the single largest-magnitude
    dimension (massive-activation-style concentration)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langflow.data import load_probe_corpus


@torch.no_grad()
def profile_model(model_name, languages, n_examples_per_lang, max_length):
    print(f"[{model_name}] loading (cpu, float32)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers

    corpus = load_probe_corpus(languages=languages, n_examples=n_examples_per_lang, min_chars=200)

    sum_norm = torch.zeros(n_layers)
    sum_top1_frac = torch.zeros(n_layers)
    n_tokens = torch.zeros(n_layers)

    n_texts = 0
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            tokens, mask = enc["input_ids"], enc["attention_mask"]
            if tokens.shape[1] < 2:
                continue
            out = model(input_ids=tokens, attention_mask=mask)
            hidden_states = out.hidden_states  # tuple len n_layers+1: [embeddings, layer_0_out, ..., layer_{n-1}_out]
            m = mask[0].bool()
            for l in range(n_layers):
                r = hidden_states[l + 1][0][m]  # (n_valid_tokens, d_model)
                if r.shape[0] == 0:
                    continue
                norms = r.norm(dim=-1)
                sum_norm[l] += norms.sum().item()
                top1 = r.abs().max(dim=-1).values ** 2
                sum_top1_frac[l] += (top1 / (norms ** 2 + 1e-9)).sum().item()
                n_tokens[l] += r.shape[0]
            n_texts += 1
        print(f"  {lang} done ({n_texts} texts so far)")

    results = []
    for l in range(n_layers):
        nt = max(n_tokens[l].item(), 1)
        results.append({
            "layer": l,
            "mean_resid_norm": sum_norm[l].item() / nt,
            "mean_top1_dim_frac_of_norm_sq": sum_top1_frac[l].item() / nt,
        })
    return n_layers, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--languages", default="en,zh")
    parser.add_argument("--n-examples-per-lang", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    languages = args.languages.split(",")
    n_layers, results = profile_model(args.model_name, languages, args.n_examples_per_lang, args.max_length)

    for r in results:
        print(f"  layer {r['layer']:2d}: mean_norm={r['mean_resid_norm']:8.2f}  "
              f"top1_frac={r['mean_top1_dim_frac_of_norm_sq']:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model_name": args.model_name, "n_layers": n_layers, "results": results}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
