"""Flow 2's RQ6: can steering through the attributed latents alone shift
output language as effectively as neuron-level "language arithmetics"
(Gurgurov et al.'s style of steering by amplifying a language-specific
neuron's activation)?

For each target language, take its top LAPE-selected neuron and its
already-computed attributed latents (from `flow2_rq1_overlap.py`'s saved
`per_neuron_results`), then on a set of English prompts compare three
conditions: baseline (no intervention), neuron-level steering (amplify the
neuron), and latent-level steering (add the attributed latents' decoder
directions to the residual stream). Effectiveness is scored as the
log-probability shift toward a set of target-language "marker tokens"
(frequent content tokens sampled from that language's own probe text).
"""

import argparse
import json
import sys
from collections import Counter

sys.path.insert(0, "src")

import torch

from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.models import run_with_cache
from langflow.patching import language_logprob_shift, steer_latents, steer_neuron


def get_marker_tokens(tokenizer, texts: list[str], n_markers: int = 15, min_len: int = 3) -> list[int]:
    """Frequent, language-distinctive token ids from a sample of text in one
    language: tokenize every text, count token frequency, keep the most
    common tokens whose decoded form is at least `min_len` characters (to
    skip single-character/punctuation/whitespace tokens common to every
    language)."""

    counts = Counter()
    for text in texts:
        ids = tokenizer.encode(text)
        counts.update(ids)
    candidates = [
        (tid, c) for tid, c in counts.items() if len(tokenizer.decode([tid]).strip()) >= min_len
    ]
    candidates.sort(key=lambda x: -x[1])
    return [tid for tid, _ in candidates[:n_markers]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="bloom-560m")
    parser.add_argument("--model-name", default="bigscience/bloom-560m")
    parser.add_argument("--mlsae-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--rq1-results", required=True, help="flow2_rq1_overlap.py output, for attributed_latents")
    parser.add_argument("--languages", nargs="+", default=["zh", "ar", "hi", "sw"])
    parser.add_argument("--n-english-prompts", type=int, default=15)
    parser.add_argument("--prompt-length", type=int, default=20)
    parser.add_argument("--n-marker-texts", type=int, default=30)
    parser.add_argument("--n-markers", type=int, default=15)
    parser.add_argument("--n-attributed-latents", type=int, default=20)
    parser.add_argument("--neuron-steer-multiplier", type=float, default=5.0)
    parser.add_argument("--latent-steer-alpha", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = args.device
    print(f"Loading {args.model_key} and its trained MLSAE on {device}...")
    model, tokenizer = load_bloom(args.model_name, device=device)
    from langflow.mlsae import TopKSAE

    config = json.load(open(args.mlsae_config))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    sae.load_state_dict(torch.load(args.mlsae_checkpoint, map_location=device))
    sae.to(device)
    sae.eval()

    rq1 = json.load(open(args.rq1_results))
    per_neuron = {(n["layer"], n["neuron"]): n for n in rq1["per_neuron"]}

    print("Computing per-layer FVU to screen out layers with unreliable SAE reconstruction...")
    calib_text = load_probe_corpus(languages=["en"], n_examples=1, min_chars=400)["en"][0]
    enc = tokenizer(calib_text, return_tensors="pt", truncation=True, max_length=64)
    calib_acts = run_with_cache(model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
    recon, _ = sae(calib_acts.resid)
    layer_fvu = []
    for l in range(calib_acts.resid.shape[0]):
        r, c = calib_acts.resid[l], recon[l]
        fvu = ((c - r).pow(2).sum() / (r - r.mean(-1, keepdim=True)).pow(2).sum()).item()
        layer_fvu.append(fvu)
    fvu_threshold = sorted(layer_fvu)[len(layer_fvu) // 2]  # median FVU across layers
    reliable_layers = {l for l, f in enumerate(layer_fvu) if f <= fvu_threshold}
    print(f"  per-layer FVU: {[round(f, 3) for f in layer_fvu]}")
    print(f"  reliable layers (FVU <= median {fvu_threshold:.3f}): {sorted(reliable_layers)}")

    print(f"Streaming English prompts and {args.n_marker_texts} marker texts per target language...")
    english_corpus = load_probe_corpus(languages=["en"], n_examples=args.n_english_prompts, min_chars=200)["en"]
    marker_corpus = load_probe_corpus(
        languages=args.languages, n_examples=args.n_marker_texts, min_chars=200
    )

    prompts = []
    for text in english_corpus:
        ids = tokenizer.encode(text)[: args.prompt_length]
        prompts.append(torch.tensor([ids], device=device))

    results = {}
    for lang in args.languages:
        # Find this language's best candidate neuron: not forced onto the
        # literal last layer, AND with a target_layer whose SAE reconstruction
        # is reliable enough that its decoder directions are trustworthy for
        # steering (an unreliable layer can make a "small" coefficient produce
        # a saturating, uncontrollable effect regardless of its magnitude).
        candidates = [
            n
            for n in rq1["per_neuron"]
            if n["language"] == lang
            and not n.get("is_last_layer_forced")
            and n["target_layer"] in reliable_layers
        ]
        if not candidates:
            print(f"  {lang}: no candidate neuron with a reliable target_layer found, skipping")
            continue
        candidates.sort(key=lambda n: -len(n["attributed_latents"]))
        target = candidates[0]
        layer, neuron, target_layer = target["layer"], target["neuron"], target["target_layer"]
        attributed = torch.tensor(
            target["attributed_latents"][: args.n_attributed_latents], device=device, dtype=torch.long
        )

        marker_tokens = get_marker_tokens(tokenizer, marker_corpus[lang], n_markers=args.n_markers)
        print(
            f"  {lang}: neuron=L{layer}N{neuron}, target_layer={target_layer}, "
            f"{len(attributed)} attributed latents, {len(marker_tokens)} marker tokens"
        )

        # Calibrate the neuron's "on" activation level from a real forward pass.
        enc = tokenizer(marker_corpus[lang][0], return_tensors="pt", truncation=True, max_length=64)
        acts = run_with_cache(model, enc["input_ids"].to(device), enc["attention_mask"].to(device))
        own_high_activation = acts.neurons[layer, 0, :, neuron].max().item()
        neuron_value = max(own_high_activation, 0.1) * args.neuron_steer_multiplier

        neuron_shifts, latent_shifts = [], []
        for tokens in prompts:
            attention_mask = torch.ones_like(tokens)
            resid_acts = run_with_cache(model, tokens, attention_mask)
            typical_norm = resid_acts.resid[target_layer].norm(dim=-1).mean().item()
            coefficient = args.latent_steer_alpha * typical_norm / max(len(attributed), 1)

            neuron_result = steer_neuron(model, tokens, attention_mask, layer, neuron, value=neuron_value)
            neuron_shifts.append(language_logprob_shift(neuron_result, marker_tokens, position=-1))

            latent_result = steer_latents(
                model, sae, tokens, attention_mask, target_layer, attributed, coefficient=coefficient
            )
            latent_shifts.append(language_logprob_shift(latent_result, marker_tokens, position=-1))

        neuron_mean = sum(neuron_shifts) / len(neuron_shifts)
        latent_mean = sum(latent_shifts) / len(latent_shifts)
        results[lang] = {
            "neuron": [layer, neuron],
            "target_layer": target_layer,
            "n_attributed_latents": len(attributed),
            "n_marker_tokens": len(marker_tokens),
            "neuron_steer_logprob_shift_mean": neuron_mean,
            "latent_steer_logprob_shift_mean": latent_mean,
            "latent_over_neuron_fraction": (latent_mean / neuron_mean) if neuron_mean != 0 else None,
            "n_prompts": len(prompts),
        }
        print(
            f"    neuron steer shift={neuron_mean:.4f}, latent steer shift={latent_mean:.4f}, "
            f"fraction={results[lang]['latent_over_neuron_fraction']}"
        )

    out = {"model": args.model_key, "languages": args.languages, "results": results}
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. Results saved to {args.out}")


if __name__ == "__main__":
    main()
