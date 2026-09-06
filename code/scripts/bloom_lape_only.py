"""Standalone LAPE neuron identification on a BLOOM checkpoint, independent of
the MLSAE (which may still be training). Lets us see real language-specific-
neuron results for BLOOM well before the day-long MLSAE training finishes,
and gives an early, real check on whether BLOOM's genuine multilingual
pretraining produces cleaner neuron separation than Pythia's incidental
exposure did. Parameterized by model size so it can run on BLOOM-560M (a
scale ablation against BLOOM-1.7B, per flow_2.md's RQ5) as well.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "src")

from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.lape import LapeAccumulator, select_language_neurons
from langflow.models import run_with_cache

MODEL_NAMES = {
    "bloom-560m": "bigscience/bloom-560m",
    "bloom-1b7": "bigscience/bloom-1b7",
}

# BLOOM's own pretraining covers 46 natural languages; this is a representative
# subset spanning several scripts and resource levels.
LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=list(MODEL_NAMES.keys()), default="bloom-1b7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-examples-per-lang", type=int, default=1500)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = args.device
    out_path = args.out or f"results/{args.model_key.replace('-', '')}_lape.json"

    print(f"Loading {args.model_key} on {device}...")
    model, tokenizer = load_bloom(MODEL_NAMES[args.model_key], device=device)

    print(f"Streaming probe corpus: {LANGUAGES}, {args.n_examples_per_lang} examples/language...")
    t0 = time.time()
    corpus = load_probe_corpus(languages=LANGUAGES, n_examples=args.n_examples_per_lang, min_chars=200)
    for lang, texts in corpus.items():
        print(f"  {lang}: {len(texts)} examples")
    print(f"  corpus streamed in {time.time()-t0:.0f}s")

    print("Streaming through every example, accumulating LAPE statistics...")
    print("  (each example's activations are discarded immediately after updating the running")
    print("   sums, since materializing all of them at once OOMs at BLOOM-1.7B's width)")
    t0 = time.time()
    acc = LapeAccumulator(LANGUAGES)
    n_done = 0
    total = sum(len(t) for t in corpus.values())
    for lang, texts in corpus.items():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            acts = run_with_cache(model, tokens, mask)
            acc.update(lang, acts.neurons)
            del acts
            n_done += 1
            if n_done % 2000 == 0:
                print(f"  {n_done}/{total} examples processed, {time.time()-t0:.0f}s elapsed")
    print(f"  LAPE accumulation done in {time.time()-t0:.0f}s")

    print("Finalizing LAPE scores...")
    scores = acc.finalize()
    language_neurons = select_language_neurons(scores, entropy_percentile=0.01)
    print(f"  {len(language_neurons)} language-specific neurons found")

    from collections import Counter

    langs = Counter(n.language for n in language_neurons)
    layers = Counter(n.layer for n in language_neurons)
    n_layers = model.config.n_layer
    print(f"  language distribution: {langs.most_common()}")
    print(f"  layer distribution (of {n_layers} layers):")
    for layer in sorted(layers):
        print(f"    layer {layer}: {layers[layer]} ({100*layers[layer]/len(language_neurons):.1f}%)")

    results = {
        "model": args.model_key,
        "languages": LANGUAGES,
        "n_examples_per_lang": args.n_examples_per_lang,
        "n_language_neurons_found": len(language_neurons),
        "lape_neurons": [
            {"layer": n.layer, "neuron": n.neuron, "language": n.language, "entropy": n.entropy,
             "firing_rate": n.firing_rate_own_language}
            for n in language_neurons
        ],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
