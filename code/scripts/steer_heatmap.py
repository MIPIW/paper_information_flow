"""Disentangle "offset from own layer" from "absolute layer" in the steering
causal effect (see cross_layer_persistence.py and the confound flagged in
conversation: the earlier 44-neuron sample was 64% concentrated in own_layer
22-23, so offset and absolute layer were nearly collinear there and the
"steering favors very negative offsets" finding could equally be "steering
favors early absolute layers" -- the two could not be told apart).

Selects a NEW sample of neurons stratified across own_layer (not top-K per
language, which reproduces the back-loaded bias), then sweeps steering
effect across every absolute layer for each. Output is raw (own_layer,
target_layer, steer_shift_norm) triples -- plotting/smoothing happens in
plot_steer_heatmap.py.
"""

import argparse
import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, "src")

import torch

from langflow.attribution import gather_top_firing_selective, top_latents, weight_based_attribution
from langflow.bloom import load_bloom
from langflow.data import load_probe_corpus
from langflow.mlsae import TopKSAE
from langflow.models import get_layers, run_with_selective_cache
from langflow.patching import language_logprob_shift, steer_latents

BLOOM_LANGUAGES = ["en", "fr", "es", "pt", "zh", "ar", "vi", "hi", "id", "bn", "sw"]


def get_marker_tokens(tokenizer, texts, n_markers=15, min_len=3):
    from collections import Counter
    counts = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text))
    candidates = [(tid, c) for tid, c in counts.items() if len(tokenizer.decode([tid]).strip()) >= min_len]
    candidates.sort(key=lambda x: -x[1])
    return [tid for tid, _ in candidates[:n_markers]]


def left_pad_batch(token_lists, pad_id, device):
    maxlen = max(len(t) for t in token_lists)
    input_ids = torch.full((len(token_lists), maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(token_lists), maxlen), dtype=torch.long)
    for i, t in enumerate(token_lists):
        input_ids[i, maxlen - len(t):] = torch.tensor(t)
        attention_mask[i, maxlen - len(t):] = 1
    return input_ids.to(device), attention_mask.to(device)


@torch.no_grad()
def resid_at_layer(model, tokens, attention_mask, layer):
    captured = {}

    def hook(_module, _args, output):
        captured["resid"] = output[0] if isinstance(output, tuple) else output

    handle = get_layers(model)[layer].register_forward_hook(hook)
    try:
        model(input_ids=tokens, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["resid"]


def stratified_select(lape_top, n_layers, per_layer_cap_late=3, late_threshold=18):
    """All candidates from layers below `late_threshold` (where LAPE-selected
    neurons are naturally sparse in BLOOM), capped at `per_layer_cap_late`
    per layer from `late_threshold` upward (where they are naturally dense)
    -- trades the "top-K per language" bias for a sample spread across
    own_layer, at a similar total neuron budget."""
    by_layer = defaultdict(list)
    for n in lape_top:
        by_layer[n["layer"]].append(n)
    selected = []
    for layer in sorted(by_layer.keys()):
        cands = sorted(by_layer[layer], key=lambda n: n["entropy"])
        if layer < late_threshold:
            selected.extend(cands)
        else:
            selected.extend(cands[:per_layer_cap_late])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bloom-560m", "bloom-1b7"], required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mlsae-checkpoint", required=True)
    parser.add_argument("--mlsae-config", required=True)
    parser.add_argument("--existing-results", required=True)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--n-examples-per-lang", type=int, default=1500)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-steer-prompts", type=int, default=8)
    parser.add_argument("--steer-alpha", type=float, default=6.0)
    parser.add_argument("--per-layer-cap-late", type=int, default=3)
    parser.add_argument("--late-threshold", type=int, default=18)
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
    target_neurons_list = stratified_select(existing["lape_top"], n_layers, args.per_layer_cap_late, args.late_threshold)
    from collections import Counter
    print(f"Selected {len(target_neurons_list)} neurons, own_layer distribution: "
          f"{sorted(Counter(n['layer'] for n in target_neurons_list).items())}")

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
        tn = neurons_by_lang.get(lang, [])
        if not tn:
            continue
        acts_by_language[lang] = []
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            sel = run_with_selective_cache(model, tokens, mask, target_neurons=tn)
            acts_by_language[lang].append(sel.cpu())
    print(f"  selective caching done in {time.time()-t0:.0f}s")

    english_corpus = load_probe_corpus(languages=["en"], n_examples=args.n_steer_prompts, min_chars=200)["en"]
    steer_token_lists = [tokenizer.encode(text)[:20] for text in english_corpus]
    steer_input_ids, steer_attention_mask = left_pad_batch(steer_token_lists, pad_id, device)

    by_lang_for_markers = defaultdict(list)
    for n in target_neurons_list:
        by_lang_for_markers[n["language"]].append(n)
    marker_corpus = load_probe_corpus(languages=list(by_lang_for_markers.keys()), n_examples=30, min_chars=200)

    print(f"\nSweeping steering effect on {len(target_neurons_list)} neurons x {n_layers} layers...")
    results = []
    t0 = time.time()
    for i, target in enumerate(target_neurons_list):
        own_layer, neuron, language = target["layer"], target["neuron"], target["language"]
        lang_acts = acts_by_language.get(language)
        if not lang_acts:
            continue

        resid, neuron_act, pairs = gather_top_firing_selective(lang_acts, own_layer, neuron, top_n=40)
        resid, neuron_act = resid.to(device), neuron_act.to(device)
        w_result = weight_based_attribution(model, sae, resid[own_layer], neuron_act, own_layer, neuron, own_layer)
        _, candidates = top_latents(w_result, n=args.n_candidates)

        marker_tokens = get_marker_tokens(tokenizer, marker_corpus[language], n_markers=15)

        shifts = []
        for l in range(n_layers):
            typical_norm = resid[l].norm(dim=-1).mean().item()
            coeff = args.steer_alpha * typical_norm / max(args.n_candidates, 1)
            result = steer_latents(model, sae, steer_input_ids, steer_attention_mask, l, candidates, coefficient=coeff)
            shifts.append(language_logprob_shift(result, marker_tokens, position=-1))

        max_abs = max(abs(s) for s in shifts) or 1.0
        results.append({
            "own_layer": own_layer, "neuron": neuron, "language": language,
            "shifts_abs_norm": [abs(s) / max_abs for s in shifts],
        })

        # Incremental checkpoint after every neuron: write-to-temp-then-rename
        # so a crash/kill mid-write never corrupts the on-disk file, and a
        # partial run (killed after neuron 30/48, say) still leaves usable
        # data for those 30 rather than losing the whole run.
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        tmp_path = args.out + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"model": args.model, "n_layers": n_layers, "results": results}, f, indent=2)
        os.replace(tmp_path, args.out)

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(target_neurons_list)} neurons swept, {time.time()-t0:.0f}s elapsed "
                  f"(checkpointed to {args.out})")

    print(f"\nDone. {len(results)} neurons saved to {args.out}")


if __name__ == "__main__":
    main()
