"""Flow 2's RQ4: do languages that share overlapping attributed latents also
sit close together typologically? Gurgurov et al. report that language-
specific neurons overlap more between typologically close languages; this
tests whether the same pattern holds for our attributed latent sets.

For every pair of languages in the RQ1 run, computes the Jaccard overlap of
their pooled attributed-latent sets (from flow2_rq1_overlap.py's saved
per_neuron results) and a typological distance (cosine distance over
lang2vec/URIEL syntactic feature vectors), then correlates the two across
all pairs.
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict

sys.path.insert(0, "src")

import lang2vec.lang2vec as l2v
import numpy as np
from scipy.stats import spearmanr

# ISO 639-1 -> ISO 639-3, since lang2vec expects 639-3 codes.
ISO_639_3 = {
    "en": "eng", "fr": "fra", "es": "spa", "pt": "por", "zh": "cmn",
    "ar": "arb", "vi": "vie", "hi": "hin", "id": "ind", "bn": "ben", "sw": "swh",
}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq1-results", required=True)
    parser.add_argument("--feature-set", default="syntax_knn")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rq1 = json.load(open(args.rq1_results))
    attributed_by_lang: dict[str, set[int]] = defaultdict(set)
    for n in rq1["per_neuron"]:
        if n.get("is_last_layer_forced"):
            continue
        attributed_by_lang[n["language"]].update(n["attributed_latents"])

    languages = sorted(attributed_by_lang.keys())
    print(f"Languages with attributed latents: {languages}")

    iso_codes = [ISO_639_3[lang] for lang in languages]
    features = l2v.get_features(iso_codes, args.feature_set)

    def to_vector(code: str) -> np.ndarray:
        raw = features[code]
        # lang2vec marks missing feature values as "--"; treat as 0 (neutral)
        # for cosine distance, following common practice with sparse URIEL vectors.
        return np.array([0.0 if v == "--" else float(v) for v in raw])

    vectors = {lang: to_vector(ISO_639_3[lang]) for lang in languages}

    pairs = []
    for lang_a, lang_b in itertools.combinations(languages, 2):
        set_a, set_b = attributed_by_lang[lang_a], attributed_by_lang[lang_b]
        latent_overlap = jaccard(set_a, set_b)

        v_a, v_b = vectors[lang_a], vectors[lang_b]
        cos_sim = float(np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-8))
        typological_distance = 1.0 - cos_sim

        pairs.append(
            {
                "language_a": lang_a,
                "language_b": lang_b,
                "latent_jaccard_overlap": latent_overlap,
                "n_attributed_a": len(set_a),
                "n_attributed_b": len(set_b),
                "typological_distance": typological_distance,
            }
        )

    distances = [p["typological_distance"] for p in pairs]
    overlaps = [p["latent_jaccard_overlap"] for p in pairs]
    corr, pvalue = spearmanr(distances, overlaps)

    print(f"\n{len(pairs)} language pairs compared.")
    print("Sorted by typological distance (closest first):")
    for p in sorted(pairs, key=lambda p: p["typological_distance"]):
        print(
            f"  {p['language_a']}-{p['language_b']}: typological_distance={p['typological_distance']:.3f}, "
            f"latent_overlap={p['latent_jaccard_overlap']:.4f}"
        )

    print(f"\nSpearman correlation (typological distance vs. latent overlap): rho={corr:.3f}, p={pvalue:.4f}")
    print("A negative rho means typologically closer languages (smaller distance) share MORE attributed latents,")
    print("matching Gurgurov et al.'s neuron-level finding.")

    out = {
        "feature_set": args.feature_set,
        "languages": languages,
        "pairs": pairs,
        "spearman_rho": corr,
        "spearman_pvalue": pvalue,
    }
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDone. Results saved to {args.out}")


if __name__ == "__main__":
    main()
