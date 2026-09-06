"""Static PNG plots for the cross-layer persistence sweep
(scripts/cross_layer_persistence.py), saved locally under plots/.

Five figures, all x-axis = layer offset from the neuron's own layer (0 =
neuron's own layer, negative = earlier/input-side layers, positive =
later/output-side layers), BLOOM-560M and BLOOM-1.7B overlaid:

  1. language_specificity_by_offset.png -- normalized language-specific
     activation strength (0-1 per neuron, then averaged) vs offset.
  2. participation_ratio_by_offset.png -- effective dimensionality of the
     candidate latents' activation covariance (PR = (sum lambda)^2 /
     sum(lambda^2)) vs offset; near 1 = monosemantic, higher = superposed.
  3. pairwise_interference_by_offset.png -- Toy-Models-of-Superposition-style
     interference (dot product between unit-norm decoder columns of the
     latents actually active at that layer) vs offset.
  4. steering_effect_by_offset.png -- normalized causal steering effect
     (|logprob shift| toward the target language's marker tokens) vs offset.
  5. best_steer_layer_histogram.png -- distribution of (best-steering-layer
     minus neuron's own layer) across all 44 neurons per model.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = "plots"
COLOR = {"bloom560m": "#2a78d6", "bloom1b7": "#eb6834"}
LABEL = {"bloom560m": "BLOOM-560M", "bloom1b7": "BLOOM-1.7B"}
OFFSET_MIN, OFFSET_MAX = -19, 8

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#c3cad3",
    "axes.labelcolor": "#171b24",
    "text.color": "#171b24",
    "xtick.color": "#5b6472",
    "ytick.color": "#5b6472",
    "axes.grid": True,
    "grid.color": "#dce1e7",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_profiles():
    data = {}
    for m, path in [("bloom560m", "results/cross_layer_bloom560m.json"), ("bloom1b7", "results/cross_layer_bloom1b7.json")]:
        d = json.load(open(path))
        results = d["results"]
        offset_bins = {}
        best_steer_offsets = []
        for r in results:
            own = r["layer"]
            prof = r["layer_profile"]
            spec = [p["language_specificity"] for p in prof]
            pr = [p["participation_ratio"] for p in prof]
            interf = [p["mean_pairwise_interference"] for p in prof]
            steer = [abs(p["steer_shift_mean"]) for p in prof]
            max_spec = max(spec) if max(spec) > 1e-9 else 1.0
            max_steer = max(steer) if max(steer) > 1e-9 else 1.0
            best_steer_offsets.append(max(range(d["n_layers"]), key=lambda l: steer[l]) - own)
            for l in range(d["n_layers"]):
                off = l - own
                b = offset_bins.setdefault(off, {"spec": [], "steer": [], "pr": [], "interf": []})
                b["spec"].append(spec[l] / max_spec)
                b["steer"].append(steer[l] / max_steer)
                if pr[l] == pr[l]:
                    b["pr"].append(pr[l])
                if interf[l] == interf[l]:
                    b["interf"].append(interf[l])
        offs = sorted(o for o in offset_bins if OFFSET_MIN <= o <= OFFSET_MAX)
        data[m] = {
            "offsets": offs,
            "spec": [sum(offset_bins[o]["spec"]) / len(offset_bins[o]["spec"]) for o in offs],
            "steer": [sum(offset_bins[o]["steer"]) / len(offset_bins[o]["steer"]) for o in offs],
            "pr": [sum(offset_bins[o]["pr"]) / len(offset_bins[o]["pr"]) if offset_bins[o]["pr"] else float("nan") for o in offs],
            "interf": [sum(offset_bins[o]["interf"]) / len(offset_bins[o]["interf"]) if offset_bins[o]["interf"] else float("nan") for o in offs],
            "best_steer_offsets": best_steer_offsets,
        }
    return data


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(0, color="#c3cad3", linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xlabel("layer offset from the neuron's own layer (0 = own layer)")


def line_plot(data, key, ylabel, title, subtitle, out_path, ymin_floor=None):
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    for m in ["bloom560m", "bloom1b7"]:
        ax.plot(data[m]["offsets"], data[m][key], color=COLOR[m], linewidth=2.2,
                solid_capstyle="round", label=LABEL[m], marker="o", markersize=3)
    style_axes(ax)
    ax.set_ylabel(ylabel)
    if ymin_floor is not None:
        ax.set_ylim(bottom=ymin_floor)
    ax.legend(frameon=False, loc="best", fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    fig.subplots_adjust(top=0.80, left=0.11, right=0.97, bottom=0.14)
    fig.text(0.045, 0.965, title, fontsize=14.5, fontweight="bold", ha="left", va="top", color="#171b24")
    fig.text(0.045, 0.905, subtitle, fontsize=9.8, ha="left", va="top", color="#5b6472", wrap=True)
    fig.savefig(out_path)
    plt.close(fig)
    print("saved", out_path)


def histogram_plot(data, out_path):
    bins = [-24, -20, -16, -12, -8, -4, 0, 4, 8, 12, 16]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    width = 1.6
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    for i, m in enumerate(["bloom560m", "bloom1b7"]):
        counts = [sum(1 for o in data[m]["best_steer_offsets"] if bins[j] <= o < bins[j + 1]) for j in range(len(bins) - 1)]
        offset = (-1) ** i * (width / 2 + 0.15)
        ax.bar([c + offset for c in centers], counts, width=width, color=COLOR[m], label=LABEL[m], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#c3cad3", linewidth=1, linestyle=(0, (3, 3)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("offset of the best-steering layer from the neuron's own layer (bin width 4)")
    ax.set_ylabel("number of neurons")
    ax.legend(frameon=False, fontsize=10)
    fig.subplots_adjust(top=0.80, left=0.11, right=0.97, bottom=0.14)
    fig.text(0.045, 0.965, "5. Distribution of the best-steering-layer offset", fontsize=14.5, fontweight="bold", ha="left", va="top", color="#171b24")
    fig.text(0.045, 0.905, "layer (of 24) where steering had the largest effect, minus the neuron's own layer (44 neurons/model)",
             fontsize=9.8, ha="left", va="top", color="#5b6472")
    fig.savefig(out_path)
    plt.close(fig)
    print("saved", out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_profiles()

    line_plot(data, "spec", "language specificity (normalized, 0-1)",
              "1. Language specificity vs layer offset",
              "how distinctively the candidate latents respond to this language vs. a cross-language background, at that layer",
              os.path.join(OUT_DIR, "1_language_specificity_by_offset.png"), ymin_floor=0)

    line_plot(data, "pr", "Participation Ratio",
              "2. Participation Ratio vs layer offset",
              "near 1 = monosemantic (concentrated in one latent), higher = superposed (spread across many)",
              os.path.join(OUT_DIR, "2_participation_ratio_by_offset.png"))

    line_plot(data, "interf", "mean pairwise interference",
              "3. Pairwise decoder-direction interference vs layer offset",
              "Toy Models of Superposition (2209.10652) style: dot product between unit-norm decoder directions. 0 = orthogonal, higher = interfering",
              os.path.join(OUT_DIR, "3_pairwise_interference_by_offset.png"), ymin_floor=0)

    line_plot(data, "steer", "|steering effect| (normalized, 0-1)",
              "4. Steering causal effect vs layer offset",
              "log-probability shift toward the target language's marker tokens when injecting the candidate latents' direction at that layer",
              os.path.join(OUT_DIR, "4_steering_effect_by_offset.png"), ymin_floor=0)

    histogram_plot(data, os.path.join(OUT_DIR, "5_best_steer_layer_histogram.png"))


if __name__ == "__main__":
    main()
