"""Heatmap of steering causal effect over (neuron's own layer x absolute
target layer), from the own_layer-stratified sample in steer_heatmap.py --
this is the plot that disentangles "offset from own layer" from "absolute
layer" (see conversation: the earlier 44-neuron sample was 64% own_layer
22-23, so offset and absolute layer were nearly collinear there).

Smoothing: own_layer is sparsely and irregularly sampled (0,1,2,3,12,14-23),
so the heatmap is built with Gaussian-kernel-weighted (Nadaraya-Watson)
smoothing across own_layer for each target_layer row, onto a fine grid --
not a naive per-cell average, which would leave gaps.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "plots"
SIGMA = 2.2  # own_layer smoothing bandwidth, ~1 typical gap between sampled layers


def smooth_heatmap(results, n_layers, grid_size=96):
    own_layers = np.array([r["own_layer"] for r in results], dtype=float)  # (n_neurons,)
    shifts = np.array([r["shifts_abs_norm"] for r in results])  # (n_neurons, n_layers)

    grid = np.linspace(0, n_layers - 1, grid_size)  # fine own_layer axis
    # weights[g, n] = kernel(grid[g], own_layers[n])
    diffs = grid[:, None] - own_layers[None, :]
    weights = np.exp(-(diffs ** 2) / (2 * SIGMA ** 2))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12

    # smoothed[g, l] = weighted mean over neurons of shifts[n, l]
    smoothed = weights @ shifts  # (grid_size, n_layers)
    return grid, smoothed, own_layers


def plot_model(model_key, label, out_path):
    d = json.load(open(f"results/steer_heatmap_{model_key}.json"))
    n_layers = d["n_layers"]
    grid, smoothed, own_layers = smooth_heatmap(d["results"], n_layers)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=200)
    im = ax.imshow(
        smoothed.T, origin="lower", aspect="auto", cmap="magma",
        extent=[grid.min(), grid.max(), -0.5, n_layers - 0.5],
        vmin=0, vmax=1,
    )
    # diagonal: target_layer == own_layer (offset 0)
    ax.plot([0, n_layers - 1], [0, n_layers - 1], color="#8fd6ff", linewidth=1.4, linestyle=(0, (4, 3)), label="target layer = own layer (offset 0)")
    # rug of actually-sampled own_layer values
    ax.scatter(own_layers, np.full_like(own_layers, -1.4), marker="^", s=22, color="white", edgecolor="#171b24", linewidth=0.5, clip_on=False, zorder=5, label="sampled neuron (own layer)")

    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(-2.0, n_layers - 0.5)
    ax.set_xlabel("neuron's own layer")
    ax.set_ylabel("absolute target layer (where steering is applied)")
    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("|steering effect| (normalized per neuron, 0-1)")
    ax.legend(loc="upper left", frameon=True, fontsize=9, facecolor="white", framealpha=0.85)

    fig.subplots_adjust(top=0.85, left=0.10, right=1.0, bottom=0.11)
    fig.text(0.045, 0.965, f"Steering effect: own layer x target layer -- {label}", fontsize=14.5, fontweight="bold", ha="left", va="top", color="#171b24")
    fig.text(0.045, 0.915, "Gaussian-smoothed (sigma=2.2) over own_layer; white triangles mark the neurons actually sampled",
             fontsize=9.8, ha="left", va="top", color="#5b6472")
    fig.savefig(out_path)
    plt.close(fig)
    print("saved", out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_model("bloom560m", "BLOOM-560M", os.path.join(OUT_DIR, "6_steer_heatmap_bloom560m.png"))
    plot_model("bloom1b7", "BLOOM-1.7B", os.path.join(OUT_DIR, "7_steer_heatmap_bloom1b7.png"))


if __name__ == "__main__":
    main()
