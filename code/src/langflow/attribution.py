"""Attribute a language-specific neuron's causal influence onto MLSAE latents.

Two independent methods, cross-checked against each other (RQ2 / RQ7):

- Weight-based: a closed-form, gradient-free projection. A neuron's direct,
  un-mediated contribution to the residual stream is `activation * W_down[:, n]`
  (residual connections are additive, so this same vector persists unchanged
  into every later layer). We push that vector through the SAE's per-token
  standardization (freezing the standardization's own scale at its observed
  value, the same linearization convention used for LayerNorm in direct-logit-
  attribution / path-patching work) and take its dot product with each
  latent's encoder row. This is a *direct-path* attribution: it ignores
  indirect effects mediated by later nonlinear reads of the neuron's output.

- Gradient-based: an exact local sensitivity, `d(latent pre-activation) /
  d(neuron activation)`, computed by autograd through the real (non-frozen)
  forward pass, in the same spirit as Shu et al.'s GradSAE.

Both operate on plain `resid`/`neuron_act` tensors rather than a specific
activation-cache type, so they work identically whether the caller sourced
those tensors from a dense `ModelActivations` cache or a `SelectiveActivations`
cache that only retains a handful of (layer, neuron) columns (the latter is
what large-corpus, large-model runs must use — see `gather_top_firing` and
`gather_top_firing_selective` below — since caching every neuron for every
token in a large corpus is what OOM'd the whole machine on BLOOM-1.7B).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from langflow.mlsae import TopKSAE, standardize
from langflow.models import ModelActivations, SelectiveActivations, down_proj_weight


def choose_target_layer(neuron_layer: int, n_layers: int) -> tuple[int, bool]:
    """Pick the layer to attribute a neuron's influence onto, avoiding the
    model's last layer whenever an alternative exists.

    The original rule, `target_layer = min(neuron_layer + 2, n_layers - 1)`,
    silently piles neurons from the last few layers onto `n_layers - 1`
    specifically. That layer has the worst SAE reconstruction fidelity (FVU)
    of any layer in every model checked so far (Pythia-70M, Pythia-410M,
    BLOOM-560M, BLOOM-1.7B), plausibly from "massive activation" outlier
    dimensions that concentrate late in a transformer. In BLOOM, where
    language-specific neurons themselves concentrate in the last 2-4 layers,
    most analyzed neurons ended up targeting exactly this volatile layer,
    inflating patching effect sizes in a way that is a property of the layer,
    not of the method.

    Returns `(target_layer, is_last_layer_forced)`. `is_last_layer_forced` is
    True only for a neuron at `n_layers - 1` itself, where no later layer
    exists and the direct effect has nowhere else to land — those neurons
    should be reported separately from the main pooled statistics rather
    than silently mixed in.
    """
    if neuron_layer >= n_layers - 1:
        return n_layers - 1, True
    return min(neuron_layer + 2, n_layers - 2), False


def gather_top_firing(
    acts_list: list[ModelActivations], layer: int, neuron: int, top_n: int
) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    """Find the `top_n` (example, position) pairs across `acts_list` (a dense,
    full-`neurons`-tensor cache) where neuron (layer, neuron) fires strongest.

    Returns `(resid, neuron_act, pairs)`: `resid` has shape
    (n_layers, top_n, 1, d_model), `neuron_act` has shape (top_n, 1), and
    `pairs` are the (example_index, position_index) sources (needed by
    patching, which replays each example's real token sequence).
    """

    candidates = []  # (activation_value, example_index, position_index)
    for ex_idx, acts in enumerate(acts_list):
        vals = acts.neurons[layer, 0, :, neuron]
        for pos in range(vals.shape[0]):
            candidates.append((vals[pos].item(), ex_idx, pos))
    candidates.sort(key=lambda c: -c[0])
    chosen = candidates[:top_n]

    resid_slices, neuron_slices = [], []
    pairs = []
    for _, ex_idx, pos in chosen:
        acts = acts_list[ex_idx]
        resid_slices.append(acts.resid[:, 0, pos, :])  # (n_layers, d_model)
        neuron_slices.append(acts.neurons[layer, 0, pos, neuron])  # scalar
        pairs.append((ex_idx, pos))

    resid = torch.stack(resid_slices, dim=1).unsqueeze(2)  # (n_layers, top_n, 1, d_model)
    neuron_act = torch.stack(neuron_slices).unsqueeze(-1)  # (top_n, 1)
    return resid, neuron_act, pairs


def gather_top_firing_selective(
    acts_list: list[SelectiveActivations], layer: int, neuron: int, top_n: int
) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    """Same as `gather_top_firing`, but reads from a `SelectiveActivations`
    cache (only a few (layer, neuron) columns retained, not the full
    d_mlp-wide activation) — use this whenever the corpus-wide cache was
    built with `run_with_selective_cache` instead of `run_with_cache`."""

    candidates = []
    for ex_idx, acts in enumerate(acts_list):
        vals = acts.neuron_values[(layer, neuron)][0]  # (pos,)
        for pos in range(vals.shape[0]):
            candidates.append((vals[pos].item(), ex_idx, pos))
    candidates.sort(key=lambda c: -c[0])
    chosen = candidates[:top_n]

    resid_slices, neuron_slices = [], []
    pairs = []
    for _, ex_idx, pos in chosen:
        acts = acts_list[ex_idx]
        resid_slices.append(acts.resid[:, 0, pos, :])
        neuron_slices.append(acts.neuron_values[(layer, neuron)][0, pos])
        pairs.append((ex_idx, pos))

    resid = torch.stack(resid_slices, dim=1).unsqueeze(2)
    neuron_act = torch.stack(neuron_slices).unsqueeze(-1)
    return resid, neuron_act, pairs


def selection_hit_rate(
    sae: TopKSAE, resid_at_target_layer: Tensor, latents: Tensor
) -> float:
    """Fraction of tokens for which at least one of `latents` is actually
    among the SAE's selected top-k for that token — a direct check of
    whether ablating `latents` can possibly do anything at all, since
    ablating a latent that was never selected is a no-op by construction."""

    encoded, _ = sae.encode(resid_at_target_layer)
    hit = torch.isin(encoded.indices, latents).any(dim=-1)  # (batch, pos)
    return hit.float().mean().item()


@dataclass
class AttributionResult:
    layer: int
    neuron: int
    target_layer: int
    scores: Tensor  # (n_latents,) attribution of this neuron onto every latent


@torch.no_grad()
def weight_based_attribution(
    model,
    sae: TopKSAE,
    resid_at_own_layer: Tensor,
    neuron_act: Tensor,
    layer: int,
    neuron: int,
    target_layer: int,
) -> AttributionResult:
    """Average, over every real token, the direct linear attribution of
    neuron (layer, neuron) onto every MLSAE latent (the SAE dictionary has
    no layer-specific parameters, so `target_layer` is not otherwise used in
    this computation and is carried through only for bookkeeping/labeling).

    `resid_at_own_layer`: (batch, pos, d_model) residual stream at the
    neuron's OWN layer (`layer`), for the same tokens `neuron_act` is drawn
    from. Earlier versions of this function took the residual at
    `target_layer` here instead, to freeze the standardization's std at the
    value the SAE would actually observe at the point of attribution. That
    was wrong: `target_layer`'s aggregate std reflects everything else that
    accumulated into the residual between `layer` and `target_layer`
    (other neurons, attention, etc.), which has nothing to do with this one
    neuron's own contribution, and borrowing it has no principled
    justification. Using the origin layer's own std is not a perfect
    normalization either (it's still borrowed from the full residual rather
    than from the neuron's contribution alone), but at least it reflects the
    scale the signal actually originates at, not a downstream point it is
    hypothesized to persist into.
    """

    assert target_layer >= layer, "the neuron's direct contribution only reaches later layers"

    w_down_col = down_proj_weight(model, layer)[:, neuron]  # (d_model,)
    contribution = neuron_act.unsqueeze(-1) * w_down_col  # (batch, pos, d_model)

    _, stats = standardize(resid_at_own_layer)

    # The centering step is exactly linear, so the neuron's marginal contribution
    # to the centered vector is its own contribution minus its own mean. The
    # division by std is data-dependent and nonlinear; we freeze it at the
    # value observed at the neuron's own layer rather than recomputing it
    # exactly (which would require isolating this one neuron's marginal
    # effect on a nonlinear denominator).
    centered = contribution - contribution.mean(dim=-1, keepdim=True)
    standardized_contribution = centered / (stats.std + 1e-5)

    scores = standardized_contribution @ sae.encoder.weight.T  # (batch, pos, n_latents)
    return AttributionResult(
        layer=layer,
        neuron=neuron,
        target_layer=target_layer,
        scores=scores.mean(dim=(0, 1)),
    )


def gradient_based_attribution(
    model,
    sae: TopKSAE,
    resid: Tensor,
    neuron_act: Tensor,
    layer: int,
    neuron: int,
    target_layer: int,
    candidate_latents: Tensor,
) -> AttributionResult:
    """Exact `d(latent pre-activation) / d(neuron activation)` for a chosen
    subset of latents (typically the top candidates from the weight-based
    screen), averaged over every real token, backpropagated through the true
    (non-frozen) forward pass of the standardization and the SAE encoder.

    `resid`: (batch, pos, d_model) residual stream at `target_layer`.
    `neuron_act`: (batch, pos) the neuron's activation for the same tokens.

    Restricted to a subset because the full Jacobian is `n_latents` rows
    (tens of thousands for the checkpoints used here), each requiring its
    own backward pass.
    """

    assert target_layer >= layer

    neuron_act = neuron_act.clone().requires_grad_(True)
    w_down_col = down_proj_weight(model, layer)[:, neuron].detach()

    # Residual at target_layer with this neuron's contribution isolated as a
    # differentiable leaf: everything else in the residual stream is treated
    # as a constant, since we only want the sensitivity to this one neuron.
    other = (resid - neuron_act.detach().unsqueeze(-1) * w_down_col).detach()
    resid_reconstructed = other + neuron_act.unsqueeze(-1) * w_down_col

    pre_acts, _ = sae.pre_acts(resid_reconstructed)  # (batch, pos, n_latents)
    pre_acts_summed = pre_acts.sum(dim=(0, 1))[candidate_latents]  # (n_candidates,)

    # One-hot rows select one candidate latent per vmapped backward pass, so
    # row `i` gives exactly d(pre_acts_summed[i]) / d(neuron_act).
    one_hot = torch.eye(len(candidate_latents), device=pre_acts.device)
    grads = torch.autograd.grad(
        pre_acts_summed, neuron_act, grad_outputs=one_hot, is_grads_batched=True
    )[0]  # (n_candidates, batch, pos)

    full_scores = torch.zeros(sae.n_latents, device=pre_acts.device)
    full_scores[candidate_latents] = grads.mean(dim=(1, 2))
    return AttributionResult(
        layer=layer, neuron=neuron, target_layer=target_layer, scores=full_scores
    )


def top_latents(result: AttributionResult, n: int = 20) -> tuple[Tensor, Tensor]:
    values, indices = result.scores.abs().topk(n)
    return result.scores[indices], indices


def jaccard_overlap(a_indices: Tensor, b_indices: Tensor) -> float:
    a, b = set(a_indices.tolist()), set(b_indices.tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
