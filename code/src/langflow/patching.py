"""Causal validation: does patching the attributed latents reproduce the effect
of patching the neuron itself? (RQ4)

Two interventions are compared on the same prompts, both implemented the same
way: a forward hook overrides one module's real output during a single real
`model(...)` call, so the library's own masking, rotary embeddings, and batch
handling are used unmodified rather than reimplemented by hand.

  - neuron patching: zero one MLP neuron's activation.
  - latent patching: decode the real residual stream at `target_layer`
    through the MLSAE, zero the attributed latents in that decomposition, and
    re-encode to a modified residual vector that replaces the layer's output.

Both are scored by the resulting shift in output-language log-probability.
"""

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedModel

from langflow.mlsae import TopKLatents, TopKSAE
from langflow.models import get_layers


@dataclass
class PatchResult:
    baseline_logits: Tensor  # (batch, pos, vocab)
    patched_logits: Tensor  # (batch, pos, vocab)


@torch.no_grad()
def baseline_logits(model: PreTrainedModel, tokens: Tensor, attention_mask: Tensor) -> Tensor:
    return model(input_ids=tokens, attention_mask=attention_mask).logits


@torch.no_grad()
def patch_neuron(
    model: PreTrainedModel,
    tokens: Tensor,
    attention_mask: Tensor,
    layer: int,
    neuron: int,
    value: float = 0.0,
) -> PatchResult:
    baseline = baseline_logits(model, tokens, attention_mask)

    def zero_neuron(_module, args):
        h = args[0].clone()
        h[..., neuron] = value
        return (h,) + args[1:]

    handle = get_layers(model)[layer].mlp.dense_4h_to_h.register_forward_pre_hook(
        zero_neuron
    )
    try:
        patched = model(input_ids=tokens, attention_mask=attention_mask).logits
    finally:
        handle.remove()

    return PatchResult(baseline_logits=baseline, patched_logits=patched)


@torch.no_grad()
def patch_latents(
    model: PreTrainedModel,
    sae: TopKSAE,
    tokens: Tensor,
    attention_mask: Tensor,
    target_layer: int,
    latent_indices: Tensor,
) -> PatchResult:
    baseline = baseline_logits(model, tokens, attention_mask)

    def zero_latents(_module, _args, output):
        real_resid = output[0] if isinstance(output, tuple) else output
        latents, stats = sae.encode(real_resid)
        zero_mask = torch.isin(latents.indices, latent_indices)
        patched_values = latents.values.masked_fill(zero_mask, 0.0)
        patched_resid = sae.decode(TopKLatents(patched_values, latents.indices), stats)
        if isinstance(output, tuple):
            return (patched_resid,) + output[1:]
        return patched_resid

    handle = get_layers(model)[target_layer].register_forward_hook(zero_latents)
    try:
        patched = model(input_ids=tokens, attention_mask=attention_mask).logits
    finally:
        handle.remove()

    return PatchResult(baseline_logits=baseline, patched_logits=patched)


@torch.no_grad()
def steer_neuron(
    model: PreTrainedModel,
    tokens: Tensor,
    attention_mask: Tensor,
    layer: int,
    neuron: int,
    value: float,
) -> PatchResult:
    """Set a neuron's activation to a chosen (typically large positive) value
    rather than zeroing it, matching Gurgurov et al.'s "language arithmetics"
    style of steering by amplifying a language-specific neuron's activation,
    not by ablating it. Mechanically identical to `patch_neuron`; separated
    only so steering and ablation experiments read as distinct in call sites."""

    return patch_neuron(model, tokens, attention_mask, layer, neuron, value=value)


@torch.no_grad()
def steer_latents(
    model: PreTrainedModel,
    sae: TopKSAE,
    tokens: Tensor,
    attention_mask: Tensor,
    target_layer: int,
    latent_indices: Tensor,
    coefficient: float,
) -> PatchResult:
    """Add `coefficient` times the sum of the given latents' (unit-norm)
    decoder directions directly to the residual stream at `target_layer`,
    the standard activation-addition style of steering (Turner et al.) as
    opposed to `patch_latents`'s ablate-via-the-SAE's-own-encode/decode
    approach. Bypasses the SAE's top-k selection entirely: the point of
    steering a latent that is not naturally among a token's active top-k is
    to force its direction into the residual stream regardless."""

    baseline = baseline_logits(model, tokens, attention_mask)

    direction = sae.decoder.weight[:, latent_indices].sum(dim=-1) * coefficient  # (n_inputs,)

    def add_direction(_module, _args, output):
        real_resid = output[0] if isinstance(output, tuple) else output
        steered = real_resid + direction
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    handle = get_layers(model)[target_layer].register_forward_hook(add_direction)
    try:
        patched = model(input_ids=tokens, attention_mask=attention_mask).logits
    finally:
        handle.remove()

    return PatchResult(baseline_logits=baseline, patched_logits=patched)


def language_logprob_shift(result: PatchResult, token_ids: list[int], position: int = -1) -> float:
    """Average log-probability shift, at `position`, over a set of tokens that
    mark a candidate output language (e.g. the first content token of a
    translation in that language)."""

    base_logprobs = torch.log_softmax(result.baseline_logits[:, position, :], dim=-1)
    patched_logprobs = torch.log_softmax(result.patched_logits[:, position, :], dim=-1)
    idx = torch.tensor(token_ids, device=result.baseline_logits.device)
    return (patched_logprobs[:, idx] - base_logprobs[:, idx]).mean().item()
