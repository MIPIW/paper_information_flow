"""Architecture-agnostic accessors and activation capture, shared by both
flows: Pythia (GPTNeoXForCausalLM) and BLOOM (BloomForCausalLM). Both put
their transformer blocks' MLP under `<block>.mlp.dense_h_to_4h` /
`dense_4h_to_h`, so only the path to the block list itself differs.
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedModel


@dataclass
class ModelActivations:
    """Cached activations for one forward pass over a batch of tokens."""

    tokens: Tensor  # (batch, pos)
    resid: Tensor  # (n_layers, batch, pos, d_model), true output of each block
    neurons: Tensor  # (n_layers, batch, pos, d_mlp), MLP activation post-GELU, pre down_proj


@dataclass
class SelectiveActivations:
    """Like `ModelActivations`, but only a chosen handful of (layer, neuron)
    columns from the MLP activation are kept, instead of the full d_mlp-wide
    tensor at every layer. Caching every neuron for every example in a large
    corpus is what OOM'd the whole machine on BLOOM-1.7B (d_mlp=8192, 24
    layers, ~16,500 examples): the full-`neurons` cache alone needs on the
    order of 500GB+ at that scale. Since attribution only ever inspects a
    handful of specific, already-known (layer, neuron) pairs (the neurons
    LAPE selected), caching just those columns cuts memory by roughly
    d_mlp / len(target_neurons_at_that_layer), typically several hundred x.
    """

    tokens: Tensor  # (batch, pos)
    resid: Tensor  # (n_layers, batch, pos, d_model), true output of each block
    neuron_values: dict[tuple[int, int], Tensor]  # (layer, neuron) -> (batch, pos)

    def cpu(self) -> "SelectiveActivations":
        """`run_with_selective_cache` returns tensors on whatever device the
        model ran on. Callers that accumulate one of these per example across
        a whole corpus (rather than using it immediately) must move to CPU
        before appending to that list, or GPU memory fills up exactly the way
        the old dense per-example `neurons` cache used to fill up system RAM."""
        return SelectiveActivations(
            tokens=self.tokens.cpu(),
            resid=self.resid.cpu(),
            neuron_values={k: v.cpu() for k, v in self.neuron_values.items()},
        )


def get_backbone(model: PreTrainedModel) -> torch.nn.Module:
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox
    if hasattr(model, "transformer"):
        return model.transformer
    raise ValueError(f"Unsupported model type: {type(model)}")


def get_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    backbone = get_backbone(model)
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "h"):
        return backbone.h
    raise ValueError(f"Unsupported model type: {type(model)}")


def down_proj_weight(model: PreTrainedModel, layer: int) -> Tensor:
    """The MLP output weight matrix for one layer, shape (d_model, d_mlp).

    Column n is the direction that neuron n writes into the residual stream,
    scaled by that neuron's (post-activation) scalar value.
    """
    return get_layers(model)[layer].mlp.dense_4h_to_h.weight  # (d_model, d_mlp)


@torch.no_grad()
def run_with_cache(
    model: PreTrainedModel,
    tokens: Tensor,
    attention_mask: Tensor | None = None,
) -> ModelActivations:
    """Run a forward pass, capturing every block's true residual-stream output
    (via a forward hook on each transformer block, not `output_hidden_states`,
    whose semantics around the final norm vary across model/library versions)
    and every block's MLP neuron activations (post-GELU, pre down_proj)."""

    layers = get_layers(model)
    neuron_acts: dict[int, Tensor] = {}
    layer_out: dict[int, Tensor] = {}
    hooks = []

    def make_pre_hook(layer_idx: int):
        def pre_hook(_module: torch.nn.Module, args):
            neuron_acts[layer_idx] = args[0]

        return pre_hook

    def make_post_hook(layer_idx: int):
        def post_hook(_module: torch.nn.Module, _args, output):
            layer_out[layer_idx] = output[0] if isinstance(output, tuple) else output

        return post_hook

    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(make_pre_hook(i)))
        hooks.append(layer.register_forward_hook(make_post_hook(i)))

    try:
        get_backbone(model)(input_ids=tokens, attention_mask=attention_mask)
    finally:
        for h in hooks:
            h.remove()

    n_layers = len(layers)
    resid = torch.stack([layer_out[i] for i in range(n_layers)], dim=0)
    neurons = torch.stack([neuron_acts[i] for i in range(n_layers)], dim=0)
    return ModelActivations(tokens=tokens, resid=resid, neurons=neurons)


@torch.no_grad()
def run_with_selective_cache(
    model: PreTrainedModel,
    tokens: Tensor,
    attention_mask: Tensor | None,
    target_neurons: list[tuple[int, int]],
) -> SelectiveActivations:
    """Like `run_with_cache`, but only captures the MLP activation columns
    named in `target_neurons` (a list of (layer, neuron) pairs), not the
    full d_mlp-wide activation at every layer. Full residual stream is still
    captured at every layer, since attribution needs that regardless."""

    layers = get_layers(model)
    neurons_needed_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, neuron in target_neurons:
        neurons_needed_by_layer[layer].append(neuron)

    captured: dict[int, Tensor] = {}
    layer_out: dict[int, Tensor] = {}
    hooks = []

    def make_pre_hook(layer_idx: int):
        def pre_hook(_module: torch.nn.Module, args):
            if layer_idx in neurons_needed_by_layer:
                idx = neurons_needed_by_layer[layer_idx]
                captured[layer_idx] = args[0][..., idx].detach().clone()

        return pre_hook

    def make_post_hook(layer_idx: int):
        def post_hook(_module: torch.nn.Module, _args, output):
            layer_out[layer_idx] = output[0] if isinstance(output, tuple) else output

        return post_hook

    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(make_pre_hook(i)))
        hooks.append(layer.register_forward_hook(make_post_hook(i)))

    try:
        get_backbone(model)(input_ids=tokens, attention_mask=attention_mask)
    finally:
        for h in hooks:
            h.remove()

    n_layers = len(layers)
    resid = torch.stack([layer_out[i] for i in range(n_layers)], dim=0)
    neuron_values: dict[tuple[int, int], Tensor] = {}
    for layer, idx_list in neurons_needed_by_layer.items():
        vals = captured[layer]
        for j, neuron in enumerate(idx_list):
            neuron_values[(layer, neuron)] = vals[..., j]
    return SelectiveActivations(tokens=tokens, resid=resid, neuron_values=neuron_values)


def tokenize_batch(tokenizer, texts: list[str], max_length: int, device: str = "cpu"):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
