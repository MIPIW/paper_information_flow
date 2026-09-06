"""Two baselines specified in flow_1.md/flow_2.md's Experimental Plan, run
against the same gathered tokens as our weight-based/gradient-based
attribution so the comparison is apples-to-apples:

- Correlational: Pearson correlation between a neuron's activation and each
  latent's dense (pre-top-k) encoder output, across the same high-firing
  tokens `gather_top_firing_selective` already gathered. Uses neither
  weights nor gradients, just co-activation statistics.

- GradSAE-style (Shu et al.): attributes latents by their gradient
  contribution to the model's OUTPUT rather than to the neuron. This is the
  method flow_1.md's baselines section names directly, and it demonstrates
  what an attribution finds when it starts from the output side instead of
  from the neuron side of the pathway.
"""

import torch
from torch import Tensor
from transformers import PreTrainedModel

from langflow.attribution import AttributionResult
from langflow.mlsae import TopKSAE
from langflow.models import get_layers


@torch.no_grad()
def correlational_attribution(
    sae: TopKSAE, resid: Tensor, neuron_act: Tensor, layer: int, neuron: int, target_layer: int
) -> AttributionResult:
    """Pearson correlation between `neuron_act` and each latent's dense
    encoder pre-activation, across the token dimension. `resid`:
    (batch, pos, d_model) at `target_layer`; `neuron_act`: (batch, pos)."""

    pre_acts, _ = sae.pre_acts(resid)  # (batch, pos, n_latents)
    x = neuron_act.reshape(-1)  # (n_tokens,)
    y = pre_acts.reshape(-1, pre_acts.shape[-1])  # (n_tokens, n_latents)

    x_centered = x - x.mean()
    y_centered = y - y.mean(dim=0, keepdim=True)
    cov = (x_centered.unsqueeze(-1) * y_centered).sum(dim=0)
    x_std = x_centered.pow(2).sum().sqrt()
    y_std = y_centered.pow(2).sum(dim=0).sqrt()
    corr = cov / (x_std * y_std + 1e-8)

    return AttributionResult(layer=layer, neuron=neuron, target_layer=target_layer, scores=corr)


def gradsae_output_attribution(
    model: PreTrainedModel,
    sae: TopKSAE,
    tokens: Tensor,
    attention_mask: Tensor,
    layer: int,
    neuron: int,
    target_layer: int,
    position: int,
) -> AttributionResult:
    """Shu et al.'s GradSAE idea: rank latents by the gradient of the model's
    output with respect to each latent's activation value, computed through
    a real forward pass. Unlike our neuron-to-latent attribution, this never
    looks at the neuron at all; it is purely output-side.

    Implemented with a forward hook that replaces the real residual stream
    at `target_layer` with a reconstruction computed from a dense, leaf
    latent tensor `z` (initialized to exactly the true top-k reconstruction,
    so the forward pass value is unchanged), then backpropagates from the
    output logits at `position` to `z`. The gradient is well-defined for
    every entry of `z`, not just the k that were originally selected, since
    the decode step is a linear map from all of `z`.
    """

    captured = {}

    def hook(_module, _args, output):
        real_resid = output[0] if isinstance(output, tuple) else output
        latents, stats = sae.encode(real_resid)
        dense = torch.zeros(
            *latents.indices.shape[:-1], sae.n_latents, device=real_resid.device
        )
        dense.scatter_(-1, latents.indices, latents.values)
        z = dense.clone().requires_grad_(True)
        captured["z"] = z
        recon = (z @ sae.decoder.weight.T) + sae.pre_encoder_bias
        recon = recon * stats.std + stats.mean
        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon

    handle = get_layers(model)[target_layer].register_forward_hook(hook)
    try:
        logits = model(input_ids=tokens, attention_mask=attention_mask).logits
    finally:
        handle.remove()

    z = captured["z"]
    # Sum of logits at the chosen position, the standard output-side scalar
    # objective for this kind of attribution when no specific target token
    # is singled out.
    objective = logits[:, position, :].sum()
    grad = torch.autograd.grad(objective, z)[0]  # (batch, pos, n_latents)
    scores = grad[:, position, :].mean(dim=0)

    return AttributionResult(
        layer=layer, neuron=neuron, target_layer=target_layer, scores=scores.detach()
    )
