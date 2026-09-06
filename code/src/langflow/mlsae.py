"""A TopK sparse autoencoder matching Lawson et al.'s released MLSAE checkpoints.

Reimplemented directly from the checkpoint's state dict and from
https://github.com/tim-lawson/mlsae (mlsae/model/autoencoders/topk.py), rather
than depending on the `mlsae` package, since only inference is needed here:

  encoder.weight: (n_latents, n_inputs), no bias
  decoder.weight: (n_inputs, n_latents), no bias
  pre_encoder_bias: (n_inputs,)

Forward pass: per-token standardize -> subtract pre_encoder_bias -> encode ->
keep top-k -> ReLU -> decode with only the k latents -> add back
pre_encoder_bias -> un-standardize.
"""

from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn


@dataclass
class Standardization:
    mean: Tensor  # (..., 1)
    std: Tensor  # (..., 1)


@dataclass
class TopKLatents:
    values: Tensor  # (..., k), post-ReLU
    indices: Tensor  # (..., k)


def standardize(x: Tensor, eps: float = 1e-5) -> tuple[Tensor, Standardization]:
    mean = x.mean(dim=-1, keepdim=True)
    x = x - mean
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, Standardization(mean, std)


class TopKSAE(nn.Module):
    def __init__(self, n_inputs: int, n_latents: int, k: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_latents = n_latents
        self.k = k
        self.encoder = nn.Linear(n_inputs, n_latents, bias=False)
        self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.pre_encoder_bias = nn.Parameter(torch.zeros(n_inputs))

        # Tied init, then unit-normalize: matches Lawson et al.'s training setup.
        self.decoder.weight.data = self.encoder.weight.data.T.clone().contiguous()
        self.unit_norm_decoder()

    def pre_acts(self, resid: Tensor) -> tuple[Tensor, Standardization]:
        """Encoder pre-activations (before top-k / ReLU), and the standardization
        stats used, so callers can invert them for reconstruction."""
        x, stats = standardize(resid)
        pre_acts = self.encoder(x - self.pre_encoder_bias)
        return pre_acts, stats

    def encode(self, resid: Tensor) -> tuple[TopKLatents, Standardization]:
        pre_acts, stats = self.pre_acts(resid)
        values, indices = torch.topk(pre_acts, k=self.k, dim=-1, sorted=False)
        values = torch.relu(values)
        return TopKLatents(values, indices), stats

    def decode(self, latents: TopKLatents, stats: Standardization) -> Tensor:
        # Gather only the k active decoder directions per token instead of
        # materializing a dense (..., n_latents) tensor: at real training
        # batch sizes with n_latents in the tens of thousands, the dense
        # version allocates tens of GB and OOMs, even though only k of those
        # columns are ever nonzero.
        gathered = torch.nn.functional.embedding(latents.indices, self.decoder.weight.T)  # (..., k, n_inputs)
        recon = (gathered * latents.values.unsqueeze(-1)).sum(dim=-2) + self.pre_encoder_bias
        return recon * stats.std + stats.mean

    @torch.no_grad()
    def unit_norm_decoder(self) -> None:
        """Renormalize each decoder latent direction to unit norm. Lawson et
        al.'s checkpoints are trained this way (decoder columns are directions,
        not free-scaled), and it should be called after every optimizer step
        during training, not just once at init."""
        self.decoder.weight.data /= self.decoder.weight.data.norm(dim=0, keepdim=True)

    def forward(self, resid: Tensor) -> tuple[Tensor, TopKLatents]:
        latents, stats = self.encode(resid)
        recon = self.decode(latents, stats)
        return recon, latents


def load_mlsae(repo_id: str, device: str = "cpu") -> TopKSAE:
    config_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "model.safetensors")

    import json

    config = json.load(open(config_path))
    sae = TopKSAE(n_inputs=config["n_inputs"], n_latents=config["n_latents"], k=config["k"])
    state_dict = load_file(weights_path)
    sae.load_state_dict(state_dict, strict=True)
    sae.to(device)
    sae.eval()
    return sae
