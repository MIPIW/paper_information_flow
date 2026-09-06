"""Reproductions of two published methods for scoring SAE latents as
language-specific, used as external reference sets for Flow 2's central RQ1
comparison (does our neuron-to-latent attribution land on the same latents
that these independently-derived methods flag?).

Neither Andrylie et al. nor Deng et al. tested BLOOM or released feature
indices for any model (verified directly against both papers and their
released code), so both methods are reproduced here from their published
formulas and applied to our own trained BLOOM MLSAE.

- SAE-LAPE (Andrylie et al., 2507.11230): identical algorithm to Tang et
  al.'s LAPE, applied to SAE latent "fires" (top-k selection > 0) instead of
  neuron activations. Confirmed structurally identical via the paper's own
  description ("each SAE feature is treated analogously to a neuron").

- Monolinguality (Deng et al., 2505.05111, Equation 3): for feature s and
  language L, nu_s^L = mean_activation_in_L(s) - mean_activation_in_others(s).
  Features are ranked per language by nu, not thresholded.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from langflow.lape import LapeAccumulator, LapeScores, select_language_neurons
from langflow.mlsae import TopKSAE


class SAELapeAccumulator:
    """SAE-LAPE (Andrylie et al.): LAPE's entropy criterion applied to SAE
    latent activations instead of neuron activations. One example at a time,
    streamed, to avoid materializing a full corpus's worth of latent
    activations (the same OOM risk `LapeAccumulator` was built to avoid).
    """

    def __init__(self, sae: TopKSAE, languages: list[str]):
        self.sae = sae
        self.languages = languages
        self._fires_sum = {lang: torch.zeros(sae.n_latents) for lang in languages}
        self._n_tokens = {lang: 0 for lang in languages}

    @torch.no_grad()
    def update(self, language: str, resid: Tensor) -> None:
        """`resid`: (n_layers, 1, pos, d_model) residual stream for one
        unpadded example, exactly what feeds the MLSAE at every layer."""
        latents, _ = self.sae.encode(resid)
        indices_flat = latents.indices.reshape(-1, latents.indices.shape[-1]).cpu()
        n_tok = indices_flat.shape[0]
        fires = torch.ones_like(indices_flat, dtype=torch.float32).reshape(-1)
        self._fires_sum[language].scatter_add_(0, indices_flat.reshape(-1), fires)
        self._n_tokens[language] += n_tok

    def finalize(self, eps: float = 1e-8) -> LapeScores:
        firing_rates = [self._fires_sum[lang] / self._n_tokens[lang] for lang in self.languages]
        firing_rate = torch.stack(firing_rates, dim=-1).unsqueeze(0)  # (1, n_latents, n_languages)
        counts_t = torch.tensor(
            [self._n_tokens[lang] for lang in self.languages], dtype=firing_rate.dtype
        )
        overall_firing_rate = (firing_rate * counts_t).sum(-1) / counts_t.sum()

        probs = firing_rate / (firing_rate.sum(dim=-1, keepdim=True) + eps)
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)
        dominant_language = firing_rate.argmax(dim=-1)

        return LapeScores(
            languages=self.languages,
            firing_rate=firing_rate,
            entropy=entropy,
            dominant_language=dominant_language,
            overall_firing_rate=overall_firing_rate,
        )


class MonolingualityAccumulator:
    """Deng et al.'s monolinguality metric: nu_s^L = mean_activation_in_L(s)
    - mean_activation_in_others(s), where "activation" is the latent's raw
    value when selected in the top-k, and 0 otherwise (matching a sparse
    autoencoder's usual convention that an unselected latent contributes 0).
    """

    def __init__(self, sae: TopKSAE, languages: list[str]):
        self.sae = sae
        self.languages = languages
        self._activation_sum = {lang: torch.zeros(sae.n_latents) for lang in languages}
        self._n_tokens = {lang: 0 for lang in languages}

    @torch.no_grad()
    def update(self, language: str, resid: Tensor) -> None:
        latents, _ = self.sae.encode(resid)
        indices_flat = latents.indices.reshape(-1, latents.indices.shape[-1]).cpu()
        values_flat = latents.values.reshape(-1, latents.values.shape[-1]).cpu()
        n_tok = indices_flat.shape[0]
        self._activation_sum[language].scatter_add_(
            0, indices_flat.reshape(-1), values_flat.reshape(-1)
        )
        self._n_tokens[language] += n_tok

    def finalize(self) -> dict[str, Tensor]:
        """Returns {language: nu} where `nu` has shape (n_latents,)."""
        mean_act = {
            lang: self._activation_sum[lang] / self._n_tokens[lang] for lang in self.languages
        }
        nu = {}
        for lang in self.languages:
            others = torch.stack([mean_act[o] for o in self.languages if o != lang], dim=-1)
            nu[lang] = mean_act[lang] - others.mean(dim=-1)
        return nu


@dataclass
class ScoredLatent:
    latent: int
    language: str
    score: float


def top_sae_lape_latents(
    scores: LapeScores,
    entropy_percentile: float = 0.01,
    k: int = 32,
    n_latents: int | None = None,
) -> list[ScoredLatent]:
    """`select_language_neurons`'s default `min_firing_rate=0.05` assumes
    dense neuron activations, where firing rates well above 5% are common.
    A top-k SAE latent's chance firing rate is only k/n_latents (about 0.05%
    for a 65536-wide, k=32 dictionary), so a latent firing in only one of
    several languages can easily have a *pooled* firing rate under 5% even
    when it is highly language-specific within that one language. We instead
    set the live-latent floor relative to chance level, so a latent only
    needs to fire meaningfully above the rate a uniformly-selected latent
    would by pure luck.
    """
    n_latents = n_latents or scores.entropy.shape[-1]
    chance_rate = k / n_latents
    neurons = select_language_neurons(
        scores,
        entropy_percentile=entropy_percentile,
        min_firing_rate=10 * chance_rate,
        max_firing_rate=0.95,
    )
    return [ScoredLatent(latent=n.neuron, language=n.language, score=n.entropy) for n in neurons]


def top_monolinguality_latents(
    nu: dict[str, Tensor], top_k_per_language: int = 50
) -> list[ScoredLatent]:
    """Deng et al. rank features per language and take the top-ranked ones,
    rather than applying a fixed threshold."""
    selected = []
    for lang, scores in nu.items():
        top_scores, top_indices = scores.topk(top_k_per_language)
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            selected.append(ScoredLatent(latent=idx, language=lang, score=score))
    return selected
