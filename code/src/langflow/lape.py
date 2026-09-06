"""Language Activation Probability Entropy (Tang et al., 2024, arXiv:2402.16438).

For each neuron, compute how often it fires (post-activation > 0) on tokens of
each language, normalize that firing-rate vector into a probability
distribution over languages, and take its entropy. A neuron with low entropy
fires almost exclusively for one language; a neuron with high entropy fires
about equally often regardless of language.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from langflow.pythia import PythiaActivations


@dataclass
class LapeScores:
    languages: list[str]
    firing_rate: Tensor  # (n_layers, n_neurons, n_languages), P(neuron fires | language)
    entropy: Tensor  # (n_layers, n_neurons), entropy of firing_rate normalized over languages
    dominant_language: Tensor  # (n_layers, n_neurons), argmax index into `languages`
    overall_firing_rate: Tensor  # (n_layers, n_neurons), P(neuron fires) pooled over all languages


class LapeAccumulator:
    """Incrementally accumulate per-language firing-rate statistics one
    example at a time, so a large corpus never needs every example's
    activations held in memory at once (materializing them all, even on CPU,
    is what OOMed on BLOOM-1.7B: ~1TB for 16,500 examples at that model's
    width). Call `update` per example, discard the example's activations
    immediately after, and call `finalize` once at the end.
    """

    def __init__(self, languages: list[str]):
        self.languages = languages
        self._fires_sum: dict[str, Tensor | None] = {lang: None for lang in languages}
        self._n_tokens: dict[str, int] = {lang: 0 for lang in languages}

    @torch.no_grad()
    def update(self, language: str, neurons: Tensor) -> None:
        """`neurons`: (n_layers, 1, pos, n_neurons) for one unpadded example."""
        fires = (neurons > 0).float().sum(dim=(1, 2))  # (n_layers, n_neurons)
        prev = self._fires_sum[language]
        self._fires_sum[language] = fires if prev is None else prev + fires
        self._n_tokens[language] += neurons.shape[2]

    def finalize(self, eps: float = 1e-8) -> LapeScores:
        firing_rates = [self._fires_sum[lang] / self._n_tokens[lang] for lang in self.languages]
        firing_rate = torch.stack(firing_rates, dim=-1)  # (n_layers, n_neurons, n_languages)
        counts_t = torch.tensor(
            [self._n_tokens[lang] for lang in self.languages],
            dtype=firing_rate.dtype,
            device=firing_rate.device,
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


@torch.no_grad()
def compute_lape_scores(
    acts_by_language: dict[str, list[PythiaActivations]], eps: float = 1e-8
) -> LapeScores:
    """`acts_by_language[lang]` holds one `PythiaActivations` per example in
    that language, each an unpadded single sequence (batch dim 1), so that
    every token counted is real text and padding never biases firing rates.

    Convenience wrapper around `LapeAccumulator` for corpora small enough to
    hold in memory all at once; use `LapeAccumulator` directly for larger runs.
    """

    acc = LapeAccumulator(list(acts_by_language.keys()))
    for lang, acts_list in acts_by_language.items():
        for acts in acts_list:
            acc.update(lang, acts.neurons)
    return acc.finalize(eps)


@dataclass
class LanguageNeuron:
    layer: int
    neuron: int
    language: str
    entropy: float
    firing_rate_own_language: float


def select_language_neurons(
    scores: LapeScores,
    entropy_percentile: float = 0.01,
    min_firing_rate: float = 0.05,
    max_firing_rate: float = 0.95,
) -> list[LanguageNeuron]:
    """Follow Tang et al.'s selection rule: take the lowest-entropy neurons
    among those that fire often enough to be meaningfully "on" but not so
    often that they are just always-active, language-agnostic neurons."""

    live = (scores.overall_firing_rate > min_firing_rate) & (
        scores.overall_firing_rate < max_firing_rate
    )
    live_entropy = scores.entropy[live]
    threshold = torch.quantile(live_entropy, entropy_percentile)

    selected = []
    n_layers, n_neurons = scores.entropy.shape
    for layer in range(n_layers):
        for neuron in range(n_neurons):
            if not live[layer, neuron] or scores.entropy[layer, neuron] > threshold:
                continue
            lang_idx = scores.dominant_language[layer, neuron].item()
            selected.append(
                LanguageNeuron(
                    layer=layer,
                    neuron=neuron,
                    language=scores.languages[lang_idx],
                    entropy=scores.entropy[layer, neuron].item(),
                    firing_rate_own_language=scores.firing_rate[
                        layer, neuron, lang_idx
                    ].item(),
                )
            )
    selected.sort(key=lambda n: n.entropy)
    return selected
