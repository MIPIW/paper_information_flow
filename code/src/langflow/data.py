"""A small real multilingual probe corpus, streamed from Wikipedia so no gated
dataset access is required. Good enough to exercise the full pipeline; the
real experimental plan (flow_1.md) calls for a Pile-derived multilingual
subset plus Flores-200 translations, which is a separate, larger data step.
"""

from datasets import load_dataset

WIKIPEDIA_DUMP = "20231101"

# `wikimedia/wikipedia` config suffixes match ISO 639-1 codes directly for
# every language we use; keep this map only for the rare exception.
LANGUAGE_OVERRIDES = {}

DEFAULT_LANGUAGES = ["en", "fr", "de", "es"]


def load_probe_corpus(
    languages: list[str] | None = None, n_examples: int = 20, min_chars: int = 200
) -> dict[str, list[str]]:
    """Stream `n_examples` Wikipedia articles per language, each truncated to a
    single paragraph-scale chunk of at least `min_chars` characters."""

    languages = languages or DEFAULT_LANGUAGES
    corpus: dict[str, list[str]] = {}
    for lang in languages:
        config = f"{WIKIPEDIA_DUMP}.{LANGUAGE_OVERRIDES.get(lang, lang)}"
        ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
        texts = []
        for example in ds:
            text = example["text"].strip()
            if len(text) < min_chars:
                continue
            texts.append(text[: min_chars * 3])
            if len(texts) >= n_examples:
                break
        corpus[lang] = texts
    return corpus
