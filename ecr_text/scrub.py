"""Remove or tokenize identifiers that survived GM's masking.

GM's masked columns replace programs, part numbers and amounts with tokens such as
``[Program 276]``, ``[12345678]`` and ``[AMT]``, but the masking is incomplete: raw
program codes (``PROG 12345``), WO/ECR/PCCR numbers, unmasked part numbers, e-mails,
phone numbers and personal names still appear in the text.

Two modes:
* ``placeholder`` (used before anything is sent to the LLM): replace each identifier
  with a typed placeholder such as ``<PN>`` so the sentence keeps its meaning.
* ``remove`` (used for NLP word counts): drop the identifier entirely so it cannot
  show up as a "word" in a Pareto.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import lru_cache

# Order matters: the specific patterns run before the generic digit-run catch-all.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("<PROGRAM>", re.compile(r"\[+\s*Program\s+\d+\s*\]+", re.I)),
    ("<AMT>", re.compile(r"\[+\s*AMT\s*\]+", re.I)),
    ("<PN>", re.compile(r"\[+\s*\d[\d\s.\-]*\]+")),
    ("<EMAIL>", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("<PHONE>", re.compile(r"\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b")),
    ("<PROGRAM>", re.compile(r"\bPROG(?:RAM)?\s*#?\s*\d{3,}\b", re.I)),
    # Reference numbers that follow a document/system label: keep the label, mask the number.
    ("<ID>", re.compile(r"\b(ECR|EWO|VWO|WO|PCCR|PC-CR|PRTS|LOG|PO|CG)([\s#:\-]{1,3})\d[\d\-]{3,}\b", re.I)),
    ("<AMT>", re.compile(r"[-+(]?\s?\$\s?\(?\d[\d,]*(?:\.\d+)?\)?")),
    ("<ID>", re.compile(r"\b\d{6,}\b")),
]

_PERSON_LABEL = "<PERSON>"


def scrub_text(text: str, mode: str = "placeholder") -> str:
    """Mask leftover identifiers in one string.

    Args:
        text: Masked GM text.
        mode: ``"placeholder"`` to substitute typed tokens, ``"remove"`` to delete them.
    """
    if mode not in {"placeholder", "remove"}:
        raise ValueError(f"Unknown scrub mode {mode!r}")
    out = text
    for label, pattern in _PATTERNS:
        repl = f" {label} " if mode == "placeholder" else " "
        if pattern.groups:  # labelled reference: keep the label word, mask only the number
            out = pattern.sub(lambda m, r=repl: f"{m.group(1)}{m.group(2)}{r}", out)
        else:
            out = pattern.sub(repl, out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


@lru_cache(maxsize=1)
def _ner_model():
    import spacy

    return spacy.load("en_core_web_sm", disable=["lemmatizer", "attribute_ruler"])


def scrub_person_names(texts: Iterable[str], batch_size: int = 256) -> list[str]:
    """Replace spaCy PERSON entities with ``<PERSON>``.

    The small English model over-detects names in engineering text (it sometimes tags part
    descriptions), which costs a little meaning but is the safe direction before an
    external API call.
    """
    nlp = _ner_model()
    texts = list(texts)
    out: list[str] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        text = doc.text
        for ent in reversed([e for e in doc.ents if e.label_ == "PERSON"]):
            text = text[: ent.start_char] + _PERSON_LABEL + text[ent.end_char:]
        out.append(text)
    return out


_N_GM_MASK_PATTERNS = 3  # the first three patterns match GM's own mask tokens


def residual_identifier_counts(texts: Iterable[str]) -> dict[str, int]:
    """Count identifiers GM's masking missed, by type (for the data-quality log)."""
    counts: dict[str, int] = {}
    for text in texts:
        for _, pattern in _PATTERNS[:_N_GM_MASK_PATTERNS]:
            text = pattern.sub(" ", text)  # GM's own mask tokens are not leaks
        for label, pattern in _PATTERNS[_N_GM_MASK_PATTERNS:]:
            text, n = pattern.subn(" ", text)  # remove once counted so nothing is counted twice
            if n:
                counts[label] = counts.get(label, 0) + n
    return counts
