"""Rule-based NLP parsing of the masked ECR text columns.

Steps for each text (all deterministic and reproducible):
    1. Scrub leftover identifiers (``scrub.scrub_text(mode="remove")``).
    2. Lowercase; normalize ``w/`` -> with, ``w/o`` -> without, ``ed&d`` -> edd, ``&`` -> and.
    3. Split into segments on line breaks, pipes, semicolons and sentence ends, so n-grams
       never span two separate statements.
    4. Tokenize; drop any token that contains a digit (model years, program codes, dims).
    5. Expand abbreviations from ``resources/abbreviations.csv`` (GM-confirmed rows only).
    6. Lemmatize with spaCy ``en_core_web_sm`` (POS-aware); acronyms listed in
       ``resources/acronym_glossary.csv`` are kept verbatim.
    7. Remove spaCy English stop words + ``resources/stopwords_domain.txt``, keeping
       negations ("no cost impact") and position words ("front", "side") - see KEEP_WORDS.

The output for each text is a string of lemmas, with segments separated by `` | ``.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .config import RESOURCE_DIR
from .scrub import scrub_text

log = logging.getLogger(__name__)

SEGMENT_SEP = " | "
# spaCy's English stop list contains words that carry meaning in automotive text
# (negations for "no cost impact"; position words for "front bumper", "side door").
KEEP_WORDS = frozenset({
    "no", "not", "without", "none",
    "front", "back", "side", "top", "bottom", "full", "empty", "move", "off", "first", "last",
})

_SEGMENT_SPLIT = re.compile(r"[\r\n|;]+|(?<=[a-z])[.!?](?=\s|$)")
_TOKEN = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
_HAS_DIGIT = re.compile(r"\d")


def load_abbreviations(path: Path = RESOURCE_DIR / "abbreviations.csv") -> dict[str, list[str]]:
    """Only rows marked confirmed_by_gm = yes are applied (expanding unconfirmed shorthand
    would be an inference about its meaning)."""
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["abbreviation"].strip().lower(): row["expansion"].strip().lower().split()
            for row in csv.DictReader(f)
            if row["abbreviation"].strip() and row.get("confirmed_by_gm", "").strip().lower() == "yes"
        }


def load_acronyms(path: Path = RESOURCE_DIR / "acronym_glossary.csv") -> set[str]:
    """Acronyms kept verbatim (never lemmatized, never expanded)."""
    with path.open(newline="", encoding="utf-8") as f:
        return {row["acronym"].strip().lower() for row in csv.DictReader(f) if row["acronym"].strip()}


def load_stopwords(path: Path = RESOURCE_DIR / "stopwords_domain.txt") -> set[str]:
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            words.add(line)
    return words


@lru_cache(maxsize=1)
def _spacy_model():
    import spacy

    # Only the tagger/attribute_ruler/lemmatizer are needed; parser and NER are slow.
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


@dataclass
class NLPParser:
    """Callable text parser with an audit trail of what was removed."""

    abbreviations: dict[str, list[str]] = field(default_factory=load_abbreviations)
    domain_stopwords: set[str] = field(default_factory=load_stopwords)
    acronyms: set[str] = field(default_factory=load_acronyms)
    min_token_len: int = 2
    removed: Counter = field(default_factory=Counter)  # (reason, word) -> count
    # Step 1 audit: (surface form, lemma) for kept tokens whose lemma differs from the surface form.
    lemma_pairs: Counter = field(default_factory=Counter)
    # Step 2 audit: every lemma before stop-word removal -> occurrences (over unique texts).
    lemma_totals: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        nlp = _spacy_model()
        self._nlp = nlp
        self._spacy_stop = (set(nlp.Defaults.stop_words) - KEEP_WORDS)

    # ---- pre-tokenization ---------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        t = scrub_text(text, mode="remove").lower()
        t = re.sub(r"\bw/o\b", " without ", t)
        t = re.sub(r"\bw/(?=\s|[a-z])", " with ", t)
        t = t.replace("ed&d", "edd").replace("&", " and ")
        return t

    def _segment_tokens(self, text: str) -> list[list[str]]:
        segments = []
        for seg in _SEGMENT_SPLIT.split(self._normalize(text)):
            tokens: list[str] = []
            for tok in _TOKEN.findall(seg):
                if _HAS_DIGIT.search(tok):
                    self.removed[("contains_digit", "<digits>")] += 1
                    continue
                tokens.extend(self.abbreviations.get(tok, [tok]))
            if tokens:
                segments.append(tokens)
        return segments

    # ---- main API -------------------------------------------------------------------
    def parse_many(self, texts: Iterable[str | None], batch_size: int = 500) -> list[str | None]:
        """Parse texts; returns None where the input is blank or nothing survives cleaning."""
        from spacy.tokens import Doc

        texts = list(texts)
        unique = {t for t in texts if isinstance(t, str) and t.strip()}
        log.info("NLP: %d texts (%d unique non-blank)", len(texts), len(unique))

        unique_list = sorted(unique)
        seg_lists = [self._segment_tokens(t) for t in unique_list]
        # One spaCy Doc per segment keeps POS context local to each statement.
        flat_words = [seg for segs in seg_lists for seg in segs]
        docs = (Doc(self._nlp.vocab, words=w) for w in flat_words)

        lemmatized: list[list[str]] = []
        for doc in self._nlp.pipe(docs, batch_size=batch_size):
            kept = []
            for tok in doc:
                if tok.text in self.acronyms:
                    lemma = tok.text  # e.g. keep "prts", which spaCy would lemmatize to "prt"
                else:
                    lemma = tok.lemma_.lower() if tok.lemma_ else tok.text
                self.lemma_totals[lemma] += 1
                if lemma in self._spacy_stop or tok.text in self._spacy_stop:
                    self.removed[("english_stopword", lemma)] += 1
                elif lemma in self.domain_stopwords:
                    self.removed[("domain_stopword", lemma)] += 1
                elif len(lemma) < self.min_token_len or not re.search(r"[a-z]", lemma):
                    self.removed[("too_short", lemma)] += 1
                else:
                    kept.append(lemma)
                    if tok.text != lemma:
                        self.lemma_pairs[(tok.text, lemma)] += 1
            lemmatized.append(kept)

        parsed: dict[str, str | None] = {}
        i = 0
        for text, segs in zip(unique_list, seg_lists):
            parts = [" ".join(lemmatized[i + k]) for k in range(len(segs))]
            i += len(segs)
            parts = [p for p in parts if p]
            parsed[text] = SEGMENT_SEP.join(parts) if parts else None
        return [parsed.get(t) if isinstance(t, str) else None for t in texts]

    def lemma_audit(self):
        """Step 1: which surface forms were merged into each lemma (kept tokens only)."""
        import pandas as pd

        rows = [{"surface_form": s_, "lemma": l_, "occurrences": n} for (s_, l_), n in self.lemma_pairs.items()]
        return (pd.DataFrame(rows, columns=["surface_form", "lemma", "occurrences"])
                .sort_values("occurrences", ascending=False).reset_index(drop=True))

    def stopword_check(self, top_n: int = 200):
        """Step 2: the most frequent lemmas before stop-word removal and what happened to each."""
        import pandas as pd

        removed_reason = {w: r for (r, w) in self.removed}
        rows = []
        for rank, (lemma, n) in enumerate(self.lemma_totals.most_common(top_n), start=1):
            rows.append({"rank": rank, "lemma": lemma, "occurrences": n,
                         "status": removed_reason.get(lemma, "kept"),
                         "kept_by_exception": lemma in KEEP_WORDS})
        return pd.DataFrame(rows)

    def removal_audit(self):
        import pandas as pd

        rows = [{"reason": r, "word": w, "times_removed": n} for (r, w), n in self.removed.items()]
        return (pd.DataFrame(rows, columns=["reason", "word", "times_removed"])
                .sort_values(["reason", "times_removed"], ascending=[True, False])
                .reset_index(drop=True))


def tokens(parsed: str | None) -> list[str]:
    """Unigrams from an NLP field value."""
    if not isinstance(parsed, str) or not parsed:
        return []
    return [t for seg in parsed.split(SEGMENT_SEP) for t in seg.split()]


def ngrams(parsed: str | None, n: int) -> list[str]:
    """N-grams within segments (never across a ' | ' boundary)."""
    if not isinstance(parsed, str) or not parsed:
        return []
    out = []
    for seg in parsed.split(SEGMENT_SEP):
        w = seg.split()
        out.extend(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    return out
