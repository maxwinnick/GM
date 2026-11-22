"""Step 7 for one WO partition: build the ECR x term count matrix from Phase 1 cleaned text.

* Text = the partition's free-text fields, already lemmatized and stop-worded in Phase 1
  (the ``*_NLP`` columns). Each field is split into its Phase 1 segments (" | ") so a
  two-word phrase never crosses a field or sentence boundary.
* Optional Step 7 theme stop words (position / generic-action single words) are removed;
  two-word phrases that contain them (e.g. "rear fascia") are kept.
* Optional structured tokens (e.g. driver_prts, track_exp_launch) are added as extra terms.
* Vocabulary: terms in at least ``min_df`` and at most ``max_df_share`` of the partition's ECRs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from . import common as C


@dataclass
class TermMatrix:
    X: sp.csr_matrix          # partition ECRs x terms, raw counts
    terms: np.ndarray
    ngram: np.ndarray         # 1, 2 (0 = structured token)
    wo: pd.Series             # WO_NO in row order


def theme_stopwords(cfg: dict) -> set[str]:
    if not cfg["vocabulary"].get("use_theme_stopwords", True):
        return set()
    lines = cfg["paths"]["theme_stopwords"].read_text().splitlines()
    return {w.strip().lower() for w in lines if w.strip() and not w.strip().startswith("#")}


def _segments_to_terms(text: str, ngrams: list[int], stop: set[str]) -> list[str]:
    out = []
    for seg in str(text).split(" | "):
        toks = seg.split()
        if 1 in ngrams:
            out += [t for t in toks if t not in stop]
        if 2 in ngrams:
            out += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return out


def build(df: pd.DataFrame, cfg: dict, partition: str, use_stop: bool | None = None,
          use_structured: bool | None = None) -> TermMatrix:
    p = cfg["partitions"][partition]
    v = cfg["vocabulary"]
    stop = theme_stopwords(cfg) if (use_stop if use_stop is not None else v.get("use_theme_stopwords", True)) else set()
    sub = df[C.partition_mask(df, cfg, partition)]
    nlp_cols = [f"{f}_NLP" for f in p["text_fields"]]
    missing = [c for c in nlp_cols if c not in sub]
    assert not missing, f"Phase 1 cleaned columns missing: {missing}"
    docs = [[t for c in nlp_cols for t in _segments_to_terms(r[c] if isinstance(r[c], str) else "", v["ngrams"], stop)]
            for _, r in sub[nlp_cols].iterrows()]
    struct_on = p["structured_fields"] and (use_structured if use_structured is not None else True)
    if struct_on:
        st = C.structured_tokens(sub, p["structured_fields"])
        docs = [d + s.split() for d, s in zip(docs, st)]
    cv = CountVectorizer(analyzer=lambda d: d, min_df=v["min_df"], max_df=v["max_df_share"])
    X = cv.fit_transform(docs).tocsr()
    terms = cv.get_feature_names_out()
    is_struct = np.array([t.startswith(("driver_", "track_", "kind_")) and " " not in t for t in terms])
    ngram = np.where(is_struct, 0, np.char.count(terms.astype(str), " ") + 1)
    return TermMatrix(X=X, terms=terms, ngram=ngram, wo=sub["WO_NO"].reset_index(drop=True))


def term_field_share(df: pd.DataFrame, cfg: dict, partition: str, field: str = "COST_COMMENT") -> pd.Series:
    """For each term, the share of its occurrences (in this partition) that come from ``field``.

    Used to flag themes built mostly from cost-comment wording (cost handling) rather than from
    the title (the change itself). Empty if the partition does not use the field.
    """
    p = cfg["partitions"][partition]
    if field not in p["text_fields"]:
        return pd.Series(dtype=float)
    sub = df[C.partition_mask(df, cfg, partition)]
    v = cfg["vocabulary"]
    stop = theme_stopwords(cfg) if v.get("use_theme_stopwords", True) else set()
    counts = {}
    for f in p["text_fields"]:
        c = pd.Series([t for x in sub[f"{f}_NLP"].dropna() for t in _segments_to_terms(x, v["ngrams"], stop)]).value_counts()
        counts[f] = c
    tab = pd.concat(counts, axis=1).fillna(0)
    return (tab[field] / tab.sum(axis=1)).rename(f"{field.lower()}_share")
