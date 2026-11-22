"""Phase 1 text frequency model, Steps 3, 5 and 6 (mid-term report, Equations 1-5).

Document d_i = the cleaned (Steps 1-2) title + description + reason for change of ECR i.

Step 3  Count words per ECR and per group (program, supplier, VPPS L1/L2/L3):
        f_iw (occurrences), x_iw (presence), DF_w = sum_i x_iw            (Eq. 1)
        s_w = DF_w / |E|                                                   (Eq. 1)
        cumulative share S_m over terms sorted by DF                      (Eq. 2)
        s_gw = DF_gw / |E_g|                                               (Eq. 3)
        lift L_gw = s_gw / s_w                                             (Eq. 4)
        ubiquity U_w = share of groups where rank_g(w) <= N (N = 10)       (Eq. 5)
Step 5  Vocabulary: every unigram, bigram and trigram with DF_w >= min_df (default 5 ECRs),
        built with scikit-learn's CountVectorizer.
Step 6  Term frequency per ECR: the document-term matrix tf_{i,w} = f_iw (raw counts) over the
        vocabulary, saved as a sparse matrix plus a long table.

Steps 1-2 (lemmatize, stop words) run in ``nlp.py``; Step 4 (histograms) in ``figures.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .config import TEXT_COLUMNS, ParetoConfig
from .nlp import SEGMENT_SEP
from .pareto import NO_GROUP_PREFIX, add_dimensions, filter_scope

log = logging.getLogger(__name__)

DOC_FIELDS = ("TITLE", "DESC_FOR_CHANGE", "REASON_FOR_CHANGE")
NGRAMS = {1: "unigram", 2: "bigram", 3: "trigram"}


@dataclass
class Phase1Config:
    min_df: int = 5                 # Step 5: keep terms in >= 5 ECRs
    top_n_ubiquity: int = 10        # Eq. 5: N
    min_group_ecrs: int = 25        # groups smaller than this are not ranked (Eq. 3-5)
    min_group_df_for_lift: int = 5  # lift reported only when DF_gw >= 5
    pareto: ParetoConfig = field(default_factory=ParetoConfig)


def segment_ngrams(doc: str) -> list[str]:
    """Analyzer for CountVectorizer: unigrams, bigrams, trigrams within a segment."""
    out: list[str] = []
    if not isinstance(doc, str) or not doc:
        return out
    for seg in doc.split(SEGMENT_SEP):
        w = seg.split()
        for n in (1, 2, 3):
            out.extend(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    return out


def build_documents(parsed: pd.DataFrame, cfg: Phase1Config) -> pd.DataFrame:
    df = filter_scope(add_dimensions(parsed), cfg.pareto).reset_index(drop=True)
    parts = df[[f"{f}_NLP" for f in DOC_FIELDS]]
    df["DOC"] = parts.apply(lambda r: SEGMENT_SEP.join(v for v in r if isinstance(v, str) and v), axis=1)
    df["HAS_TEXT"] = df[[TEXT_COLUMNS[f] for f in DOC_FIELDS]].notna().any(axis=1)
    return df


def _group_labels(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Display labels per dimension. VPPS assemblies (L3) carry GM's most frequent FNA_DESC."""
    labels: dict[str, dict[str, str]] = {}
    sub = df[~df["DIM_VPPS_L3"].astype(str).str.startswith(NO_GROUP_PREFIX)]
    fna = sub.dropna(subset=["FNA_DESC"]).groupby("DIM_VPPS_L3")["FNA_DESC"].agg(lambda s: s.value_counts().index[0])
    labels["VPPS_L3"] = {g: f"{g} {d}" for g, d in fna.items()}
    return labels


def run_phase1(parsed: pd.DataFrame, out_dir: Path, cfg: Phase1Config | None = None) -> dict:
    cfg = cfg or Phase1Config()
    out_dir = Path(out_dir)
    s3, s5, s6 = (out_dir / d for d in ("step3_counts", "step5_vocabulary", "step6_term_frequency"))
    for d in (s3, s5, s6):
        d.mkdir(parents=True, exist_ok=True)

    df = build_documents(parsed, cfg)
    n_ecr = len(df)
    log.info("Phase 1: %d ECRs", n_ecr)

    # ---- full count matrix (all terms) --------------------------------------------------------
    vec_all = CountVectorizer(analyzer=segment_ngrams, lowercase=False)
    X_all = vec_all.fit_transform(df["DOC"]).tocsc()
    terms_all = vec_all.get_feature_names_out()
    order = np.array([t.count(" ") + 1 for t in terms_all])
    df_all = np.asarray((X_all > 0).sum(axis=0)).ravel()
    tf_all = np.asarray(X_all.sum(axis=0)).ravel()

    # ---- Step 5: vocabulary ----------------------------------------------------------------------
    keep = df_all >= cfg.min_df
    vocab = pd.DataFrame({"term": terms_all, "n": order, "ngram": [NGRAMS[k] for k in order],
                          "DF": df_all, "TF": tf_all, "in_vocabulary": keep})
    vocab["exclusion_reason"] = np.where(keep, "", f"DF < {cfg.min_df}")
    vocab = vocab.sort_values(["n", "DF", "TF", "term"], ascending=[True, False, False, True]).reset_index(drop=True)
    vocab[vocab.in_vocabulary].drop(columns=["in_vocabulary", "exclusion_reason"]).to_csv(
        s5 / "vocabulary.csv", index=False)
    vocab.to_csv(s5 / "all_terms_with_vocabulary_flag.csv.gz", index=False, compression="gzip")
    vsum = {}
    for k, name in NGRAMS.items():
        m = order == k
        vsum[name] = {
            "distinct_terms": int(m.sum()), "in_vocabulary": int((m & keep).sum()),
            "share_of_terms_kept": float((m & keep).sum() / m.sum()) if m.sum() else 0.0,
            "occurrences_total": int(tf_all[m].sum()),
            "share_of_occurrences_kept": float(tf_all[m & keep].sum() / tf_all[m].sum()) if m.sum() else 0.0,
            "terms_in_exactly_1_ecr": int((m & (df_all == 1)).sum()),
        }
    vsum["min_df"] = cfg.min_df
    vsum["vocabulary_size"] = int(keep.sum())
    (s5 / "vocabulary_summary.json").write_text(json.dumps(vsum, indent=1))

    # ---- Step 6: term frequency per ECR (vocabulary only) -------------------------------------------
    X = X_all[:, np.flatnonzero(keep)].tocsr()
    terms = terms_all[keep]
    t_order = order[keep]
    sparse.save_npz(s6 / "tf_matrix.npz", X)
    pd.DataFrame({"row": np.arange(n_ecr), "WO_NO": df["WO_NO"]}).to_csv(s6 / "tf_matrix_rows.csv", index=False)
    pd.DataFrame({"col": np.arange(len(terms)), "term": terms, "n": t_order}).to_csv(
        s6 / "tf_matrix_columns.csv", index=False)
    coo = X.tocoo()
    doc_len = np.asarray(X_all[:, np.flatnonzero(order == 1)].sum(axis=1)).ravel()  # unigram tokens per ECR
    tf_long = pd.DataFrame({"WO_NO": df["WO_NO"].to_numpy()[coo.row], "term": terms[coo.col],
                            "n": t_order[coo.col], "tf": coo.data})
    tf_long["unigram_tokens_in_ecr"] = doc_len[coo.row]
    tf_long.sort_values(["WO_NO", "n", "tf"], ascending=[True, True, False]).to_csv(
        s6 / "tf_long.csv.gz", index=False, compression="gzip")
    t6 = {"ecrs": n_ecr, "vocabulary_terms": int(len(terms)), "nonzero_cells": int(X.nnz),
          "density": float(X.nnz / (X.shape[0] * X.shape[1])),
          "ecrs_with_no_vocabulary_term": int((np.asarray(X.sum(axis=1)).ravel() == 0).sum()),
          "median_distinct_vocab_terms_per_ecr": float(np.median(np.diff(X.indptr)))}
    (s6 / "tf_summary.json").write_text(json.dumps(t6, indent=1))

    # ---- Step 3: counts per ECR ------------------------------------------------------------------------
    X_all_csr = X_all.tocsr()
    per_ecr = df[["WO_NO", "STATE", "SYSTEM_ERA", "DIM_PROGRAM", "DIM_SUPPLIER",
                  "DIM_VPPS_L1", "DIM_VPPS_L2", "DIM_VPPS_L3"]].copy()
    for k, name in NGRAMS.items():
        sub = X_all_csr[:, np.flatnonzero(order == k)]
        per_ecr[f"{name}_tokens"] = np.asarray(sub.sum(axis=1)).ravel()
        per_ecr[f"distinct_{name}s"] = np.diff(sub.indptr)
    for f in DOC_FIELDS + ("COST_COMMENT",):
        per_ecr[f"{f}_unigram_tokens"] = df[f"{f}_NLP"].map(
            lambda v: sum(len(s.split()) for s in v.split(SEGMENT_SEP)) if isinstance(v, str) and v else 0)
    per_ecr.to_csv(s3 / "ecr_word_counts.csv", index=False)

    # Eq. 1-2 overall (DF Pareto per n-gram order)
    xb = (X_all > 0).astype(np.int32).tocsc()
    overall = vocab[["term", "n", "ngram", "DF", "TF"]].copy()
    overall["s_w"] = overall["DF"] / n_ecr
    overall = overall.sort_values(["n", "DF", "TF", "term"], ascending=[True, False, False, True])
    overall["rank"] = overall.groupby("n").cumcount() + 1
    overall["cum_share_S_m"] = overall.groupby("n")["DF"].cumsum() / overall.groupby("n")["DF"].transform("sum")
    prev = overall["cum_share_S_m"] - overall["DF"] / overall.groupby("n")["DF"].transform("sum")
    overall["vital_few_to_80pct"] = prev < 0.80
    overall.to_csv(s3 / "term_counts_overall.csv.gz", index=False, compression="gzip")

    # Eq. 3-5 by group
    term_index = {t: j for j, t in enumerate(terms_all)}
    s_w = df_all / n_ecr
    group_tables, ubiq_tables, group_sizes = {}, {}, {}
    for dim, col in cfg.pareto.dimensions.items():
        g = df[col].astype(str)
        sizes = g.value_counts()
        group_sizes[dim] = sizes
        ranked = sizes[(~sizes.index.str.startswith(NO_GROUP_PREFIX)) & (sizes >= cfg.min_group_ecrs)]
        codes = pd.Categorical(g, categories=ranked.index)
        rows = np.flatnonzero(codes.codes >= 0)
        G = sparse.csr_matrix((np.ones(len(rows)), (codes.codes[rows], rows)), shape=(len(ranked), n_ecr))
        DFg = (G @ xb).tocoo()  # groups x terms
        t = pd.DataFrame({"group": ranked.index[DFg.row], "term": terms_all[DFg.col], "n": order[DFg.col],
                          "DF_gw": DFg.data.astype(int)})
        t["group_ecrs"] = ranked.reindex(t["group"]).to_numpy()
        t["s_gw"] = t["DF_gw"] / t["group_ecrs"]
        t["s_w"] = s_w[DFg.col]
        t["lift_L_gw"] = np.where(t["DF_gw"] >= cfg.min_group_df_for_lift, t["s_gw"] / t["s_w"], np.nan)
        t = t.sort_values(["group", "n", "DF_gw", "term"], ascending=[True, True, False, True])
        t["rank_in_group"] = t.groupby(["group", "n"]).cumcount() + 1
        t.insert(0, "dimension", dim)
        t[t["DF_gw"] >= 2].to_csv(s3 / f"group_term_counts_{dim}.csv.gz", index=False, compression="gzip")
        group_tables[dim] = t
        top = t[t["rank_in_group"] <= cfg.top_n_ubiquity]
        u = (top.groupby(["n", "term"])["group"].nunique().rename("groups_with_term_in_top_N").reset_index())
        u["groups_ranked"] = len(ranked)
        u["ubiquity_U_w"] = u["groups_with_term_in_top_N"] / len(ranked)
        u = u.sort_values(["n", "ubiquity_U_w", "term"], ascending=[True, False, True])
        u.insert(0, "dimension", dim)
        ubiq_tables[dim] = u
    pd.concat(ubiq_tables.values()).to_csv(s3 / "ubiquity.csv", index=False)
    pd.concat([s.rename("ecrs").rename_axis("group").reset_index().assign(dimension=d)
               for d, s in group_sizes.items()])[["dimension", "group", "ecrs"]].to_csv(
        s3 / "group_sizes.csv", index=False)

    result = {"df": df, "vocab": vocab, "overall": overall, "group_tables": group_tables,
              "ubiquity": ubiq_tables, "group_sizes": group_sizes, "per_ecr": per_ecr,
              "vocab_summary": vsum, "tf_summary": t6, "labels": _group_labels(df),
              "term_index": term_index, "n_ecr": n_ecr}
    log.info("Phase 1 Steps 3, 5, 6 written to %s (vocabulary %d terms)", out_dir, len(terms))
    return result
