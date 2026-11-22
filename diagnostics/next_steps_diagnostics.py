"""Next-step diagnostics on the Phase 1 outputs (facts only, no labels inferred).

Reads the masked workbook and the Step 5-6 outputs, then measures:
  1. text coverage by WO range and state
  2. the existing reason codes, and what the text says inside "Reqmts Not Met - Other"
  3. agreement between reason codes and the phrases that name the same thing
  4. how far top-k bigram rules reach (coverage toward the >=95% classification target)
  5. a pilot of Step 7-8: TF-IDF + NMF (k=20), with a split-half stability check
Writes CSV/JSON to <out_dir> (the pipeline uses outputs/04_diagnostics/).

Usage: python -m diagnostics.next_steps_diagnostics <extract.xlsx> <phase1_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer

from ecr_text import config
from ecr_text.pipeline import load_masked

OTHER = "Reqmts Not Met - Other"
K = 20
SEED = 0


def run(input_path: Path, phase1_dir: Path, out_dir: Path, sheet: str = config.DEFAULT_SHEET) -> dict:
    """Run all five diagnostics; returns the summary dict (also written to <out_dir>/summary.json)."""
    # ---------------------------------------------------------------- load
    df = load_masked(input_path, sheet)
    OUT = Path(out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    P1 = Path(phase1_dir)
    rows = pd.read_csv(P1 / "step6_term_frequency" / "tf_matrix_rows.csv")
    cols = pd.read_csv(P1 / "step6_term_frequency" / "tf_matrix_columns.csv")
    X = sparse.load_npz(P1 / "step6_term_frequency" / "tf_matrix.npz").tocsr()
    df = rows.merge(df, on="WO_NO", how="left")
    assert len(df) == X.shape[0]
    terms = cols["term"].to_numpy()
    ngram_n = cols["n"].to_numpy()
    B = (X > 0).astype(np.int8).tocsc()
    n = len(df)
    res: dict = {"ecrs": n}

    # ---------------------------------------------------------------- 1. coverage
    df["era"] = np.where(df["WO_NO"] >= config.WO_RANGE_SPLIT, config.WO_RANGE_LABELS[1], config.WO_RANGE_LABELS[0])
    has = {f: df[c].notna() & (df[c].astype(str).str.strip() != "") for f, c in config.TEXT_COLUMNS.items()}
    cov = df.assign(**{f"has_{f}": v for f, v in has.items()}).groupby("era")[[f"has_{f}" for f in has]].mean()
    cov.insert(0, "ECRs", df.groupby("era").size())
    cov.to_csv(OUT / "1_text_coverage_by_wo_range.csv")
    res["title_only_ecrs"] = int((has["TITLE"] & ~has["DESC_FOR_CHANGE"] & ~has["REASON_FOR_CHANGE"]).sum())
    res["ecrs_with_desc_or_reason"] = int((has["DESC_FOR_CHANGE"] | has["REASON_FOR_CHANGE"]).sum())
    st = df.groupby("STATE").agg(ECRs=("WO_NO", "size"), share_with_desc_or_reason=("WO_NO", lambda s: (has["DESC_FOR_CHANGE"] | has["REASON_FOR_CHANGE"])[s.index].mean()))
    st.sort_values("ECRs", ascending=False).to_csv(OUT / "1_text_coverage_by_state.csv")

    # ---------------------------------------------------------------- 2. reason codes and "Other"
    rc = df["REASON"].fillna("(blank)").value_counts().rename_axis("REASON").reset_index(name="ECRs")
    rc["share"] = rc["ECRs"] / n
    rc["cum_share"] = rc["share"].cumsum()
    rc.to_csv(OUT / "2_reason_code_pareto.csv", index=False)
    res["reason_codes_used"] = int(df["REASON"].nunique())
    res["other_ecrs"] = int((df["REASON"] == OTHER).sum())
    res["other_share"] = res["other_ecrs"] / n
    res["reason_codes_for_80pct"] = int((rc["cum_share"] < 0.80).sum() + 1)
    is_other = (df["REASON"] == OTHER).to_numpy()
    res["other_by_wo_range"] = df[is_other].groupby("era").size().to_dict()
    res["other_with_desc_or_reason"] = int((is_other & (has["DESC_FOR_CHANGE"] | has["REASON_FOR_CHANGE"]).to_numpy()).sum())

    df_all = np.asarray(B.sum(axis=0)).ravel()
    df_oth = np.asarray(B[is_other].sum(axis=0)).ravel()
    s_w = df_all / n
    s_o = df_oth / is_other.sum()
    other_terms = pd.DataFrame({"term": terms, "n": ngram_n, "DF_other": df_oth, "share_other": s_o, "DF_all": df_all,
                                "lift_other": np.where(s_w > 0, s_o / s_w, 0)})
    other_terms = other_terms[other_terms["DF_other"] >= 5]
    top_other = (other_terms[other_terms["n"] == 2].sort_values("DF_other", ascending=False).head(25))
    top_other.to_csv(OUT / "2_other_top_bigrams.csv", index=False)
    (other_terms[(other_terms["n"] == 2) & (other_terms["DF_other"] >= 25)]
     .sort_values("lift_other", ascending=False).head(25).to_csv(OUT / "2_other_highest_lift_bigrams.csv", index=False))
    (other_terms[other_terms["n"] == 1].sort_values("DF_other", ascending=False).head(25)
     .to_csv(OUT / "2_other_top_unigrams.csv", index=False))

    # ---------------------------------------------------------------- 3. code vs phrase agreement
    tidx = {t: j for j, t in enumerate(terms)}
    def has_term(t: str) -> np.ndarray:
        return B[:, tidx[t]].toarray().ravel().astype(bool) if t in tidx else np.zeros(n, bool)

    agree = []
    for phrase in ["cost reduction", "cost saving", "program direction", "tier resiliency", "exterior color",
                   "harness batch", "batch release", "exp launch", "supplier request", "initial release", "quality survey"]:
        m = has_term(phrase)
        if m.sum() == 0:
            continue
        vc = df.loc[m, "REASON"].fillna("(blank)").value_counts()
        agree.append({"phrase": phrase, "ECRs_with_phrase": int(m.sum()),
                      "top_code": vc.index[0], "top_code_share": vc.iloc[0] / m.sum(),
                      "second_code": vc.index[1] if len(vc) > 1 else "", "second_code_share": vc.iloc[1] / m.sum() if len(vc) > 1 else 0,
                      "share_coded_Other": (df.loc[m, "REASON"] == OTHER).mean(), "codes_used": int(vc.size)})
    pd.DataFrame(agree).to_csv(OUT / "3_phrase_vs_reason_code.csv", index=False)
    code = "Cost Reduction or Value Optimization"
    m_code = (df["REASON"] == code).to_numpy()
    res["cost_reduction_code_ecrs"] = int(m_code.sum())
    res["cost_reduction_code_with_cost_phrase"] = int((m_code & (has_term("cost reduction") | has_term("cost saving"))).sum())

    # ---------------------------------------------------------------- 4. reach of top-k bigram rules
    bi_cols = np.flatnonzero(ngram_n == 2)
    bi_order = bi_cols[np.argsort(-df_all[bi_cols], kind="stable")]
    reach = []
    for k in (10, 25, 50, 100, 250, 500, 1000):
        hit = np.asarray(B[:, bi_order[:k]].sum(axis=1)).ravel() > 0
        reach.append({"top_k_bigrams": k, "share_all_ecrs": hit.mean(), "share_other_ecrs": hit[is_other].mean()})
    pd.DataFrame(reach).to_csv(OUT / "4_bigram_rule_reach.csv", index=False)

    # ---------------------------------------------------------------- 5. pilot Step 7-8: TF-IDF + NMF
    keep = np.flatnonzero(ngram_n <= 2)
    Xu = X[:, keep]
    nonempty = np.asarray(Xu.sum(axis=1)).ravel() > 0
    T = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(Xu[nonempty])
    kt = terms[keep]
    res["pilot_ecrs"] = int(nonempty.sum())
    res["pilot_terms"] = int(len(keep))

    def fit(M, seed=SEED):
        nmf = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=400)
        W = nmf.fit_transform(M)
        return W, nmf.components_

    W, H = fit(T)
    dom = W.argmax(axis=1)
    wmax = W.max(axis=1)
    sub = df.loc[nonempty].reset_index(drop=True)
    sub["component"] = dom
    sub["w_max"] = wmax
    thr = np.quantile(wmax, 0.05)
    res["pilot_weak_assignment_threshold_p5"] = float(thr)
    sub["cost_sign"] = np.sign(sub["COST"].fillna(0))
    comp = []
    for c in range(K):
        g = sub[sub["component"] == c]
        top = kt[np.argsort(-H[c])[:8]]
        vc = g["REASON"].fillna("(blank)").value_counts()
        comp.append({"component": c + 1, "ECRs": len(g), "share": len(g) / len(sub), "top_terms": ", ".join(top),
                     "top_reason_code": vc.index[0] if len(vc) else "", "top_code_share": vc.iloc[0] / len(g) if len(g) else 0,
                     "codes_used": int(vc.size), "share_coded_Other": (g["REASON"] == OTHER).mean(),
                     "share_cost_pos": (g["cost_sign"] > 0).mean(), "share_cost_zero": (g["cost_sign"] == 0).mean(),
                     "share_cost_neg": (g["cost_sign"] < 0).mean(),
                     "share_WO_7M": (g["WO_NO"] >= config.WO_RANGE_SPLIT).mean()})
    comp = pd.DataFrame(comp).sort_values("ECRs", ascending=False)
    comp.to_csv(OUT / "5_pilot_nmf_components.csv", index=False)
    sub[["WO_NO", "REASON", "component", "w_max"]].to_csv(OUT / "5_pilot_nmf_assignments.csv.gz", index=False, compression="gzip")
    res["pilot_largest_component_share"] = float(comp["share"].max())
    res["pilot_smallest_component_share"] = float(comp["share"].min())
    res["pilot_median_top_code_share"] = float(comp["top_code_share"].median())

    # where do "Other"-coded ECRs land?
    oth = sub[sub["REASON"] == OTHER]["component"].value_counts(normalize=True).rename("share_of_Other").reset_index()
    oth["component"] += 1
    oth.to_csv(OUT / "5_other_by_component.csv", index=False)
    res["other_components_for_80pct"] = int((oth["share_of_Other"].cumsum() < 0.80).sum() + 1)

    # split-half stability: fit on two random halves, match components by cosine of H
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(T.shape[0])
    A_, B_ = perm[: len(perm) // 2], perm[len(perm) // 2:]
    _, Ha = fit(T[A_])
    _, Hb = fit(T[B_])
    na = Ha / np.linalg.norm(Ha, axis=1, keepdims=True)
    nb = Hb / np.linalg.norm(Hb, axis=1, keepdims=True)
    S = na @ nb.T
    r, c = linear_sum_assignment(-S)
    cos = S[r, c]
    res["split_half_cosine_median"] = float(np.median(cos))
    res["split_half_cosine_min"] = float(cos.min())
    res["split_half_components_cos_ge_0_8"] = int((cos >= 0.8).sum())
    res["k"] = K

    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), indent=1, default=float))
