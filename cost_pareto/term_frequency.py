"""Global term-frequency snapshot of the ECR free text (Phase 1, Steps 5-6 outputs).

One row per vocabulary term (unigram, bigram or trigram in >= 5 ECRs) with its ECR count,
occurrences, spread across WO ranges / programs / suppliers / assemblies, most common reason
code, and the COST_USD (both bases) of the ECRs that contain it.

Usage: python -m cost_pareto.term_frequency <phase1_results_dir> <ecr_data.csv> <out_dir>
  phase1_results_dir  folder with step3_counts/, step5_vocabulary/, step6_term_frequency/
  ecr_data.csv        per-ECR cost table written by cost_pareto.pareto_cost (tables/ecr_data.csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOP_N = 500
EXAMPLE_WOS = 3


def main(p1: Path, ecr_csv: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    vocab = pd.read_csv(p1 / "step5_vocabulary" / "vocabulary.csv")
    tf = pd.read_csv(p1 / "step6_term_frequency" / "tf_long.csv.gz", usecols=["WO_NO", "term", "tf"])
    dims = pd.read_csv(p1 / "step3_counts" / "ecr_word_counts.csv",
                       usecols=["WO_NO", "SYSTEM_ERA", "DIM_PROGRAM", "DIM_SUPPLIER", "DIM_VPPS_L3"])
    cost = pd.read_csv(ecr_csv, usecols=["WO_NO", "REASON", "COST_USD", "cost_recalc"])
    n_ecrs = len(dims)

    e = dims.merge(cost, on="WO_NO", how="left", validate="1:1")
    assert e["COST_USD"].notna().all(), "every ECR in the text outputs should have a cost row"
    for c, blank in (("DIM_PROGRAM", "(no lead program)"),):
        e.loc[e[c] == blank, c] = pd.NA
    x = tf.merge(e, on="WO_NO", how="left", validate="m:1")

    g = x.groupby("term")
    agg = pd.DataFrame({
        "ecrs": g["WO_NO"].nunique(),
        "occurrences": g["tf"].sum(),
        "ecrs_wo_7.0M_plus": g["SYSTEM_ERA"].apply(lambda s: (s == "WO 7.0M+").sum()),
        "lead_programs": g["DIM_PROGRAM"].nunique(),
        "suppliers": g["DIM_SUPPLIER"].nunique(),
        "assemblies": g["DIM_VPPS_L3"].nunique(),
        "cost_usd_as_given_of_ecrs": g["COST_USD"].sum(),
        "cost_usd_recalculated_of_ecrs": g["cost_recalc"].sum(),
    })
    agg["ecrs_wo_4.1M_4.8M"] = agg["ecrs"] - agg["ecrs_wo_7.0M_plus"]
    rc = x.groupby(["term", "REASON"]).size().rename("n").reset_index()
    rc = rc.sort_values(["term", "n", "REASON"], ascending=[True, False, True]).drop_duplicates("term").set_index("term")
    agg["top_reason_code"] = rc["REASON"]
    agg["top_reason_code_share"] = rc["n"] / agg["ecrs"]
    ex = (x.sort_values(["term", "COST_USD", "WO_NO"], ascending=[True, False, True])
            .groupby("term")["WO_NO"].apply(lambda s: ", ".join(map(str, s.head(EXAMPLE_WOS)))))
    agg["largest_cost_example_wos"] = ex

    v = vocab.rename(columns={"DF": "ecrs_vocab", "TF": "occurrences_vocab"}).set_index("term").join(agg, how="left")
    assert (v["ecrs"] == v["ecrs_vocab"]).all(), "ECR counts from tf_long must match the vocabulary DF"
    v = v.drop(columns=["ecrs_vocab", "occurrences_vocab", "n"]).reset_index()
    v["share_of_ecrs"] = v["ecrs"] / n_ecrs
    v = v.sort_values(["ecrs", "term"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    v.insert(0, "rank_by_ecrs", range(1, len(v) + 1))
    v["rank_by_cost_usd_as_given"] = v["cost_usd_as_given_of_ecrs"].rank(ascending=False, method="min").astype(int)
    cols = ["rank_by_ecrs", "term", "ngram", "ecrs", "share_of_ecrs", "occurrences", "ecrs_wo_7.0M_plus",
            "ecrs_wo_4.1M_4.8M", "lead_programs", "suppliers", "assemblies", "top_reason_code", "top_reason_code_share",
            "cost_usd_as_given_of_ecrs", "cost_usd_recalculated_of_ecrs", "rank_by_cost_usd_as_given",
            "largest_cost_example_wos"]
    v = v[cols]
    for c in ("share_of_ecrs", "top_reason_code_share"):
        v[c] = v[c].round(4)
    for c in ("cost_usd_as_given_of_ecrs", "cost_usd_recalculated_of_ecrs"):
        v[c] = v[c].round(2)

    v.to_csv(out / "term_frequency_full_vocabulary.csv", index=False)
    v.head(TOP_N).to_csv(out / "term_frequency_top500_all_ngrams.csv", index=False)
    by = pd.concat([t.head(TOP_N).assign(rank_within_ngram=range(1, min(TOP_N, len(t)) + 1))
                    for _, t in v.groupby("ngram", sort=False)])
    by = by[["ngram", "rank_within_ngram"] + [c for c in cols if c != "ngram"]]
    order = {"unigram": 0, "bigram": 1, "trigram": 2}
    by = by.sort_values(["ngram", "rank_within_ngram"], key=lambda s: s.map(order) if s.name == "ngram" else s)
    by.to_csv(out / "term_frequency_top500_by_ngram.csv", index=False)
    print(f"{len(v):,} vocabulary terms; top {TOP_N} files written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
