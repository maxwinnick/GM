"""ITIS Pareto reports on the GM cost extract, by ECR count and by COST_USD.

Reports
-------
Pareto A  assembly (VPPS first 3 segments) -> full VPPS code -> PART_NUMBER / NEW_PART_NUMBER
Pareto B  reason code, overall and inside the top assemblies, VPPS codes and part numbers
No VPPS   ECRs without an assembly code, reported separately

Every Pareto is produced for three metrics:
    count        number of ECRs (one row per WO_NO)
    cost_given   sum of COST_USD exactly as supplied
    cost_recalc  sum of COST x EXCHANGE_RATE where a rate exists, else COST_USD as supplied

Usage: python -m cost_pareto.pareto_cost <extract.xlsx> <out_dir> [sheet]
Writes <out_dir>/tables/*.parquet-free CSVs and <out_dir>/summary.json; charts.py and
workbook.py read those.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHEET = "result"
METRICS = {"count": "ecr_count", "cost_given": "cost_given", "cost_recalc": "cost_recalc"}
TOP_DRILL = 3     # groups drilled into at each level
TOP_N = 10        # rows kept per drill-down table
NO_REASON = "No code mentioned"
MISMATCH_TOL = 0.01  # relative difference that counts as COST_USD != COST x EXCHANGE_RATE


def load(path: Path, sheet: str = SHEET) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, dtype={"PART_NUMBER": "string", "NEW_PART_NUMBER": "string", "VPPS": "string"})
    assert df["WO_NO"].is_unique, "expected one row per ECR (WO_NO)"
    for c in ("PART_NUMBER", "NEW_PART_NUMBER", "VPPS"):
        df[c] = df[c].str.strip()
    # Raw part columns are kept; values that are not part numbers (no digit at all, or an
    # "NPN" placeholder) are left out of the part-number Paretos.
    for c in ("PART_NUMBER", "NEW_PART_NUMBER"):
        df[c + "_RAW"] = df[c]
        not_part = df[c].notna() & (~df[c].str.contains(r"\d", na=False) | df[c].str.startswith("NPN", na=False))
        df[c] = df[c].mask(not_part)
    df["REASON"] = df["REASON"].fillna(NO_REASON)
    # Part description as written on the ECR, normalised for case and spacing, and its leading term
    # (the noun before the first space or hyphen, e.g. HARNESS in "HARNESS ASM-I/P WRG").
    df["desc"] = df["FNA_DESC"].astype("string").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    df["term"] = df["desc"].str.extract(r"^([A-Z0-9/&.]+)")[0]
    has_rate = df["EXCHANGE_RATE"].notna()
    df["cost_given"] = df["COST_USD"].astype(float)
    df["cost_recalc"] = np.where(has_rate, df["COST"] * df["EXCHANGE_RATE"], df["COST_USD"])
    seg = df["VPPS"].str.split(".")
    df["assembly"] = seg.str[:3].str.join(".").where(seg.str.len() >= 3)
    df["ecr_count"] = 1
    return df


def pareto(df: pd.DataFrame, key: str, metric: str) -> pd.DataFrame:
    """Pareto table of `key`, ranked by `metric` (ties broken by key). All three measures kept."""
    col = METRICS[metric]
    g = (df.dropna(subset=[key]).groupby(key, as_index=False)
           .agg(ecr_count=("ecr_count", "sum"), cost_given=("cost_given", "sum"), cost_recalc=("cost_recalc", "sum")))
    g = g.sort_values([col, key], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    total = g[col].sum()
    g["pct"] = g[col] / total if total else 0.0
    g["cum_pct"] = g["pct"].cumsum()
    before = g[col].cumsum() - g[col]            # running total ahead of each group
    g["vital_few"] = before < 0.8 * total - 1e-9 * abs(total)   # tolerance keeps exact-80% ties stable
    g.insert(0, "rank", range(1, len(g) + 1))
    g.insert(0, "metric", metric)
    return g


NAME_SHARE = 0.5   # a description or term must be on MORE than this share of the code's ECRs
NAME_MIN_ECRS = 2  # ...and on at least this many ECRs


def name_table(df: pd.DataFrame, key: str, level: str) -> pd.DataFrame:
    """Name each code from FNA_DESC only where the data supports it.

    tier 'name'    most common FNA_DESC is on >50% of the code's ECRs and on >=2 ECRs
    tier 'term'    otherwise, one leading term is on >50% of the code's ECRs and on >=2 ECRs
    tier 'single'  the code has exactly one ECR and that ECR has a description (cannot be cross-checked)
    tier 'none'    no description meets a rule; the code is shown without a name
    """
    s = df.dropna(subset=[key])
    rows = []
    for k, g in s.groupby(key, sort=False):
        n = len(g)
        dv, tv = g["desc"].value_counts(), g["term"].value_counts()
        top_d, n_d = (dv.index[0], int(dv.iloc[0])) if len(dv) else (None, 0)
        top_t, n_t = (tv.index[0], int(tv.iloc[0])) if len(tv) else (None, 0)
        tie_d = len(dv) > 1 and dv.iloc[1] == dv.iloc[0]
        if n_d >= NAME_MIN_ECRS and n_d > NAME_SHARE * n:
            tier, name, on = "name", top_d, n_d
        elif n_t >= NAME_MIN_ECRS and n_t > NAME_SHARE * n:
            tier, name, on = "term", f"{top_t} (various)", n_t
        elif n == 1 and n_d == 1:
            tier, name, on = "single", top_d, 1
        else:
            tier, name, on = "none", None, 0
        rows.append({"level": level, "key": k, "ecrs": n, "ecrs_with_desc": int(g["desc"].notna().sum()), "distinct_desc": int(len(dv)),
                     "top_desc": top_d, "top_desc_ecrs": n_d, "top_desc_tied": bool(tie_d), "top_term": top_t, "top_term_ecrs": n_t,
                     "tier": tier, "name": name, "name_ecrs": on, "name_share": on / n if n else 0.0})
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame, metric: str) -> dict:
    col = METRICS[metric]
    return {
        "groups": int(len(t)),
        "total": float(t[col].sum()),
        "top1": str(t.iloc[0, 2]) if len(t) else None,
        "top1_value": float(t[col].iloc[0]) if len(t) else None,
        "top10_share": float(t["pct"].head(10).sum()),
        "groups_to_80": int(t["vital_few"].sum()),
    }


def main(src: Path, out: Path, sheet: str = SHEET, verbose: bool = True) -> None:
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df = load(src, sheet)
    has_asm = df["assembly"].notna()
    asm, noasm = df[has_asm], df[~has_asm]
    S: dict = {"source": src.name, "sheet": sheet}

    # ---- scope and cost bases -------------------------------------------------------------
    rate = df["EXCHANGE_RATE"].notna()
    diff = (df["cost_recalc"] - df["cost_given"]).abs()
    mism = rate & (diff > MISMATCH_TOL * df["cost_given"].abs().clip(lower=1.0))
    unconv = mism & np.isclose(df["COST"], df["COST_USD"], rtol=1e-6, atol=0.005)
    norate_diff = ~rate & ~np.isclose(df["COST"], df["COST_USD"], rtol=1e-3, atol=0.5)
    S["scope"] = {
        "ecrs": int(len(df)),
        "created_min": str(df["CREATED_ON"].min().date()), "created_max": str(df["CREATED_ON"].max().date()),
        "with_assembly": int(has_asm.sum()), "without_assembly": int((~has_asm).sum()),
        "vpps_null": int(df["VPPS"].isna().sum()), "vpps_short": int((df["VPPS"].notna() & ~has_asm).sum()),
        "assemblies": int(asm["assembly"].nunique()), "vpps_codes": int(asm["VPPS"].nunique()),
        "part_filled": int(df["PART_NUMBER_RAW"].notna().sum()), "part_unique": int(df["PART_NUMBER_RAW"].nunique()),
        "part_one_ecr": int((df["PART_NUMBER_RAW"].value_counts() == 1).sum()),
        "part_max_ecrs": int(df["PART_NUMBER_RAW"].value_counts().max()),
        "newpart_filled": int(df["NEW_PART_NUMBER_RAW"].notna().sum()), "newpart_unique": int(df["NEW_PART_NUMBER_RAW"].nunique()),
        "newpart_one_ecr": int((df["NEW_PART_NUMBER_RAW"].value_counts() == 1).sum()),
        "newpart_max_ecrs": int(df["NEW_PART_NUMBER_RAW"].value_counts().max()),
        "newpart_equals_part": int((df["PART_NUMBER_RAW"] == df["NEW_PART_NUMBER_RAW"]).fillna(False).sum()),
        "part_excluded": int((df["PART_NUMBER_RAW"].notna() & df["PART_NUMBER"].isna()).sum()),
        "newpart_excluded": int((df["NEW_PART_NUMBER_RAW"].notna() & df["NEW_PART_NUMBER"].isna()).sum()),
        "newpart_nonnumeric": int((~df["NEW_PART_NUMBER"].dropna().str.fullmatch(r"\d+")).sum()),
        "part_nonnumeric": int((~df["PART_NUMBER"].dropna().str.fullmatch(r"\d+")).sum()),
        "reasons": int(df["REASON"].nunique()), "reason_blank": int((df["REASON"] == NO_REASON).sum()),
        "parts_in_multiple_assemblies": int((asm.dropna(subset=["PART_NUMBER"]).groupby("PART_NUMBER")["assembly"].nunique() > 1).sum()),
        "parts_with_assembly": int(asm["PART_NUMBER"].nunique()),
    }
    S["cost"] = {
        "given_total": float(df["cost_given"].sum()), "recalc_total": float(df["cost_recalc"].sum()),
        "zero": int((df["cost_given"] == 0).sum()), "negative": int((df["cost_given"] < 0).sum()),
        "negative_sum": float(df.loc[df["cost_given"] < 0, "cost_given"].sum()), "positive": int((df["cost_given"] > 0).sum()),
        "rows_with_rate": int(rate.sum()), "rows_mismatch": int(mism.sum()), "rows_unconverted": int(unconv.sum()),
        "mismatch_given": float(df.loc[mism, "cost_given"].sum()), "mismatch_recalc": float(df.loc[mism, "cost_recalc"].sum()),
        "norate_rows": int((~rate).sum()), "norate_cost_ne_usd": int(norate_diff.sum()),
        "norate_cost_ne_usd_given": float(df.loc[norate_diff, "cost_given"].sum()),
        "norate_cost_ne_usd_cost": float(df.loc[norate_diff, "COST"].sum()),
        "median_given": float(df["cost_given"].median()),
        "cost_types": int(df["COST_TYPE"].nunique()),
    }
    for m in ("cost_given", "cost_recalc"):
        top = df[m].sort_values(ascending=False)
        S["cost"][f"top10_ecr_share_{m}"] = float(top.head(10).sum() / top.sum())
        S["cost"][f"top1pct_ecr_share_{m}"] = float(top.head(int(round(len(df) * 0.01))).sum() / top.sum())
        S["cost"][f"asm_total_{m}"] = float(asm[m].sum())
        S["cost"][f"noasm_total_{m}"] = float(noasm[m].sum())

    states = (df.groupby("STATE").agg(ecr_count=("ecr_count", "sum"), cost_given=("cost_given", "sum"), cost_recalc=("cost_recalc", "sum"),
                                       no_assembly=("assembly", lambda s: int(s.isna().sum())))
                .sort_values("ecr_count", ascending=False).reset_index())
    states.to_csv(tables / "states.csv", index=False)

    recon = df.loc[mism | norate_diff, ["WO_NO", "STATE", "REASON", "assembly", "VPPS", "PART_NUMBER", "COST_TYPE", "COST", "EXCHANGE_RATE", "cost_given", "cost_recalc"]].copy()
    recon["difference"] = recon["cost_given"] - recon["cost_recalc"]
    recon["rule"] = np.where(recon["EXCHANGE_RATE"].notna(), "rate present: COST x EXCHANGE_RATE", "no rate: COST_USD kept (COST differs)")
    recon.sort_values("difference", key=np.abs, ascending=False).to_csv(tables / "cost_reconciliation.csv", index=False)

    cols = ["WO_NO", "STATE", "REASON", "COST_TYPE", "assembly", "VPPS", "PART_NUMBER", "NEW_PART_NUMBER", "FNA_DESC", "COST", "EXCHANGE_RATE", "cost_given", "cost_recalc"]
    tops = []
    for m in ("cost_given", "cost_recalc"):
        t = df.nlargest(25, m)[cols].copy(); t.insert(0, "rank", range(1, 26)); t.insert(0, "metric", m); tops.append(t)
    pd.concat(tops).to_csv(tables / "top_ecrs.csv", index=False)

    ecr = df[["WO_NO", "STATE", "REASON", "COST_TYPE", "assembly", "VPPS", "PART_NUMBER_RAW", "NEW_PART_NUMBER_RAW", "COST", "EXCHANGE_RATE", "COST_USD", "cost_recalc", "FNA_DESC"]]
    ecr.rename(columns={"PART_NUMBER_RAW": "PART_NUMBER", "NEW_PART_NUMBER_RAW": "NEW_PART_NUMBER"}).to_csv(tables / "ecr_data.csv", index=False)

    # ---- names from FNA_DESC ----------------------------------------------------------------
    names = pd.concat([name_table(df, "assembly", "assembly"), name_table(df, "VPPS", "vpps_code"),
                       name_table(df, "PART_NUMBER", "part_number"), name_table(df, "NEW_PART_NUMBER", "new_part_number")])
    names.to_csv(tables / "names.csv", index=False)
    S["names"] = {"desc_filled": int(df["desc"].notna().sum()), "desc_filled_with_assembly": int((df["desc"].notna() & has_asm).sum()),
                  "desc_filled_no_assembly": int((df["desc"].notna() & ~has_asm).sum()), "distinct_desc": int(df["desc"].nunique())}
    for lv, t in names.groupby("level"):
        multi = t[t["ecrs_with_desc"] >= 2]
        S["names"][lv] = {"codes": int(len(t)), **{f"tier_{k}": int(v) for k, v in t["tier"].value_counts().items()},
                          "codes_2plus_described": int(len(multi)),
                          "codes_2plus_one_desc": int((multi["distinct_desc"] == 1).sum()),
                          "codes_2plus_majority_desc": int((multi["top_desc_ecrs"] > NAME_SHARE * multi["ecrs"]).sum()),
                          "ecrs_in_named_codes": int(t.loc[t["tier"].isin(["name", "term"]), "ecrs"].sum()), "ecrs_total": int(t["ecrs"].sum())}

    # ---- Pareto A and B -------------------------------------------------------------------
    S["pareto"] = {}
    level1, drill, b_drill, heat = [], [], [], []
    keys = {"A1_assembly": "assembly", "A2_vpps_code": "VPPS", "A3_part_number": "PART_NUMBER", "A3_new_part_number": "NEW_PART_NUMBER", "B1_reason": "REASON"}
    for m in METRICS:
        S["pareto"][m] = {}
        tabs = {}
        for name, key in keys.items():
            base = df if name == "B1_reason" else asm
            t = pareto(base, key, m)
            tabs[name] = t
            S["pareto"][m][name] = stats(t, m)
            level1.append(t.rename(columns={key: "key"}).assign(report=name))
        a1 = tabs["A1_assembly"]
        top_asm = a1["assembly"].head(TOP_DRILL).tolist()
        S["pareto"][m]["top_assemblies"] = top_asm
        S["pareto"][m]["top_vpps_codes"] = tabs["A2_vpps_code"]["VPPS"].head(TOP_DRILL).tolist()
        S["pareto"][m]["top_parts"] = tabs["A3_part_number"]["PART_NUMBER"].head(TOP_DRILL).tolist()

        # A2/A3 inside the top assemblies
        for i, a in enumerate(top_asm, 1):
            sub = asm[asm["assembly"] == a]
            for lvl, key in (("A2_vpps_code", "VPPS"), ("A3_part_number", "PART_NUMBER"), ("A3_new_part_number", "NEW_PART_NUMBER")):
                t = pareto(sub, key, m)
                drill.append(t.head(TOP_N).rename(columns={key: "key"}).assign(report=lvl, parent="assembly", parent_key=a, parent_rank=i,
                                                                                n_groups=len(t), level_total=float(t[METRICS[m]].sum()), parent_ecrs=len(sub), parent_total=float(sub[METRICS[m]].sum())))
        # B2: reason inside top assemblies; B3: inside top VPPS codes; B4: inside top part numbers
        for lvl, key, tops_ in (("B2_reason_in_assembly", "assembly", top_asm),
                                ("B3_reason_in_vpps_code", "VPPS", S["pareto"][m]["top_vpps_codes"]),
                                ("B4_reason_in_part_number", "PART_NUMBER", S["pareto"][m]["top_parts"])):
            for i, k in enumerate(tops_, 1):
                sub = asm[asm[key] == k]
                t = pareto(sub, "REASON", m)
                b_drill.append(t.head(TOP_N).rename(columns={"REASON": "key"}).assign(report=lvl, parent=key, parent_key=k, parent_rank=i,
                                                                                       n_groups=len(t), level_total=float(t[METRICS[m]].sum()), parent_ecrs=len(sub), parent_total=float(sub[METRICS[m]].sum())))
        # heatmap: reason mix across the top 10 assemblies (share of each assembly's metric)
        col = METRICS[m]
        top10 = a1["assembly"].head(10).tolist()
        top_reasons = tabs["B1_reason"]["REASON"].head(8).tolist()
        sub = asm[asm["assembly"].isin(top10)].copy()
        sub["reason_grp"] = sub["REASON"].where(sub["REASON"].isin(top_reasons), "All other codes")
        h = sub.groupby(["assembly", "reason_grp"])[col].sum().unstack().fillna(0.0)
        h = h.reindex(index=top10, columns=top_reasons + ["All other codes"], fill_value=0.0)
        share = h.div(h.sum(axis=1), axis=0)
        long = share.stack().rename("share").reset_index()
        long["value"] = h.stack().values
        long["metric"] = m
        long["assembly_rank"] = long["assembly"].map({a: i for i, a in enumerate(top10, 1)})
        heat.append(long)

        # ECRs without an assembly
        for name, key in (("N1_part_number", "PART_NUMBER"), ("N2_new_part_number", "NEW_PART_NUMBER"), ("N3_reason", "REASON")):
            t = pareto(noasm, key, m)
            S["pareto"][m][name] = stats(t, m)
            level1.append(t.rename(columns={key: "key"}).assign(report=name))

    pd.concat(level1).to_csv(tables / "level_tables.csv", index=False)
    pd.concat(drill).to_csv(tables / "a_drill.csv", index=False)
    pd.concat(b_drill).to_csv(tables / "b_drill.csv", index=False)
    pd.concat(heat).to_csv(tables / "heatmap.csv", index=False)

    # part-number detail used in the write-up: ECRs, states, reasons, assemblies per top part
    detail = []
    for m in METRICS:
        for scope, base in (("with_assembly", asm), ("no_assembly", noasm)):
            for key in ("PART_NUMBER", "NEW_PART_NUMBER"):
                t = pareto(base, key, m).head(TOP_N)
                for _, r in t.iterrows():
                    s = base[base[key] == r[key]]
                    detail.append({"metric": m, "scope": scope, "level": key, "rank": r["rank"], "key": r[key], "ecr_count": r["ecr_count"],
                                   "cost_given": r["cost_given"], "cost_recalc": r["cost_recalc"], "pct": r["pct"], "cum_pct": r["cum_pct"],
                                   "assemblies": "; ".join(sorted(s["assembly"].dropna().unique())),
                                   "states": "; ".join(f"{k} ({v})" for k, v in s["STATE"].value_counts().items()),
                                   "reasons": "; ".join(f"{k} ({v})" for k, v in s["REASON"].value_counts().head(3).items()),
                                   "n_reasons": s["REASON"].nunique(),
                                   "largest_ecr": int(s.loc[s[METRICS[m]].idxmax(), "WO_NO"]),
                                   "largest_ecr_desc": s.loc[s[METRICS[m]].idxmax(), "desc"],
                                   "largest_ecr_value": float(s[METRICS[m]].max())})
    pd.DataFrame(detail).to_csv(tables / "part_detail.csv", index=False)

    (out / "summary.json").write_text(json.dumps(S, indent=2))
    if verbose:
        print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), *(sys.argv[3:4] or [SHEET]))
