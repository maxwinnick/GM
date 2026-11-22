"""List the ECRs (real WO numbers) behind every bar in the ITIS Pareto charts and deck.

For each group plotted in a chart (top 10 per Pareto, every drill-down row), this lists up to
10 ECRs in that group, ranked by the chart's measure (ECR-count charts: ranked by COST_USD as
given). Reads the tables written by pareto_cost.py.

Usage: python -m cost_pareto.ecrs_behind_slides <pareto_cost out_dir> <output.xlsx>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ECRS_PER_GROUP = 10
MSLUG = {"count": "ecr_count", "cost_given": "cost_usd_as_given", "cost_recalc": "cost_usd_recalculated"}
TOP_GROUPS = 10

METRICS = {
    "count": ("ECR count", "01_ecr_count", "cost_given"),
    "cost_given": ("COST_USD as given", "02_cost_usd_as_given", "cost_given"),
    "cost_recalc": ("COST_USD recalculated", "03_cost_usd_recalculated", "cost_recalc"),
}
# report -> (ECR column the group key lives in, names.csv level, chart file stem, scope)
LEVEL = {
    "A1_assembly": ("assembly", "assembly", "A1_by_assembly", "with_assembly"),
    "A2_vpps_code": ("VPPS", "vpps_code", "A2_00_by_vpps_code_all_assemblies", "with_assembly"),
    "A3_part_number": ("PART_NUMBER", "part_number", "A3_00_by_part_number_all_assemblies", "with_assembly"),
    "A3_new_part_number": ("NEW_PART_NUMBER", "new_part_number", "A3N_00_by_new_part_number_all_assemblies", "with_assembly"),
    "B1_reason": ("REASON", None, "B1_by_reason_code", "all"),
    "N1_part_number": ("PART_NUMBER", "part_number", "N1_{m}_by_part_number", "no_assembly"),
    "N2_new_part_number": ("NEW_PART_NUMBER", "new_part_number", "N2_{m}_by_new_part_number", "no_assembly"),
    "N3_reason": ("REASON", None, "N3_{m}_by_reason_code", "no_assembly"),
}
DRILL = {  # report -> (key column, key name level, parent column, chart stem pattern)
    "A2_vpps_code": ("VPPS", "vpps_code", "assembly", "A2_{r:02d}_vpps_code_in_assembly_{p}"),
    "A3_part_number": ("PART_NUMBER", "part_number", "assembly", "A3_{r:02d}_part_number_in_assembly_{p}"),
    "A3_new_part_number": ("NEW_PART_NUMBER", "new_part_number", "assembly", "A3N_{r:02d}_new_part_number_in_assembly_{p}"),
    "B2_reason_in_assembly": ("REASON", None, "assembly", "B2_{r:02d}_reasons_for_assembly_{p}"),
    "B3_reason_in_vpps_code": ("REASON", None, "VPPS", "B3_{r:02d}_reasons_for_vpps_code_{p}"),
    "B4_reason_in_part_number": ("REASON", None, "PART_NUMBER", "B4_{r:02d}_reasons_for_part_number_{p}"),
}
# Deck slide -> chart, from matching the deck's images to the chart PNGs (identical files)
SLIDES = [
    (7, "A1_assembly", "count"), (8, "A1_assembly", "cost_given"), (9, "A1_assembly", "cost_recalc"),
    (10, "A2_vpps_code", "count"), (11, "A2_vpps_code", "cost_given"), (12, "A2_vpps_code", "cost_recalc"),
    (14, "A3_part_number", "count"), (15, "A3_part_number", "cost_given"), (16, "A3_part_number", "cost_recalc"),
    (18, "B1_reason", "count"), (19, "B1_reason", "cost_given"), (20, "B1_reason", "cost_recalc"),
    (27, "N1_part_number", "cost_given"),
]
SLIDE_TABLES = {  # deck slides that show drill-down tables rather than a chart
    13: "A2_vpps_code", 23: "B2_reason_in_assembly", 24: "B3_reason_in_vpps_code", 25: "B4_reason_in_part_number",
}
ECR_COLS = ["WO_NO", "STATE", "REASON", "assembly", "VPPS", "PART_NUMBER", "NEW_PART_NUMBER", "FNA_DESC",
            "COST_TYPE", "COST", "EXCHANGE_RATE", "COST_USD", "cost_recalc"]


def load(out: Path):
    t = out / "tables"
    s = {c: str for c in ("key", "parent_key", "assembly", "VPPS", "PART_NUMBER", "NEW_PART_NUMBER")}
    ecr = pd.read_csv(t / "ecr_data.csv", dtype=s)
    ecr["cost_given"] = ecr["COST_USD"]
    names = pd.read_csv(t / "names.csv", dtype=str)
    names = {(r.level, r.key): (r.name if isinstance(r.name, str) else "") for r in names.itertuples()}
    return (ecr, pd.read_csv(t / "level_tables.csv", dtype=s), pd.read_csv(t / "a_drill.csv", dtype=s),
            pd.read_csv(t / "b_drill.csv", dtype=s), names)


def ecrs_for(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    rank_col = METRICS[metric][2]
    x = sub.sort_values([rank_col, "WO_NO"], ascending=[False, True], kind="mergesort").head(ECRS_PER_GROUP).copy()
    x.insert(0, "ecr_rank_in_group", range(1, len(x) + 1))
    return x


def main(out: Path, xlsx: Path) -> None:
    ecr, L, AD, BD, names = load(out)
    meta = json.loads((out / "summary.json").read_text())
    src_line = f"{meta.get('source', 'extract')}, sheet {meta.get('sheet', 'result')}, {len(ecr):,} ECRs, every STATE"
    scope = {"all": ecr, "with_assembly": ecr[ecr["assembly"].notna()], "no_assembly": ecr[ecr["assembly"].isna()]}
    rows = []

    def add(chart_set, report, metric, chart, parent_col, parent_key, parent_name, key_col, key, name_level, g):
        # level charts use their own scope; every drill-down sits inside the ECRs with an assembly
        base = scope[LEVEL[report][3]] if parent_col is None else scope["with_assembly"]
        sub = base[base[key_col] == key]
        if parent_col is not None:
            sub = sub[sub[parent_col] == parent_key]
        val_col = METRICS[metric][2]
        e = ecrs_for(sub, metric)
        total = float(g[val_col]) if metric != "count" else None
        for r in e.itertuples(index=False):
            d = r._asdict()
            rows.append({
                "chart_set": chart_set, "chart_file": chart, "report": report, "measure": METRICS[metric][0],
                "parent_level": parent_col or "", "parent_key": parent_key or "", "parent_name": parent_name or "",
                "group_rank": int(g["rank"]), "group_level": key_col, "group_key": key,
                "group_name": names.get((name_level, key), "") if name_level else "",
                "group_ecrs": int(g["ecr_count"]), "group_cost_usd_as_given": float(g["cost_given"]),
                "group_cost_usd_recalculated": float(g["cost_recalc"]),
                "ecr_rank_in_group": d["ecr_rank_in_group"],
                **{c: d[c] for c in ECR_COLS},
                "ecr_share_of_group_value": (float(d[val_col]) / total) if total else None,
            })

    for (report, metric), t in L.groupby(["report", "metric"], sort=False):
        key_col, lvl, stem, _ = LEVEL[report]
        folder = "04_no_vpps_assembly" if report.startswith("N") else METRICS[metric][1]
        chart = f"{folder}/{stem.format(m=MSLUG[metric])}.png"
        for g in t.sort_values("rank").head(TOP_GROUPS).to_dict("records"):
            add("Level chart (top 10)", report, metric, chart, None, None, None, key_col, g["key"], lvl, g)

    for D in (AD, BD):
        for (report, metric, pk), t in D.groupby(["report", "metric", "parent_key"], sort=False):
            key_col, lvl, parent_col, stem = DRILL[report]
            r = int(t["parent_rank"].iloc[0])
            chart = f"{METRICS[metric][1]}/{stem.format(r=r, p=str(pk).replace('.', '-'))}.png"
            if report == "B4_reason_in_part_number" and metric != "count":
                chart = "(table only, deck slide 25)"
            plevel = {"assembly": "assembly", "VPPS": "vpps_code", "PART_NUMBER": "part_number"}[parent_col]
            for g in t.sort_values("rank").to_dict("records"):
                add(f"Drill-down within #{r} {parent_col}", report, metric, chart, parent_col, pk,
                    names.get((plevel, pk), ""), key_col, g["key"], lvl, g)

    df = pd.DataFrame(rows)
    slide = []
    for s, rep, m in SLIDES:
        k = LEVEL[rep]
        folder = "04_no_vpps_assembly" if rep.startswith("N") else METRICS[m][1]
        slide.append({"deck_slide": s, "shows": f"chart {folder}/{k[2].format(m=MSLUG[m])}.png", "report": rep, "measure": METRICS[m][0],
                      "filter_ECR_lookup_on": f"report = {rep}, measure = {METRICS[m][0]}, chart_set = Level chart (top 10)"})
    slide.append({"deck_slide": 22, "shows": "chart 01_ecr_count/B2_01_reasons_for_assembly_80-06-01.png",
                  "report": "B2_reason_in_assembly", "measure": "ECR count",
                  "filter_ECR_lookup_on": "report = B2_reason_in_assembly, parent_key = 80.06.01, measure = ECR count"})
    for s, rep in SLIDE_TABLES.items():
        slide.append({"deck_slide": s, "shows": "drill-down table (all three measures)", "report": rep, "measure": "all",
                      "filter_ECR_lookup_on": f"report = {rep}"})
    slides = pd.DataFrame(slide).sort_values("deck_slide")

    readme = pd.DataFrame({"README": [
        "ECRs behind the ITIS Pareto charts and the ITIS Pareto Reports deck.",
        "Sheet ECR_lookup: for every bar in every chart (top 10 groups per Pareto, every drill-down row), the ECRs in that group, up to 10.",
        "ECRs are ranked by the chart's measure. ECR-count charts rank their ECRs by COST_USD as given.",
        "ecr_share_of_group_value = the ECR's cost / the group's cost on the chart's cost basis (blank on ECR-count charts).",
        "COST_USD recalculated = COST x EXCHANGE_RATE where a rate exists, else COST_USD as given.",
        "Group names follow the FNA_DESC naming rule in the deck (slide 5); blank = no name meets the rule.",
        "Sheet Slide_index: which deck slide shows which chart, and how to filter ECR_lookup for it.",
        f"Source: {src_line}.",
    ]})
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="README", index=False)
        slides.to_excel(w, sheet_name="Slide_index", index=False)
        df.to_excel(w, sheet_name="ECR_lookup", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = min(60, max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:200]) + 2)
                ws.column_dimensions[col[0].column_letter].width = width
            ws.auto_filter.ref = ws.dimensions
        lk = w.book["ECR_lookup"]
        hdr = [c.value for c in lk[1]]
        for name in ("group_cost_usd_as_given", "group_cost_usd_recalculated", "COST", "COST_USD", "cost_recalc"):
            j = hdr.index(name) + 1
            for (c,) in lk.iter_rows(min_row=2, min_col=j, max_col=j):
                c.number_format = "#,##0.00"
        j = hdr.index("ecr_share_of_group_value") + 1
        for (c,) in lk.iter_rows(min_row=2, min_col=j, max_col=j):
            c.number_format = "0.0%"
    print(f"saved {xlsx}: {len(df):,} ECR rows, {df.groupby(['report','measure','parent_key','group_key']).ngroups:,} groups")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
