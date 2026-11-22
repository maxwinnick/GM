"""Excel workbook of every Pareto table. Usage: python workbook.py <out_dir> <workbook.xlsx>

Small tables (Summary, States, A1_assembly, B1_reason) are formulas over the ECR_data sheet, so they
recalculate if a cost or code is corrected there. Larger tables are values written by pareto_cost.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
TITLE = Font(name="Arial", size=12, bold=True)
USD = '$#,##0;($#,##0);-'
PCT = '0.0%'
INT = '#,##0'


def write_table(ws, df: pd.DataFrame, formats: dict[str, str] | None = None, start_row: int = 1) -> None:
    formats = formats or {}
    for j, c in enumerate(df.columns, 1):
        cell = ws.cell(row=start_row, column=j, value=c)
        cell.font = BOLD
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if pd.isna(v):
                continue
            v = v.item() if hasattr(v, "item") else v
            if isinstance(v, bool):  # text keeps LibreOffice from rewriting booleans as =TRUE()/=FALSE()
                v = "Yes" if v else "No"
            cell = ws.cell(row=i, column=j, value=v)
            cell.font = FONT
            f = formats.get(df.columns[j - 1])
            if f:
                cell.number_format = f
    for j, c in enumerate(df.columns, 1):
        width = max(len(str(c)), *(len(str(v)) for v in df.iloc[:200, j - 1].astype(str))) if len(df) else len(str(c))
        ws.column_dimensions[get_column_letter(j)].width = min(max(width + 2, 10), 48)
    ws.freeze_panes = ws.cell(row=start_row + 1, column=1)


def level_sheet(L: pd.DataFrame, report: str, key_name: str) -> pd.DataFrame:
    t = L[L["report"] == report]
    base = t[t["metric"] == "count"][["key", "ecr_count", "cost_given", "cost_recalc"]].copy()
    for m, lab in (("count", "ECR count"), ("cost_given", "COST_USD as given"), ("cost_recalc", "COST_USD recalculated")):
        s = t[t["metric"] == m].set_index("key")
        base[f"Rank by {lab}"] = base["key"].map(s["rank"])
        base[f"Share of {lab}"] = base["key"].map(s["pct"])
        base[f"Cumulative share, ranked by {lab}"] = base["key"].map(s["cum_pct"])
    return base.rename(columns={"key": key_name, "ecr_count": "ECRs", "cost_given": "COST_USD as given ($)", "cost_recalc": "COST_USD recalculated ($)"})


BASIS = {"name": "Most common FNA_DESC, on >50% of the code's ECRs", "term": "Shared leading term, on >50% of the code's ECRs",
         "single": "FNA_DESC on the code's only ECR (not cross-checked)", "none": ""}
LEVEL_OF = {"A1_assembly": "assembly", "A2_vpps_code": "vpps_code", "A3_part_number": "part_number", "A3_new_part_number": "new_part_number",
            "N1_part_number": "part_number", "N2_new_part_number": "new_part_number", "assembly": "assembly", "VPPS": "vpps_code",
            "PART_NUMBER": "part_number", "NEW_PART_NUMBER": "new_part_number"}


def add_names(df: pd.DataFrame, names: pd.DataFrame, level: str, key_col: str, prefix: str = "Name") -> pd.DataFrame:
    """Insert name columns right after key_col. names: the table written by pareto_cost.name_table."""
    n = names[names["level"] == level].set_index("key")
    out = df.copy()
    pos = out.columns.get_loc(key_col) + 1
    out.insert(pos, f"{prefix} (from FNA_DESC)", out[key_col].map(n["name"]))
    out.insert(pos + 1, f"{prefix} basis", out[key_col].map(n["tier"]).map(BASIS))
    out.insert(pos + 2, f"ECRs carrying that {prefix.lower()}", out[key_col].map(n["name_ecrs"]).where(out[key_col].map(n["tier"]).isin(["name", "term", "single"])))
    out.insert(pos + 3, "ECRs on the code, whole extract", out[key_col].map(n["ecrs"]))
    return out


def level_formats(df: pd.DataFrame) -> dict[str, str]:
    f = {"ECRs": INT, "COST_USD as given ($)": USD, "COST_USD recalculated ($)": USD}
    f.update({c: PCT for c in df.columns if "hare" in c})
    return f


def main(out: Path, dest: Path) -> None:
    T = out / "tables"
    S = json.loads((out / "summary.json").read_text())
    str_cols = {"key": str, "parent_key": str, "PART_NUMBER": str, "NEW_PART_NUMBER": str, "VPPS": str}
    L = pd.read_csv(T / "level_tables.csv", dtype=str_cols)
    ecr = pd.read_csv(T / "ecr_data.csv", dtype=str_cols)
    NM = pd.read_csv(T / "names.csv", dtype=str_cols)
    n = len(ecr)
    last = n + 1
    rng = lambda col: f"ECR_data!${col}$2:${col}${last}"

    wb = Workbook()
    # ---- README ---------------------------------------------------------------------------
    ws = wb.active
    ws.title = "README"
    lines = [
        ("ITIS Pareto reports with COST_USD: assembly, VPPS code and part number", TITLE),
        ("IOE 424 · GM Case Study No. 6 · Cost Commandos Consulting", FONT),
        (f"Source: {S['source']}, sheet '{S['sheet']}' ({n:,} ECRs, one row per WO_NO, every STATE). Built by cost_pareto/pareto_cost.py and cost_pareto/workbook.py.", FONT),
        ("", FONT),
        ("Definitions", BOLD),
        ("Assembly = first three segments of VPPS (e.g. 80.06.01). ECRs with a blank or one-segment VPPS have no assembly.", FONT),
        ("COST_USD as given = the COST_USD column exactly as supplied.", FONT),
        ("COST_USD recalculated = COST x EXCHANGE_RATE where EXCHANGE_RATE is filled; COST_USD as supplied where it is blank (ECR_data column L).", FONT),
        (f"Cost Paretos use the net sum per group ({int((ecr['COST_USD'].astype(float) < 0).sum()):,} ECRs have a negative COST_USD).", FONT),
        ("Rank: descending by the measure, ties broken by code. Cumulative share: of the total over all groups in that Pareto.", FONT),
        ("Vital few: groups up to and including the one that crosses 80% cumulative share.", FONT),
        ("Part-number Paretos leave out values that are not part numbers: no digit at all (----****, ------**, A) or starting with NPN.", FONT),
        ("", FONT),
        ("Names", BOLD),
        ("The extract has no table of official assembly or part names. Names come from FNA_DESC (the part description on each ECR) and are shown only when a rule is met:", FONT),
        ("1. Most common FNA_DESC, when it is on more than 50% of the code's ECRs and on at least 2 ECRs.", FONT),
        ("2. Otherwise the shared leading term followed by '(various)', when one leading term (e.g. HARNESS) is on more than 50% of the code's ECRs and on at least 2.", FONT),
        ("3. Otherwise, for a code with exactly one ECR, that ECR's FNA_DESC, flagged as not cross-checked (marked * on the charts).", FONT),
        ("4. Otherwise no name. Shares are of all ECRs carrying the code in the whole extract. The Names sheet lists every code with its counts.", FONT),
        ("", FONT),
        ("Sheets", BOLD),
        ("Summary, States, A1_assembly, B1_reason: formulas over ECR_data; they recalculate if ECR_data is edited.", FONT),
        ("A2_vpps_code, A3_part_number, A3_new_part_number: Pareto A levels 2 and 3 for ECRs with an assembly (values).", FONT),
        ("A_drilldowns, B_drilldowns: top 10 rows inside the top 3 assemblies / VPPS codes / part numbers, per measure (values).", FONT),
        ("B2_heatmap: reason-code mix across the top 10 assemblies, per measure (values).", FONT),
        ("NoVPPS_*: ECRs with no assembly, by PART_NUMBER, NEW_PART_NUMBER and reason code (values).", FONT),
        ("Part_detail: states, reason codes and largest ECR for the top 10 part numbers of each Pareto (values).", FONT),
        ("Cost_reconciliation: every ECR where the two cost bases differ. Top_ECRs: 25 largest ECRs on each basis.", FONT),
        ("Names: every assembly, VPPS code, PART_NUMBER and NEW_PART_NUMBER with its description counts and the rule it met.", FONT),
        ("ECR_data: the ECR-level fields the reports use, with the recalculated cost as a formula.", FONT),
    ]
    for i, (txt, font) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=txt)
        c.font = font
    ws.column_dimensions["A"].width = 150

    # ---- ECR_data (source for the formula sheets) -------------------------------------------
    wd = wb.create_sheet("ECR_data")
    cols = ["WO_NO", "STATE", "REASON", "COST_TYPE", "assembly", "VPPS", "PART_NUMBER", "NEW_PART_NUMBER", "COST", "EXCHANGE_RATE", "COST_USD"]
    d = ecr[cols].rename(columns={"assembly": "ASSEMBLY"})
    write_table(wd, d, {"COST": '#,##0.00', "COST_USD": '#,##0.00', "EXCHANGE_RATE": '0.000000'})
    h = wd.cell(row=1, column=12, value="COST_USD_RECALC")
    h.font = BOLD
    for r in range(2, last + 1):
        c = wd.cell(row=r, column=12, value=f'=IF(J{r}="",K{r},I{r}*J{r})')
        c.font = FONT
        c.number_format = '#,##0.00'
    wd.column_dimensions["L"].width = 18
    hm_ = wd.cell(row=1, column=13, value="FNA_DESC")
    hm_.font = BOLD
    for r, v in enumerate(ecr["FNA_DESC"], 2):
        if isinstance(v, str):
            wd.cell(row=r, column=13, value=v).font = FONT
    wd.column_dimensions["M"].width = 40

    # ---- Summary ----------------------------------------------------------------------------
    wsum = wb.create_sheet("Summary", 1)
    rows = [
        ("Measure", "Value", "Note"),
        ("ECRs", f"=COUNTA({rng('A')})", "One row per WO_NO, all 7 states"),
        ("ECRs with an assembly", f'=COUNTIF({rng("E")},"?*")', "VPPS with at least three segments"),
        ("ECRs with no assembly", "=B2-B3", "Blank or one-segment VPPS"),
        ("COST_USD as given, total ($)", f"=SUM({rng('K')})", "COST_USD column as supplied"),
        ("COST_USD recalculated, total ($)", f"=SUM({rng('L')})", "COST x EXCHANGE_RATE where a rate exists"),
        ("Difference ($)", "=B5-B6", ""),
        ("COST_USD as given, ECRs with an assembly ($)", f'=SUMIF({rng("E")},"?*",{rng("K")})', ""),
        ("COST_USD as given, ECRs with no assembly ($)", "=B5-B8", ""),
        ("COST_USD recalculated, ECRs with an assembly ($)", f'=SUMIF({rng("E")},"?*",{rng("L")})', ""),
        ("COST_USD recalculated, ECRs with no assembly ($)", "=B6-B10", ""),
        ("ECRs with an exchange rate", f"=COUNT({rng('J')})", ""),
        ("ECRs with COST_USD = 0", f"=COUNTIF({rng('K')},0)", ""),
        ("ECRs with COST_USD < 0", f'=COUNTIF({rng("K")},"<0")', ""),
    ]
    for i, r in enumerate(rows, 1):
        for j, v in enumerate(r, 1):
            c = wsum.cell(row=i, column=j, value=v)
            c.font = BOLD if i == 1 else FONT
    for i in (2, 3, 4, 12, 13, 14):
        wsum.cell(row=i, column=2).number_format = INT
    for i in (5, 6, 7, 8, 9, 10, 11):
        wsum.cell(row=i, column=2).number_format = USD
    wsum.column_dimensions["A"].width = 50
    wsum.column_dimensions["B"].width = 20
    wsum.column_dimensions["C"].width = 48

    # ---- formula sheets: States, A1_assembly, B1_reason ----------------------------------------
    def formula_sheet(name: str, key_name: str, keys: list[str], key_col: str, name_level: str | None = None) -> None:
        w = wb.create_sheet(name)
        heads = [key_name, "ECRs", "COST_USD as given ($)", "COST_USD recalculated ($)", "Share of ECRs", "Cumulative share of ECRs",
                 "Share of COST_USD as given", "Share of COST_USD recalculated", "Rank by ECR count", "Rank by COST_USD as given", "Rank by COST_USD recalculated"]
        for j, hd in enumerate(heads, 1):
            c = w.cell(row=1, column=j, value=hd)
            c.font = BOLD
            c.alignment = Alignment(wrap_text=True, vertical="top")
        end = len(keys) + 1
        for i, k in enumerate(keys, 2):
            vals = [k, f"=COUNTIF({rng(key_col)},$A{i})", f"=SUMIF({rng(key_col)},$A{i},{rng('K')})", f"=SUMIF({rng(key_col)},$A{i},{rng('L')})",
                    f"=B{i}/SUM(B$2:B${end})", f"=SUM(B$2:B{i})/SUM(B$2:B${end})", f"=C{i}/SUM(C$2:C${end})", f"=D{i}/SUM(D$2:D${end})",
                    f"=RANK(B{i},B$2:B${end})", f"=RANK(C{i},C$2:C${end})", f"=RANK(D{i},D$2:D${end})"]
            for j, v in enumerate(vals, 1):
                c = w.cell(row=i, column=j, value=v)
                c.font = FONT
                c.number_format = {2: INT, 3: USD, 4: USD, 5: PCT, 6: PCT, 7: PCT, 8: PCT}.get(j, "General")
        t = end + 1
        w.cell(row=t, column=1, value="Total").font = BOLD
        for j, colL in ((2, "B"), (3, "C"), (4, "D")):
            c = w.cell(row=t, column=j, value=f"=SUM({colL}2:{colL}{end})")
            c.font = BOLD
            c.number_format = INT if j == 2 else USD
        w.column_dimensions["A"].width = 48 if name == "B1_reason" else 16
        for j in range(2, 12):
            w.column_dimensions[get_column_letter(j)].width = 18
        if name_level:
            n = NM[NM["level"] == name_level].set_index("key")
            for j, hd in enumerate(("Name (from FNA_DESC)", "Name basis", "ECRs carrying that name"), 12):
                c = w.cell(row=1, column=j, value=hd)
                c.font = BOLD
                c.alignment = Alignment(wrap_text=True, vertical="top")
            for i, k in enumerate(keys, 2):
                tier = n.at[k, "tier"]
                if tier != "none":
                    for j, v in enumerate((n.at[k, "name"], BASIS[tier], int(n.at[k, "name_ecrs"])), 12):
                        w.cell(row=i, column=j, value=v).font = FONT
            w.column_dimensions["L"].width = 34
            w.column_dimensions["M"].width = 46
            w.column_dimensions["N"].width = 14
        w.freeze_panes = "B2"

    states = pd.read_csv(T / "states.csv")["STATE"].tolist()
    formula_sheet("States", "State", states, "B")
    a1 = L[(L["report"] == "A1_assembly") & (L["metric"] == "count")]["key"].tolist()
    formula_sheet("A1_assembly", "Assembly (VPPS first 3 segments)", a1, "E", name_level="assembly")
    b1 = L[(L["report"] == "B1_reason") & (L["metric"] == "count")]["key"].tolist()
    reasons_in_data = set(ecr["REASON"])
    assert set(b1) == reasons_in_data
    formula_sheet("B1_reason", "Reason code", b1, "C")

    # ---- value sheets ------------------------------------------------------------------------
    for rep, name, key_name in (("A2_vpps_code", "A2_vpps_code", "VPPS code"), ("A3_part_number", "A3_part_number", "PART_NUMBER"),
                                ("A3_new_part_number", "A3_new_part_number", "NEW_PART_NUMBER"), ("N1_part_number", "NoVPPS_part_number", "PART_NUMBER"),
                                ("N2_new_part_number", "NoVPPS_new_part_number", "NEW_PART_NUMBER"), ("N3_reason", "NoVPPS_reason", "Reason code")):
        df = level_sheet(L, rep, key_name)
        if rep in LEVEL_OF:
            df = add_names(df, NM, LEVEL_OF[rep], key_name)
        write_table(wb.create_sheet(name), df, level_formats(df) | {"ECRs on the code, whole extract": INT})

    names = {"metric": "Measure", "report": "Report", "parent": "Within", "parent_key": "Within value", "parent_rank": "Rank of parent", "rank": "Rank",
             "key": "Value", "ecr_count": "ECRs", "cost_given": "COST_USD as given ($)", "cost_recalc": "COST_USD recalculated ($)",
             "pct": "Share of measure", "cum_pct": "Cumulative share", "vital_few": "Vital few", "n_groups": "Groups at this level",
             "parent_ecrs": "ECRs in parent", "level_total": "Total of measure at this level"}
    order = ["metric", "report", "parent", "parent_key", "parent_rank", "rank", "key", "ecr_count", "cost_given", "cost_recalc", "pct", "cum_pct", "vital_few", "n_groups", "parent_ecrs", "level_total"]
    for fn, name in (("a_drill.csv", "A_drilldowns"), ("b_drill.csv", "B_drilldowns")):
        df = pd.read_csv(T / fn, dtype=str_cols)[order]
        nm_all = NM.assign(k=NM["level"] + "|" + NM["key"]).set_index("k")
        show = nm_all["name"].where(nm_all["tier"] != "single", nm_all["name"] + " *")
        parent_lv = df["parent"].map(LEVEL_OF)
        df.insert(df.columns.get_loc("parent_key") + 1, "Within name (from FNA_DESC)", (parent_lv + "|" + df["parent_key"]).map(show))
        if fn == "a_drill.csv":
            df.insert(df.columns.get_loc("key") + 1, "Value name (from FNA_DESC)", (df["report"].map(LEVEL_OF) + "|" + df["key"]).map(show))
        df = df.rename(columns=names)
        fm = level_formats(df) | {"Share of measure": PCT, "Cumulative share": PCT, "Total of measure at this level": '#,##0'}
        write_table(wb.create_sheet(name), df, fm)

    hm = pd.read_csv(T / "heatmap.csv")[["metric", "assembly_rank", "assembly", "reason_grp", "value", "share"]]
    hm = hm.rename(columns={"metric": "Measure", "assembly_rank": "Assembly rank", "assembly": "Assembly", "reason_grp": "Reason code", "value": "Value of measure", "share": "Share of assembly"})
    write_table(wb.create_sheet("B2_heatmap"), hm, {"Value of measure": '#,##0', "Share of assembly": PCT})

    pdet = pd.read_csv(T / "part_detail.csv", dtype=str_cols).rename(columns={
        "metric": "Measure", "scope": "Scope", "level": "Column", "rank": "Rank", "key": "Value", "ecr_count": "ECRs", "cost_given": "COST_USD as given ($)",
        "cost_recalc": "COST_USD recalculated ($)", "pct": "Share of measure", "cum_pct": "Cumulative share", "assemblies": "Assemblies", "states": "States (ECRs)",
        "reasons": "Top reason codes (ECRs)", "n_reasons": "Reason codes used", "largest_ecr": "Largest ECR (WO_NO)", "largest_ecr_value": "Largest ECR value",
        "largest_ecr_desc": "FNA_DESC on largest ECR"})
    lv = pdet["Column"].map(LEVEL_OF)
    nm_all = NM.assign(k=NM["level"] + "|" + NM["key"]).set_index("k")
    pdet.insert(pdet.columns.get_loc("Value") + 1, "Name (from FNA_DESC)", (lv + "|" + pdet["Value"]).map(nm_all["name"]))
    pdet.insert(pdet.columns.get_loc("Value") + 2, "Name basis", (lv + "|" + pdet["Value"]).map(nm_all["tier"]).map(BASIS))
    write_table(wb.create_sheet("Part_detail"), pdet, level_formats(pdet) | {"Share of measure": PCT, "Cumulative share": PCT, "Largest ECR value": '#,##0'})

    rec = pd.read_csv(T / "cost_reconciliation.csv", dtype=str_cols).rename(columns={"assembly": "ASSEMBLY", "cost_given": "COST_USD as given ($)", "cost_recalc": "COST_USD recalculated ($)", "difference": "Difference ($)", "rule": "Rule applied"})
    write_table(wb.create_sheet("Cost_reconciliation"), rec, {"COST": '#,##0.00', "EXCHANGE_RATE": '0.000000', "COST_USD as given ($)": '#,##0.00', "COST_USD recalculated ($)": '#,##0.00', "Difference ($)": '#,##0.00'})

    top = pd.read_csv(T / "top_ecrs.csv", dtype=str_cols).rename(columns={"metric": "Measure", "rank": "Rank", "assembly": "ASSEMBLY", "cost_given": "COST_USD as given ($)", "cost_recalc": "COST_USD recalculated ($)"})
    write_table(wb.create_sheet("Top_ECRs"), top, {"COST": '#,##0.00', "EXCHANGE_RATE": '0.000000', "COST_USD as given ($)": '#,##0.00', "COST_USD recalculated ($)": '#,##0.00'})

    nsheet = NM.rename(columns={"level": "Level", "key": "Code", "ecrs": "ECRs on the code", "ecrs_with_desc": "ECRs with an FNA_DESC", "distinct_desc": "Distinct FNA_DESC",
                                "top_desc": "Most common FNA_DESC", "top_desc_ecrs": "ECRs with it", "top_desc_tied": "Tied for most common", "top_term": "Most common leading term",
                                "top_term_ecrs": "ECRs with that term", "tier": "Rule met", "name": "Name shown", "name_ecrs": "ECRs carrying the name shown", "name_share": "Share of the code's ECRs"})
    nsheet["Rule met"] = nsheet["Rule met"].map({"name": "1 most common description", "term": "2 shared leading term", "single": "3 single ECR, not cross-checked", "none": "4 no name"})
    nsheet["Level"] = nsheet["Level"].map({"assembly": "Assembly", "vpps_code": "VPPS code", "part_number": "PART_NUMBER", "new_part_number": "NEW_PART_NUMBER"})
    nsheet = nsheet.sort_values(["Level", "ECRs on the code"], ascending=[True, False], kind="mergesort")
    write_table(wb.create_sheet("Names"), nsheet, {"Share of the code's ECRs": PCT, "ECRs on the code": INT})

    # ECR_data last
    wb.move_sheet("ECR_data", offset=len(wb.sheetnames))
    wb.save(dest)
    print("saved", dest, wb.sheetnames)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
