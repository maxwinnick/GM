"""Write a synthetic ECR extract with the same schema as GM's masked workbook (sheet ``result``).

Every value is randomly generated from small hand-written templates; nothing comes from GM data.
It exists so the pipeline, the tests and ``make demo`` can run on any machine.

    python tools/synthetic_extract.py data/synthetic_extract.xlsx --rows 1500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "WO_NO", "STATE", "DRIVER", "REASON", "CREATED_ON", "MASKED_CREATED_ON", "LEAD_PROGRAM",
    "MASKED_LEAD_PROGRAM", "MODEL_YEAR", "MASKED_MODEL_YEAR", "PROGRAMS", "MASKED_PROGRAMS", "TITLE",
    "MASKED_TITLE", "TYPE", "SUBTYPE", "COST_AREA", "COST_TYPE", "FNA_DESC", "IS_OVERALL",
    "COST_DELTA_INDICATOR", "TOOLING_TYPE", "COST_COMMENT", "MASKED_COST_COMMENT", "SUPPLIER_NAME",
    "MASKED_SUPPLIER_NAME", "VPPS", "PRTS_NO", "PRTS_ISSUE_NUMBER_LINK", "PART_NUMBER",
    "MASKED_PART_NUMBER", "NEW_PART_NUMBER", "MASKED_NEW_PART_NUMBER", "DESC_FOR_CHANGE",
    "MASKED_DESC_FOR_CHANGE", "REASON_FOR_CHANGE", "MASKED_REASON_FOR_CHANGE", "COST", "MASKED_COST",
    "EXCHANGE_RATE", "MASKED_EXCHANGE_RATE", "COST_USD", "MASKED_COST_USD",
]

STATES = (["Closed"] * 14 + ["Execute"] * 3 + ["Cancelled", "Returned", "Elaborate", "Review", "Investigate"])
REASONS = ["Reqmts Not Met - Other", "Cost Reduction or Value Optimization", "Supplier or Purchasing Request",
           "Initial Release", "Mfg No Other Mfg Code Applies", "PC-CR", "Design Improvement", "Quality Issue"]
REASON_P = [0.30, 0.14, 0.12, 0.10, 0.10, 0.08, 0.08, 0.08]
DRIVERS = ["EGM", "Engineering Group Manager (EGM)", "PRTS", "Problem Resolution Tracking System", "CRDN",
           "Supplier", "EGM|PRTS", None]
SUBTYPES = ["PWO - Express - Part Chg with New Part Number", "PWO - Exp Launch - Part Chg with New Part Number",
            "PWO - Sourcing", "PWO - Initial P Release", "PWO - Usage and Spec Change", "AWO - Standard - Cost Change"]
ASSEMBLIES = {  # VPPS assembly -> (component words, FNA_DESC)
    "80.06.01": ("harness connector clip", "HARNESS ASM-I/P WRG"),
    "40.03.02": ("door trim panel", "TRIM ASM-FRT S/D"),
    "60.05.01": ("headlamp lens housing", "HEADLAMP ASM-FRT"),
    "10.01.01": ("turbocharger hose clamp", "TURBOCHARGER ASM"),
    "40.01.04": ("seat foam cushion", "SEAT ASM-FRT"),
    "20.02.01": ("coolant hose pipe", "HOSE ASM-RAD INL"),
    "70.03.02": ("front fascia bracket", "FASCIA ASM-FRT BPR"),
    "50.01.01": ("floor console armrest", "CONSOLE ASM-FLR"),
}
WHY = ["cost reduction", "supplier request", "tier resiliency", "design update", "express launch issue",
       "batch release", "durability concern", "material change for cost saving", "noise issue", "fit issue"]
ACTIONS = ["add", "remove", "update", "change", "replace", "revise", "release"]
COMMENTS = ["No tooling cost impact", "Piece cost impact per supplier quote", "Tooling cost approved by PPCO",
            "Supplier quote pending", "No piece cost impact", "Cost reconciliation with supplier"]


def make(rows: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_old = rows // 2
    wo = np.concatenate([rng.choice(np.arange(4_131_983, 4_800_660), n_old, replace=False),
                         rng.choice(np.arange(7_000_105, 7_186_370), rows - n_old, replace=False)])
    asm_keys = list(ASSEMBLIES)
    recs = []
    for i, w in enumerate(wo):
        old = w < 5_000_000
        has_vpps = rng.random() < 0.76
        asm = asm_keys[rng.integers(len(asm_keys))]
        comp, fna = ASSEMBLIES[asm]
        part = comp.split()[rng.integers(3)]
        why = WHY[rng.integers(len(WHY))]
        act = ACTIONS[rng.integers(len(ACTIONS))]
        prog = int(rng.integers(2, 25))
        title = f"[Program {prog}] {act} {part} {comp.split()[0]} for {why}"
        desc = None if old else f"{act.capitalize()} {comp} PN [{rng.integers(1e7, 9e7)}] to new design; {part} {why}."
        reason = None if old else f"{why.capitalize()} identified by program team; {part} {act} required."
        comment = COMMENTS[rng.integers(len(COMMENTS))] if rng.random() < 0.88 else None
        cost = float(np.round(rng.lognormal(8, 2.2) * (1 if rng.random() > 0.3 else 0), 2))
        rate = float(rng.choice([0.92, 1.08, 0.14, 0.0069])) if rng.random() < 0.23 else np.nan
        cost_usd = cost if np.isnan(rate) else (cost * rate if rng.random() < 0.7 else cost)
        created = pd.Timestamp("2020-10-01") + pd.Timedelta(days=int(rng.integers(0, 2160)))
        pn = str(rng.integers(10_000_000, 99_999_999))
        recs.append({
            "WO_NO": int(w), "STATE": STATES[rng.integers(len(STATES))],
            "DRIVER": DRIVERS[rng.integers(len(DRIVERS))],
            "REASON": rng.choice(REASONS, p=REASON_P), "CREATED_ON": created, "MASKED_CREATED_ON": created.normalize(),
            "LEAD_PROGRAM": "-" if not old else f"P{prog:03d}", "MASKED_LEAD_PROGRAM": "Program 1" if not old else f"Program {prog}",
            "MODEL_YEAR": int(rng.integers(2024, 2031)), "MASKED_MODEL_YEAR": 2027,
            "PROGRAMS": f"P{prog:03d}", "MASKED_PROGRAMS": f"Program {prog}",
            "TITLE": title, "MASKED_TITLE": title, "TYPE": "PWO", "SUBTYPE": SUBTYPES[rng.integers(len(SUBTYPES))],
            "COST_AREA": "Engineering Impact", "COST_TYPE": "Piece Cost", "FNA_DESC": fna if has_vpps and rng.random() < 0.8 else None,
            "IS_OVERALL": int(rng.integers(2)), "COST_DELTA_INDICATOR": int(rng.integers(2)),
            "TOOLING_TYPE": "Tooling" if rng.random() < 0.35 else None,
            "COST_COMMENT": comment, "MASKED_COST_COMMENT": comment,
            "SUPPLIER_NAME": f"Supplier Co {rng.integers(1, 30)}", "MASKED_SUPPLIER_NAME": f"Supplier {rng.integers(1, 30)}",
            "VPPS": f"{asm}.{rng.integers(1, 12):02d}" if has_vpps else None,
            "PRTS_NO": None, "PRTS_ISSUE_NUMBER_LINK": None,
            "PART_NUMBER": pn, "MASKED_PART_NUMBER": float(i), "NEW_PART_NUMBER": float(int(pn) + 1),
            "MASKED_NEW_PART_NUMBER": float(i + 1),
            "DESC_FOR_CHANGE": desc, "MASKED_DESC_FOR_CHANGE": desc,
            "REASON_FOR_CHANGE": reason, "MASKED_REASON_FOR_CHANGE": reason,
            "COST": cost, "MASKED_COST": int(cost), "EXCHANGE_RATE": rate, "MASKED_EXCHANGE_RATE": 1.0,
            "COST_USD": float(np.round(cost_usd, 2)), "MASKED_COST_USD": float(np.round(cost_usd, 0)),
        })
    return pd.DataFrame(recs, columns=COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    make(a.rows, a.seed).to_excel(a.out, sheet_name="result", index=False)
    print(f"wrote {a.out} ({a.rows:,} synthetic ECRs)")


if __name__ == "__main__":
    main()
