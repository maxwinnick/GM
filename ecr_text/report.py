"""Write the per-field Pareto tables: Excel workbook and Power BI CSVs (figures: ecr_text.figures)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import ParetoConfig

log = logging.getLogger(__name__)

README_ROWS = [
    ("What this is", "Word-frequency Pareto reports on the four masked ECR text fields, parsed two ways: "
                     "rule-based NLP (<FIELD>_NLP) and LLM key-term extraction (<FIELD>_AI)."),
    ("Counting unit", "ECRs (document frequency). A term counts once per ECR however often it repeats."),
    ("Scope", "All ECRs in every STATE (Closed and non-closed). See Summary sheet for counts."),
    ("WO 7.0M+ only fields", "DESC_FOR_CHANGE and REASON_FOR_CHANGE are blank for every ECR with WO_NO in "
                             "4,131,983-4,800,660, so their Paretos cover WO_NO 7,000,105+ only."),
    ("NLP method", "Scrub leftover identifiers; lowercase; split into statements; drop tokens with digits; "
                   "expand only GM-confirmed abbreviations (none confirmed yet); spaCy lemmatize; remove English + "
                   "domain stop words (resources/stopwords_domain.txt; negations kept). Bigrams never "
                   "cross a statement boundary."),
    ("AI method", "Scrubbed text (typed placeholders, person names removed) sent to the configured LLM with the "
                  "process-flow guide as context; returns up to 6 normalized 1-3 word key terms per text."),
    ("share_of_mentions / cum_share", "Term's share of all term-ECR mentions, and running total (the Pareto curve)."),
    ("abc_class", "A = terms making up the first 80% of mentions, B = next 15%, C = rest."),
    ("Concentration", "Number of terms needed to reach 50% / 80% of all term mentions."),
    ("log_odds_z", "Weighted log-odds (Monroe et al. 2008) of the term in the group vs. all other groups. "
                   "z > 1.96 = used notably more in that group."),
    ("Ubiquity", "Share of reported groups whose top-N frequency list contains the term. A value of 1.0 "
                 "means the term is in the top-N list of every reported group."),
    ("Stopword_Audit", "Every removed word and how many unique texts it was removed from - the list to review with GM."),
]


def top_terms_wide(overall: pd.DataFrame, n: int) -> pd.DataFrame:
    """Side-by-side top-N lists, one column pair per field x method x term type."""
    cols = {}
    for (f, m, t), g in overall.groupby(["field", "method", "term_type"], sort=False):
        g = g.nsmallest(n, "rank")
        cols[f"{f} {m} {t}"] = g["term"].tolist()
        cols[f"{f} {m} {t} %ECRs"] = (g["pct_of_ecrs"] * 100).round(1).tolist()
    return pd.DataFrame({k: pd.Series(v) for k, v in cols.items()})


def write_outputs(reports: dict[str, pd.DataFrame], out_dir: Path, cfg: ParetoConfig,
                  stopword_audit: pd.DataFrame | None = None,
                  leak_counts: pd.DataFrame | None = None) -> Path:
    out_dir = Path(out_dir)
    bi_dir = out_dir / "powerbi"
    bi_dir.mkdir(parents=True, exist_ok=True)

    overall = reports["overall"]
    sheets: dict[str, pd.DataFrame] = {
        "README": pd.DataFrame(README_ROWS, columns=["item", "explanation"]),
        "Summary": reports["summary"],
        "Concentration": reports["concentration"],
        "Top_Terms": top_terms_wide(overall, cfg.top_terms),
        "Overall_Pareto": overall,
    }
    if not reports["nlp_vs_ai"].empty:
        sheets["NLP_vs_AI"] = reports["nlp_vs_ai"]
    for key, df in reports.items():
        if key.startswith("by_"):
            sheets[f"By_{key[3:]}"] = df
    sheets["Ubiquity"] = reports["ubiquity"]
    if stopword_audit is not None:
        sheets["Stopword_Audit"] = stopword_audit
    if leak_counts is not None:
        sheets["Masking_Leaks"] = leak_counts

    xlsx = out_dir / "ecr_text_pareto.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                width = min(60, max(10, len(str(col)) + 2,
                                    int(df[col].astype(str).str.len().quantile(0.9)) + 2 if len(df) else 10))
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
            pct_cols = [c for c in df.columns if str(c).startswith(("pct", "share", "cum_share"))
                        or str(c).endswith(("share", "jaccard", "overlap")) or "pct_" in str(c)]
            for c in pct_cols:
                j = list(df.columns).index(c) + 1
                for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                    row[0].number_format = "0.0%"
            for c in [c for c in df.columns if "cost_usd" in str(c)]:
                j = list(df.columns).index(c) + 1
                for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                    row[0].number_format = "$#,##0"

    for name in ["Overall_Pareto", "Ubiquity"] + [n for n in sheets if n.startswith("By_")]:
        sheets[name].to_csv(bi_dir / f"{name.lower()}.csv", index=False)

    log.info("Wrote %s and Power BI CSVs in %s (figures: ecr_text.figures)", xlsx, bi_dir)
    return xlsx
