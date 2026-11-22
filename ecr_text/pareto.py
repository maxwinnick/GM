"""Word-frequency Pareto reports on the NLP and AI fields.

Counting unit is the ECR (document frequency): a term counts once per ECR no matter how
often the ECR repeats it, so a single verbose record cannot dominate a Pareto.

Reports
* Overall Pareto per field x method x term type (NLP unigram, bigram, trigram; AI key term):
  ECR count, % of ECRs with text, share of mentions, cumulative share, ABC class and,
  optionally, the absolute USD cost of the ECRs that mention the term.
* Pareto by dimension (Program, Supplier, VPPS L1, VPPS L2): the most frequent terms per
  group and the most *distinctive* terms per group. Distinctiveness is the weighted
  log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008),
  reported as a z-score: z > 1.96 means the term is used notably more in that group than
  in the rest of the corpus.
* Ubiquity: in how many groups' top-N list each term appears. Terms that are in almost every
  group's list are the "same words in every change" GM asked about.
* NLP vs AI comparison per field.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ai import ai_terms
from .config import BLANK_PROGRAM_MASK, TEXT_COLUMNS, WO_RANGE_LABELS, WO_RANGE_SPLIT, ParetoConfig
from .nlp import ngrams, tokens

log = logging.getLogger(__name__)

NO_GROUP_PREFIX = "(no "


@dataclass(frozen=True)
class TermSpec:
    field: str
    method: str      # "NLP" | "AI"
    term_type: str   # "unigram" | "bigram" | "trigram" | "key_term"

    @property
    def column(self) -> str:
        return f"{self.field}_{self.method}"

    @property
    def label(self) -> str:
        return f"{self.field} | {self.method} {self.term_type}"

    def extract(self, value) -> set[str]:
        if self.method == "AI":
            return set(ai_terms(value))
        if self.term_type == "unigram":
            return set(tokens(value))
        return set(ngrams(value, {"bigram": 2, "trigram": 3}[self.term_type]))


def term_specs(df: pd.DataFrame) -> list[TermSpec]:
    specs = []
    for f in TEXT_COLUMNS:
        specs += [TermSpec(f, "NLP", "unigram"), TermSpec(f, "NLP", "bigram"), TermSpec(f, "NLP", "trigram")]
        if f"{f}_AI" in df and df[f"{f}_AI"].notna().any():
            specs.append(TermSpec(f, "AI", "key_term"))
    return specs


# ---------------------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------------------
def add_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prog = out["MASKED_LEAD_PROGRAM"].astype("string")
    out["DIM_PROGRAM"] = prog.mask(prog.eq(BLANK_PROGRAM_MASK) | prog.isna(), "(no lead program)")
    out["DIM_SUPPLIER"] = out["MASKED_SUPPLIER_NAME"].astype("string").fillna("(no supplier)")
    vpps = out["VPPS"].astype("string").str.strip()
    parts = vpps.str.split(".")
    out["DIM_VPPS_L1"] = parts.str[0].fillna("(no VPPS)")
    out["DIM_VPPS_L2"] = (parts.str[0] + "." + parts.str[1]).fillna("(no VPPS)")
    out["DIM_VPPS_L3"] = (parts.str[0] + "." + parts.str[1] + "." + parts.str[2]).fillna("(no VPPS)")
    out["SYSTEM_ERA"] = np.where(out["WO_NO"] >= WO_RANGE_SPLIT, WO_RANGE_LABELS[1], WO_RANGE_LABELS[0])
    # UNCONFIRMED cost rule (only used when ParetoConfig.cost_weighted is True).
    rate = pd.to_numeric(out.get("EXCHANGE_RATE"), errors="coerce").fillna(1.0)
    out["COST_USD_CALC"] = pd.to_numeric(out.get("COST"), errors="coerce") * rate
    return out


def filter_scope(df: pd.DataFrame, cfg: ParetoConfig) -> pd.DataFrame:
    if cfg.states is None:
        return df
    return df[df["STATE"].isin(cfg.states)].copy()


def _doc_terms(df: pd.DataFrame, spec: TermSpec) -> pd.DataFrame:
    """Long frame: one row per (ECR, distinct term). Only ECRs with source text."""
    has_text = df[TEXT_COLUMNS[spec.field]].notna()
    sub = df.loc[has_text, ["WO_NO", spec.column, "COST_USD_CALC"]]
    sets = sub[spec.column].map(spec.extract)
    long = (sub.assign(term=sets.map(sorted)).explode("term").dropna(subset=["term"]))
    return long


# ---------------------------------------------------------------------------------------
# overall
# ---------------------------------------------------------------------------------------
def overall_pareto(df: pd.DataFrame, spec: TermSpec, cfg: ParetoConfig) -> pd.DataFrame:
    n_docs = int(df[TEXT_COLUMNS[spec.field]].notna().sum())
    long = _doc_terms(df, spec)
    if long.empty:
        return pd.DataFrame()
    agg = long.groupby("term").agg(n_ecrs=("WO_NO", "size"),
                                   abs_cost_usd=("COST_USD_CALC", lambda s: s.abs().sum()))
    agg = agg.sort_values(["n_ecrs", "abs_cost_usd"], ascending=False).reset_index()
    agg["rank"] = np.arange(1, len(agg) + 1)
    agg["pct_of_ecrs"] = agg["n_ecrs"] / n_docs
    agg["share_of_mentions"] = agg["n_ecrs"] / agg["n_ecrs"].sum()
    agg["cum_share"] = agg["share_of_mentions"].cumsum()
    a, b = cfg.abc_breaks
    prev = agg["cum_share"] - agg["share_of_mentions"]  # class by where the term starts
    agg["abc_class"] = np.select([prev < a, prev < b], ["A", "B"], "C")
    total_cost = df.loc[df[TEXT_COLUMNS[spec.field]].notna(), "COST_USD_CALC"].abs().sum()
    agg["pct_of_abs_cost"] = agg["abs_cost_usd"] / total_cost if total_cost else np.nan
    if not cfg.cost_weighted:
        agg = agg.drop(columns=["abs_cost_usd", "pct_of_abs_cost"])
    agg.insert(0, "field", spec.field)
    agg.insert(1, "method", spec.method)
    agg.insert(2, "term_type", spec.term_type)
    agg.insert(3, "ecrs_with_text", n_docs)
    return agg


# ---------------------------------------------------------------------------------------
# by dimension
# ---------------------------------------------------------------------------------------
def _log_odds_z(group_counts: pd.Series, rest_counts: pd.Series, prior: pd.Series) -> pd.Series:
    """Weighted log-odds ratio with informative Dirichlet prior (Monroe et al. 2008)."""
    terms = prior.index
    y_i = group_counts.reindex(terms, fill_value=0).astype(float)
    y_j = rest_counts.reindex(terms, fill_value=0).astype(float)
    a = prior.astype(float)
    n_i, n_j, a0 = y_i.sum(), y_j.sum(), a.sum()
    delta = (np.log((y_i + a) / (n_i + a0 - y_i - a)) - np.log((y_j + a) / (n_j + a0 - y_j - a)))
    var = 1.0 / (y_i + a) + 1.0 / (y_j + a)
    return delta / np.sqrt(var)


def pareto_by_dimension(df: pd.DataFrame, spec: TermSpec, dim_label: str, dim_col: str,
                        cfg: ParetoConfig) -> pd.DataFrame:
    has_text = df[TEXT_COLUMNS[spec.field]].notna()
    docs_per_group = df.loc[has_text, dim_col].value_counts()
    docs_per_group = docs_per_group[~docs_per_group.index.astype(str).str.startswith(NO_GROUP_PREFIX)]
    groups = docs_per_group[docs_per_group >= cfg.min_group_docs].head(cfg.top_groups)
    if groups.empty:
        return pd.DataFrame()

    long = _doc_terms(df, spec).merge(df[["WO_NO", dim_col]], on="WO_NO")
    total = long["term"].value_counts()
    vocab = total[total >= cfg.min_term_docs]
    # Prior scaled to a modest pseudo-count mass so rare terms are shrunk toward zero.
    prior = vocab / vocab.sum() * min(1000.0, float(vocab.sum()))

    rows = []
    for group, n_docs in groups.items():
        in_g = long[dim_col] == group
        g_counts = long.loc[in_g, "term"].value_counts()
        rest = (total - g_counts.reindex(total.index, fill_value=0))
        z = _log_odds_z(g_counts, rest, prior)

        top = g_counts.head(cfg.terms_per_group)
        for rank, (term, n) in enumerate(top.items(), start=1):
            rows.append({"metric": "frequency", "group": group, "group_ecrs": n_docs, "rank": rank,
                         "term": term, "n_ecrs_in_group": n, "pct_of_group": n / n_docs,
                         "log_odds_z": z.get(term, np.nan)})
        cand = z[g_counts.reindex(z.index, fill_value=0) >= 3].sort_values(ascending=False)
        for rank, (term, zval) in enumerate(cand.head(cfg.terms_per_group).items(), start=1):
            n = int(g_counts.get(term, 0))
            rows.append({"metric": "distinctive", "group": group, "group_ecrs": n_docs, "rank": rank,
                         "term": term, "n_ecrs_in_group": n, "pct_of_group": n / n_docs,
                         "log_odds_z": zval})
    out = pd.DataFrame(rows)
    out.insert(0, "dimension", dim_label)
    out.insert(1, "field", spec.field)
    out.insert(2, "method", spec.method)
    out.insert(3, "term_type", spec.term_type)
    return out


def ubiquity(by_dim: pd.DataFrame) -> pd.DataFrame:
    """How many groups list each term in their top-N by frequency."""
    if by_dim.empty:
        return by_dim
    freq = by_dim[by_dim["metric"] == "frequency"]
    keys = ["dimension", "field", "method", "term_type"]
    n_groups = freq.groupby(keys)["group"].nunique().rename("groups_reported")
    u = (freq.groupby(keys + ["term"])
             .agg(groups_with_term_in_top=("group", "nunique"),
                  median_pct_of_group=("pct_of_group", "median"))
             .reset_index().merge(n_groups.reset_index(), on=keys))
    u["pct_of_groups"] = u["groups_with_term_in_top"] / u["groups_reported"]
    return u.sort_values(keys + ["pct_of_groups", "median_pct_of_group"],
                         ascending=[True] * 4 + [False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------
def field_summary(df_all: pd.DataFrame, df_scope: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f, src in TEXT_COLUMNS.items():
        has = df_scope[src].notna()
        nlp_ok = df_scope.loc[has, f"{f}_NLP"].notna()
        row = {
            "field": f, "source_column": src,
            "ecrs_all_states": len(df_all), "with_text_all_states": int(df_all[src].notna().sum()),
            "ecrs_in_scope": len(df_scope), "with_text_in_scope": int(has.sum()),
            "pct_wo_7m_plus_of_text": float((df_scope.loc[has, "SYSTEM_ERA"] == WO_RANGE_LABELS[1]).mean())
            if has.any() else np.nan,
            "nlp_nonempty": int(nlp_ok.sum()),
            "nlp_avg_unigrams": float(df_scope.loc[has, f"{f}_NLP"].map(lambda v: len(tokens(v))).mean()),
        }
        ai_col = f"{f}_AI"
        if ai_col in df_scope:
            ai = df_scope.loc[has, ai_col]
            row["ai_parsed"] = int(ai.notna().sum())
            row["ai_nonempty"] = int(ai.fillna("").ne("").sum())
            row["ai_avg_terms"] = float(ai.map(lambda v: len(ai_terms(v))).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def concentration(overall_full: pd.DataFrame) -> pd.DataFrame:
    """How concentrated each Pareto is: terms needed for 50% / 80% of mentions."""
    rows = []
    for (f, m, t), g in overall_full.groupby(["field", "method", "term_type"], sort=False):
        cum = g.sort_values("rank")["cum_share"].to_numpy()
        rows.append({
            "field": f, "method": m, "term_type": t, "distinct_terms": len(g),
            "terms_for_50pct": int(np.searchsorted(cum, 0.5) + 1),
            "terms_for_80pct": int(np.searchsorted(cum, 0.8) + 1),
            "top10_share": float(cum[min(9, len(cum) - 1)]),
            "top30_share": float(cum[min(29, len(cum) - 1)]),
        })
    return pd.DataFrame(rows)


def nlp_vs_ai(df: pd.DataFrame, overall: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    rows = []
    for f, src in TEXT_COLUMNS.items():
        ai_col = f"{f}_AI"
        if ai_col not in df or df[ai_col].isna().all():
            continue
        both = df[df[src].notna() & df[ai_col].notna()]
        nlp_tok = both[f"{f}_NLP"].map(lambda v: set(tokens(v)))
        ai_tok = both[ai_col].map(lambda v: {w for t in ai_terms(v) for w in t.split()})
        overlap = [len(a & b) / len(a | b) if (a | b) else np.nan for a, b in zip(nlp_tok, ai_tok)]
        any_hit = [bool(a & b) for a, b in zip(nlp_tok, ai_tok) if b]

        def top(method, term_type):
            sel = overall[(overall.field == f) & (overall.method == method) & (overall.term_type == term_type)]
            return sel.nsmallest(top_n, "rank")["term"].tolist()

        ai_top_words = {w for t in top("AI", "key_term") for w in t.split()}
        nlp_top = set(top("NLP", "unigram"))
        rows.append({
            "field": f, "ecrs_compared": len(both),
            "mean_per_ecr_word_jaccard": float(np.nanmean(overlap)) if overlap else np.nan,
            "pct_ecrs_ai_shares_a_word_with_nlp": float(np.mean(any_hit)) if any_hit else np.nan,
            f"top{top_n}_word_overlap": len(ai_top_words & nlp_top) / len(ai_top_words | nlp_top)
            if (ai_top_words | nlp_top) else np.nan,
            "ai_distinct_terms": int(overall[(overall.field == f) & (overall.method == "AI")]["term"].nunique()),
            "nlp_distinct_unigrams": int(overall[(overall.field == f) & (overall.method == "NLP")
                                                 & (overall.term_type == "unigram")]["term"].nunique()),
        })
    return pd.DataFrame(rows)


def run_reports(df_parsed: pd.DataFrame, cfg: ParetoConfig) -> dict[str, pd.DataFrame]:
    df_all = add_dimensions(df_parsed)
    df = filter_scope(df_all, cfg)
    log.info("Pareto scope: %d of %d ECRs (states=%s)", len(df), len(df_all), cfg.states)
    specs = term_specs(df)

    overall_full = pd.concat([overall_pareto(df, s, cfg) for s in specs], ignore_index=True)
    overall = overall_full[overall_full["rank"] <= cfg.max_overall_terms].reset_index(drop=True)
    by_dim = {}
    for label, col in cfg.dimensions.items():
        parts = [p for p in (pareto_by_dimension(df, s, label, col, cfg) for s in specs) if not p.empty]
        if not parts:
            log.warning("No %s group has %d+ ECRs; its Pareto sheet is empty.", label, cfg.min_group_docs)
        by_dim[label] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    filled = [v for v in by_dim.values() if not v.empty]
    all_dim = pd.concat(filled, ignore_index=True) if filled else pd.DataFrame()

    return {
        "summary": field_summary(df_all, df),
        "concentration": concentration(overall_full),
        "overall": overall,
        **{f"by_{k}": v for k, v in by_dim.items()},
        "ubiquity": ubiquity(all_dim),
        "nlp_vs_ai": nlp_vs_ai(df, overall),
        "scope": df,
    }
