"""Shared loading, corpus, split and evaluation used by all three approaches.

Every approach writes the same two tables, so they can be scored identically:

* ``assignments.csv``: WO_NO, primary_topic (int, -1 = unassigned), topics (";"-joined ids),
  primary_weight (approach-specific confidence, 0-1)

Every model is fit inside one WO partition (config ``partitions``), because the WO range
decides which free text an ECR has.
* ``topics.csv``: topic_id, label, top_terms, n_ecrs, share, ...

``evaluate()`` turns an assignment into the comparison metrics.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
WO_SPLIT = 7_000_000
RANGE_OLD, RANGE_NEW = "WO 4.1M-4.8M (title only)", "WO 7.0M+ (full text)"
OTHER_CODE = "Reqmts Not Met - Other"
_MASK = re.compile(r"\[+\s*(?:Program\s+\d+|\d+)\s*\]+", flags=re.IGNORECASE)
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- config / IO
def resolve_input(base: Path, workbook: str | Path) -> Path:
    """The configured workbook, or the only .xlsx in data/ if the configured one is missing."""
    p = Path(workbook)
    p = p if p.is_absolute() else base / p
    if p.exists():
        return p
    found = sorted(x for x in (base / "data").glob("*.xlsx") if not x.name.startswith("~$"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"Input workbook not found: {p}. Put GM's extract in data/ (see data/README.md) "
        "or set input.workbook in config.yaml.")


def load_config(path: str | Path = ROOT / "config.yaml", outputs: str | Path | None = None) -> dict:
    """Read the ``topics`` section of the project config and resolve every path it needs.

    ``outputs`` overrides the config's output folder (the pipeline passes its --outputs value).
    """
    path = Path(path)
    full = yaml.safe_load(path.read_text())
    base = path.parent
    cfg = dict(full["topics"])
    outputs = base / (outputs or full.get("outputs", "outputs"))
    emb = Path(cfg["bertopic"]["embedding_model"])
    cfg["sheet"] = full["input"].get("sheet", "result")
    cfg["paths"] = {
        "extract_xlsx": resolve_input(base, full["input"]["workbook"]),
        "phase1_parsed": outputs / "01_text" / "ecr_text_parsed.csv",
        "embedding_model": emb if emb.is_absolute() else base / emb,
        "driver_map": ROOT / "resources" / "driver_map.csv",
        "theme_stopwords": ROOT / "resources" / "theme_stopwords.txt",
        "outputs": outputs / "05_topics",
    }
    cfg["paths"]["extract"] = cfg["paths"]["outputs"] / "extract.parquet"
    cfg["paths"]["outputs"].mkdir(parents=True, exist_ok=True)
    return cfg


def out_dir(cfg: dict, name: str) -> Path:
    d = cfg["paths"]["outputs"] / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------- extract
def load_extract(cfg: dict) -> pd.DataFrame:
    """One row per ECR with derived fields. Asserts the known row count and cost totals."""
    p, src = cfg["paths"]["extract"], cfg["paths"]["extract_xlsx"]
    # The parquet copy is a cache of the workbook; it is rebuilt whenever the workbook is newer.
    if p.exists() and p.stat().st_mtime >= src.stat().st_mtime:
        df = pd.read_parquet(p)
    else:
        df = pd.read_excel(src, sheet_name=cfg.get("sheet", "result"),
                           dtype={"WO_NO": str, "VPPS": str, "PART_NUMBER": str, "NEW_PART_NUMBER": str})
        for c in df.columns[df.dtypes == object]:      # mixed-type columns break parquet
            df[c] = df[c].map(lambda v: v if v is None or isinstance(v, str) or pd.isna(v) else str(v))
        df.to_parquet(p)
    df["WO_NO"] = df["WO_NO"].astype(str).str.strip()
    assert df["WO_NO"].is_unique, "WO_NO must be unique"
    chk = cfg.get("checks") or {}
    if chk.get("n_ecrs") is not None:
        assert len(df) == chk["n_ecrs"], f"expected {chk['n_ecrs']} ECRs, got {len(df)}"

    df["cost_given"] = pd.to_numeric(df["COST_USD"], errors="coerce").fillna(0.0)
    cost = pd.to_numeric(df["COST"], errors="coerce")
    rate = pd.to_numeric(df["EXCHANGE_RATE"], errors="coerce")
    df["cost_recalc"] = np.where(rate.notna(), cost * rate, df["cost_given"])
    df["cost_recalc"] = df["cost_recalc"].fillna(0.0)
    for col, key in (("cost_given", "cost_given_musd"), ("cost_recalc", "cost_recalc_musd")):
        if chk.get(key) is None:
            continue
        got = df[col].sum() / 1e6
        assert abs(got - chk[key]) < 0.1, f"{col} total {got:.1f}M != {chk[key]}M"

    df["WO_RANGE"] = np.where(df["WO_NO"].astype(int) >= WO_SPLIT, RANGE_NEW, RANGE_OLD)
    v = df["VPPS"].fillna("").astype(str).str.strip()
    parts = v.str.split(".")
    df["VPPS_L1"] = np.where(v != "", parts.str[0], None)
    df["VPPS_L2"] = np.where(v != "", parts.str[:2].str.join("."), None)
    df["VPPS_L3"] = np.where(v != "", parts.str[:3].str.join("."), None)
    df["REASON"] = df["REASON"].fillna("(blank)")
    df["IS_OTHER"] = df["REASON"].eq(OTHER_CODE)
    df["DRIVER_N"] = normalize_driver(df["DRIVER"], cfg)
    sub = df["SUBTYPE"].fillna("").astype(str)
    df["SUB_CLASS"] = sub.str.extract(r"^(\w+)\s*-")[0].fillna("")
    df["SUB_TRACK"] = np.select([sub.str.contains("Exp Launch"), sub.str.contains("Express"),
                                 sub.str.contains("Standard")], ["Exp Launch", "Express", "Standard"], "Regular")
    df["SUB_KIND"] = (sub.str.replace(r"^\w+\s*-\s*", "", regex=True)
                      .str.replace(r"^(Express|Exp Launch|Standard)\s*-\s*", "", regex=True).str.strip())
    p = cfg["paths"].get("phase1_parsed")
    if p is not None and Path(p).exists():
        nlp = pd.read_csv(p, dtype={"WO_NO": str}, encoding="utf-8-sig",
                          usecols=lambda c: c == "WO_NO" or c.endswith("_NLP"))
        nlp["WO_NO"] = nlp["WO_NO"].str.strip()
        df = df.merge(nlp, on="WO_NO", how="left", validate="one_to_one")
    return df.reset_index(drop=True)


def normalize_driver(s: pd.Series, cfg: dict) -> pd.Series:
    """Map long and short DRIVER spellings to one code; several drivers are ';'-joined (sorted, unique)."""
    m = pd.read_csv(cfg["paths"]["driver_map"])
    mp = dict(zip(m["raw"].str.strip(), m["driver"]))

    def one(v):
        if not isinstance(v, str) or not v.strip():
            return ""
        return ";".join(sorted({mp.get(x.strip(), x.strip()) for x in v.split("|") if x.strip()}))
    return s.map(one)


def partition_mask(df: pd.DataFrame, cfg: dict, name: str) -> pd.Series:
    p = cfg["partitions"][name]
    wo = df["WO_NO"].astype(int)
    return (wo >= p["wo_min"]) & (wo <= p["wo_max"])


def structured_tokens(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    """Structured fields as pseudo-words, e.g. 'driver_prts', 'track_exp_launch', 'kind_sourcing'."""
    def slug(x):
        return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")
    out = pd.Series([[] for _ in range(len(df))], index=df.index)
    if "DRIVER" in fields:
        out = out + df["DRIVER_N"].map(lambda v: [f"driver_{slug(x)}" for x in v.split(";") if x])
    if "SUBTYPE" in fields:
        out = out + df["SUB_TRACK"].map(lambda v: [f"track_{slug(v)}"]) + df["SUB_KIND"].map(
            lambda v: [f"kind_{slug(v)}"] if v else [])
    return out.map(lambda l: " ".join(l))


def clean_raw_text(text: str, strip_masks: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("_x000d_", " ").replace("_x000D_", " ")   # Excel carriage-return escapes
    if strip_masks:
        text = _MASK.sub(" ", text)
    return _WS.sub(" ", text).strip()


FIELD_LABEL = {"TITLE": "Title", "COST_COMMENT": "Cost comment", "DESC_FOR_CHANGE": "Description",
               "REASON_FOR_CHANGE": "Reason"}


def text_col(cfg: dict, field: str) -> str:
    return f"MASKED_{field}" if cfg.get("masked", True) else field


def build_documents(df: pd.DataFrame, cfg: dict, partition: str) -> pd.Series:
    """Raw (not lemmatized) ECR text for embeddings and LLMs, using the partition's fields.

    Fields are labeled and joined with ' || ' so a reader (or LLM) can see the boundaries.
    Structured fields (if any for this partition) are appended as 'Driver: ...' and
    'Change type: ...' so they can be cited as evidence.
    """
    p = cfg["partitions"][partition]
    cols = [text_col(cfg, f) for f in p["text_fields"]]
    labels = [FIELD_LABEL[f] for f in p["text_fields"]]
    out = []
    for i, row in zip(df.index, df[cols].itertuples(index=False)):
        segs = []
        for lab, val in zip(labels, row):
            t = clean_raw_text(val, cfg.get("strip_mask_tokens", True))
            if t:
                segs.append(f"{lab}: {t}")
        if "DRIVER" in p["structured_fields"] and df.at[i, "DRIVER_N"]:
            segs.append(f"Driver: {df.at[i, 'DRIVER_N'].replace(';', ', ')}")
        if "SUBTYPE" in p["structured_fields"] and isinstance(df.at[i, "SUBTYPE"], str):
            segs.append(f"Change type: {df.at[i, 'SUBTYPE']}")
        out.append(" || ".join(segs))
    return pd.Series(out, index=df.index, name="doc")


# --------------------------------------------------------------------------- split
def make_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Boolean Series: True = held-out (30%), stratified by STATE x VPPS L1, fixed seed."""
    strata = df["STATE"].astype(str) + "|" + df["VPPS_L1"].fillna("none").astype(str)
    counts = strata.map(strata.value_counts())
    strata = strata.where(counts >= 10, "rare")
    _, test_idx = train_test_split(df.index, test_size=cfg["split"]["holdout_share"],
                                   random_state=cfg["seed"], stratify=strata)
    held = pd.Series(False, index=df.index, name="HELD_OUT")
    held.loc[test_idx] = True
    return held


# --------------------------------------------------------------------------- evaluation
def _axis(ent: float, n_vpps: int, cfg: dict) -> str:
    a = cfg["axis"]
    if n_vpps < a["min_ecrs_with_vpps"]:
        return "too few with VPPS"
    if ent <= a["what_max_entropy"]:
        return "what (component)"
    if ent >= a["why_min_entropy"]:
        return "why (driver)"
    return "mixed"


def topic_profiles(df: pd.DataFrame, labels: pd.Series, cfg: dict, topics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per topic: size, WO-range mix, reason-code mix, assembly concentration and axis proposal.

    Assembly entropy is normalized by log(number of assemblies in the whole corpus at that
    level), so topics of different sizes are comparable.
    """
    level = cfg["axis"]["level"]
    n_levels = df[level].nunique()
    rows = []
    d = df.assign(_t=labels.values)
    for t, g in d.groupby("_t"):
        if t == -1:
            continue
        rc = g["REASON"].value_counts()
        asm = g[level].dropna().value_counts()
        ent = float(entropy(asm.to_numpy()) / np.log(n_levels)) if len(asm) > 1 else 0.0
        rows.append({
            "topic_id": int(t),
            "n_ecrs": len(g),
            "share": len(g) / len(df),
            "share_title_only": (g["WO_RANGE"] == RANGE_OLD).mean(),
            "top_reason": rc.index[0],
            "top_driver": (lambda v: f"{v.index[0]} ({v.iloc[0] / len(g):.0%})" if len(v) else "")(
                g["DRIVER_N"].replace("", np.nan).dropna().value_counts()) if "DRIVER_N" in g else "",
            "top_subtype": (lambda v: f"{v.index[0]} ({v.iloc[0] / len(g):.0%})")(g["SUBTYPE"].fillna("(blank)").value_counts()),
            "top_reason_share": rc.iloc[0] / len(g),
            "other_share": g["IS_OTHER"].mean(),
            "n_reason_codes": int(rc.size),
            "ecrs_with_vpps": int(asm.sum()),
            "top_assembly": asm.index[0] if len(asm) else None,
            "top_assembly_share": asm.iloc[0] / asm.sum() if len(asm) else np.nan,
            "assembly_entropy": ent,
            "axis_proposal": _axis(ent, int(asm.sum()), cfg),
            "cost_given_musd": g["cost_given"].sum() / 1e6,
            "cost_recalc_musd": g["cost_recalc"].sum() / 1e6,
        })
    prof = pd.DataFrame(rows).sort_values("n_ecrs", ascending=False)
    if topics is not None:
        prof = topics[["topic_id", "label", "top_terms"]].merge(prof, on="topic_id", how="right")
    return prof


def evaluate(df: pd.DataFrame, assign: pd.DataFrame, cfg: dict, name: str,
             mask: pd.Series | None = None) -> dict:
    """Headline metrics for one approach on the ECRs selected by ``mask`` (default: all)."""
    a = df[["WO_NO", "REASON", "WO_RANGE", "IS_OTHER", cfg["axis"]["level"]]].merge(
        assign, on="WO_NO", how="left")
    a["primary_topic"] = a["primary_topic"].fillna(-1).astype(int)
    if mask is not None:
        a = a[mask.values]
    assigned = a["primary_topic"] != -1
    n_topics = int(a.loc[assigned, "primary_topic"].nunique())
    sizes = a.loc[assigned, "primary_topic"].value_counts()
    n_labels = a["topics"].fillna("").map(lambda s: len([x for x in str(s).split(";") if x != ""]))
    res = {
        "approach": name,
        "ecrs_scored": len(a),
        "n_topics": n_topics,
        "coverage": assigned.mean(),
        "coverage_title_only": assigned[a["WO_RANGE"] == RANGE_OLD].mean(),
        "coverage_full_text": assigned[a["WO_RANGE"] == RANGE_NEW].mean(),
        "largest_topic_share": (sizes.iloc[0] / len(a)) if len(sizes) else np.nan,
        "mean_labels_per_ecr": n_labels[assigned].mean() if assigned.any() else np.nan,
        "nmi_vs_reason_code": normalized_mutual_info_score(a.loc[assigned, "REASON"], a.loc[assigned, "primary_topic"]) if assigned.any() else np.nan,
        "other_ecrs_topics_to_80pct": _topics_to_share(a.loc[assigned & a["IS_OTHER"], "primary_topic"], 0.8),
    }
    return res


def _topics_to_share(s: pd.Series, target: float) -> int:
    if s.empty:
        return 0
    c = s.value_counts(normalize=True).cumsum()
    return int((c < target).sum() + 1)


def agreement(a1: pd.DataFrame, a2: pd.DataFrame, mask_wo: set | None = None) -> dict:
    """ARI / NMI between two approaches' primary topics on ECRs both assigned."""
    m = a1[["WO_NO", "primary_topic"]].merge(a2[["WO_NO", "primary_topic"]], on="WO_NO", suffixes=("_1", "_2"))
    if mask_wo is not None:
        m = m[m["WO_NO"].isin(mask_wo)]
    m = m[(m["primary_topic_1"] != -1) & (m["primary_topic_2"] != -1)]
    if m.empty:
        return {"n": 0, "ari": np.nan, "nmi": np.nan}
    return {"n": len(m),
            "ari": adjusted_rand_score(m["primary_topic_1"], m["primary_topic_2"]),
            "nmi": normalized_mutual_info_score(m["primary_topic_1"], m["primary_topic_2"])}


def write_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=default))
