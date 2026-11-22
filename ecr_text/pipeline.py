"""End-to-end orchestration: load -> NLP -> AI -> save parsed data -> Pareto reports."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .ai import run_ai_parsing
from .config import (CARRY_COLUMNS, DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, DEFAULT_PROCESS_FLOW,
                     DEFAULT_SHEET, KEY, TEXT_COLUMNS, LLMConfig, ParetoConfig)
from .llm import llm_settings, read_process_flow
from .nlp import NLPParser
from .pareto import run_reports
from .report import write_outputs
from .scrub import residual_identifier_counts

log = logging.getLogger(__name__)

AI_MODES = ("off", "cached", "sync")


@dataclass
class PipelineConfig:
    input_path: Path = DEFAULT_INPUT
    sheet: str = DEFAULT_SHEET
    process_flow_path: Path = DEFAULT_PROCESS_FLOW
    output_dir: Path = DEFAULT_OUTPUT_DIR
    ai_mode: str = "cached"         # off | cached (use cache only, no API calls) | sync (call the LLM)
    ai_limit: int | None = None     # pilot: only send the first N ECRs
    ai_dry_run: bool = False        # estimate tokens, do not call the API
    scrub_person_names: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)


def load_masked(path: Path, sheet: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Masked workbook not found: {path.resolve()}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_excel(path, sheet_name=sheet)
    needed = set(CARRY_COLUMNS) | set(TEXT_COLUMNS.values())
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Workbook is missing expected columns: {sorted(missing)}")
    df = df[[c for c in CARRY_COLUMNS] + list(TEXT_COLUMNS.values())].copy()
    if not df[KEY].is_unique:
        raise ValueError(f"{KEY} is not unique; the pipeline expects one row per ECR.")
    for col in TEXT_COLUMNS.values():
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.isin(["", "nan", "None", "null", "N/A"]))
    log.info("Loaded %d ECRs x %d columns from %s", len(df), df.shape[1], path.name)
    return df


def add_nlp_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, NLPParser]:
    """Steps 1-2: lemmatize and remove stop words in every masked text field."""
    parser = NLPParser()
    out = df.copy()
    for field_name, col in TEXT_COLUMNS.items():
        t0 = time.time()
        out[f"{field_name}_NLP"] = parser.parse_many(out[col].tolist())
        log.info("  %s_NLP done in %.1fs", field_name, time.time() - t0)
    return out, parser


def write_step12_audits(df: pd.DataFrame, parser: NLPParser, phase_dir: Path, n_sample: int = 40) -> None:
    """Write the Step 1 (lemmatization) and Step 2 (stop words) audit files."""
    s1, s2 = phase_dir / "step1_lemmatize", phase_dir / "step2_stopwords"
    s1.mkdir(parents=True, exist_ok=True)
    s2.mkdir(parents=True, exist_ok=True)
    lem = parser.lemma_audit()
    lem.to_csv(s1 / "lemma_merges.csv", index=False)
    (lem.groupby("lemma")
        .agg(surface_forms=("surface_form", lambda x: ", ".join(x)), occurrences=("occurrences", "sum"))
        .sort_values("occurrences", ascending=False).reset_index()
        .to_csv(s1 / "lemma_merges_by_lemma.csv", index=False))
    cols = ["WO_NO"] + [c for f, src in TEXT_COLUMNS.items() for c in (src, f"{f}_NLP")]
    sample = df[df["MASKED_DESC_FOR_CHANGE"].notna()].sample(n_sample // 2, random_state=7)
    sample = pd.concat([sample, df[df["MASKED_DESC_FOR_CHANGE"].isna()].sample(n_sample // 2, random_state=7)])
    sample[cols].to_csv(s1 / "sample_before_after.csv", index=False, encoding="utf-8-sig")
    parser.removal_audit().to_csv(s2 / "stopword_removals.csv", index=False)
    parser.stopword_check().to_csv(s2 / "top_lemmas_vs_stoplist.csv", index=False)


def leak_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field_name, col in TEXT_COLUMNS.items():
        texts = df[col].dropna().tolist()
        for label, n in residual_identifier_counts(texts).items():
            rows.append({"field": field_name, "identifier_type": label, "occurrences": n})
    return pd.DataFrame(rows, columns=["field", "identifier_type", "occurrences"])


def ordered_columns(df: pd.DataFrame) -> list[str]:
    cols = list(CARRY_COLUMNS)
    for f, src in TEXT_COLUMNS.items():
        cols += [src, f"{f}_NLP"] + ([f"{f}_AI"] if f"{f}_AI" in df else [])
    return cols


def run(cfg: PipelineConfig) -> dict:
    if cfg.ai_mode not in AI_MODES:
        raise ValueError(f"ai_mode must be one of {AI_MODES}")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.llm.model:
        cfg.llm.model = llm_settings()["model"] or "unset"
    safe_model = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in cfg.llm.model)
    cfg.llm.cache_file = out_dir / "cache" / f"ai_terms_{cfg.llm.backend}_{safe_model}.jsonl"

    df = load_masked(cfg.input_path, cfg.sheet)
    leaks = leak_report(df)

    log.info("Stage 1/3: NLP parsing")
    df, parser = add_nlp_fields(df)
    audit = parser.removal_audit()
    write_step12_audits(df, parser, out_dir / "phase1")

    ai_summary = {"mode": cfg.ai_mode}
    if cfg.ai_mode != "off":
        log.info("Stage 2/3: AI parsing (mode=%s, backend=%s, model=%s)",
                 cfg.ai_mode, cfg.llm.backend, cfg.llm.model)
        df, ai_summary = run_ai_parsing(
            df, cfg.llm,
            process_flow_text=read_process_flow(cfg.process_flow_path),
            limit=cfg.ai_limit,
            dry_run=cfg.ai_dry_run or cfg.ai_mode == "cached",
            scrub_names=cfg.scrub_person_names,
        )
        ai_summary["mode"] = cfg.ai_mode

    parsed = df[ordered_columns(df)]
    parsed.to_parquet(out_dir / "ecr_text_parsed.parquet", index=False)
    parsed.to_csv(out_dir / "ecr_text_parsed.csv", index=False, encoding="utf-8-sig")

    log.info("Stage 3/3: Pareto reports")
    reports = run_reports(parsed, cfg.pareto)
    xlsx = write_outputs(reports, out_dir, cfg.pareto, stopword_audit=audit, leak_counts=leaks)

    summary = {"ecrs": len(parsed), "ai": ai_summary, "report": str(xlsx),
               "fields": reports["summary"].to_dict("records")}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
