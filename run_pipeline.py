"""Run the whole GM ECR analytics pipeline, from GM's extract to every report, with one command.

    python run_pipeline.py                         # every stage enabled in config.yaml
    python run_pipeline.py --only cost             # one stage (repeat --only for several)
    python run_pipeline.py --skip topics           # everything except one stage
    python run_pipeline.py --input data/new.xlsx   # a different extract
    python run_pipeline.py --outputs runs/2026-12   # a different output folder
    python run_pipeline.py --states Closed         # Phase 1 text Paretos on Closed ECRs only
    python run_pipeline.py --with-llm              # also run the TopicGPT / LLooM stage (sends text)

Stages, in order (outputs under outputs/):
    text         01_text/         Phase 1 text model: lemmatize, stop words, counts, vocabulary,
                                  term-frequency matrix, figures, write-up, per-field Pareto workbook
    cost         02_cost_pareto/  ITIS Pareto reports by ECR count and COST_USD (as given and
                                  recalculated), charts, Excel workbook, ECR lookup behind each chart
    terms        03_term_frequency/  vocabulary x cost / program / assembly table (top 500 + full)
    diagnostics  04_diagnostics/  reason-code diagnostics and the k = 20 NMF pilot
    topics       05_topics/       Steps 7-8: TF-IDF + NMF, BERTopic, AI samples, comparison, report
    llm_topics   05_topics/<partition>/approach3_llm/  TopicGPT + LLooM via the LLM in .env (opt-in)

A run manifest (inputs, hashes, row counts, timings, package versions) is written to
outputs/run_manifest.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STAGES = ("text", "cost", "terms", "diagnostics", "topics", "llm_topics")
NEEDS = {  # stage -> (stage it reads from, file that must exist)
    "terms": [("text", "01_text/phase1/step6_term_frequency/tf_long.csv.gz"),
              ("cost", "02_cost_pareto/tables/ecr_data.csv")],
    "diagnostics": [("text", "01_text/phase1/step6_term_frequency/tf_matrix.npz")],
    "topics": [("text", "01_text/ecr_text_parsed.csv")],
    "llm_topics": [("topics", "05_topics/title_only/approach3_llm_inputs/llm_generation_sample.csv")],
}
log = logging.getLogger("pipeline")


# ------------------------------------------------------------------------------------ helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("pandas", "numpy", "scipy", "scikit-learn", "spacy", "matplotlib", "openpyxl", "bertopic",
                "sentence-transformers", "umap-learn", "hdbscan", "openai"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def resolve(base: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


# ------------------------------------------------------------------------------------ stages
def stage_text(ctx: dict) -> dict:
    from ecr_text.config import LLMConfig, ParetoConfig
    from ecr_text.figures import make_phase1_figures
    from ecr_text.phase1 import Phase1Config, run_phase1
    from ecr_text.phase1_report import write_report
    from ecr_text.pipeline import PipelineConfig, run

    import pandas as pd

    t = ctx["cfg"]["text"]
    states_arg = str(ctx["states"] or t.get("states", "all"))
    states = None if states_arg.lower() == "all" else tuple(s.strip() for s in states_arg.split(","))
    out = ctx["out"] / "01_text"
    summary = run(PipelineConfig(
        input_path=ctx["input"], sheet=ctx["sheet"], process_flow_path=ctx["process_flow"], output_dir=out,
        ai_mode=ctx["ai"] or t.get("ai", "cached"), ai_limit=t.get("ai_limit"),
        llm=LLMConfig(max_workers=ctx["cfg"]["llm"].get("max_workers", 4)),
        pareto=ParetoConfig(states=states)))
    parsed = pd.read_parquet(out / "ecr_text_parsed.parquet")
    phase_dir = out / "phase1"
    p1 = run_phase1(parsed, phase_dir, Phase1Config(min_df=int(t.get("min_df", 5)), pareto=ParetoConfig(states=states)))
    facts = make_phase1_figures(p1, phase_dir / "figures")
    (phase_dir / "figure_facts.json").write_text(json.dumps(facts, indent=1, default=str))
    md = write_report(p1, facts, phase_dir, {"timestamp": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "states": states_arg})
    return {"ecrs": int(p1["n_ecr"]), "vocabulary_terms": int(p1["vocab_summary"]["vocabulary_size"]),
            "figures": len(list((phase_dir / "figures").rglob("*.png"))), "write_up": str(md.relative_to(ctx["out"])),
            "ai": summary["ai"]}


def stage_cost(ctx: dict) -> dict:
    from cost_pareto import charts, ecrs_behind_slides, pareto_cost, workbook

    out = ctx["out"] / "02_cost_pareto"
    pareto_cost.main(ctx["input"], out, ctx["sheet"], verbose=False)
    charts.main(out)
    workbook.main(out, out / "ITIS_Pareto_COST_USD.xlsx")
    ecrs_behind_slides.main(out, out / "ECRs_behind_slides.xlsx")
    s = json.loads((out / "summary.json").read_text())
    return {"charts": len(list((out / "charts").rglob("*.png"))), "summary": "02_cost_pareto/summary.json",
            **{k: s[k] for k in ("ecrs",) if k in s}}


def stage_terms(ctx: dict) -> dict:
    from cost_pareto import term_frequency

    out = ctx["out"] / "03_term_frequency"
    term_frequency.main(ctx["out"] / "01_text" / "phase1", ctx["out"] / "02_cost_pareto" / "tables" / "ecr_data.csv", out)
    return {"files": sorted(p.name for p in out.glob("*.csv"))}


def stage_diagnostics(ctx: dict) -> dict:
    from diagnostics import next_steps_diagnostics as D

    res = D.run(ctx["input"], ctx["out"] / "01_text" / "phase1", ctx["out"] / "04_diagnostics", ctx["sheet"])
    return {k: res[k] for k in ("ecrs", "other_share", "split_half_components_cos_ge_0_8") if k in res}


def _topics_cfg(ctx: dict) -> dict:
    from ecr_topics import common as C

    cfg = C.load_config(ctx["config_path"], outputs=ctx["out"])
    cfg["paths"]["extract_xlsx"] = ctx["input"]
    cfg["sheet"] = ctx["sheet"]
    return cfg


def _compare_and_report(cfg: dict) -> None:
    from ecr_topics import compare, report_md

    compare.run(cfg)
    report_md.run(cfg)


def stage_topics(ctx: dict) -> dict:
    from ecr_topics import approach1_nmf, approach2_bertopic, make_llm_sample

    cfg = _topics_cfg(ctx)
    approach1_nmf.run(cfg)
    approach2_bertopic.run(cfg)          # skipped with a warning if BERTopic or the model is missing
    make_llm_sample.run(cfg)
    if "llm_topics" not in ctx["selected"]:  # otherwise compare/report run after the LLM stage
        _compare_and_report(cfg)
    return {"bertopic": approach2_bertopic.unavailable_reason(cfg) or "ran",
            "report": "05_topics/Steps7-8_Analysis.md"}


def stage_llm_topics(ctx: dict) -> dict:
    from ecr_topics import llm_topics as T
    from ecr_topics.llm_caller import default_model

    cfg = _topics_cfg(ctx)
    model = ctx["model"] or default_model()
    if not model:
        raise SystemExit("Set LLM_MODEL in .env (see .env.example) or pass --model to run the llm_topics stage.")
    workers = ctx["cfg"]["llm"].get("max_workers", 4)
    est = T.run_all(model, dry_run=True, max_workers=workers, topics_out=cfg["paths"]["outputs"])
    log.info("LLM estimate: %s", est["report"])
    if ctx["cfg"]["llm"].get("dry_run", False):
        return {"dry_run": True, **est}
    info = T.run_all(model, dry_run=False, max_workers=workers, topics_out=cfg["paths"]["outputs"])
    _compare_and_report(cfg)
    return {"model": model, **info}


RUNNERS = {"text": stage_text, "cost": stage_cost, "terms": stage_terms, "diagnostics": stage_diagnostics,
           "topics": stage_topics, "llm_topics": stage_llm_topics}


# ------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--input", type=Path, help="extract workbook (default: input.workbook in config.yaml)")
    ap.add_argument("--outputs", type=Path, help="output folder (default: outputs in config.yaml)")
    ap.add_argument("--only", action="append", choices=STAGES, help="run only this stage (repeatable)")
    ap.add_argument("--skip", action="append", choices=STAGES, default=[], help="skip this stage (repeatable)")
    ap.add_argument("--with-llm", action="store_true", help="also run the llm_topics stage")
    ap.add_argument("--states", help='Phase 1 text Paretos: "all" or a comma-separated STATE list')
    ap.add_argument("--ai", choices=("off", "cached", "sync"), help="Phase 1 LLM key terms (default from config)")
    ap.add_argument("--model", help="LLM model id (default: LLM_MODEL in .env)")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "openai", "matplotlib", "numba", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config_path = a.config.resolve()
    cfg = yaml.safe_load(config_path.read_text())
    base = config_path.parent
    sys.path.insert(0, str(ROOT))
    from ecr_topics.common import resolve_input

    inp = a.input.resolve() if a.input else resolve_input(base, cfg["input"]["workbook"])
    if not inp.exists():
        raise SystemExit(f"Input workbook not found: {inp}")
    if a.outputs:
        cfg["outputs"] = str(a.outputs.resolve())
    out = resolve(base, cfg.get("outputs", "outputs"))
    out.mkdir(parents=True, exist_ok=True)

    enabled = {s for s, on in cfg.get("stages", {}).items() if on}
    if a.with_llm:
        enabled.add("llm_topics")
    selected = [s for s in STAGES if (s in a.only if a.only else s in enabled) and s not in a.skip]
    ctx = {"cfg": cfg, "config_path": config_path, "input": inp, "sheet": cfg["input"].get("sheet", "result"),
           "process_flow": resolve(base, cfg["input"].get("process_flow") or "data/Overall Process Flow.docx"),
           "out": out, "states": a.states, "ai": a.ai, "model": a.model, "selected": selected}

    log.info("Input: %s", inp)
    log.info("Stages: %s", ", ".join(selected) or "(none)")
    manifest = {"started": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
                "input": {"path": inp.name, "sha256": sha256(inp), "sheet": ctx["sheet"]},
                "config": cfg, "stages": {}, "versions": versions()}
    t_all = time.time()
    for s in selected:
        for need, rel in NEEDS.get(s, []):
            if not (out / rel).exists():
                raise SystemExit(f"Stage '{s}' needs the '{need}' stage's outputs ({rel}). Run it first, "
                                 f"e.g. python run_pipeline.py --only {need}")
        log.info("=== stage %s ===", s)
        t0 = time.time()
        result = RUNNERS[s](ctx)
        manifest["stages"][s] = {"seconds": round(time.time() - t0, 1), **(result or {})}
        log.info("=== stage %s done in %.0fs ===", s, time.time() - t0)
    manifest["seconds"] = round(time.time() - t_all, 1)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info("All done in %.0fs. Outputs: %s", time.time() - t_all, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
