"""Every stage of run_pipeline.py on a synthetic extract, with the LLM stage on a fake endpoint."""
import json

import pandas as pd

import run_pipeline
from ecr_topics import llm_topics as T
from tests.fixtures import fake_llm_caller as fake


def test_one_command_runs_every_stage(project, monkeypatch):
    monkeypatch.setattr(T, "call_llm", fake.call_llm)
    rc = run_pipeline.main(["--config", str(project / "config.yaml"), "--with-llm", "--model", "fake-model"])
    assert rc == 0
    out = project / "outputs"
    m = json.loads((out / "run_manifest.json").read_text())
    assert list(m["stages"]) == ["text", "cost", "terms", "diagnostics", "topics", "llm_topics"]
    assert m["stages"]["text"]["ecrs"] == 900
    # one key file per stage
    for rel in ("01_text/phase1/Phase1_Steps1-8_Analysis.md", "01_text/ecr_text_pareto.xlsx",
                "02_cost_pareto/ITIS_Pareto_COST_USD.xlsx", "02_cost_pareto/ECRs_behind_slides.xlsx",
                "03_term_frequency/term_frequency_full_vocabulary.csv", "04_diagnostics/summary.json",
                "05_topics/ecr_labels_all.csv", "05_topics/full_text/approach3_llm/topicgpt_codebook.json"):
        assert (out / rel).exists(), rel
    assert len(list((out / "02_cost_pareto" / "charts").rglob("*.png"))) > 50
    labels = pd.read_csv(out / "05_topics" / "ecr_labels_all.csv", dtype={"WO_NO": str})
    assert labels["WO_NO"].is_unique and len(labels) == 900
    report = (out / "05_topics" / "Steps7-8_Analysis.md").read_text()
    assert "TopicGPT" in report and "has not been run" not in report


def test_stage_dependencies_are_checked(tmp_path):
    import pytest
    import yaml
    from tests.conftest import ROOT, synthetic_extract

    x = tmp_path / "x.xlsx"
    synthetic_extract.make(rows=60).to_excel(x, sheet_name="result", index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["input"]["workbook"], cfg["outputs"] = str(x), str(tmp_path / "out")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="needs the 'text' stage"):
        run_pipeline.main(["--config", str(tmp_path / "c.yaml"), "--only", "diagnostics"])
