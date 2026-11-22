"""Shared fixtures: a synthetic project (fake extract + config) so tests never need GM data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import synthetic_extract  # noqa: E402


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> Path:
    """A temp folder with a synthetic extract and a config.yaml pointing at it."""
    d = tmp_path_factory.mktemp("project")
    xlsx = d / "data" / "extract.xlsx"
    xlsx.parent.mkdir()
    synthetic_extract.make(rows=900, seed=1).to_excel(xlsx, sheet_name="result", index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["input"]["workbook"] = str(xlsx)
    cfg["outputs"] = str(d / "outputs")
    cfg["topics"]["nmf"]["k_grid"] = [5, 10]
    cfg["topics"]["bertopic"]["embedding_model"] = str(d / "no_model")   # BERTopic is skipped in tests
    cfg["topics"]["llm"]["generation_sample_per_partition"] = 40
    cfg["topics"]["llm"]["evaluation_sample_per_partition"] = 30
    (d / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return d
