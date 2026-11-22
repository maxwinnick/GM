"""Run with:  python -m pytest -q tests"""

import pandas as pd

from ecr_text import ai
from ecr_text.config import LLMConfig
from ecr_text.llm import Backend
from ecr_text.nlp import NLPParser, ngrams, tokens
from ecr_text.pareto import _log_odds_z
from ecr_text.scrub import residual_identifier_counts, scrub_text


def test_scrub_placeholders_and_leaks():
    t = "PROG 12345 VWO 7000001 [Program 5] PN [12345678] [AMT] jane.doe@example.com 555-010-0199"
    out = scrub_text(t)
    assert "12345" not in out and "7000001" not in out and "@" not in out and "0199" not in out
    assert out.count("<PROGRAM>") == 2 and "<PN>" in out and "VWO <ID>" in out
    leaks = residual_identifier_counts([t])
    assert leaks == {"<EMAIL>": 1, "<PHONE>": 1, "<PROGRAM>": 1, "<ID>": 1}


def test_unconfirmed_abbreviations_are_not_applied():
    assert NLPParser().abbreviations == {}  # nothing is expanded until GM confirms it
    (out,) = NLPParser().parse_many(["FRT BRKT ASM"])
    assert out == "frt brkt asm"


def test_nlp_parse_segments_abbrev_and_negation():
    p = NLPParser(abbreviations={"frt": ["front"], "brkt": ["bracket"], "asm": ["assembly"]})
    (out,) = p.parse_many(["FRT BRKT ASM w/ new clip MY30 | No piece cost impact."])
    assert out.split(" | ")[0] == "front bracket assembly clip"
    assert "no" in tokens(out) and "my30" not in out
    assert "assembly clip" in ngrams(out, 2)
    assert "clip no" not in ngrams(out, 2)  # bigrams never cross segments
    assert p.parse_many([None, "  "]) == [None, None]


def test_normalize_terms_drops_identifiers():
    raw = ["Wire Harness", "wire harness", "<PN>", "MY30 change", "lh bracket", "cost reduction", 3]
    assert ai.normalize_terms(raw, 6) == ["wire harness", "cost reduction"]


class DropsOneId(Backend):
    """Fake backend that forgets the last id on the first call."""

    calls = 0

    def extract(self, items):
        DropsOneId.calls += 1
        keep = items[:-1] if DropsOneId.calls == 1 else items
        return {it["id"]: [f"term {it['id']}" ] for it in keep}


def test_ai_retry_and_attach(tmp_path, monkeypatch):
    cfg = LLMConfig(backend="mock", items_per_request=10, max_workers=1,
                    cache_file=tmp_path / "cache.jsonl")
    monkeypatch.setattr(ai, "make_backend", lambda c, s: DropsOneId(c, s))
    df = pd.DataFrame({
        "WO_NO": [1, 2, 3],
        "MASKED_TITLE": ["Front bracket change", "Front bracket change", "Seat foam"],
        "MASKED_COST_COMMENT": ["No tooling", None, "Supplier quote"],
        "MASKED_DESC_FOR_CHANGE": [None, None, None],
        "MASKED_REASON_FOR_CHANGE": [None, None, None],
    })
    out, summary = ai.run_ai_parsing(df, cfg, scrub_names=False)
    assert summary["unique_items"] == 4  # duplicate title sent once
    assert summary["failed_chunks"] == [] and DropsOneId.calls == 2
    assert out["TITLE_AI"].notna().all() and out.loc[1, "COST_COMMENT_AI"] is None or pd.isna(out.loc[1, "COST_COMMENT_AI"])
    assert out.loc[0, "TITLE_AI"] == out.loc[1, "TITLE_AI"]
    # second run is fully cached
    _, s2 = ai.run_ai_parsing(df, cfg, scrub_names=False)
    assert s2["to_send"] == 0


def test_log_odds_sign():
    prior = pd.Series({"a": 10.0, "b": 10.0})
    z = _log_odds_z(pd.Series({"a": 50, "b": 5}), pd.Series({"a": 50, "b": 500}), prior)
    assert z["a"] > 1.96 > z["b"]


def test_trigrams_stay_within_segments():
    v = "front bracket assembly clip | no piece cost impact"
    assert ngrams(v, 3) == ["front bracket assembly", "bracket assembly clip",
                            "no piece cost", "piece cost impact"]
