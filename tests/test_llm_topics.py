"""TopicGPT / LLooM module end to end against a fake LLM endpoint (no network).

The fake cuts off any reply over ~3,100 characters (like a default output limit), so this also
checks that oversized batches are split and that renamed topics are used in assignments.
"""
import pandas as pd

from ecr_topics import llm_topics as T
from tests.fixtures import fake_llm_caller as fake


def _samples(root, n=30):
    for part in ("title_only", "full_text"):
        d = root / part / "approach3_llm_inputs"
        d.mkdir(parents=True)
        docs = [f"Title: replace seat foam cushion {i} for cost reduction || Reason: supplier cost save on part {i}"
                for i in range(n)]
        df = pd.DataFrame({"WO_NO": [str(7_000_000 + i) for i in range(n)], "doc": docs})
        df.to_csv(d / "llm_generation_sample.csv", index=False)
        df.to_csv(d / "llm_evaluation_sample.csv", index=False)


def test_llm_pipeline_with_fake_caller(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "call_llm", fake.call_llm)
    monkeypatch.setattr(fake, "LIMIT", 800)              # force batch-of-10 replies over the limit
    monkeypatch.setitem(fake.calls, "too_long", 0)
    _samples(tmp_path)
    est = T.run_all("m", dry_run=True, topics_out=tmp_path)
    assert est["calls"] > 0
    T.run_all("m", dry_run=False, max_workers=2, topics_out=tmp_path)
    for part in ("title_only", "full_text"):
        tg = pd.read_csv(tmp_path / part / "approach3_llm" / "topicgpt_assignments.csv")
        assert len(tg) == 30 and tg["returned"].all()
        assert set(tg["why_topics"]) == {"Cost X"}          # renamed name is the one assigned
    assert fake.calls["too_long"] > 0                     # oversized replies were split, not lost
    again = T.run_all("m", dry_run=False, topics_out=tmp_path)
    assert again["calls"] == 0                            # everything served from cache
