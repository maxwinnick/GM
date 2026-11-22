"""Build the per-partition ECR samples for Approach 3 (TopicGPT / LLooM).

For each WO partition:
* generation sample: ``generation_sample_per_partition`` ECRs from the discovery split, read by
  the LLM to propose topics; stratified by coded "Other" x the partition's NMF theme
* evaluation sample: ``evaluation_sample_per_partition`` ECRs from the held-out split, labeled
  by the LLM with the finished codebook; same stratification. If a fixed list of evaluation
  WO numbers exists in ``approach3_llm_inputs/evaluation_wo_numbers.csv`` (column WO_NO), it is
  used instead, so results stay comparable across reruns.

Only the partition's text fields (masked by default) and its structured fields are written.

Run:  python -m ecr_topics.make_llm_sample   (after approach1_nmf)
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from . import common as C

log = logging.getLogger("llm_sample")
MAX_CHARS = 2000


def _stratified(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(pool) <= n:
        return pool
    strata = pool["IS_OTHER"].astype(str) + "|" + pool["nmf_topic"].astype(str)
    strata = strata.where(strata.map(strata.value_counts()) >= 4, pool["IS_OTHER"].astype(str))
    if strata.value_counts().min() < 2:
        strata = None
    picked, _ = train_test_split(pool, train_size=n, random_state=seed, stratify=strata)
    return picked


def run(cfg: dict | None = None) -> None:
    cfg = cfg or C.load_config()
    df = C.load_extract(cfg)
    held = C.make_split(df, cfg)
    fixed = cfg["paths"]["outputs"] / "approach3_llm_inputs" / "evaluation_wo_numbers.csv"
    fixed_eval = set(pd.read_csv(fixed, dtype={"WO_NO": str})["WO_NO"]) if fixed.exists() else None
    for part in cfg["partitions"]:
        out = C.out_dir(cfg, f"{part}/approach3_llm_inputs")
        pm = C.partition_mask(df, cfg, part)
        d = df[pm].copy()
        d["doc"] = C.build_documents(d, cfg, part)
        d["doc"] = d["doc"].where(d["doc"].str.len() <= MAX_CHARS, d["doc"].str[:MAX_CHARS] + " [...]")
        nmf = pd.read_csv(cfg["paths"]["outputs"] / part / "approach1_nmf" / "assignments.csv", dtype={"WO_NO": str})
        d["nmf_topic"] = d["WO_NO"].map(dict(zip(nmf["WO_NO"], nmf["primary_topic"]))).fillna(-1).astype(int)
        d["held"] = held[pm].values
        usable = d["doc"].str.len() > 0
        gen = _stratified(d[~d["held"] & usable], cfg["llm"]["generation_sample_per_partition"], cfg["seed"])
        if fixed_eval is not None:
            ev = d[d["WO_NO"].isin(fixed_eval)]
        else:
            ev = _stratified(d[d["held"] & usable], cfg["llm"].get("evaluation_sample_per_partition", 250), cfg["seed"])
        assert ev["held"].all(), "evaluation ECRs must be held out"
        cols = ["WO_NO", "doc"]
        gen[cols].to_csv(out / "llm_generation_sample.csv", index=False)
        ev[cols].to_csv(out / "llm_evaluation_sample.csv", index=False)
        log.info("[%s] generation %d, evaluation %d", part, len(gen), len(ev))


if __name__ == "__main__":
    C.setup_logging()
    run()
