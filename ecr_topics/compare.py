"""Score the three approaches side by side, separately for each WO partition.

Per partition:
* held-out set: TF-IDF + NMF, BERTopic (raw and all-assigned)
* AI evaluation sample (held-out ECRs): all of the above plus TopicGPT
  ("why" and "what") and LLooM, when outputs/<partition>/approach3_llm/ exists
* agreement between every pair of approaches (and the existing reason code)
* where the AI's evidence quotes came from (which field)

Run:  python -m ecr_topics.compare
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import common as C

log = logging.getLogger("compare")


def _read_assign(path) -> pd.DataFrame:
    a = pd.read_csv(path, dtype={"WO_NO": str, "topics": str})
    a["topics"] = a["topics"].fillna("")
    return a


def llm_to_assign(names_col: pd.Series, codebook_names: list[str], wo: pd.Series) -> pd.DataFrame:
    idx = {n: i for i, n in enumerate(codebook_names)}
    ids = names_col.fillna("").map(lambda s: [idx[x] for x in str(s).split(";") if x in idx])
    return pd.DataFrame({"WO_NO": wo.values, "primary_topic": ids.map(lambda l: l[0] if l else -1).values,
                         "topics": ids.map(lambda l: ";".join(map(str, l))).values, "primary_weight": np.nan})


def load_llm(d) -> dict:
    res = {}
    if (d / "topicgpt_assignments.csv").exists():
        tg = pd.read_csv(d / "topicgpt_assignments.csv", dtype={"WO_NO": str})
        book = json.loads((d / "topicgpt_codebook.json").read_text())
        res["TopicGPT (why)"] = (llm_to_assign(tg["why_topics"], [t["name"] for t in book["why"]], tg["WO_NO"]), book["why"])
        res["TopicGPT (what)"] = (llm_to_assign(tg["what_topic"], [t["name"] for t in book["what"]], tg["WO_NO"]), book["what"])
        res["_tg"] = tg
    if (d / "lloom_scores.csv").exists():
        lm = pd.read_csv(d / "lloom_scores.csv", dtype={"WO_NO": str})
        concepts = json.loads((d / "lloom_concepts.json").read_text())["final"]
        res["LLooM (why)"] = (llm_to_assign(lm["concepts"], [c["name"] for c in concepts], lm["WO_NO"]), concepts)
        res["_lm"] = lm
    return res


def quote_fields(doc: str, quotes: str) -> list[str]:
    """Which labeled field (Title, Cost comment, Description, Reason, Driver, Change type) each quote is in."""
    segs = []
    for part in str(doc).split(" || "):
        if ": " in part:
            lab, txt = part.split(": ", 1)
            segs.append((lab, " ".join(txt.lower().split())))
    out = []
    for q in [x for x in str(quotes).split(" || ") if x.strip()]:
        qn = " ".join(q.lower().split())
        hit = [lab for lab, txt in segs if qn in txt]
        out.append(hit[0] if hit else "spans fields")
    return out


def run_partition(cfg: dict, part: str, df: pd.DataFrame) -> dict:
    o = cfg["paths"]["outputs"] / part
    out = C.out_dir(cfg, f"{part}/comparison")
    pm = C.partition_mask(df, cfg, part)
    dfp = df[pm].reset_index(drop=True)
    held = C.make_split(df, cfg)[pm].reset_index(drop=True)
    ev_docs = pd.read_csv(o / "approach3_llm_inputs" / "llm_evaluation_sample.csv", dtype={"WO_NO": str})
    ev_wo = set(ev_docs["WO_NO"])
    ev_mask = dfp["WO_NO"].isin(ev_wo)

    A = {"TF-IDF + NMF": _read_assign(o / "approach1_nmf" / "assignments.csv")}
    if (o / "approach2_bertopic" / "assignments.csv").exists():     # BERTopic is optional
        A["BERTopic"] = _read_assign(o / "approach2_bertopic" / "assignments.csv")
        A["BERTopic (all assigned)"] = _read_assign(o / "approach2_bertopic" / "assignments_outliers_reduced.csv")
    llm = load_llm(o / "approach3_llm")

    held_rows = [C.evaluate(dfp, a, cfg, k, mask=held) for k, a in A.items()]
    ev_rows = [C.evaluate(dfp, a, cfg, k, mask=ev_mask) for k, a in A.items()]
    for k in ("TopicGPT (why)", "TopicGPT (what)", "LLooM (why)"):
        if k in llm:
            ev_rows.append(C.evaluate(dfp, llm[k][0], cfg, k, mask=ev_mask))
    pd.DataFrame(held_rows).to_csv(out / "metrics_heldout.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(out / "metrics_eval_sample.csv", index=False)

    pool = dict(A)
    pool.update({k: v[0] for k, v in llm.items() if not k.startswith("_")})
    pool["Existing reason code"] = dfp[["WO_NO"]].assign(primary_topic=pd.factorize(dfp["REASON"])[0])
    keys = list(pool)
    agr = pd.DataFrame([{"a": a, "b": b, **C.agreement(pool[a], pool[b], ev_wo)}
                        for i, a in enumerate(keys) for b in keys[i + 1:]])
    agr.to_csv(out / "agreement_eval_sample.csv", index=False)

    evidence = {}
    if "_tg" in llm or "_lm" in llm:
        docs = dict(zip(ev_docs["WO_NO"], ev_docs["doc"]))
        for k, col in (("_tg", "why_quotes"), ("_lm", "quotes")):
            if k in llm:
                f = [x for w, q in zip(llm[k]["WO_NO"], llm[k][col].fillna("")) for x in quote_fields(docs.get(w, ""), q)]
                evidence["TopicGPT (why)" if k == "_tg" else "LLooM (why)"] = pd.Series(f).value_counts().to_dict()
        if "_tg" in llm:
            f = [x for w, q in zip(llm["_tg"]["WO_NO"], llm["_tg"]["what_quote"].fillna("")) for x in quote_fields(docs.get(w, ""), q)]
            evidence["TopicGPT (what)"] = pd.Series(f).value_counts().to_dict()

    summary = {"partition": part, "label": cfg["partitions"][part]["label"], "ecrs": int(len(dfp)),
               "heldout_ecrs": int(held.sum()), "eval_sample_ecrs": int(ev_mask.sum()),
               "heldout": held_rows, "eval_sample": ev_rows, "agreement": agr.to_dict("records"),
               "llm_available": [k for k in llm if not k.startswith("_")], "evidence_fields": evidence}
    for k, key in (("_tg", "topicgpt_validity"), ("_lm", "lloom_validity")):
        if k in llm:
            summary[key] = {"records_returned": float(llm[k]["returned"].mean()),
                            "labels_dropped_invalid": int(llm[k]["invalid_dropped"].sum())}
    C.write_json(summary, out / "comparison_summary.json")
    log.info("[%s] eval sample:\n%s", part, pd.DataFrame(ev_rows)[["approach", "n_topics", "coverage", "largest_topic_share",
                                                                    "mean_labels_per_ecr", "nmi_vs_reason_code"]].round(3).to_string())
    return summary


def run(cfg: dict | None = None) -> None:
    cfg = cfg or C.load_config()
    df = C.load_extract(cfg)
    for p in cfg["partitions"]:
        run_partition(cfg, p, df)


if __name__ == "__main__":
    C.setup_logging()
    run()
