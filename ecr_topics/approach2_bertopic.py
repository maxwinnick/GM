"""Approach 2 — BERTopic (embeddings -> UMAP -> HDBSCAN -> class-based TF-IDF), per WO partition.

Data handling: the embedding model (all-MiniLM-L6-v2, pinned revision, SHA256-checked by `make models`) is
loaded from a local folder with Hugging Face access switched off, so ECR text never leaves
the machine.

Per partition:
1. Embed the partition's documents (its free-text fields, plus structured fields where the
   partition uses them).
2. Sweep HDBSCAN ``min_cluster_size`` x 3 UMAP seeds, fit on the discovery split, apply to the
   held-out split. Record themes, unassigned share and seed agreement (ARI).
3. Choose the size with the best seed agreement among those with unassigned share <=
   ``max_outlier_share`` and at least ``min_topics`` themes (a 2-theme split agrees with
   itself trivially); else the lowest unassigned share.
4. Save the chosen model's assignments, plus a variant with every unassigned ECR moved to its
   nearest theme (BERTopic's embedding strategy).

Run:  python -m ecr_topics.approach2_bertopic [--partition title_only|full_text]
"""

from __future__ import annotations

import argparse
import logging
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import adjusted_rand_score

from . import common as C
from . import features as F

log = logging.getLogger("approach2")


def embed(docs: list[str], cfg: dict, cache) -> np.ndarray:
    if cache.exists():
        E = np.load(cache)
        if E.shape[0] == len(docs):
            return E
    from sentence_transformers import SentenceTransformer

    path = cfg["paths"]["embedding_model"]
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(f"Embedding model not found at {path} (see models/README.md).")
    model = SentenceTransformer(str(path), device="cpu")
    t0 = time.time()
    E = model.encode(docs, batch_size=cfg["bertopic"]["batch_size"], normalize_embeddings=True,
                     show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    log.info("embedded %d docs in %.0fs", len(docs), time.time() - t0)
    np.save(cache, E)
    return E


def vectorizer(cfg: dict) -> CountVectorizer:
    domain = [ln.split("#")[0].strip() for ln in (C.ROOT / "resources" / "stopwords_domain.txt").read_text().splitlines()]
    domain = [w for w in domain if w]
    stop = set(ENGLISH_STOP_WORDS) | set(domain) | F.theme_stopwords(cfg) | {
        "title", "description", "reason", "comment", "driver", "type", "chg", "pwo", "awo", "two",
        "without"}   # field labels and SUBTYPE boilerplate added by build_documents
    # BERTopic fits this vectorizer on one concatenated document PER TOPIC (class-based TF-IDF), so
    # min_df counts topics, not ECRs; it must stay at 1 or topic words skew toward generic terms.
    return CountVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True, stop_words=sorted(stop),
                           token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")


def build_model(cfg: dict, seed: int, mcs: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    b = cfg["bertopic"]
    return BERTopic(embedding_model=None,
                    umap_model=UMAP(**b["umap"], random_state=seed, low_memory=True),
                    hdbscan_model=HDBSCAN(min_cluster_size=mcs, min_samples=max(5, mcs // 4), metric="euclidean",
                                          cluster_selection_method="eom", prediction_data=True),
                    vectorizer_model=vectorizer(cfg), calculate_probabilities=False, verbose=False)


def fit_apply(cfg, seed, mcs, docs, E, disc):
    tm = build_model(cfg, seed, mcs)
    t_disc, _ = tm.fit_transform([docs[i] for i in disc], E[disc])
    t_all, _ = tm.transform(docs, E)
    t_all = np.asarray(t_all)
    t_all[disc] = np.asarray(t_disc)
    return tm, t_all


def run_partition(cfg: dict, partition: str, df: pd.DataFrame | None = None) -> dict:
    df = df if df is not None else C.load_extract(cfg)
    out = C.out_dir(cfg, f"{partition}/approach2_bertopic")
    pmask = C.partition_mask(df, cfg, partition)
    dfp = df[pmask].reset_index(drop=True)
    held = C.make_split(df, cfg)[pmask].to_numpy()
    docs = C.build_documents(dfp, cfg, partition).tolist()
    E = embed(docs, cfg, out / "embeddings_minilm.npy")
    disc = np.flatnonzero(~held)

    runs, models = [], {}
    for mcs in cfg["bertopic"]["min_cluster_sizes"]:
        labs = {}
        for seed in cfg["bertopic"]["seeds"]:
            t0 = time.time()
            tm, lab = fit_apply(cfg, seed, mcs, docs, E, disc)
            labs[seed] = lab
            if seed == cfg["bertopic"]["seeds"][0]:
                models[mcs] = (tm, lab)
            runs.append({"min_cluster_size": mcs, "seed": seed, "n_topics": int(len(set(lab)) - (1 if -1 in lab else 0)),
                         "unassigned_share": float(np.mean(lab == -1)),
                         "unassigned_share_heldout": float(np.mean(lab[held] == -1)),
                         "seconds": round(time.time() - t0, 1)})
            log.info("[%s] mcs=%d seed=%d topics=%d unassigned=%.2f", partition, mcs, seed,
                     runs[-1]["n_topics"], runs[-1]["unassigned_share"])
        s0 = labs[cfg["bertopic"]["seeds"][0]]
        aris = []
        for s in cfg["bertopic"]["seeds"][1:]:
            both = (s0 != -1) & (labs[s] != -1)
            aris.append(adjusted_rand_score(s0[both], labs[s][both]))
        for r in runs:
            if r["min_cluster_size"] == mcs:
                r["seed_ari_both_assigned"] = float(np.mean(aris))
    runs = pd.DataFrame(runs)
    runs.to_csv(out / "cluster_size_sweep.csv", index=False)

    by = runs.groupby("min_cluster_size").agg(n_topics=("n_topics", "mean"), unassigned=("unassigned_share", "mean"),
                                              ari=("seed_ari_both_assigned", "first")).reset_index()
    min_t = cfg["bertopic"].get("min_topics", 10)
    ok = by[(by["unassigned"] <= cfg["bertopic"]["max_outlier_share"]) & (by["n_topics"] >= min_t)]
    if len(ok):
        mcs = int(ok.sort_values("ari", ascending=False).iloc[0]["min_cluster_size"])
        why = (f"best seed agreement among sizes with <= {cfg['bertopic']['max_outlier_share']:.0%} unassigned "
               f"and at least {min_t} themes")
    else:
        mcs = int(by.sort_values("unassigned").iloc[0]["min_cluster_size"])
        why = "no size kept unassigned under the limit; chose the lowest unassigned share"
    tm, lab = models[mcs]

    rows = []
    for t in sorted(set(lab) - {-1}):
        words = [w for w, _ in tm.get_topic(t)][:12]
        rows.append({"topic_id": int(t), "top_terms": ", ".join(words), "label": " / ".join(words[:3])})
    topics = pd.DataFrame(rows)
    assign = pd.DataFrame({"WO_NO": dfp["WO_NO"], "primary_topic": lab.astype(int),
                           "topics": [str(t) if t != -1 else "" for t in lab], "primary_weight": np.nan})
    assign.to_csv(out / "assignments.csv", index=False)
    prof = C.topic_profiles(dfp, assign["primary_topic"], cfg, topics)
    prof.to_csv(out / "topics.csv", index=False)
    red = (np.asarray(tm.reduce_outliers(docs, lab.tolist(), strategy="embeddings", embeddings=E))
           if (lab == -1).any() else lab.copy())         # BERTopic raises when there is nothing to reduce
    a_red = assign.assign(primary_topic=red, topics=[str(t) if t != -1 else "" for t in red])
    a_red.to_csv(out / "assignments_outliers_reduced.csv", index=False)

    summary = {"partition": partition, "ecrs": int(len(dfp)), "chosen_min_cluster_size": mcs, "choice_reason": why,
               "sweep": by.to_dict("records"), "n_topics": int(len(topics)),
               "axis_counts": prof["axis_proposal"].value_counts().to_dict(),
               "eval_all": C.evaluate(dfp, assign, cfg, "BERTopic"),
               "eval_heldout": C.evaluate(dfp, assign, cfg, "BERTopic", mask=pd.Series(held)),
               "eval_heldout_reduced": C.evaluate(dfp, a_red, cfg, "BERTopic (all assigned)", mask=pd.Series(held))}
    C.write_json(summary, out / "summary.json")
    log.info("[%s] chose min_cluster_size=%d (%s)", partition, mcs, why)
    return summary


def unavailable_reason(cfg: dict) -> str | None:
    """Why BERTopic cannot run here (missing packages or embedding model), or None if it can."""
    try:
        import bertopic  # noqa: F401
        import hdbscan  # noqa: F401
        import sentence_transformers  # noqa: F401
        import umap  # noqa: F401
    except ImportError as exc:
        return f"package not installed ({exc.name}); run `make setup`"
    if not (cfg["paths"]["embedding_model"] / "model.safetensors").exists():
        return f"embedding model not found at {cfg['paths']['embedding_model']}; run `make models`"
    return None


def run(cfg: dict | None = None, partitions: list[str] | None = None) -> None:
    cfg = cfg or C.load_config()
    why = unavailable_reason(cfg)
    if why:
        log.warning("Skipping BERTopic: %s", why)
        return
    df = C.load_extract(cfg)
    for p in partitions or list(cfg["partitions"]):
        run_partition(cfg, p, df)


if __name__ == "__main__":
    C.setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default=None)
    a = ap.parse_args()
    run(partitions=[a.partition] if a.partition else None)
