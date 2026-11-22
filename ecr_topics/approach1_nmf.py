"""Approach 1 — traditional NLP: TF-IDF + NMF themes, fit separately for each WO partition.

Per partition:
1. Term matrix from Phase 1 cleaned text (features.build); TF-IDF (sublinear tf, L2 norm),
   IDF learned on the partition's discovery rows only.
2. k sweep: split-half stability averaged over several random half-splits (Hungarian match on
   component cosine), largest-topic share, NPMI coherence (reported, not used to choose).
3. Final model at the chosen k applied to every ECR in the partition; one catch-all split if a
   topic holds more than ``max_topic_share``.
4. Ablation: the same sweep without the Step 7 theme stop words and (title-only partition)
   without the structured tokens, so the effect of each change is measured.

Run:  python -m ecr_topics.approach1_nmf [--partition title_only|full_text]
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from . import common as C
from . import features as F

log = logging.getLogger("approach1")


def tfidf(Xc: sp.csr_matrix, fit_rows: np.ndarray) -> sp.csr_matrix:
    tf = TfidfTransformer(sublinear_tf=True, norm="l2").fit(Xc[fit_rows])
    return tf.transform(Xc).tocsr()


def fit_nmf(X: sp.csr_matrix, k: int, seed: int) -> NMF:
    return NMF(n_components=k, init="nndsvda", max_iter=800, tol=1e-4, random_state=seed).fit(X)


def match_components(H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
    S = normalize(H1) @ normalize(H2).T
    r, c = linear_sum_assignment(-S)
    return S[r, c]


def npmi_topics(B: sp.csr_matrix, H: np.ndarray, top_n: int = 10) -> np.ndarray:
    N = B.shape[0]
    out = []
    for h in H:
        top = np.argsort(h)[::-1][:top_n]
        co = (B[:, top].T @ B[:, top]).toarray().astype(float)
        dfv = np.diag(co)
        vals = []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                pij = co[i, j] / N
                vals.append(-1.0 if pij == 0 else np.log(pij / ((dfv[i] / N) * (dfv[j] / N))) / -np.log(pij))
        out.append(np.mean(vals))
    return np.array(out)


def assign_from_W(W: np.ndarray, min_share: float):
    tot = W.sum(1)
    share = np.divide(W, tot[:, None], out=np.zeros_like(W), where=tot[:, None] > 0)
    primary = np.where(tot > 0, share.argmax(1), -1)
    weight = np.where(tot > 0, share.max(1), 0.0)
    multi = [";".join(str(j) for j in np.flatnonzero(s >= min_share)) if t > 0 else "" for s, t in zip(share, tot)]
    return primary, multi, weight


def top_terms(H, terms, n=12):
    return [", ".join(terms[np.argsort(h)[::-1][:n]]) for h in H]


def sweep(X, B, rows, cfg) -> pd.DataFrame:
    nm = cfg["nmf"]
    recs = []
    for k in nm["k_grid"]:
        t0 = time.time()
        full = fit_nmf(X[rows], k, cfg["seed"])
        stab, med = [], []
        for rep in range(nm["stability_repeats"]):
            perm = np.random.default_rng(cfg["seed"] + rep).permutation(rows)
            ha, hb = perm[: len(perm) // 2], perm[len(perm) // 2:]
            cos = match_components(fit_nmf(X[ha], k, cfg["seed"]).components_, fit_nmf(X[hb], k, cfg["seed"]).components_)
            stab.append((cos >= nm["stability_match"]).mean())
            med.append(np.median(cos))
        primary, multi, _ = assign_from_W(full.transform(X[rows]), nm["multilabel_min_share"])
        sizes = pd.Series(primary[primary >= 0]).value_counts(normalize=True)
        recs.append({"k": k, "stable_share": float(np.mean(stab)), "stable_share_min": float(np.min(stab)),
                     "median_match_cosine": float(np.mean(med)),
                     "median_npmi": float(np.median(npmi_topics(B[rows], full.components_))),
                     "largest_topic_share": float(sizes.iloc[0]),
                     "topics_above_max_share": int((sizes > nm["max_topic_share"]).sum()),
                     "mean_labels_per_ecr": float(np.mean([len(m.split(";")) for m in multi if m])),
                     "converged": bool(full.n_iter_ < full.max_iter), "seconds": round(time.time() - t0, 1)})
        log.info("k=%d stable=%.2f largest=%.3f", k, recs[-1]["stable_share"], recs[-1]["largest_topic_share"])
    return pd.DataFrame(recs)


def choose_k(sw: pd.DataFrame, cfg: dict) -> tuple[int, str]:
    """Smallest stable k with no catch-all; else the most stable k without a catch-all.

    NPMI coherence is reported but not used (it does not track human judgment reliably;
    Hoyle et al., NeurIPS 2021).
    """
    nm = cfg["nmf"]
    sw = sw.sort_values("k")
    ok = sw[(sw["stable_share"] >= nm["stability_target"]) & (sw["topics_above_max_share"] == 0)]
    if len(ok):
        return int(ok.iloc[0]["k"]), "smallest k meeting the stability target with no catch-all topic"
    pool = sw[sw["topics_above_max_share"] == 0]
    pool = pool if len(pool) else sw
    best = pool.sort_values(["stable_share", "median_match_cosine"], ascending=False).iloc[0]
    return int(best["k"]), "no k met the stability target; chose the most stable k" + (
        " without a catch-all" if len(pool) < len(sw) or (sw["topics_above_max_share"] == 0).all() else "")


def run_partition(cfg: dict, partition: str, df: pd.DataFrame | None = None) -> dict:
    df = df if df is not None else C.load_extract(cfg)
    out = C.out_dir(cfg, f"{partition}/approach1_nmf")
    held_all = C.make_split(df, cfg)
    pmask = C.partition_mask(df, cfg, partition)
    dfp = df[pmask].reset_index(drop=True)
    held = held_all[pmask].to_numpy()
    nm = cfg["nmf"]

    # ---- ablation: which Step 7 changes help stability?
    variants = [("Phase 1 vocabulary only", False, False), ("+ theme stop words", True, False)]
    if cfg["partitions"][partition]["structured_fields"]:
        variants += [("+ structured fields only", False, True), ("+ theme stop words + structured fields", True, True)]
    abl = []
    for name, stop, struct in variants:
        tm = F.build(df, cfg, partition, use_stop=stop, use_structured=struct)
        nz = np.asarray(tm.X.sum(1)).ravel() > 0
        disc = np.flatnonzero(~held & nz)
        X = tfidf(tm.X, disc)
        sw = sweep(X, (tm.X > 0).astype(np.float32).tocsr(), disc, cfg).assign(variant=name)
        sw.to_csv(out / f"sweep_{name.replace(' ', '_').replace('+', 'plus')}.csv", index=False)
        k, why = choose_k(sw, cfg)
        r = sw[sw["k"] == k].iloc[0]
        abl.append({"variant": name, "terms": len(tm.terms), "chosen_k": k, "stable_share": r["stable_share"],
                    "largest_topic_share": r["largest_topic_share"], "median_npmi": r["median_npmi"]})
        log.info("[%s] %s: k=%d stable=%.2f", partition, name, k, r["stable_share"])
    pd.DataFrame(abl).to_csv(out / "ablation.csv", index=False)

    # ---- main model = config defaults
    tm = F.build(df, cfg, partition)
    assert (tm.wo.values == dfp["WO_NO"].values).all()
    nz = np.asarray(tm.X.sum(1)).ravel() > 0
    disc = np.flatnonzero(~held & nz)
    X = tfidf(tm.X, disc)
    sw = sweep(X, (tm.X > 0).astype(np.float32).tocsr(), disc, cfg)
    sw.to_csv(out / "theme_sweep.csv", index=False)
    k, why_k = choose_k(sw, cfg)
    model = fit_nmf(X[disc], k, cfg["seed"])
    W = model.transform(X)
    W[~nz] = 0
    primary, multi, weight = assign_from_W(W, nm["multilabel_min_share"])
    labels = top_terms(model.components_, tm.terms)

    splits, next_id = [], k
    sizes = pd.Series(primary[primary >= 0]).value_counts(normalize=True)
    for t in sizes[sizes > nm["max_topic_share"]].index:
        rows_t = np.flatnonzero(primary == t)
        best = None
        for ks in range(3, 7):
            perm = np.random.default_rng(cfg["seed"]).permutation(rows_t)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
            st = float((match_components(fit_nmf(X[a], ks, cfg["seed"]).components_,
                                         fit_nmf(X[b], ks, cfg["seed"]).components_) >= nm["stability_match"]).mean())
            if best is None or st >= best[1]:
                best = (ks, st)
        ks, st = best
        sm = fit_nmf(X[rows_t], ks, cfg["seed"])
        sub = sm.transform(X[rows_t]).argmax(1)
        for j in range(ks):
            for r in rows_t[sub == j]:
                primary[r] = next_id + j
                multi[r] = ";".join([m for m in multi[r].split(";") if m and m != str(t)] + [str(next_id + j)])
            labels.append(", ".join(tm.terms[np.argsort(sm.components_[j])[::-1][:12]]))
        splits.append({"parent_topic": int(t), "parent_share": float(sizes[t]), "sub_k": ks, "sub_stable_share": st})
        labels[t] = "(split) " + labels[t]
        next_id += ks

    assign = pd.DataFrame({"WO_NO": dfp["WO_NO"], "primary_topic": primary, "topics": multi, "primary_weight": weight})
    assign.to_csv(out / "assignments.csv", index=False)
    topics = pd.DataFrame({"topic_id": range(len(labels)), "top_terms": labels})
    topics["label"] = topics["top_terms"].str.split(", ").str[:3].str.join(" / ")
    prof = C.topic_profiles(dfp, assign["primary_topic"], cfg, topics)
    cc = F.term_field_share(df, cfg, partition)
    if len(cc):   # share of each theme's top-12 terms that come mostly (>= 70%) from the cost comment
        prof["cost_comment_term_share"] = prof["top_terms"].map(
            lambda s: np.mean([cc.get(t, 0) >= 0.7 for t in str(s).split(", ") if not t.startswith(("driver_", "track_", "kind_"))]))
    prof.to_csv(out / "topics.csv", index=False)
    np.savez_compressed(out / "model_components.npz", H=model.components_, terms=tm.terms)

    summary = {"partition": partition, "ecrs": int(len(dfp)), "vocabulary_terms": int(len(tm.terms)),
               "structured_terms": int((tm.ngram == 0).sum()), "chosen_k": k, "chosen_k_reason": why_k,
               "catch_all_splits": splits, "final_topics": int(len(labels) - len(splits)),
               "axis_counts": prof["axis_proposal"].value_counts().to_dict(), "ablation": abl,
               "eval_all": C.evaluate(dfp, assign, cfg, "TF-IDF + NMF"),
               "eval_heldout": C.evaluate(dfp, assign, cfg, "TF-IDF + NMF", mask=pd.Series(held))}
    C.write_json(summary, out / "summary.json")
    log.info("[%s] done: k=%d (%s)", partition, k, why_k)
    return summary


def run(cfg: dict | None = None, partitions: list[str] | None = None) -> None:
    cfg = cfg or C.load_config()
    df = C.load_extract(cfg)
    for p in partitions or list(cfg["partitions"]):
        run_partition(cfg, p, df)


if __name__ == "__main__":
    C.setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=None, default=None)
    a = ap.parse_args()
    run(partitions=[a.partition] if a.partition else None)
