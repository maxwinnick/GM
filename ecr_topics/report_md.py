"""Write outputs/05_topics/Steps7-8_Analysis.md: the three approaches side by side, separately for each WO
partition. Every list of up to 50 items is shown in full; cost is shown only as a share of a total.

Run:  python -m ecr_topics.report_md   (after approach1_nmf, approach2_bertopic, compare)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import common as C

KIND = {"why (driver)": "Why", "what (component)": "What", "mixed": "Mixed", "too few with VPPS": "n/a"}


def pct(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def code(s):
    return str(s).replace(" - ", " – ")


def esc(s):
    return str(s).replace("|", "/").replace("\n", " ")


def pretty_terms(label: str) -> str:
    """Show structured-field terms readably: driver_pem -> [Driver: PEM], track_exp_launch -> [Exp Launch]."""
    out = []
    for t in str(label).split(" / "):
        if t.startswith("driver_"):
            t = f"[Driver: {t[7:].replace('_', '/').upper()}]"
        elif t.startswith("track_"):
            t = f"[{t[6:].replace('_', ' ').title()}]"
        elif t.startswith("kind_"):
            t = f"[Type: {t[5:].replace('_', ' ')}]"
        out.append(t)
    return " / ".join(out)


def kind_of(x) -> str:
    if pd.notna(x.get("cost_comment_term_share", np.nan)) and x["cost_comment_term_share"] >= 0.5:
        return "Cost handling"
    return KIND.get(x["axis_proposal"], x["axis_proposal"])


def theme_table(dfp, assign, topics, limit=50, outlier_label=None):
    d = dfp[["WO_NO", "STATE", "REASON", "DRIVER_N", "SUB_KIND", "SUB_TRACK", "cost_given", "cost_recalc"]].merge(
        assign[["WO_NO", "primary_topic"]], on="WO_NO", how="left")
    d["primary_topic"] = d["primary_topic"].fillna(-1).astype(int)
    tot_n, tot_g, tot_r = len(d), d["cost_given"].sum(), d["cost_recalc"].sum()
    rows = []
    for t, g in d.groupby("primary_topic"):
        rc = g["REASON"].value_counts()
        dr = g["DRIVER_N"].replace("", "(blank)").value_counts()
        sub = (g["SUB_TRACK"].where(g["SUB_TRACK"] != "Regular", "") + " " + g["SUB_KIND"]).str.strip().value_counts()
        top2 = g.nlargest(2, "cost_recalc")
        cr = g["cost_recalc"].sum()
        rows.append({"topic_id": t, "n": len(g), "ecr_share": len(g) / tot_n, "cost_g": g["cost_given"].sum() / tot_g,
                     "cost_r": cr / tot_r, "reason": f"{code(rc.index[0])} ({rc.iloc[0] / len(g):.0%})",
                     "driver": f"{dr.index[0]} ({dr.iloc[0] / len(g):.0%})",
                     "subtype": f"{sub.index[0]} ({sub.iloc[0] / len(g):.0%})",
                     "top2": ", ".join(f"WO {w} ({s})" for w, s in zip(top2["WO_NO"], top2["STATE"])) if cr > 0 else "–",
                     "top2_share": top2["cost_recalc"].sum() / cr if cr > 0 else np.nan})
    keep = [c for c in ("topic_id", "label", "axis_proposal", "cost_comment_term_share") if c in topics]
    r = pd.DataFrame(rows).merge(topics[keep], on="topic_id", how="left")
    out_row = r[r["topic_id"] == -1]
    r = r[r["topic_id"] != -1].sort_values("n", ascending=False).reset_index(drop=True)
    r["cum"] = r["n"].cumsum() / r["n"].sum()
    shown = r.head(limit)
    L = ["| # | Theme (top words) | ECRs | % of ECRs | Cumul. % | % of cost (given) | % of cost (recalc.) | Kind | Top existing reason code | Top driver | Top change type | 2 largest ECRs (share of theme's recalc. cost) |",
         "|---:|---|---:|---:|---:|---:|---:|---|---|---|---|---|"]
    for i, x in shown.iterrows():
        L.append(f"| {i + 1} | {esc(pretty_terms(x['label']))} | {x['n']:,} | {pct(x['ecr_share'])} | {pct(x['cum'], 0)} | {pct(x['cost_g'])} | "
                 f"{pct(x['cost_r'])} | {kind_of(x)} | {x['reason']} | {x['driver']} | "
                 f"{x['subtype']} | {x['top2']} ({pct(x['top2_share'], 0)}) |")
    if outlier_label and len(out_row):
        x = out_row.iloc[0]
        L.append(f"| – | *{outlier_label}* | {x['n']:,} | {pct(x['ecr_share'])} | – | {pct(x['cost_g'])} | {pct(x['cost_r'])} | – | "
                 f"{x['reason']} | {x['driver']} | {x['subtype']} | {x['top2']} ({pct(x['top2_share'], 0)}) |")
    note = "" if len(r) <= limit else f"\n\n*Top {limit} of {len(r)} themes; the rest are in the topics file.*"
    return "\n".join(L) + note, r


def llm_table(dfp, labels, col, book, fields, head):
    d = dfp[["WO_NO", "cost_given", "cost_recalc"]].merge(labels[["WO_NO", col]], on="WO_NO")
    d[col] = d[col].fillna("")
    tg, tr, n = d["cost_given"].sum(), d["cost_recalc"].sum(), len(d)
    rows = []
    for t in book:
        g = d[d[col].map(lambda s: t["name"] in [x for x in str(s).split(";") if x])]
        rows.append((t, len(g), len(g) / n, g["cost_given"].sum() / tg if tg else np.nan, g["cost_recalc"].sum() / tr if tr else np.nan))
    rows.sort(key=lambda x: -x[1])
    hdr = " | ".join(f[1] for f in fields)
    L = [f"| # | {head} | {hdr} | ECRs (of {n}) | % of sample ECRs | % of sample cost (given) | % of sample cost (recalc.) |",
         "|---:|---|" + "---|" * len(fields) + "---:|---:|---:|---:|"]
    for i, (t, k, s, cg, cr) in enumerate(rows, 1):
        nm = t["name"] + (f" *(was: {t['given_name']})*" if t.get("given_name") and t["given_name"] != t["name"] else "")
        L.append(f"| {i} | {esc(nm)} | " + " | ".join(esc(t.get(f[0], "")) for f in fields) +
                 f" | {k} | {pct(s)} | {pct(cg)} | {pct(cr)} |")
    none = d[d[col] == ""]
    L.append(f"| – | *No label* | " + " | ".join("–" for _ in fields) +
             f" | {len(none)} | {pct(len(none) / n)} | {pct(none['cost_given'].sum() / tg if tg else np.nan)} | "
             f"{pct(none['cost_recalc'].sum() / tr if tr else np.nan)} |")
    return "\n".join(L)


def label_sets(df_lab, col):
    sets = {}
    for w, v in zip(df_lab["WO_NO"], df_lab[col].fillna("")):
        for x in [y for y in str(v).split(";") if y]:
            sets.setdefault(x, set()).add(w)
    return sets


def review_flags(df_lab, col, n, name) -> list[str]:
    """Measured warning signs: very broad labels and pairs of labels that mostly co-occur."""
    sets = label_sets(df_lab, col)
    out = [f"- **{name}: '{k}' is on {len(v)} of {n} ECRs ({len(v) / n:.0%})**, broad enough to act as a catch-all."
           for k, v in sorted(sets.items(), key=lambda kv: -len(kv[1])) if len(v) / n >= 0.25]
    keys = sorted(sets, key=lambda k: -len(sets[k]))
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = len(sets[a] & sets[b])
            small = min(len(sets[a]), len(sets[b]))
            if small >= 5 and inter / small >= 0.7:
                out.append(f"- **{name}: '{a}' and '{b}' overlap.** {inter} of the {small} ECRs with the smaller label carry both, so they may be one reason.")
    return out


def structured_only(docs, df_lab, qcol) -> tuple[int, int]:
    from .compare import quote_fields
    d = dict(zip(docs["WO_NO"], docs["doc"]))
    lab = df_lab[df_lab[qcol].fillna("") != ""]
    k = sum(1 for w, q in zip(lab["WO_NO"], lab[qcol]) if (f := quote_fields(d.get(w, ""), q)) and all(x in ("Change type", "Driver") for x in f))
    return k, len(lab)


def sweep_md(sw):
    L = ["| Themes tried (k) | Themes that came back on a rebuild (avg of 3) | Worst of 3 | Biggest theme's share | Themes over 12% |",
         "|---:|---:|---:|---:|---:|"]
    for _, x in sw.iterrows():
        L.append(f"| {int(x['k'])} | {pct(x['stable_share'], 0)} | {pct(x['stable_share_min'], 0)} | {pct(x['largest_topic_share'])} | {int(x['topics_above_max_share'])} |")
    return "\n".join(L)


NAMES = {"TF-IDF + NMF": "Word counting (NMF)", "BERTopic": "BERTopic", "BERTopic (all assigned)": "BERTopic (all forced into a theme)",
         "TopicGPT (why)": "AI: TopicGPT why", "TopicGPT (what)": "AI: TopicGPT what", "LLooM (why)": "AI: LLooM",
         "Existing reason code": "Existing reason code"}


def partition_section(cfg, part, df) -> str:
    o = cfg["paths"]["outputs"] / part
    P = cfg["partitions"][part]
    pm = C.partition_mask(df, cfg, part)
    dfp = df[pm].reset_index(drop=True)
    ra = lambda p: pd.read_csv(p, dtype={"WO_NO": str})
    s1 = json.loads((o / "approach1_nmf/summary.json").read_text())
    has_bt = (o / "approach2_bertopic/summary.json").exists()
    s2 = json.loads((o / "approach2_bertopic/summary.json").read_text()) if has_bt else {}
    sc = json.loads((o / "comparison/comparison_summary.json").read_text())
    nmf_tab, nmf_r = theme_table(dfp, ra(o / "approach1_nmf/assignments.csv"), pd.read_csv(o / "approach1_nmf/topics.csv"),
                                 outlier_label="No theme (no usable words)")
    if has_bt:
        bt_tab, bt_r = theme_table(dfp, ra(o / "approach2_bertopic/assignments.csv"), pd.read_csv(o / "approach2_bertopic/topics.csv"),
                                   outlier_label="Unassigned")
    abl = pd.DataFrame(s1["ablation"])
    abl_md = "\n".join(["| Vocabulary version | Terms | Themes chosen | Themes that came back on a rebuild | Biggest theme |", "|---|---:|---:|---:|---:|"] +
                       [f"| {r['variant']} | {r['terms']:,} | {r['chosen_k']} | {pct(r['stable_share'], 0)} | {pct(r['largest_topic_share'])} |" for _, r in abl.iterrows()])
    if has_bt:
        bsw = pd.DataFrame(s2["sweep"])
        bsw_md = "\n".join(["| Smallest group size | Themes (avg of 3 runs) | Unassigned ECRs | Agreement between runs (0–1) |", "|---:|---:|---:|---:|"] +
                           [f"| {int(r['min_cluster_size'])} | {r['n_topics']:.0f} | {pct(r['unassigned'], 0)} | {r['ari']:.2f} |" for _, r in bsw.iterrows()])
        bt_md = f"""{bsw_md}

Chosen smallest group size: **{s2['chosen_min_cluster_size']}** ({s2['choice_reason']}).

{bt_tab}"""
    else:
        bt_md = "*BERTopic was not run (packages or the local embedding model are missing; see `make models`).*"

    ev = pd.DataFrame(sc["eval_sample"]).set_index("approach")
    cols = [c for c in ["TF-IDF + NMF", "BERTopic", "TopicGPT (why)", "LLooM (why)", "TopicGPT (what)"] if c in ev.index]
    side = ["| | " + " | ".join(NAMES[c] for c in cols) + " |", "|---|" + "---:|" * len(cols),
            "| Themes used as a main label | " + " | ".join(f"{int(ev.loc[c, 'n_topics'])}" for c in cols) + " |",
            "| ECRs given a label | " + " | ".join(pct(ev.loc[c, 'coverage'], 0) for c in cols) + " |",
            "| Biggest theme's share | " + " | ".join(pct(ev.loc[c, 'largest_topic_share'], 0) for c in cols) + " |",
            "| Labels per labeled ECR | " + " | ".join(f"{ev.loc[c, 'mean_labels_per_ecr']:.2f}" for c in cols) + " |",
            "| Themes to cover 80% of 'Other' ECRs | " + " | ".join(f"{int(ev.loc[c, 'other_ecrs_topics_to_80pct'])}" for c in cols) + " |"]
    agr = pd.DataFrame(sc["agreement"]).sort_values("ari", ascending=False)
    agr = agr[~(agr["a"].str.startswith("BERTopic") & agr["b"].str.startswith("BERTopic"))]
    agr_md = "\n".join(["| Method A | Method B | ECRs both labeled | Agreement (0 = none, 1 = identical) |", "|---|---|---:|---:|"] +
                       [f"| {NAMES.get(r['a'], r['a'])} | {NAMES.get(r['b'], r['b'])} | {int(r['n'])} | {r['ari']:.2f} |" for _, r in agr.iterrows()])

    llm_md = "*The AI codebook step (stage `llm_topics`) has not been run for this partition.*"
    d3 = o / "approach3_llm"
    if (d3 / "topicgpt_codebook.json").exists():
        book = json.loads((d3 / "topicgpt_codebook.json").read_text())
        tg = ra(d3 / "topicgpt_assignments.csv")
        lm = ra(d3 / "lloom_scores.csv")
        concepts = json.loads((d3 / "lloom_concepts.json").read_text())["final"]
        f_tg = [("definition", "What it means (AI)"), ("include_rule", "Include when"), ("exclude_rule", "Not when")]
        evid = sc.get("evidence_fields", {})
        ev_md = "\n".join(["| Labels from | " + " | ".join(evid) + " |", "|---|" + "---:|" * len(evid)] +
                          [f"| {fld} | " + " | ".join(str(evid[k].get(fld, 0)) for k in evid) + " |"
                           for fld in sorted({f for v in evid.values() for f in v})])
        run_info = json.loads((cfg["paths"]["outputs"] / "run_info.json").read_text()) if (cfg["paths"]["outputs"] / "run_info.json").exists() else {}
        ev_docs = ra(o / "approach3_llm_inputs/llm_evaluation_sample.csv")
        n_ev = len(ev_docs)
        flags = review_flags(tg, "why_topics", n_ev, "TopicGPT") + review_flags(lm, "concepts", n_ev, "LLooM")
        so_tg, n_tg = structured_only(ev_docs, tg, "why_quotes")
        so_lm, n_lm = structured_only(ev_docs, lm, "quotes")
        struct_line = (f"\n\n**Labels supported only by Driver or Change type** (no supporting words in the title or cost comment): TopicGPT {so_tg} of {n_tg} labeled ECRs, LLooM {so_lm} of {n_lm}."
                       if P["structured_fields"] else "")
        n_gen = len(ra(o / "approach3_llm_inputs/llm_generation_sample.csv"))
        llm_md = f"""**Run:** model `{run_info.get('model', '?')}`. The codebook was built from {n_gen} ECRs; {n_ev} held-out ECRs were labeled.

**Validity:** every label needs an exact quote from the ECR. TopicGPT returned {pct(sc['topicgpt_validity']['records_returned'], 0)} of records; {sc['topicgpt_validity']['labels_dropped_invalid']} labels were dropped for a bad quote or topic name. LLooM: {pct(sc['lloom_validity']['records_returned'], 0)} returned, {sc['lloom_validity']['labels_dropped_invalid']} dropped.

**Where the evidence came from** (number of labels whose quote sits in each field):

{ev_md}{struct_line}

**Review flags (measured):**

{chr(10).join(flags) if flags else "- none"}

#### All {len(book['why'])} TopicGPT "why" topics

{llm_table(dfp, tg, 'why_topics', book['why'], f_tg, 'Topic')}

#### All {len(book['what'])} TopicGPT "what" topics

{llm_table(dfp, tg, 'what_topic', book['what'], f_tg[:2], 'Topic')}

#### All {len(concepts)} LLooM concepts

{llm_table(dfp, lm, 'concepts', concepts, [('definition', 'What it means (AI)'), ('criterion', 'Yes/no test')], 'Concept')}"""

    return f"""## {P['label']}

**{len(dfp):,} ECRs** ({pct(len(dfp) / len(df), 0)} of all) · **free text used:** {', '.join(C.FIELD_LABEL[f] for f in P['text_fields'])}{' · **structured fields used:** ' + ', '.join(P['structured_fields']) if P['structured_fields'] else ''} · held-out test ECRs: {sc['heldout_ecrs']:,} · AI test sample: {sc['eval_sample_ecrs']}

Cost columns below are shares of **this partition's** total. This partition holds {pct(dfp['cost_given'].sum() / df['cost_given'].sum(), 0)} of all cost as given and {pct(dfp['cost_recalc'].sum() / df['cost_recalc'].sum(), 0)} of recalculated cost.

### Step 7: vocabulary

{s1['vocabulary_terms']:,} terms (single words and two-word phrases in at least 5 and at most 25% of this partition's ECRs){f", including {s1['structured_terms']} structured-field terms" if s1['structured_terms'] else ''}.

**What each vocabulary change did to theme stability** (measured, not assumed):

{abl_md}

### Method 1: Word counting (TF-IDF + NMF)

{sweep_md(pd.read_csv(o / 'approach1_nmf/theme_sweep.csv'))}

Chosen: **{s1['chosen_k']} themes** ({s1['chosen_k_reason']}).

{nmf_tab}

### Method 2: BERTopic

{bt_md}

### Method 3: AI codebook (TopicGPT and LLooM)

{llm_md}

### Side by side ({sc['eval_sample_ecrs']} AI test ECRs)

{chr(10).join(side)}

### How much the methods agree ({sc['eval_sample_ecrs']} AI test ECRs)

{agr_md}
"""


def key_findings(cfg, df) -> str:
    out = []
    for part in cfg["partitions"]:
        o = cfg["paths"]["outputs"] / part
        lab = cfg["partitions"][part]["label"]
        s1 = json.loads((o / "approach1_nmf/summary.json").read_text())
        has_bt = (o / "approach2_bertopic/summary.json").exists()
        t = pd.read_csv(o / "approach1_nmf/topics.csv")
        sw = pd.read_csv(o / "approach1_nmf/theme_sweep.csv").set_index("k")
        cost_h = int((t.get("cost_comment_term_share", pd.Series(dtype=float)) >= 0.5).sum())
        why = int((t["axis_proposal"] == "why (driver)").sum()) - (int(((t.get("cost_comment_term_share", 0) >= 0.5) & (t["axis_proposal"] == "why (driver)")).sum()) if cost_h else 0)
        ev = json.loads((o / "comparison/comparison_summary.json").read_text())
        out.append(f"\n**{lab}**\n" if out else f"**{lab}**\n")
        out.append(f"- Word counting: {s1['chosen_k']} themes; {pct(sw.loc[s1['chosen_k'], 'stable_share'], 0)} came back when rebuilt on other data. "
                   f"{why} read as a reason (why), {int((t['axis_proposal'] == 'what (component)').sum())} as a part (what)"
                   + (f", and **{cost_h} are built mostly from cost-comment wording** (cost handling, not a reason)." if cost_h else "."))
        if has_bt:
            s2 = json.loads((o / "approach2_bertopic/summary.json").read_text())
            bt = pd.read_csv(o / "approach2_bertopic/topics.csv")
            held_bt = [r for r in ev["heldout"] if r["approach"] == "BERTopic"][0]
            out.append(f"- BERTopic: {s2['n_topics']} themes, {pct(1 - held_bt['coverage'], 0)} of held-out ECRs left unassigned; "
                       f"{int((bt.get('axis_proposal', pd.Series(dtype=str)) == 'what (component)').sum()) if 'axis_proposal' in bt else s2['axis_counts'].get('what (component)', 0)} of {s2['n_topics']} themes are about one part (what).")
        mt = {r["approach"]: r for r in ev["eval_sample"]}
        if "TopicGPT (why)" in mt:
            out.append(f"- AI on {ev['eval_sample_ecrs']} test ECRs: TopicGPT found a stated reason for {pct(mt['TopicGPT (why)']['coverage'], 0)}, "
                       f"LLooM labeled {pct(mt['LLooM (why)']['coverage'], 0)}, and TopicGPT named a part for {pct(mt['TopicGPT (what)']['coverage'], 0)}. "
                       "Review flags for broad or overlapping labels are listed in the AI section.")
    return "\n".join(out)


def export_labels(cfg, df) -> pd.DataFrame:
    """One row per ECR with every approach's label, for charts and Power BI (outputs/ecr_labels_all.csv)."""
    frames = []
    for part in cfg["partitions"]:
        o = cfg["paths"]["outputs"] / part
        dfp = df[C.partition_mask(df, cfg, part)]
        base = dfp[["WO_NO", "STATE", "CREATED_ON", "REASON", "DRIVER_N", "SUBTYPE", "VPPS", "FNA_DESC",
                    "cost_given", "cost_recalc"]].rename(columns={"DRIVER_N": "DRIVER_NORMALIZED"})
        base.insert(1, "WO_PARTITION", cfg["partitions"][part]["label"])
        for tag, folder in (("NMF", "approach1_nmf"), ("BERTOPIC", "approach2_bertopic")):
            if not (o / folder / "assignments.csv").exists():
                continue
            a = pd.read_csv(o / folder / "assignments.csv", dtype={"WO_NO": str})
            t = pd.read_csv(o / folder / "topics.csv")
            lab = dict(zip(t["topic_id"], t["label"].map(pretty_terms)))
            kind = {r["topic_id"]: kind_of(r) for _, r in t.iterrows()} if "axis_proposal" in t else {}
            a[f"{tag}_THEME"] = a["primary_topic"].map(lab).fillna("(unassigned)")
            a[f"{tag}_KIND"] = a["primary_topic"].map(lambda i: kind.get(i, ""))
            if tag == "NMF":
                a["NMF_ALL_THEMES"] = a["topics"].fillna("").map(lambda v: "; ".join(lab.get(int(x), x) for x in str(v).split(";") if x))
            base = base.merge(a.drop(columns=["primary_topic", "topics", "primary_weight"]), on="WO_NO", how="left")
        d3 = o / "approach3_llm"
        if (d3 / "topicgpt_assignments.csv").exists():
            tg = pd.read_csv(d3 / "topicgpt_assignments.csv", dtype={"WO_NO": str})[["WO_NO", "why_topics", "why_quotes", "what_topic", "what_quote"]]
            lm = pd.read_csv(d3 / "lloom_scores.csv", dtype={"WO_NO": str})[["WO_NO", "concepts", "quotes"]]
            tg.columns = ["WO_NO", "AI_TOPICGPT_WHY", "AI_TOPICGPT_WHY_QUOTES", "AI_TOPICGPT_WHAT", "AI_TOPICGPT_WHAT_QUOTE"]
            lm.columns = ["WO_NO", "AI_LLOOM_WHY", "AI_LLOOM_QUOTES"]
            base = base.merge(tg, on="WO_NO", how="left").merge(lm, on="WO_NO", how="left")
            base["IN_AI_SAMPLE"] = base["WO_NO"].isin(set(tg["WO_NO"]))
        frames.append(base)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cfg["paths"]["outputs"] / "ecr_labels_all.csv", index=False)
    return out


def run(cfg: dict | None = None) -> None:
    cfg = cfg or C.load_config()
    o = cfg["paths"]["outputs"]
    df = C.load_extract(cfg)
    tot_g, tot_r = df["cost_given"].sum(), df["cost_recalc"].sum()
    top5 = df.nlargest(5, "cost_recalc")
    parts = list(cfg["partitions"])
    ov = ["| Partition | ECRs | % of ECRs | % of cost (given) | % of cost (recalc.) | Free text | Structured fields |", "|---|---:|---:|---:|---:|---|---|"]
    for p in parts:
        m = C.partition_mask(df, cfg, p)
        P = cfg["partitions"][p]
        ov.append(f"| {P['label']} | {m.sum():,} | {pct(m.mean(), 0)} | {pct(df.loc[m, 'cost_given'].sum() / tot_g, 0)} | "
                  f"{pct(df.loc[m, 'cost_recalc'].sum() / tot_r, 0)} | {', '.join(C.FIELD_LABEL[f] for f in P['text_fields'])} | "
                  f"{', '.join(P['structured_fields']) or '–'} |")
    kf = key_findings(cfg, df)
    md = f"""# Steps 7–8 Analysis: Three Approaches, Run Separately for Each WO Range

IOE 424 · GM Case Study No. 6 · Cost Commandos Consulting · generated {pd.Timestamp.today():%B %d, %Y} from `{cfg['paths']['extract_xlsx'].name}`

## Method summary

- **Partitions.** Every model is fit separately for each WO range, because the range decides which free text an ECR has.
- **Title-only ECRs** also use their cost comment and the structured Driver and Change type (SUBTYPE) fields.
- **Step 7 vocabulary.** Position and generic-action single words are dropped (`resources/theme_stopwords.txt`); two-word phrases that contain them are kept. The effect is measured below.
- **DRIVER** long and short spellings are normalized (`resources/driver_map.csv`).
- **BERTopic** sweeps the smallest group size (with 3 seeds each) and states its choice rule.
- **AI codebook.** Merged topics are renamed to fit everything merged into them, every topic has include and exclude rules, and each ECR gets at most 2 "why" labels, each backed by an exact quote.
- Results are coverage, stability and agreement only; accuracy against hand labels is not measured here.

## Key findings (measured)

{kf}

## How to read the cost columns

Cost is shown only as a share of a total, never in dollars, because GM said the extract's amounts are directional.
- "Given" = COST_USD. "Recalc." = COST × EXCHANGE_RATE, or COST_USD where there is no rate.
- Inside each partition, shares are of **that partition's** total.
- For AI topics, shares are of the AI test sample's total. An ECR can have 2 labels, so AI shares can add up to more than 100%.

**Cost is very concentrated:**
- {pct((df['cost_recalc'] <= 0).mean(), 0)} of ECRs carry no cost.
- The 5 largest ECRs hold {pct(top5['cost_recalc'].sum() / tot_r, 0)} of all recalculated cost: WO {', '.join(f"{w} ({s})" for w, s in zip(top5['WO_NO'], top5['STATE']))}.
- Savings are not recorded as negative cost.

Every theme table names its 2 largest ECRs and their state. When those two hold most of a theme's cost, the theme's cost share reflects them, not a pattern.

## The two partitions

{chr(10).join(ov)}

---

""" + "\n---\n\n".join(partition_section(cfg, p, df) for p in parts) + """
---

## Files

All under `outputs/05_topics/`, with one folder per partition (`title_only/`, `full_text/`):

| What | File |
|---|---|
| Word counting: themes, labels, vocabulary test, stability sweep | `<partition>/approach1_nmf/` |
| BERTopic: themes, labels, group-size sweep | `<partition>/approach2_bertopic/` |
| AI samples (text sent to the LLM) | `<partition>/approach3_llm_inputs/` |
| AI codebooks, labels, quotes | `<partition>/approach3_llm/` |
| Side-by-side numbers, agreement | `<partition>/comparison/` |
| One row per ECR, every approach's label | `ecr_labels_all.csv` |
| Rebuild everything | `python run_pipeline.py` |
"""
    (o / "Steps7-8_Analysis.md").write_text(md)
    export_labels(cfg, df)


if __name__ == "__main__":
    C.setup_logging()
    run()
