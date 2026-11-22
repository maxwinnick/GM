"""Approach 3 — LLM-induced topics for ECR free text, run through an OpenAI-compatible endpoint.

Two published methods, adapted to the ECR "what changed / why it changed" framework:

* TopicGPT (Pham et al., NAACL 2024): the LLM reads ECRs and proposes topics with
  definitions (generate), the list is merged and pruned (refine), every evaluation ECR is
  assigned topics with a supporting verbatim quote (assign), and invalid answers are
  re-prompted once (correct). Here it produces two lists: WHY (drivers, multi-select) and
  WHAT (component, single).
* LLooM (Lam et al., CHI 2024): the LLM distills each ECR into short "why" bullets,
  the bullets are grouped (TF-IDF + k-means here; the paper uses embeddings), the LLM
  synthesizes each group into concepts with explicit inclusion criteria, concepts are
  merged, and every evaluation ECR is scored against every concept.

Every request goes through ``llm_caller.call_llm`` (credentials stay in .env and are never
read here). Responses are cached on disk by a hash of (model, instructions, prompt, data), so a
rerun costs nothing. Only the ECR text in the sample files (masked by default) is sent.

The method runs separately for each WO partition (title-only vs full-text ECRs), each with its
own codebook. Title-only ECRs also carry their cost comment, Driver and Change type, which may be
quoted as evidence. Merged topics are renamed so the name fits everything merged into them;
topics get include and exclude rules; at most 2 "why" labels per ECR.

Inputs:  <topics_out>/<partition>/approach3_llm_inputs/llm_{generation,evaluation}_sample.csv
Outputs: <topics_out>/<partition>/approach3_llm/, <topics_out>/llm_cache/, call_log.jsonl, run_info.json
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .llm_caller import call_llm

ROOT = Path(__file__).resolve().parents[1]
TOPICS_OUT = ROOT / "outputs" / "05_topics"   # set by run_all
CACHE = TOPICS_OUT / "llm_cache"
OUT_ROOT = TOPICS_OUT
OUT = OUT_ROOT                       # set per partition by run_all
LOG = OUT_ROOT / "call_log.jsonl"


def _set_root(topics_out: Path) -> None:
    global TOPICS_OUT, CACHE, OUT_ROOT, OUT, LOG
    TOPICS_OUT = Path(topics_out)
    CACHE, OUT_ROOT = TOPICS_OUT / "llm_cache", TOPICS_OUT
    OUT, LOG = OUT_ROOT, OUT_ROOT / "call_log.jsonl"

_BASE = (
    "You are helping General Motors cost engineers analyze Engineering Change Requests (ECRs). "
    "Text is masked: program names and part numbers were removed. Abbreviations are GM shorthand; "
    "keep them as written and never guess what an abbreviation stands for. Use only what is written. "
)
CONTEXTS = {
    "title_only": _BASE + (
        "These are older ECRs. Each has only a short Title and usually a Cost comment, plus two fields "
        "from the ECR system: Driver (who or what raised the change, e.g. EGM, PRTS, CRDN) and Change type "
        "(e.g. 'PWO - Exp Launch - Part Chg with New Part Number', 'PWO - Sourcing'). Driver and Change type "
        "count as evidence only when they themselves state a reason (e.g. Exp Launch, Sourcing, Cost "
        "Reduction, Rework); a role such as EGM or a generic type such as 'Part Chg with New Part Number' is "
        "not a reason."),
    "full_text": _BASE + "These are newer ECRs. Each has a Title, a Description of the change and a Reason for the change.",
}
CONTEXT = CONTEXTS["full_text"]      # set per partition by run_all
MAX_WHY = 2

QUOTE_MAX_WORDS = 20


class OutputTooLong(Exception):
    """The model hit its output limit (finish_reason='length'); the batch must be smaller."""


# --------------------------------------------------------------------------- plumbing
def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def estimate_tokens(*texts) -> int:
    return int(sum(len(json.dumps(t, default=str)) for t in texts) / 4)


class Caller:
    """call_llm with an on-disk cache, a call log and a dry-run mode for token estimates."""

    def __init__(self, model: str, dry_run: bool = True, max_workers: int = 4):
        self.model, self.dry_run, self.max_workers = model, dry_run, max_workers
        self.est_in = self.est_out = self.calls = self.cached = 0
        self._lock = threading.Lock()
        CACHE.mkdir(parents=True, exist_ok=True)
        OUT_ROOT.mkdir(parents=True, exist_ok=True)

    def __call__(self, step: str, instructions: str, prompt: str, data: list[dict], out_tokens: int) -> dict | None:
        k = _key(self.model, instructions, prompt, data)
        f = CACHE / f"{k}.json"
        if f.exists():
            with self._lock:
                self.cached += 1
            res = json.loads(f.read_text())
            if isinstance(res, dict) and res.get("__output_too_long__"):
                raise OutputTooLong(step)         # known to overflow: go straight to the fallback
            return res
        with self._lock:
            self.est_in += estimate_tokens(instructions, prompt, data)
            self.est_out += out_tokens
            self.calls += 1
        if self.dry_run:
            return None
        t0 = time.time()
        for attempt in range(3):
            try:
                res = call_llm(model=self.model, instructions=instructions, prompt=prompt, data=data)
                break
            except Exception as e:
                if "finish_reason='length'" in str(e):      # truncated reply: retrying won't help
                    f.write_text(json.dumps({"__output_too_long__": True}))
                    raise OutputTooLong(step) from e
                if attempt == 2:                             # network / JSON / rate-limit errors: retry twice
                    raise
                time.sleep(10 * (attempt + 1))
        f.write_text(json.dumps(res))
        with self._lock, LOG.open("a") as lg:
            lg.write(json.dumps({"step": step, "model": self.model, "cache_key": k, "n_items": len(data),
                                 "seconds": round(time.time() - t0, 1)}) + "\n")
        return res

    def split(self, step: str, instructions: str, prompt: str, data: list[dict], out_per_item: int) -> dict | None:
        """Call on a batch; if the reply is cut off, halve the batch and merge the list fields."""
        try:
            return self(step, instructions, prompt, data, out_per_item * len(data))
        except OutputTooLong:
            if len(data) == 1:
                raise
            mid = len(data) // 2
            parts = [self.split(step, instructions, prompt, d, out_per_item) for d in (data[:mid], data[mid:])]
            merged: dict = {}
            for p in parts:
                for k, v in (p or {}).items():
                    if isinstance(v, list):
                        merged.setdefault(k, []).extend(v)
            return merged

    def map_split(self, step: str, instructions: str, prompt: str, batch_list: list[list[dict]],
                  out_per_item: int) -> list[dict | None]:
        """Run independent batches in parallel (max_workers at a time); results keep input order."""
        if self.max_workers <= 1 or len(batch_list) <= 1:
            return [self.split(step, instructions, prompt, b, out_per_item) for b in batch_list]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda b: self.split(step, instructions, prompt, b, out_per_item), batch_list))

    def report(self) -> str:
        return (f"{self.calls} new calls (+{self.cached} cached); est. input ~{self.est_in:,} tokens, "
                f"output ~{self.est_out:,} tokens")


def batches(items: list, n: int):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def quote_ok(quote: str, text: str) -> bool:
    q = _norm(quote)
    return bool(q) and len(q.split()) <= QUOTE_MAX_WORDS + 5 and q in _norm(text)


# --------------------------------------------------------------------------- TopicGPT
GEN_PROMPT = """Read each ECR and maintain two topic lists.
WHY topics = the driver or reason the change was made (for example a cost reduction, a supplier request, a launch issue). An ECR may have up to 2 WHY topics or none.
WHAT topics = the component or system being changed. An ECR has at most one WHAT topic.
For each ECR, reuse an existing topic when it fits; propose a new topic only when none fits. If an ECR does not state a reason, give it no WHY topic. Keep topics general enough to cover many ECRs but specific enough to be useful to a cost engineer. A topic name is 1-4 words; a definition is one sentence.
Return only compact JSON (no markdown): {"new_topics": [{"axis": "why"|"what", "name": str, "definition": str}], "ecrs": [{"id": str, "why": [topic names], "what": topic name or null}]}. "new_topics" lists ONLY topics not already in the existing list (often none). Use each supplied id exactly once."""

REFINE_PROMPT = """Below are candidate {axis} topics with how many ECRs used them during generation. Produce the final {axis} list:
1. Merge topics only when they mean the same thing. Do not merge different reasons just to shorten the list.
2. Drop topics used by fewer than {min_count} ECRs, unless clearly distinct and important.
3. Keep at most {max_topics} topics.
Return only compact JSON (no markdown), names only: {{"topics": [{{"name": str, "merged_from": [str]}}]}}."""

DESCRIBE_PROMPT = """Each final {axis} topic below lists the candidate topics (with definitions) merged into it. For each one:
- name: 1-4 words that are accurate for ALL merged candidates. Replace the given name if it does not fit them all.
- definition: one sentence.
- include_rule: one sentence on what the ECR text must say for the topic to apply.
- exclude_rule: one sentence on a similar case that does NOT belong in this topic.
Return only compact JSON (no markdown): {{"topics": [{{"given_name": str, "name": str, "definition": str, "include_rule": str, "exclude_rule": str}}]}}."""

ASSIGN_PROMPT = """Assign topics from the codebook to each ECR.
WHY: at most 2 topics whose include rule the text meets and whose exclude rule it does not; pick the most directly stated. WHAT: at most one topic.
For every assigned topic give an evidence quote copied exactly from the ECR (at most 20 words). If no topic fits, return an empty list (WHY) or null (WHAT) - do not force a fit.
Return only compact JSON (no markdown): {"records": [{"id": str, "why": [{"topic": str, "quote": str}], "what": {"topic": str, "quote": str} or null}]}. Use each supplied id exactly once and only topic names from the codebook."""


def merge_names(items, call, step, make_prompt, key, cap, min_count, depth=0, chunk=30):
    """Ask the model to merge a list of names into at most ``cap`` groups ({name, merged_from}).

    The first try sends the whole list (same request as before, so cached results are reused).
    If the reply is cut off by the endpoint's output limit, the list is sorted by name, merged in
    chunks of ``chunk`` (no pruning yet), and the chunk results are merged again; ``merged_from``
    is expanded back to the original names.
    """
    try:
        res = call(step, CONTEXT, make_prompt(cap, min_count), items, 25 * cap)
        return None if res is None else res.get(key, [])
    except OutputTooLong:
        if len(items) <= 12 or depth >= 3:
            raise
    count = {it["name"]: it.get("count", 1) for it in items}
    sub_cap = max(5, cap // 2)
    members: dict[str, set] = {}
    for part in batches(sorted(items, key=lambda it: str(it["name"]).lower()), chunk):
        for g in merge_names(part, call, f"{step}.chunk", make_prompt, key, sub_cap, 1, depth + 1, chunk) or []:
            members.setdefault(g["name"], set()).update(g.get("merged_from") or [g["name"]])
    level = [{"name": n, "count": sum(count.get(m, 0) for m in ms)} for n, ms in members.items()]
    top = merge_names(level, call, f"{step}.top", make_prompt, key, cap, min_count, depth + 1, chunk) or []
    return [{"name": g["name"], "merged_from": sorted(set().union(*[members.get(m, {m}) for m in (g.get("merged_from") or [g["name"]])]))}
            for g in top]


def _refine_axis(axis, cand, call, min_count, cap, tag):
    c_ax = [{"name": c["name"], "count": c["count"]} for c in cand if c["axis"] == axis]
    defs = {c["name"]: c["definition"] for c in cand if c["axis"] == axis}
    res = merge_names(c_ax, call, f"{tag}.refine.{axis}",
                      lambda k, mc: REFINE_PROMPT.format(axis=axis.upper(), min_count=mc, max_topics=k), "topics", cap, min_count)
    final = res if res else [{"name": f"{axis}{i}", "merged_from": [f"{axis}{i}"]} for i in range(cap)]
    data = [{"given_name": t["name"], "merged_from": [{"name": m, "definition": defs.get(m, "")}
                                                      for m in (t.get("merged_from") or [t["name"]])]} for t in final]
    out = []
    for r in call.map_split(f"{tag}.describe.{axis}", CONTEXT, DESCRIBE_PROMPT.format(axis=axis.upper()),
                            list(batches(data, 4)), 90):
        out += (r or {}).get("topics", [])
    got = {t.get("given_name"): t for t in out}
    book = []
    for t in final:
        g = got.get(t["name"], {})
        book.append({"name": g.get("name") or t["name"], "given_name": t["name"], "definition": g.get("definition", ""),
                     "include_rule": g.get("include_rule", ""), "exclude_rule": g.get("exclude_rule", ""),
                     "merged_from": t.get("merged_from", [])})
    seen, uniq = set(), []                      # renamed topics must stay unique
    for b in book:
        if b["name"] in seen:
            b["name"] = f'{b["name"]} ({b["given_name"]})'
        seen.add(b["name"])
        uniq.append(b)
    return uniq


def topicgpt(gen, ev, call, gen_batch=10, assign_batch=10, min_count=3):
    topics: dict[tuple[str, str], dict] = {}
    for b in batches(gen.to_dict("records"), gen_batch):
        current = [{"axis": a, "name": n, "definition": t["definition"]} for (a, n), t in topics.items()]
        data = [{"id": r["WO_NO"], "text": r["doc"]} for r in b]
        res = call.split("topicgpt.generate", CONTEXT + "\nExisting topics:\n" + json.dumps(current), GEN_PROMPT, data, 60)
        if res is None:
            continue
        for t in res.get("new_topics", []):
            k = (t.get("axis", "why"), str(t.get("name", "")).strip())
            topics.setdefault(k, {"definition": t.get("definition", ""), "count": 0})
        for e in res.get("ecrs", []):
            for n in e.get("why", []) or []:
                if ("why", n) in topics:
                    topics[("why", n)]["count"] += 1
            if e.get("what") and ("what", e["what"]) in topics:
                topics[("what", e["what"])]["count"] += 1
    if call.dry_run and not topics:
        topics = {("why", f"t{i}"): {"definition": "x" * 120, "count": 5} for i in range(30)}
        topics.update({("what", f"c{i}"): {"definition": "x" * 80, "count": 5} for i in range(60)})
    cand = [{"axis": a, "name": n, "definition": t["definition"], "count": t["count"]} for (a, n), t in topics.items()]
    if not call.dry_run:
        (OUT / "topicgpt_candidates.json").write_text(json.dumps(cand, indent=1))
    book = {"why": _refine_axis("why", cand, call, min_count, 20, "topicgpt"),
            "what": _refine_axis("what", cand, call, min_count, 25, "topicgpt")}
    if not call.dry_run:
        (OUT / "topicgpt_codebook.json").write_text(json.dumps(book, indent=1))
    why_names = {t["name"] for t in book["why"]}
    what_names = {t["name"] for t in book["what"]}
    cb = {ax: [{k: t[k] for k in ("name", "definition", "include_rule", "exclude_rule")} for t in book[ax]] for ax in book}
    instr = CONTEXT + "\nCODEBOOK:\n" + json.dumps(cb)
    text = dict(zip(ev["WO_NO"], ev["doc"]))
    rows = {}

    def run(records, step):
        blist = [[{"id": w, "text": text[w]} for w in b] for b in batches(records, assign_batch)]
        for res in call.map_split(step, instr, ASSIGN_PROMPT, blist, 90):
            for r in (res or {}).get("records", []):
                rows[str(r.get("id"))] = r

    run(list(ev["WO_NO"]), "topicgpt.assign")
    bad = [w for w in ev["WO_NO"] if not _valid(rows.get(w), text[w], why_names, what_names)]
    if bad and not call.dry_run:
        run(bad, "topicgpt.correct")
    out = []
    for w in ev["WO_NO"]:
        r = rows.get(w) or {}
        why = [x for x in (r.get("why") or []) if x.get("topic") in why_names and quote_ok(x.get("quote", ""), text[w])][:MAX_WHY]
        what = r.get("what") if isinstance(r.get("what"), dict) else None
        what_ok = bool(what and what.get("topic") in what_names and quote_ok(what.get("quote", ""), text[w]))
        out.append({"WO_NO": w, "why_topics": ";".join(x["topic"] for x in why),
                    "why_quotes": " || ".join(x["quote"] for x in why),
                    "what_topic": what["topic"] if what_ok else "", "what_quote": what["quote"] if what_ok else "",
                    "returned": bool(r),
                    "invalid_dropped": len(r.get("why") or []) - len(why) + (1 if (what and not what_ok) else 0)})
    return pd.DataFrame(out), book


def _valid(r, text, why_names, what_names) -> bool:
    if not r:
        return False
    for x in r.get("why") or []:
        if x.get("topic") not in why_names or not quote_ok(x.get("quote", ""), text):
            return False
    w = r.get("what")
    if isinstance(w, dict) and (w.get("topic") not in what_names or not quote_ok(w.get("quote", ""), text)):
        return False
    return True


# --------------------------------------------------------------------------- LLooM
DISTILL_PROMPT = """For each ECR, write up to 2 short bullets (each at most 15 words) stating WHY the change is being made, using only what is written. If nothing says why, return an empty list.
Return only compact JSON (no markdown): {"records": [{"id": str, "bullets": [str]}]}. Use each supplied id exactly once."""

SYNTH_PROMPT = """These bullets each state why an engineering change was made. Propose 1-3 high-level concepts that several bullets share. Each concept needs a short name (2-5 words), a one-sentence definition, and an explicit inclusion criterion phrased as a yes/no question a reviewer could answer from ECR text.
Return only compact JSON (no markdown): {"concepts": [{"name": str, "definition": str, "criterion": str, "example_bullet_ids": [int]}]}."""

MERGE_PROMPT = """Below are concepts proposed from different groups of ECR bullets. Merge only true duplicates, keep concepts distinct from each other, and return at most {max_concepts} concepts, most general first.
Return only compact JSON (no markdown), names only: {{"concepts": [{{"name": str, "merged_from": [str]}}]}}."""

CONCEPT_DESCRIBE_PROMPT = """Each final concept below lists the candidate concepts merged into it. For each one:
- name: 2-5 words accurate for ALL merged candidates (replace the given name if needed).
- definition: one sentence.
- criterion: a yes/no question a reviewer could answer from ECR text.
Return only compact JSON (no markdown): {"concepts": [{"given_name": str, "name": str, "definition": str, "criterion": str}]}."""

SCORE_PROMPT = """Score each ECR against the concepts. List at most 2 concepts per ECR: the ones whose criterion the ECR most clearly meets. For each, give an evidence quote copied exactly from the ECR (at most 20 words). If none clearly applies, list none.
Return only compact JSON (no markdown): {"records": [{"id": str, "concepts": [{"name": str, "quote": str}]}]}. Use each supplied id exactly once."""


def lloom(gen, ev, call, distill_batch=10, group_size=20, max_concepts=20, score_batch=5, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    bullets = []
    gb = list(batches(gen.to_dict("records"), distill_batch))
    blist = [[{"id": r["WO_NO"], "text": r["doc"]} for r in b] for b in gb]
    for b, res in zip(gb, call.map_split("lloom.distill", CONTEXT, DISTILL_PROMPT, blist, 45)):
        if res is None:
            bullets += [{"WO_NO": r["WO_NO"], "bullet": f"reason {i % 7} words here"} for i, r in enumerate(b)]
            continue
        for r in res.get("records", []):
            for t in r.get("bullets") or []:
                bullets.append({"WO_NO": str(r.get("id")), "bullet": str(t)})
    bl = pd.DataFrame(bullets)
    if not call.dry_run:
        bl.to_csv(OUT / "lloom_bullets.csv", index=False)
    k = max(2, min(len(bl) - 1, math.ceil(len(bl) / group_size)))
    Xb = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(bl["bullet"])
    bl["group"] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xb)
    groups = [[{"bullet_id": int(i), "bullet": t} for i, t in zip(sub.index, sub["bullet"])] for _, sub in bl.groupby("group")]
    concepts = []
    with ThreadPoolExecutor(max_workers=max(1, call.max_workers)) as ex:
        for res in ex.map(lambda d: call("lloom.synthesize", CONTEXT, SYNTH_PROMPT, d, 400), groups):
            concepts += res.get("concepts", []) if res else [{"name": "x", "definition": "x" * 100, "criterion": "x" * 100}] * 2
    res = merge_names([{"name": c.get("name"), "definition": c.get("definition")} for c in concepts], call, "lloom.merge",
                      lambda k, mc: MERGE_PROMPT.format(max_concepts=k), "concepts", max_concepts, 1)
    merged = res if res else [{"name": f"c{i}", "merged_from": [f"c{i}"]} for i in range(max_concepts)]
    by_name = {c.get("name"): c for c in concepts}
    data = [{"given_name": m["name"], "merged_from": [{"name": n, "definition": by_name.get(n, {}).get("definition", ""),
                                                      "criterion": by_name.get(n, {}).get("criterion", "")}
                                                     for n in (m.get("merged_from") or [m["name"]])]} for m in merged]
    described = []
    for r in call.map_split("lloom.describe", CONTEXT, CONCEPT_DESCRIBE_PROMPT, list(batches(data, 4)), 80):
        described += (r or {}).get("concepts", [])
    got = {c.get("given_name"): c for c in described}
    final, seen = [], set()
    for m in merged:
        g = got.get(m["name"], {})
        nm = g.get("name") or m["name"]
        if nm in seen:
            nm = f'{nm} ({m["name"]})'
        seen.add(nm)
        final.append({"name": nm, "given_name": m["name"], "definition": g.get("definition", ""),
                      "criterion": g.get("criterion", ""), "merged_from": m.get("merged_from", [])})
    if res:
        (OUT / "lloom_concepts.json").write_text(json.dumps({"candidates": concepts, "final": final}, indent=1))
    names = {c["name"] for c in final}
    instr = CONTEXT + "\nCONCEPTS:\n" + json.dumps([{k: c[k] for k in ("name", "definition", "criterion")} for c in final])
    text = dict(zip(ev["WO_NO"], ev["doc"]))
    rows = {}
    blist = [[{"id": w, "text": text[w]} for w in b] for b in batches(list(ev["WO_NO"]), score_batch)]
    for r in call.map_split("lloom.score", instr, SCORE_PROMPT, blist, 70):
        for rec in (r or {}).get("records", []):
            rows[str(rec.get("id"))] = rec
    out = []
    for w in ev["WO_NO"]:
        rec = rows.get(w) or {}
        ok = [c for c in rec.get("concepts") or [] if c.get("name") in names and quote_ok(c.get("quote", ""), text[w])][:MAX_WHY]
        out.append({"WO_NO": w, "concepts": ";".join(c["name"] for c in ok),
                    "quotes": " || ".join(c["quote"] for c in ok), "returned": bool(rec),
                    "invalid_dropped": len(rec.get("concepts") or []) - len(ok)})
    return pd.DataFrame(out), final


# --------------------------------------------------------------------------- entry point
PARTITIONS = ("title_only", "full_text")


def run_all(model: str, dry_run: bool = True, partitions=PARTITIONS, max_workers: int = 4,
            topics_out: Path | None = None) -> dict:
    """Run TopicGPT and LLooM separately for each WO partition.

    ``dry_run=True`` sends nothing and returns the request count and token estimate.
    """
    global OUT, CONTEXT
    if topics_out is not None:
        _set_root(topics_out)
    call = Caller(model, dry_run=dry_run, max_workers=max_workers)
    for part in partitions:
        src = OUT_ROOT / part / "approach3_llm_inputs"
        gen = pd.read_csv(src / "llm_generation_sample.csv", dtype={"WO_NO": str})
        ev = pd.read_csv(src / "llm_evaluation_sample.csv", dtype={"WO_NO": str})
        OUT, CONTEXT = OUT_ROOT / part / "approach3_llm", CONTEXTS[part]
        OUT.mkdir(parents=True, exist_ok=True)
        tg, _ = topicgpt(gen, ev, call)
        lm, _ = lloom(gen, ev, call)
        if not dry_run:
            tg.to_csv(OUT / "topicgpt_assignments.csv", index=False)
            lm.to_csv(OUT / "lloom_scores.csv", index=False)
    if not dry_run:
        (OUT_ROOT / "run_info.json").write_text(json.dumps({"model": model, "partitions": list(partitions),
                                                            "calls": call.calls, "cached": call.cached}, indent=1))
    return {"report": call.report(), "calls": call.calls, "est_input_tokens": call.est_in,
            "est_output_tokens": call.est_out}
