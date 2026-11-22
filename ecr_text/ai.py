"""AI parsing stage: LLM key-term extraction for the four masked text columns.

Design choices
* Unit of work is a unique (field, scrubbed text) pair, so repeated comments such as
  "No tooling" are sent once. Results are written to an append-only JSONL cache keyed by
  hash(model, prompt_version, field, text); re-runs only pay for new or changed texts.
* Text is scrubbed of leftover identifiers (typed placeholders) and, by default, spaCy
  PERSON names before it leaves the machine.
* Each model reply is validated: every id must come back exactly once, terms are
  normalized and anything that looks like an identifier is dropped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .config import TEXT_COLUMNS, LLMConfig
from .llm import build_system_prompt, make_backend
from .scrub import scrub_person_names, scrub_text

log = logging.getLogger(__name__)

TERM_SEP = "; "
_BAD_TERM = re.compile(r"<[a-z]+>|\d|\b(?:lh|rh)\b", re.I)


# ---------------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------------
class TermCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.data: dict[str, list[str]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["terms"]

    @staticmethod
    def key(model: str, prompt_version: str, field: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x1f{prompt_version}\x1f{field}\x1f{text}".encode()).hexdigest()

    def put_many(self, records: dict[str, list[str]]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            for k, terms in records.items():
                self.data[k] = terms
                f.write(json.dumps({"key": k, "terms": terms}, ensure_ascii=False) + "\n")


def normalize_terms(raw: list, max_terms: int) -> list[str]:
    out: list[str] = []
    for t in raw or []:
        if not isinstance(t, str):
            continue
        t = re.sub(r"\s+", " ", t.strip().lower().strip(".,;:'\""))
        if not t or _BAD_TERM.search(t) or len(t.split()) > 4:
            continue
        if t not in out:
            out.append(t)
    return out[:max_terms]


# ---------------------------------------------------------------------------------------
# item preparation
# ---------------------------------------------------------------------------------------
def prepare_items(df: pd.DataFrame, cfg: LLMConfig, scrub_names: bool = True) -> pd.DataFrame:
    """Long table of (WO_NO, field, text_sent) with the exact text that would be sent."""
    frames = []
    for field, col in TEXT_COLUMNS.items():
        s = df[col].dropna().astype(str).str.strip()
        s = s[s.ne("")]
        frames.append(pd.DataFrame({"WO_NO": df.loc[s.index, "WO_NO"].values, "field": field,
                                    "raw": s.values}))
    long = pd.concat(frames, ignore_index=True)

    uniq = long["raw"].drop_duplicates()
    scrubbed = uniq.map(lambda t: scrub_text(t, "placeholder"))
    if scrub_names:
        log.info("Scrubbing person names from %d unique texts", len(scrubbed))
        scrubbed = pd.Series(scrub_person_names(scrubbed.tolist()), index=scrubbed.index)
    scrubbed = scrubbed.str.slice(0, cfg.max_chars_per_text)
    long["text_sent"] = long["raw"].map(dict(zip(uniq, scrubbed)))
    return long.drop(columns="raw")


def estimate_tokens(units: pd.DataFrame, system_prompt: str, cfg: LLMConfig) -> dict:
    """Rough token estimate (~4 characters per token)."""
    n_requests = -(-len(units) // cfg.items_per_request)
    item_tokens = int((units["text_sent"].str.len().sum() + 40 * len(units)) / 4)
    return {
        "unique_items": len(units),
        "requests": n_requests,
        "system_tokens_per_request": len(system_prompt) // 4,
        "input_tokens_items": item_tokens,
        "output_tokens_est": 15 * len(units) + 30 * n_requests,
    }


# ---------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------
def run_ai_parsing(
    df: pd.DataFrame,
    cfg: LLMConfig,
    process_flow_text: str = "",
    limit: int | None = None,
    dry_run: bool = False,
    scrub_names: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Add ``<FIELD>_AI`` columns to ``df``. Returns (df, run summary).

    Args:
        limit: only parse the first N ECRs (pilot runs). Others keep cached values or NaN.
        dry_run: build items and print the token estimate without calling the API.
    """
    system_prompt = build_system_prompt(cfg, process_flow_text)
    scope = df if limit is None else df.head(limit)
    long = prepare_items(scope, cfg, scrub_names=scrub_names)
    cache = TermCache(cfg.cache_file)

    long["key"] = [TermCache.key(cfg.model, cfg.prompt_version, f, t)
                   for f, t in zip(long["field"], long["text_sent"])]
    units = long.drop_duplicates("key")[["key", "field", "text_sent"]]
    todo = units[~units["key"].isin(cache.data)]
    est = estimate_tokens(todo, system_prompt, cfg)
    summary = {"ecrs_in_scope": len(scope), "item_rows": len(long), "unique_items": len(units),
               "already_cached": len(units) - len(todo), "to_send": len(todo), "estimate": est}
    log.info("AI parsing plan: %s", json.dumps(summary, indent=1))

    if dry_run or todo.empty:
        return _attach(df, long, cache, cfg), summary

    backend = make_backend(cfg, system_prompt)
    chunks = {f"c{i:05d}": [{"id": k, "field": f, "text": t}
                            for k, f, t in todo.iloc[i:i + cfg.items_per_request].itertuples(index=False)]
              for i in range(0, len(todo), cfg.items_per_request)}
    # The model sees short ids instead of 64-char hashes (fewer tokens, fewer copy errors).
    id_maps = {}
    for cid, items in chunks.items():
        id_maps[cid] = {str(j): it["id"] for j, it in enumerate(items, start=1)}
        for j, it in enumerate(items, start=1):
            it["id"] = str(j)

    failures = _run_sync(backend, chunks, id_maps, cache, cfg)

    summary["failed_chunks"] = failures
    summary["cached_after_run"] = int(units["key"].isin(cache.data).sum())
    return _attach(df, long, cache, cfg), summary


def _store(records: dict[str, list[str]], items: list[dict], id_map: dict[str, str],
           cache: TermCache, cfg: LLMConfig) -> int:
    """Validate one chunk's reply and cache it. Returns how many ids were missing."""
    sent = {it["id"] for it in items}
    good = {id_map[i]: normalize_terms(records[i], cfg.max_terms) for i in sent if i in records}
    cache.put_many(good)
    return len(sent - set(records))


def _run_sync(backend, chunks, id_maps, cache, cfg) -> list[str]:
    failures = []
    t0 = time.time()

    def work(cid):
        items = chunks[cid]
        recs = backend.extract(items)
        missing = _store(recs, items, id_maps[cid], cache, cfg)
        if missing:  # retry the missing ids once, on their own
            retry = [it for it in items if it["id"] not in recs]
            missing = _store(backend.extract(retry), retry, id_maps[cid], cache, cfg)
        return cid, missing

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {pool.submit(work, cid): cid for cid in chunks}
        for n, fut in enumerate(as_completed(futures), start=1):
            cid = futures[fut]
            try:
                _, missing = fut.result()
                if missing:
                    failures.append(cid)
            except Exception as exc:  # keep going; failed chunks are retried on the next run
                log.error("Chunk %s failed: %s", cid, exc)
                failures.append(cid)
            if n % 20 == 0 or n == len(chunks):
                log.info("  %d/%d requests done (%.0fs)", n, len(chunks), time.time() - t0)
    return failures


def _attach(df: pd.DataFrame, long: pd.DataFrame, cache: TermCache, cfg: LLMConfig) -> pd.DataFrame:
    """Write ``<FIELD>_AI`` columns. Blank text -> NaN; parsed with no terms -> ''."""
    long = long.assign(terms=long["key"].map(cache.data))
    long = long[long["terms"].notna()]
    long["value"] = long["terms"].map(TERM_SEP.join)
    out = df.copy()
    for field in TEXT_COLUMNS:
        m = long[long["field"] == field].drop_duplicates("WO_NO").set_index("WO_NO")["value"]
        out[f"{field}_AI"] = out["WO_NO"].map(m)
    return out


def ai_terms(value) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [t for t in value.split(TERM_SEP) if t]
