"""LLM backends for AI key-term extraction.

Every backend takes a list of ``{"id", "field", "text"}`` items and returns
``{id: [key terms]}``. Backends:

* ``openai`` - any OpenAI-compatible chat-completions endpoint (the U-M GPT Toolkit, OpenAI,
  Azure OpenAI, or an internal gateway). Credentials and the model id come from ``.env``
  (``LLM_API_KEY``, ``LLM_API_BASE``, ``LLM_MODEL``); see ``.env.example``.
* ``mock``   - offline stand-in that returns NLP-style terms. For testing plumbing only.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

from .config import FIELD_DEFINITIONS, PROJECT_DIR, LLMConfig

log = logging.getLogger(__name__)


def build_system_prompt(cfg: LLMConfig, process_flow_text: str = "") -> str:
    field_lines = "\n".join(f"- {k}: {v}" for k, v in FIELD_DEFINITIONS.items())
    guide = f"\n\n<process_flow_guide>\n{process_flow_text}\n</process_flow_guide>" if process_flow_text else ""
    return f"""You are a text-analytics assistant for General Motors cost engineers (PPCO). \
You read free-text fields from Engineering Change Requests (ECRs) and extract short, \
normalized key terms so that thousands of ECRs can be counted and compared in Pareto charts.

Field definitions:
{field_lines}

Rules for key terms:
1. Return 0 to {cfg.max_terms} key terms per item, most important first. Each term is 1-3 words.
2. Capture what matters for the field: the affected component or system, the change action \
(e.g. "material change", "hole addition", "part deletion"), and the driver or cost nature \
(e.g. "cost reduction", "supplier request", "durability issue", "tooling cost", \
"no piece cost impact", "supplier quote pending").
3. Normalize so the same concept is always written the same way: lowercase, singular nouns, \
US spelling, abbreviations expanded (frt -> front, rr -> rear, asm -> assembly, \
brkt -> bracket, ip -> instrument panel, w/h -> wire harness only if context is electrical).
4. Never include identifiers or numbers: no placeholders such as <PN>, <PROGRAM>, <AMT>, \
<ID>, <PERSON>, <EMAIL>, <PHONE>; no part numbers, program or vehicle codes (e.g. T1XX, BT1YL), \
model years, dates, amounts, units, people's names, or LH/RH side markers.
5. Use only information stated in the text. If the text has no substantive content \
(e.g. "see VCD", "N/A", "TBD", only identifiers), return an empty list.
6. Return exactly one record per supplied item, copying each id exactly.{guide}"""


def build_user_prompt(items: list[dict]) -> str:
    payload = json.dumps(items, ensure_ascii=False)
    return (
        f"Extract key terms for each of the {len(items)} items below. Each item has an id, the "
        "ECR field it came from, and the text. Return only a JSON object: "
        '{"records": [{"id": "...", "key_terms": ["..."]}]}.\n\n'

        f"<items>\n{payload}\n</items>"
    )


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional if variables are already exported
        return
    load_dotenv(PROJECT_DIR / ".env")


class Backend(ABC):
    def __init__(self, cfg: LLMConfig, system_prompt: str):
        self.cfg = cfg
        self.system_prompt = system_prompt

    @abstractmethod
    def extract(self, items: list[dict]) -> dict[str, list[str]]:
        """Return {item id: raw key terms} for one request's worth of items."""


def llm_settings() -> dict[str, str | None]:
    """Read endpoint settings from the environment / project ``.env`` (never from code)."""
    _load_env()
    return {
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("api_key"),
        "base_url": os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE") or os.getenv("base_url"),
        "model": os.getenv("LLM_MODEL") or os.getenv("MODEL"),
    }


class OpenAICompatibleBackend(Backend):
    """OpenAI-compatible chat-completions endpoint. JSON is requested in plain text."""

    def __init__(self, cfg: LLMConfig, system_prompt: str):
        super().__init__(cfg, system_prompt)
        from openai import OpenAI

        env = llm_settings()
        if not env["api_key"]:
            raise RuntimeError("Set LLM_API_KEY in the project .env file (see .env.example).")
        if not self.cfg.model:
            raise RuntimeError("Set LLM_MODEL in .env or pass --model.")
        # base_url may be blank for the public OpenAI endpoint.
        self.client = OpenAI(api_key=env["api_key"], base_url=env["base_url"] or None, max_retries=6)

    def extract(self, items: list[dict]) -> dict[str, list[str]]:
        prompt = build_user_prompt(items)
        kwargs = {}
        if self.cfg.temperature is not None:
            kwargs["temperature"] = self.cfg.temperature
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": prompt}],
            **kwargs,
        )
        text = (resp.choices[0].message.content or "").strip()
        start = text.find("{")
        if start < 0:
            raise ValueError(f"No JSON object in response: {text[:300]!r}")
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return {str(r["id"]): r.get("key_terms") or [] for r in obj.get("records", [])}


class MockBackend(Backend):
    """Offline stand-in: returns the first content words. Never use for reported results."""

    def extract(self, items: list[dict]) -> dict[str, list[str]]:
        out = {}
        for it in items:
            words = [w for w in re.findall(r"[a-z]{4,}", it["text"].lower())][: self.cfg.max_terms]
            out[it["id"]] = words
        return out


def make_backend(cfg: LLMConfig, system_prompt: str) -> Backend:
    backends = {"openai": OpenAICompatibleBackend, "mock": MockBackend}
    if cfg.backend not in backends:
        raise ValueError(f"backend must be one of {sorted(backends)}")
    return backends[cfg.backend](cfg, system_prompt)


def read_process_flow(path: Path) -> str:
    """Flatten the Overall Process Flow .docx (paragraphs + tables) into plain text."""
    if not path or not Path(path).is_file():
        log.warning("Process-flow guide not found at %s; continuing without it.", path)
        return ""
    from docx import Document

    doc = Document(path)
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(c.text.strip().replace("\n", " / ") for c in row.cells))
    return "\n".join(lines)
