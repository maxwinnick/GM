"""Small caller for an OpenAI-compatible chat-completions API (e.g. the U-M GPT Toolkit).

Credentials and the model id come from the project ``.env`` (see ``.env.example``):
``LLM_API_KEY``, ``LLM_API_BASE`` (blank for the public OpenAI endpoint) and ``LLM_MODEL``.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str | None]:
    load_dotenv(ROOT / ".env")
    return {
        "api_key": os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("api_key"),
        "base_url": os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE") or os.getenv("base_url"),
        "model": os.getenv("LLM_MODEL") or os.getenv("MODEL"),
    }


def default_model() -> str | None:
    """The model id configured in .env (LLM_MODEL), if any."""
    return _env()["model"]


def _client() -> OpenAI:
    """Create a client using credentials from the project .env file."""
    env = _env()
    if not env["api_key"]:
        raise ValueError("Set LLM_API_KEY in the project .env file (see .env.example).")
    return OpenAI(api_key=env["api_key"], base_url=env["base_url"] or None)


def available_models() -> list[str]:
    """Return model IDs available to the configured account."""
    return sorted(model.id for model in _client().models.list().data)


def call_llm(
    *,
    model: str,
    instructions: str,
    prompt: str,
    data: list[dict[str, object]],
) -> dict[str, object]:
    """Send instructions, prompt and data; return the decoded JSON object."""
    response = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Analyze only the supplied data. Return the requested JSON.\n\n"
                    f"Input data:\n{json.dumps(data, ensure_ascii=False, default=str)}"
                ),
            },
        ],
    )
    if not response.choices:
        raise RuntimeError("The selected model returned no completion choices.")

    choice = response.choices[0]
    content = choice.message.content
    if not isinstance(content, str) or not content.strip():
        refusal = getattr(choice.message, "refusal", None)
        raise RuntimeError(
            "The selected model returned no text content. "
            f"finish_reason={choice.finish_reason!r}; refusal={refusal!r}."
        )

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    json_start = text.find("{")
    if json_start < 0:
        raise ValueError(
            "The model response did not contain a JSON object. "
            f"finish_reason={choice.finish_reason!r}; response preview={text[:500]!r}"
        )
    try:
        result, _ = json.JSONDecoder().raw_decode(text[json_start:])
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The model response did not contain valid JSON. "
            f"finish_reason={choice.finish_reason!r}; response preview={text[:500]!r}"
        ) from exc
    if not isinstance(result, dict):
        raise ValueError("The model JSON response must be an object.")
    return result
