"""Central configuration for the ECR text pipeline.

Edit values here (or override from the CLI / notebook) instead of editing module code.
Items not yet confirmed by GM are marked ``UNCONFIRMED`` and are switched off by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
RESOURCE_DIR = PROJECT_DIR / "resources"

# --------------------------------------------------------------------------------------
# Source data
# --------------------------------------------------------------------------------------
# The masked workbook GM provided ("Masking_Lookup_Reference 1.xlsx", sheet "result").
DEFAULT_INPUT = PROJECT_DIR / "data" / "Masking_Lookup_Reference.xlsx"
DEFAULT_SHEET = "result"
DEFAULT_PROCESS_FLOW = PROJECT_DIR / "data" / "Overall Process Flow.docx"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "01_text"

KEY = "WO_NO"

# The four free-text columns we parse. Only the MASKED_ versions are ever read for text,
# so unmasked text never reaches the NLP outputs or the LLM.
TEXT_COLUMNS: dict[str, str] = {
    # output prefix : source column
    "TITLE": "MASKED_TITLE",
    "COST_COMMENT": "MASKED_COST_COMMENT",
    "DESC_FOR_CHANGE": "MASKED_DESC_FOR_CHANGE",
    "REASON_FOR_CHANGE": "MASKED_REASON_FOR_CHANGE",
}

# Field definitions sent to the LLM (paraphrased from the Overall Process Flow guide).
FIELD_DEFINITIONS: dict[str, str] = {
    "TITLE": "Short ECR title naming the affected component/system and the change being made.",
    "COST_COMMENT": (
        "Free-text context explaining a cost entry: cost increase/decrease, supplier quote "
        "details, tooling assumptions, approvals, or why there is no cost impact."
    ),
    "DESC_FOR_CHANGE": "Description of WHAT is being changed (parts, design, material, process).",
    "REASON_FOR_CHANGE": "Explanation of WHY the change is needed (root cause / business driver).",
}

# Columns carried into the parsed output / used as Pareto dimensions. Masked versions only.
CARRY_COLUMNS = [
    "WO_NO", "STATE", "REASON", "CREATED_ON", "MASKED_LEAD_PROGRAM", "MASKED_PROGRAMS",
    "MASKED_SUPPLIER_NAME", "VPPS", "FNA_DESC", "TYPE", "SUBTYPE", "COST", "EXCHANGE_RATE",
]

# Verified in the data: WO_NO falls in two ranges, 4,131,983-4,800,660 and 7,000,105-7,186,370.
# DESC_FOR_CHANGE / REASON_FOR_CHANGE are blank for every WO in the lower range.
WO_RANGE_SPLIT = 5_000_000
WO_RANGE_LABELS = ("WO 4.1M-4.8M", "WO 7.0M+")

# Verified in the data: MASKED_LEAD_PROGRAM == "Program 1" exactly where LEAD_PROGRAM == "-" (8,865 rows).
BLANK_PROGRAM_MASK = "Program 1"


@dataclass
class ParetoConfig:
    """Settings for the Pareto report stage."""

    # All 18,625 ECRs by default. Pass e.g. ("Closed",) to filter by STATE.
    states: tuple[str, ...] | None = None  # None = all states (Closed + non-closed)
    top_terms: int = 30             # rows in each overall Pareto chart
    max_overall_terms: int = 500    # rows kept in each overall Pareto table
    top_groups: int = 15            # groups (programs / suppliers / VPPS) reported per dimension
    terms_per_group: int = 10       # frequency and distinctiveness terms per group
    min_group_docs: int = 25        # skip groups with too few ECRs to say anything
    min_term_docs: int = 5          # drop very rare terms from distinctiveness scoring
    abc_breaks: tuple[float, float] = (0.80, 0.95)
    # UNCONFIRMED: cost rules (COST x EXCHANGE_RATE, blank rate) await GM confirmation; off by default.
    cost_weighted: bool = False
    # Group dimensions: label -> derived column name (see pareto.add_dimensions)
    dimensions: dict[str, str] = field(default_factory=lambda: {
        "Program": "DIM_PROGRAM",
        "Supplier": "DIM_SUPPLIER",
        "VPPS_L1": "DIM_VPPS_L1",
        "VPPS_L2": "DIM_VPPS_L2",
        "VPPS_L3": "DIM_VPPS_L3",   # assembly: first three VPPS segments, as in the mid-term Pareto A
    })


@dataclass
class LLMConfig:
    """Settings for the AI parsing stage."""

    backend: str = "openai"               # "openai" (any OpenAI-compatible endpoint) | "mock"
    model: str = ""                       # blank = LLM_MODEL from .env
    max_tokens: int = 4096
    temperature: float = 0.0
    items_per_request: int = 25           # unique texts sent per request
    max_chars_per_text: int = 1500        # long descriptions are truncated (head) before sending
    max_workers: int = 4                  # parallel requests (sync mode)
    max_terms: int = 6
    prompt_version: str = "v1"            # bump when the prompt changes -> invalidates cache
    cache_file: Path = DEFAULT_OUTPUT_DIR / "cache" / "ai_terms_cache.jsonl"
