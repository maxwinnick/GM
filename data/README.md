# data/

Put GM's ECR extract here. Nothing in this folder except this README is committed (see `.gitignore`).

| File | Required | Used by |
|---|---|---|
| `Masking_Lookup_Reference.xlsx` (sheet `result`) | Yes | every stage |
| `Overall Process Flow.docx` | No | context for the LLM stages only |
| `ECR Analytics Project Data 1.xlsx` | No | `notebooks/01_*` and `notebooks/02_*` only (GM's first sample extract) |

Any file name works: set `input.workbook` in `config.yaml`, pass `--input <file>`, or leave exactly
one `.xlsx` in this folder and the pipeline picks it up.

**Expected sheet `result`:** one row per ECR, unique `WO_NO`, with these columns:

- **Keys and dimensions:** `WO_NO`, `STATE`, `REASON`, `DRIVER`, `TYPE`, `SUBTYPE`, `CREATED_ON`,
  `VPPS`, `FNA_DESC`, `PART_NUMBER`, `NEW_PART_NUMBER`, `MASKED_LEAD_PROGRAM`, `MASKED_PROGRAMS`,
  `MASKED_SUPPLIER_NAME`
- **Free text (masked):** `MASKED_TITLE`, `MASKED_DESC_FOR_CHANGE`, `MASKED_REASON_FOR_CHANGE`,
  `MASKED_COST_COMMENT`
- **Cost:** `COST`, `EXCHANGE_RATE`, `COST_USD`

`tools/synthetic_extract.py` writes a fake file with the full 43-column layout if you want to see the
exact shape.
