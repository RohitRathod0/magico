# PRD — Invoice Extraction Pipeline (Es Magico AI Intern Assignment)

> **Audience:** Claude Code. This document is the single source of truth for building the submission.
> **Owner:** Rohit Rathod. **Language:** Python 3.11+. **LLM:** Google Gemini (vision, native PDF input).

---

## 0. Instructions for Claude Code (read first)

1. Build **phase by phase** (Section 9). Do not start a phase until the previous phase's acceptance checks pass.
2. After each phase: run its tests, show the output, and stop for the user's go-ahead.
3. **Strict scope.** Build only what is in this PRD. No speculative abstractions, no plugin systems, no multi-provider layers, no "future-proofing".
4. Every module must be independently testable. Unit tests must **not** call the real LLM (use recorded/mocked responses). Only `extract.py`, `run_all.py` and `eval.py` hit the API.
5. Never commit API keys, the sample documents, or the ground-truth labels (they are confidential). Only generated `outputs/` and `reports/` are committed.
6. If something is blocked (no API key, data missing, model name rejected), **stop and ask** — do not invent data or fake results.
7. Never hand-edit numbers in the report. Every number in `REPORT.md` must come from a file the code generated.

---

## 0.1 Amendment A1 (2026-11-21): Groq instead of Gemini

**This section overrides any conflicting text below.** Owner decision: use **Groq**, not Gemini.
Facts below come from console.groq.com/docs (vision, structured-outputs, models), read 2026-11-21.

| Topic | Was (Gemini) | Now (Groq) |
|---|---|---|
| SDK | `google-genai` | `groq` (`from groq import Groq`, OpenAI-style `client.chat.completions.create`) |
| Env vars | `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_THINKING_BUDGET` | `GROQ_API_KEY`, `GROQ_MODEL` (default `qwen/qwen3.8-27b`) |
| Model | `gemini-2.5-flash` | `qwen/qwen3.8-27b`, the only publicly available Groq vision model. It is a **preview** model; note the risk in README. |
| PDF input | Native PDF bytes | **Not supported.** Render each PDF page to PNG with `pypdfium2`, then send the pages as images. Images (JPEG/PNG/WEBP) are sent as-is, base64 data URIs. Still vision-first, with a rendering step added. |
| Page limit | `TOO_MANY_PAGES` > 10 | Max **3 images per request** → `TOO_MANY_PAGES` if a PDF has > 3 pages. All 20 sample docs are ≤ 2 pages. |
| Response schema | `response_schema=InvoiceExtraction` | `response_format={"type":"json_schema","json_schema":{"name":...,"schema":<pydantic JSON schema>,"strict":true}}`. If strict + image is rejected, fall back to `json_object` and say so. We still parse and validate ourselves. |
| `llm.py` signature | `call_model(file_bytes, mime, prompt)` | `call_model(images: list[tuple[bytes, mime]], prompt) -> (text, usage)`. `pipeline.py` does the rendering. |
| Tokens | `usage_metadata` | `response.usage` (`prompt_tokens`, `completion_tokens`, plus reasoning tokens if reported). The Phase 1 smoke test must confirm these fields exist, since the docs don't show them. Docs say each image counts as 2048 input tokens. |
| Thinking | Gemini thinking budget | If the model emits reasoning, disable it or count it as output tokens. Strip any `<think>` text before JSON parsing. |
| Pricing | Gemini rates | `config/pricing.json` for `qwen/qwen3.8-27b`. The docs models table shows $0.80 in / $4.00 out per 1M tokens; the **user must confirm** these before Phase 4 is done. |
| Phase 6 text-only path | Same model, text only | Text-only via a Groq text model (`openai/gpt-oss-120b`, or the same Qwen model with text only). Decide in Phase 6. |

Phase 0 also found that the pack's real layout and labelling conventions differ from §2 and §4.1. See `docs/data_notes.md`: data lives in `data/labelled` + `data/unlablled`; `tax_total` is `0` when there's no tax; line-item `description`/`amount` are non-null.

---

## 1. Goal

Build a pipeline that takes a PDF or image and returns **validated, structured JSON** for invoices, rejects non-invoices, and — equally important — **measures itself honestly**: field-level accuracy, grouped failure analysis, and real cost per document.

Evaluation weights (from the brief) drive priorities:

| Area | Weight | Where in this PRD |
|---|---|---|
| Eval harness & metrics | 25% | §6, Phase 5 |
| Extraction & validation | 25% | §4, §5, Phases 2–4 |
| Failure analysis | 20% | §8, Phase 7 |
| Bad-input handling | 10% | §5.3, Phase 3 |
| Cost awareness | 10% | §7, Phase 4 |
| Code & communication | 10% | §10, §11 |
| Bonus (review flag, frontend) | +10 | Phases 8–9 |

**Rule from the brief:** a strong core beats a weak core with extras. Bonus phases only after core is done.

---

## 2. Inputs

Sample pack (user downloads from Google Drive and places locally; **gitignored**):

```
data/
  docs/      # 20 files: ~12 clean PDFs, ~5 scans/photos, 3 non-invoices
  labels/    # 10 ground-truth JSON files (for 10 of the 20 docs)
```

Phase 0 determines the exact filenames, extensions and how labels map to docs. Do not assume.

---

## 3. Tech decisions (with the "why" for the README)

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Pydantic, PDF/image libs, fast to build |
| LLM | Gemini Flash-tier (default `gemini-2.5-flash`, set via `GEMINI_MODEL` in `.env`) | Native PDF + image input (no OCR stage to maintain), structured JSON output with a schema, very low per-token price → fits "a couple of dollars" budget |
| Approach | **Vision-first**: send the raw PDF/image bytes to the model | ~25% of docs are scans/photos; a text-only path fails on those (this is tested as the "thing that didn't work", Phase 6) |
| SDK | `google-genai` | Official SDK, returns `usage_metadata` for token counting |
| Schema validation | Pydantic v2 | Same model used for LLM response schema and validation |
| Interface | CLI (`extract.py`) — required. Streamlit UI — bonus only | Brief accepts CLI; least code |
| Retries | `tenacity` for transport errors | Standard, small |
| Matching in eval | `rapidfuzz` | Fuzzy vendor/description matching |
| Tables | `rich` | Readable terminal results table |
| Tests | `pytest` | — |

Settings: `temperature=0`. If the model supports a thinking budget, set it via `GEMINI_THINKING_BUDGET` (default `0`) and **count thinking tokens as output tokens** in cost.

Phase 1 must verify the model name works with a 1-call smoke test. If it is rejected, stop and ask the user which model to use.

---

## 4. Output schema

### 4.1 Pydantic models (`invoice_pipeline/schema.py`)

```python
class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str | None
    quantity: float | None
    amount: float | None          # line total, NOT unit price

class InvoiceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_invoice: bool
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None      # YYYY-MM-DD, must be a real calendar date
    currency: str | None          # ISO 4217, 3 uppercase letters
    line_items: list[LineItem] | None
    tax_total: float | None
    total: float | None
```

Validators:
- `invoice_date`: regex `^\d{4}-\d{2}-\d{2}$` **and** `date.fromisoformat` succeeds.
- `currency`: regex `^[A-Z]{3}$`.
- Model validator: if `is_invoice is False` → every other field must be `None` (including `line_items`, not `[]`).

> Phase 0 checks the ground-truth conventions (e.g. `null` vs `[]`, quantity when not printed). Align the schema/prompt to those **conventions** and document it — that is convention alignment, not overfitting.

### 4.2 Files written per document

- `outputs/<doc_stem>.json` — the schema fields exactly, plus two extra keys:
  `"needs_review": bool` and `"review_reasons": list[str]`. (Bonus asks for the flag; keep it in the main file.)
- `outputs/meta/<doc_stem>.json` — run metadata:
  ```json
  {
    "file": "...", "model": "...", "prompt_version": "v1",
    "attempts": 1, "latency_s": 3.2,
    "tokens_in": 1830, "tokens_out": 210, "tokens_thinking": 0,
    "cost_usd": 0.00061,
    "validation": {"schema_ok": true, "totals_ok": true, "errors": []},
    "error": null
  }
  ```

---

## 5. Pipeline

```
file → preflight → LLM call → parse JSON → schema validate → business checks
          │                                         │ fail
          │                                         ▼
          │                              1 repair retry (errors fed back)
          │                                         │ still fail
          ▼                                         ▼
   input error JSON                    output + needs_review=true + reasons
```

### 5.1 Modules

```
invoice_pipeline/
  __init__.py
  config.py        # loads .env, model name, pricing, tolerances
  schema.py        # Pydantic models (§4.1)
  preflight.py     # file checks (§5.3)
  prompt.py        # PROMPT_VERSION + system prompt + repair prompt
  llm.py           # single function: call_model(file_bytes, mime, prompt) -> (text, usage)
  validate.py      # schema + totals + consistency checks -> list[str] errors
  pipeline.py      # extract_document(path) -> (output_dict, meta_dict)
  cost.py          # usage -> USD
extract.py         # CLI: one file -> stdout JSON
run_all.py         # all 20 docs -> outputs/
eval.py            # labelled 10 -> results table + reports/eval_results.{json,md}
```

### 5.2 Prompt (`prompt.py`, `PROMPT_VERSION = "v1"`)

System prompt must state, briefly:
- Task: extract fields from the attached document into the given JSON schema.
- **Invoice definition:** a bill from a seller requesting payment (tax invoice, invoice, bill). **Not invoices:** bank statements, purchase orders, cover letters, quotes, delivery notes, receipts-of-payment-only. If not an invoice → `is_invoice: false`, all other fields `null`.
- `vendor` = the **issuer/seller**, not the buyer/"Bill To".
- `invoice_number` = the invoice's own number, not PO/order/GST number.
- `invoice_date` = issue date (not due date), output `YYYY-MM-DD`. Watch DD/MM vs MM/DD; use vendor country/context.
- `currency` = ISO 4217 code inferred from symbols/text (₹/Rs → INR, € → EUR, £ → GBP; `$` → use context).
- Numbers as plain JSON numbers: no commas, no symbols.
- `line_items[].amount` = line total (qty × rate after line discount), not unit price.
- `tax_total` = sum of all taxes (CGST+SGST+IGST, VAT, GST, sales tax). `null` if none shown.
- `total` = final grand total payable.
- Never guess. If a field isn't on the document, use `null`.

No few-shot examples taken from the labelled set (overfitting risk).

Repair prompt (used once on failure): original instructions + `"Your previous output failed these checks: <errors>. Re-read the document and return corrected JSON."`

Pass the Pydantic model as the response schema (`response_mime_type="application/json"`, `response_schema=InvoiceExtraction`). Still parse and validate ourselves — never trust the SDK's parse alone.

### 5.3 Preflight & bad inputs (`preflight.py`)

**Chosen extra bad-input case (state in README): corrupt / unreadable / unsupported files.** Also handle API timeouts/errors (robustness, not the "chosen" case).

Checks, in order — each failure returns a structured error **without** calling the LLM (zero cost):

| Check | Error code |
|---|---|
| File missing | `FILE_NOT_FOUND` |
| 0 bytes | `EMPTY_FILE` |
| > 20 MB | `FILE_TOO_LARGE` |
| Magic bytes not PDF/PNG/JPEG/WEBP (don't trust extension) | `UNSUPPORTED_TYPE` |
| PDF fails to open with `pypdf`, or encrypted, or 0 pages | `CORRUPT_PDF` |
| PDF > 10 pages | `TOO_MANY_PAGES` |
| Image fails `PIL.Image.verify()` | `CORRUPT_IMAGE` |
| Image < 300 px on short side, or near-blank (grayscale std-dev < 5) | `UNREADABLE_IMAGE` |

Error output (stdout JSON for CLI, also written to `outputs/` by `run_all.py`):
```json
{"is_invoice": null, "vendor": null, "...": null,
 "needs_review": true, "review_reasons": ["CORRUPT_PDF"],
 "error": {"code": "CORRUPT_PDF", "message": "..."}}
```
(Note: `is_invoice: null` here means "could not determine" — this is an error envelope, not a schema-valid extraction. Document it.)

**LLM transport errors** (`llm.py`): per-request timeout 60 s; `tenacity` retry up to 3 attempts with exponential backoff on timeout / 429 / 5xx. After final failure → error code `LLM_UNAVAILABLE`, same envelope.

### 5.4 Validation (`validate.py`)

Returns a list of error strings (empty = pass).

1. **Parse:** `json.loads` succeeds → else `JSON_PARSE_ERROR`.
2. **Schema:** `InvoiceExtraction.model_validate` succeeds → else `SCHEMA_ERROR: <pydantic message>`.
3. **Totals check** (only if `is_invoice` and `line_items` non-empty and `total` not null):
   - `s = sum(item.amount for items with amount not null) + (tax_total or 0)`
   - `tol = max(1.00, 0.005 * total)` (rounding tolerance: ₹1 or 0.5%, whichever larger)
   - `|s - total| <= tol` → pass; else `TOTALS_MISMATCH: lines+tax=<s>, total=<total>`.
   - Any line item with `amount is None` → `LINE_AMOUNT_MISSING`.
4. **Required for invoices:** if `is_invoice` and `total is None` → `MISSING_TOTAL`; `vendor is None` → `MISSING_VENDOR`.

Failure policy:
- `JSON_PARSE_ERROR`, `SCHEMA_ERROR`, `TOTALS_MISMATCH` → **one** repair retry.
- Still failing → keep the best schema-valid attempt (prefer the latest), set `needs_review: true`, `review_reasons` = the errors. If no attempt is schema-valid → error envelope with `SCHEMA_ERROR`. **Never silently pass bad data.**
- Legitimate mismatches (discounts, shipping, round-off lines, tax-inclusive pricing) will be flagged, not "fixed". That's intended; call it out in failure analysis.

---

## 6. Eval harness (`eval.py`)

**One command:** `python eval.py` (defaults: `--docs data/docs --labels data/labels`).
Flags: `--use-cached` (score existing `outputs/` without calling the API — default **off**; when off, it runs the extractor on the 10 labelled docs and writes their outputs).

### 6.1 Matching rules (state these verbatim in REPORT.md)

Normalisation helpers live in `eval.py` (or `invoice_pipeline/matching.py` if it grows past ~150 lines).

| Field | Rule |
|---|---|
| `is_invoice` | Exact boolean. Scored over **all 10** labelled docs. |
| All other fields | Scored only over docs where GT `is_invoice == true`. For GT non-invoices, a separate check "all other fields null" is reported as `non_invoice_nulls`. |
| `vendor` | Normalise: lowercase, `&`→`and`, strip punctuation, collapse spaces, remove legal suffixes (`private limited, pvt, ltd, limited, llp, llc, inc, corp, corporation, co, company, gmbh, plc`). Match if normalised strings equal **or** `rapidfuzz.fuzz.token_set_ratio >= 90`. So `"Acme Pvt Ltd"` ≡ `"Acme"`. |
| `invoice_number` | Uppercase, drop all non-alphanumerics (`"INV-2041"` ≡ `"inv 2041"` ≡ `"#INV2041"`). Exact after that. Leading zeros kept. |
| `invoice_date` | Both parsed to `date`; equal. Pred not ISO → wrong. |
| `currency` | Uppercase; map symbols (`₹`,`Rs`,`INR`→INR; `€`→EUR; `£`→GBP; `$`→USD). Exact. |
| `tax_total`, `total` | Coerce to float (strip `,`, currency symbols, spaces: `"1,200.00"` ≡ `1200`). Match if `|a-b| <= 0.01`. `null` matches only `null`. |
| `line_items` | Doc-level correct iff **same count** and every GT item is matched one-to-one (Hungarian assignment via `scipy.optimize.linear_sum_assignment` on description similarity, or greedy if scipy is avoided) where an item matches when: description `token_set_ratio >= 80` after lowercase/punctuation strip, `quantity` equal (abs diff ≤ 1e-6, null≡null), `amount` within 0.01. Also report item-level precision / recall / F1 as a secondary metric. |
| Doc exact match | All 8 fields correct. |

Null-handling: GT `null` vs pred value → wrong; GT value vs pred `null` → wrong (counted as "missed"). Track `wrong_value` vs `missed` vs `hallucinated` separately for failure analysis.

### 6.2 Output

Terminal (rich table):

```
Field           Correct   N    Accuracy
is_invoice      10        10   100.0%
vendor          6         7    85.7%
...
line_items      4         7    57.1%
doc_exact       3         7    42.9%
```

Plus a per-document grid (rows = docs, cols = fields, ✓/✗/–), and a footer: total tokens, total cost, avg cost/doc, prompt version, model.

Files:
- `reports/eval_results.json` — per-doc, per-field: gt, pred, normalised values, correct, error_type.
- `reports/eval_results_<prompt_version>.md` — same tables in markdown (pasted into REPORT.md by a script, not by hand).

### 6.3 Overfitting protocol (mandatory)

1. Write prompt `v1` **before** looking at any eval errors. Run `python eval.py` once. Save as `reports/eval_results_v1.md`. This is the **headline, untuned** number.
2. If the prompt is changed after inspecting labelled errors, bump to `v2`, re-run, save `eval_results_v2.md`.
3. REPORT.md shows **both** v1 and v2, and states plainly: "v2 was tuned on the same 10 labelled docs, so its score is optimistic." Max 1 tuning round.

---

## 7. Cost (`cost.py`)

- Token counts come from the API response `usage_metadata` (prompt tokens, candidate tokens, thinking tokens if present). Never estimate from character counts.
- `config/pricing.json`:
  ```json
  {"model": "gemini-2.5-flash",
   "usd_per_1m_input": <fill>, "usd_per_1m_output": <fill>,
   "source_url": "<official pricing page>", "checked_on": "YYYY-MM-DD"}
  ```
  **Claude Code must not fill prices from memory.** Ask the user to confirm the current numbers from the official pricing page (or fetch the page if a web tool is available) before Phase 4 is marked done.
- `cost = tokens_in * in_price/1e6 + (tokens_out + tokens_thinking) * out_price/1e6`, summed over **all attempts** including repair retries and transport retries that returned usage.
- Report: avg cost/doc over all 20 docs, avg tokens in/out, min/max, cost split clean-PDF vs scan vs non-invoice, total spend of the whole eval run, and the formula. Preflight-rejected docs cost $0 and are listed separately.

---

## 8. Failure analysis (`scripts/failure_analysis.py`)

Reads `reports/eval_results.json` + `outputs/meta/` and produces `reports/failures.md`.

For every wrong field, assign one cause category (automatic heuristics first, then Claude Code reviews and corrects labels manually with the user — note in report that categorisation was partly manual):

| Category | Heuristic |
|---|---|
| `FORMAT_NORMALISATION` | values equal after looser normalisation (e.g. date format, suffixes) → matching rule issue, not model issue |
| `WRONG_ENTITY` | vendor = buyer name, invoice_number = PO number, date = due date |
| `OCR_READ_ERROR` | doc is a scan/photo and value is close (edit distance ≤ 2 or digit swap) |
| `LINE_ITEM_SEGMENTATION` | line count differs (merged/split rows, header/subtotal rows captured as items) |
| `TAX_INTERPRETATION` | tax_total off by a component (e.g. only CGST, not CGST+SGST) |
| `TOTALS_SEMANTICS` | discount / shipping / round-off / tax-inclusive pricing |
| `MISSED_FIELD` | pred null, GT not null |
| `HALLUCINATION` | pred not null, GT null |
| `CLASSIFICATION` | `is_invoice` wrong |
| `GT_AMBIGUITY` | GT itself debatable (only after manual look; show evidence) |

Output: counts per category, 2–3 concrete examples each (doc, field, GT, pred, why), and a **ranked "fix first" list** ordered by (errors removed × ease). Also report: did the totals check catch the error? (validation recall).

---

## 9. Build phases

Each phase lists **deliverables** and **acceptance checks**. Stop after each.

### Phase 0 — Repo + data inspection
- `git init`, `.gitignore` (`.env`, `data/`, `__pycache__/`, `.venv/`), `.env.example` (`GEMINI_API_KEY=`, `GEMINI_MODEL=gemini-2.5-flash`, `GEMINI_THINKING_BUDGET=0`), `requirements.txt` (pinned), empty package skeleton.
- `scripts/inspect_data.py`: lists every doc (name, magic-byte type, size, pages/resolution), which have labels, and prints 2 label files.
- Write `docs/data_notes.md`: filename→label mapping rule, GT conventions observed (null vs [], quantity defaults, number formats, date formats, how non-invoices are labelled), which docs are scans vs PDFs vs non-invoices (based on inspection; mark any guesses).
- **Accept:** script runs; 20 docs and 10 labels found; mapping is unambiguous. If not, stop and ask.

### Phase 1 — Schema + config + LLM smoke test
- `schema.py`, `config.py`, `llm.py`, `cost.py` (pricing placeholders).
- `tests/test_schema.py`: valid invoice passes; bad date (`2026-02-30`, `14/03/2026`) fails; `currency="Rs"` fails; non-invoice with a non-null field fails; extra key fails.
- `scripts/smoke_llm.py`: one real call on one labelled doc, prints raw text + usage.
- **Accept:** tests green; smoke call returns JSON + token counts. Model name confirmed.

### Phase 2 — Extractor CLI (happy path)
- `prompt.py` (v1), `pipeline.py` (no retry yet), `extract.py`.
- `python extract.py data/docs/<file> > out.json` prints schema JSON (+ `needs_review`, `review_reasons`) to stdout; logs go to stderr only. `--meta` flag also prints meta to stderr.
- Exit codes: `0` ok, `2` ok-but-needs_review, `3` input error, `4` LLM error.
- **Accept:** works on 1 clean PDF, 1 scan, 1 non-invoice (non-invoice → `is_invoice: false`, rest null). Tests with a mocked `call_model` for each of those shapes.

### Phase 3 — Validation, retry, bad inputs
- `validate.py`, `preflight.py`, repair retry + transport retry wired into `pipeline.py`.
- Test fixtures generated by `tests/conftest.py` (not committed binaries if avoidable): truncated PDF, 0-byte file, `.pdf` that is actually a text file, tiny 50×50 PNG, blank white PNG.
- `tests/test_validate.py`: totals pass within tolerance; mismatch caught; tolerance edge (±1.00 and 0.5%); `LINE_AMOUNT_MISSING`; non-invoice skips totals.
- `tests/test_pipeline.py` (mocked LLM): bad JSON → retry → success (attempts=2); persistent mismatch → `needs_review=true` with reason; timeout ×3 → `LLM_UNAVAILABLE`; each preflight error returns correct code with 0 LLM calls.
- **Accept:** all tests green; CLI on a corrupt file returns error JSON + exit code 3, no API call made.

### Phase 4 — Run all 20 + cost
- `run_all.py`: processes every file in `data/docs/`, writes `outputs/*.json` and `outputs/meta/*.json`, prints a summary (ok / needs_review / errors, total tokens, total + avg cost). Skips docs whose output already exists unless `--force`.
- Pricing confirmed by user in `config/pricing.json`.
- **Accept:** 20 output files exist (all schema-valid or proper error envelopes); summary cost is a real number; total spend well under $2.

### Phase 5 — Eval harness
- `eval.py` per §6. `tests/test_matching.py`: `"1,200.00"`≡`1200`; `"Acme Pvt Ltd"`≡`"Acme"`; `"INV-2041"`≡`"inv2041"`; date mismatch caught; line items reordered still match; extra line item → wrong; null vs 0 → wrong.
- Run once with prompt v1 → `reports/eval_results_v1.md` (headline number, §6.3).
- **Accept:** `python eval.py` from a fresh shell prints the field table + per-doc grid; matching tests green.

### Phase 6 — "Thing that didn't work" experiment
- Add `--mode text` to `pipeline.py`/`extract.py`: extract text with `pdfplumber` (images → no text) and send text only.
- `python eval.py --mode text --out reports/eval_results_text_only.md`.
- **Accept:** comparison table vision vs text-only saved; expected finding: text-only fails on scans/photos (verify, don't assume — report whatever actually happens). If text-only turns out fine, report the honest result and pick another genuine failed attempt (e.g. no response schema → parse failures), run it, and report that.

### Phase 7 — Failure analysis + (optional) one tuning round
- `scripts/failure_analysis.py` → `reports/failures.md` per §8. Review categories with the user.
- Optional v2 prompt per §6.3 (only fixing categories identified as fixable). Re-run eval and `run_all.py --force` so `outputs/` reflect the final prompt. Keep v1 results file.
- **Accept:** failures.md has category counts, examples, ranked fix list.

### Phase 8 — Bonus: needs_review validation
- `needs_review` triggers (already partly in place): any validation error after retry; preflight warnings; `is_invoice` true but ≥2 of {vendor, invoice_number, invoice_date, total} null; scan/photo input with totals mismatch.
- Add model-reported confidence? **No** — keep rule-based (cheaper, explainable). Mention as a future idea only in the report.
- In `eval.py`, add a section: confusion matrix of `needs_review` vs "doc has ≥1 field error"; flag precision, flag recall, and "% of all field errors that sit in flagged docs".
- **Accept:** numbers printed and saved; small-sample caveat (n=10) stated.

### Phase 9 — Bonus: Streamlit frontend
- `app.py`: file uploader → left: rendered preview (PDF first page via `pypdfium2`, or image), right: extracted fields table + line items + validation status + cost. Calls `pipeline.extract_document`.
- `streamlit run app.py`.
- **Accept:** works for a PDF, an image, and a corrupt file (shows error, no crash).

### Phase 10 — README + REPORT + fresh-clone check
- Write README (§10) and REPORT (§11). `scripts/build_report_tables.py` injects generated tables into REPORT.md between markers (`<!-- ACCURACY_TABLE -->` etc.) so no numbers are hand-typed.
- Fresh clone test: clone to a temp dir, create venv, `pip install -r requirements.txt`, copy `.env` and `data/`, run `pytest`, `python extract.py <file>`, `python eval.py --use-cached`. All must work.
- `git grep -i "api_key\s*=\s*['\"]A"` and a scan for key-like strings return nothing. `data/` not tracked.
- **Accept:** fresh-clone run passes; repo pushed as **private**; user shares access with Es Magico.

---

## 10. README.md contents

1. One-paragraph summary + headline accuracy (v1) and avg cost/doc.
2. Setup: Python version, `python -m venv .venv`, `pip install -r requirements.txt`, `cp .env.example .env` and add key, place sample pack in `data/docs` and `data/labels` (not included — confidential).
3. Run:
   ```
   python extract.py data/docs/invoice.pdf > out.json
   python run_all.py            # all 20 → outputs/
   python eval.py               # labelled 10 → results table
   python eval.py --use-cached  # score existing outputs, no API spend
   pytest
   streamlit run app.py         # bonus
   ```
4. Approach & why (vision-first Gemini; one paragraph).
5. Bad-input handling: non-invoices + **chosen case = corrupt/unreadable/unsupported files**, plus API timeout retries. Table of error codes + exit codes.
6. Validation & failure policy (retry once, then flag; never silent).
7. Repo layout.
8. Link to REPORT.md.

## 11. REPORT.md structure

1. **Approach** — model, why vision-first, prompting approach (schema-constrained JSON, explicit field rules, temperature 0, one repair retry), **what didn't work** (Phase 6 numbers).
2. **Matching rules** — §6.1 table.
3. **Accuracy** — field-level table for v1 (headline) and v2 (if any) with the overfitting disclaimer; doc exact-match; line-item P/R/F1; `non_invoice_nulls`.
4. **Failure analysis** — from `reports/failures.md`: categories, examples, fix-first list, validation recall (did totals check catch errors?).
5. **Cost per document** — avg USD, avg tokens in/out (+thinking), by doc type, formula, pricing source + date, total spend of all runs.
6. **Human-review flag** (bonus) — confusion matrix + caveat.
7. **Limitations & honest notes** — n=10, any manual categorisation, any conventions aligned to GT, what's not handled (multi-invoice PDFs, handwritten, >10 pages).
8. **Outputs** — pointer to `outputs/` (20 files) and the 10 unlabelled outputs listed.

Tone: plain, honest, short. No marketing language. If a number is bad, say it and explain why.

---

## 12. Final repo layout

```
.
├── README.md
├── REPORT.md
├── PRD.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/pricing.json
├── invoice_pipeline/ (config, schema, preflight, prompt, llm, validate, pipeline, cost)
├── extract.py
├── run_all.py
├── eval.py
├── app.py                      # bonus
├── scripts/ (inspect_data, smoke_llm, failure_analysis, build_report_tables)
├── tests/ (conftest, test_schema, test_validate, test_pipeline, test_matching)
├── docs/data_notes.md
├── outputs/  (20 *.json + meta/)
└── reports/  (eval_results.json, eval_results_v1.md, [v2], eval_results_text_only.md, failures.md)
```

## 13. Definition of done (submission checklist)

- [ ] `python extract.py <file>` works for PDF, image, non-invoice, corrupt file
- [ ] `python eval.py` = one command, prints per-field table
- [ ] `outputs/` has JSON for all 20 docs
- [ ] All tests pass; tests make no real API calls
- [ ] REPORT.md: approach, what didn't work, field accuracy table, grouped failures + fix-first, cost per doc with method
- [ ] v1 (untuned) score reported; any tuning disclosed
- [ ] No keys, sample docs or labels committed; `.env.example` present
- [ ] Fresh-clone run verified
- [ ] Total API spend < $2 (logged)
- [ ] Rohit can explain every file (walkthrough-ready)
