# Invoice extraction pipeline

Takes a PDF or image and returns validated, structured JSON for invoices. It rejects non-invoices and bad files, and it measures itself: field-level accuracy against labelled data, grouped failure analysis, and real cost per document.

<!-- HEADLINE -->
**Headline (prompt v1, untuned):** 6/9 labelled invoices fully correct (66.7% doc exact match), is_invoice 10/10. After one disclosed tuning round (v2): 8/9 (88.9%), optimistic because it was tuned on the same documents. Average cost **$0.0034 per document** at Groq list price over 20 documents (actual spend $0 on the free plan).
<!-- /HEADLINE -->

Full write-up, including what didn't work: **[REPORT.md](REPORT.md)**.

## Setup

Python 3.11+ (developed on 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env              # then put your Groq key in GROQ_API_KEY
```

Place the sample pack (not included, confidential) as delivered:

```
data/labelled/     L01..L10 documents + L01.json..L10.json ground truth
data/unlablled/    U01..U10 documents   (the pack's spelling; "unlabelled" also works)
```

## Run

```bash
python extract.py data/labelled/L01.pdf > out.json   # one file -> JSON on stdout (--meta: run info on stderr)
python run_all.py                                    # all 20 docs -> outputs/ (skips done ones; --force to redo)
python eval.py                                       # labelled 10 -> accuracy table + per-doc grid
python eval.py --use-cached                          # score existing outputs/, no API calls
pytest                                               # unit tests, no API calls
streamlit run app.py                                 # bonus: upload a file, see preview + fields
```

Extras: `extract.py --mode text` / `eval.py --mode text` run the text-only experiment. `run_all.py --retry-errors` re-runs documents that were cut off by Groq's rate limits. `python scripts/failure_analysis.py` and `python scripts/build_report_tables.py` regenerate the report material.

## Approach

**Vision-first with one model.** Every page goes to Groq's `qwen/qwen3.8-27b` vision model as an image (PDF pages are rendered with pypdfium2, since Groq takes images only). About a quarter of real documents are scans or phone photos with no text layer. The text-only alternative was tested, and it fails on exactly those documents ([REPORT.md](REPORT.md#what-didnt-work)). The model is asked for JSON constrained by a schema (`response_format: json_schema`, strict), at temperature 0, with explicit field rules. We still parse and validate everything ourselves, and then check the arithmetic.

Note: `qwen/qwen3.8-27b` is a **preview** model on Groq and could be withdrawn. Changing `GROQ_MODEL` in `.env` is the only change needed if it is.

## Bad-input handling

Two kinds of bad input are handled:
1. **Non-invoices** (purchase orders, bank statements, letters…) return `is_invoice: false` with every other field `null`.
2. **Chosen extra case: corrupt, unreadable or unsupported files.** These are caught *before* any model call, so they cost $0. File type is detected from the file's content, never its extension.

| Code | When | Exit code |
|---|---|---|
| `FILE_NOT_FOUND` | path doesn't exist | 3 |
| `EMPTY_FILE` | 0 bytes | 3 |
| `FILE_TOO_LARGE` | over 20 MB | 3 |
| `UNSUPPORTED_TYPE` | content isn't PDF/PNG/JPEG/WEBP (e.g. a `.pdf` that is really text) | 3 |
| `CORRUPT_PDF` | can't be opened or rendered, encrypted, or 0 pages | 3 |
| `TOO_MANY_PAGES` | PDF over 3 pages (Groq allows 3 images per request) | 3 |
| `CORRUPT_IMAGE` | image can't be decoded | 3 |
| `UNREADABLE_IMAGE` | shorter side under 300 px, or nearly blank | 3 |
| `LLM_UNAVAILABLE` | model unreachable after retries | 4 |
| `SCHEMA_ERROR` | no valid JSON from the model after the repair retry | 4 |

Errors come back as the normal JSON shape with every field `null`, `needs_review: true`, and an `error: {code, message}` object. Here `is_invoice: null` means "could not determine", so this is an error envelope, not a valid extraction. Successful runs exit with `0`, or `2` if the result needs review.

**API robustness.** Timeouts, connection errors and server errors get 3 attempts with backoff. Groq's per-minute rate limit is waited out (up to 10 attempts, waiting as long as Groq asks). When the free plan's *daily* token quota is exhausted, the tool stops immediately, never overwrites a good result with an error, and `--retry-errors` finishes the remaining documents later.

## Validation and failure policy

Every response is checked in order: valid JSON → schema (real calendar date, ISO currency code, a non-invoice has nothing else filled in) → **line items + tax = total** (within max(1.00, 0.05% of total)) → required fields present.

- A JSON, schema or totals failure triggers **one repair retry**, with the exact problems fed back to the model.
- If it still fails, the result is kept but marked `needs_review: true` with the reasons. **Bad data is never passed on silently.**
- An invoice missing its total, its vendor, or 2+ of vendor/number/date/total is also flagged for review.

## Repo layout

```
extract.py            CLI: one file -> JSON
run_all.py            all documents -> outputs/
eval.py               accuracy against labels -> reports/
app.py                Streamlit page (bonus)
invoice_pipeline/
  config.py           settings, limits, paths        preflight.py   file checks (no API cost)
  schema.py           output + model-response schemas render.py     PDF -> images / text
  prompt.py           prompts (v1 kept, v2 active)    llm.py         Groq call + retries
  validate.py         schema/totals/required checks   pipeline.py    extract_document()
  matching.py         eval matching rules             cost.py        tokens -> USD
  dataset.py          finds documents and labels
scripts/              inspect_data, smoke_llm, compare_modes, failure_analysis, build_report_tables
tests/                pytest suite (mocked model, generated fixtures)
outputs/              one JSON per document + outputs/meta/ run records
reports/              eval results, failure analysis, vision vs text comparison
docs/data_notes.md    sample-pack layout and labelling conventions
config/pricing.json   Groq prices used for cost, with source and date
```
