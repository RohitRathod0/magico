# Data notes (Phase 0)

Generated from `python scripts/inspect_data.py` plus a manual look at the documents.
The sample pack is confidential and gitignored (`data/`). Everything in it is synthetic.

## Layout and label mapping

```
data/
  README (1).md      pack README: field meanings + labelling conventions
  schema.json        official output JSON Schema
  labelled/          L01..L10 docs + L01.json..L10.json
  unlablled/         U01..U10 docs (folder name is misspelled in the pack)
```

- **Mapping rule:** a label is `labelled/<stem>.json` for the doc `labelled/<stem>.<ext>` (e.g. `L02.jpg` ↔ `L02.json`).
  Inspection found 10/10 labels matched one-to-one, no duplicate stems, no unlabelled doc in `labelled/`.
- Code accepts both `unlablled/` (as shipped) and `unlabelled/`.
- This differs from the PRD's assumed `data/docs` + `data/labels`; `run_all.py` / `eval.py` defaults follow the real layout.
- Output stems (`L01`, `U05`, …) are unique across both folders, so `outputs/<stem>.json` has no collisions.

## Document inventory

| Doc | Type (magic bytes) | Pages / size | Text layer | Category | Invoice? |
|---|---|---|---|---|---|
| L01 | pdf | 1 p | yes | clean PDF | yes (GBP) |
| L02 | jpeg | 1500×2070 | – | phone photo (viewed) | yes (INR) |
| L03 | pdf | 1 p | yes | clean PDF | yes |
| L04 | pdf | **2 p** | yes | clean PDF, 24 line items | yes |
| L05 | pdf | 1 p | yes | clean PDF | yes |
| L06 | pdf | 1 p | yes | clean PDF | yes |
| L07 | pdf | 1 p | yes | clean PDF | yes (USD, has a discount row) |
| L08 | pdf | 1 p | yes | clean PDF | **no** – purchase order |
| L09 | pdf | 1 p | **none** | scan (image-only PDF) | yes |
| L10 | pdf | 1 p | **none** | scan (image-only PDF) | yes |
| U01 | pdf | 1 p | yes | clean PDF | yes (from text) |
| U02 | jpeg | 1050×1449 | – | phone photo, rotated (viewed) | yes (USD, US date format) |
| U03 | pdf | 1 p | yes | clean PDF | yes (from text) |
| U04 | pdf | 1 p | yes | clean PDF | yes (from text) |
| U05 | pdf | 1 p | **none** | scan (image-only PDF) | *guess: invoice – not viewed* |
| U06 | pdf | 1 p | yes | clean PDF | **no** – bank account statement |
| U07 | pdf | 1 p | yes | clean PDF, German ("Rechnung") | yes (from text) |
| U08 | pdf | 1 p | yes | clean PDF, Dutch vendor | yes (from text) |
| U09 | pdf | 1 p | yes | clean PDF | yes (from text) |
| U10 | pdf | 1 p | yes | clean PDF | **no** – job cover letter |

Totals: **12 clean PDFs, 5 scans/photos (L02, L09, L10, U02, U05), 3 non-invoices (L08, U06, U10)**. That matches the PRD's estimate.
"Scan" means the PDF has no extractable text (`text_chars=0`). "Invoice?" for unlabelled docs comes from the first lines of extracted text, so treat it as informed, not authoritative. All files are ≤ 2 pages and ≤ 530 KB.

Labelled eval set: 9 invoices + 1 non-invoice (L08). It covers 3 scans/photos (L02, L09, L10), so the image path is tested.

## Ground-truth conventions observed

From the pack README and all 10 label files:

1. **Non-invoice:** `is_invoice: false`, every other field is `null` (`line_items` is `null`, not `[]`). Same as the PRD.
2. **`tax_total` is `0` (not `null`) when the doc shows zero tax or no tax.** ⚠ The PRD says "`null` if none shown". **Align to the pack: use `0`.** (All 9 labelled invoices have tax > 0, so this only matters for unlabelled docs.)
3. **Line items:** in printed order. `description` as printed (always a string). `amount` = row total **before tax** (always a number). `quantity` = number printed, or `null` if the row has none (L06 has 2 such rows, L07 has 1).
   ⚠ The PRD's Pydantic model allows `description`/`amount` to be `null`; the official `schema.json` does not. **Align to schema.json:** `description: str`, `amount: float`, `quantity: float | None`.
4. **Charges and discounts printed as rows are line items. A discount is a negative amount** (L07: "Returning client discount (10%)", `-450.0`).
5. **Totals identity holds exactly in all 9 labelled invoices:** `sum(line amounts) + tax_total == total`. So the PRD's totals check is consistent with GT (discount rows included as negative lines).
6. **Quantities can be fractional** (L09: `12.5`, `9.5` kg).
7. **Numbers:** plain JSON floats, no separators or symbols. Docs print them as `8,700.00`, `₹19,610.00`.
8. **Dates:** ISO `YYYY-MM-DD`. Source formats vary: `19/08/2026` (DD/MM, Indian doc L02), `09/14/2026` (MM/DD, US doc U02), and German/Dutch docs probably use `DD.MM.YYYY`. The prompt must read the date in the context of the vendor's country.
9. **Vendor:** as printed at the top, legal suffix kept (`"Kaveri Office Mart Pvt Ltd"`, `"Meridian Print & Pack LLP"`). The eval normalises suffixes, so this convention doesn't affect scoring.
10. **Invoice number:** kept verbatim with separators (`KOM/26-27/00418`, `MAP/D/09-0312`).
11. **Currencies in the labelled set:** INR ×7, GBP ×1, USD ×1. The unlabelled set adds probably EUR (U07, U08).

## Decisions taken from this (convention alignment, not tuning)

- Schema: `LineItem.description: str`, `LineItem.amount: float`, `quantity: float | None`.
  Consequence: the PRD's `LINE_AMOUNT_MISSING` check becomes a `SCHEMA_ERROR` (a null amount can't pass the schema). I'll keep the check in `validate.py` anyway, in case the schema ever loosens.
- Prompt: `tax_total = 0` when no tax is shown; discount/charge rows go in `line_items` (negative for discounts).
- Data paths: `data/labelled` and `data/unlablled`, with `unlabelled` also accepted.
