"""Prompts. v1 was written before looking at any eval errors (PRD §6.3).

Field rules follow PRD §5.2 plus the pack's published labelling conventions
(docs/data_notes.md). No examples are taken from the labelled set.
"""
from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You extract data from business documents. The attached image(s) are the pages of ONE document, in order.
Return a single JSON object that matches the given schema. Output JSON only.

Step 1 - Is it an invoice?
An invoice is a bill from a seller asking the buyer for payment (invoice, tax invoice, bill, "Rechnung", "Factuur").
NOT invoices: purchase orders, quotes/estimates, bank or account statements, delivery notes, payment receipts only, letters, CVs, other documents.
If it is not an invoice: is_invoice=false and every other field null (line_items null, not []).

Step 2 - If it is an invoice, fill the fields:
- vendor: the seller/issuer name as printed at the top. NOT the buyer / "Bill To" / "Billed to" party.
- invoice_number: the invoice's own number, as printed (keep its separators). NOT the PO, order, GSTIN, VAT or tax ID number.
- invoice_date: the issue date (not the due, order or PO date), as YYYY-MM-DD. For dates like 03/04/2026, decide day-first vs month-first from the vendor's country (India, UK and Europe are day-first; US is month-first) and other dates on the page.
- currency: ISO 4217 code from symbols, words or context: Rs/INR/₹ -> INR, € -> EUR, £ -> GBP, $ -> USD unless the context shows another dollar (CAD, AUD, SGD...).
- line_items: one object per row of the item table, in printed order.
  - description: the row's description as printed.
  - quantity: the number printed in the row's quantity column, or null if the row has none.
  - amount: the row total BEFORE tax (qty x rate after any line discount), NOT the unit price.
  - Charges or discounts printed as rows of the item table (shipping, fees, discounts) are line items too. A discount is a negative amount.
  - Do NOT include subtotal, tax or total rows as line items.
- tax_total: sum of ALL tax lines (CGST+SGST+IGST, VAT/USt/BTW, GST, sales tax). If the invoice shows no tax or zero tax, use 0.
- total: the final amount payable (grand total, including tax).

Rules:
- Numbers are plain JSON numbers: no thousands separators, no currency symbols. "1.234,56" in European style means 1234.56.
- Never guess. If a field is not on the document, use null."""

REPAIR_TEMPLATE = """{system}

Your previous output failed these checks:
{errors}
Re-read the document and return corrected JSON."""


# Text-only mode (the "thing that didn't work" experiment): same rules, the document
# arrives as text extracted from the PDF instead of page images.
_VISION_INTRO = "The attached image(s) are the pages of ONE document, in order."
_TEXT_INTRO = ("The text below was extracted from ONE document (layout may be lost; it may be empty "
               "if the document had no text layer).")
assert _VISION_INTRO in SYSTEM_PROMPT


def text_mode_prompt(doc_text: str) -> str:
    return (SYSTEM_PROMPT.replace(_VISION_INTRO, _TEXT_INTRO)
            + f"\n\nDOCUMENT TEXT:\n<<<\n{doc_text}\n>>>")


def repair_prompt(errors: list[str], system: str = SYSTEM_PROMPT) -> str:
    return REPAIR_TEMPLATE.format(system=system, errors="\n".join(f"- {e}" for e in errors))
