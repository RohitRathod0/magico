"""Validation (PRD §5.4). Every function returns a list of error strings; empty means pass.

Error codes:
  JSON_PARSE_ERROR, SCHEMA_ERROR, TOTALS_MISMATCH   -> trigger one repair retry
  LINE_AMOUNT_MISSING, MISSING_TOTAL, MISSING_VENDOR,
  KEY_FIELDS_MISSING (invoice with >= 2 of vendor/number/date/total null) -> flag for review only

needs_review is rule-based by design (cheap, explainable). A doc is flagged if any of these
remain after the repair retry, or if extraction failed (error envelope). A scan/photo with a
totals mismatch is already flagged by TOTALS_MISMATCH. Preflight warnings: none are defined
(preflight either passes or rejects), so nothing extra is needed for that trigger.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from . import config
from .schema import InvoiceExtraction, LLMExtraction

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
RETRYABLE = ("JSON_PARSE_ERROR", "SCHEMA_ERROR", "TOTALS_MISMATCH")


def parse_and_validate_schema(text: str) -> tuple[InvoiceExtraction | None, list[str]]:
    """Strip any reasoning text, parse JSON, validate against the schema, compute tax_total."""
    cleaned = THINK_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, [f"JSON_PARSE_ERROR: {e}"]
    try:
        # v2: the model returns tax_lines; tax_total is added up here, in code
        return LLMExtraction.model_validate(data).to_output(), []
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors())
        return None, [f"SCHEMA_ERROR: {msg}"]


def totals_tolerance(total: float) -> float:
    return max(config.TOTALS_ABS_TOL, config.TOTALS_REL_TOL * abs(total))


def check_totals(inv: InvoiceExtraction) -> list[str]:
    """sum(line amounts) + tax must equal total, within max(1.00, 0.05% of total)."""
    if not inv.is_invoice or not inv.line_items or inv.total is None:
        return []
    errors = []
    if any(i.amount is None for i in inv.line_items):  # unreachable with the strict schema; kept on purpose
        errors.append("LINE_AMOUNT_MISSING")
    lines = sum(i.amount for i in inv.line_items if i.amount is not None)
    tax = inv.tax_total or 0
    s = lines + tax
    if abs(s - inv.total) > totals_tolerance(inv.total) + 1e-9:
        # Spelled out so the repair prompt tells the model exactly what doesn't add up (v2).
        errors.append(f"TOTALS_MISMATCH: lines+tax={s:.2f}, total={inv.total:.2f} "
                      f"(line items sum to {lines:.2f}, tax lines sum to {tax:.2f}, gap {s - inv.total:+.2f})")
    return errors


def check_required(inv: InvoiceExtraction) -> list[str]:
    if not inv.is_invoice:
        return []
    errors = []
    if inv.total is None:
        errors.append("MISSING_TOTAL")
    if inv.vendor is None:
        errors.append("MISSING_VENDOR")
    # Review flag (PRD Phase 8): an "invoice" missing most of its key fields is suspicious
    # (possibly a misclassified non-invoice or an unreadable page).
    key = {"vendor": inv.vendor, "invoice_number": inv.invoice_number,
           "invoice_date": inv.invoice_date, "total": inv.total}
    missing = [k for k, v in key.items() if v is None]
    if len(missing) >= 2:
        errors.append(f"KEY_FIELDS_MISSING: {', '.join(missing)}")
    return errors


def validate(text: str) -> tuple[InvoiceExtraction | None, list[str]]:
    """Full check of one model response. Returns (model or None, errors)."""
    inv, errors = parse_and_validate_schema(text)
    if inv is None:
        return None, errors
    return inv, check_totals(inv) + check_required(inv)


def is_retryable(errors: list[str]) -> bool:
    return any(e.startswith(RETRYABLE) for e in errors)
