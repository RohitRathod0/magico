"""Output schema (PRD §4.1, aligned to the pack's schema.json; see docs/data_notes.md).

Differences from the PRD draft, taken from the pack's conventions:
- LineItem.description and LineItem.amount are required (non-null); quantity may be null.
- tax_total is 0 (not null) when an invoice shows no tax. That's a prompt rule, not a schema rule.
"""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
FIELDS = ("is_invoice", "vendor", "invoice_number", "invoice_date", "currency",
          "line_items", "tax_total", "total")


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    quantity: float | None
    amount: float  # line total before tax, NOT unit price; discounts negative


class InvoiceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_invoice: bool
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None  # YYYY-MM-DD, a real calendar date
    currency: str | None      # ISO 4217, 3 uppercase letters
    line_items: list[LineItem] | None
    tax_total: float | None
    total: float | None

    @field_validator("invoice_date")
    @classmethod
    def _check_date(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not DATE_RE.match(v):
            raise ValueError(f"invoice_date must be YYYY-MM-DD, got {v!r}")
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"invoice_date is not a real calendar date: {v!r}") from None
        return v

    @field_validator("currency")
    @classmethod
    def _check_currency(cls, v: str | None) -> str | None:
        if v is not None and not CURRENCY_RE.match(v):
            raise ValueError(f"currency must be a 3-letter uppercase ISO 4217 code, got {v!r}")
        return v

    @model_validator(mode="after")
    def _non_invoice_all_null(self) -> "InvoiceExtraction":
        if self.is_invoice is False:
            filled = [f for f in FIELDS[1:] if getattr(self, f) is not None]
            if filled:
                raise ValueError(f"is_invoice is false but these fields are not null: {filled}")
        return self


def response_json_schema() -> dict:
    """JSON Schema for the LLM's response_format: $refs inlined, titles dropped.

    Cross-field rules (non-invoice -> nulls, real dates) can't be expressed here,
    so we always re-validate with InvoiceExtraction.
    """
    raw = InvoiceExtraction.model_json_schema()
    defs = raw.pop("$defs", {})

    def clean(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return clean(defs[node["$ref"].split("/")[-1]])
            return {k: clean(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    return clean(raw)
