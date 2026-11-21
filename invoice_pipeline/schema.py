"""Output schema (PRD §4.1, aligned to the pack's schema.json; see docs/data_notes.md).

Differences from the PRD draft, taken from the pack's conventions:
- LineItem.description and LineItem.amount are required (non-null); quantity may be null.
- tax_total is 0 (not null) when an invoice shows no tax.

Two models:
- InvoiceExtraction: the OUTPUT schema (exactly the pack's fields).
- LLMExtraction (prompt v2): what we ask the model for. Same fields, except the model
  lists each printed tax line (tax_lines) instead of adding them up; the code computes
  tax_total = sum(tax_lines). Failure analysis on v1 showed the model mis-adding tax lines.
"""
from __future__ import annotations

import re
from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
FIELDS = ("is_invoice", "vendor", "invoice_number", "invoice_date", "currency",
          "line_items", "tax_total", "total")
LLM_FIELDS = ("is_invoice", "vendor", "invoice_number", "invoice_date", "currency",
              "line_items", "tax_lines", "total")


def _check_date(v: str | None) -> str | None:
    if v is None:
        return v
    if not DATE_RE.match(v):
        raise ValueError(f"invoice_date must be YYYY-MM-DD, got {v!r}")
    try:
        date.fromisoformat(v)
    except ValueError:
        raise ValueError(f"invoice_date is not a real calendar date: {v!r}") from None
    return v


def _check_currency(v: str | None) -> str | None:
    if v is not None and not CURRENCY_RE.match(v):
        raise ValueError(f"currency must be a 3-letter uppercase ISO 4217 code, got {v!r}")
    return v


def _check_non_invoice(obj, fields) -> None:
    if obj.is_invoice is False:
        filled = [f for f in fields[1:] if getattr(obj, f) is not None]
        if filled:
            raise ValueError(f"is_invoice is false but these fields are not null: {filled}")


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
    def _date(cls, v):
        return _check_date(v)

    @field_validator("currency")
    @classmethod
    def _currency(cls, v):
        return _check_currency(v)

    @model_validator(mode="after")
    def _non_invoice_all_null(self) -> "InvoiceExtraction":
        _check_non_invoice(self, FIELDS)
        return self


class TaxLine(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str     # as printed, e.g. "CGST @ 9%"
    amount: float


class LLMExtraction(BaseModel):
    """What the model returns (prompt v2). Converted to InvoiceExtraction by to_output()."""
    model_config = ConfigDict(extra="forbid")
    is_invoice: bool
    vendor: str | None
    invoice_number: str | None
    invoice_date: str | None
    currency: str | None
    line_items: list[LineItem] | None
    tax_lines: list[TaxLine] | None  # every tax line printed; [] if none
    total: float | None

    @field_validator("invoice_date")
    @classmethod
    def _date(cls, v):
        return _check_date(v)

    @field_validator("currency")
    @classmethod
    def _currency(cls, v):
        return _check_currency(v)

    @model_validator(mode="after")
    def _non_invoice_all_null(self) -> "LLMExtraction":
        _check_non_invoice(self, LLM_FIELDS)
        return self

    def to_output(self) -> InvoiceExtraction:
        """tax_total = sum of tax lines (0 when an invoice shows none, per pack convention)."""
        tax_total = round(sum(t.amount for t in self.tax_lines or []), 2) if self.is_invoice else None
        data = self.model_dump(exclude={"tax_lines"}) | {"tax_total": tax_total}
        return InvoiceExtraction.model_validate(data)


def response_json_schema(model: type[BaseModel] = LLMExtraction) -> dict:
    """JSON Schema for the LLM's response_format: $refs inlined, titles dropped.

    Cross-field rules (non-invoice -> nulls, real dates) can't be expressed here,
    so we always re-validate with Pydantic.
    """
    raw = model.model_json_schema()
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
