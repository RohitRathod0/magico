import pytest
from pydantic import ValidationError

from invoice_pipeline.schema import InvoiceExtraction, response_json_schema


def valid_invoice(**over):
    d = {
        "is_invoice": True,
        "vendor": "Acme Pvt Ltd",
        "invoice_number": "INV-2041",
        "invoice_date": "2026-03-14",
        "currency": "INR",
        "line_items": [
            {"description": "Widget", "quantity": 2, "amount": 200.0},
            {"description": "Discount", "quantity": None, "amount": -20.0},
        ],
        "tax_total": 32.4,
        "total": 212.4,
    }
    d.update(over)
    return d


NON_INVOICE = {k: None for k in ("vendor", "invoice_number", "invoice_date", "currency",
                                 "line_items", "tax_total", "total")} | {"is_invoice": False}


def test_valid_invoice_passes():
    m = InvoiceExtraction.model_validate(valid_invoice())
    assert m.line_items[1].amount == -20.0


def test_valid_non_invoice_passes():
    assert InvoiceExtraction.model_validate(NON_INVOICE).is_invoice is False


@pytest.mark.parametrize("bad", ["2026-02-30", "14/03/2026", "2026-3-14", "March 14 2026"])
def test_bad_date_fails(bad):
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(valid_invoice(invoice_date=bad))


@pytest.mark.parametrize("bad", ["Rs", "inr", "₹", "INRS"])
def test_bad_currency_fails(bad):
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(valid_invoice(currency=bad))


@pytest.mark.parametrize("field,value", [("vendor", "Acme"), ("line_items", []), ("tax_total", 0)])
def test_non_invoice_with_non_null_field_fails(field, value):
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(NON_INVOICE | {field: value})


def test_extra_key_fails():
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(valid_invoice(po_number="PO-1"))


def test_extra_key_in_line_item_fails():
    items = [{"description": "x", "quantity": 1, "amount": 1.0, "unit_price": 1.0}]
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(valid_invoice(line_items=items))


def test_line_item_null_amount_fails():
    items = [{"description": "x", "quantity": 1, "amount": None}]
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(valid_invoice(line_items=items))


def test_missing_field_fails():
    d = valid_invoice()
    del d["total"]
    with pytest.raises(ValidationError):
        InvoiceExtraction.model_validate(d)


def test_response_schema_is_self_contained():
    s = response_json_schema()
    text = str(s)
    assert "$ref" not in text and "$defs" not in text
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])
