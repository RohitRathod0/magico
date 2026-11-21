import json

import pytest

from invoice_pipeline.schema import InvoiceExtraction, LineItem
from invoice_pipeline.validate import check_totals, is_retryable, validate


def inv(lines, tax, total, **over):
    d = {"is_invoice": True, "vendor": "V", "invoice_number": "1", "invoice_date": "2026-01-01",
         "currency": "INR", "line_items": [{"description": f"i{n}", "quantity": 1, "amount": a}
                                           for n, a in enumerate(lines)],
         "tax_total": tax, "total": total} | over
    return d


def errs(d):
    return validate(json.dumps(d))[1]


def test_totals_exact_pass():
    assert errs(inv([100, 50], 27, 177)) == []


def test_totals_with_discount_line_pass():
    assert errs(inv([3200, 850, 450, -450], 334.13, 4384.13)) == []


def test_totals_mismatch_caught():
    e = errs(inv([100, 50], 27, 200))
    assert len(e) == 1 and e[0].startswith("TOTALS_MISMATCH") and is_retryable(e)


@pytest.mark.parametrize("total,ok", [
    (101.00, True), (99.00, True),        # small total: tolerance is 1.00
    (101.02, False), (98.98, False),
])
def test_tolerance_abs_edge(total, ok):
    # lines+tax = 100; tol = max(1.00, 0.005*total) = 1.00 near 100
    assert (errs(inv([100], 0, total)) == []) is ok


@pytest.mark.parametrize("total,ok", [(1005.0, True), (995.1, True), (1005.2, False), (994.9, False)])
def test_tolerance_relative_edge(total, ok):
    # lines+tax = 1000; tol = 0.005*total (5.025 at 1005, 4.9755 at 995.1, 4.9745 at 994.9)
    assert (errs(inv([1000], 0, total)) == []) is ok


def test_line_amount_missing_flagged():
    # The strict schema forbids a null amount, so build the object without validation.
    m = InvoiceExtraction.model_construct(
        is_invoice=True, vendor="V", invoice_number="1", invoice_date=None, currency=None,
        line_items=[LineItem.model_construct(description="a", quantity=1, amount=None),
                    LineItem.model_construct(description="b", quantity=1, amount=10.0)],
        tax_total=0, total=10.0)
    assert check_totals(m) == ["LINE_AMOUNT_MISSING"]


def test_non_invoice_skips_totals():
    d = {"is_invoice": False} | {k: None for k in ("vendor", "invoice_number", "invoice_date", "currency",
                                                    "line_items", "tax_total", "total")}
    assert errs(d) == []


def test_no_line_items_skips_totals():
    assert errs(inv([], 10, 999)) == []


def test_missing_total_and_vendor_flagged_not_retryable():
    e = errs(inv([100], 0, None, vendor=None))
    assert e == ["MISSING_TOTAL", "MISSING_VENDOR"] and not is_retryable(e)


def test_json_and_schema_errors():
    assert validate("{oops")[1][0].startswith("JSON_PARSE_ERROR")
    e = validate(json.dumps(inv([1], 0, 1, invoice_date="31/12/2026")))[1]
    assert e[0].startswith("SCHEMA_ERROR") and "invoice_date" in e[0] and is_retryable(e)
