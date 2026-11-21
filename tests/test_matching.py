import pytest

import eval as ev
from invoice_pipeline.matching import (compare_amount, compare_currency, compare_date, compare_invoice_number,
                                       compare_is_invoice, compare_line_items, compare_vendor)


# --- required by the PRD ---

def test_amount_string_with_commas_matches_number():
    assert compare_amount(1200, "1,200.00").correct
    assert compare_amount(1200.0, "₹ 1,200").correct


def test_vendor_legal_suffix_ignored():
    assert compare_vendor("Acme Pvt Ltd", "Acme").correct
    assert compare_vendor("Meridian Print & Pack LLP", "Meridian Print and Pack").correct


def test_invoice_number_normalised():
    assert compare_invoice_number("INV-2041", "inv2041").correct
    assert compare_invoice_number("INV-2041", "#INV 2041").correct
    assert not compare_invoice_number("INV-02041", "INV-2041").correct  # leading zeros kept


def test_date_mismatch_caught():
    assert compare_date("2026-03-14", "2026-03-14").correct
    r = compare_date("2026-03-14", "2026-14-03")
    assert not r.correct and r.error_type == "wrong_value"
    assert not compare_date("2026-03-14", "14/03/2026").correct  # not ISO -> wrong


ITEMS = [{"description": "Brake pad set, front", "quantity": 8, "amount": 14800.0},
         {"description": "Air filter element", "quantity": 12, "amount": 5040.0},
         {"description": "Discount", "quantity": None, "amount": -450.0}]


def test_line_items_reordered_still_match():
    r = compare_line_items(ITEMS, list(reversed(ITEMS)))
    assert r.correct and r.details["matched"] == 3


def test_line_items_extra_item_wrong():
    r = compare_line_items(ITEMS, ITEMS + [{"description": "Subtotal", "quantity": None, "amount": 19390.0}])
    assert not r.correct and r.details["matched"] == 3 and r.details["n_pred"] == 4


def test_null_vs_zero_wrong():
    r = compare_amount(0, None)
    assert not r.correct and r.error_type == "missed"
    r = compare_amount(None, 0)
    assert not r.correct and r.error_type == "hallucinated"


# --- additional coverage ---

def test_line_item_amount_sign_and_qty():
    flipped = [dict(ITEMS[2], amount=450.0)]
    assert not compare_line_items([ITEMS[2]], flipped).correct
    assert not compare_line_items([ITEMS[0]], [dict(ITEMS[0], quantity=None)]).correct
    assert compare_line_items([ITEMS[0]], [dict(ITEMS[0], description="Brake pad set (front)")]).correct


def test_line_items_null_handling():
    assert compare_line_items(None, None).correct
    r = compare_line_items(ITEMS, None)
    assert r.error_type == "missed" and r.details["n_gt"] == 3


def test_currency_symbols_and_is_invoice():
    assert compare_currency("INR", "₹").correct and compare_currency("GBP", "gbp").correct
    assert not compare_currency("USD", "EUR").correct
    assert compare_is_invoice(False, False).correct
    assert compare_is_invoice(True, None).error_type == "missed"


def test_vendor_different_names_wrong():
    assert not compare_vendor("Kaveri Office Mart Pvt Ltd", "Sahyadri Coaching Classes").correct


GT_INV = {"is_invoice": True, "vendor": "Acme Ltd", "invoice_number": "A-1", "invoice_date": "2026-01-02",
          "currency": "INR", "line_items": ITEMS, "tax_total": 0, "total": 19390.0}
GT_NON = {"is_invoice": False} | {k: None for k in GT_INV if k != "is_invoice"}


def test_score_doc_exact_and_non_invoice():
    s = ev.score_doc(GT_INV, dict(GT_INV, vendor="ACME"))
    assert s["doc_exact"]["correct"] and "non_invoice_nulls" not in s
    s = ev.score_doc(GT_INV, dict(GT_INV, tax_total=None))
    assert not s["doc_exact"]["correct"] and s["tax_total"]["error_type"] == "missed"
    s = ev.score_doc(GT_NON, dict(GT_NON, vendor="X"))
    assert s["is_invoice"]["correct"] and not s["non_invoice_nulls"]["correct"] and "vendor" not in s


def test_aggregate_counts():
    docs = [{"fields": ev.score_doc(GT_INV, GT_INV), "meta": {"cost_usd": 0.002, "tokens_in": 10}},
            {"fields": ev.score_doc(GT_INV, dict(GT_INV, total=1.0)), "meta": {"cost_usd": 0.004, "tokens_in": 20}},
            {"fields": ev.score_doc(GT_NON, GT_NON), "meta": None}]
    a = ev.aggregate(docs)
    assert a["fields"]["is_invoice"] == {"correct": 3, "n": 3, "accuracy": 1.0, "error_types": {}}
    assert a["fields"]["total"]["correct"] == 1 and a["fields"]["total"]["n"] == 2
    assert a["fields"]["doc_exact"]["accuracy"] == 0.5
    assert a["non_invoice_nulls"] == {"correct": 1, "n": 1}
    assert a["line_items_item_level"]["f1"] == pytest.approx(1.0)
    assert a["cost"]["cost_usd"] == pytest.approx(0.006) and a["cost"]["docs"] == 2


def test_review_flag_metrics():
    ok = ev.score_doc(GT_INV, GT_INV)
    bad = ev.score_doc(GT_INV, dict(GT_INV, total=1.0, tax_total=5.0))
    docs = [{"fields": bad, "needs_review": True, "meta": None},    # flagged, has errors (2)
            {"fields": ok, "needs_review": True, "meta": None},     # flagged, no error
            {"fields": bad, "needs_review": False, "meta": None},   # missed (2 errors)
            {"fields": ok, "needs_review": False, "meta": None}]    # fine
    r = ev.review_flag_metrics(docs)
    assert (r["flagged_with_error"], r["flagged_no_error"], r["unflagged_with_error"], r["unflagged_no_error"]) \
        == (1, 1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
    assert r["field_errors"] == 4 and r["field_errors_in_flagged_docs"] == 2
