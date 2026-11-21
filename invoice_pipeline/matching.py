"""Field matching rules for the eval (PRD §6.1). Pure functions, no I/O.

Each compare returns a FieldResult: correct, error_type, normalised gt/pred.
error_type: None (correct) | "missed" (pred null, gt not) | "hallucinated" (gt null,
pred not) | "wrong_value" (both present, different).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

VENDOR_FUZZ_MIN = 90
DESC_FUZZ_MIN = 80
AMOUNT_TOL = 0.01
QTY_TOL = 1e-6
LEGAL_SUFFIXES = {"private", "limited", "pvt", "ltd", "llp", "llc", "inc", "corp", "corporation",
                  "co", "company", "gmbh", "plc"}
CURRENCY_SYMBOLS = {"₹": "INR", "RS": "INR", "RS.": "INR", "INR": "INR", "€": "EUR", "£": "GBP", "$": "USD"}


@dataclass
class FieldResult:
    correct: bool
    error_type: str | None
    gt_norm: object = None
    pred_norm: object = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _null_result(gt, pred) -> FieldResult | None:
    """Shared null handling: null matches only null."""
    if gt is None and pred is None:
        return FieldResult(True, None)
    if gt is None:
        return FieldResult(False, "hallucinated", None, pred)
    if pred is None:
        return FieldResult(False, "missed", gt, None)
    return None


def _result(ok: bool, g, p, **details) -> FieldResult:
    return FieldResult(ok, None if ok else "wrong_value", g, p, details)


# ---- normalisers ----

def _strip_punct(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def norm_vendor(s: str) -> str:
    s = _strip_punct(s.lower().replace("&", " and "))
    return " ".join(t for t in s.split() if t not in LEGAL_SUFFIXES)


def norm_invoice_number(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def norm_currency(s: str) -> str:
    s = str(s).strip().upper()
    return CURRENCY_SYMBOLS.get(s, s)


def to_float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("bool is not a number")
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[,\s₹$€£]|Rs\.?|INR", "", str(v), flags=re.IGNORECASE)
    return float(s)


def parse_iso_date(s) -> date | None:
    if not isinstance(s, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


# ---- field comparators ----

def compare_is_invoice(gt, pred) -> FieldResult:
    return FieldResult(gt is pred, None if gt is pred else ("missed" if pred is None else "wrong_value"), gt, pred)


def compare_vendor(gt, pred) -> FieldResult:
    if (r := _null_result(gt, pred)) is not None:
        return r
    g, p = norm_vendor(gt), norm_vendor(pred)
    score = fuzz.token_set_ratio(g, p)
    return _result(g == p or score >= VENDOR_FUZZ_MIN, g, p, fuzz=round(score, 1))


def compare_invoice_number(gt, pred) -> FieldResult:
    if (r := _null_result(gt, pred)) is not None:
        return r
    g, p = norm_invoice_number(gt), norm_invoice_number(pred)
    return _result(g == p, g, p)


def compare_date(gt, pred) -> FieldResult:
    if (r := _null_result(gt, pred)) is not None:
        return r
    g, p = parse_iso_date(gt), parse_iso_date(pred)
    return _result(g is not None and g == p, str(g), str(p) if p else f"not ISO: {pred!r}")


def compare_currency(gt, pred) -> FieldResult:
    if (r := _null_result(gt, pred)) is not None:
        return r
    g, p = norm_currency(gt), norm_currency(pred)
    return _result(g == p, g, p)


def compare_amount(gt, pred) -> FieldResult:
    if (r := _null_result(gt, pred)) is not None:
        return r
    try:
        g, p = to_float(gt), to_float(pred)
    except ValueError:
        return _result(False, gt, pred)
    return _result(abs(g - p) <= AMOUNT_TOL, g, p)


def _qty_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= QTY_TOL


def item_match(g: dict, p: dict) -> tuple[bool, float]:
    """(matches, description similarity) for one GT item vs one predicted item."""
    sim = fuzz.token_set_ratio(_strip_punct(g.get("description") or ""), _strip_punct(p.get("description") or ""))
    try:
        amt_ok = abs(to_float(g.get("amount")) - to_float(p.get("amount"))) <= AMOUNT_TOL
    except (TypeError, ValueError):
        amt_ok = False
    return sim >= DESC_FUZZ_MIN and _qty_equal(g.get("quantity"), p.get("quantity")) and amt_ok, sim


def compare_line_items(gt, pred) -> FieldResult:
    """Doc-level correct iff same count and every GT item is matched one-to-one.

    One-to-one assignment by the Hungarian algorithm: maximise full matches first,
    then description similarity. Also returns item-level matched/gt/pred counts for P/R/F1.
    """
    if (r := _null_result(gt, pred)) is not None:
        n_g, n_p = len(gt or []), len(pred or [])
        r.details = {"matched": 0, "n_gt": n_g, "n_pred": n_p}
        return r
    n_g, n_p = len(gt), len(pred)
    matched, pairs = 0, []
    if n_g and n_p:
        ok = np.zeros((n_g, n_p), dtype=bool)
        sim = np.zeros((n_g, n_p))
        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                ok[i, j], sim[i, j] = item_match(g, p)
        rows, cols = linear_sum_assignment(-(ok * 1000 + sim))
        for i, j in zip(rows, cols):
            pairs.append({"gt": int(i), "pred": int(j), "match": bool(ok[i, j]), "desc_sim": round(sim[i, j], 1)})
            matched += int(ok[i, j])
    correct = n_g == n_p and matched == n_g
    unmatched_gt = sorted(set(range(n_g)) - {p["gt"] for p in pairs if p["match"]})
    unmatched_pred = sorted(set(range(n_p)) - {p["pred"] for p in pairs if p["match"]})
    return FieldResult(correct, None if correct else "wrong_value", n_g, n_p,
                       {"matched": matched, "n_gt": n_g, "n_pred": n_p, "pairs": pairs,
                        "unmatched_gt": unmatched_gt, "unmatched_pred": unmatched_pred})


COMPARATORS = {
    "is_invoice": compare_is_invoice,
    "vendor": compare_vendor,
    "invoice_number": compare_invoice_number,
    "invoice_date": compare_date,
    "currency": compare_currency,
    "line_items": compare_line_items,
    "tax_total": compare_amount,
    "total": compare_amount,
}
