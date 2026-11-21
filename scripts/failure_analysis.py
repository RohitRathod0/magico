"""Failure analysis (PRD §8): reports/eval_results.json + outputs/meta -> reports/failures.md

Every wrong field gets one cause category: automatic heuristics first, then manual
overrides (MANUAL_REVIEW below) where a human looked at the document. The report
marks which labels are manual and shows the evidence.

    python scripts/failure_analysis.py [--eval reports/eval_results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invoice_pipeline.schema import FIELDS  # noqa: E402

CATEGORIES = {
    "FORMAT_NORMALISATION": "Equal after looser normalisation: a matching-rule issue, not a model issue",
    "WRONG_ENTITY": "Right kind of value, wrong one (buyer as vendor, PO number, due date)",
    "OCR_READ_ERROR": "Scan/photo and the value is a near-miss (edit distance <= 2 or digit swap)",
    "LINE_ITEM_SEGMENTATION": "Line count differs (merged/split rows, subtotal rows captured)",
    "TAX_INTERPRETATION": "tax_total off by a component, or tax lines combined incorrectly",
    "TOTALS_SEMANTICS": "Discount / shipping / round-off / tax-inclusive handling (e.g. discount sign)",
    "MISSED_FIELD": "Predicted null, GT has a value",
    "HALLUCINATION": "Predicted a value, GT is null",
    "CLASSIFICATION": "is_invoice wrong",
    "GT_AMBIGUITY": "The ground truth itself is debatable",
    "UNCATEGORISED": "No heuristic matched and no manual review yet",
}

# Manual review: (doc, field) -> (category, evidence). Reviewed by looking at the rendered page.
MANUAL_REVIEW = {
    ("L02", "tax_total"): (
        "TAX_INTERPRETATION",
        "Photo prints CGST 1,764.90 and SGST 1,764.90 (sum 3,529.80), both clearly legible. Predicted 3,549.80: "
        "the line items and total were read correctly, so this is most likely an addition error when the model "
        "combined two tax lines, not a misread. (The heuristic alone would have said OCR_READ_ERROR.)"),
    ("L10", "tax_total"): (
        "TAX_INTERPRETATION",
        "Scan prints CGST 9,844.80 and SGST 9,844.80 (sum 19,689.60), clearly legible. Predicted 18,689.60, "
        "exactly 1,000 short: again the components are readable and the line items and total are right, so most "
        "likely an addition error. (The heuristic alone would have said OCR_READ_ERROR.)"),
}

# Fix-first candidates: category -> (fix, ease 1=hard..3=easy, how)
FIXES = {
    "TAX_INTERPRETATION": (
        "Ask the model for each tax line separately and add them up in code", 2,
        "Extend the internal LLM schema with tax_lines: [{label, amount}]; set tax_total = sum(tax_lines) in "
        "code. The output schema is unchanged. Removes the model's arithmetic from the loop."),
    "TOTALS_SEMANTICS": (
        "State the discount-sign rule more forcefully, and give the repair prompt the actual arithmetic", 3,
        "The v1 prompt already says discounts are negative, but the model still returned +450, and the generic "
        "repair message didn't fix it. Make the repair prompt say: lines sum to X, tax Y, total Z, gap D. "
        "Check signs of discount rows and re-add the tax lines."),
    "OCR_READ_ERROR": (
        "Straighten rotated or skewed scans before sending, and render at a higher DPI", 2,
        "Observed on unlabelled U05 (rotated 90°): money read correctly, header misread. Not in the labelled errors."),
    "LINE_ITEM_SEGMENTATION": ("Prompt: exclude subtotal/tax rows; one object per printed row", 3, ""),
    "WRONG_ENTITY": ("Prompt: stronger seller vs buyer and issue vs due date rules", 3, ""),
    "MISSED_FIELD": ("Inspect case by case", 1, ""),
    "HALLUCINATION": ("Inspect case by case", 1, ""),
    "CLASSIFICATION": ("Tighten the invoice definition in the prompt", 2, ""),
    "FORMAT_NORMALISATION": ("Adjust the matching rule (eval only)", 3, ""),
}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def digits(v) -> str:
    return f"{float(v):.2f}"


def heuristic(doc: dict, field: str, res: dict) -> tuple[str, str]:
    gt, pred = doc["gt"].get(field), doc["pred"].get(field)
    scan = doc["input_kind"] in ("pdf_scan", "image")
    if field == "is_invoice":
        return "CLASSIFICATION", f"GT {gt}, predicted {pred}"
    if res.get("error_type") == "missed":
        return "MISSED_FIELD", f"GT {gt!r}, predicted null"
    if res.get("error_type") == "hallucinated":
        return "HALLUCINATION", f"GT null, predicted {pred!r}"
    if field == "line_items":
        d = res["details"]
        if d["n_gt"] != d["n_pred"]:
            return "LINE_ITEM_SEGMENTATION", f"{d['n_gt']} GT rows vs {d['n_pred']} predicted"
        notes = []
        for i, j in zip(d["unmatched_gt"], d["unmatched_pred"]):
            g, p = gt[i], pred[j]
            if num(g["amount"]) is not None and num(p["amount"]) is not None \
                    and abs(num(g["amount"]) + num(p["amount"])) < 0.01 and num(g["amount"]) != 0:
                return "TOTALS_SEMANTICS", (f"row '{g['description']}': GT amount {g['amount']}, predicted "
                                            f"{p['amount']} (sign flipped)")
            notes.append(f"GT {g} vs pred {p}")
        if scan:
            return "OCR_READ_ERROR", "; ".join(notes)
        return "UNCATEGORISED", "; ".join(notes)
    if field in ("tax_total", "total"):
        g, p = num(gt), num(pred)
        if g and p and (abs(p * 2 - g) < 0.01 or abs(p - g * 2) < 0.01):
            return "TAX_INTERPRETATION" if field == "tax_total" else "TOTALS_SEMANTICS", f"GT {g}, predicted {p} (x2)"
        if scan and g is not None and p is not None and Levenshtein.distance(digits(g), digits(p)) <= 2:
            return "OCR_READ_ERROR", f"GT {g}, predicted {p} (edit distance {Levenshtein.distance(digits(g), digits(p))})"
        return ("TAX_INTERPRETATION" if field == "tax_total" else "TOTALS_SEMANTICS"), f"GT {g}, predicted {p}"
    if scan and isinstance(gt, str) and isinstance(pred, str) and Levenshtein.distance(gt, pred) <= 2:
        return "OCR_READ_ERROR", f"GT {gt!r}, predicted {pred!r}"
    return "UNCATEGORISED", f"GT {gt!r}, predicted {pred!r}"


def flagged_by_validation(doc: dict) -> tuple[bool, list[str]]:
    """Did any validation check fire on any attempt for this doc?"""
    meta = doc.get("meta") or {}
    attempts = meta.get("attempt_errors") or [doc["pred"].get("review_reasons") or []]
    fired = sorted({e.split(":")[0] for errs in attempts for e in errs})
    return bool(fired) or bool(doc["pred"].get("needs_review")), fired


def analyse(report: dict) -> dict:
    errors = []
    for doc in report["docs"]:
        for f in FIELDS:
            res = doc["fields"].get(f)
            if res is None or res["correct"]:
                continue
            if (doc["doc"], f) in MANUAL_REVIEW:
                cat, why = MANUAL_REVIEW[(doc["doc"], f)]
                manual = True
                auto_cat = heuristic(doc, f, res)[0]
            else:
                cat, why = heuristic(doc, f, res)
                manual, auto_cat = False, cat
            caught, fired = flagged_by_validation(doc)
            errors.append({"doc": doc["doc"], "field": f, "input_kind": doc["input_kind"], "category": cat,
                           "auto_category": auto_cat, "manual": manual, "why": why,
                           "gt": doc["gt"].get(f) if f != "line_items" else f"{len(doc['gt'][f] or [])} rows",
                           "pred": doc["pred"].get(f) if f != "line_items" else f"{len(doc['pred'][f] or [])} rows",
                           "caught_by_validation": caught, "checks_fired": fired})

    retries = []
    for doc in report["docs"]:
        ae = (doc.get("meta") or {}).get("attempt_errors") or []
        if len(ae) > 1:
            fixed = ae[-1] == []
            retries.append({"doc": doc["doc"], "first": ae[0], "fixed": fixed})

    err_docs = sorted({e["doc"] for e in errors})
    caught_docs = sorted({e["doc"] for e in errors if e["caught_by_validation"]})
    return {"errors": errors, "retries": retries, "err_docs": err_docs, "caught_docs": caught_docs}


def to_markdown(report: dict, a: dict) -> str:
    errs = a["errors"]
    counts = {}
    for e in errs:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    n_manual = sum(e["manual"] for e in errs)
    out = ["# Failure analysis", "",
           f"Source: `reports/eval_results.json` (prompt `{report['prompt_version']}`, model `{report['model']}`, "
           f"{len(report['docs'])} labelled docs) + `outputs/meta/`. Generated by `scripts/failure_analysis.py`.", "",
           f"**{len(errs)} wrong fields across {len(a['err_docs'])} docs.** Categories come from automatic "
           f"heuristics; **{n_manual} of {len(errs)} were changed after a manual look at the document** "
           "(marked *manual*, with the evidence). n = 10 docs, so treat every rate here as indicative only.", "",
           "## Errors by category", "", "| Category | Count | Meaning |", "|---|---|---|"]
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        out.append(f"| {cat} | {n} | {CATEGORIES[cat]} |")

    out += ["", "## Every error", "",
            "| Doc | Input | Field | GT | Predicted | Category | Caught by validation? |", "|---|---|---|---|---|---|---|"]
    for e in errs:
        cat = e["category"] + (f" *(manual; heuristic said {e['auto_category']})*" if e["manual"] else "")
        caught = ("yes: " + ", ".join(e["checks_fired"])) if e["caught_by_validation"] else "**no**"
        out.append(f"| {e['doc']} | {e['input_kind']} | {e['field']} | {e['gt']} | {e['pred']} | {cat} | {caught} |")

    out += ["", "## Examples and evidence", ""]
    for cat in sorted(counts, key=lambda c: -counts[c]):
        out.append(f"### {cat}")
        for e in [x for x in errs if x["category"] == cat][:3]:
            out.append(f"- **{e['doc']} · {e['field']}**: {e['why']}")
        out.append("")

    # validation recall
    n_err_fields = len(errs)
    caught_fields = sum(e["caught_by_validation"] for e in errs)
    out += ["## Did validation catch the errors?", "",
            f"- Field level: **{caught_fields}/{n_err_fields}** wrong fields were in a doc where a validation check "
            "fired (so it was retried and/or flagged).",
            f"- Doc level: **{len(a['caught_docs'])}/{len(a['err_docs'])}** docs with an error were caught "
            f"({', '.join(a['caught_docs']) or 'none'}); missed: "
            f"{', '.join(sorted(set(a['err_docs']) - set(a['caught_docs']))) or 'none'}."]
    missed = [e for e in errs if not e["caught_by_validation"]]
    for e in missed:
        if e["field"] in ("tax_total", "total") and isinstance(e["gt"], (int, float)):
            gap = abs(float(e["pred"]) - float(e["gt"]))
            total = report_doc_total(report, e["doc"])
            if total:
                out.append(f"- Why {e['doc']} slipped through: the error is {gap:.2f}, but the totals tolerance is "
                           f"max(1.00, 0.5% x {total:.2f}) = {max(1.0, 0.005 * total):.2f}. The tolerance is too "
                           "loose for large invoices.")
    out += ["", "## Did the repair retry help?", "", "| Doc | First-attempt errors | Fixed by retry? |", "|---|---|---|"]
    for r in a["retries"]:
        out.append(f"| {r['doc']} | {'; '.join(x[:80] for x in r['first'])} | {'yes' if r['fixed'] else 'no'} |")
    fixed = sum(r["fixed"] for r in a["retries"])
    out += ["", f"{fixed}/{len(a['retries'])} retries fixed the problem. The schema error (a non-invoice with a "
                "filled field) was fixed; the totals mismatches were not. The generic 'totals don't add up' message "
                "doesn't tell the model which number is wrong.", ""]

    # fix-first
    ranked = []
    for cat, n in counts.items():
        fix, ease, how = FIXES.get(cat, ("Inspect", 1, ""))
        ranked.append((n * ease, n, ease, cat, fix, how))
    ranked.sort(key=lambda r: (-r[0], -r[1]))
    out += ["## Fix first (ranked by errors removed x ease)", "",
            "| # | Fix | Category | Errors it targets | Ease (3 = easy) | Score |", "|---|---|---|---|---|---|"]
    for i, (score, n, ease, cat, fix, how) in enumerate(ranked, 1):
        out.append(f"| {i} | {fix} | {cat} | {n} | {ease} | {score} |")
    out.append("")
    for i, (score, n, ease, cat, fix, how) in enumerate(ranked, 1):
        if how:
            out.append(f"{i}. **{fix}.** {how}")
    out += [f"{len(ranked) + 1}. **Tighten the totals tolerance** (validation, not extraction). Use e.g. "
            "max(1.00, 0.05% of total), so a 20-rupee error on a 23,000-rupee invoice gets flagged. It doesn't "
            "fix errors, but it raises validation recall.", "",
            "Observed outside the labelled set (no GT, not counted above): U05 is a scan rotated 90°. The model read "
            "every amount correctly but returned invoice number `LP-10` (printed `LP-10932`) and date 2021-10-21 "
            "(printed 21 Oct 2026). The totals check cannot catch header errors like this.", ""]
    return "\n".join(out)


def report_doc_total(report: dict, doc_id: str) -> float | None:
    for d in report["docs"]:
        if d["doc"] == doc_id:
            return num(d["gt"].get("total"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "reports" / "eval_results.json"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "failures.md"))
    args = ap.parse_args()
    report = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    md = to_markdown(report, analyse(report))
    Path(args.out).write_text(md, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
