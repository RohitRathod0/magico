"""Inject generated tables into REPORT.md and README.md (PRD §11: no hand-typed numbers).

Each block lives between markers:
    <!-- NAME -->
    ...generated...
    <!-- /NAME -->
and is rebuilt from files the code produced:
    reports/eval_results_v1.json, reports/eval_results_v2.json, reports/eval_results_text_only.json,
    reports/failures.md, outputs/*.json, outputs/meta/*.json, config/pricing.json,
    invoice_pipeline/matching.py (thresholds)

    python scripts/build_report_tables.py          # rewrite blocks in place
    python scripts/build_report_tables.py --check  # exit 1 if any block is out of date
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import TABLE_ROWS, pct  # noqa: E402
from invoice_pipeline import config, matching  # noqa: E402
from scripts.compare_modes import build as build_comparison  # noqa: E402

REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
PENDING = "_Pending: this run has not been completed yet. Rebuild with `python scripts/build_report_tables.py`._"
KIND = {"pdf_text": "digital PDF", "pdf_scan": "scanned PDF", "image": "photo (JPG)"}


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def eval_run(version: str) -> dict | None:
    """A completed eval run only (an LLM_UNAVAILABLE doc means the run is incomplete)."""
    r = load(REPORTS / f"eval_results_{version}.json")
    if r and any((d["pred"].get("error") or {}).get("code") == "LLM_UNAVAILABLE" for d in r["docs"]):
        return None
    return r


# ---------- blocks ----------

def accuracy_table(r: dict | None) -> str:
    if r is None:
        return PENDING
    s = r["summary"]
    li, nin = s["line_items_item_level"], s["non_invoice_nulls"]
    rows = ["| Field | Correct | N | Accuracy |", "|---|---|---|---|"]
    for f in TABLE_ROWS:
        x = s["fields"][f]
        name = "**doc exact match**" if f == "doc_exact" else f"`{f}`"
        rows.append(f"| {name} | {x['correct']} | {x['n']} | {pct(x['accuracy'])} |")
    rows += ["", f"- Line items, item level: precision {pct(li['precision'])}, recall {pct(li['recall'])}, "
                 f"F1 {pct(li['f1'])} ({li['matched']} of {li['n_gt']} GT rows matched; {li['n_pred']} predicted).",
             f"- `non_invoice_nulls` (non-invoices returned with every other field null): {nin['correct']}/{nin['n']}.",
             f"- Model `{r['model']}`, prompt `{r['prompt_version']}`, {len(r['docs'])} labelled docs. "
             f"Source: `reports/eval_results_{r['prompt_version']}.json`."]
    return "\n".join(rows)


def accuracy_compare(v1: dict | None, v2: dict | None) -> str:
    if v1 is None or v2 is None:
        return PENDING
    rows = ["| Field | v1 (untuned, headline) | v2 (tuned on the same docs) |", "|---|---|---|"]
    for f in TABLE_ROWS:
        a, b = v1["summary"]["fields"][f], v2["summary"]["fields"][f]
        rows.append(f"| {f} | {a['correct']}/{a['n']} ({pct(a['accuracy'])}) | "
                    f"{b['correct']}/{b['n']} ({pct(b['accuracy'])}) |")
    grid = ["", "Per document (✓ = every scored field correct):", "",
            "| Doc | Input | v1 | v2 |", "|---|---|---|---|"]
    b_docs = {d["doc"]: d for d in v2["docs"]}
    for d in v1["docs"]:
        def ok(doc):
            return all(v["correct"] for k, v in doc["fields"].items() if k != "doc_exact")
        grid.append(f"| {d['doc']} | {KIND.get(d['input_kind'], d['input_kind'])} | {'✓' if ok(d) else '✗'} | "
                    f"{'✓' if ok(b_docs[d['doc']]) else '✗'} |")
    return "\n".join(rows + grid)


def remaining_errors(r: dict | None) -> str:
    if r is None:
        return PENDING
    rows = []
    for d in r["docs"]:
        for f, v in d["fields"].items():
            if f in ("doc_exact",) or v["correct"]:
                continue
            gt, pred = d["gt"].get(f), d["pred"].get(f)
            if f == "line_items":
                det = v["details"]
                gt = "; ".join(str(d["gt"][f][i]) for i in det.get("unmatched_gt", [])) or f"{det['n_gt']} rows"
                pred = "; ".join(str(d["pred"][f][i]) for i in det.get("unmatched_pred", [])) or f"{det['n_pred']} rows"
            rows.append(f"| {d['doc']} | {f} | {gt} | {pred} | {'yes' if d['needs_review'] else 'no'} |")
    if not rows:
        return f"No wrong fields with prompt `{r['prompt_version']}`."
    return "\n".join([f"Wrong fields with prompt `{r['prompt_version']}`:", "",
                      "| Doc | Field | GT | Predicted | Flagged for review? |", "|---|---|---|---|---|"] + rows)


def headline(v1: dict | None) -> str:
    cost = cost_stats()
    if v1 is None:
        return PENDING
    s = v1["summary"]["fields"]
    parts = [f"**Headline (prompt v1, untuned):** {s['doc_exact']['correct']}/{s['doc_exact']['n']} labelled "
             f"invoices fully correct ({pct(s['doc_exact']['accuracy'])} doc exact match), "
             f"is_invoice {s['is_invoice']['correct']}/{s['is_invoice']['n']}."]
    v2 = eval_run("v2")
    if v2:
        t = v2["summary"]["fields"]["doc_exact"]
        parts.append(f"After one disclosed tuning round (v2): {t['correct']}/{t['n']} ({pct(t['accuracy'])}), "
                     "optimistic because it was tuned on the same documents.")
    if cost:
        parts.append(f"Average cost **${cost['avg']:.4f} per document** at Groq list price over "
                     f"{cost['n']} documents (actual spend $0 on the free plan).")
    return " ".join(parts)


def matching_rules() -> str:
    m = matching
    return "\n".join([
        "| Field | Rule |", "|---|---|",
        "| `is_invoice` | Exact boolean, scored over all labelled docs. |",
        "| all other fields | Scored only where GT `is_invoice` is true. For GT non-invoices, a separate "
        "`non_invoice_nulls` check asks that every other field is null. |",
        f"| `vendor` | Lowercase, `&`→`and`, strip punctuation, drop legal suffixes "
        f"({', '.join(sorted(m.LEGAL_SUFFIXES))}). Correct if equal, or rapidfuzz `token_set_ratio` "
        f"≥ {m.VENDOR_FUZZ_MIN}. |",
        "| `invoice_number` | Uppercase, drop non-alphanumerics, exact (leading zeros kept). |",
        "| `invoice_date` | Both parsed as ISO dates and equal; a non-ISO prediction is wrong. |",
        "| `currency` | Uppercase; symbols mapped (₹/Rs→INR, €→EUR, £→GBP, $→USD); exact. |",
        f"| `tax_total`, `total` | Coerced to float (commas and symbols stripped); correct if "
        f"\\|a−b\\| ≤ {m.AMOUNT_TOL}. Null matches only null (so 0 vs null is wrong). |",
        f"| `line_items` | Correct only if the counts are equal and every GT row is matched one-to-one "
        f"(Hungarian assignment). A row matches if description `token_set_ratio` ≥ {m.DESC_FUZZ_MIN}, "
        f"quantity is equal (null≡null), and amount is within {m.AMOUNT_TOL}. Item-level P/R/F1 is reported too. |",
        "| doc exact match | All 8 fields correct. |",
        "| null handling | GT null vs a value = `hallucinated`; GT value vs null = `missed`; both present but "
        "different = `wrong_value`. |",
    ])


def vision_vs_text() -> str:
    a, b = load(REPORTS / "eval_results_v1.json"), load(REPORTS / "eval_results_text_only.json")
    if a is None or b is None:
        return PENDING
    md = build_comparison(a, b)
    return md.split("\n", 2)[2].strip().replace("## ", "#### ")  # drop the title; demote headings


def failures() -> str:
    p = REPORTS / "failures.md"
    if not p.is_file():
        return PENDING
    body = p.read_text(encoding="utf-8").split("\n", 1)[1].strip()  # drop the H1
    return body.replace("## ", "#### ")


def outputs_with_meta() -> list[tuple[dict, dict]]:
    pairs = []
    for p in sorted(OUTPUTS.glob("*.json")):
        meta = load(OUTPUTS / "meta" / p.name)
        if meta:
            pairs.append((load(p), meta))
    return pairs


def cost_stats() -> dict | None:
    pairs = [(o, m) for o, m in outputs_with_meta() if m.get("cost_usd") is not None]
    if not pairs:
        return None
    costs = [m["cost_usd"] for _, m in pairs]
    return {"n": len(pairs), "avg": sum(costs) / len(pairs), "total": sum(costs), "pairs": pairs}


def doc_group(out: dict, meta: dict) -> str:
    if out.get("error"):
        return "rejected / error"
    if out.get("is_invoice") is False:
        return "non-invoice"
    return KIND.get(meta.get("input_kind"), "other") + " invoice"


def cost_block() -> str:
    st = cost_stats()
    if st is None:
        return PENDING
    pairs, n = st["pairs"], st["n"]
    pricing = config.load_pricing()
    tin = [m["tokens_in"] for _, m in pairs]
    tout = [m["tokens_out"] for _, m in pairs]
    tth = [m.get("tokens_thinking") or 0 for _, m in pairs]
    costs = [m["cost_usd"] for _, m in pairs]
    lo = min(pairs, key=lambda p: p[1]["cost_usd"])
    hi = max(pairs, key=lambda p: p[1]["cost_usd"])
    versions = sorted({m.get("prompt_version") for _, m in pairs})
    rows = [f"Over the {n} documents in `outputs/` (prompt {', '.join(versions)}; every attempt counted, "
            "including repair retries):", "",
            "| Metric | Value |", "|---|---|",
            f"| Average cost per document (list price) | **${st['avg']:.5f}** |",
            f"| Total for these {n} documents | ${st['total']:.4f} |",
            f"| Min / max per document | ${lo[1]['cost_usd']:.5f} ({Path(lo[1]['file']).stem}) / "
            f"${hi[1]['cost_usd']:.5f} ({Path(hi[1]['file']).stem}) |",
            f"| Average tokens in / out / thinking | {sum(tin) / n:.0f} / {sum(tout) / n:.0f} / {sum(tth) / n:.0f} |",
            f"| Actual spend | $0. Groq free plan (`billing_plan: {pricing.get('billing_plan')}`) |", "",
            "By document type:", "", "| Type | Docs | Avg cost | Avg tokens in | Avg attempts |", "|---|---|---|---|---|"]
    groups: dict[str, list] = {}
    for o, m in pairs:
        groups.setdefault(doc_group(o, m), []).append(m)
    for g in sorted(groups):
        ms = groups[g]
        rows.append(f"| {g} | {len(ms)} | ${sum(x['cost_usd'] for x in ms) / len(ms):.5f} | "
                    f"{sum(x['tokens_in'] for x in ms) / len(ms):.0f} | "
                    f"{sum(x['attempts'] for x in ms) / len(ms):.1f} |")
    rejected = [Path(m["file"]).stem for o, m in pairs if o.get("error") and m.get("attempts") == 0]
    rows += ["", f"Preflight-rejected documents (cost $0): {', '.join(rejected) or 'none in the sample pack'}.", "",
             "**Formula:** `cost = tokens_in × in_price / 1e6 + tokens_out × out_price / 1e6`, summed over all "
             "attempts. Token counts come from Groq's `usage` field, not estimates. `tokens_out` already includes "
             "any reasoning tokens (none were used).", "",
             f"**Prices:** ${pricing['usd_per_1m_input']:.2f} per 1M input tokens, ${pricing['usd_per_1m_output']:.2f} "
             f"per 1M output tokens for `{pricing['model']}`, from {pricing['source_url']}, checked "
             f"{pricing['checked_on']} and confirmed by the owner.", "", "Eval runs recorded in `reports/`:", "",
             "| Run | Docs | Tokens in | Tokens out | List-price cost |", "|---|---|---|---|---|"]
    for label, name in [("Eval, prompt v1 (vision)", "eval_results_v1.json"),
                        ("Eval, prompt v1 (text-only experiment)", "eval_results_text_only.json"),
                        ("Eval, prompt v2 (vision)", "eval_results_v2.json")]:
        r = load(REPORTS / name)
        if r:
            c = r["summary"]["cost"]
            rows.append(f"| {label} | {c['docs']} | {c['tokens_in']} | {c['tokens_out']} | ${c['cost_usd']:.4f} |")
    return "\n".join(rows)


def review_block() -> str:
    r = eval_run("v2") or load(REPORTS / "eval_results_v1.json")
    if r is None or "review_flag" not in r["summary"]:
        return PENDING
    rf = r["summary"]["review_flag"]
    return "\n".join([
        f"Prompt `{r['prompt_version']}`, {len(r['docs'])} labelled docs. A doc 'has an error' if any scored field "
        "is wrong.", "",
        "| | has ≥1 field error | no field error |", "|---|---|---|",
        f"| **flagged `needs_review`** | {rf['flagged_with_error']} | {rf['flagged_no_error']} |",
        f"| **not flagged** | {rf['unflagged_with_error']} | {rf['unflagged_no_error']} |", "",
        f"- Flag precision: {pct(rf['precision'])} · flag recall: {pct(rf['recall'])} · field errors inside "
        f"flagged docs: {rf['field_errors_in_flagged_docs']}/{rf['field_errors']} "
        f"({pct(rf['share_of_errors_flagged'])}).",
        f"- **Small sample:** only {rf['flagged_with_error'] + rf['unflagged_with_error']} of {len(r['docs'])} docs "
        "have any error, so one document moves these rates by tens of points."])


def outputs_block() -> str:
    pairs = outputs_with_meta()
    if not pairs:
        return PENDING
    rows = ["| Doc | Set | Input | is_invoice | Status | Review reasons |", "|---|---|---|---|---|---|"]
    for o, m in pairs:
        stem = Path(m["file"]).stem
        status = "error: " + o["error"]["code"] if o.get("error") else ("needs review" if o.get("needs_review") else "ok")
        rows.append(f"| [{stem}](outputs/{stem}.json) | {'labelled' if stem.startswith('L') else 'unlabelled'} | "
                    f"{KIND.get(m.get('input_kind'), m.get('input_kind'))} | {o.get('is_invoice')} | {status} | "
                    f"{'; '.join(r[:70] for r in o.get('review_reasons') or []) or '-'} |")
    return "\n".join(rows)


BLOCKS = {
    "HEADLINE": lambda: headline(load(REPORTS / "eval_results_v1.json")),
    "MATCHING_RULES": matching_rules,
    "ACCURACY_V1": lambda: accuracy_table(load(REPORTS / "eval_results_v1.json")),
    "ACCURACY_V2": lambda: accuracy_table(eval_run("v2")),
    "ACCURACY_COMPARE": lambda: accuracy_compare(load(REPORTS / "eval_results_v1.json"), eval_run("v2")),
    "REMAINING_ERRORS_V2": lambda: remaining_errors(eval_run("v2")),
    "VISION_VS_TEXT": vision_vs_text,
    "FAILURES": failures,
    "COST": cost_block,
    "REVIEW_FLAG": review_block,
    "OUTPUTS": outputs_block,
}
MARKER_RE = re.compile(r"(<!-- (\w+) -->)(.*?)(<!-- /\2 -->)", re.DOTALL)


def render(text: str) -> str:
    def sub(m):
        name = m.group(2)
        if name not in BLOCKS:
            raise SystemExit(f"Unknown marker {name}")
        return m.group(1) + "\n" + BLOCKS[name]().strip() + "\n" + m.group(4)
    return MARKER_RE.sub(sub, text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    stale = []
    for name in ("REPORT.md", "README.md"):
        p = ROOT / name
        if not p.is_file():
            continue
        old = p.read_text(encoding="utf-8")
        new = render(old)
        if new != old:
            stale.append(name)
            if not a.check:
                p.write_text(new, encoding="utf-8")
    print(("out of date: " if a.check else "updated: ") + (", ".join(stale) or "nothing"))
    return 1 if a.check and stale else 0


if __name__ == "__main__":
    sys.exit(main())
