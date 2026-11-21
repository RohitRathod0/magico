"""Eval harness: score the 10 labelled docs field by field against ground truth.

    python eval.py                 # runs the extractor on the labelled docs (writes outputs/), then scores
    python eval.py --use-cached    # scores the existing outputs/ without calling the API

Writes reports/eval_results.json (per-doc, per-field detail) and
reports/eval_results_<prompt_version>.md (the tables). Matching rules: invoice_pipeline/matching.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from invoice_pipeline import config
from invoice_pipeline.dataset import label_path, labelled_docs
from invoice_pipeline.matching import COMPARATORS
from invoice_pipeline.prompt import PROMPT_VERSION
from invoice_pipeline.schema import FIELDS

console = Console()
NON_INVOICE_FIELDS = FIELDS[1:]
TABLE_ROWS = list(FIELDS) + ["doc_exact"]


# ---------- scoring ----------

def score_doc(gt: dict, pred: dict) -> dict:
    """Per-field results for one doc. Fields other than is_invoice are scored only if GT is an invoice."""
    res = {"is_invoice": COMPARATORS["is_invoice"](gt["is_invoice"], pred.get("is_invoice")).to_dict()}
    if gt["is_invoice"]:
        for f in NON_INVOICE_FIELDS:
            res[f] = COMPARATORS[f](gt[f], pred.get(f)).to_dict()
        res["doc_exact"] = {"correct": all(res[f]["correct"] for f in FIELDS)}
    else:
        filled = [f for f in NON_INVOICE_FIELDS if pred.get(f) is not None]
        res["non_invoice_nulls"] = {"correct": not filled, "filled_fields": filled}
    return res


def aggregate(docs: list[dict]) -> dict:
    rows = {}
    for f in TABLE_ROWS:
        scored = [d for d in docs if f in d["fields"]]
        correct = sum(d["fields"][f]["correct"] for d in scored)
        types = {}
        for d in scored:
            et = d["fields"][f].get("error_type")
            if et:
                types[et] = types.get(et, 0) + 1
        rows[f] = {"correct": correct, "n": len(scored),
                   "accuracy": correct / len(scored) if scored else None, "error_types": types}

    m = g = p = 0
    for d in docs:
        li = d["fields"].get("line_items")
        if li:
            m += li["details"].get("matched", 0)
            g += li["details"].get("n_gt", 0)
            p += li["details"].get("n_pred", 0)
    prec = m / p if p else 0.0
    rec = m / g if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    nin = [d for d in docs if "non_invoice_nulls" in d["fields"]]
    metas = [d["meta"] for d in docs if d["meta"]]
    cost = sum(mt.get("cost_usd") or 0 for mt in metas)
    return {
        "fields": rows,
        "line_items_item_level": {"matched": m, "n_gt": g, "n_pred": p,
                                  "precision": prec, "recall": rec, "f1": f1},
        "non_invoice_nulls": {"correct": sum(d["fields"]["non_invoice_nulls"]["correct"] for d in nin),
                              "n": len(nin)},
        "cost": {"docs": len(metas), "tokens_in": sum(mt.get("tokens_in") or 0 for mt in metas),
                 "tokens_out": sum(mt.get("tokens_out") or 0 for mt in metas),
                 "tokens_thinking": sum(mt.get("tokens_thinking") or 0 for mt in metas),
                 "cost_usd": cost, "avg_cost_usd": cost / len(metas) if metas else None,
                 "billing_plan": config.load_pricing().get("billing_plan")},
        "review_flag": review_flag_metrics(docs),
    }


def field_errors(fields: dict) -> int:
    """Wrong scored fields in one doc (doc_exact excluded; non_invoice_nulls counts as one)."""
    return sum(1 for k, v in fields.items() if k != "doc_exact" and not v["correct"])


def review_flag_metrics(docs: list[dict]) -> dict:
    """needs_review vs 'doc has >= 1 field error' (PRD Phase 8)."""
    tp = fp = fn = tn = 0
    err_total = err_in_flagged = 0
    for d in docs:
        n_err = field_errors(d["fields"])
        flagged = bool(d.get("needs_review"))
        err_total += n_err
        err_in_flagged += n_err if flagged else 0
        if flagged and n_err:
            tp += 1
        elif flagged:
            fp += 1
        elif n_err:
            fn += 1
        else:
            tn += 1
    return {"flagged_with_error": tp, "flagged_no_error": fp, "unflagged_with_error": fn,
            "unflagged_no_error": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "field_errors": err_total, "field_errors_in_flagged_docs": err_in_flagged,
            "share_of_errors_flagged": err_in_flagged / err_total if err_total else None}


# ---------- running ----------

def run_extractor(docs: list[Path], out_dir: Path, mode: str = "vision") -> None:
    from run_all import DailyQuotaExhausted, TokenPacer, process_doc

    pacer = TokenPacer(6000)
    for doc in docs:
        try:
            process_doc(doc, out_dir, pacer, mode)
        except DailyQuotaExhausted as e:
            raise SystemExit(
                f"Stopped at {doc.stem}: Groq's daily token quota is used up, so this eval would be incomplete. "
                f"Wait for the quota to refill, then finish with `python run_all.py --only <stems> --force` "
                f"and score with `python eval.py --use-cached`.\n{str(e)[:300]}") from None


def load_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def evaluate(data_dir: Path, out_dir: Path, use_cached: bool, mode: str = "vision") -> dict:
    docs = [d for d in labelled_docs(data_dir) if label_path(d)]
    if not docs:
        raise SystemExit(f"No labelled documents found under {data_dir}")
    if not use_cached:
        run_extractor(docs, out_dir, mode)

    results = []
    for doc in docs:
        gt = load_json(label_path(doc))
        pred = load_json(out_dir / f"{doc.stem}.json")
        meta = load_json(out_dir / "meta" / f"{doc.stem}.json")
        if pred is None:
            raise SystemExit(f"Missing output for {doc.stem}; run without --use-cached or run run_all.py")
        results.append({"doc": doc.stem, "input_kind": (meta or {}).get("input_kind"),
                        "needs_review": pred.get("needs_review"), "review_reasons": pred.get("review_reasons"),
                        "error": pred.get("error"), "gt": gt, "pred": pred,
                        "meta": meta, "fields": score_doc(gt, pred)})
    model = next((r["meta"]["model"] for r in results if r["meta"]), config.GROQ_MODEL)
    pv = next((r["meta"]["prompt_version"] for r in results if r["meta"]), PROMPT_VERSION)
    return {"model": model, "prompt_version": pv, "mode": mode, "summary": aggregate(results), "docs": results}


# ---------- presentation ----------

def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def mark(fields: dict, f: str) -> str:
    return "–" if f not in fields else ("✓" if fields[f]["correct"] else "✗")


def grid_columns() -> list[str]:
    return TABLE_ROWS + ["non_invoice_nulls"]


def to_markdown(report: dict, title: str) -> str:
    s = report["summary"]
    c = s["cost"]
    li = s["line_items_item_level"]
    out = [f"# {title}", "",
           f"Model `{report['model']}` · prompt `{report['prompt_version']}` · "
           f"input mode `{report.get('mode', 'vision')}` · "
           f"{len(report['docs'])} labelled docs (generated by `eval.py`; do not edit by hand)", "",
           "## Field accuracy", "", "| Field | Correct | N | Accuracy | missed | hallucinated | wrong_value |",
           "|---|---|---|---|---|---|---|"]
    for f in TABLE_ROWS:
        r = s["fields"][f]
        et = r["error_types"]
        out.append(f"| {f} | {r['correct']} | {r['n']} | {pct(r['accuracy'])} | {et.get('missed', 0)} | "
                   f"{et.get('hallucinated', 0)} | {et.get('wrong_value', 0)} |")
    nin = s["non_invoice_nulls"]
    out += ["", f"- **non_invoice_nulls** (GT non-invoices with every other field null): "
                f"{nin['correct']}/{nin['n']}",
            f"- **Line items, item level:** precision {pct(li['precision'])}, recall {pct(li['recall'])}, "
            f"F1 {pct(li['f1'])} ({li['matched']} matched / {li['n_gt']} GT / {li['n_pred']} predicted)", "",
            "## Per-document grid", "", "✓ correct · ✗ wrong · – not scored (GT is not an invoice)", "",
            "| Doc | Input | " + " | ".join(grid_columns()) + " | needs_review |",
            "|" + "---|" * (len(grid_columns()) + 3)]
    for d in report["docs"]:
        out.append(f"| {d['doc']} | {d['input_kind'] or '-'} | "
                   + " | ".join(mark(d["fields"], f) for f in grid_columns())
                   + f" | {'yes' if d['needs_review'] else 'no'} |")
    plan = " (actual spend $0, Groq free plan)" if c["billing_plan"] == "free" else ""
    out += ["", "## Cost (this eval run)", "",
            f"- Docs: {c['docs']} · tokens in {c['tokens_in']} · out {c['tokens_out']} · "
            f"thinking {c['tokens_thinking']}",
            f"- List-price cost: total ${c['cost_usd']:.4f} · avg/doc ${c['avg_cost_usd']:.5f}{plan}", ""]
    rf = s.get("review_flag")
    if rf:
        out += ["## Human-review flag (needs_review) vs actual errors", "",
                "Does `needs_review` point at the docs that really have a wrong field? "
                "(A doc 'has an error' if any scored field is wrong.)", "",
                "| | has ≥1 field error | no field error |", "|---|---|---|",
                f"| **flagged** | {rf['flagged_with_error']} | {rf['flagged_no_error']} |",
                f"| **not flagged** | {rf['unflagged_with_error']} | {rf['unflagged_no_error']} |", "",
                f"- Flag precision (flagged docs that really had an error): {pct(rf['precision'])}",
                f"- Flag recall (docs with an error that were flagged): {pct(rf['recall'])}",
                f"- Field errors sitting in flagged docs: {rf['field_errors_in_flagged_docs']}/{rf['field_errors']} "
                f"({pct(rf['share_of_errors_flagged'])})",
                f"- **Caveat:** n = {len(report['docs'])} docs, with only {rf['flagged_with_error'] + rf['unflagged_with_error']} "
                "containing an error. One doc moves these rates by 10–50 points, so read them as indicative.", ""]
    return "\n".join(out)


def print_console(report: dict) -> None:
    s = report["summary"]
    t = Table(title=f"Field accuracy · prompt {report['prompt_version']}")
    for col in ("Field", "Correct", "N", "Accuracy"):
        t.add_column(col, justify="left" if col == "Field" else "right")
    for f in TABLE_ROWS:
        r = s["fields"][f]
        t.add_row(f, str(r["correct"]), str(r["n"]), pct(r["accuracy"]))
    console.print(t)

    g = Table(title="Per-document grid (✓ correct, ✗ wrong, – not scored)")
    g.add_column("doc")
    g.add_column("input")
    short = {"is_invoice": "inv?", "invoice_number": "inv_no", "invoice_date": "date", "currency": "cur",
             "line_items": "lines", "tax_total": "tax", "doc_exact": "exact", "non_invoice_nulls": "nulls"}
    for f in grid_columns():
        g.add_column(short.get(f, f), justify="center")
    g.add_column("review", justify="center")
    for d in report["docs"]:
        g.add_row(d["doc"], d["input_kind"] or "-", *[mark(d["fields"], f) for f in grid_columns()],
                  "yes" if d["needs_review"] else "")
    console.print(g)

    li, nin, c = s["line_items_item_level"], s["non_invoice_nulls"], s["cost"]
    console.print(f"line items (item level): P={pct(li['precision'])} R={pct(li['recall'])} F1={pct(li['f1'])}"
                  f"   non_invoice_nulls: {nin['correct']}/{nin['n']}")
    plan = "  (actual spend $0, Groq free plan)" if c["billing_plan"] == "free" else ""
    console.print(f"tokens in={c['tokens_in']} out={c['tokens_out']} thinking={c['tokens_thinking']} · "
                  f"list-price cost ${c['cost_usd']:.4f}, avg/doc ${c['avg_cost_usd']:.5f}{plan} · "
                  f"model {report['model']} · prompt {report['prompt_version']}")
    rf = s.get("review_flag")
    if rf:
        console.print(f"needs_review: flagged&error={rf['flagged_with_error']} flagged&ok={rf['flagged_no_error']} "
                      f"missed={rf['unflagged_with_error']} ok={rf['unflagged_no_error']} · "
                      f"precision {pct(rf['precision'])}, recall {pct(rf['recall'])}, "
                      f"errors in flagged docs {rf['field_errors_in_flagged_docs']}/{rf['field_errors']} "
                      f"(small sample, n={len(report['docs'])})")


def rescore(json_path: Path) -> int:
    """Recompute scores and summary for a saved run (same gt/pred, current matching + metrics)."""
    report = json.loads(json_path.read_text(encoding="utf-8"))
    for d in report["docs"]:
        d["fields"] = score_doc(d["gt"], d["pred"])
    report["summary"] = aggregate(report["docs"])
    print_console(report)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    mode = report.get("mode", "vision")
    title = f"Eval results: prompt {report['prompt_version']}" + (" (TEXT-ONLY input)" if mode == "text" else "")
    md_path.write_text(to_markdown(report, title), encoding="utf-8")
    console.print(f"[dim]rescored {json_path} and wrote {md_path}[/dim]")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score extractions of the labelled docs against ground truth.")
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--outputs", default=str(config.OUTPUTS_DIR), help="where extractions are read/written")
    ap.add_argument("--use-cached", action="store_true", help="score existing outputs, no API calls")
    ap.add_argument("--out", help="markdown path (default reports/eval_results_<prompt_version>.md); "
                                  "the JSON is written next to it with the same name")
    ap.add_argument("--mode", choices=["vision", "text"], default="vision",
                    help="text = PDF text layer only (experiment); its outputs go to reports/text_only_outputs/")
    ap.add_argument("--rescore", metavar="EVAL_JSON",
                    help="re-score a saved eval JSON from its stored gt/pred (no API, no outputs/ needed) and "
                         "rewrite it plus its .md, e.g. reports/eval_results_v1.json")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    if a.rescore:
        return rescore(Path(a.rescore))

    outputs = Path(a.outputs)
    if a.mode == "text" and a.outputs == str(config.OUTPUTS_DIR):
        outputs = config.REPORTS_DIR / "text_only_outputs"  # never overwrite the vision outputs/
    report = evaluate(Path(a.data), outputs, a.use_cached, a.mode)
    print_console(report)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    if a.out:
        md_path = Path(a.out)
        json_path = md_path.with_suffix(".json")
    else:
        md_path = config.REPORTS_DIR / f"eval_results_{report['prompt_version']}.md"
        json_path = config.REPORTS_DIR / "eval_results.json"
    payload = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    json_path.write_text(payload, encoding="utf-8")
    if not a.out:  # keep a per-version copy so later prompt versions never overwrite earlier results
        (config.REPORTS_DIR / f"eval_results_{report['prompt_version']}.json").write_text(payload, encoding="utf-8")
    title = f"Eval results: prompt {report['prompt_version']}" + (" (TEXT-ONLY input)" if a.mode == "text" else "")
    md_path.write_text(to_markdown(report, title), encoding="utf-8")
    console.print(f"[dim]wrote {json_path} and {md_path}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
