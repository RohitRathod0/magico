"""Run the extractor on every sample document and write outputs/.

    python run_all.py            # skip docs that already have outputs/<stem>.json
    python run_all.py --force    # re-run everything
    python run_all.py --only L01 U05

Writes outputs/<stem>.json (extraction + needs_review + review_reasons [+ error])
and outputs/meta/<stem>.json (tokens, cost, attempts, validation). Prints a summary.

Pacing: Groq's free plan allows 7,000 input tokens per minute. Before each doc we
wait until the last 60 s of usage plus this doc's estimate fits under --itpm
(default 6,000: headroom for unpaced repair retries and Groq's own token count).
Transport retries in llm.py remain the backstop if the estimate is off.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import deque
from pathlib import Path

from pypdf import PdfReader
from rich.console import Console
from rich.table import Table

from invoice_pipeline import config
from invoice_pipeline.dataset import all_docs
from invoice_pipeline.llm import is_daily_limit
from invoice_pipeline.pipeline import extract_document

EST_TOKENS_PER_PAGE = 2500  # observed ~2,390 per rendered page, plus the prompt
console = Console(stderr=True)


def estimate_tokens(path: Path) -> int:
    try:
        if path.read_bytes()[:4] == b"%PDF":
            return EST_TOKENS_PER_PAGE * max(1, len(PdfReader(io.BytesIO(path.read_bytes())).pages))
    except Exception:
        pass
    return EST_TOKENS_PER_PAGE


class TokenPacer:
    """Sliding 60-second window of input tokens."""

    def __init__(self, itpm: int, sleep=time.sleep, clock=time.monotonic):
        self.itpm, self.sleep, self.clock = itpm, sleep, clock
        self.window: deque[tuple[float, int]] = deque()

    def _used(self, now: float) -> int:
        while self.window and now - self.window[0][0] >= 60:
            self.window.popleft()
        return sum(t for _, t in self.window)

    def wait_for(self, tokens: int) -> float:
        if self.itpm <= 0:
            return 0.0
        waited = 0.0
        while self.window and self._used(self.clock()) + tokens > self.itpm:
            secs = 60 - (self.clock() - self.window[0][0]) + 0.5
            console.print(f"[dim]  pacing: waiting {secs:.0f}s for the token-per-minute limit[/dim]")
            self.sleep(secs)
            waited += secs
        return waited

    def record(self, tokens: int) -> None:
        if tokens:
            self.window.append((self.clock(), tokens))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def status_of(out: dict) -> str:
    if out.get("error"):
        return "error"
    return "needs_review" if out.get("needs_review") else "ok"


def was_unavailable(out: dict) -> bool:
    """The saved output is incomplete because the model was unreachable (first call or repair call)."""
    if (out.get("error") or {}).get("code") == "LLM_UNAVAILABLE":
        return True
    return any(r.startswith("LLM_UNAVAILABLE") for r in out.get("review_reasons") or [])


class DailyQuotaExhausted(Exception):
    pass


def process_doc(doc: Path, out_dir: Path, pacer: TokenPacer, mode: str = "vision") -> tuple[dict, dict]:
    """Extract one doc and save it. Shared by run_all and eval.

    An LLM_UNAVAILABLE result never overwrites an existing good output, and the daily-quota
    case raises DailyQuotaExhausted so the caller stops instead of failing every remaining doc.
    """
    out_path, meta_path = out_dir / f"{doc.stem}.json", out_dir / "meta" / f"{doc.stem}.json"
    pacer.wait_for(estimate_tokens(doc))
    console.print(f"[bold]{doc.parent.name}/{doc.name}[/bold] ...", highlight=False)
    out, meta = extract_document(doc, mode=mode)
    pacer.record(meta["tokens_in"])
    err = out.get("error") or {}
    if err.get("code") == "LLM_UNAVAILABLE":
        old = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else None
        if old and not old.get("error"):
            console.print(f"[yellow]  {doc.stem}: model unavailable; keeping the previous output[/yellow]")
        else:
            write_json(out_path, out)
            write_json(meta_path, meta)
        if is_daily_limit(err.get("message", "")):
            raise DailyQuotaExhausted(err["message"])
        return out, meta
    write_json(out_path, out)
    write_json(meta_path, meta)
    return out, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.OUTPUTS_DIR))
    ap.add_argument("--force", action="store_true", help="re-run docs that already have outputs")
    ap.add_argument("--only", nargs="*", help="doc stems to run, e.g. L01 U05")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run only docs whose saved output is incomplete because the model was unavailable")
    ap.add_argument("--itpm", type=int, default=6000, help="input tokens/minute budget (0 = no pacing)")
    a = ap.parse_args(argv)

    out_dir = Path(a.out)
    docs = all_docs(Path(a.data))
    if a.only:
        docs = [d for d in docs if d.stem in set(a.only)]
    if not docs:
        console.print(f"[red]No documents found under {a.data}[/red]")
        return 1

    pacer = TokenPacer(a.itpm)
    rows = []
    quota_msg = None
    for doc in docs:
        out_path, meta_path = out_dir / f"{doc.stem}.json", out_dir / "meta" / f"{doc.stem}.json"
        cached = None
        if out_path.exists() and meta_path.exists():
            cached = (json.loads(out_path.read_text(encoding="utf-8")),
                      json.loads(meta_path.read_text(encoding="utf-8")))
        unavailable = bool(cached) and was_unavailable(cached[0])
        skip = cached and not a.force and not (a.retry_errors and unavailable)
        if skip or quota_msg:
            if cached:
                rows.append((doc, *cached, "cached"))
            continue
        try:
            out, meta = process_doc(doc, out_dir, pacer)
        except DailyQuotaExhausted as e:
            quota_msg = str(e)
            out, meta = (json.loads(out_path.read_text(encoding="utf-8")),
                         json.loads(meta_path.read_text(encoding="utf-8")))
        rows.append((doc, out, meta, "run"))
    if quota_msg:
        console.print(f"[red]Stopped early: Groq's daily token quota is used up. Remaining docs were "
                      f"not run. Re-run later with --retry-errors (or --force).[/red]\n[dim]{quota_msg[:300]}[/dim]")

    # ---- summary ----
    t = Table(title="run_all summary")
    for col in ("doc", "kind", "status", "attempts", "tok in", "tok out", "cost $", "reasons"):
        t.add_column(col)
    for doc, out, meta, src in rows:
        t.add_row(doc.stem, meta.get("input_kind") or "-", status_of(out) + (" (cached)" if src == "cached" else ""),
                  str(meta["attempts"]), str(meta["tokens_in"]), str(meta["tokens_out"]),
                  f"{meta['cost_usd']:.5f}" if meta["cost_usd"] is not None else "-",
                  "; ".join(r[:60] for r in out.get("review_reasons", [])))
    console.print(t)

    counts = {s: sum(1 for _, o, _, _ in rows if status_of(o) == s) for s in ("ok", "needs_review", "error")}
    tin = sum(m["tokens_in"] for _, _, m, _ in rows)
    tout = sum(m["tokens_out"] for _, _, m, _ in rows)
    cost = sum(m["cost_usd"] or 0 for _, _, m, _ in rows)
    plan = config.load_pricing().get("billing_plan")
    console.print(f"docs={len(rows)}  ok={counts['ok']}  needs_review={counts['needs_review']}  "
                  f"errors={counts['error']}")
    console.print(f"tokens in={tin} out={tout}  list-price cost: total=${cost:.4f} avg/doc=${cost / len(rows):.5f}"
                  + ("  (actual spend $0: Groq free plan)" if plan == "free" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
