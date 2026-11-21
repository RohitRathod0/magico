"""Phase 1 smoke test: ONE real Groq call on one labelled doc.

Prints raw response text + usage, and checks that the text parses against the schema.
Usage: python scripts/smoke_llm.py [path] [--no-strict]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invoice_pipeline import config, llm  # noqa: E402
from invoice_pipeline.cost import usage_cost  # noqa: E402
from invoice_pipeline.render import to_images  # noqa: E402
from invoice_pipeline.schema import InvoiceExtraction  # noqa: E402

MIME = {".pdf": "application/pdf", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
PROMPT = ("Extract the invoice fields from the attached document image(s) into the given JSON schema. "
          "Use null for fields not on the document. If it is not an invoice, set is_invoice=false and all "
          "other fields null.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(config.LABELLED_DIR / "L01.pdf"))
    ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args()

    p = Path(a.path)
    images = to_images(p.read_bytes(), MIME[p.suffix.lower()])
    print(f"model={config.GROQ_MODEL} file={p.name} images={len(images)} "
          f"sizes_kb={[round(len(b) / 1024) for b, _ in images]}", file=sys.stderr)

    t = time.perf_counter()
    text, usage = llm.call_model(images, PROMPT, fmt=llm.response_format(strict=not a.no_strict))
    print(f"latency_s={time.perf_counter() - t:.2f}", file=sys.stderr)

    print("--- raw text ---")
    print(text)
    print("--- usage ---")
    print(json.dumps(usage), f"cost_usd={usage_cost(usage)}")
    try:
        InvoiceExtraction.model_validate_json(text)
        print("schema: OK")
    except Exception as e:
        print(f"schema: FAIL {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
