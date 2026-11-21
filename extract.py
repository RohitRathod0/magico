"""CLI: extract one document and print the JSON to stdout.

    python extract.py data/labelled/L01.pdf > out.json
    python extract.py <file> --meta      # also print run metadata to stderr

Exit codes: 0 ok | 2 ok but needs_review | 3 input error | 4 LLM error
"""
from __future__ import annotations

import argparse
import json
import sys

from invoice_pipeline.pipeline import extract_document

LLM_ERRORS = {"LLM_UNAVAILABLE", "SCHEMA_ERROR"}


def exit_code(output: dict) -> int:
    err = output.get("error")
    if err:
        return 4 if err["code"] in LLM_ERRORS else 3
    return 2 if output.get("needs_review") else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract invoice fields from a PDF or image.")
    ap.add_argument("path")
    ap.add_argument("--meta", action="store_true", help="print run metadata to stderr")
    ap.add_argument("--mode", choices=["vision", "text"], default="vision",
                    help="vision (default): page images; text: PDF text layer only (experiment)")
    a = ap.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    output, meta = extract_document(a.path, mode=a.mode)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if a.meta:
        print(json.dumps(meta, indent=2), file=sys.stderr)
    return exit_code(output)


if __name__ == "__main__":
    sys.exit(main())
