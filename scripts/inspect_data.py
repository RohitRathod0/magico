"""Phase 0: inspect the sample pack.

Lists every doc (name, magic-byte type, size, pages/resolution), shows which
docs have a label file, and prints two label files so GT conventions can be read.

Pack layout (as delivered):
  data/labelled/    L01..L10 docs + L01.json..L10.json ground truth (same stem)
  data/unlablled/   U01..U10 docs, no labels (folder name misspelled in the pack;
                    "unlabelled" is also accepted)

Usage: python scripts/inspect_data.py [--data data]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

MAGIC = [
    (b"%PDF", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
]


def sniff(path: Path) -> str:
    head = path.read_bytes()[:16]
    for sig, name in MAGIC:
        if head.startswith(sig):
            return name
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def describe(path: Path, kind: str) -> str:
    try:
        if kind == "pdf":
            r = PdfReader(str(path))
            enc = " encrypted" if r.is_encrypted else ""
            text_chars = sum(len(p.extract_text() or "") for p in r.pages)
            return f"{len(r.pages)} page(s){enc}, text_chars={text_chars}"
        if kind in ("png", "jpeg", "webp"):
            with Image.open(path) as im:
                return f"{im.width}x{im.height} {im.mode}"
    except Exception as e:  # inspection only; report and continue
        return f"ERROR: {type(e).__name__}: {e}"
    return "-"


LABELLED_DIR = "labelled"
UNLABELLED_DIRS = ("unlabelled", "unlablled")
LABEL_EXT = ".json"


def find_dirs(data: Path) -> tuple[Path, Path | None]:
    unl = next((data / d for d in UNLABELLED_DIRS if (data / d).is_dir()), None)
    return data / LABELLED_DIR, unl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    lab_dir, unl_dir = find_dirs(Path(args.data))
    if not lab_dir.is_dir() or unl_dir is None:
        print(f"Expected {lab_dir} and an unlabelled folder under {args.data}.", file=sys.stderr)
        return 1

    def doc_files(d: Path) -> list[Path]:
        return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() != LABEL_EXT)

    docs = doc_files(lab_dir) + doc_files(unl_dir)
    labels = sorted(lab_dir.glob("*" + LABEL_EXT))
    label_stems = {p.stem: p for p in labels}
    doc_stems = [p.stem for p in docs]

    print(f"{'doc':28} {'split':10} {'magic':7} {'ext':5} {'size_kb':>8}  {'label':5} details")
    for p in docs:
        kind = sniff(p)
        has_label = "yes" if p.stem in label_stems else "-"
        print(f"{p.parent.name + '/' + p.name:28} {'labelled' if p.parent == lab_dir else 'unlabelled':10} "
              f"{kind:7} {p.suffix:5} {p.stat().st_size / 1024:8.1f}  {has_label:5} {describe(p, kind)}")

    print(f"\nDocs: {len(docs)}   Labels: {len(labels)}")
    dupes = sorted({s for s in doc_stems if doc_stems.count(s) > 1})
    orphan = sorted(s for s in label_stems if s not in doc_stems)
    unlabelled_in_lab = sorted(p.name for p in doc_files(lab_dir) if p.stem not in label_stems)
    print(f"Labels matched to exactly one doc by stem: {len(labels) - len(orphan)}")
    for name, items in [("Duplicate doc stems", dupes), ("Labels with no doc", orphan),
                        ("Docs in labelled/ without a label", unlabelled_in_lab)]:
        if items:
            print(f"PROBLEM - {name}: {items}")

    print("\n--- Label summary ---")
    for p in labels:
        d = json.loads(p.read_text(encoding="utf-8"))
        items = d.get("line_items")
        n = len(items) if isinstance(items, list) else items
        qnull = sum(1 for i in items or [] if i.get("quantity") is None)
        neg = sum(1 for i in items or [] if (i.get("amount") or 0) < 0)
        check = ""
        if d.get("is_invoice") and items and d.get("total") is not None:
            s_ = sum(i["amount"] for i in items) + (d.get("tax_total") or 0)
            check = f" lines+tax={s_:.2f} total={d['total']} {'OK' if abs(s_ - d['total']) < 0.01 else 'MISMATCH'}"
        print(f"{p.name}: is_invoice={d.get('is_invoice')} cur={d.get('currency')} items={n} "
              f"qty_null={qnull} negative={neg} tax={d.get('tax_total')}{check}")

    for p in labels[:2]:
        print(f"\n--- {p.name} ---")
        print(p.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
