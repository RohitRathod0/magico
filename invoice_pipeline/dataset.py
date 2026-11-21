"""Locate sample-pack documents and labels (layout in docs/data_notes.md)."""
from __future__ import annotations

from pathlib import Path

from . import config


def unlabelled_dir(data_dir: Path = config.DATA_DIR) -> Path | None:
    for name in ("unlabelled", "unlablled"):  # the pack ships the misspelling
        if (data_dir / name).is_dir():
            return data_dir / name
    return None


def _docs_in(d: Path | None) -> list[Path]:
    if d is None or not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() != ".json")


def labelled_docs(data_dir: Path = config.DATA_DIR) -> list[Path]:
    return _docs_in(data_dir / "labelled")


def all_docs(data_dir: Path = config.DATA_DIR) -> list[Path]:
    return labelled_docs(data_dir) + _docs_in(unlabelled_dir(data_dir))


def label_path(doc: Path) -> Path | None:
    p = doc.with_suffix(".json")
    return p if p.is_file() else None
