"""Fixtures generated at test time (no committed binaries, no sample-pack data)."""
from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

from invoice_pipeline import llm


def _page(w=800, h=1100) -> Image.Image:
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((50, 50), "INVOICE  No. T-1  Total 118.00", fill="black")
    for y in range(150, 600, 40):
        d.line((50, y, 750, y), fill="black")
    return img


@pytest.fixture
def pdf_file(tmp_path):
    p = tmp_path / "clean.pdf"
    _page().save(p, format="PDF")
    return p


@pytest.fixture
def two_page_pdf(tmp_path):
    p = tmp_path / "two.pdf"
    _page().save(p, format="PDF", save_all=True, append_images=[_page()])
    return p


@pytest.fixture
def jpg_file(tmp_path):
    p = tmp_path / "photo.jpg"
    _page().rotate(4, expand=True, fillcolor="gray").save(p, format="JPEG")
    return p


def _png(tmp_path, name, img):
    p = tmp_path / name
    img.save(p, format="PNG")
    return p


@pytest.fixture
def bad_files(tmp_path):
    """Bad inputs from PRD Phase 3, keyed by the error code each should produce."""
    from pypdf import PdfReader, PdfWriter

    good_pdf = tmp_path / "good.pdf"
    _page().save(good_pdf, format="PDF")

    truncated = tmp_path / "truncated.pdf"
    truncated.write_bytes(good_pdf.read_bytes()[:200])

    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    text_as_pdf = tmp_path / "notes.pdf"
    text_as_pdf.write_text("this is just a text file pretending to be a PDF")

    w = PdfWriter()
    w.append(PdfReader(good_pdf))
    w.encrypt("secret")
    encrypted = tmp_path / "locked.pdf"
    with open(encrypted, "wb") as f:
        w.write(f)

    four_pages = tmp_path / "four.pdf"
    _page().save(four_pages, format="PDF", save_all=True, append_images=[_page() for _ in range(3)])

    corrupt_jpg = tmp_path / "broken.jpg"
    corrupt_jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00garbage" * 50)

    return {
        "FILE_NOT_FOUND": tmp_path / "does_not_exist.pdf",
        "EMPTY_FILE": empty,
        "UNSUPPORTED_TYPE": text_as_pdf,
        "CORRUPT_PDF": truncated,
        "CORRUPT_PDF_encrypted": encrypted,
        "TOO_MANY_PAGES": four_pages,
        "CORRUPT_IMAGE": corrupt_jpg,
        "UNREADABLE_IMAGE": _png(tmp_path, "tiny.png", _page(50, 50)),
        "UNREADABLE_IMAGE_blank": _png(tmp_path, "blank.png", Image.new("RGB", (800, 1100), "white")),
    }


INVOICE_JSON = {
    "is_invoice": True, "vendor": "Test Traders", "invoice_number": "T-1",
    "invoice_date": "2026-05-01", "currency": "INR",
    "line_items": [{"description": "Item A", "quantity": 2, "amount": 100.0}],
    "tax_total": 18.0, "total": 118.0,
}
NON_INVOICE_JSON = {"is_invoice": False} | {k: None for k in (
    "vendor", "invoice_number", "invoice_date", "currency", "line_items", "tax_total", "total")}
USAGE = {"tokens_in": 1800, "tokens_out": 150, "tokens_thinking": 0}


class FakeLLM:
    """Stands in for llm.call_model. Replies are returned in order; an Exception reply is raised."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, images, prompt, **kw):
        self.calls.append({"images": images, "prompt": prompt})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return (r if isinstance(r, str) else json.dumps(r)), dict(USAGE)


@pytest.fixture
def fake_llm(monkeypatch):
    def install(*replies):
        fake = FakeLLM(replies)
        monkeypatch.setattr(llm, "call_model", fake)
        return fake
    return install
