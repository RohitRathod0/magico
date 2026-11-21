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
