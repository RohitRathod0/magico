"""Turn an input file into the images sent to the model.

Groq vision accepts images only, not PDFs (PRD Amendment A1), so each PDF page
is rendered to PNG. JPEG/PNG/WEBP inputs pass through unchanged.
"""
from __future__ import annotations

import io

import pdfplumber
import pypdfium2 as pdfium

from . import config


def render_pdf(pdf_bytes: bytes, dpi: int = config.PDF_RENDER_DPI) -> list[tuple[bytes, str]]:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        out = []
        for page in pdf:
            img = page.render(scale=dpi / 72).to_pil()
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            out.append((buf.getvalue(), "image/png"))
        return out
    finally:
        pdf.close()


def to_images(data: bytes, mime: str) -> list[tuple[bytes, str]]:
    if mime == "application/pdf":
        return render_pdf(data)
    return [(data, mime)]


def extract_text(data: bytes, mime: str) -> str:
    """Text layer of a PDF via pdfplumber. Images (and scanned PDFs) give ''. Text mode only."""
    if mime != "application/pdf":
        return ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n\n".join((page.extract_text() or "") for page in pdf.pages).strip()
