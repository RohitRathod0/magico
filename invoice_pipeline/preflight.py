"""Input file checks that run before any LLM call (PRD §5.3).

Phase 2: file type sniffing only. The full check list arrives in Phase 3.
"""
from __future__ import annotations

MAGIC = [
    (b"%PDF", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
]


def sniff_mime(data: bytes) -> str | None:
    """MIME type from magic bytes (the extension is not trusted)."""
    for sig, mime in MAGIC:
        if data.startswith(sig):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
