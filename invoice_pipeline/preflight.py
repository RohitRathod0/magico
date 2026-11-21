"""Input file checks that run before any LLM call (PRD §5.3). A failed check costs $0.

Chosen extra bad-input case: corrupt / unreadable / unsupported files.
Checks run in this order; the first failure raises InputError(code, message).
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageStat, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from . import config

logging.getLogger("pypdf").setLevel(logging.ERROR)  # we report PDF problems ourselves

MAGIC = [
    (b"%PDF", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
]


class InputError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def sniff_mime(data: bytes) -> str | None:
    """MIME type from magic bytes (the extension is not trusted)."""
    for sig, mime in MAGIC:
        if data.startswith(sig):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _check_pdf(data: bytes) -> None:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise InputError("CORRUPT_PDF", "PDF is encrypted / password-protected")
        n = len(reader.pages)
    except InputError:
        raise
    except (PyPdfError, ValueError, KeyError, TypeError, OSError) as e:
        raise InputError("CORRUPT_PDF", f"PDF could not be opened: {type(e).__name__}: {e}") from None
    if n == 0:
        raise InputError("CORRUPT_PDF", "PDF has 0 pages")
    if n > config.MAX_PAGES:
        raise InputError("TOO_MANY_PAGES", f"PDF has {n} pages; the limit is {config.MAX_PAGES}")


def _check_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:  # verify() leaves the image unusable; reopen
            im.load()
            w, h = im.size
            stddev = ImageStat.Stat(im.convert("L")).stddev[0]
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise InputError("CORRUPT_IMAGE", f"Image could not be decoded: {type(e).__name__}: {e}") from None
    if min(w, h) < config.MIN_IMAGE_SHORT_SIDE:
        raise InputError("UNREADABLE_IMAGE",
                         f"Image is {w}x{h}; the short side must be at least {config.MIN_IMAGE_SHORT_SIDE}px")
    if stddev < config.MIN_IMAGE_STDDEV:
        raise InputError("UNREADABLE_IMAGE", f"Image is nearly blank (grayscale std-dev {stddev:.1f})")


def preflight(path: str | Path) -> tuple[bytes, str]:
    """Return (file bytes, mime) or raise InputError."""
    path = Path(path)
    if not path.is_file():
        raise InputError("FILE_NOT_FOUND", f"No such file: {path}")
    size = path.stat().st_size
    if size == 0:
        raise InputError("EMPTY_FILE", "File is 0 bytes")
    if size > config.MAX_FILE_BYTES:
        raise InputError("FILE_TOO_LARGE", f"File is {size / 1e6:.1f} MB; the limit is "
                                           f"{config.MAX_FILE_BYTES / 1e6:.0f} MB")
    data = path.read_bytes()
    mime = sniff_mime(data)
    if mime is None:
        raise InputError("UNSUPPORTED_TYPE", "File content is not PDF, PNG, JPEG or WEBP (the extension is ignored)")
    if mime == "application/pdf":
        _check_pdf(data)
    else:
        _check_image(data)
    return data, mime
