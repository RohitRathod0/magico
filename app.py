"""Bonus UI: upload a PDF or image, see the page next to the extracted fields.

    streamlit run app.py

Calls the same pipeline as the CLI (invoice_pipeline.pipeline.extract_document).
Extraction only runs when you click the button, and results are cached per file,
so re-rendering the page doesn't spend tokens.
"""
from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image

from invoice_pipeline import config
from invoice_pipeline.pipeline import extract_document
from invoice_pipeline.preflight import sniff_mime

HEADER_FIELDS = ("is_invoice", "vendor", "invoice_number", "invoice_date", "currency", "tax_total", "total")


# ---------- helpers (plain functions, unit-tested without Streamlit) ----------

def preview_image(data: bytes) -> tuple[Image.Image | None, str | None]:
    """First page as an image, or (None, reason) if it can't be drawn. Never raises."""
    mime = sniff_mime(data)
    try:
        if mime == "application/pdf":
            pdf = pdfium.PdfDocument(data)
            try:
                n = len(pdf)
                img = pdf[0].render(scale=100 / 72).to_pil()
            finally:
                pdf.close()
            return img, (f"Page 1 of {n}" if n > 1 else None)
        if mime in ("image/png", "image/jpeg", "image/webp"):
            img = Image.open(io.BytesIO(data))
            img.load()
            return img, None
        return None, "Not a PDF or image (by content), so there is nothing to preview."
    except Exception as e:  # broken file: the pipeline will report the exact error code
        return None, f"Preview unavailable: {type(e).__name__}"


def header_table(out: dict) -> pd.DataFrame:
    return pd.DataFrame({"field": HEADER_FIELDS, "value": ["" if out.get(f) is None else str(out.get(f))
                                                           for f in HEADER_FIELDS]})


def items_table(out: dict) -> pd.DataFrame | None:
    items = out.get("line_items")
    return pd.DataFrame(items, columns=["description", "quantity", "amount"]) if items else None


def status_of(out: dict) -> tuple[str, str]:
    """(level, message) where level is one of error / warning / success."""
    if out.get("error"):
        return "error", f"{out['error']['code']}: {out['error']['message']}"
    if out.get("needs_review"):
        return "warning", "Needs human review: " + "; ".join(out.get("review_reasons") or [])
    if out.get("is_invoice") is False:
        return "success", "Not an invoice. All other fields are null."
    return "success", "All checks passed (schema, totals, required fields)."


@st.cache_data(show_spinner=False, max_entries=32)
def run_extraction(data: bytes, filename: str, mode: str) -> tuple[dict, dict]:
    """Write to a temp file (the pipeline takes a path) and extract. Cached by file content."""
    suffix = Path(filename).suffix or ".bin"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"upload{suffix}"
        p.write_bytes(data)
        out, meta = extract_document(p, mode=mode)
    meta["file"] = filename
    return out, meta


# ---------- page ----------

def main() -> None:
    st.set_page_config(page_title="Invoice extractor", layout="wide")
    st.title("Invoice extractor")
    st.caption(f"Model `{config.GROQ_MODEL}` on Groq · upload a PDF (up to {config.MAX_PAGES} pages) "
               "or a JPG/PNG/WEBP image")

    up = st.file_uploader("Document", type=None, accept_multiple_files=False)
    if up is None:
        st.info("Upload a file to start. The file type is detected from its content, not its extension.")
        return
    data = up.getvalue()
    key = hashlib.sha256(data).hexdigest()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Preview")
        img, note = preview_image(data)
        if img is not None:
            st.image(img, use_container_width=True)
        if note:
            st.caption(note)

    with right:
        st.subheader("Extraction")
        mode = st.radio("Input to the model", ["vision", "text"], horizontal=True,
                        help="vision = page images (default). text = PDF text layer only (the experiment that "
                             "fails on scans and photos).")
        if st.button("Extract", type="primary") or st.session_state.get("done") == (key, mode):
            with st.spinner("Calling the model..."):
                out, meta = run_extraction(data, up.name, mode)
            st.session_state["done"] = (key, mode)

            level, msg = status_of(out)
            getattr(st, level)(msg)

            st.dataframe(header_table(out), hide_index=True, use_container_width=True)
            items = items_table(out)
            if items is not None:
                st.markdown("**Line items**")
                st.dataframe(items, hide_index=True, use_container_width=True)

            cost = meta.get("cost_usd")
            plan = config.load_pricing().get("billing_plan")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cost (list price)", f"${cost:.5f}" if cost is not None else "-",
                      help="Groq free plan: actual spend $0" if plan == "free" else None)
            c2.metric("Tokens in / out", f"{meta['tokens_in']} / {meta['tokens_out']}")
            c3.metric("Attempts", meta["attempts"])
            c4.metric("Latency", f"{meta['latency_s']}s")
            with st.expander("Raw JSON output and run metadata"):
                st.json(out)
                st.json(meta)


if __name__ == "__main__":
    main()
