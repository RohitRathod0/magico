"""UI helper tests (no Streamlit server, no API)."""
import app


def test_preview_pdf(pdf_file, two_page_pdf):
    img, note = app.preview_image(pdf_file.read_bytes())
    assert img is not None and note is None
    img, note = app.preview_image(two_page_pdf.read_bytes())
    assert img is not None and note == "Page 1 of 2"


def test_preview_image(jpg_file):
    img, note = app.preview_image(jpg_file.read_bytes())
    assert img is not None and note is None


def test_preview_broken_files_do_not_raise(bad_files):
    for key in ("CORRUPT_PDF", "CORRUPT_IMAGE", "UNSUPPORTED_TYPE"):
        img, note = app.preview_image(bad_files[key].read_bytes())
        assert img is None and note


def test_status_and_tables():
    out = {"is_invoice": True, "vendor": "V", "invoice_number": "1", "invoice_date": "2026-01-01",
           "currency": "INR", "line_items": [{"description": "a", "quantity": None, "amount": 1.0}],
           "tax_total": 0.0, "total": 1.0, "needs_review": False, "review_reasons": []}
    assert app.status_of(out)[0] == "success"
    assert app.status_of(out | {"needs_review": True, "review_reasons": ["TOTALS_MISMATCH: x"]})[0] == "warning"
    assert app.status_of({"error": {"code": "CORRUPT_PDF", "message": "m"}})[0] == "error"
    assert list(app.header_table(out)["field"])[0] == "is_invoice"
    assert len(app.items_table(out)) == 1 and app.items_table(out | {"line_items": None}) is None


def test_run_extraction_corrupt_file_no_crash(fake_llm, bad_files):
    fake = fake_llm()
    out, meta = app.run_extraction.__wrapped__(bad_files["CORRUPT_PDF"].read_bytes(), "broken.pdf", "vision")
    assert out["error"]["code"] == "CORRUPT_PDF" and fake.calls == [] and meta["file"] == "broken.pdf"
