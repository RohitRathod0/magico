"""Pipeline tests with a mocked LLM (no network)."""
import json

import pytest

import extract
from invoice_pipeline.pipeline import extract_document, parse_response
from tests.conftest import INVOICE_JSON, NON_INVOICE_JSON

PRICING_COST = 1800 * 0.80 / 1e6 + 150 * 4.00 / 1e6


def test_clean_pdf_invoice(fake_llm, pdf_file):
    fake = fake_llm(INVOICE_JSON)
    out, meta = extract_document(pdf_file)
    assert out["vendor"] == "Test Traders" and out["needs_review"] is False and out["review_reasons"] == []
    # PDF was rendered to one PNG page before the call
    assert [m for _, m in fake.calls[0]["images"]] == ["image/png"]
    assert meta["attempts"] == 1 and meta["validation"]["schema_ok"] is True
    assert meta["tokens_in"] == 1800 and meta["cost_usd"] == pytest.approx(PRICING_COST)


def test_multipage_pdf_sends_all_pages(fake_llm, two_page_pdf):
    fake = fake_llm(INVOICE_JSON)
    extract_document(two_page_pdf)
    assert len(fake.calls[0]["images"]) == 2


def test_scan_image_passed_through(fake_llm, jpg_file):
    fake = fake_llm(INVOICE_JSON)
    out, _ = extract_document(jpg_file)
    assert out["is_invoice"] is True
    assert [m for _, m in fake.calls[0]["images"]] == ["image/jpeg"]


def test_non_invoice(fake_llm, pdf_file):
    fake_llm(NON_INVOICE_JSON)
    out, _ = extract_document(pdf_file)
    assert out["is_invoice"] is False
    assert all(out[k] is None for k in NON_INVOICE_JSON if k != "is_invoice")


def test_think_tags_stripped():
    model, errors = parse_response("<think>hmm</think>\n" + json.dumps(INVOICE_JSON))
    assert errors == [] and model.total == 118.0


def test_bad_json_gives_schema_error_envelope(fake_llm, pdf_file):
    fake_llm("not json at all")
    out, meta = extract_document(pdf_file)
    assert out["error"]["code"] == "SCHEMA_ERROR" and out["is_invoice"] is None
    assert meta["validation"]["errors"][0].startswith("JSON_PARSE_ERROR")
    assert meta["cost_usd"] > 0  # the failed call still cost tokens


@pytest.mark.parametrize("reply,code", [(INVOICE_JSON, 0), (NON_INVOICE_JSON, 0), ("garbage", 4)])
def test_cli_exit_codes(fake_llm, pdf_file, capsys, reply, code):
    fake_llm(reply)
    assert extract.main([str(pdf_file)]) == code
    json.loads(capsys.readouterr().out)  # stdout is always valid JSON


def test_cli_unsupported_file_exit_3(fake_llm, tmp_path, capsys):
    fake = fake_llm()
    p = tmp_path / "fake.pdf"
    p.write_text("hello, I am text")
    assert extract.main([str(p)]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "UNSUPPORTED_TYPE"
    assert fake.calls == []
