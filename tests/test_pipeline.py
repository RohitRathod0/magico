"""Pipeline tests with a mocked LLM (no network)."""
import json

import pytest

import extract
import groq
import httpx

from invoice_pipeline import llm
from invoice_pipeline.pipeline import extract_document
from invoice_pipeline.validate import parse_and_validate_schema
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
    model, errors = parse_and_validate_schema("<think>hmm</think>\n" + json.dumps(INVOICE_JSON))
    assert errors == [] and model.total == 118.0


def test_bad_json_twice_gives_schema_error_envelope(fake_llm, pdf_file):
    fake_llm("not json at all", "still not json")
    out, meta = extract_document(pdf_file)
    assert out["error"]["code"] == "SCHEMA_ERROR" and out["is_invoice"] is None
    assert meta["validation"]["errors"][0].startswith("JSON_PARSE_ERROR")
    assert meta["cost_usd"] > 0  # the failed call still cost tokens


@pytest.mark.parametrize("reply,code", [(INVOICE_JSON, 0), (NON_INVOICE_JSON, 0), ("garbage", 4),
                                        (INVOICE_JSON | {"total": 999.0}, 2)])
def test_cli_exit_codes(fake_llm, pdf_file, capsys, reply, code):
    fake_llm(reply, reply)
    assert extract.main([str(pdf_file)]) == code
    json.loads(capsys.readouterr().out)  # stdout is always valid JSON


def test_cli_unsupported_file_exit_3(fake_llm, tmp_path, capsys):
    fake = fake_llm()
    p = tmp_path / "fake.pdf"
    p.write_text("hello, I am text")
    assert extract.main([str(p)]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "UNSUPPORTED_TYPE"
    assert fake.calls == []


# ---- repair retry ----

def test_bad_json_then_success(fake_llm, pdf_file):
    fake = fake_llm("{not json", INVOICE_JSON)
    out, meta = extract_document(pdf_file)
    assert meta["attempts"] == 2 and out["needs_review"] is False and out["total"] == 118.0
    assert "JSON_PARSE_ERROR" in fake.calls[1]["prompt"]  # errors were fed back
    assert meta["tokens_in"] == 2 * 1800  # both attempts are costed


def test_persistent_totals_mismatch_flagged(fake_llm, pdf_file):
    bad = INVOICE_JSON | {"total": 999.0}
    fake_llm(bad, bad)
    out, meta = extract_document(pdf_file)
    assert meta["attempts"] == 2 and out["needs_review"] is True and out["total"] == 999.0
    assert out["review_reasons"][0].startswith("TOTALS_MISMATCH")
    assert meta["validation"]["totals_ok"] is False


def test_valid_then_invalid_keeps_first_valid(fake_llm, pdf_file):
    fake_llm(INVOICE_JSON | {"total": 999.0}, "garbage")
    out, meta = extract_document(pdf_file)
    assert out["total"] == 999.0 and out["needs_review"] is True and "error" not in out
    assert out["review_reasons"][0].startswith("TOTALS_MISMATCH")


def test_missing_vendor_flagged_without_retry(fake_llm, pdf_file):
    fake = fake_llm(INVOICE_JSON | {"vendor": None})
    out, meta = extract_document(pdf_file)
    assert len(fake.calls) == 1 and out["needs_review"] is True and out["review_reasons"] == ["MISSING_VENDOR"]


# ---- transport retry (real llm.call_model, fake Groq client) ----

class FakeCompletions:
    def __init__(self, excs, reply=None):
        self.excs, self.reply, self.n = list(excs), reply, 0

    def create(self, **kw):
        self.n += 1
        if self.excs:
            raise self.excs.pop(0)
        return self.reply


def _timeout():
    return groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))


def _rate_limit(retry_after="7"):
    req = httpx.Request("POST", "https://api.groq.com")
    resp = httpx.Response(429, request=req, headers={"retry-after": retry_after})
    return groq.RateLimitError("Rate limit reached ... Please try again in 20.5s.", response=resp, body=None)


def _install_client(monkeypatch, completions):
    class Client:
        chat = type("Chat", (), {"completions": completions})()
    sleeps = []
    monkeypatch.setattr(llm, "_get_client", lambda: Client())
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    return sleeps


def test_timeout_three_times_gives_llm_unavailable(monkeypatch, pdf_file):
    comp = FakeCompletions([_timeout(), _timeout(), _timeout()])
    sleeps = _install_client(monkeypatch, comp)
    out, meta = extract_document(pdf_file)
    assert comp.n == 3 and out["error"]["code"] == "LLM_UNAVAILABLE" and meta["cost_usd"] == 0.0
    assert sleeps == [2.0, 4.0]


def test_rate_limit_waits_as_asked_then_succeeds(monkeypatch, pdf_file):
    reply = type("R", (), {
        "choices": [type("C", (), {"message": type("M", (), {"content": json.dumps(INVOICE_JSON)})()})()],
        "usage": type("U", (), {"prompt_tokens": 1800, "completion_tokens": 150,
                                "completion_tokens_details": None})()})()
    comp = FakeCompletions([_rate_limit("7")], reply)
    sleeps = _install_client(monkeypatch, comp)
    out, meta = extract_document(pdf_file)
    assert comp.n == 2 and sleeps == [8.0] and out["total"] == 118.0 and meta["tokens_in"] == 1800


def test_rate_limit_wait_from_message_when_no_header():
    exc = _rate_limit("")
    assert llm.retry_wait_seconds(exc, 1) == pytest.approx(21.5)


def test_non_transient_error_not_retried(monkeypatch, pdf_file):
    req = httpx.Request("POST", "https://api.groq.com")
    bad = groq.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    comp = FakeCompletions([bad])
    _install_client(monkeypatch, comp)
    out, _ = extract_document(pdf_file)
    assert comp.n == 1 and out["error"]["code"] == "LLM_UNAVAILABLE"


# ---- preflight: every bad input fails with the right code and zero LLM calls ----

@pytest.mark.parametrize("key", ["FILE_NOT_FOUND", "EMPTY_FILE", "UNSUPPORTED_TYPE", "CORRUPT_PDF",
                                 "CORRUPT_PDF_encrypted", "TOO_MANY_PAGES", "CORRUPT_IMAGE",
                                 "UNREADABLE_IMAGE", "UNREADABLE_IMAGE_blank"])
def test_preflight_errors(fake_llm, bad_files, key):
    fake = fake_llm()
    out, meta = extract_document(bad_files[key])
    code = key.split("_encrypted")[0].split("_blank")[0]
    assert out["error"]["code"] == code, out["error"]
    assert out["review_reasons"] == [code] and out["is_invoice"] is None
    assert fake.calls == [] and meta["cost_usd"] == 0.0 and meta["attempts"] == 0


def test_file_too_large(fake_llm, pdf_file, monkeypatch):
    from invoice_pipeline import config
    monkeypatch.setattr(config, "MAX_FILE_BYTES", 100)
    fake = fake_llm()
    out, _ = extract_document(pdf_file)
    assert out["error"]["code"] == "FILE_TOO_LARGE" and fake.calls == []


def test_cli_corrupt_file_exit_3(fake_llm, bad_files, capsys):
    fake = fake_llm()
    assert extract.main([str(bad_files["CORRUPT_PDF"])]) == 3
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "CORRUPT_PDF" and fake.calls == []
