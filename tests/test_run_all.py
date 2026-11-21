import json

import run_all
from tests.conftest import INVOICE_JSON


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_pacer_waits_only_when_over_budget():
    c = FakeClock()
    p = run_all.TokenPacer(7000, sleep=c.sleep, clock=c)
    assert p.wait_for(2500) == 0
    p.record(2400)
    assert p.wait_for(2500) == 0
    p.record(2400)
    c.t = 10
    waited = p.wait_for(2500)  # 4800 + 2500 > 7000 -> wait until the first entry expires
    assert waited > 0 and c.t >= 60


def test_pacer_disabled():
    p = run_all.TokenPacer(0)
    p.record(10**6)
    assert p.wait_for(10**6) == 0


def test_run_all_writes_outputs_and_skips_existing(fake_llm, tmp_path, pdf_file):
    data = tmp_path / "data"
    (data / "labelled").mkdir(parents=True)
    (data / "unlablled").mkdir()
    (data / "labelled" / "L01.pdf").write_bytes(pdf_file.read_bytes())
    (data / "labelled" / "L01.json").write_text("{}")
    (data / "unlablled" / "U01.pdf").write_bytes(b"not a pdf")
    out = tmp_path / "outputs"

    fake = fake_llm(INVOICE_JSON)
    assert run_all.main(["--data", str(data), "--out", str(out), "--itpm", "0"]) == 0
    assert json.loads((out / "L01.json").read_text())["total"] == 118.0
    assert json.loads((out / "U01.json").read_text())["error"]["code"] == "UNSUPPORTED_TYPE"
    assert json.loads((out / "meta" / "L01.json").read_text())["input_kind"] == "pdf_scan"
    assert len(fake.calls) == 1

    fake2 = fake_llm()  # second run: everything cached, no calls
    run_all.main(["--data", str(data), "--out", str(out), "--itpm", "0"])
    assert fake2.calls == []


def test_quota_stops_run_and_keeps_good_output(tmp_path, pdf_file, monkeypatch):
    from invoice_pipeline import pipeline
    data = tmp_path / "data"
    (data / "labelled").mkdir(parents=True)
    (data / "unlablled").mkdir()
    for s in ("L01", "L02", "L03"):
        (data / "labelled" / f"{s}.pdf").write_bytes(pdf_file.read_bytes())
    out = tmp_path / "outputs"
    (out / "meta").mkdir(parents=True)
    good = {"is_invoice": True, "total": 1.0, "needs_review": False, "review_reasons": []}
    (out / "L01.json").write_text(json.dumps(good))
    (out / "meta" / "L01.json").write_text(json.dumps({"tokens_in": 1, "tokens_out": 1, "cost_usd": 0, "attempts": 1}))

    calls = []

    def fake_extract(doc, mode="vision"):
        calls.append(doc.stem)
        msg = "RateLimitError: ... on tokens per day (TPD) ... try again in 10m2s"
        return ({"is_invoice": None, "needs_review": True, "review_reasons": ["LLM_UNAVAILABLE"],
                 "error": {"code": "LLM_UNAVAILABLE", "message": msg}},
                {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "attempts": 1, "input_kind": "pdf_scan"})
    monkeypatch.setattr(run_all, "extract_document", fake_extract)
    run_all.main(["--data", str(data), "--out", str(out), "--itpm", "0", "--force"])
    assert calls == ["L01"]                                   # stopped after the first quota error
    assert json.loads((out / "L01.json").read_text()) == good  # good output not overwritten
    assert not (out / "L02.json").exists()


def test_was_unavailable():
    assert run_all.was_unavailable({"error": {"code": "LLM_UNAVAILABLE", "message": ""}})
    assert run_all.was_unavailable({"review_reasons": ["TOTALS_MISMATCH: x", "LLM_UNAVAILABLE on repair: RateLimitError"]})
    assert not run_all.was_unavailable({"review_reasons": ["TOTALS_MISMATCH: x"]})
    assert not run_all.was_unavailable({"error": {"code": "CORRUPT_PDF", "message": ""}})
