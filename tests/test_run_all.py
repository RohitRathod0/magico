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
