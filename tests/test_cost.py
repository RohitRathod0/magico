import pytest

from invoice_pipeline.cost import sum_usage, usage_cost

PRICING = {"usd_per_1m_input": 0.80, "usd_per_1m_output": 4.00}


def test_cost_formula():
    u = {"tokens_in": 2_000_000, "tokens_out": 500_000, "tokens_thinking": 100_000}
    # thinking is already inside tokens_out, so it is not added again
    assert usage_cost(u, PRICING) == pytest.approx(1.60 + 2.00)


def test_cost_none_when_price_missing():
    assert usage_cost({"tokens_in": 10, "tokens_out": 10}, {"usd_per_1m_input": None}) is None


def test_sum_usage_handles_none():
    assert sum_usage([{"tokens_in": 5, "tokens_out": None}, {"tokens_in": 1, "tokens_out": 2}]) == {
        "tokens_in": 6, "tokens_out": 2, "tokens_thinking": 0}
