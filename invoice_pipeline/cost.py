"""Token usage -> USD.

cost = tokens_in * in_price / 1e6 + tokens_out * out_price / 1e6

tokens_out is Groq's `completion_tokens`, which already includes any reasoning
tokens (OpenAI-compatible usage), so thinking is charged as output exactly once.
tokens_thinking is recorded for reporting only.
"""
from __future__ import annotations

from . import config


def usage_cost(usage: dict, pricing: dict | None = None) -> float | None:
    p = pricing or config.load_pricing()
    pin, pout = p.get("usd_per_1m_input"), p.get("usd_per_1m_output")
    if pin is None or pout is None:
        return None
    return (usage.get("tokens_in") or 0) * pin / 1e6 + (usage.get("tokens_out") or 0) * pout / 1e6


def sum_usage(usages: list[dict]) -> dict:
    keys = ("tokens_in", "tokens_out", "tokens_thinking")
    return {k: sum(u.get(k) or 0 for u in usages) for k in keys}
