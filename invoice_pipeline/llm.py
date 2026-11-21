"""The only module that talks to Groq.

call_model(images, prompt) -> (text, usage)

Transport errors are retried by tenacity:
  - timeout / connection / 5xx: up to MAX_TRANSPORT_ATTEMPTS, exponential backoff
  - 429 rate limit: up to MAX_RATE_LIMIT_ATTEMPTS, each time waiting as long as Groq
    asks (the free plan has a 7,000 input-tokens/minute limit that resets every minute).
    If Groq asks for more than MAX_RATE_LIMIT_WAIT_S, it is the daily quota (200k tokens/day
    on the free plan): give up immediately instead of hammering the API.
After the last failure the exception propagates and pipeline.py maps it to LLM_UNAVAILABLE.
"""
from __future__ import annotations

import base64
import re
import sys
import time

import groq
from groq import Groq
from tenacity import Retrying, retry_if_exception

from . import config
from .schema import response_json_schema

MAX_TRANSPORT_ATTEMPTS = 3        # timeouts, connection errors, 5xx (PRD §5.3)
MAX_RATE_LIMIT_ATTEMPTS = 10      # 429s: keep waiting for the per-minute window to reset
MAX_RATE_LIMIT_WAIT_S = 120.0     # if Groq asks for longer, it is the daily quota: stop now
MAX_WAIT_S = 60.0                 # backoff cap for non-429 errors
_sleep = time.sleep  # patched in tests
_TRY_AGAIN_RE = re.compile(r"try again in ((?:[\d.]+(?:h|ms|m|s))+)")
_DURATION_PART_RE = re.compile(r"([\d.]+)(h|ms|m|s)")
_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
        _client = Groq(api_key=config.GROQ_API_KEY, timeout=config.REQUEST_TIMEOUT_S, max_retries=0)
    return _client


def response_format(strict: bool = True) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "invoice_extraction", "schema": response_json_schema(), "strict": strict},
    }


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    details = getattr(u, "completion_tokens_details", None) if u else None
    return {
        "tokens_in": getattr(u, "prompt_tokens", None) if u else None,
        "tokens_out": getattr(u, "completion_tokens", None) if u else None,
        # Reported separately when the API gives it. Per Groq/OpenAI convention it is
        # already included in completion_tokens, so cost.py does NOT add it again.
        "tokens_thinking": (getattr(details, "reasoning_tokens", None) or 0) if details else 0,
    }


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (groq.APITimeoutError, groq.APIConnectionError, groq.RateLimitError)):
        return True
    return isinstance(exc, groq.APIStatusError) and exc.status_code >= 500


def parse_duration(text: str) -> float | None:
    """'10m20.784s' -> 620.784; '59.6s' -> 59.6; '1h2m' -> 3720."""
    parts = _DURATION_PART_RE.findall(text or "")
    return sum(float(n) * _UNIT_S[u] for n, u in parts) if parts else None


def asked_wait_seconds(exc: BaseException) -> float | None:
    """How long a 429 asks us to wait, from the message ('try again in 10m20s') or retry-after header."""
    m = _TRY_AGAIN_RE.search(str(exc))
    if m:
        return parse_duration(m.group(1))
    resp = getattr(exc, "response", None)
    hdr = resp.headers.get("retry-after") if resp is not None else None
    try:
        return float(hdr) if hdr else None
    except ValueError:
        return None


def is_daily_limit(text: str) -> bool:
    """Groq names the limit in the message: '... on tokens per day (TPD)' / 'requests per day (RPD)'."""
    return "per day" in text or "(TPD)" in text or "(RPD)" in text


def retry_wait_seconds(exc: BaseException | None, attempt: int) -> float:
    """Seconds to wait before the next attempt (attempt = number of attempts made so far)."""
    if isinstance(exc, groq.RateLimitError):
        asked = asked_wait_seconds(exc)
        return min(asked + 1.0, MAX_RATE_LIMIT_WAIT_S) if asked is not None else 30.0
    return min(2.0 ** attempt, MAX_WAIT_S)  # 2s, 4s, ...


def _max_attempts(exc: BaseException | None) -> int:
    return MAX_RATE_LIMIT_ATTEMPTS if isinstance(exc, groq.RateLimitError) else MAX_TRANSPORT_ATTEMPTS


def _stop(retry_state) -> bool:
    """Per-minute rate limits get a larger retry budget; a long wait means the daily quota is gone."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, groq.RateLimitError):
        asked = asked_wait_seconds(exc)
        if asked is not None and asked > MAX_RATE_LIMIT_WAIT_S:
            return True
    return retry_state.attempt_number >= _max_attempts(exc)


def _wait(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return retry_wait_seconds(exc, retry_state.attempt_number)


def _log_before_sleep(retry_state) -> None:
    """Runs only when tenacity is really about to sleep (not on the final, stopping attempt)."""
    exc = retry_state.outcome.exception()
    print(f"[llm] {type(exc).__name__}; retry {retry_state.attempt_number}/{_max_attempts(exc) - 1} "
          f"in {retry_state.next_action.sleep:.1f}s", file=sys.stderr)


def call_model(images: list[tuple[bytes, str]], prompt: str, *, model: str | None = None,
               fmt: dict | None = None) -> tuple[str, dict]:
    content = [{"type": "text", "text": prompt}]
    for data, mime in images:
        b64 = base64.b64encode(data).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    retrying = Retrying(stop=_stop, wait=_wait, before_sleep=_log_before_sleep,
                        retry=retry_if_exception(is_transient), sleep=_sleep, reraise=True)
    resp = retrying(
        _get_client().chat.completions.create,
        model=model or config.GROQ_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=config.TEMPERATURE,
        max_completion_tokens=config.MAX_COMPLETION_TOKENS,
        response_format=fmt or response_format(),
    )
    return resp.choices[0].message.content or "", _usage(resp)
