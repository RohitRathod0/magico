"""The only module that talks to Groq.

call_model(images, prompt) -> (text, usage)

Transport errors (timeout, connection, 429, 5xx) are retried by tenacity: up to
MAX_TRANSPORT_ATTEMPTS attempts. On 429 we wait as long as Groq asks (the free plan
has a 7,000 input-tokens/minute limit); otherwise exponential backoff.
After the last failure the exception propagates and pipeline.py maps it to LLM_UNAVAILABLE.
"""
from __future__ import annotations

import base64
import re
import sys
import time

import groq
from groq import Groq
from tenacity import Retrying, retry_if_exception, stop_after_attempt

from . import config
from .schema import response_json_schema

MAX_TRANSPORT_ATTEMPTS = 3
MAX_WAIT_S = 60.0
_sleep = time.sleep  # patched in tests
_TRY_AGAIN_RE = re.compile(r"try again in ([\d.]+)s")

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


def retry_wait_seconds(exc: BaseException | None, attempt: int) -> float:
    """Seconds to wait before the next attempt (attempt = number of attempts made so far)."""
    if isinstance(exc, groq.RateLimitError):
        hdr = exc.response.headers.get("retry-after") if exc.response is not None else None
        m = _TRY_AGAIN_RE.search(str(exc))
        for v in (hdr, m.group(1) if m else None):
            try:
                if v is not None:
                    return min(float(v) + 1.0, MAX_WAIT_S)
            except ValueError:
                pass
        return MAX_WAIT_S / 2
    return min(2.0 ** attempt, MAX_WAIT_S)  # 2s, 4s, ...


def _wait(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    secs = retry_wait_seconds(exc, retry_state.attempt_number)
    print(f"[llm] {type(exc).__name__}; retry {retry_state.attempt_number}/"
          f"{MAX_TRANSPORT_ATTEMPTS - 1} in {secs:.1f}s", file=sys.stderr)
    return secs


def call_model(images: list[tuple[bytes, str]], prompt: str, *, model: str | None = None,
               fmt: dict | None = None) -> tuple[str, dict]:
    content = [{"type": "text", "text": prompt}]
    for data, mime in images:
        b64 = base64.b64encode(data).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    retrying = Retrying(stop=stop_after_attempt(MAX_TRANSPORT_ATTEMPTS), wait=_wait,
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
