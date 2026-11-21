"""extract_document(path) -> (output_dict, meta_dict)

file -> preflight -> render -> LLM -> parse -> schema -> business checks
  - preflight failure: error envelope, no LLM call, $0
  - JSON/schema/totals failure: ONE repair retry with the errors fed back
  - still failing: keep the latest schema-valid attempt with needs_review=true and the
    errors as review_reasons; if no attempt was schema-valid, error envelope SCHEMA_ERROR
  - transport failure after tenacity retries: error envelope LLM_UNAVAILABLE
"""
from __future__ import annotations

import time
from pathlib import Path

from . import config, llm
from .cost import sum_usage, usage_cost
from .preflight import InputError, input_kind, preflight
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT, repair_prompt
from .render import to_images
from .schema import FIELDS
from .validate import is_retryable, validate

MAX_LLM_ATTEMPTS = 2  # first try + one repair retry


def error_envelope(code: str, message: str) -> dict:
    """Not a schema-valid extraction: is_invoice=None means 'could not determine'."""
    return {f: None for f in FIELDS} | {
        "needs_review": True, "review_reasons": [code], "error": {"code": code, "message": message}}


def extract_document(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    t0 = time.perf_counter()
    usages: list[dict] = []
    meta = {"file": path.name, "input_kind": None, "model": config.GROQ_MODEL, "prompt_version": PROMPT_VERSION,
            "attempts": 0, "latency_s": None, "tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0,
            "cost_usd": 0.0, "validation": {"schema_ok": False, "totals_ok": None, "errors": []},
            "attempt_errors": [], "error": None}

    def finish(output: dict) -> tuple[dict, dict]:
        meta.update(sum_usage(usages))
        cost = usage_cost(meta) if usages else 0.0
        meta["cost_usd"] = round(cost, 8) if cost is not None else None
        meta["latency_s"] = round(time.perf_counter() - t0, 2)
        return output, meta

    def fail(code: str, message: str) -> tuple[dict, dict]:
        meta["error"] = {"code": code, "message": message}
        return finish(error_envelope(code, message))

    # 1. Preflight + rendering (no LLM cost)
    try:
        data, mime = preflight(path)
        meta["input_kind"] = input_kind(data, mime)
        try:
            images = to_images(data, mime)
        except Exception as e:  # pypdf accepted it but the renderer could not draw it
            raise InputError("CORRUPT_PDF", f"PDF could not be rendered: {type(e).__name__}: {e}") from None
    except InputError as e:
        return fail(e.code, e.message)

    # 2. LLM + validation, with one repair retry
    best, best_errors, errors = None, [], []
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        prompt = SYSTEM_PROMPT if attempt == 1 else repair_prompt(errors)
        meta["attempts"] = attempt
        try:
            text, usage = llm.call_model(images, prompt)
        except Exception as e:
            if best is not None:  # repair call failed: fall back to the earlier valid attempt
                best_errors = best_errors + [f"LLM_UNAVAILABLE on repair: {type(e).__name__}"]
                break
            return fail("LLM_UNAVAILABLE", f"{type(e).__name__}: {e}")
        usages.append(usage)

        inv, errors = validate(text)
        meta["attempt_errors"].append(errors)
        if inv is not None:
            best, best_errors = inv, errors  # prefer the latest schema-valid attempt
        if not is_retryable(errors):
            break

    if best is None:
        meta["validation"]["errors"] = errors
        return fail("SCHEMA_ERROR", "; ".join(errors))

    final_errors = best_errors  # the errors of the attempt we are returning
    meta["validation"] = {
        "schema_ok": True,
        "totals_ok": (None if not (best.is_invoice and best.line_items and best.total is not None)
                      else not any(e.startswith(("TOTALS_MISMATCH", "LINE_AMOUNT_MISSING")) for e in final_errors)),
        "errors": final_errors,
    }
    return finish(best.model_dump() | {"needs_review": bool(final_errors), "review_reasons": final_errors})
