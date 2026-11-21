"""extract_document(path) -> (output_dict, meta_dict)

Phase 2: happy path (sniff -> render -> LLM -> parse -> schema). No retries yet.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pydantic import ValidationError

from . import config, llm
from .cost import sum_usage, usage_cost
from .preflight import sniff_mime
from .prompt import PROMPT_VERSION, SYSTEM_PROMPT
from .render import to_images
from .schema import FIELDS, InvoiceExtraction

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class InputError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def error_envelope(code: str, message: str) -> dict:
    """Not a schema-valid extraction: is_invoice=None means 'could not determine'."""
    return {f: None for f in FIELDS} | {
        "needs_review": True, "review_reasons": [code], "error": {"code": code, "message": message}}


def parse_response(text: str) -> tuple[InvoiceExtraction | None, list[str]]:
    """Strip any reasoning text, parse JSON, validate the schema. Returns (model, errors)."""
    cleaned = THINK_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, [f"JSON_PARSE_ERROR: {e}"]
    try:
        return InvoiceExtraction.model_validate(data), []
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors())
        return None, [f"SCHEMA_ERROR: {msg}"]


def extract_document(path: str | Path) -> tuple[dict, dict]:
    path = Path(path)
    t0 = time.perf_counter()
    usages: list[dict] = []
    meta = {"file": path.name, "model": config.GROQ_MODEL, "prompt_version": PROMPT_VERSION,
            "attempts": 0, "latency_s": None, "tokens_in": 0, "tokens_out": 0, "tokens_thinking": 0,
            "cost_usd": 0.0, "validation": {"schema_ok": False, "totals_ok": None, "errors": []},
            "error": None}

    def finish(output: dict) -> tuple[dict, dict]:
        meta.update(sum_usage(usages))
        cost = usage_cost(meta) if usages else 0.0
        meta["cost_usd"] = round(cost, 8) if cost is not None else None
        meta["latency_s"] = round(time.perf_counter() - t0, 2)
        return output, meta

    try:
        if not path.is_file():
            raise InputError("FILE_NOT_FOUND", f"No such file: {path}")
        data = path.read_bytes()
        mime = sniff_mime(data)
        if mime is None:
            raise InputError("UNSUPPORTED_TYPE", "File is not a PDF, PNG, JPEG or WEBP (by content)")
        images = to_images(data, mime)
    except InputError as e:
        meta["error"] = {"code": e.code, "message": e.message}
        return finish(error_envelope(e.code, e.message))

    meta["attempts"] = 1
    try:
        text, usage = llm.call_model(images, SYSTEM_PROMPT)
    except Exception as e:  # transport errors; tenacity retries are added in Phase 3
        meta["error"] = {"code": "LLM_UNAVAILABLE", "message": f"{type(e).__name__}: {e}"}
        return finish(error_envelope("LLM_UNAVAILABLE", meta["error"]["message"]))
    usages.append(usage)

    model, errors = parse_response(text)
    meta["validation"]["errors"] = errors
    if model is None:
        meta["error"] = {"code": "SCHEMA_ERROR", "message": "; ".join(errors)}
        return finish(error_envelope("SCHEMA_ERROR", meta["error"]["message"]))

    meta["validation"]["schema_ok"] = True
    return finish(model.model_dump() | {"needs_review": False, "review_reasons": []})
