"""Settings: .env values, model name, pricing, limits and tolerances."""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# LLM call settings
TEMPERATURE = 0
MAX_COMPLETION_TOKENS = 4096
REQUEST_TIMEOUT_S = 60

# Input limits (preflight)
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 3                 # Groq vision: max 3 images per request
MIN_IMAGE_SHORT_SIDE = 300
MIN_IMAGE_STDDEV = 5.0
PDF_RENDER_DPI = 150

# Totals check tolerance: max(1.00, 0.5% of total)
TOTALS_ABS_TOL = 1.00
TOTALS_REL_TOL = 0.005

# Data / output paths
DATA_DIR = ROOT / "data"
LABELLED_DIR = DATA_DIR / "labelled"
UNLABELLED_DIRS = (DATA_DIR / "unlabelled", DATA_DIR / "unlablled")  # pack ships the misspelling
OUTPUTS_DIR = ROOT / "outputs"
META_DIR = OUTPUTS_DIR / "meta"
REPORTS_DIR = ROOT / "reports"
PRICING_PATH = ROOT / "config" / "pricing.json"


def load_pricing(path: Path = PRICING_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
