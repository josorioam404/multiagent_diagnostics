import json
import logging
import re

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from drp_system import config

logger = logging.getLogger(__name__)


def _ensure_configured() -> None:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=config.GEMINI_API_KEY)


def _get_model() -> genai.GenerativeModel:
    _ensure_configured()
    return genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )


_model: genai.GenerativeModel | None = None


def _model_singleton() -> genai.GenerativeModel:
    global _model
    if _model is None:
        _model = _get_model()
    return _model


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


@retry(
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
    wait=wait_exponential(multiplier=1, min=5, max=90),
    stop=stop_after_attempt(4),
)
async def call_gemini(prompt: str) -> dict:
    """Single async Gemini call; returns parsed JSON dict."""
    model = _model_singleton()
    response = await model.generate_content_async(prompt)
    text = _strip_json_fences(response.text or "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Gemini returned invalid JSON: %s", text[:500])
        raise ValueError("Gemini response was not valid JSON") from exc


def normalize_drug_name(medication: str) -> str:
    """Strip dose/strength for openFDA lookup (e.g. 'Metformin 850mg' -> 'Metformin')."""
    name = medication.split(",")[0].strip()
    name = re.sub(
        r"\s+\d+(\.\d+)?\s*(mg|mcg|g|ml|units?|iu|meq|%|tab|cap|po|bid|tid|qid).*$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"\s+\d+(\.\d+)?$", "", name)
    return name.strip() or medication.strip()
