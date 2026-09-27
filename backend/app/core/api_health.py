"""
core/api_health.py

Lightweight API key / configuration health check for OnionGrade AI.

Reports Gemini API status without exposing the key or crashing on failure.

Status values:
  "configured"    — key present AND passes minimal format check
  "not_configured" — no key set (demo/offline mode is active)
  "validation_failed" — key present but failed a lightweight check
  "error"         — unexpected exception during check
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("oniongrade.api_health")


def _key_looks_valid(key: str) -> bool:
    """Basic structural check: Google AI keys start with 'AIza' and are ~39 chars."""
    return bool(key) and key.startswith("AIza") and len(key) >= 30


def check_gemini_api() -> dict:
    """Return a health dict about the Gemini API configuration.

    Never raises; never prints the key.
    Does NOT make a live API request — format check only, for fast startup.
    """
    from app.core.vision_service import DEMO_MODE, GOOGLE_API_KEY, VISION_MODEL, VISION_MODELS

    if not GOOGLE_API_KEY:
        return {
            "status": "not_configured",
            "message": "Gemini API: Not configured (demo/offline mode active)",
            "demo_mode": True,
            "vision_model": VISION_MODEL,
            "models_available": VISION_MODELS,
        }

    if not _key_looks_valid(GOOGLE_API_KEY):
        return {
            "status": "validation_failed",
            "message": (
                "Gemini API: Validation failed — key format looks incorrect. "
                "Keys should start with 'AIza'. "
                "Get a key at https://aistudio.google.com/apikey"
            ),
            "demo_mode": False,
            "vision_model": VISION_MODEL,
            "models_available": VISION_MODELS,
        }

    return {
        "status": "configured",
        "message": "Gemini API: Configured",
        "demo_mode": DEMO_MODE,
        "vision_model": VISION_MODEL,
        "models_available": VISION_MODELS,
    }
