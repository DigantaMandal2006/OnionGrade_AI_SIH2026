"""
core/image_preprocessing.py

Image preprocessing pipeline for OnionGrade AI.

Before sending an image to Gemini Vision:
  1. Decode with Pillow
  2. Correct EXIF orientation (auto-rotate)
  3. Resize to max 1024px on the longest side (aspect-ratio preserved)
  4. Strip EXIF / metadata
  5. Re-encode as JPEG at quality 85

The original uploaded bytes are preserved unchanged for:
  - Display in the Streamlit UI
  - NIR analysis
  - PDF/report attachments

Logging shows original vs processed dimensions and file size.
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

logger = logging.getLogger("oniongrade.preprocess")

MAX_DIMENSION = int(1024)    # longest edge target (pixels)
JPEG_QUALITY  = int(85)      # re-encode quality


def _exif_transpose(image):
    """Return the image with EXIF rotation applied.

    Uses Pillow's ImageOps.exif_transpose if available (Pillow >= 6.0.0).
    Falls back silently if the image has no EXIF or the call fails.
    """
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(image)
    except Exception:
        return image


def preprocess_image(
    image_bytes: bytes,
    max_dimension: int = MAX_DIMENSION,
    jpeg_quality: int = JPEG_QUALITY,
) -> Tuple[bytes, str]:
    """Pre-process raw uploaded image bytes for Gemini analysis.

    Parameters
    ----------
    image_bytes   : raw bytes of the user-uploaded image
    max_dimension : longest side target in pixels (default 1024)
    jpeg_quality  : JPEG re-encode quality 1-95 (default 85)

    Returns
    -------
    (processed_bytes, mime_type)
        processed_bytes : re-encoded JPEG bytes (possibly resized/rotated)
        mime_type       : always "image/jpeg" after processing

    Notes
    -----
    - Never modifies the caller's original bytes.
    - If Pillow is unavailable or processing fails, returns the original
      bytes unchanged with its detected mime_type so the pipeline degrades
      gracefully.
    """
    original_size = len(image_bytes)

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        orig_w, orig_h = img.size
        orig_format = (img.format or "JPEG").upper()

        # 1. EXIF orientation correction
        img = _exif_transpose(img)

        # 2. Convert to RGB (handles RGBA, P, LA, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # 3. Resize if needed — maintain aspect ratio
        cur_w, cur_h = img.size
        if max(cur_w, cur_h) > max_dimension:
            scale = max_dimension / max(cur_w, cur_h)
            new_w = max(1, int(cur_w * scale))
            new_h = max(1, int(cur_h * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info(
                "Image resized: %dx%d → %dx%d (original format: %s, max_dim: %d)",
                orig_w, orig_h, new_w, new_h, orig_format, max_dimension,
            )
        else:
            logger.info(
                "Image within size limit: %dx%d (no resize needed, format: %s)",
                cur_w, cur_h, orig_format,
            )

        # 4. Strip EXIF / metadata by re-encoding without copying info
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        processed = buf.getvalue()

        processed_size = len(processed)
        reduction_pct = (1 - processed_size / original_size) * 100
        logger.info(
            "Image preprocessed: %d KB → %d KB (%.1f%% reduction)",
            original_size // 1024,
            processed_size // 1024,
            reduction_pct,
        )

        return processed, "image/jpeg"

    except ImportError:
        logger.warning("Pillow not available — skipping image preprocessing")
        return image_bytes, "image/jpeg"
    except Exception as exc:
        logger.warning("Image preprocessing failed (%s) — using original bytes", exc)
        return image_bytes, "image/jpeg"


def get_image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
    """Return (width, height) of image or (0, 0) on failure."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        return img.size
    except Exception:
        return (0, 0)
