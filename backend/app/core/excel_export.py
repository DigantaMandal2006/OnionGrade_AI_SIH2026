"""
core/excel_export.py

Excel (.xlsx) export for OnionGrade AI using openpyxl.

Workbook structure:
  Sheet 1 — Onion Results  (per-onion rows, color-coded by grade)
  Sheet 2 — Summary        (batch-level statistics)

Keeps the existing CSV export unchanged.
Falls back gracefully if openpyxl is not installed.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("oniongrade.excel")

# Grade colour fills (ARGB hex)
_GREEN  = "FF90EE90"   # Grade A
_YELLOW = "FFFFF176"   # URS
_RED    = "FFFF9999"   # Defect
_GREY   = "FFD3D3D3"   # header row


def _grade_fill(grade: str):
    """Return an openpyxl PatternFill for the given grade string."""
    from openpyxl.styles import PatternFill
    g = (grade or "").strip()
    if "Grade A" in g or g == "A":
        colour = _GREEN
    elif g == "URS":
        colour = _YELLOW
    elif g == "Defect":
        colour = _RED
    else:
        return None
    return PatternFill(fill_type="solid", fgColor=colour)


def build_xlsx(
    onions: List[dict],
    grade_result: dict,
    batch_id: str = "unknown",
    analysis_mode: str = "rgb",
    demo_mode: bool = False,
    vision_model: str = "",
) -> Optional[bytes]:
    """Build an .xlsx workbook and return the bytes.

    Returns None if openpyxl is unavailable or an error occurs.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        logger.warning("openpyxl not installed — XLSX export unavailable")
        return None

    try:
        wb = Workbook()

        # ── Sheet 1: Per-onion results ────────────────────────────────────
        ws1 = wb.active
        ws1.title = "Onion Results"

        headers = [
            "#", "Onion ID", "Source Image", "Final Grade",
            "Grade Reason", "Size", "Colour", "Confidence (%)",
            "Sprouting", "Defect Present", "Defect Description",
            "Defect Type", "Severity", "Review Status",
        ]
        header_fill = PatternFill(fill_type="solid", fgColor=_GREY)
        header_font = Font(bold=True)

        ws1.append(headers)
        for cell in ws1[1]:
            cell.fill  = header_fill
            cell.font  = header_font
            cell.alignment = Alignment(horizontal="center")

        for idx, onion in enumerate(onions, start=1):
            grade    = str(onion.get("final_grade") or "")
            conf     = float(onion.get("confidence") or 0.0)
            sprouting = bool(onion.get("sprouting"))
            defect    = bool(onion.get("defect_present"))

            # Review status: manual_review flag or low-confidence label
            review = str(onion.get("review_status") or "")

            row = [
                idx,
                str(onion.get("onion_id") or ""),
                str(onion.get("source_image") or ""),
                grade,
                str(onion.get("grade_reason") or ""),
                str(onion.get("size") or ""),
                str(onion.get("color") or ""),
                round(conf * 100, 1),
                "Yes" if sprouting else "No",
                "Yes" if defect else "No",
                str(onion.get("defect_description") or ""),
                str(onion.get("defect_type") or ""),
                str(onion.get("severity") or ""),
                review,
            ]
            ws1.append(row)

            # Colour-code the grade cell (column D = index 4)
            grade_cell = ws1.cell(row=idx + 1, column=4)
            fill = _grade_fill(grade)
            if fill:
                grade_cell.fill = fill

        # Auto-fit column widths (approx)
        for col in ws1.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws1.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)

        # ── Sheet 2: Summary ──────────────────────────────────────────────
        ws2 = wb.create_sheet("Summary")

        summary_rows = [
            ("Batch ID",       batch_id),
            ("Generated At",   datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("Analysis Mode",  analysis_mode.upper()),
            ("Demo Mode",      "Yes" if demo_mode else "No"),
            ("Vision Model",   vision_model),
            ("", ""),
            ("Total Observed",  grade_result.get("total_validated", 0)),
            ("Total Validated", grade_result.get("total_validated", 0)),
            ("Grade A Count",   grade_result.get("grade_a_count", 0)),
            ("URS Count",       grade_result.get("urs_count", 0)),
            ("Defect Count",    grade_result.get("defect_count", 0)),
            ("Grade A %",       grade_result.get("grade_a_pct", 0.0)),
            ("URS %",           grade_result.get("urs_pct", 0.0)),
            ("Defect %",        grade_result.get("defect_pct", 0.0)),
        ]

        ws2.column_dimensions["A"].width = 22
        ws2.column_dimensions["B"].width = 30
        key_font = Font(bold=True)

        for label, value in summary_rows:
            ws2.append([label, value])

        # Colour-code the grade percentage cells
        grade_colour_map = {
            "Grade A %": _GREEN,
            "URS %":     _YELLOW,
            "Defect %":  _RED,
        }
        for row in ws2.iter_rows(min_row=1, max_row=ws2.max_row):
            cell_a = row[0]
            if cell_a.value:
                cell_a.font = key_font
            colour = grade_colour_map.get(str(cell_a.value or ""))
            if colour and len(row) > 1:
                row[1].fill = PatternFill(fill_type="solid", fgColor=colour)

        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue()

    except Exception as exc:
        logger.error("XLSX build failed: %s", exc)
        return None
