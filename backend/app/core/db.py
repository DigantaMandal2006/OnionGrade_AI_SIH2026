"""
core/db.py

SQLite-backed scan history for OnionGrade AI.

Each completed analysis is stored as a row in the `scans` table.
Per-onion results are stored in the `scan_onions` table.

Design goals:
  - Prevent re-calling Gemini when reloading a past result.
  - Structured schema (not an arbitrary blob dump).
  - Safe to missing / corrupted records.
  - Single file, zero server process, zero extra dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("oniongrade.db")

# Default DB path — backend project root / data / scans.db
# Override via ONIONGRADE_DB_PATH env var.
_DEFAULT_DB = Path(__file__).resolve().parents[3] / "data" / "scans.db"
DB_PATH = Path(os.getenv("ONIONGRADE_DB_PATH", str(_DEFAULT_DB)))


def _ensure_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS scans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,   -- ISO-8601 UTC
    analysis_mode   TEXT    NOT NULL DEFAULT 'rgb',
    total_observed  INTEGER NOT NULL DEFAULT 0,
    total_validated INTEGER NOT NULL DEFAULT 0,
    grade_a_count   INTEGER NOT NULL DEFAULT 0,
    urs_count       INTEGER NOT NULL DEFAULT 0,
    defect_count    INTEGER NOT NULL DEFAULT 0,
    grade_a_pct     REAL    NOT NULL DEFAULT 0.0,
    urs_pct         REAL    NOT NULL DEFAULT 0.0,
    defect_pct      REAL    NOT NULL DEFAULT 0.0,
    overall_status  TEXT    NOT NULL DEFAULT '',
    report_text     TEXT    NOT NULL DEFAULT '',
    demo_mode       INTEGER NOT NULL DEFAULT 0,   -- 0/1 boolean
    vision_model    TEXT    NOT NULL DEFAULT '',
    -- full API response JSON for reload-without-reanalysis
    response_json   TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS scan_onions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id     INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    onion_id    TEXT    NOT NULL,
    source_image TEXT   NOT NULL DEFAULT '',   -- for multi-image batches
    final_grade TEXT    NOT NULL DEFAULT '',
    grade_reason TEXT   NOT NULL DEFAULT '',
    size        TEXT    NOT NULL DEFAULT '',
    color       TEXT    NOT NULL DEFAULT '',
    confidence  REAL    NOT NULL DEFAULT 0.0,
    sprouting   INTEGER NOT NULL DEFAULT 0,
    defect_present INTEGER NOT NULL DEFAULT 0,
    defect_description TEXT,
    defect_type TEXT    NOT NULL DEFAULT '',
    severity    TEXT    NOT NULL DEFAULT '',
    -- normalized bbox JSON string or NULL
    bbox_json   TEXT
);
"""


# ── Connection helpers ────────────────────────────────────────────────────────

@contextmanager
def _connection():
    """Yield a SQLite connection, committing or rolling back on exit."""
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist.  Safe to call multiple times."""
    try:
        with _connection() as conn:
            conn.executescript(_DDL)
        logger.info("Database initialised at %s", DB_PATH)
    except Exception as exc:
        logger.error("Database init failed: %s", exc)


# ── Write ─────────────────────────────────────────────────────────────────────

def save_scan(
    response: dict,
    analysis_mode: str = "rgb",
    demo_mode: bool = False,
    vision_model: str = "",
) -> Optional[int]:
    """Persist a completed analysis response and return the new scan_id.

    Returns None on any failure — callers should treat None as "history
    not saved" without crashing.
    """
    try:
        batch_id = str(response.get("batch_id") or "unknown")
        created_at = datetime.now(timezone.utc).isoformat()
        total_observed = int(response.get("total_observed") or 0)

        grade_result = response.get("grade_result") or {}
        total_validated = int(grade_result.get("total_validated") or 0)
        grade_a_count = int(grade_result.get("grade_a_count") or 0)
        urs_count = int(grade_result.get("urs_count") or 0)
        defect_count = int(grade_result.get("defect_count") or 0)
        grade_a_pct = float(grade_result.get("grade_a_pct") or 0.0)
        urs_pct = float(grade_result.get("urs_pct") or 0.0)
        defect_pct = float(grade_result.get("defect_pct") or 0.0)

        overall_status = str(
            (response.get("overall_quality") or {}).get("status") or ""
        )
        report_text = str(response.get("report") or "")
        response_json = json.dumps(response, default=str)

        onions = response.get("onions") or response.get("validated") or []

        with _connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO scans
                    (batch_id, created_at, analysis_mode, total_observed,
                     total_validated, grade_a_count, urs_count, defect_count,
                     grade_a_pct, urs_pct, defect_pct, overall_status,
                     report_text, demo_mode, vision_model, response_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    batch_id, created_at, analysis_mode, total_observed,
                    total_validated, grade_a_count, urs_count, defect_count,
                    grade_a_pct, urs_pct, defect_pct, overall_status,
                    report_text, int(demo_mode), vision_model, response_json,
                ),
            )
            scan_id = cur.lastrowid

            for onion in onions:
                if not isinstance(onion, dict):
                    continue
                bbox = onion.get("bbox")
                conn.execute(
                    """
                    INSERT INTO scan_onions
                        (scan_id, onion_id, source_image, final_grade,
                         grade_reason, size, color, confidence,
                         sprouting, defect_present, defect_description,
                         defect_type, severity, bbox_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        scan_id,
                        str(onion.get("onion_id") or ""),
                        str(onion.get("source_image") or ""),
                        str(onion.get("final_grade") or ""),
                        str(onion.get("grade_reason") or ""),
                        str(onion.get("size") or ""),
                        str(onion.get("color") or ""),
                        float(onion.get("confidence") or 0.0),
                        int(bool(onion.get("sprouting"))),
                        int(bool(onion.get("defect_present"))),
                        onion.get("defect_description"),
                        str(onion.get("defect_type") or ""),
                        str(onion.get("severity") or ""),
                        json.dumps(bbox) if isinstance(bbox, dict) else None,
                    ),
                )

        logger.info("Scan saved: scan_id=%s batch_id=%s", scan_id, batch_id)
        return scan_id

    except Exception as exc:
        logger.error("Failed to save scan: %s", exc)
        return None


# ── Read ──────────────────────────────────────────────────────────────────────

def list_scans(limit: int = 50) -> list[dict]:
    """Return a list of scan summaries, newest first."""
    try:
        with _connection() as conn:
            rows = conn.execute(
                """
                SELECT id, batch_id, created_at, analysis_mode,
                       total_observed, total_validated,
                       grade_a_count, urs_count, defect_count,
                       grade_a_pct, urs_pct, defect_pct,
                       overall_status, demo_mode, vision_model
                FROM scans
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to list scans: %s", exc)
        return []


def get_scan(scan_id: int) -> Optional[dict]:
    """Return the full stored response dict for a scan, or None."""
    try:
        with _connection() as conn:
            row = conn.execute(
                "SELECT response_json FROM scans WHERE id = ?",
                (scan_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["response_json"])
    except Exception as exc:
        logger.error("Failed to get scan %s: %s", scan_id, exc)
        return None


def get_scan_summary(scan_id: int) -> Optional[dict]:
    """Return the scan row metadata (no full response JSON)."""
    try:
        with _connection() as conn:
            row = conn.execute(
                """
                SELECT id, batch_id, created_at, analysis_mode,
                       total_observed, total_validated,
                       grade_a_count, urs_count, defect_count,
                       grade_a_pct, urs_pct, defect_pct,
                       overall_status, demo_mode, vision_model
                FROM scans WHERE id = ?
                """,
                (scan_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)
    except Exception as exc:
        logger.error("Failed to get scan summary %s: %s", scan_id, exc)
        return None


def delete_scan(scan_id: int) -> bool:
    """Delete a scan and its per-onion rows. Returns True on success."""
    try:
        with _connection() as conn:
            conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        return True
    except Exception as exc:
        logger.error("Failed to delete scan %s: %s", scan_id, exc)
        return False
