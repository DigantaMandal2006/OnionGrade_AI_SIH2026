"""
OnionGrade AI Report Service

Gemini/LangChain is used ONLY to make the
already-computed results readable.

Python remains the source of truth.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ============================================================
# ENV
# ============================================================

_ENV_PATH = (
    Path(__file__).resolve().parents[2]
    / ".env"
)

load_dotenv(
    dotenv_path=_ENV_PATH
)

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

# gemini-3.6-flash: confirmed working for text generation on this API key.
# Override with GEMINI_MODEL env var.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


# ============================================================
# VERDICT
# ============================================================

def _verdict_line(
    context: dict
) -> str:

    total_validated = context.get(
        "total_validated",
        0
    )

    grade_a_pct = context.get(
        "grade_a_pct",
        0
    )

    defect_pct = context.get(
        "defect_pct",
        0
    )

    if total_validated == 0:

        return (
            "No valid onions were available "
            "for grading."
        )

    if grade_a_pct >= 70:

        return (
            "Overall batch quality is strong — "
            "the majority of validated onions "
            "meet Grade A criteria."
        )

    if defect_pct >= 30:

        return (
            "Batch quality is a concern — "
            "a high proportion of onions "
            "are defective."
        )

    return (
        "Batch quality is mixed — "
        "a meaningful proportion falls under "
        "URS or defective categories."
    )


# ============================================================
# DEFECT ANALYSIS
# ============================================================

def _format_defect_analysis(
    defect_analysis: Optional[dict]
) -> str:

    if not defect_analysis:

        return (
            "No defect analysis data was available."
        )

    total_defective = defect_analysis.get(
        "total_defective",
        0
    )

    categories = defect_analysis.get(
        "categories",
        {}
    )

    if not categories:

        return (
            f"Total defective onions: "
            f"{total_defective}. "
            "No specific defect categories "
            "were recorded."
        )

    category_text = ", ".join(
        f"{name}: {count}"
        for name, count
        in categories.items()
    )

    return (
        f"Total defective onions: "
        f"{total_defective}. "
        f"Defect categories: "
        f"{category_text}."
    )


# ============================================================
# RECOMMENDATION
# ============================================================

def _default_recommendation(
    overall_status: str,
    defect_analysis: Optional[dict]
) -> str:

    if overall_status == "GOOD":

        return (
            "Batch shows generally good quality. "
            "Remove any identified defective "
            "onions before final processing."
        )

    if overall_status == "POOR":

        return (
            "Inspect the batch carefully and "
            "remove defective onions before "
            "processing."
        )

    return (
        "Perform manual quality inspection "
        "and remove defective onions before "
        "distribution."
    )


# ============================================================
# TEMPLATE REPORT
# ============================================================

REPORT_TEMPLATE = """
Quality Assessment Report — Batch {batch_id}

BATCH SUMMARY

Generated: {generated_at}
Total onions observed: {total_observed}
Passed validation: {total_validated}
Rejected before grading: {rejected_count}

GRADE DISTRIBUTION

Grade A (Premium): {grade_a_count} onions ({grade_a_pct}%)
URS: {urs_count} onions ({urs_pct}%)
Defective: {defect_count} onions ({defect_pct}%)

OVERALL ASSESSMENT

Status: {overall_status}
{verdict}

DEFECT ANALYSIS

{defect_summary}

RECOMMENDATION

{recommendation}

AUDIT NOTE

All grading figures were computed by the deterministic
Python rule engine. Gemini Vision provided only the
raw per-onion visual observations. Every result is
traceable to an individual onion record.
"""


def _template_report(
    context: dict
) -> str:

    rejected_count = (
        context.get(
            "total_observed",
            0
        )
        -
        context.get(
            "total_validated",
            0
        )
    )

    defect_summary = (
        _format_defect_analysis(
            context.get(
                "defect_analysis"
            )
        )
    )

    return REPORT_TEMPLATE.format(

        batch_id=context.get(
            "batch_id",
            "Unknown"
        ),

        generated_at=context.get(
            "generated_at",
            "N/A"
        ),

        total_observed=context.get(
            "total_observed",
            0
        ),

        total_validated=context.get(
            "total_validated",
            0
        ),

        rejected_count=rejected_count,

        grade_a_count=context.get(
            "grade_a_count",
            0
        ),

        grade_a_pct=context.get(
            "grade_a_pct",
            0
        ),

        urs_count=context.get(
            "urs_count",
            0
        ),

        urs_pct=context.get(
            "urs_pct",
            0
        ),

        defect_count=context.get(
            "defect_count",
            0
        ),

        defect_pct=context.get(
            "defect_pct",
            0
        ),

        overall_status=context.get(
            "overall_status",
            "ACCEPTABLE"
        ),

        verdict=context.get(
            "verdict",
            ""
        ),

        defect_summary=defect_summary,

        recommendation=context.get(
            "recommendation",
            ""
        )
    ).strip()


# ============================================================
# MAIN REPORT FUNCTION
# ============================================================

def generate_report(
    grade_result: dict,
    batch_id: str,
    total_observed: int,
    defect_analysis: Optional[dict] = None,
    overall_status: Optional[str] = None,
    recommendation: Optional[str] = None
) -> str:

    from datetime import datetime

    if overall_status is None:

        overall_status = (
            "GOOD"
            if grade_result.get(
                "grade_a_pct",
                0
            ) >= 70
            else "ACCEPTABLE"
        )

    if recommendation is None:

        recommendation = (
            _default_recommendation(
                overall_status,
                defect_analysis
            )
        )

    context = {

        **grade_result,

        "batch_id": batch_id,

        "total_observed": (
            total_observed
        ),

        "generated_at": (
            datetime.now().strftime(
                "%d %b %Y, %H:%M"
            )
        ),

        "overall_status": (
            overall_status
        ),

        "defect_analysis": (
            defect_analysis or {}
        ),

        "recommendation": (
            recommendation
        ),

        "verdict": (
            _verdict_line(
                grade_result
            )
        )
    }

    # --------------------------------------------------------
    # Deterministic fallback
    # --------------------------------------------------------

    if not GOOGLE_API_KEY:

        return _template_report(
            context
        )

    # --------------------------------------------------------
    # LLM report
    # --------------------------------------------------------

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            request_timeout=30,
        )

        # Build a compact defect summary for the prompt — only pass what's
        # needed so the LLM cannot invent extra numbers.
        defect_categories = list(
            (context.get("defect_analysis") or {}).get("categories", {}).items()
        )
        defect_summary_for_prompt = (
            ", ".join(f"{k}: {v}" for k, v in defect_categories)
            if defect_categories
            else "none"
        )

        prompt = ChatPromptTemplate.from_template(
            """You are an agricultural quality analyst writing a brief batch-level
agronomic interpretation for an onion grading report.

STRICT RULES — do NOT break these:
- Use ONLY the data supplied below. Never invent, estimate or change any number.
- Do NOT recalculate or re-derive percentages or counts.
- Do NOT assign new grades.
- Grade A%, URS%, Defect%, counts and overall status are already correct — quote them exactly.
- Keep the interpretation to 3-5 concise sentences.

YOUR SPECIFIC JOB:
Based on the defect pattern (types and counts below), write a short agronomic
interpretation: what does this pattern suggest about storage, handling, or
transit conditions? What should the buyer / warehouse manager do next?
This is the one thing the deterministic Python engine cannot produce.

BATCH DATA (do not change these figures):
- Batch ID: {batch_id}
- Date: {generated_at}
- Observed: {total_observed} | Validated: {total_validated}
- Grade A: {grade_a_count} onions ({grade_a_pct}%)
- URS: {urs_count} onions ({urs_pct}%)
- Defect: {defect_count} onions ({defect_pct}%)
- Overall status: {overall_status}
- Defect categories observed: {defect_summary}

Write the report now. Start with "Batch {batch_id} —" and end with
the specific recommendation for this defect pattern."""
        )

        chain = prompt | llm
        result = chain.invoke({
            "batch_id": context.get("batch_id", "Unknown"),
            "generated_at": context.get("generated_at", "N/A"),
            "total_observed": context.get("total_observed", 0),
            "total_validated": context.get("total_validated", 0),
            "grade_a_count": context.get("grade_a_count", 0),
            "grade_a_pct": context.get("grade_a_pct", 0),
            "urs_count": context.get("urs_count", 0),
            "urs_pct": context.get("urs_pct", 0),
            "defect_count": context.get("defect_count", 0),
            "defect_pct": context.get("defect_pct", 0),
            "overall_status": context.get("overall_status", "ACCEPTABLE"),
            "defect_summary": defect_summary_for_prompt,
        })

        content = result.content
        if isinstance(content, list):
            content = " ".join(str(item) for item in content)

        # Prepend structured header so the report is always machine-readable
        # even when LLM text is present.
        header = _template_report(context).split("OVERALL ASSESSMENT")[0].strip()
        return header + "\n\nOVERALL ASSESSMENT\n\n" + str(content).strip()

    except Exception:
        return _template_report(context)