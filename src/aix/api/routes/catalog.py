"""The choices the WebUI offers, for a client to offer the same (2026-10-14).

A chatbot that builds a session needs the codes the API accepts and the words
the WebUI shows for them: the domains, the pedagogical intents, the refinement
chips, the codes of the educational profile, the turn modes, the formats and
limits of the documents. They come from the same constants the WebUI renders,
so a label changed there is changed here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from aix.api.principals import require_caller
from aix.api.schemas.educational_profile import (
    CLASS_FEATURE_LABELS,
    DISABILITY_CHOICES,
    DISABILITY_LABELS,
    ENVIRONMENT_TYPE_LABELS,
    FORNITURE_MOBILITY_LABELS,
    GRADE_LABELS,
    LEGACY_DISABILITY_CODES,
    OWN_DEVICE_LABELS,
    PARTICIPATION_TYPE_LABELS,
    PEDAGOGICAL_INTENT_OPTIONS,
    PREVALENT_MODALITY_LABELS,
    STUDENT_ATTR_LABELS,
)
from aix.webui.agent.turns import MAX_QUERY_CHARS, REFINEMENT_LABELS

router = APIRouter(prefix="/catalog", tags=["catalog"])

DOMAINS = [
    {
        "code": "neuro",
        "label": "Neuroscienze dell'apprendimento",
        "description": "Memoria, attenzione, funzioni esecutive, BES e strategie didattiche.",
    },
    {
        "code": "udl",
        "label": "Universal Design for Learning",
        "description": "I principi UDL: rappresentazione, azione ed espressione, coinvolgimento.",
    },
]

TURN_MODES = [
    {"code": "auto", "label": "Come la WebUI: lezione al primo turno, poi risposte veloci"},
    {"code": "answer", "label": "Rispondi qui, senza rigenerare la lezione"},
    {"code": "lesson", "label": "Lezione completa (con il Knowledge Graph)"},
]


@router.get("", summary="Codes and labels the API accepts, as the WebUI shows them")
async def catalog(_caller=Depends(require_caller)) -> dict[str, Any]:
    from aix.webui.lessons.uploads import allowed_extensions
    from aix.webui.materials.parsing.intake import MAX_BYTES

    return {
        "domains": DOMAINS,
        "pedagogical_intents": PEDAGOGICAL_INTENT_OPTIONS,
        "refinements": [
            {"code": code, "label": label} for code, label in REFINEMENT_LABELS.items()
        ],
        "turn_modes": TURN_MODES,
        "profile": {
            "grade": GRADE_LABELS,
            "disabilities": DISABILITY_CHOICES,
            # Still accepted (saved profiles carry them), no longer offered.
            "disabilities_legacy": {
                code: DISABILITY_LABELS[code] for code in sorted(LEGACY_DISABILITY_CODES)
            },
            "class_features": CLASS_FEATURE_LABELS,
            "student_attributes": STUDENT_ATTR_LABELS,
            "forniture_mobility": FORNITURE_MOBILITY_LABELS,
            "own_device": OWN_DEVICE_LABELS,
            "environment_type": ENVIRONMENT_TYPE_LABELS,
            "participation_type": PARTICIPATION_TYPE_LABELS,
            "prevalent_modality_type": PREVALENT_MODALITY_LABELS,
            # Classroom yes/no questions take true/false or, as the platform
            # stores them, YES / NO / UNKNOWN; UNKNOWN and omitted mean "not said".
            "three_state": ["YES", "NO", "UNKNOWN"],
        },
        "materials": {
            "extensions": sorted(allowed_extensions()),
            "max_bytes": MAX_BYTES,
        },
        "limits": {"message_max_chars": MAX_QUERY_CHARS},
    }
