"""
Educational Profile Schema (CORE 1 #2.5 — ported)

Defines the per-request educational context that flows through both API surfaces:
    - GraphRAG mode: POST /api/v1/context     (optional; falls back to generic when absent)
    - Agent mode:    POST /api/v1/agent/...   (optional; richer prompt specialization when present)

Field shapes are 1:1 with the AixLearning native production models so no
field translation is required when the FEM platform consumes this schema:
    - party.models.Party              ↔ EducationalGroup
    - party.models.StudentDisabilities ↔ DisabilityType (10 BES types)
    - party.models.PartyFeature       ↔ ClassFeature
    - party.models.StudentAttribute   ↔ StudentAttribute
    - classroom.models.Classroom      ↔ ClassroomEnvironment

Source: ported from `FEM-modena/graphrag-aixlearning` branch `Angelo`,
    file `api/schemas/educational_profile.py` (original author: AG).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ============================================================================
# GRADE LEVELS (6 options) — party.models.Party.grade
# ============================================================================


class GradeLevel(str, Enum):
    """Grade levels matching `party.models.Party.grade`."""

    INFANZIA = "INFANZIA"
    PRIMARIA = "PRIMARIA"
    SECONDARIA_I_GRADO = "SECONDARIA_I_GRADO"
    SECONDARIA_II_GRADO = "SECONDARIA_II_GRADO"
    UNIVERSITA = "UNIVERSITA"
    FORMAZIONE_SUL_LAVORO = "FORMAZIONE_SUL_LAVORO"


# ============================================================================
# DISABILITIES / BES (10 options) — party.models.StudentDisabilities
# ============================================================================


class DisabilityType(str, Enum):
    """BES (Bisogni Educativi Speciali) types matching `party.models.StudentDisabilities`."""

    DSA = "DSA"  # Disturbi specifici dell'apprendimento
    ADHD = "ADHD"  # Disturbo dell'attenzione
    DOP = "DOP"  # Disturbo oppositivo provocatorio
    DF = "DF"  # Disabilità fisica
    DCGL = "DCGL"  # Disabilità cognitiva di grado lieve
    DCGM = "DCGM"  # Disabilità cognitiva di grado medio
    DCGS = "DCGS"  # Disabilità cognitiva di grado severo
    DLDS = "DLDS"  # Difficoltà linguistiche (studente straniero/a)
    PD = "PD"  # Plusdotazione
    SA = "SA"  # Disturbo dello spettro autistico


# ============================================================================
# CLASS FEATURES (5 options) — party.models.PartyFeature
# ============================================================================


class ClassFeature(str, Enum):
    """Class characteristics matching `party.models.PartyFeature`."""

    COESA = "COESA"
    DIVISA_IN_GRUPPI = "DIVISA_IN_GRUPPI"
    ELEMENTI_DI_DISTURBO = "ELEMENTI_DI_DISTURBO"
    MOTIVATA = "MOTIVATA"
    GENDER_GAP = "GENDER_GAP"


# ============================================================================
# STUDENT ATTRIBUTES (6 options) — party.models.StudentAttribute
# ============================================================================


class StudentAttribute(str, Enum):
    """Student-population attributes matching `party.models.StudentAttribute`."""

    PUNTI_DI_ECCELLENZA = "PUNTI_DI_ECCELLENZA"
    PUNTI_DI_CADUTA = "PUNTI_DI_CADUTA"
    SPINTA_MOTIVAZIONALE = "SPINTA_MOTIVAZIONALE"
    MANCANZA_DI_MOTIVAZIONE = "MANCANZA_DI_MOTIVAZIONE"
    TIMIDEZZA_O_CHIUSURA = "TIMIDEZZA_O_CHIUSURA"
    PROBLEMI_FAMILIARI = "PROBLEMI_FAMILIARI"


# ============================================================================
# FURNITURE MOBILITY (3 options) — classroom.models.Classroom.forniture_mobility
# ============================================================================


class FornitureMobility(str, Enum):
    """Furniture mobility matching `classroom.models.Classroom.forniture_mobility`.

    Field name preserved (`Forniture` with the typo) to keep 1:1 mapping with
    the AixLearning native model field. Do not rename without coordinating
    with the FEM platform team.
    """

    YES = "YES"
    NO = "NO"
    PARTIALLY = "PARTIALLY"


# ============================================================================
# DEVICE POLICY (3 options) — classroom.models.Classroom.own_device
# ============================================================================


class OwnDevicePolicy(str, Enum):
    """BYOD policy matching `classroom.models.Classroom.own_device`."""

    YES = "YES"
    NO = "NO"
    BES = "BES"


# ============================================================================
# EDUCATIONAL GROUP — party.models.Party
# ============================================================================


class EducationalGroup(BaseModel):
    """Educational group (class) matching `party.models.Party`."""

    title: Optional[str] = Field(
        None,
        description="Group / class name (e.g. '3A Liceo Scientifico').",
    )
    students_number: int = Field(
        default=20,
        ge=1,
        le=40,
        description="Number of students in the group.",
    )
    grade: Optional[GradeLevel] = Field(
        None,
        description="Grade level (school level).",
    )
    disabilities: list[DisabilityType] = Field(
        default_factory=list,
        description="Special Educational Needs (BES) present in the class.",
    )
    class_features: list[ClassFeature] = Field(
        default_factory=list,
        description="Class characteristics (cohesion, motivation, etc.).",
    )
    student_attributes: list[StudentAttribute] = Field(
        default_factory=list,
        description="Population-level attributes (excellence/difficulty, motivation, etc.).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "3A Liceo Scientifico",
                "students_number": 25,
                "grade": "SECONDARIA_II_GRADO",
                "disabilities": ["ADHD", "DSA"],
                "class_features": ["MOTIVATA"],
                "student_attributes": ["PUNTI_DI_ECCELLENZA", "PUNTI_DI_CADUTA"],
            }
        }


# ============================================================================
# CLASSROOM ENVIRONMENT — classroom.models.Classroom
# ============================================================================


class ClassroomEnvironment(BaseModel):
    """Classroom environment matching `classroom.models.Classroom`."""

    title: Optional[str] = Field(
        None,
        description="Classroom name / identifier (e.g. 'Aula 101').",
    )
    forniture_mobility: FornitureMobility = Field(
        default=FornitureMobility.NO,
        description="Whether furniture can be moved / rearranged.",
    )
    has_lim: Optional[bool] = Field(
        None,
        description="Has Interactive Whiteboard (LIM).",
    )
    has_wifi: Optional[bool] = Field(
        None,
        description="Has WiFi connection.",
    )
    has_suite: Optional[bool] = Field(
        None,
        description="Has Office Suite (Google Workspace / Microsoft 365).",
    )
    pc_station: Optional[bool] = Field(
        None,
        description="Has PC workstations.",
    )
    own_device: OwnDevicePolicy = Field(
        default=OwnDevicePolicy.NO,
        description="Student personal-device policy (BYOD).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Aula 101",
                "forniture_mobility": "PARTIALLY",
                "has_lim": True,
                "has_wifi": True,
                "has_suite": True,
                "pc_station": False,
                "own_device": "BES",
            }
        }


# ============================================================================
# COMPLETE EDUCATIONAL PROFILE
# ============================================================================


class EducationalProfile(BaseModel):
    """
    Complete educational context combining group and environment.

    This is the main model that gets attached to API requests to provide
    rich, personalized educational context. Every field is optional —
    a missing profile must fall back to current generic behavior
    (backward-compat acceptance criterion of CORE 1 #2.5).
    """

    group: Optional[EducationalGroup] = Field(
        None,
        description="Educational group / class profile.",
    )
    classroom: Optional[ClassroomEnvironment] = Field(
        None,
        description="Classroom environment.",
    )
    time_available_minutes: Optional[int] = Field(
        None,
        ge=15,
        le=480,
        description="Available time for the activity in minutes.",
    )
    subject_area: Optional[str] = Field(
        None,
        description="Subject area (e.g. 'Matematica', 'Storia').",
    )
    specific_topic: Optional[str] = Field(
        None,
        description="Specific topic within the subject.",
    )
    pedagogical_intent: Optional[str] = Field(
        None,
        description=(
            "Teacher's pedagogical intent for this lesson. Stored as "
            "'{code}: {optional_detail}' (e.g. 'misconception: practice alone "
            "doesn't increase WM capacity'). Code alone is valid when no detail "
            "is provided."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "group": {
                    "students_number": 25,
                    "grade": "SECONDARIA_I_GRADO",
                    "disabilities": ["ADHD", "DSA"],
                    "class_features": ["ELEMENTI_DI_DISTURBO"],
                    "student_attributes": ["MANCANZA_DI_MOTIVAZIONE"],
                },
                "classroom": {
                    "forniture_mobility": "PARTIALLY",
                    "has_lim": True,
                    "has_wifi": True,
                    "has_suite": False,
                    "pc_station": False,
                    "own_device": "NO",
                },
                "time_available_minutes": 60,
                "subject_area": "Scienze",
                "specific_topic": "Fotosintesi",
            }
        }


# ============================================================================
# ITALIAN LABEL MAPPINGS (for UI rendering — webui form, AixLearning embed)
# ============================================================================


GRADE_LABELS: dict[str, str] = {
    "INFANZIA": "Infanzia",
    "PRIMARIA": "Primaria",
    "SECONDARIA_I_GRADO": "Secondaria di primo grado",
    "SECONDARIA_II_GRADO": "Secondaria di secondo grado",
    "UNIVERSITA": "Università",
    "FORMAZIONE_SUL_LAVORO": "Formazione per adulti",
}

DISABILITY_LABELS: dict[str, str] = {
    "DSA": "Disturbi specifici dell'apprendimento (DSA)",
    "ADHD": "Disturbo dell'attenzione (ADHD)",
    "DOP": "Disturbo oppositivo provocatorio (DOP)",
    "DF": "Disabilità fisica",
    "DCGL": "Disabilità cognitiva lieve",
    "DCGM": "Disabilità cognitiva media",
    "DCGS": "Disabilità cognitiva severa",
    "DLDS": "Difficoltà linguistiche (straniero/a)",
    "PD": "Plusdotazione",
    "SA": "Disturbo dello spettro autistico",
}

CLASS_FEATURE_LABELS: dict[str, str] = {
    "COESA": "Classe coesa",
    "DIVISA_IN_GRUPPI": "Divisa in gruppi",
    "ELEMENTI_DI_DISTURBO": "Con elementi di disturbo",
    "MOTIVATA": "Motivata",
    "GENDER_GAP": "Gender gap",
}

STUDENT_ATTR_LABELS: dict[str, str] = {
    "PUNTI_DI_ECCELLENZA": "Punti di eccellenza in alcune discipline",
    "PUNTI_DI_CADUTA": "Punti di caduta / difficoltà",
    "SPINTA_MOTIVAZIONALE": "Spinta motivazionale",
    "MANCANZA_DI_MOTIVAZIONE": "Mancanza di motivazione",
    "TIMIDEZZA_O_CHIUSURA": "Timidezza / chiusura relazionale",
    "PROBLEMI_FAMILIARI": "Problematiche familiari",
}

FORNITURE_MOBILITY_LABELS: dict[str, str] = {
    "YES": "Sì",
    "NO": "No",
    "PARTIALLY": "Parzialmente",
}

OWN_DEVICE_LABELS: dict[str, str] = {
    "YES": "Sì",
    "NO": "No",
    "BES": "Solo studenti con BES",
}

# ============================================================================
# PEDAGOGICAL INTENT OPTIONS (for UI chip selector)
# code → {"label": ..., "prompt": ...}
# "prompt" is injected verbatim into the writer prompt when the code is used.
# ============================================================================

PEDAGOGICAL_INTENT_OPTIONS: list[dict] = [
    {
        "code": "intro",
        "label": "Prima introduzione",
        "description": "Nessun prerequisito richiesto",
        "prompt": "Introduce the concept for the first time — assume no prior knowledge",
    },
    {
        "code": "deepening",
        "label": "Approfondimento",
        "description": "La classe conosce già il concetto",
        "prompt": "Deepen understanding of a concept the class has already encountered",
    },
    {
        "code": "misconception",
        "label": "Correggere un'idea errata",
        "description": "Sfidare un errore concettuale frequente",
        "prompt": "Challenge and correct a common misconception about this topic",
    },
    {
        "code": "assessment",
        "label": "Preparazione verifica",
        "description": "Consolidare prima della valutazione",
        "prompt": "Consolidate knowledge and prepare students for an upcoming assessment",
    },
    {
        "code": "practice",
        "label": "Applicazione pratica",
        "description": "Focus su esercizi e attività concrete",
        "prompt": "Focus on practical exercises, activities, and concrete application",
    },
    {
        "code": "interdisciplinary",
        "label": "Collegamento interdisciplinare",
        "description": "Collegare più materie o ambiti",
        "prompt": "Build bridges between this concept and other subject areas",
    },
]

# Quick lookup by code
PEDAGOGICAL_INTENT_BY_CODE: dict[str, dict] = {o["code"]: o for o in PEDAGOGICAL_INTENT_OPTIONS}


__all__ = [
    # Enums
    "GradeLevel",
    "DisabilityType",
    "ClassFeature",
    "StudentAttribute",
    "FornitureMobility",
    "OwnDevicePolicy",
    # Models
    "EducationalGroup",
    "ClassroomEnvironment",
    "EducationalProfile",
    # Labels
    "GRADE_LABELS",
    "DISABILITY_LABELS",
    "CLASS_FEATURE_LABELS",
    "STUDENT_ATTR_LABELS",
    "FORNITURE_MOBILITY_LABELS",
    "OWN_DEVICE_LABELS",
    "PEDAGOGICAL_INTENT_OPTIONS",
    "PEDAGOGICAL_INTENT_BY_CODE",
]
