"""
Public Agent API schemas (CORE 2 #7).

These models freeze the JSON contract for ``POST /api/v1/agent/run`` and
``POST /api/v1/agent/stream``. They are auto-published in the OpenAPI spec
at ``/openapi.json`` and rendered in Swagger UI at ``/docs``.

Design principles:
    1. **Reuse the existing CORE 1 #2.5 EducationalProfile model** — no
       duplication, no field translation. The webui already accepts the
       same shape, so a teacher's profile JSON is interchangeable across
       both surfaces.
    2. **Mirror the internal ``StreamEvent`` taxonomy 1:1** — the SSE route
       serialises ``aix.webui.agent.service.StreamEvent`` instances into
       these models. Adding a new ``kind`` upstream means adding one new
       schema variant here; we will catch missing variants in tests by
       exhaustively asserting every ``kind`` value round-trips.
    3. **Backward compatibility with the existing GraphRAG context API** —
       this module only *adds* schemas; nothing in ``models.py`` /
       ``educational_profile.py`` is touched.

The discriminator field on the SSE event union is ``kind``. Clients can
``switch`` on it the same way the htmx webui dispatches per-card partials.

See: docs/product/ClickUp_Agentic_GraphRAG_Update.md → Subtask 7.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from aix.api.schemas.educational_profile import EducationalProfile

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AgentRunRequest(BaseModel):
    """
    Input to either ``POST /api/v1/agent/run`` (sync JSON) or
    ``POST /api/v1/agent/stream`` (SSE-encoded JSON events).

    Mandatory fields are ``query`` + ``domain``. Everything else has sensible
    defaults so the smallest valid call from /docs is just::

        {"query": "Crea una lezione su attenzione", "domain": "neuro"}

    A richer call attaches an Educational Profile so the planner / writer
    can specialise prompts to grade level, BES, classroom resources, etc.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description=(
            "Teacher query in natural language (Italian). Same shape the "
            "webui chat input accepts, e.g. 'Crea una lezione di 45 "
            "minuti sulla fotosintesi adattata a una classe con 2 "
            "studenti DSA'."
        ),
    )
    domain: Literal["neuro", "udl"] = Field(
        ...,
        description=(
            "Knowledge graph domain to retrieve from. 'neuro' is the "
            "neuroscience-of-learning KG; 'udl' is the Universal Design "
            "for Learning KG."
        ),
    )
    language: Literal["it", "en"] = Field(
        default="it",
        description="Output language for the lesson plan and agent reasoning.",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "Optional client-side correlation id, surfaced into the agent "
            "state. Useful for tracing a single request across logs / "
            "Langfuse (CORE 2 #11). When omitted, a UUID4 is generated."
        ),
    )
    educational_profile: Optional[EducationalProfile] = Field(
        default=None,
        description=(
            "CORE 1 #2.5 — per-request class/classroom context. Same Pydantic "
            "model the webui /webui/lesson/new form serialises. When omitted, "
            "the agent falls back to generic prompts."
        ),
    )
    teacher_provided_context: Optional[str] = Field(
        default=None,
        max_length=48000,
        description=(
            "CORE 2 #6.6 P3 — joined plain text from teacher-uploaded files "
            "(PDF / TXT / Markdown) that the Writer should treat as "
            "additional context. Truncated to 48k chars to fit the Writer "
            "prompt budget. NOT ingested into the Knowledge Graph."
        ),
    )
    max_revisions: Optional[int] = Field(
        default=None,
        ge=0,
        le=4,
        description=(
            "Cap on critic revision loops. ``None`` defers to the agent "
            "pipeline default (AIX_MAX_REVISIONS env var, default 1). "
            "Setting 0 disables critic revisions for fast smoke testing."
        ),
    )

    model_config = ConfigDict(
        # Single canonical example for the OpenAPI Schema tab and any
        # client-codegen tool (Postman / openapi-generator) that consumes
        # the bare schema. The /docs "Try it out" picker also shows the
        # ``minimal`` / ``rich`` dropdown driven by ``openapi_examples``
        # in routes/agent.py — this single example is the richer one
        # so a reader of the schema alone still sees the full surface.
        json_schema_extra={
            "example": {
                "query": (
                    "Crea una lezione di 45 minuti sulla fotosintesi "
                    "clorofilliana adattata a una classe con 2 studenti DSA"
                ),
                "domain": "neuro",
                "language": "it",
                "max_revisions": 2,
                "educational_profile": {
                    "group": {
                        "title": "3A Liceo Scientifico",
                        "students_number": 25,
                        "grade": "SECONDARIA_II_GRADO",
                        "disabilities": ["ADHD", "DSA"],
                        "class_features": ["MOTIVATA"],
                        "student_attributes": [
                            "PUNTI_DI_ECCELLENZA",
                            "PUNTI_DI_CADUTA",
                        ],
                    },
                    "classroom": {
                        "title": "Aula 101",
                        "forniture_mobility": "PARTIALLY",
                        "has_lim": True,
                        "has_wifi": True,
                        "has_suite": True,
                        "pc_station": False,
                        "own_device": "BES",
                    },
                    "time_available_minutes": 45,
                    "subject_area": "Scienze",
                    "specific_topic": "Fotosintesi",
                },
                "teacher_provided_context": (
                    "Estratto dal manuale (opzionale): 'La fotosintesi è "
                    "il processo con cui le piante convertono energia "
                    "luminosa in energia chimica…'"
                ),
            }
        }
    )


# ---------------------------------------------------------------------------
# Sync response (POST /api/v1/agent/run)
# ---------------------------------------------------------------------------


class CriticScores(BaseModel):
    """Critic-agent per-criterion scores. Shape matches the internal critic
    output and is left intentionally permissive — the critic prompt is the
    source of truth and may evolve in CORE 3."""

    model_config = ConfigDict(extra="allow")

    average_score: Optional[float] = Field(
        default=None,
        description="Average score on a 0-5 scale. Convert to % via "
        "``round(average_score / 5 * 100)`` if needed.",
    )


class MediaCounts(BaseModel):
    """Aggregated counts of curated media for the right sidebar / API summary."""

    videos: int = Field(default=0, ge=0)
    articles: int = Field(default=0, ge=0)
    oer: int = Field(default=0, ge=0)


class AgentRunMeta(BaseModel):
    """Run-summary metadata returned alongside the final lesson plan."""

    duration_seconds: float = Field(..., description="Wall-clock duration of the agent run.")
    approved: bool = Field(..., description="Whether the critic approved the final draft.")
    revision_count: int = Field(
        ..., ge=0, description="Number of critic-driven revisions performed."
    )
    scores: Optional[CriticScores] = Field(default=None, description="Critic per-criterion scores.")
    nodes_count: int = Field(default=0, ge=0, description="Knowledge-graph nodes retrieved.")
    recommendations_count: int = Field(default=0, ge=0)
    media_counts: MediaCounts = Field(default_factory=MediaCounts)
    search_queries_count: int = Field(default=0, ge=0)


class PlannerInfo(BaseModel):
    """Planner output included in the sync response for explainability."""

    intent: Optional[str] = None
    intent_label: Optional[str] = None
    scope: Optional[str] = None
    scope_label: Optional[str] = None
    scope_confidence: Optional[float] = None
    key_concepts: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    lesson_type: Optional[str] = None
    target_grade: Optional[str] = None


class RetrieverInfo(BaseModel):
    """Retriever output included in the sync response for explainability."""

    nodes_count: int = Field(default=0, ge=0)
    relationships_count: int = Field(default=0, ge=0)
    recommendations_count: int = Field(default=0, ge=0)
    media_counts: MediaCounts = Field(default_factory=MediaCounts)
    media: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full curated_media payload (videos / resources / citations / "
            "open_textbooks). Same shape consumed by the webui side panel."
        ),
    )
    top_concepts: list[str] = Field(default_factory=list)
    retrieval_confidence: Optional[str] = None


class AgentRunResponse(BaseModel):
    """
    Final result of ``POST /api/v1/agent/run`` (sync).

    Holds the full lesson plan plus the explainability fields the webui
    surfaces in its planner / retriever / critic cards. Consumers that only
    need the lesson plan can read ``lesson_plan_md`` and ignore the rest.
    """

    lesson_plan_md: str = Field(
        ...,
        description="Final lesson plan as Markdown. Empty string indicates "
        "the agent finished without producing output (rare).",
    )
    meta: AgentRunMeta
    planner: Optional[PlannerInfo] = Field(
        default=None,
        description="Planner-stage explainability. Absent only on early "
        "errors that bailed before the planner ran.",
    )
    retriever: Optional[RetrieverInfo] = Field(
        default=None,
        description="Retriever-stage explainability. Absent only on early "
        "errors that bailed before the retriever ran.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lesson_plan_md": "# Lezione: ...",
                "meta": {
                    "duration_seconds": 73.2,
                    "approved": True,
                    "revision_count": 0,
                    "nodes_count": 14,
                    "recommendations_count": 5,
                    "media_counts": {"videos": 3, "articles": 2, "oer": 4},
                    "search_queries_count": 4,
                },
                "planner": {
                    "intent": "lesson_creation",
                    "intent_label": "Creazione lezione",
                    "scope": "in_scope",
                    "scope_label": "Nel Knowledge Graph",
                    "key_concepts": ["attenzione", "memoria di lavoro"],
                    "search_queries": ["strategie attenzione DSA", "..."],
                },
                "retriever": {
                    "nodes_count": 14,
                    "relationships_count": 31,
                    "recommendations_count": 5,
                    "top_concepts": ["Attenzione sostenuta", "Self-regulation"],
                    "retrieval_confidence": "HIGH",
                },
            }
        }
    )


# ---------------------------------------------------------------------------
# SSE event payloads (POST /api/v1/agent/stream)
# ---------------------------------------------------------------------------
#
# Each variant corresponds to one ``StreamEvent.kind`` produced by
# ``stream_agent_events`` (see service.py). The discriminator is ``kind``;
# ``data`` carries the kind-specific payload. Clients should switch on
# ``kind`` and validate against the matching ``data`` model — or rely on
# Pydantic's discriminated-union machinery if they regenerate from the
# OpenAPI spec.
#
# We keep ``data`` as ``Dict[str, Any]`` rather than nesting another typed
# model per kind to avoid coupling the public contract to internal field
# renames in the agent layer. The contract guarantees only the *outer*
# shape (``kind`` + ``data`` + optional ``lesson_plan_md`` + ``error``);
# the *content* of ``data`` mirrors ``StreamEvent.payload`` and is
# documented in the StreamEvent docstring.


class _SSEEventBase(BaseModel):
    """Base class for SSE event variants. Not used directly by clients."""

    kind: str
    data: dict[str, Any] = Field(default_factory=dict)
    lesson_plan_md: Optional[str] = None
    error: Optional[str] = None


class PlannerEvent(_SSEEventBase):
    kind: Literal["planner"] = "planner"


class RetrieverEvent(_SSEEventBase):
    kind: Literal["retriever"] = "retriever"


class WriterPendingEvent(_SSEEventBase):
    kind: Literal["writer_pending"] = "writer_pending"


class WriterEvent(_SSEEventBase):
    kind: Literal["writer"] = "writer"


class CriticEvent(_SSEEventBase):
    kind: Literal["critic"] = "critic"


class DoneEvent(_SSEEventBase):
    kind: Literal["done"] = "done"


class ErrorEvent(_SSEEventBase):
    kind: Literal["error"] = "error"


# Public union — exported so partner SDKs / OpenAPI codegen pick it up.
AgentStreamEvent = Union[
    PlannerEvent,
    RetrieverEvent,
    WriterPendingEvent,
    WriterEvent,
    CriticEvent,
    DoneEvent,
    ErrorEvent,
]


__all__ = [
    "AgentRunRequest",
    "AgentRunResponse",
    "AgentRunMeta",
    "CriticScores",
    "MediaCounts",
    "PlannerInfo",
    "RetrieverInfo",
    "AgentStreamEvent",
    "PlannerEvent",
    "RetrieverEvent",
    "WriterPendingEvent",
    "WriterEvent",
    "CriticEvent",
    "DoneEvent",
    "ErrorEvent",
]
