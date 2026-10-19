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

    **Unknown fields are kept and logged, not rejected — for now.** A field the
    caller misspells (``languag`` for ``language``) is silently dropped under
    Pydantic's default, so the lesson comes back in the wrong language and
    nothing anywhere says why. ``extra="forbid"`` would turn that into an
    immediate 422, which is the behaviour this API should end up with.

    It is not on yet because we do not know what the integrating worker
    actually sends: one undeclared field in its payload — a trace id, a client
    version, a leftover — and every call it makes starts failing, on code we do
    not control. So the route logs the *names* of unknown fields on each
    request (never the values: an unknown field may carry a teacher's data),
    and once the logs show nobody is sending any, this becomes
    ``extra="forbid"`` and the guesswork ends (2026-09-22).

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
            "The conversation's identity, not a log tag. It becomes LangGraph's "
            "``thread_id``: send the same value again and the run continues that "
            "conversation, with its checkpointed history and possibly its reused "
            "retrieval corpus. **Two callers that send the same value share a "
            "conversation**, so an integrator must scope it per end user — a "
            "counter, or a key that is not guessable from outside. When omitted, "
            "a UUID4 is generated and the run is a fresh conversation. It is also "
            "the correlation id in the logs (CORE 2 #11)."
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
        # Unknown fields are kept rather than dropped, so the route can name
        # them in the log; see the class docstring. It lives in *this*
        # ConfigDict and not in a second one earlier in the class, because a
        # class body has one ``model_config`` and the last assignment wins —
        # an earlier one is simply dead code.
        extra="allow",
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
        },
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
    udl_coverage: Optional[dict[str, Any]] = Field(
        default=None,
        description="How the lesson covers the three UDL principles (representation, "
        "action, engagement), as the WebUI's lesson card shows it.",
    )
    review_notes: list[ReviewNote] = Field(
        default_factory=list,
        description="Problems the judge left for the teacher to read (AIX_JUDGE_ENABLED): "
        "minor ones, and blocking ones no repair could make safely. Empty when the judge "
        "is off or found nothing.",
    )
    judge: Optional[dict[str, Any]] = Field(
        default=None,
        description="What the judge did: found, blocking, replaced, revised, second_check, "
        "undone, cost, seconds. Null when the judge is off.",
    )


class ReviewNote(BaseModel):
    """One problem the judge found, quoting the lesson (``validation.judge``)."""

    model_config = ConfigDict(extra="ignore")

    check: str = Field(..., description="The checklist question, e.g. «esempi_lingua», «esercizi».")
    severity: str = Field(..., description="«bloccante» or «minore».")
    quote: str = Field(..., description="The sentence of the lesson, as written.")
    why: str = ""
    fix: str = Field("", description="The corrected text, or what to change.")
    kind: str = Field("revisione", description="«sostituzione» or «revisione».")


AgentRunMeta.model_rebuild()


# The three cards below take the WebUI's payloads as they are (``extra="ignore"``):
# every field the WebUI card shows is a field here, and
# ``tests/unit/test_api_webui_parity.py`` fails when the WebUI gains one that is not.


class PlannerInfo(BaseModel):
    """What the planner understood: the WebUI's planner card."""

    model_config = ConfigDict(extra="ignore")

    intent: Optional[str] = Field(None, description="lesson_creation, activity_design, …")
    intent_label: Optional[str] = Field(None, description="The intent in Italian.")
    scope: Optional[str] = Field(None, description="in_scope · partial_scope · out_of_scope")
    scope_label: Optional[str] = None
    scope_variant: Optional[str] = Field(
        None, description="success · warning · danger: how the card colours the scope."
    )
    scope_confidence: Optional[float] = None
    key_concepts: list[str] = Field(default_factory=list)
    subject_concepts: list[str] = Field(
        default_factory=list,
        description="The subject the lesson is about (searched outside the graph when out of scope).",
    )
    search_queries: list[str] = Field(default_factory=list)
    lesson_type: Optional[str] = None
    target_grade: Optional[str] = None


class RetrieverInfo(BaseModel):
    """What the search found: the WebUI's retriever card."""

    model_config = ConfigDict(extra="ignore")

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
    media_total: int = Field(default=0, ge=0)
    kg_unreachable: bool = Field(
        False,
        description="The Knowledge Graph did not answer: the lesson was written without it. "
        "Not the same as an out-of-scope question.",
    )
    retrieval_outcome: Optional[str] = Field(
        None,
        description="success · adapted_with_hybrid (outside sources filled in) · "
        "limited_kg_only · grader_error",
    )
    coverage_tier: Optional[str] = Field(
        None, description="How much of the request the graph covers, for the coverage banner."
    )
    domain: Optional[str] = None
    domain_label_short: Optional[str] = None
    domain_label_long: Optional[str] = None
    reused: bool = Field(
        False, description="This turn reused an earlier turn's search instead of searching again."
    )
    reuse_reason: Optional[str] = Field(None, description="Why the planner chose to reuse it.")
    retrieval_grade: Optional[str] = Field(
        None,
        description="Corrective RAG only (off in the pilot): relevant · ambiguous · irrelevant.",
    )
    retrieval_grade_label: Optional[str] = None
    retrieval_grade_reason: Optional[str] = None
    retrieval_attempts: Optional[int] = None
    retrieval_attempts_max: Optional[int] = None
    retrieval_warning: bool = False
    retrieval_rewritten_query: Optional[str] = None


class CriticInfo(BaseModel):
    """The critic's judgement: the WebUI's critic card."""

    model_config = ConfigDict(extra="ignore")

    approved: bool = False
    revision_count: int = Field(0, ge=0)
    max_revisions: int = Field(1, ge=0)
    score: Optional[float] = Field(None, description="0-5.")
    score_pct: Optional[int] = Field(None, description="The score as a percentage.")
    critique: str = Field("", description="The critic's assessment, in the lesson's language.")
    revision_instructions: str = Field(
        "", description="What the critic asked the writer to change, when it did."
    )


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
    critic: Optional[CriticInfo] = Field(
        default=None, description="The critic's last judgement on the lesson."
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
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Run summary, populated on the terminal ``done`` event: "
            "``duration_seconds``, ``approved``, ``revision_count``, ``scores``, "
            "``nodes_count``, ``media_counts``. Empty on every other event. "
            "Until 2026-09-21 this was dropped on the SSE path, so a client that "
            "wanted both the summary and incremental phases had to call ``/run`` "
            "as well."
        ),
    )


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
