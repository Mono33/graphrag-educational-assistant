"""Request and response shapes of the sessions API (2026-10-14).

A session is a conversation with the agent about one lesson: the same object as
a lesson in the WebUI, with its turns, its current lesson and its documents.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from aix.api.schemas.agent import AgentRunMeta, CriticInfo, JudgeInfo, PlannerInfo, RetrieverInfo
from aix.api.schemas.educational_profile import EducationalProfile

#: The AI Act Art. 50 notice, in the body as well as the ``X-AI-Generated`` header.
AI_DISCLOSURE = "Contenuto generato dall'intelligenza artificiale."


class SessionCreate(BaseModel):
    domain: Literal["neuro", "udl"] = Field(
        "neuro", description="Knowledge Graph domain: neuro (neuroscience) or udl."
    )
    title: Optional[str] = Field(None, max_length=200)
    educational_profile: Optional[EducationalProfile] = Field(
        None,
        description=(
            "The class and classroom, as in the WebUI's profile form. Set "
            "``pedagogical_intent`` here: the API does not ask for it with a card."
        ),
    )
    material_ids: list[str] = Field(
        default_factory=list,
        description="Documents from ``/api/v1/materials`` this session's turns may search.",
    )


class SessionMaterials(BaseModel):
    material_ids: list[str] = Field(..., min_length=1)


class Refinement(BaseModel):
    code: Literal["simplify", "deepen", "more_activities", "adapt_class", "custom"]
    detail: Optional[str] = Field(None, max_length=2000)


class TurnRequest(BaseModel):
    message: Optional[str] = Field(
        None,
        max_length=2000,
        description="What the user writes. May be omitted only with a ``refinement``.",
    )
    mode: Literal["auto", "answer", "lesson"] = Field(
        "auto",
        description=(
            "auto: as the WebUI -- the first turn writes a lesson, later turns answer in a "
            "few seconds, and a request for a new lesson comes back as ``handoff``. "
            "answer: answer here, never hand off (needs a lesson). lesson: the whole chain, "
            "a new version of the lesson."
        ),
    )
    refinement: Optional[Refinement] = Field(
        None, description="A refinement chip of the WebUI: rewrites the whole lesson."
    )
    search_graph: Optional[bool] = Field(
        None,
        description=(
            "On a follow-up that rewrites the lesson: false reuses what the Knowledge Graph "
            "already found, true searches again. Omitted: the agent decides."
        ),
    )
    material_ids: list[str] = Field(
        default_factory=list, description="Documents to bind to the session before this turn."
    )


class LessonSection(BaseModel):
    title: str
    level: int = Field(..., description="2 for a ## section, 3 for a ### part of it.")
    minutes: Optional[int] = Field(None, description="The minutes the section declares, if any.")


class LessonStructure(BaseModel):
    """The lesson as sections and minutes, read from its Markdown (the export model)."""

    title: str
    duration_minutes: Optional[int] = None
    sections: list[LessonSection] = Field(default_factory=list)
    timed_outline: list[LessonSection] = Field(
        default_factory=list,
        description="The timed sections without double counting: the outline export.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="What the reader noticed, e.g. minutes that do not add up.",
    )


class AnswerMeta(BaseModel):
    model: Optional[str] = None
    seconds: Optional[float] = None
    first_token_seconds: Optional[float] = None
    passages: int = Field(0, description="Passages of the user's documents the answer was given.")


class TurnResult(BaseModel):
    session_id: str
    turn_index: Optional[int] = None
    # A string, not an enum: a later version may add a kind (an agentic step, say),
    # and a client generated from an enum would fail to read it (2026-10-21).
    kind: str = Field(
        ...,
        description=(
            "Today one of lesson, answer, handoff; treat any other value as text to "
            "show (``markdown``). lesson: a new version of the lesson (``markdown``). answer: a reply about the "
            "lesson and its documents (``markdown``); the lesson did not change. handoff: "
            "the request needs a new lesson -- send ``handoff_query`` again with "
            "``mode: lesson`` (or ``mode: answer`` to have it answered here)."
        ),
    )
    markdown: Optional[str] = Field(
        None,
        description=(
            "The lesson or the answer, citations of the user's documents resolved to "
            "document name and place, with the sources listed at the end."
        ),
    )
    handoff_query: Optional[str] = None
    cited_documents: list[str] = Field(
        default_factory=list, description="Names of the user's documents the text cites."
    )
    structure: Optional[LessonStructure] = Field(None, description="For a lesson.")
    planner: Optional[PlannerInfo] = Field(None, description="For a lesson: the planner card.")
    retriever: Optional[RetrieverInfo] = Field(None, description="For a lesson: the search card.")
    critic: Optional[CriticInfo] = Field(
        None,
        description="For a lesson: the critic card. Null when the judge is on (the Critic is "
        "then not in the chain).",
    )
    judge: Optional[JudgeInfo] = Field(
        None,
        description="For a lesson, when the judge is on (AIX_JUDGE_ENABLED): the targeted "
        "review -- what it corrected (already in ``markdown``) and what is left to check.",
    )
    lesson_meta: Optional[AgentRunMeta] = Field(None, description="For a lesson: the run summary.")
    answer_meta: Optional[AnswerMeta] = Field(None, description="For an answer.")
    ai_disclosure: str = AI_DISCLOSURE


class TurnOut(BaseModel):
    turn_index: int
    role: str = Field(..., description="Today user or assistant; a later version may add others.")
    kind: str = Field(
        ...,
        description="Today message (the user's), lesson or answer; treat any other value "
        "as text to show (``markdown``).",
    )
    markdown: str
    created_at: Optional[datetime] = None
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="For a lesson: the run summary (scores, counts, UDL coverage). "
        "For an answer: model, seconds, passages.",
    )
    planner: Optional[PlannerInfo] = Field(
        None, description="For a lesson: its planner card, as when it was written."
    )
    retriever: Optional[RetrieverInfo] = Field(
        None,
        description="For a lesson: its Knowledge Graph card. The curated media are in "
        "``GET /sessions/{id}/media?turn_index=…`` → ``curated``.",
    )
    critic: Optional[CriticInfo] = Field(None, description="For a lesson, with the judge off.")
    judge: Optional[JudgeInfo] = Field(
        None, description="For a lesson, with the judge on: the «Revisione mirata» card."
    )


class SessionMaterial(BaseModel):
    material_id: str
    name: str
    status: str


class SessionSummary(BaseModel):
    session_id: str
    title: Optional[str] = None
    domain: str
    status: str = Field(..., description="draft (no turn yet) · running · complete · error")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SessionOut(SessionSummary):
    educational_profile: dict[str, Any] = Field(default_factory=dict)
    lesson_markdown: Optional[str] = Field(
        None, description="The current lesson, citations resolved."
    )
    lesson_structure: Optional[LessonStructure] = None
    cited_documents: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    materials: list[SessionMaterial] = Field(default_factory=list)
    turns: list[TurnOut] = Field(default_factory=list)
    ai_disclosure: str = AI_DISCLOSURE


class SessionMedia(BaseModel):
    """The resources shown with a lesson: the Knowledge Graph's curated ones, and
    those found live (papers, Wikipedia, videos) for the WebUI's right panel."""

    turn_index: Optional[int] = Field(
        None, description="The lesson turn these resources belong to; null when none."
    )
    curated: dict[str, Any] = Field(
        default_factory=dict,
        description="The Knowledge Graph's curated resources, the lesson result's "
        "``retriever.media``: videos, resources, citations, open textbooks.",
    )
    videos: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    wikipedia: list[dict[str, Any]] = Field(default_factory=list)
