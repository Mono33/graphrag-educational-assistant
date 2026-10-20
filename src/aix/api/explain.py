"""The WebUI's cards, as the API's typed explainability (2026-10-14).

The WebUI builds one payload per phase (``_build_planner_payload``,
``_build_retriever_payload``, ``_build_critic_payload``, ``_build_judge_payload`` and the run summary in
``aix.webui.agent.service``) and renders it as a card. The API returns the very
same payloads, validated into published models -- nothing is recomputed, so the
two cannot disagree. A model ignores what it does not declare; the parity test
(``tests/unit/test_api_webui_parity.py``) is what keeps it declaring everything.
"""

from __future__ import annotations

from typing import Any, Optional

from aix.api.schemas.agent import AgentRunMeta, CriticInfo, JudgeInfo, PlannerInfo, RetrieverInfo


def planner_info(payload: Optional[dict[str, Any]]) -> PlannerInfo:
    return PlannerInfo.model_validate(payload or {})


def retriever_info(payload: Optional[dict[str, Any]]) -> RetrieverInfo:
    return RetrieverInfo.model_validate(payload or {})


def critic_info(payload: Optional[dict[str, Any]]) -> CriticInfo:
    return CriticInfo.model_validate(payload or {})


def judge_info(payload: Optional[dict[str, Any]]) -> JudgeInfo:
    return JudgeInfo.model_validate(payload or {})


def run_meta(meta: Optional[dict[str, Any]]) -> AgentRunMeta:
    """The run summary the ``done`` event carries; a missing number is a zero, not an error."""
    data = dict(meta or {})
    data.setdefault("duration_seconds", 0.0)
    data.setdefault("approved", False)
    data.setdefault("revision_count", 0)
    if not isinstance(data.get("scores"), dict):
        data["scores"] = None
    if not isinstance(data.get("media_counts"), dict):
        data.pop("media_counts", None)
    return AgentRunMeta.model_validate(data)


def lesson_structure(markdown: str, *, lesson_id: Optional[str], title: Optional[str]):
    """Sections and minutes, read the way the PDF and outline exports read the lesson."""
    from aix.api.schemas.sessions import LessonSection, LessonStructure
    from aix.export import adapt_lesson_markdown, build_export_ir

    ir = build_export_ir(
        adapt_lesson_markdown(markdown, lesson_id=lesson_id, fallback_title=title or "Lezione")
    )
    return LessonStructure(
        title=ir.title,
        duration_minutes=ir.duration_minutes,
        sections=[
            LessonSection(title=s.title, level=s.level, minutes=s.minutes) for s in ir.sections
        ],
        timed_outline=[
            LessonSection(title=name, level=2, minutes=minutes)
            for name, minutes in ir.timed_outline
        ],
        warnings=list(ir.warnings),
    )


def cited_documents(raw_markdown: str, labels: Optional[dict[str, str]]) -> list[str]:
    """Names of the documents the text's markers point at, in order, once each."""
    import re

    names: list[str] = []
    for version_id in re.findall(r"\[\[material:([0-9a-fA-F-]{36})", raw_markdown or ""):
        name = (labels or {}).get(version_id.lower())
        if name and name not in names:
            names.append(name)
    return names
