"""What the WebUI shows, the API returns (2026-10-14).

The API's explainability models validate the WebUI's own payloads and ignore
what they do not declare. That makes a new field on a WebUI card invisible to
the API until someone adds it -- and nobody would notice. This test notices: it
builds each payload the way the WebUI does and requires every key to be either
a field of the API model or listed below as WebUI-only, with the reason.
"""

from __future__ import annotations

import pytest

from aix.api.schemas.agent import AgentRunMeta, CriticInfo, JudgeInfo, PlannerInfo, RetrieverInfo
from aix.api.schemas.sessions import AnswerMeta
from aix.webui.agent.service import (
    _build_critic_payload,
    _build_judge_payload,
    _build_planner_payload,
    _build_retriever_payload,
    _extract_meta,
)

pytestmark = pytest.mark.unit

#: Keys a WebUI payload carries that the API leaves out on purpose.
WEBUI_ONLY = {
    "retrieval_grade_emoji": "a decoration of the card; the grade itself is returned",
    "write_revision": "the DOM id counter of the card that follows the critic",
    "kind": "the light answer's own marker, the API says it in TurnResult.kind",
}

STATE = {
    "query_intent": "lesson_creation",
    "scope_status": "partial_scope",
    "scope_confidence": 0.8,
    "key_concepts": ["memoria di lavoro"],
    "subject_concepts": ["Guerra fredda"],
    "search_queries": ["memoria di lavoro"],
    "lesson_type": "full_lesson",
    "target_grade": "secondaria",
    "retrieved_nodes": [{"name": "Memoria di lavoro"}],
    "retrieved_relationships": [{"source": "a", "target": "b"}],
    "recommendations": [{"name": "Chunking"}],
    "curated_media": {"videos": [{"title": "v"}]},
    "external_resources": {"wikipedia": [{"title": "Guerra fredda"}]},
    "retrieval_confidence": "medium",
    "retrieval_reused": True,
    "plan": {"retrieval_reason": "stesso argomento"},
    "domain": "udl",
    "approved": True,
    "revision_count": 1,
    "max_revisions": 1,
    "critique": "Buona lezione.",
    "revision_instructions": "",
    "final_metadata": {"scores": {"average_score": 4.2}},
    "lesson_plan_draft": "# Lezione",
    "judge_summary": {
        "found": 3,
        "blocking": 1,
        "replaced": 1,
        "revised": 1,
        "reverted": 0,
        "seconds": 6.2,
        "fixed": [
            {
                "check": "esempi_lingua",
                "why": "gatto si divide gat-to",
                "before": "ga-to",
                "after": "gat-to",
                "kind": "sostituzione",
                "teacher_material": False,
            }
        ],
    },
    "review_notes": [
        {"check": "fatti", "severity": "bloccante", "quote": "q", "why": "w", "fix": "f"}
    ],
}

ANSWER_META = {
    "kind": "answer",
    "model": "m",
    "seconds": 4.8,
    "first_token_seconds": 1.1,
    "passages": 3,
}


def _missing(payload: dict, model) -> list[str]:
    return sorted(set(payload) - set(model.model_fields) - set(WEBUI_ONLY))


@pytest.mark.parametrize(
    ("name", "payload", "model"),
    [
        ("planner", lambda: _build_planner_payload(STATE), PlannerInfo),
        ("retriever", lambda: _build_retriever_payload(STATE), RetrieverInfo),
        ("critic", lambda: {**_build_critic_payload(STATE), "write_revision": 1}, CriticInfo),
        ("judge", lambda: _build_judge_payload(STATE, 1), JudgeInfo),
        ("run summary", lambda: {**_extract_meta(STATE), "duration_seconds": 1.0}, AgentRunMeta),
        ("answer", lambda: ANSWER_META, AnswerMeta),
    ],
)
def test_every_field_of_a_webui_card_is_in_the_api(name, payload, model):
    missing = _missing(payload(), model)

    assert not missing, (
        f"the WebUI's {name} payload has {missing}, which the API's {model.__name__} does not "
        "return: add the field, or list it in WEBUI_ONLY with the reason"
    )


def test_the_api_returns_the_values_not_only_the_names():
    retriever = RetrieverInfo.model_validate(_build_retriever_payload(STATE))
    planner = PlannerInfo.model_validate(_build_planner_payload(STATE))

    assert retriever.reused is True
    assert retriever.reuse_reason == "stesso argomento"
    assert planner.subject_concepts == ["Guerra fredda"]


def test_the_judge_s_corrections_reach_the_api_typed():
    """With the judge on, the Critic's card is empty: the judge's is the one to read."""
    judge = JudgeInfo.model_validate(_build_judge_payload(STATE, 1))

    assert (judge.found, judge.replaced, judge.revised) == (3, 1, 1)
    assert judge.fixed[0].after == "gat-to" and judge.fixed[0].teacher_material is False
    assert [p.quote for p in judge.points] == ["q"] and judge.notes == 1
