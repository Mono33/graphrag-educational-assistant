"""The teacher's switch for the Knowledge Graph search on a follow-up turn.

On 2026-09-29 a teacher asked about a chapter of her attached book and the
run searched the Knowledge Graph and the web again, because the planner judged
the chapter "content not yet retrieved" -- which it was, from the book, and the
book is searched on every turn anyway. The switch lets her say: this turn,
work from what the thread already has.
"""

from __future__ import annotations

import uuid

import pytest
from jinja2 import Environment, FileSystemLoader

from aix.agent.graph.nodes import should_retrieve
from aix.webui.agent.service import graph_search_override
from aix.webui.lessons.routes import remember_graph_search

pytestmark = pytest.mark.unit

TEMPLATES = "src/aix/webui/templates"


class _Lesson:
    def __init__(self, status="complete", profile=None):
        self.id = uuid.UUID("11111111-1111-1111-1111-111111111111")
        self.status = status
        self.uploaded_files_json = []
        self.educational_profile_json = profile


def _input(lesson, *, conditional=True, light=False):
    env = Environment(loader=FileSystemLoader(TEMPLATES))
    env.globals.update(
        upload_accept=lambda: ".pdf",
        upload_hint=lambda: "PDF",
        conditional_retrieval_enabled=lambda: conditional,
        light_turns_enabled=lambda: light,
    )
    return env.get_template("partials/chat_input.html").render(lesson=lesson, material_states={})


def test_a_follow_up_input_offers_the_switch_on_by_default():
    html = _input(_Lesson())

    assert "Cerca nel Knowledge Graph" in html
    assert 'name="graph_search" value="on" checked' in html
    assert 'name="graph_search_choice" value="1"' in html


def test_switched_off_it_stays_off_on_the_next_turn():
    html = _input(_Lesson(profile={"__graph_search__": False}))

    assert 'name="graph_search" value="on">' in html, "rendered unticked"


def test_the_first_request_has_no_switch():
    """A first turn has nothing to reuse: it always searches."""
    assert "Cerca nel Knowledge Graph" not in _input(_Lesson(status="draft"))


def test_without_conditional_retrieval_there_is_no_switch():
    assert "Cerca nel Knowledge Graph" not in _input(_Lesson(), conditional=False)


@pytest.mark.parametrize(
    "form, expected",
    [
        ({"graph_search_choice": "1"}, False),  # an unticked box sends nothing
        ({"graph_search_choice": "1", "graph_search": "on"}, True),
    ],
)
def test_the_form_s_choice_is_kept_on_the_lesson(form, expected):
    lesson = _Lesson(profile={"pedagogical_intent": "x"})

    remember_graph_search(lesson, form)

    assert lesson.educational_profile_json["__graph_search__"] is expected
    assert lesson.educational_profile_json["pedagogical_intent"] == "x"


def test_a_form_without_the_switch_changes_nothing():
    lesson = _Lesson(profile={"__graph_search__": False})

    remember_graph_search(lesson, {"query": "altro"})

    assert lesson.educational_profile_json == {"__graph_search__": False}


@pytest.mark.parametrize("switch, expected", [(False, False), (True, None), (None, None)])
def test_off_waives_the_search_and_on_lets_the_planner_decide(switch, expected):
    assert graph_search_override(switch) is expected


def test_switched_off_the_router_reuses_even_when_the_planner_wants_to_search(monkeypatch):
    monkeypatch.setenv("AIX_CONDITIONAL_RETRIEVAL_ENABLED", "true")
    state = {
        "conversation_history": [{"role": "user", "content": "crea una lezione"}],
        "retrieved_nodes": [{"name": "Scaffolding"}],
        "requires_new_retrieval": True,
        "force_retrieval": graph_search_override(False),
    }

    assert should_retrieve(state) == "reuse"
