"""A cut in the conversation history never reads as a broken answer (2026-10-16).

The neuro replay of the water-cycle lesson: turn 2 answered in 3,500 characters,
the history kept 2,500 and a «…», and turn 3 told the teacher «la mia risposta
precedente si è interrotta» -- about an answer she had read whole.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aix.webui.agent.light import _ANSWER_CHARS, _CLIPPED_NOTE, _turn_line

pytestmark = pytest.mark.unit

LONG = "parola " * 800  # ~5,600 characters


def _answer(text):
    return SimpleNamespace(content_md=text, role="assistant", agent_kind="tutor")


def test_the_latest_answer_is_kept_whole():
    line = _turn_line(_answer(LONG), "# Lezione", whole=True)

    assert line == f"**Tu:** {LONG.strip()}"


def test_an_older_long_answer_says_the_cut_is_ours():
    line = _turn_line(_answer(LONG), "# Lezione")

    assert line.endswith(_CLIPPED_NOTE)
    assert len(line) < _ANSWER_CHARS + len(_CLIPPED_NOTE) + 20
    assert not line.endswith("…")


def test_a_short_answer_is_untouched():
    assert _turn_line(_answer("Sì, certo."), "# Lezione") == "**Tu:** Sì, certo."
