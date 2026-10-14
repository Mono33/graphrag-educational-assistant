"""Removing a citation whose passage does not support the sentence (§8.6).

The judging itself is measured elsewhere, on real passages, by
``scripts/eval/check_citation_support.py``. What is pinned here is everything
around the judgement, which is where this kind of feature usually goes wrong:

* the sentence survives — only the attribution is removed;
* a check that fails leaves the lesson exactly as it was, in every way it can
  fail. A checker that strips citations because the network hiccuped is worse
  than no checker, and the teacher cannot tell the two apart;
* the same marker used twice is judged twice, because the claims differ.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aix.agent.validation.citation_support import drop_unsupported_citations

pytestmark = pytest.mark.unit

YALTA = "[[material:a#page=1]]"
STANDARDS = "[[material:b#page=6]]"

ITEMS = [
    {"citation": YALTA, "text": "A Yalta l'Europa fu divisa in sfere d'influenza."},
    {"citation": STANDARDS, "text": "Common Core Standard RH.9-10.2: Determine the central ideas."},
]

DRAFT = (
    f"La conferenza di Yalta divise l'Europa {YALTA}.\n"
    f"La NATO nacque per contenere l'espansione sovietica {STANDARDS}."
)


def _answer(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _judge(*verdicts: str) -> SimpleNamespace:
    """A client that answers one citation per call, in order.

    One call per citation, not one for the lesson: asked to rule on six
    citations of the same document at once, the judge marked the two sound ones
    unsupported along with the four invented ones. 11/11 on the golden cases one
    at a time, 10/11 batched (2026-09-22).
    """
    create = AsyncMock(side_effect=[_answer(v) for v in verdicts])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def test_the_unsupported_citation_goes_and_the_sentence_stays():
    out = await drop_unsupported_citations(DRAFT, ITEMS, client=_judge("supported", "unsupported"))

    assert YALTA in out, "a citation that holds is left alone"
    assert STANDARDS not in out, "one that does not is removed"
    assert "La NATO nacque per contenere l'espansione sovietica." in out, (
        "the claim is usually right — it is the attribution that was wrong"
    )


async def test_a_citation_removed_mid_sentence_leaves_no_gap():
    out = await drop_unsupported_citations(DRAFT, ITEMS, client=_judge("unsupported", "supported"))

    assert "divise l'Europa." in out, "no space before the full stop"


async def test_the_same_marker_is_judged_once_per_sentence():
    """Two claims, one passage: one may hold and the other may not.

    On two lines: the claim is the marker's line (see ``_claim_around``), so two
    markers on one line are the same question, asked once since 2026-10-13.
    """
    draft = f"Yalta divise l'Europa {YALTA}.\nLa fotosintesi avviene nei cloroplasti {YALTA}."

    out = await drop_unsupported_citations(
        draft, ITEMS[:1], client=_judge("supported", "unsupported")
    )

    assert out.count(YALTA) == 1
    assert "Yalta divise l'Europa" in out and "La fotosintesi avviene nei cloroplasti" in out


async def test_a_bulleted_list_is_one_claim_per_bullet():
    """A list has no blank lines between its items.

    Taking "the block between two blank lines" swallowed the whole list, and
    the judge was handed «page=3&block=e8d9…]] - Guerra del Vietnam
    [[material:5c07…» — several claims and their neighbours' markers run
    together. Four citations were removed on that (2026-09-21).
    """
    a, b = "[[material:a#page=1]]", "[[material:b#page=2]]"
    items = [
        {"citation": a, "text": "La guerra di Corea cominciò nel 1950."},
        {"citation": b, "text": "La crisi dei missili di Cuba del 1962."},
    ]
    draft = f"## Temi\n\n- Guerra di Corea {a}\n- Crisi di Cuba {b}\n"
    client = _judge("supported", "supported")

    await drop_unsupported_citations(draft, items, client=client)

    # One call per citation now, so the claims are in different prompts.
    prompts = [
        call.kwargs["messages"][0]["content"]
        for call in client.chat.completions.create.await_args_list
    ]
    assert len(prompts) == 2
    assert any("FRASE: - Guerra di Corea" in p for p in prompts)
    assert any("FRASE: - Crisi di Cuba" in p for p in prompts)
    assert all("[[material:" not in p for p in prompts), "an address is noise inside a claim"


async def test_a_title_on_its_own_line_travels_with_the_claim():
    """A numbered item splits in two, and the second line alone says nothing."""
    marker = "[[material:a#page=6]]"
    items = [{"citation": marker, "text": "Common Core Standard RH.9-10.2."}]
    draft = f"9. **Il ruolo della NATO nel contenimento**\n   Approfondimento {marker}.\n"
    client = _judge("unsupported")

    await drop_unsupported_citations(draft, items, client=client)

    prompt = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "**Il ruolo della NATO nel contenimento** Approfondimento" in prompt


async def test_passages_sharing_one_marker_are_judged_together():
    """A citation anchors on the page and the first block, and a PDF page is
    usually one block — so several chunks carry the same marker.

    Judged per chunk, one position was cut once per chunk: on a page with four,
    the second cut took the text *after* the marker with it. And a claim was
    called unsupported because the first chunk behind the marker happened not to
    carry it, while another one did.
    """
    marker = "[[material:a#page=2]]"
    items = [
        {"citation": marker, "text": "La divisione della Germania in quattro zone."},
        {"citation": marker, "text": "Il muro di Berlino fu costruito nel 1961."},
        {"citation": marker, "text": "La conferenza di Bandung del 1955."},
    ]
    draft = f"Il muro di Berlino è del 1961 {marker}. La lezione prosegue qui."
    client = _judge("supported")

    out = await drop_unsupported_citations(draft, items, client=client)

    assert out == draft, "one marker, one judgement, and the text is untouched"
    assert client.chat.completions.create.await_count == 1

    prompt = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "muro di Berlino fu costruito" in prompt, "the judge sees every passage behind it"
    assert "Bandung" in prompt


async def test_a_removal_never_eats_the_text_after_it():
    marker = "[[material:a#page=2]]"
    items = [{"citation": marker, "text": "x"}, {"citation": marker, "text": "y"}]
    draft = f"Una affermazione {marker}. Una frase che deve sopravvivere."

    out = await drop_unsupported_citations(draft, items, client=_judge("unsupported"))

    assert out == "Una affermazione. Una frase che deve sopravvivere."


@pytest.mark.parametrize(
    "content, why",
    [
        ("perhaps", "a verdict that is neither word"),
        ("", "an empty answer"),
        ("{}", "an answer in the wrong shape"),
    ],
)
async def test_an_unreadable_verdict_keeps_its_citation(content, why):
    """Fail open, one citation at a time.

    The batch could only keep everything or judge everything; per citation, an
    answer nobody can read costs that one citation its judgement and nothing
    else.
    """
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_answer(content)))
        )
    )

    assert await drop_unsupported_citations(DRAFT, ITEMS, client=client) == DRAFT, why


async def test_a_judge_that_raises_changes_nothing():
    """The API is down, or the key is wrong. The lesson is not the place to find out."""
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("502")))
        )
    )

    assert await drop_unsupported_citations(DRAFT, ITEMS, client=client) == DRAFT


async def test_nothing_to_check_costs_nothing():
    """No markers in the draft, no passages, no draft: no call at all."""
    client = _judge("unsupported")

    assert await drop_unsupported_citations("Una lezione senza citazioni.", ITEMS, client=client)
    assert await drop_unsupported_citations(DRAFT, [], client=client) == DRAFT
    assert await drop_unsupported_citations("", ITEMS, client=client) == ""
    client.chat.completions.create.assert_not_awaited()


async def test_the_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("AIX_CITATION_CHECK_ENABLED", "false")
    client = _judge("unsupported", "unsupported")

    assert await drop_unsupported_citations(DRAFT, ITEMS, client=client) == DRAFT
    client.chat.completions.create.assert_not_awaited()


async def test_the_judge_sees_the_whole_cited_page_not_just_the_marked_chunk():
    """The teacher reads "p. 44", so the evidence is all of page 44.

    From a real lesson (2026-09-25): the Writer attached the marker of one chunk
    of page 44 to a poll printed in another chunk of page 44. Judged against its
    own chunk, the citation was removed -- and the page it pointed to has the
    poll. Two markers that differ only in their block are the same place.
    """
    uid = "4c37abf4-d86d-4a61-ad1f-782856081317"
    regularisation = f"[[material:{uid}#page=44&block=a8bbc968c208e4a9]]"
    poll = f"[[material:{uid}#page=44&block=3f8c2efcd9433261]]"
    elsewhere = f"[[material:{uid}#page=40&block=c570087f4c4a53ba]]"
    items = [
        {
            "citation": regularisation,
            "text": "Percorsi di regolarizzazione per chi è arrivato da bambino.",
        },
        {"citation": poll, "text": "Sondaggio Economist e YouGov: consenso negativo, -13 punti."},
        {"citation": elsewhere, "text": "Trump sta mantenendo la promessa delle espulsioni."},
    ]
    draft = f"Il sondaggio Economist/YouGov dà -13 punti {regularisation}."
    client = _judge("supported")

    await drop_unsupported_citations(draft, items, client=client)

    prompt = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert "Economist e YouGov" in prompt, "the other chunk of the same page is evidence"
    assert "regolarizzazione" in prompt
    assert "promessa delle espulsioni" not in prompt, "page 40 is another place"
