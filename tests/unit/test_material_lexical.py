"""The keyword half of the search and the fusion of the two (§8.5).

The properties tested here are the ones the golden-case evaluation cannot see.
The evaluation says the hybrid ranking is better on twelve real requests; it
would say so just as happily with a fusion that reshuffles equal scores from run
to run, or with a tokenizer that matches "guerra" against "guerre" by accident.

So: that a passage about the subject beats one that merely name-drops it, that
the fusion is a function of the content and not of the order the rows arrived
in, and that either half going silent leaves the other half's ranking intact.
"""

from __future__ import annotations

import pytest

from aix.webui.materials.lexical import bm25_scores, fuse, tokenize

pytestmark = pytest.mark.unit


def test_the_words_that_carry_meaning_survive_tokenizing():
    tokens = tokenize("La crisi dei missili di Cuba del 1962")
    assert "crisi" in tokens
    assert "cuba" in tokens
    assert "1962" in tokens  # a year is evidence
    assert "dei" not in tokens and "del" not in tokens  # stopwords
    assert "la" not in tokens  # too short to discriminate


def test_the_inflections_of_one_word_meet():
    """ "missili" in the request, "missile" on the page — they must match."""
    assert tokenize("missili")[0] == tokenize("missile")[0]
    assert tokenize("conferenza")[0] == tokenize("conferenze")[0]
    assert tokenize("sovietici")[0] == tokenize("sovietico")[0]


def test_a_page_about_the_subject_beats_one_that_only_mentions_it():
    about = (
        "La conferenza di Yalta del 1945 divise l'Europa in sfere di influenza. "
        "A Yalta i tre alleati decisero il destino della Germania."
    )
    mentions = (
        "Bibliografia. Si veda anche il capitolo su Yalta in AA.VV., Storia "
        "contemporanea, Milano 1998; e la voce «Guerra fredda» dell'enciclopedia."
    )
    scores = bm25_scores("conferenza di Yalta sfere di influenza", [about, mentions])
    assert scores[0] > scores[1]


def test_a_query_with_nothing_to_match_scores_nothing():
    assert bm25_scores("fotosintesi clorofilliana", ["La guerra fredda", "Yalta"]) == [0.0, 0.0]
    assert bm25_scores("", ["qualunque testo"]) == [0.0]


def test_what_both_searches_like_beats_what_only_one_loves():
    """The point of fusing: agreement outranks a single strong opinion."""
    order = fuse([["both", "only-vector"], ["both", "only-words"]])
    assert order[0] == "both"


def test_a_passage_only_one_search_found_is_still_ranked():
    """Neither half may be silenced: a cross-language match has no keywords."""
    order = fuse([["vector-only"], []])
    assert order == ["vector-only"]


def test_equal_scores_are_settled_by_the_similarity_not_by_the_order():
    """Both keys are rank 1 of one ranking: the tie is real, the answer is not arbitrary.

    Without the tiebreak this order followed the order the rows came back in,
    which changed between two runs of the same evaluation (2026-09-18) and made
    every before/after measurement unreliable.
    """
    rankings = [["a"], ["b"]]
    assert fuse(rankings, tiebreak={"a": 0.9, "b": 0.4})[0] == "a"
    assert fuse(rankings, tiebreak={"a": 0.4, "b": 0.9})[0] == "b"
    # And reversing the order of the rankings changes nothing.
    assert fuse(rankings[::-1], tiebreak={"a": 0.9, "b": 0.4})[0] == "a"
