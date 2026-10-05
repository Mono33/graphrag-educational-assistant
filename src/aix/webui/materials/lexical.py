"""The lexical half of the hybrid search, and the fusion of the two (§8.5).

An embedding is good at "la crisi dei missili" ≈ "the Cuban missile crisis" and
bad at telling a page *about* the Cold War from a page that merely *mentions* it.
The first real evaluation showed exactly that: a bibliography and a page of
Common Core standards, both wholly generic, took places in the top five of two
thirds of the cases because they are full of the vocabulary of the subject.

A keyword ranking has the opposite bias — it knows nothing about meaning but a
great deal about which words this page actually spends itself on. Fusing the two
rankings is the standard answer, and §8.5 asks for it before anything heavier.

**Why rank fusion and not a weighted sum of scores.** A cosine lives in [0, 1]
and a BM25 score has no upper bound; combining them means choosing a scale
factor, and that factor turns out to depend on the corpus. Reciprocal Rank
Fusion (Cormack, Clarke and Buettcher, 2009) sums ``1 / (k + rank)`` instead:
only the *positions* are combined, so nothing needs to be made comparable and
a passage both rankings like beats one that only one of them loves.

**Why the crude stemmer.** Italian inflects: the query says "missili sovietici",
the page says "missile sovietico". Without stemming those do not match and the
lexical half contributes nothing to precisely the cases it should help. A real
stemmer (snowball) is a dependency and a model per language; truncating a token
to its first ``STEM_PREFIX`` characters conflates missil-e/-i, sovietic-o/-i and
confer-enza/-enze, which is most of what Italian needs here. It over-conflates
sometimes ("costituzione"/"costituito" both become "costit"), and that is the
trade accepted: on PostgreSQL the engine's own ``italian`` configuration does
this properly, and this path is what SQLite and the tests use.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypeVar

#: Reciprocal Rank Fusion's constant. 60 is the value from the paper: large
#: enough that the top few ranks are not overwhelmingly dominant, small enough
#: that rank 1 still counts for much more than rank 20.
RRF_K = 60

#: BM25's usual parameters. k1 governs how quickly repeating a word stops
#: helping, b how much a long passage is penalised for being long.
BM25_K1 = 1.5
BM25_B = 0.75

#: See the module docstring. Tokens shorter than this are left alone.
STEM_PREFIX = 6

#: Words that say nothing about what a passage is about. Italian first, then the
#: English of the imported handouts; both languages are in one corpus here.
#: BM25's IDF already discounts a word present everywhere, but a small corpus
#: makes that estimate noisy, and these are never evidence in any corpus.
STOPWORDS = frozenset(
    """
    ad agli ai al alla alle allo anche c che chi ci come con cui da dagli dai dal
    dalla dalle dallo degli dei del della delle dello di due e ed gli ha hanno ho
    i il in io la le lo loro ma me mi ne negli nei nel nella nelle no non nostro
    o od per piu più qual quale quando quanto quella quelle quello questa queste
    questi questo qui se sei senza si sia siamo sono su sui sul sulla sulle suo
    sua tra tre tu tuo un una uno vi

    a about after all also an and any are as at be been but by can could did do
    does for from had has have he her his how i if in into is it its more most
    no not of on one or other our out over said she should so some such than
    that the their them then there these they this those through to too under
    up very was we were what when where which who will with would you your
    """.split()
)

_WORD = re.compile(r"[^\W\d_]+|\d{4}", re.UNICODE)

T = TypeVar("T")


def tokenize(text: str) -> list[str]:
    """Words worth matching on: lowercased, stemmed crudely, stopwords dropped.

    Four-digit runs survive as tokens — a year is one of the most discriminating
    things a history page contains — while other numbers (page numbers, list
    markers) are dropped.
    """
    tokens = []
    for match in _WORD.findall(text.lower()):
        if len(match) < 3 or match in STOPWORDS:
            continue
        tokens.append(match[:STEM_PREFIX] if len(match) > STEM_PREFIX else match)
    return tokens


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Okapi BM25 of every document against ``query``, in the given order.

    The corpus is the set of passages being searched — a handful of documents
    belonging to one teacher — so the statistics are computed here and now
    rather than maintained. At this size that is both exact and instant.
    """
    query_terms = set(tokenize(query))
    if not query_terms or not documents:
        return [0.0] * len(documents)

    tokenized = [Counter(tokenize(document)) for document in documents]
    lengths = [sum(counts.values()) for counts in tokenized]
    average = (sum(lengths) / len(lengths)) or 1.0
    total = len(documents)

    idf = {}
    for term in query_terms:
        appearances = sum(1 for counts in tokenized if term in counts)
        # The "plus one" form: always positive, even for a term in every
        # passage, so a common word contributes little instead of subtracting.
        idf[term] = math.log(1 + (total - appearances + 0.5) / (appearances + 0.5))

    scores = []
    for counts, length in zip(tokenized, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + BM25_K1 * (1 - BM25_B + BM25_B * length / average)
            score += idf[term] * (frequency * (BM25_K1 + 1)) / denominator
        scores.append(score)
    return scores


def fuse(
    rankings: Iterable[Sequence[T]],
    *,
    k: int = RRF_K,
    tiebreak: Optional[Mapping[T, float]] = None,
) -> list[T]:
    """Reciprocal Rank Fusion of several rankings, best first.

    Each ranking is a sequence of keys, best first; a key missing from one
    ranking simply earns nothing from it, which is what makes the fusion robust
    when one half of the search has nothing to say (a query in a language the
    documents are not written in, say).

    **Ties are the common case, not the corner case.** A passage the vector
    search ranked third and the keyword search did not see at all scores exactly
    what a passage the keyword search ranked third and the vector search missed
    scores. Left to the dictionary, which of them is quoted depends on the order
    the rows came back in — and that order changed between two runs of the
    evaluation on the same data (2026-09-18), which makes every before/after
    comparison meaningless. ``tiebreak`` (the similarity, in practice) settles
    those from the content instead.
    """
    fused: dict[T, float] = {}
    best_rank: dict[T, int] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            best_rank[key] = min(best_rank.get(key, rank), rank)

    scores = tiebreak or {}
    return sorted(
        fused,
        key=lambda key: (-fused[key], -scores.get(key, 0.0), best_rank[key], str(key)),
    )
