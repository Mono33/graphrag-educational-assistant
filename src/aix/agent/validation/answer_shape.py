"""The shape of the Writer's answer, corrected before anyone reads it.

Two things the model does that no prompt has managed to stop for good, both
seen in real lessons, both harmless to the content and ugly everywhere the
content goes:

* it **wraps the whole answer in a code fence**. Every intent prompt but one
  shows the expected format inside a ```` ```markdown ```` block, and the model
  sometimes copies the fence along with the format. The WebUI then shows a grey
  box of source code, the PDF prints ``#`` and ``**`` literally, and an API
  caller receives a string that is not the Markdown it was promised
  (2026-09-22, a list of topics for a future lesson);
* it **writes the source disclaimer twice**, because two instructions each
  claimed the last position in the answer (2026-09-22).

**Why here, and not where the text is rendered.** The first fix for the
disclaimer went into the WebUI's renderer. The PDF export reaches the text by a
path that skips it for lessons without private citations, and the public API
skips every renderer: it returns the draft as the graph wrote it. A correction
applied at one exit is a correction the other exits do not get. ``write_node``
is the one place every draft passes through, first version and revisions alike,
so the answer is tidied there. The renderers call it too, for lessons stored
before this existed.

Both rules are deliberately narrow. Each removes only what it can prove is
packaging, and leaves the text alone whenever it is unsure: a lesson that
legitimately contains a code block keeps it.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: A fence line: three or more backticks, then an optional language tag.
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,})[ \t]*([\w+-]*)[ \t]*$")

#: The languages a wrapped *answer* is announced with. A fence that says
#: ``python`` is a code example, not packaging, and is never touched.
_ANSWER_LANGUAGES = {"", "markdown", "md"}

#: Leading HTML comments -- the AI-generated marking the service adds -- and
#: blank lines, which may precede the fence in a stored lesson.
_PREAMBLE = re.compile(r"\A(?:\s*<!--.*?-->)*\s*", re.DOTALL)


def unwrap_whole_answer_fence(text: str) -> str:
    """The answer without a code fence around all of it.

    Unwrapped only when the fence is the packaging of the whole answer: the
    first thing after any leading comment, announced as Markdown or not at all,
    and either closed on the very last line with no other fence in between, or
    -- when the model ran out of tokens -- never closed at all but announced as
    Markdown. Anything else is left exactly as it is.
    """
    if not text or "```" not in text:
        return text

    preamble = _PREAMBLE.match(text).group(0)
    body = text[len(preamble) :]
    lines = body.split("\n")

    opening = _FENCE.match(lines[0])
    if not opening or opening.group(2).lower() not in _ANSWER_LANGUAGES:
        return text

    ticks = opening.group(1)
    fence_lines = [i for i, line in enumerate(lines) if _FENCE.match(line)]

    last = len(lines) - 1
    while last > 0 and not lines[last].strip():
        last -= 1

    closed = (
        fence_lines == [0, last]
        and _FENCE.match(lines[last]).group(1) == ticks
        and not _FENCE.match(lines[last]).group(2)
    )
    never_closed = fence_lines == [0] and opening.group(2).lower() in {"markdown", "md"}

    if closed:
        inner = lines[1:last]
    elif never_closed:
        inner = lines[1:]
    else:
        return text

    logger.info("[answer-shape] the whole answer was wrapped in a code fence -- unwrapped")
    return preamble + "\n".join(inner).strip("\n") + "\n"


#: The disclaimer the Writer must end on (EU AI Act Art. 50, §12).
#:
#: Matched on its opening words rather than in full: the model reproduces the
#: sentence but not always the whitespace, and in the lesson that exposed this
#: one of the two copies carried the pedagogy references appended to it.
_DISCLAIMER = re.compile(
    r"\n*(?:---\s*\n+)?⚠️?\s*\*\*Nota sulle fonti\*\*.*?(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def once_only_disclaimer(text: str) -> str:
    """The source disclaimer kept once, and never twice.

    Two instructions in the Writer's prompt each claimed the last position --
    "Output order: ... -> `*Fonti:*` footer" and "at the end of EVERY response,
    include this disclaimer" -- and the model satisfied both the only way it
    could: writing the disclaimer before the footer *and* again after it.

    The prompt now says which of the two comes last. This is the net under it,
    because a prompt is a request: the two-lists-of-sources defect came back
    after being asked away once already.

    The first copy is the one kept. It is the one the model wrote in context,
    so where it added the pedagogy references they are not lost.
    """
    found = list(_DISCLAIMER.finditer(text or ""))
    if len(found) < 2:
        return text

    # Right to left, so the offsets of the ones not yet removed stay valid.
    for match in reversed(found[1:]):
        text = text[: match.start()] + text[match.end() :]
    logger.info("[answer-shape] the disclaimer appeared %d times -- kept one", len(found))
    return text


def tidy_answer(text: str) -> str:
    """Both corrections, in the order that lets each see the text it expects.

    The fence goes first: inside a fence the disclaimer is code, and the second
    rule would be counting copies of a sentence the reader never sees rendered.
    """
    return once_only_disclaimer(unwrap_whole_answer_fence(text))
