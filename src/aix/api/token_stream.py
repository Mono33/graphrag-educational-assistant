"""The text of a turn as it is written, for the session stream (2026-10-16).

A chatbot platform shows the answer while the model writes it: AIxLearning sends
its own model's deltas to the browser every few tokens. Our phases alone left
its users looking at an empty card for the minute a lesson takes.

The Writer and the fast answer already put their tokens on a per-lesson queue
(``aix.agent.graph.write_stream``), which the WebUI drains through a second SSE
connection. The API merges that queue into the turn's own stream instead, so a
client opens one connection and gets, in order, the phases, the text and the
result.

What a client must know, and the guide says:

* the text is a **preview**. The final ``result.markdown`` is the text after the
  checks (tidied, unsupported citations removed, citations resolved to document
  names) and replaces it;
* a ``draft`` number counts the Writer's drafts in the turn: when it grows, the
  Critic asked for a revision and the text starts over;
* the internal ``[[material:…]]`` markers never reach the preview: they are
  held back while incomplete and dropped when complete.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Union

from aix.agent.graph import write_stream

_MARKER = re.compile(r"\[\[material:[^\]]*\]\]")

#: How often the merge looks for the queue, which the run registers only once it starts.
_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class TextDelta:
    """A piece of the text being written (``text`` is never empty)."""

    text: str
    draft: int


@dataclass(frozen=True)
class DraftEnd:
    """The Writer finished a draft; a revision, if any, starts a new one."""

    draft: int


class PreviewText:
    """Deltas in, deltas out, without the citation markers.

    A marker can arrive split over several deltas (``[[mate`` + ``rial:ab…]]``),
    so an unfinished one is held back until it closes or the draft ends.
    """

    def __init__(self) -> None:
        self._held = ""

    def feed(self, delta: str) -> str:
        text = _MARKER.sub("", self._held + delta)
        cut = text.rfind("[[")
        if cut != -1 and "]]" not in text[cut:]:
            text, self._held = text[:cut], text[cut:]
        elif text.endswith("["):
            text, self._held = text[:-1], "["
        else:
            self._held = ""
        return text

    def flush(self) -> str:
        text, self._held = _MARKER.sub("", self._held), ""
        return "" if text.startswith("[[material:") else text


class _Failed:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_END = object()


async def with_text(
    items: AsyncIterator[Any], lesson_id: str
) -> AsyncIterator[Union[Any, TextDelta, DraftEnd]]:
    """``items`` as they come, with the turn's text interleaved.

    The text that is ready is always yielded before the next item, so a
    ``writer`` or ``done`` phase never overtakes the draft it reports on.
    An exception raised by ``items`` is raised here.
    """
    out: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for item in items:
                await out.put(item)
        except BaseException as exc:  # noqa: BLE001 — re-raised by the consumer
            await out.put(_Failed(exc))
        finally:
            await out.put(_END)

    task = asyncio.create_task(pump())
    bus = None
    draft = 1
    preview = PreviewText()

    def drain() -> list[Union[TextDelta, DraftEnd]]:
        nonlocal draft, preview
        ready: list[Union[TextDelta, DraftEnd]] = []
        text = ""
        while bus is not None:
            try:
                token = bus.get_nowait()
            except asyncio.QueueEmpty:
                break
            if token is None:  # the draft is over
                text += preview.flush()
                if text:
                    ready.append(TextDelta(text, draft))
                    text = ""
                ready.append(DraftEnd(draft))
                draft += 1
                preview = PreviewText()
                continue
            kind, data = token if isinstance(token, tuple) else ("content", token)
            if kind == "content" and data:
                text += preview.feed(str(data))
        if text:
            ready.append(TextDelta(text, draft))
        return ready

    try:
        while True:
            # The run registers its queue once it starts, and the fast answer
            # registers a fresh one: look again whenever there is nothing to do.
            bus = write_stream.get_bus(lesson_id) or bus
            try:
                item = await asyncio.wait_for(out.get(), timeout=_POLL_SECONDS)
            except asyncio.TimeoutError:
                for ready in drain():
                    yield ready
                continue
            bus = write_stream.get_bus(lesson_id) or bus
            for ready in drain():
                yield ready
            if item is _END:
                return
            if isinstance(item, _Failed):
                raise item.exc
            yield item
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 — the turn's own cleanup ran
                pass
