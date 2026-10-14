"""Tools a light turn may call: the documents and the graph only when the request needs them.

A light turn used to search the attached documents before every answer and send
what it found: on the rehearsals' 25 light turns those passages were 70% of the
input (~24k characters), for questions like «perché hai scelto l'attività 1?»
that never needed them. With these tools the model answers from the lesson, and
looks things up only when the request is about the documents or needs strategies
the lesson does not have.

Off unless ``AIX_LIGHT_TOOLS_ENABLED`` is on; off, the light turn is unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

#: At most this many rounds of look-ups before the answer.
MAX_ROUNDS = 2

TOOLS_RULE = """

Hai due strumenti, da usare solo se servono davvero:
- `cerca_nei_documenti`: cerca nei documenti che la docente ha allegato alla \
lezione. Usalo se la richiesta riguarda che cosa dicono i documenti o se devi \
citarli; i passaggi trovati hanno i riferimenti [[material:…]] da citare.
- `cerca_nel_grafo`: cerca strategie didattiche nel Knowledge Graph. Usalo solo \
se servono strategie che la lezione non contiene già.
Se la richiesta si risponde con la lezione e la conversazione, rispondi subito \
senza strumenti. Non annunciare le ricerche: scrivi solo la risposta."""

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cerca_nei_documenti",
            "description": "Cerca nei documenti allegati alla lezione i passaggi pertinenti.",
            "parameters": {
                "type": "object",
                "properties": {"richiesta": {"type": "string", "description": "Che cosa cercare."}},
                "required": ["richiesta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cerca_nel_grafo",
            "description": "Cerca strategie e concetti didattici nel Knowledge Graph.",
            "parameters": {
                "type": "object",
                "properties": {"richiesta": {"type": "string", "description": "Che cosa cercare."}},
                "required": ["richiesta"],
            },
        },
    },
]


def enabled() -> bool:
    raw = os.getenv("AIX_LIGHT_TOOLS_ENABLED")
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Toolbox:
    """Runs the look-ups for one turn and keeps the passages found, for the citation check."""

    def __init__(self, lesson: Any) -> None:
        self.lesson = lesson
        self.items: list[dict[str, Any]] = []
        self.calls: list[str] = []

    async def run(self, name: str, arguments: str) -> str:
        try:
            query = str(json.loads(arguments or "{}").get("richiesta") or "").strip()
        except json.JSONDecodeError:
            query = ""
        self.calls.append(name)
        if not query:
            return "Richiesta vuota: niente da cercare."
        try:
            if name == "cerca_nei_documenti":
                return await self._documents(query)
            if name == "cerca_nel_grafo":
                return await self._graph(query)
        except Exception as exc:  # noqa: BLE001 — a failed look-up is an answer
            logger.warning("[light-tools] %s failed (%s: %s)", name, type(exc).__name__, exc)
            return "La ricerca non è riuscita: rispondi con quello che hai."
        return f"Strumento sconosciuto: {name}."

    async def _documents(self, query: str) -> str:
        from aix.webui.agent.light import _documents

        text, items = await _documents(self.lesson, query)
        self.items.extend(items)
        return text or "Nessun passaggio pertinente nei documenti allegati."

    async def _graph(self, query: str) -> str:
        from aix.agent.graph.nodes import get_retriever

        result = await get_retriever(self.lesson.domain or "neuro")._get_tool().search(query)
        lines = []
        for node in (result.nodes or [])[:10]:
            name = node.get("name") or node.get("title") or ""
            about = str(node.get("description") or "")[:200]
            if name:
                lines.append(f"- **{name}**: {about}" if about else f"- **{name}**")
        return "\n".join(lines) or "Nessun risultato nel Knowledge Graph."


async def stream_with_tools(
    client: Any, messages: list[dict[str, Any]], kwargs: dict[str, Any], toolbox: Toolbox
) -> AsyncIterator[str]:
    """The answer's text as it streams; look-ups the model asks for run in between.

    Each round streams: text goes straight out, tool calls are collected and run,
    and their results go back to the model. The last round offers no tools, so
    the answer always comes.
    """
    conversation = list(messages)
    for round_no in range(MAX_ROUNDS + 1):
        offer = round_no < MAX_ROUNDS
        stream = await client.chat.completions.create(
            messages=conversation,
            stream=True,
            **kwargs,
            **({"tools": SCHEMAS, "tool_choice": "auto"} if offer else {}),
        )
        said: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            delta = choice.delta if choice else None
            if delta is None:
                continue
            if delta.content:
                said.append(delta.content)
                yield delta.content
            for call in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                slot["id"] = call.id or slot["id"]
                if call.function is not None:
                    slot["name"] += call.function.name or ""
                    slot["arguments"] += call.function.arguments or ""
        if not calls:
            return
        conversation.append(
            {
                "role": "assistant",
                "content": "".join(said) or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    }
                    for c in calls.values()
                ],
            }
        )
        for c in calls.values():
            result = await toolbox.run(c["name"], c["arguments"])
            conversation.append({"role": "tool", "tool_call_id": c["id"], "content": result})
