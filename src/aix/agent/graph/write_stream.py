"""
Per-session writer token bus.

When the /webui SSE stream is active, write_node registers an asyncio.Queue
here before calling writer.write(). The writer streams tokens into the queue
as it generates them (stream=True on the OpenAI call). A concurrent task in
service.py drains the queue and forwards each token as a ``writer_chunk``
SSE event, giving the user a typewriter effect while waiting.

Lifecycle:
    register(session_id)   — called by write_node before writer.write()
    put(session_id, token) — called by WriterAgent with each token chunk
    sentinel(session_id)   — called by write_node after writer.write() returns
    deregister(session_id) — called by service.py cleanup

None token == sentinel (stream done).
"""

import asyncio
from typing import Optional

_buses: dict[str, asyncio.Queue] = {}


def register(session_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _buses[session_id] = q
    return q


def get_bus(session_id: Optional[str]) -> Optional[asyncio.Queue]:
    if not session_id:
        return None
    return _buses.get(session_id)


def deregister(session_id: str) -> None:
    _buses.pop(session_id, None)
