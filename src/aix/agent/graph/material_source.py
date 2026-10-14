"""Per-session access to the teacher's own material (§8.1).

The graph has two retrievers. The Knowledge Graph one lives in this package and
talks to Neo4j; the private one searches rows the *WebUI* owns, under the
authorization of the logged-in teacher, in the WebUI's database session.

``aix.agent`` must not import ``aix.webui`` — the dependency runs the other way,
and the agent is also driven by the API, the MCP server and the CLI, none of
which have a teacher's session. So the WebUI registers a search function here
before the run and removes it after, and the node asks for one by session id. No
session, no function, no private retrieval: the API and the CLI keep the exact
topology they had.

The same shape as ``write_stream``, and for the same reason: something the graph
needs, owned by a layer the graph cannot name.

The registered callable takes the query and returns the plain dicts of
``MaterialContextItem.to_dict()`` — plain because they go into the graph state,
and the state is checkpointed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Optional

#: query -> list of MaterialContextItem dicts
MaterialSearch = Callable[[str], Awaitable[list[dict[str, Any]]]]

_sources: dict[str, MaterialSearch] = {}


def register(session_id: str, search: MaterialSearch) -> None:
    _sources[session_id] = search


def get(session_id: Optional[str]) -> Optional[MaterialSearch]:
    if not session_id:
        return None
    return _sources.get(session_id)


def deregister(session_id: Optional[str]) -> None:
    if session_id:
        _sources.pop(session_id, None)
