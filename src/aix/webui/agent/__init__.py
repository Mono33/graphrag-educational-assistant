"""
WebUI ↔ Agent seam (CORE 2 #6.6 P2).

This subpackage is the *only* place ``aix.webui`` is allowed to reach into
the LangGraph agent stack. Routes import from here and never directly from
``aix.agent.*`` — that boundary keeps the webui swap-in/swap-out friendly
and means a future remote-agent variant (HTTP / message bus) only needs to
swap this file.

Public API:
    run_agent_stream(lesson, session)   async generator yielding StreamEvents
    StreamEvent                         normalized event dataclass
    PHASE_ORDER, PHASE_LABELS           canonical node order + Italian labels
"""

from aix.webui.agent.service import (
    PHASE_LABELS,
    PHASE_ORDER,
    StreamEvent,
    run_agent_stream,
)

__all__ = [
    "PHASE_LABELS",
    "PHASE_ORDER",
    "StreamEvent",
    "run_agent_stream",
]
