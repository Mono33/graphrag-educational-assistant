"""
WebUI — internal HTML+SSE surface for end-to-end agent testing (Path C).

This package is the "Mirror Stack" frontend recommended by ADR-0001
(see docs/architecture/Frontend_Platform_Evaluation.md). It is mounted on
the same FastAPI process as the public API (src/aix/api) and is the
canonical end-to-end test surface for the LangGraph agent (replaces the
Streamlit app for that role from CORE 2 onward).

Stack:
    - FastAPI + Jinja2 (server-side templates)
    - htmx 2 (hypermedia for partial swaps and SSE streaming)
    - WebAwesome 3.x (web-component UI library; same family as AixLearning native)
    - Tailwind CSS (styling; loaded via CDN in P0, replaced by CLI build later)
    - sse-starlette (Server-Sent Events for token / tool / critic streaming)
    - Alpine.js (small client-side islands — theme toggle, mobile nav)

Mount point:
    src/aix/api/main.py:
        from aix.webui import router as webui_router
        app.include_router(webui_router)

Routes are prefixed with /webui so they cannot collide with the existing
public JSON API (/api/v1/*) or future /api/v1/agent/* endpoints.
"""

from aix.webui.routes import router

__all__ = ["router"]
__version__ = "0.1.0"  # P0 skeleton
