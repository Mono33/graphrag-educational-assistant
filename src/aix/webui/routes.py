"""
WebUI routes — server-rendered HTML + SSE for the Agentic GraphRAG UI.

Mounted routers (this file collects them under a single ``router`` so
``aix.api.main`` only needs one ``include_router``):

    P0 (skeleton):
        GET  /webui/             — landing page; htmx + WebAwesome wiring check
        GET  /webui/health       — htmx-target HTML fragment

    P1 (auth + lesson form):
        GET  /auth/register      — register page             (auth subpackage)
        POST /auth/register      — create user + auto-login  (auth subpackage)
        GET  /auth/login         — login page                (auth subpackage)
        POST /auth/login         — set cookie + redirect     (auth subpackage)
        GET  /auth/logout        — clear cookie              (auth subpackage)
        GET  /webui/lesson/new   — EducationalProfile form   (lessons subpackage)
        POST /webui/lesson       — persist Lesson row        (lessons subpackage)
        GET  /webui/lesson/{id}  — lesson placeholder page   (lessons subpackage)

    P2+ (not yet implemented):
        GET  /webui/lesson/{id}/stream  — SSE stream of agent events (P2)
        POST /webui/lesson/{id}/decide  — tool-approval modal callback (P3)

All HTML is rendered via Jinja2 templates under src/aix/webui/templates/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from aix.webui.auth import router as auth_router
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.lessons import router as lessons_router

logger = logging.getLogger(__name__)

# Resolve template + static directories relative to this package, so the webui
# works regardless of where uvicorn is launched from.
_PACKAGE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PACKAGE_DIR / "templates"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Top-level webui router. Sub-routers carry their own prefixes (/auth, /webui)
# so we can mount everything via a single ``app.include_router(router)`` call
# in aix.api.main.
router = APIRouter(tags=["webui"])

# Public landing routes — anonymous access OK, but the page adapts to user state.
_pages_router = APIRouter(prefix="/webui")


@_pages_router.get("/", response_class=HTMLResponse, name="webui_home")
async def home(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
) -> HTMLResponse:
    """Landing page — Path C webui skeleton, now auth-aware (P1)."""
    return templates.TemplateResponse(
        "pages/home.html",
        {
            "request": request,
            "title": "AixLearning · Agentic GraphRAG",
            "phase": "P1 — Auth + Form",
            "user": user,
        },
    )


@_pages_router.get("/health", response_class=HTMLResponse, name="webui_health")
async def health_fragment() -> HTMLResponse:
    """
    htmx target — returns a small HTML fragment that the landing page swaps in
    when the user clicks the "Health check" button. Proves htmx + WebAwesome
    + FastAPI are correctly wired together end-to-end.
    """
    return HTMLResponse(
        '<wa-callout variant="success" appearance="filled outlined">'
        '  <wa-icon slot="icon" name="circle-check"></wa-icon>'
        "  <strong>WebUI healthy.</strong> "
        "  htmx + WebAwesome + Tailwind + FastAPI sono cablati correttamente."
        "</wa-callout>"
    )


# Compose: landing pages + auth + lessons under a single exported router.
router.include_router(_pages_router)
router.include_router(auth_router)
router.include_router(lessons_router)
