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
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aix.webui.auth import router as auth_router
from aix.webui.auth.dependencies import optional_current_user
from aix.webui.auth.models import User
from aix.webui.db import get_async_session
from aix.webui.lessons import router as lessons_router
from aix.webui.lessons.display import (
    activity_event_for_lesson,
    lesson_to_row,
    today_label_it,
)
from aix.webui.lessons.models import Lesson, SavedProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dashboard helpers (CORE 2 #6.6 P5 — warm-academic brand pass)
#
# The lesson-row / status-display / time-formatting helpers live in
# ``aix.webui.lessons.display`` so they can be reused by the Library and
# Workspace pages without circular imports. This file keeps only the
# *dashboard-shaped* aggregation logic.
#
# All queries below are read-only derivations from the existing ``lesson``
# table — no new endpoints, no schema changes, no agent / SSE touches.
# ---------------------------------------------------------------------------


async def _build_dashboard_context(session: AsyncSession, user: Optional[User]) -> dict[str, Any]:
    """
    Assemble the dashboard's per-section context.

    Anonymous visitors get an empty-ish context (template renders the
    sign-in CTA instead of the personalised dashboard). Authenticated
    users get four real, read-only derivations:

        - lesson_count   : total lessons owned
        - resume_lesson  : most-recently-touched draft (if any)
        - recent_lessons : last 3 lessons by updated_at (any status)
        - activity       : last 5 activity events derived from updated_at
    """
    empty: dict[str, Any] = {
        "lesson_count": 0,
        "resume_lesson": None,
        "recent_lessons": [],
        "activity": [],
        "today_label": today_label_it(),
    }
    if user is None:
        return empty

    total = (
        await session.scalar(select(func.count(Lesson.id)).where(Lesson.owner_id == user.id)) or 0
    )

    resume_q = await session.execute(
        select(Lesson)
        .where(Lesson.owner_id == user.id, Lesson.status == "draft")
        .order_by(Lesson.updated_at.desc())
        .limit(1)
    )
    resume = resume_q.scalar_one_or_none()

    recent_q = await session.execute(
        select(Lesson).where(Lesson.owner_id == user.id).order_by(Lesson.updated_at.desc()).limit(3)
    )
    recent = list(recent_q.scalars().all())

    activity_q = await session.execute(
        select(Lesson).where(Lesson.owner_id == user.id).order_by(Lesson.updated_at.desc()).limit(5)
    )
    activity_rows = list(activity_q.scalars().all())

    profiles_q = await session.execute(
        select(SavedProfile)
        .where(SavedProfile.owner_id == user.id)
        .order_by(SavedProfile.created_at.desc())
    )
    saved_profiles = list(profiles_q.scalars().all())

    return {
        "lesson_count": int(total),
        "resume_lesson": lesson_to_row(resume) if resume is not None else None,
        "recent_lessons": [lesson_to_row(lesson) for lesson in recent],
        "activity": [activity_event_for_lesson(lesson) for lesson in activity_rows],
        "today_label": today_label_it(),
        "saved_profiles": saved_profiles,
    }


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
    session: AsyncSession = Depends(get_async_session),
) -> HTMLResponse:
    """
    Teacher-facing dashboard (CORE 2 #6.6 P5 — warm-academic brand pass).

    Surfaces a small, read-only derivation of the user's lessons (count,
    most-recent draft to resume, last 3 lessons, last 5 activity events).
    No backend logic / agent / SSE / route changes — the dashboard simply
    queries the existing ``lesson`` table and renders it. Anonymous visitors
    see a sign-in CTA instead of the personalised dashboard.
    """
    dashboard = await _build_dashboard_context(session, user)
    return templates.TemplateResponse(
        "pages/home.html",
        {
            "request": request,
            "title": "AixLearning · Agentic GraphRAG",
            "user": user,
            "active_nav": "dashboard",
            "dashboard": dashboard,
        },
    )


@_pages_router.get("/about-ai", response_class=HTMLResponse, name="webui_about_ai")
async def about_ai(
    request: Request,
    user: Optional[User] = Depends(optional_current_user),
) -> HTMLResponse:
    """
    "Come funziona l'IA" — teacher AI-literacy guide (Wave 5 #23).

    EU AI Act Art. 4 (AI literacy) deliverable: a static, non-technical page
    explaining the 4-agent pipeline, the curated knowledge base, the
    grounded-vs-general coverage signal, the human-in-the-loop design, and the
    system's limits. Purely additive — no agent / SSE / DB / route changes.
    Anonymous access is allowed so the guide is reachable from the public
    footer without bouncing through the login wall.
    """
    return templates.TemplateResponse(
        "pages/about_ai.html",
        {
            "request": request,
            "title": "Come funziona l'IA · AixLearning",
            "user": user,
            "active_nav": "about",
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
