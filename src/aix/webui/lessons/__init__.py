"""
WebUI lessons subpackage (CORE 2 #6.6 P1).

Holds the persistent ``Lesson`` row created when a teacher submits the
EducationalProfile form, and the HTML routes that render the form and the
lesson placeholder page.

P1 scope:
    - GET  /webui/lesson/new       — render the form (auth required)
    - POST /webui/lesson           — create Lesson row, redirect to /lesson/{id}
    - GET  /webui/lesson/{id}      — show placeholder page (P2 will stream the
                                      agent response into it via SSE)

The Lesson row stores the EducationalProfile as a JSON blob (no per-field
columns) so the schema can evolve in CORE 1 without DB migrations during
this iteration. Once the schema is frozen we can normalize hot fields into
proper columns for indexed queries (e.g. "all lessons for grade X").
"""

from aix.webui.lessons.routes import router

__all__ = ["router"]
