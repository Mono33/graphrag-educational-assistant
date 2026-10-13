"""Pytest configuration for graphaixlearning tests.

With ``pip install -e .`` (Phase 3A), all project modules are importable
without ``sys.path`` manipulation.

The one shared fixture: every test starts with the feature flags switched
off, whatever the developer's ``.env`` -- or the code's defaults -- say.

``aix.core.config`` loads ``.env`` into the process environment on import, so a
flag switched on for a manual test on a laptop used to switch it on for the
suite too. On 2026-09-16 turning on ``AIX_PRIVATE_RAG_ENABLED`` locally made four
tests fail — the default graph suddenly had a second retriever — and the same
leak is the likeliest reason a topology test failed on one reviewer's machine and
passed on another. A test that needs a flag sets it itself, with ``monkeypatch``,
which runs after this and wins.

Switched off, not merely unset: since 2026-10-05 the code's defaults are the
pilot's -- the new surfaces on -- so an unset flag would turn them on for every
test. The suite keeps testing what it always tested, against the previous
product; ``test_feature_defaults.py`` pins the defaults themselves.
"""

from __future__ import annotations

import pytest

#: The flags the suite runs with, and the value that switches each off. Listed
#: explicitly rather than every ``AIX_*`` variable: some are configuration a test
#: legitimately inherits (timeouts, model names).
FEATURE_FLAGS_OFF = {
    "AIX_UPLOAD_PARSER_ENABLED": "false",
    "AIX_MATERIALS_ENABLED": "false",
    "AIX_PRIVATE_RAG_ENABLED": "false",
    "AIX_VALIDATOR_ENABLED": "false",
    "AIX_CONDITIONAL_RETRIEVAL_ENABLED": "false",
    "AIX_CORRECTIVE_RAG_ENABLED": "false",
    "AIX_LIGHT_TURNS_ENABLED": "false",
    "AIX_OCR_BACKEND": "tesseract",
    "AIX_TRACING_ENABLED": "false",
    "AIX_MEDIA_LIVE_ENABLED": "false",
    "AIX_MEDIA_RERANK_ENABLED": "false",
    "AIX_PROMPT_CACHE_ENABLED": "false",
    "AIX_RESULT_CACHE_ENABLED": "false",
    # Not a flag but a switch of the same kind: a cheaper planner on a laptop's
    # .env must not change what the suite plans with. Empty is today's behavior.
    "AIX_PLANNER_MODEL": "",
    "AIX_LIGHT_TOOLS_ENABLED": "false",
}

#: On by default since before 2026-10-05, and tested at their default.
FEATURE_FLAGS_AT_DEFAULT = ("AIX_EXPORT_PDF_ENABLED", "AIX_EXPORT_OUTLINE_ENABLED")


@pytest.fixture(autouse=True)
def _feature_flags_switched_off(monkeypatch):
    for name, value in FEATURE_FLAGS_OFF.items():
        monkeypatch.setenv(name, value)
    for name in FEATURE_FLAGS_AT_DEFAULT:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _process_singletons_that_hold_a_loop():
    """Reset the process-wide objects that own asyncio primitives.

    Two of them, and both fail the same way: an object created on one test's
    event loop is reused on the next test's, and raises «is bound to a
    different event loop». Each test gets its own loop, so this only bites from
    the second test onwards — and only in some file orders, which is exactly
    the kind of failure that gets dismissed as a flake. On 2026-09-21 it was
    ``tests/api`` followed by ``tests/unit``: the first SSE response of the run
    bound sse-starlette's shutdown Event, and the next test to open a stream
    died on it.

    * ``sse_starlette.sse.AppStatus.should_exit_event`` — one Event per
      process, created on first use. A library-level issue, not ours.
    * ``aix.core.run_registry._REGISTRY`` — the in-memory backend owns locks.

    Production has a single loop per process and hits neither; both singletons
    are right there. This resets them between tests rather than weakening them.
    """
    import sse_starlette.sse as sse

    import aix.core.run_registry as registry

    registry._REGISTRY = None
    sse.AppStatus.should_exit_event = None
    yield
    registry._REGISTRY = None
    sse.AppStatus.should_exit_event = None
