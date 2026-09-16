"""Pytest configuration for graphaixlearning tests.

With ``pip install -e .`` (Phase 3A), all project modules are importable
without ``sys.path`` manipulation.

The one shared fixture: every test starts with the feature flags at their
defaults, whatever the developer's ``.env`` says.

``aix.core.config`` loads ``.env`` into the process environment on import, so a
flag switched on for a manual test on a laptop used to switch it on for the
suite too. On 2026-09-16 turning on ``AIX_PRIVATE_RAG_ENABLED`` locally made four
tests fail — the default graph suddenly had a second retriever — and the same
leak is the likeliest reason a topology test failed on one reviewer's machine and
passed on another. A test that needs a flag sets it itself, with ``monkeypatch``,
which runs after this and wins.
"""

from __future__ import annotations

import pytest

#: Flags whose default is part of what the tests assert. Listed explicitly
#: rather than every ``AIX_*`` variable: some are configuration a test legitimately
#: inherits (timeouts, model names), and clearing those would hide real defaults.
FEATURE_FLAGS = (
    "AIX_UPLOAD_PARSER_ENABLED",
    "AIX_MATERIALS_ENABLED",
    "AIX_PRIVATE_RAG_ENABLED",
    "AIX_VALIDATOR_ENABLED",
    "AIX_CONDITIONAL_RETRIEVAL_ENABLED",
    "AIX_CORRECTIVE_RAG_ENABLED",
    "AIX_EXPORT_PDF_ENABLED",
    "AIX_EXPORT_OUTLINE_ENABLED",
)


@pytest.fixture(autouse=True)
def _feature_flags_at_their_defaults(monkeypatch):
    for name in FEATURE_FLAGS:
        monkeypatch.delenv(name, raising=False)
