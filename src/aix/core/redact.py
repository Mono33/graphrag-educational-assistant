"""What the logs may say of a teacher's words (2026-10-23).

A teacher's message can describe her class -- «due bambini con ADHD», a name, a
diagnosis -- and her documents and the lessons built on them can too. On
staging the logs carried whole requests (``query=``, the live-media concepts)
and sentences of lessons (the citation check). Where ``ENVIRONMENT`` is
staging or production they now carry only a length and a fingerprint: the same
text gives the same fingerprint, so the lines of one request still read
together. Locally the text stays, for debugging; ``AIX_LOG_TEACHER_TEXT``
decides either way.
"""

from __future__ import annotations

import hashlib
import os

_LOCAL = {"development", "dev", "local", "test"}


def teacher_text_in_logs() -> bool:
    raw = os.getenv("AIX_LOG_TEACHER_TEXT", "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return os.getenv("ENVIRONMENT", "production").strip().lower() in _LOCAL


def private(text: object, keep: int = 50) -> str:
    """``text`` for a log line: its first ``keep`` characters locally, else its
    length and a fingerprint («<142 car. #3fa9c2d1>»)."""
    value = "" if text is None else str(text)
    if teacher_text_in_logs():
        return value if len(value) <= keep else value[:keep] + "…"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"<{len(value)} car. #{digest}>"
