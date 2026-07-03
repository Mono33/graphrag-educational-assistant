"""EU AI Act Article 50 — AI-generated content marking (Wave 5 #21/#22).

Single source of truth for how this system declares that lesson content is
AI-generated, across every surface:

  * **machine-readable** — an HTML comment prepended to each finalized
    ``lesson_plan_md`` (provenance for downstream tools / crawlers), plus the
    ``X-AI-Generated: true`` HTTP header on content-bearing endpoints;
  * **human-readable** — the Italian "Generato dall'IA" disclosure shown in
    the WebUI (footer + per-card badge) and in exported artifacts.

Defining the wording, the comment format, and the header here means they are
declared once; the API, WebUI, exports, and (where practical) templates all
reference the same values. The machine comment carries a ``trace_id`` — until
Langfuse run-tracing is wired (#24), callers pass the best identifier they
have (``lesson.id`` on the WebUI path, ``session_id`` on the API path).
"""

from __future__ import annotations

# Stable system identifier embedded in the provenance comment.
AI_SYSTEM_NAME = "agentic-graphrag"

# HTTP response header (machine-readable signal on content endpoints).
AI_GENERATED_HEADER = "X-AI-Generated"
AI_GENERATED_HEADER_VALUE = "true"

# Opening token of the machine-readable comment — used for idempotency checks
# and for stripping the comment back out (e.g. plain-text export).
_MARKER_PREFIX = "<!-- ai-generated:"

# Human-readable Italian disclosures (Article 50 transparency).
DISCLOSURE_SHORT = "Generato dall'IA"
DISCLOSURE_LONG = (
    "Contenuto generato dall'intelligenza artificiale. "
    "Verificare sempre l'accuratezza prima dell'uso in classe."
)


def build_marking_comment(trace_id: str | None = None) -> str:
    """Return the machine-readable HTML comment (no trailing newline)."""
    tid = trace_id or "n/a"
    return f"<!-- ai-generated: true, system: {AI_SYSTEM_NAME}, trace_id: {tid} -->"


def is_marked(md: str) -> bool:
    """True if ``md`` already carries the machine-readable marking comment."""
    return bool(md) and md.lstrip().startswith(_MARKER_PREFIX)


def ensure_marking(md: str, trace_id: str | None = None) -> str:
    """Idempotently prepend the marking comment to ``md``.

    Safe to call more than once: returns ``md`` unchanged when it is already
    marked. Blank input is returned as-is (an empty plan is treated as an
    error upstream, before this is called).
    """
    if not md or not md.strip():
        return md
    if is_marked(md):
        return md
    return f"{build_marking_comment(trace_id)}\n\n{md}"


def strip_marking(md: str) -> str:
    """Remove a leading marking comment (for plain-text rendering/exports)."""
    if not md:
        return md
    stripped = md.lstrip()
    if stripped.startswith(_MARKER_PREFIX):
        end = stripped.find("-->")
        if end != -1:
            return stripped[end + 3 :].lstrip("\n")
    return md
