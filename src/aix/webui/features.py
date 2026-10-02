"""Which of the new surfaces are switched on (Appendix B, §5.2).

The rule the architecture states and the review of 2026-09-15 found broken:
**with every new flag off, the product is the previous product.** Not "almost",
and not "except the paperclip": a teacher on the pilot must get the WebUI they
had until someone decides otherwise, and that decision must be a line in
``.env`` rather than a merge.

What was on without a flag, and is now behind one:

* the new parser on the chat attachment — DOCX, PPTX, images, magic-byte checks
  and OCR. Off by default: the paperclip goes back to PDF, TXT and Markdown read
  with pypdf;
* the PDF and outline exports. On by default, because Louis accepted them as an
  additive slice (MD, TXT and ``/print`` are untouched and remain the default
  path) — but a flag exists now, so they can be switched off without a deploy.

The functions are registered as Jinja globals, so a template asks the same
question the server does instead of keeping its own list of extensions in sync.
"""

from __future__ import annotations

import os


def _flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def upload_parser_enabled() -> bool:
    """The new intake and parser on the chat attachment. Off by default."""
    return _flag("AIX_UPLOAD_PARSER_ENABLED", default=False)


def export_pdf_enabled() -> bool:
    return _flag("AIX_EXPORT_PDF_ENABLED", default=True)


def export_outline_enabled() -> bool:
    return _flag("AIX_EXPORT_OUTLINE_ENABLED", default=True)


def conditional_retrieval_enabled() -> bool:
    """A follow-up may skip the Knowledge Graph search: the toggle is shown only then."""
    return _flag("AIX_CONDITIONAL_RETRIEVAL_ENABLED", default=False)


def light_turns_enabled() -> bool:
    """After the first lesson, a turn is one streamed answer unless she asks for a
    whole new lesson. Off by default; the pilot switches it on."""
    return _flag("AIX_LIGHT_TURNS_ENABLED", default=False)


def upload_accept_attribute() -> str:
    """What the file picker offers, which is what the server will accept."""
    from aix.webui.lessons.uploads import allowed_extensions

    return ",".join(sorted(allowed_extensions()))


def upload_hint() -> str:
    """The tooltip on the paperclip, in the teacher's words."""
    from aix.webui.materials.parsing.intake import MAX_BYTES

    limit = f"max {MAX_BYTES // (1024 * 1024)} MB"
    if upload_parser_enabled():
        return (
            "Allega PDF, Word, PowerPoint, testo o immagini (anche scansioni) "
            f"come contesto per la lezione ({limit})"
        )
    return f"Allega PDF, TXT o Markdown come contesto per la lezione ({limit})"


def register(templates) -> None:
    """Make the flags available to templates, next to ``csrf_token``."""
    templates.env.globals["upload_accept"] = upload_accept_attribute
    templates.env.globals["upload_hint"] = upload_hint
    templates.env.globals["conditional_retrieval_enabled"] = conditional_retrieval_enabled
    templates.env.globals["light_turns_enabled"] = light_turns_enabled
    templates.env.globals["export_pdf_enabled"] = export_pdf_enabled
    templates.env.globals["export_outline_enabled"] = export_outline_enabled
