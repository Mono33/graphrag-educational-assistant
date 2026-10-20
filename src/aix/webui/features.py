"""Which of the new surfaces are switched on (Appendix B, §5.2).

**On by default since 2026-10-05**, by Angelo's decision: the defaults are the
pilot's configuration, and ``false`` in the ``.env`` is how a surface is turned
off. Until then the rule was the opposite -- with every flag off the product was
the previous one -- and a deploy with a forgotten line silently served the old
product. Every flag still exists, so any surface can be switched off without a
deploy; ``tests/unit/test_feature_defaults.py`` pins the defaults, and the suite
itself runs with them off (``tests/conftest.py``).

* the new parser on the chat attachment — DOCX, PPTX, images, magic-byte checks
  and OCR. Off, the paperclip goes back to PDF, TXT and Markdown read with
  pypdf;
* conditional retrieval and the light follow-up turns;
* the PDF and outline exports, additive slices Louis accepted (MD, TXT and
  ``/print`` are untouched), and the Word export added on 2026-10-14.

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
    """The new intake and parser on the chat attachment. On by default."""
    return _flag("AIX_UPLOAD_PARSER_ENABLED", default=True)


def export_pdf_enabled() -> bool:
    return _flag("AIX_EXPORT_PDF_ENABLED", default=True)


def export_docx_enabled() -> bool:
    """The Word download: the lesson as an editable document. On by default."""
    return _flag("AIX_EXPORT_DOCX_ENABLED", default=True)


def export_outline_enabled() -> bool:
    return _flag("AIX_EXPORT_OUTLINE_ENABLED", default=True)


def conditional_retrieval_enabled() -> bool:
    """A follow-up may skip the Knowledge Graph search: the toggle is shown only then."""
    return _flag("AIX_CONDITIONAL_RETRIEVAL_ENABLED", default=True)


def light_turns_enabled() -> bool:
    """After the first lesson, a turn is one streamed answer unless she asks for a
    whole new lesson. On by default."""
    return _flag("AIX_LIGHT_TURNS_ENABLED", default=True)


def material_library_enabled() -> bool:
    """Enable reuse/detach/delete actions in the new material library.

    The read-only library is additive and always available.  Lifecycle changes
    stay opt-in until the deployment has passed the migration and persistence
    checks, so disabling this flag restores the exact legacy remove-and-retire
    behaviour.
    """
    return _flag("AIX_MATERIAL_LIBRARY_ENABLED", default=False)


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
    templates.env.globals["material_library_enabled"] = material_library_enabled
    from aix.core.config import api_docs_enabled

    templates.env.globals["api_docs_enabled"] = api_docs_enabled

    from aix.webui.materials.saved_resources import resource_key

    templates.env.globals["saved_resource_key"] = resource_key
    templates.env.globals["export_pdf_enabled"] = export_pdf_enabled
    templates.env.globals["export_outline_enabled"] = export_outline_enabled
    templates.env.globals["export_docx_enabled"] = export_docx_enabled
