"""
Teacher file uploads (CORE 2 #6.6 P3).

Files are stored on disk per lesson; excerpts are joined server-side into
``AgentState.teacher_provided_context`` for the Writer only — they are never
ingested into the shared Knowledge Graph.

Since Fase 2 · PR 5 the excerpt comes from ``aix.webui.materials.parsing``: the
bytes are checked before anything reads them, DOCX, PPTX and images are
accepted, and pages without a text layer go through OCR. What the parser could
not do is recorded on the manifest entry under ``parse``, so the chip in the
chat can say it instead of the lesson silently being written from less.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from aix.webui.materials.parsing import (
    IntakeRejected,
    ParseOptions,
    check_upload,
    parse_document,
)
from aix.webui.materials.parsing import intake as intake_limits
from aix.webui.materials.parsing.intake import ALLOWED
from aix.webui.materials.parsing.parsers import FormatDisabled

logger = logging.getLogger(__name__)

#: The attachment limit, which is ``intake.MAX_BYTES`` and not a second opinion.
#:
#: Kept as a name here because this module is the legacy chat-attachment path
#: and its callers import it, but it is an alias: the two gates a file passes
#: — this one and the parser's — used to hold separate copies of the number,
#: and raising one of them left the other refusing the file (2026-09-22).
MAX_FILE_BYTES = intake_limits.MAX_BYTES
MAX_FILES_PER_LESSON = 12
#: Was 8000: a three-page history handout (9985 characters) lost its last fifth
#: — distensione, Gorbaciov, 1989 — and the lesson never mentioned them. The
#: total across files stays at 48k, which is what bounds the Writer's prompt.
MAX_EXCERPT_PER_FILE = 24_000
MAX_TOTAL_EXCERPT = 48_000

#: What the attachment accepted before this work, and accepts again whenever
#: ``AIX_UPLOAD_PARSER_ENABLED`` is off (§5.2: flags off means the previous
#: product). The new parser's allowlist is ``parsing.intake.ALLOWED``.
LEGACY_EXT = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

#: Kept as module-level names because other modules and tests import them.
#: They describe the *parser* allowlist; what is accepted right now is
#: ``allowed_extensions()``, which depends on the flag.
ALLOWED_EXT = frozenset(ALLOWED)

MIME_FOR_EXT = dict(ALLOWED)


def too_large_message() -> str:
    """One wording for one limit, defined with the limit itself."""
    return intake_limits.too_large_message()


def allowed_extensions() -> dict[str, str]:
    """Extension → media type, for whichever intake is switched on."""
    from aix.webui.features import upload_parser_enabled

    return dict(ALLOWED) if upload_parser_enabled() else dict(LEGACY_EXT)


_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_DIR.parents[2]
UPLOAD_ROOT = _REPO_ROOT / "data" / "webui" / "uploads"


def lesson_upload_dir(lesson_id: uuid.UUID) -> Path:
    d = UPLOAD_ROOT / str(lesson_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _normalize_manifest(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and item.get("id"):
            out.append(item)
    return out


def total_excerpt_len(files: list[dict[str, Any]]) -> int:
    return sum(len(f.get("text_excerpt") or "") for f in files)


def _legacy_read(content: bytes, filename: str) -> tuple[str, str, None]:
    """The extraction this project had before the parser: pypdf, or a decode.

    Restored, not kept for nostalgia: §5.2 requires that with the new flags off
    the teacher gets the previous product, and the review of 2026-09-15 found
    the new parser running on the chat attachment regardless of any flag.

    One deliberate difference: a corrupt PDF raises ``ValueError`` and becomes a
    422 with an Italian message. The old path let pypdf's exception escape as a
    500, and restoring a server error is not backward compatibility.
    """
    import io

    ext = Path(filename).suffix.lower()
    if ext not in LEGACY_EXT:
        raise ValueError("Tipo file non supportato (solo PDF, TXT, Markdown).")

    if ext == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages[:50]]
        except Exception as exc:  # noqa: BLE001 — an unreadable file is an answer
            logger.info("[uploads] legacy path could not read %s", ext, exc_info=True)
            raise ValueError(
                "Il documento non può essere letto: è danneggiato o incompleto."
            ) from exc
        text = "\n".join(page for page in pages if page.strip())
    else:
        text = content.decode("utf-8", errors="replace")

    return LEGACY_EXT[ext], text, None


def read_upload(content: bytes, filename: str) -> tuple[str, str, Optional[dict[str, Any]]]:
    """Check and parse one upload: ``(media_type, text, parse summary)``.

    The summary is ``None`` on the legacy path, which has nothing to report and
    whose manifest entries must stay exactly the shape they were.

    Raises ``ValueError`` with a message the teacher can read.
    """
    from aix.webui.features import upload_parser_enabled

    if not upload_parser_enabled():
        return _legacy_read(content, filename)

    accepted = check_upload(filename, content)  # IntakeRejected is a ValueError

    # One OCR, the worker's. With the private RAG on, this request's parse only
    # produces a fallback excerpt nobody sends, and the worker parses the same
    # bytes again for the index. Running OCR here too meant the teacher waited
    # through it (about 12 s for a 114-page magazine) and the VM ran two OCRs at
    # once -- this one outside the worker's one-at-a-time limit, which exists
    # for the pilot's two CPUs (2026-09-25). With the RAG off the excerpt is
    # the only way the document reaches the Writer, so there OCR stays.
    from aix.webui.materials.embedding import private_rag_enabled

    options = ParseOptions(ocr_enabled=not private_rag_enabled())
    try:
        ir = parse_document(content, accepted.media_type, options)
    except FormatDisabled as exc:
        raise ValueError("Questo tipo di file è momentaneamente disattivato.") from exc
    except Exception as exc:  # noqa: BLE001 — an unreadable file is the teacher's answer
        logger.info("[uploads] %s could not be parsed", accepted.media_type, exc_info=True)
        raise ValueError("Il documento non può essere letto: è danneggiato o incompleto.") from exc

    quality = ir.quality()
    summary = {
        "parser": quality["parser"],
        "pages": quality["pages"],
        "ocr_pages": quality["ocr_pages"],
        "unread_pages": quality["unread_pages"],
        "ocr_mean_confidence": quality["ocr_mean_confidence"],
        "empty": quality["empty"],
        "warnings": quality["warnings"],
    }
    return accepted.media_type, ir.text, summary


def extract_text(content: bytes, filename: str) -> str:
    """The text of an upload, as the Writer receives it."""
    return read_upload(content, filename)[1]


def truncate_excerpt(text: str) -> str:
    t = text.strip()
    if len(t) <= MAX_EXCERPT_PER_FILE:
        return t
    return t[: MAX_EXCERPT_PER_FILE - 1] + "…"


def save_upload(
    lesson_id: uuid.UUID,
    filename: str,
    content: bytes,
    existing_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Validate, parse, write disk, return (updated manifest list, new entry).

    Synchronous and potentially slow — OCR runs here — so the route calls it
    in a thread.

    Raises:
        ValueError: validation failed (message is user-safe Italian where possible).
    """
    if not content:
        raise ValueError("File vuoto.")

    if len(content) > MAX_FILE_BYTES:
        raise ValueError(too_large_message())

    manifest = _normalize_manifest(existing_raw)
    if len(manifest) >= MAX_FILES_PER_LESSON:
        raise ValueError(f"Troppi file (massimo {MAX_FILES_PER_LESSON}).")

    media_type, text, summary = read_upload(content, filename)

    excerpt = truncate_excerpt(text)
    if total_excerpt_len(manifest) + len(excerpt) > MAX_TOTAL_EXCERPT:
        # The excerpt is what the Writer is *pasted* when there is no private
        # index — a fallback, not the document. With the private RAG on, the
        # lesson is written from retrieved passages and this text is never sent,
        # so refusing an upload over its size refuses a file for the weight of
        # something nobody reads.
        #
        # On 2026-09-22 that is what happened: the Indicazioni nazionali filled
        # the budget on their own, and the third attachment came back 422. The
        # teacher read it as "this cannot take more than two documents".
        #
        # So the file is stored with no excerpt. It is parsed, chunked, indexed
        # and searchable exactly like the others — only the fallback text is
        # dropped, which is the thing there was no room for. With the private
        # RAG off the excerpt is the *only* channel, and then the refusal is
        # honest: there is nowhere for the document to go.
        from aix.webui.materials.embedding import private_rag_enabled

        if not private_rag_enabled():
            raise ValueError("Troppo testo estratto complessivo: rimuovi un file o accorcia.")

        logger.info(
            "[upload] %s: excerpt dropped, the total budget is full (%d chars). "
            "The document is still indexed and searchable.",
            filename,
            total_excerpt_len(manifest),
        )
        excerpt = ""

    ext = Path(filename).suffix.lower()
    file_id = str(uuid.uuid4())
    stored_name = f"{file_id}{ext}"
    path = lesson_upload_dir(lesson_id) / stored_name
    path.write_bytes(content)

    entry: dict[str, Any] = {
        "id": file_id,
        "filename": Path(filename).name[:240] or stored_name,
        "mime": media_type,
        "size": len(content),
        "text_excerpt": excerpt,
        "stored_name": stored_name,
    }
    # Only the parser has something to say about how the reading went. On the
    # legacy path the entry keeps exactly the keys it always had.
    if summary is not None:
        entry["parse"] = summary
    manifest.append(entry)
    return manifest, entry


def delete_upload(
    lesson_id: uuid.UUID,
    file_id: str,
    existing_raw: Any,
) -> list[dict[str, Any]]:
    manifest = _normalize_manifest(existing_raw)
    kept: list[dict[str, Any]] = []
    removed: Optional[dict[str, Any]] = None
    for item in manifest:
        if item.get("id") == file_id:
            removed = item
            continue
        kept.append(item)
    if removed is None:
        raise ValueError("File non trovato.")

    stored = removed.get("stored_name")
    if isinstance(stored, str) and stored:
        path = lesson_upload_dir(lesson_id) / stored
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    return kept


__all__ = [
    "ALLOWED_EXT",
    "MAX_FILE_BYTES",
    "MIME_FOR_EXT",
    "IntakeRejected",
    "delete_upload",
    "extract_text",
    "read_upload",
    "save_upload",
]
