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

from aix.webui.materials.parsing import IntakeRejected, check_upload, parse_document
from aix.webui.materials.parsing.intake import ALLOWED
from aix.webui.materials.parsing.parsers import FormatDisabled

logger = logging.getLogger(__name__)

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES_PER_LESSON = 12
MAX_EXCERPT_PER_FILE = 8000
MAX_TOTAL_EXCERPT = 48_000

ALLOWED_EXT = frozenset(ALLOWED)

MIME_FOR_EXT = dict(ALLOWED)

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


def read_upload(content: bytes, filename: str) -> tuple[str, str, dict[str, Any]]:
    """Check and parse one upload: ``(media_type, text, parse summary)``.

    Raises ``ValueError`` with a message the teacher can read. A file that
    breaks the parser is a refusal, not a 500 — the legacy path answered a
    corrupt PDF with a server error.
    """
    accepted = check_upload(filename, content)  # IntakeRejected is a ValueError
    try:
        ir = parse_document(content, accepted.media_type)
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
        raise ValueError("File troppo grande (massimo 10 MB).")

    manifest = _normalize_manifest(existing_raw)
    if len(manifest) >= MAX_FILES_PER_LESSON:
        raise ValueError(f"Troppi file (massimo {MAX_FILES_PER_LESSON}).")

    media_type, text, summary = read_upload(content, filename)

    excerpt = truncate_excerpt(text)
    if total_excerpt_len(manifest) + len(excerpt) > MAX_TOTAL_EXCERPT:
        raise ValueError("Troppo testo estratto complessivo: rimuovi un file o accorcia.")

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
        "parse": summary,
    }
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
