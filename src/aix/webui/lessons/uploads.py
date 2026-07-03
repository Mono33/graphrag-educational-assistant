"""
Teacher file uploads (CORE 2 #6.6 P3).

Files are stored on disk per lesson; excerpts are joined server-side into
``AgentState.teacher_provided_context`` for the Writer only — they are never
ingested into the shared Knowledge Graph.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any, Optional

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES_PER_LESSON = 12
MAX_EXCERPT_PER_FILE = 8000
MAX_TOTAL_EXCERPT = 48_000

ALLOWED_EXT = frozenset({".pdf", ".txt", ".md", ".markdown"})

MIME_FOR_EXT = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

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


def extract_text(content: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        parts: list[str] = []
        for page in reader.pages[:50]:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts)
    return content.decode("utf-8", errors="replace")


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
    Validate, write disk, return (updated manifest list, new entry).

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

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValueError("Tipo file non supportato (solo PDF, TXT, Markdown).")

    excerpt_full = extract_text(content, filename)
    excerpt = truncate_excerpt(excerpt_full)
    if total_excerpt_len(manifest) + len(excerpt) > MAX_TOTAL_EXCERPT:
        raise ValueError("Troppo testo estratto complessivo: rimuovi un file o accorcia.")

    file_id = str(uuid.uuid4())
    stored_name = f"{file_id}{ext}"
    path = lesson_upload_dir(lesson_id) / stored_name
    path.write_bytes(content)

    entry: dict[str, Any] = {
        "id": file_id,
        "filename": Path(filename).name[:240] or stored_name,
        "mime": MIME_FOR_EXT.get(ext, "application/octet-stream"),
        "size": len(content),
        "text_excerpt": excerpt,
        "stored_name": stored_name,
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
