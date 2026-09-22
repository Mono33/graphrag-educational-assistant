"""Whether a file may be parsed at all (§7.2, §7.4).

Every check here exists because a file is a claim made by whoever uploaded it.
The filename is chosen by a person, the Content-Type by a browser, and neither
says what the bytes are. So the bytes are read:

* the **extension** must be on the allowlist, and must not hide an active one —
  ``programma.pdf.exe`` is refused, not accepted as a PDF;
* the **magic bytes** must agree with the extension: a PDF renamed ``.docx`` is
  refused rather than handed to the wrong parser;
* **macro-enabled Office** is refused even when renamed ``.docx``: the package is
  opened and a ``vbaProject.bin`` inside is enough;
* **encrypted documents** are refused, because a parser that cannot read them
  would silently produce nothing and call it success;
* **archives are bounded** before anything is extracted — entry count,
  uncompressed size and compression ratio — since a DOCX is a ZIP, and a ZIP is
  how a 10 MB upload becomes 10 GB in memory;
* **pages and pixels are bounded**, because on a 4 GB VM an OCR pass over a
  300-page scan is a denial of service whether anyone meant it or not.

A rejection carries a code and an Italian sentence. The code goes to logs and
metrics; the sentence is what the teacher reads.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Optional

import filetype

#: The biggest upload anything in this system will look at, and the **only**
#: place this number is written.
#:
#: It was 10 MB here and 10 MB again in ``lessons.uploads``, each with its own
#: hand-written Italian sentence, and a third copy in the paperclip's tooltip.
#: On 2026-09-22 the limit was raised in one of the three: the teacher went on
#: being refused, by the other gate, with a message quoting the old number. Two
#: checks that must agree are one check written twice.
#:
#: Raised from 10 MB when a 13.5 MB magazine issue was refused: 114 pages,
#: 375 000 characters, read in four seconds with no OCR. Bytes are a poor proxy
#: for the work — that is bounded by ``MAX_PAGES`` and by OCR concurrency. What
#: this guards is memory, since the upload is held whole in RAM while it is
#: parsed, and on the 4 GB pilot VM that is the ceiling to respect.
MAX_BYTES = 100 * 1024 * 1024


def too_large_message() -> str:
    """What the teacher reads, derived from the limit actually in force."""
    return f"File troppo grande (massimo {MAX_BYTES // (1024 * 1024)} MB)."

#: Reading a text layer is cheap, and the ministerial documents in the corpus run
#: to 68 and 97 pages. What the VM cannot afford is OCR on hundreds of pages, and
#: that is capped separately, per document, by ``ParseOptions.max_ocr_pages``.
MAX_PAGES = 200
MAX_PIXELS = 40_000_000
MAX_ZIP_ENTRIES = 2_000
MAX_ZIP_UNCOMPRESSED = 100 * 1024 * 1024
MAX_ZIP_RATIO = 100

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
TEXT = "text/plain"
MARKDOWN = "text/markdown"

#: Extension → the media type it must turn out to be.
ALLOWED: dict[str, str] = {
    ".pdf": PDF,
    ".docx": DOCX,
    ".pptx": PPTX,
    ".txt": TEXT,
    ".md": MARKDOWN,
    ".markdown": MARKDOWN,
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

#: Suffixes that make a name dangerous wherever they appear in it.
_ACTIVE = {
    ".exe", ".dll", ".bat", ".cmd", ".com", ".scr", ".msi", ".ps1", ".vbs", ".js",
    ".jar", ".sh", ".html", ".htm", ".svg", ".xhtml", ".hta", ".lnk",
    ".docm", ".dotm", ".pptm", ".potm", ".xlsm", ".xltm", ".zip", ".rar", ".7z",
}  # fmt: skip

IMAGE_TYPES = {v for v in ALLOWED.values() if v.startswith("image/")}


class IntakeRejected(ValueError):
    """The file may not be parsed. ``code`` for logs, ``str()`` for the teacher."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class IntakeResult:
    media_type: str
    extension: str
    page_count: Optional[int] = None


def _extension(filename: str) -> str:
    suffixes = [s.lower() for s in PurePosixPath(filename.replace("\\", "/")).suffixes]
    if not suffixes:
        raise IntakeRejected("no_extension", "Il file non ha un'estensione riconoscibile.")
    if any(s in _ACTIVE for s in suffixes):
        raise IntakeRejected(
            "active_extension",
            "Tipo di file non consentito: l'estensione indica un contenuto eseguibile o attivo.",
        )
    extension = suffixes[-1]
    if extension not in ALLOWED:
        raise IntakeRejected(
            "extension_not_allowed",
            "Tipo di file non supportato (PDF, Word, PowerPoint, testo o immagine).",
        )
    return extension


def _check_ooxml(content: bytes, expected: str) -> None:
    """Open the package, bound it, and make sure it is what it claims."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise IntakeRejected("not_ooxml", "Il file non è un documento Office valido.") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_ENTRIES:
            raise IntakeRejected(
                "archive_too_many_entries", "Il documento contiene troppi elementi."
            )

        uncompressed = sum(info.file_size for info in infos)
        compressed = max(1, sum(info.compress_size for info in infos))
        if uncompressed > MAX_ZIP_UNCOMPRESSED or uncompressed / compressed > MAX_ZIP_RATIO:
            raise IntakeRejected(
                "archive_bomb",
                "Il documento si espande in modo anomalo e non può essere elaborato.",
            )

        names = {info.filename for info in infos}
        if any(name.endswith("vbaProject.bin") for name in names):
            raise IntakeRejected(
                "macro_enabled",
                "I documenti con macro non sono consentiti: salvalo come .docx o .pptx normale.",
            )

        marker = "word/document.xml" if expected == DOCX else "ppt/presentation.xml"
        if "[Content_Types].xml" not in names or marker not in names:
            raise IntakeRejected(
                "magic_mismatch", "Il contenuto del file non corrisponde alla sua estensione."
            )


def _check_pdf(content: bytes) -> int:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(content))
        # Many official PDFs are "encrypted" only against editing, with an empty
        # opening password: the DM 254/2012 in the corpus is one. Those open; a
        # document that really needs a password does not.
        if reader.is_encrypted and not reader.decrypt(""):
            raise IntakeRejected(
                "encrypted", "Il PDF è protetto da password e non può essere letto."
            )
        pages = len(reader.pages)
    except IntakeRejected:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise IntakeRejected("unreadable_pdf", "Il PDF è danneggiato o incompleto.") from exc

    if pages > MAX_PAGES:
        raise IntakeRejected(
            "too_many_pages", f"Il documento ha {pages} pagine: il limite è {MAX_PAGES}."
        )
    return pages


def _check_image(content: bytes) -> None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise IntakeRejected(
            "unreadable_image", "L'immagine è danneggiata o non leggibile."
        ) from exc

    if width * height > MAX_PIXELS:
        raise IntakeRejected(
            "image_too_large", "L'immagine è troppo grande: riducine la risoluzione e riprova."
        )


def _check_text(content: bytes) -> None:
    if b"\x00" in content:
        raise IntakeRejected("magic_mismatch", "Il file di testo contiene dati binari.")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IntakeRejected(
            "not_utf8", "Il file di testo non è in UTF-8: salvalo con quella codifica."
        ) from exc


def check_upload(filename: str, content: bytes) -> IntakeResult:
    """Accept the file for parsing, or raise ``IntakeRejected``."""
    if not content:
        raise IntakeRejected("empty", "File vuoto.")
    if len(content) > MAX_BYTES:
        raise IntakeRejected("too_large", too_large_message())

    extension = _extension(filename)
    expected = ALLOWED[extension]

    if content[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        # OLE compound file: legacy .doc/.ppt, or an encrypted OOXML package.
        raise IntakeRejected(
            "ole_container",
            "Formato Office non supportato o documento cifrato: salvalo come .docx o .pptx.",
        )

    if expected in (DOCX, PPTX):
        _check_ooxml(content, expected)
        return IntakeResult(expected, extension)

    if expected in (TEXT, MARKDOWN):
        _check_text(content)
        return IntakeResult(expected, extension)

    sniffed = filetype.guess_mime(content)
    if sniffed != expected:
        raise IntakeRejected(
            "magic_mismatch", "Il contenuto del file non corrisponde alla sua estensione."
        )

    if expected == PDF:
        return IntakeResult(expected, extension, page_count=_check_pdf(content))

    _check_image(content)
    return IntakeResult(expected, extension, page_count=1)
