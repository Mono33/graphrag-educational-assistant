"""Parsing uploaded material into ``DocumentIRV1`` (Fase 2 · PR 5).

Three layers, each of which can refuse a file on its own:

* ``intake`` decides whether a file may be parsed at all — extension allowlist,
  magic bytes, double extensions, macro-enabled Office, encrypted documents,
  archive bombs, page and pixel limits. §7.2: the Content-Type a browser sends
  is a claim, not a fact, and a file that fails here never reaches a parser;
* ``parsers`` turn an accepted file into blocks with a locator and a confidence;
* ``ocr`` reads the pages that have no usable text layer, one page at a time.

Everything is chosen to fit the pilot VM (2 vCPU, 4 GB, shared with Postgres):
pure-Python parsers for PDF, DOCX and PPTX, and Tesseract for scans, with a page
cap, a per-page timeout and no parallelism.
"""

from aix.webui.materials.parsing.intake import IntakeRejected, IntakeResult, check_upload
from aix.webui.materials.parsing.ir import IR_SCHEMA_VERSION, DocumentIRV1, IRBlock
from aix.webui.materials.parsing.parsers import (
    PARSER_PROFILE,
    ParseOptions,
    parse_document,
)

__all__ = [
    "IR_SCHEMA_VERSION",
    "PARSER_PROFILE",
    "DocumentIRV1",
    "IRBlock",
    "IntakeRejected",
    "IntakeResult",
    "ParseOptions",
    "check_upload",
    "parse_document",
]
