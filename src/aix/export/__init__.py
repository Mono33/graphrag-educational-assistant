"""Exporting a lesson: one structured reading, many renderings (Fase 3 · PR 7).

The pipeline follows §10.1 of the implementation guide::

    lesson_plan_md
      → LegacyLessonAdapter   (lesson_document.py)
      → LessonDocumentV1
      → ExportIRV1            (export_ir.py)
          ├─ OutlineTextRenderer (outline.py)
          └─ PdfRenderer         (pdf.py)

Markdown and plain text keep their existing renderers in the route: the guide
requires them not to change, and they do not need a document model to work.

The adapter exists because the Writer still produces a Markdown string rather
than a structure. When a structured Writer arrives, it produces
``LessonDocumentV1`` directly and everything downstream of the adapter stays as
it is — which is the reason for putting a document model in the middle at all.
"""

from aix.export.export_ir import ExportIRV1, ExportSection, build_export_ir
from aix.export.lesson_document import (
    Block,
    LessonDocumentV1,
    Run,
    adapt_lesson_markdown,
)
from aix.export.outline import render_outline

__all__ = [
    "Block",
    "ExportIRV1",
    "ExportSection",
    "LessonDocumentV1",
    "Run",
    "adapt_lesson_markdown",
    "build_export_ir",
    "render_outline",
]
