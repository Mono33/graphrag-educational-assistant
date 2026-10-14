"""The lesson as a printable running order (§10.4, ``format=outline``).

A teacher in front of a class does not read the lesson; they glance at what
comes next and how long it takes. The outline is that glance: the sections in
order, indented by level, with their minutes, and the total checked against the
duration the lesson declares.

Plain text on purpose. It prints on anything, pastes into a register or a chat,
and reads the same in a terminal as on paper.
"""

from __future__ import annotations

from aix.export.export_ir import ExportIRV1

_RULE = "─" * 48


def _minutes(value: int | None) -> str:
    return f"{value:>3}′" if value is not None else "    "


def render_outline(ir: ExportIRV1) -> str:
    lines = [ir.title, ""]

    facts = []
    if ir.duration_minutes:
        facts.append(f"Durata: {ir.duration_minutes} minuti")
    facts.append(ir.disclosure_short)
    lines += [" · ".join(facts), _RULE, ""]

    if not ir.sections:
        lines.append("(la lezione non ha sezioni)")
    for section in ir.sections:
        indent = "  " * max(0, section.level - 2)
        lines.append(f"{_minutes(section.minutes)}  {indent}{section.title}")

    total = ir.outline_total_minutes
    if total is not None:
        lines += ["", _RULE]
        summary = f"{total:>3}′  totale delle sezioni"
        if ir.duration_minutes and total != ir.duration_minutes:
            summary += f" (la lezione dichiara {ir.duration_minutes})"
        lines.append(summary)

    lines += ["", ir.disclosure_long]
    return "\n".join(lines) + "\n"
