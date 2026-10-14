"""What a violation is, and what a run of the rules produces.

Plain dataclasses rather than Pydantic models: these end up in ``AgentState``,
which the checkpointer serialises, so they have to survive a round trip through
JSON without help. ``as_dict`` is what the graph stores; ``from_dict`` is how the
UI and the retry read them back.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

#: A lesson cannot be delivered with one of these outstanding. Triggers a retry.
BLOCKING = "blocking"

#: Worth telling the teacher, not worth regenerating the lesson over. Rules that
#: rely on heuristics rather than on a declared fact start here, because a false
#: positive that silently rewrites a good lesson is worse than a visible note.
WARNING = "warning"

Severity = Literal["blocking", "warning"]


@dataclass(frozen=True)
class Violation:
    """One thing the lesson asks for that this classroom cannot do."""

    severity: Severity
    #: Stable identifier — ``byod_off``, ``duration_mismatch``. Used by tests,
    #: metrics and the UI; never shown raw to the teacher.
    rule: str
    #: The sentence from the lesson that triggered it. Quoting the evidence is
    #: what lets a teacher check the validator rather than trust it.
    evidence: str
    #: The instruction injected into the retry. Written as something to *do*,
    #: because it is read by the Writer, not by a person.
    remediation: str
    #: Which section it came from, when the rule knows. Becomes a block id in
    #: Fase 3, once the lesson is structured rather than a Markdown string.
    section: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Violation:
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in allowed})


@dataclass(frozen=True)
class ValidationReport:
    """Everything the rules found, and whether the lesson may proceed."""

    violations: list[Violation] = field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == BLOCKING]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == WARNING]

    @property
    def passed(self) -> bool:
        """True when nothing blocking is outstanding. Warnings do not block."""
        return not self.blocking

    def as_dicts(self) -> list[dict[str, Any]]:
        return [v.as_dict() for v in self.violations]

    def remediation_prompt(self) -> str:
        """The retry instruction: only what is blocking, and only what to do.

        Deliberately not the whole constraint list again — the Writer already
        has the classroom in its prompt. Repeating it would bury the one thing
        that has to change under the fifteen things that did not.
        """
        if not self.blocking:
            return ""

        lines = [
            "La versione precedente della lezione violava questi vincoli d'aula. "
            "Correggili, mantenendo intatto tutto il resto:"
        ]
        for violation in self.blocking:
            lines.append(f"- {violation.remediation} (nel testo: «{violation.evidence}»)")
        return "\n".join(lines)


def report_from_dicts(raw: list[dict[str, Any]] | None) -> ValidationReport:
    """Rebuild a report from what the graph state carries."""
    return ValidationReport([Violation.from_dict(item) for item in (raw or [])])
