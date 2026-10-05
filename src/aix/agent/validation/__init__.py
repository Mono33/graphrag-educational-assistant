"""Deterministic feasibility checks on a generated lesson (Q1 · 1.3).

The Critic is an LLM with a 1–5 rubric, and it judges pedagogical quality and
practical feasibility with the same subjective instrument. That is why a lesson
asking twenty-five device-less students to use Kahoot on their phones came back
at 91%: nothing in the pipeline could *count*.

This package counts. Pure Python, no model call — instant, free, and testable
offline. It answers only objective questions: do the minutes add up, is a tool
being used that this classroom cannot run, is there a BES acronym in the text
that the teacher never declared.

The division of labour is the point. The Critic keeps judging whether a lesson
is *good*, which is a matter of opinion and belongs to a model. The validator
decides whether it is *possible*, which is a matter of fact and must not depend
on a model having a good day.
"""

from aix.agent.validation.models import (
    BLOCKING,
    WARNING,
    ValidationReport,
    Violation,
)
from aix.agent.validation.validator import (
    max_retries,
    validate_lesson,
    validator_enabled,
)

__all__ = [
    "BLOCKING",
    "WARNING",
    "ValidationReport",
    "Violation",
    "max_retries",
    "validate_lesson",
    "validator_enabled",
]
