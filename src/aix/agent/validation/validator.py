"""Running the rules, and the two knobs that govern them."""

from __future__ import annotations

import logging
import os
from typing import Any

from aix.agent.validation.models import ValidationReport, Violation
from aix.agent.validation.rules import ALL_RULES

logger = logging.getLogger(__name__)

#: Two, like the Critic's. A lesson that still violates the classroom after two
#: targeted rewrites is not going to be fixed by a third, and the teacher is
#: better served by seeing the violations than by waiting.
DEFAULT_MAX_RETRIES = 2


def validator_enabled() -> bool:
    """Off by default — read the literal below, not the shape of the expression.

    It shipped on, and that changed the product for everyone (see the comment
    below). The flag exists so a run can be compared against the previous
    behaviour, and so it can be switched on deliberately rather than inherited.

    This docstring used to open with «On by default», contradicting both the code
    under it and ``docs/ENVIRONMENT.md``. These flag helpers are all written as
    ``(os.getenv(...) or "<default>") not in (...)``: the default is the string
    literal, and reading the ``not in`` as if it meant "on" is exactly the
    mistake that line invited.
    """
    # Default off (2026-09-16). It shipped on, and that changed the product for
    # everyone: a lesson is blocked by rules nobody asked for, and the review of
    # 2026-09-15 reproduced the worst case — a lesson that merely *mentions* DSA
    # with no profile set was refused. On by choice, not by inheritance.
    return (os.getenv("AIX_VALIDATOR_ENABLED") or "false").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def max_retries() -> int:
    try:
        return max(0, int(os.getenv("AIX_VALIDATOR_MAX_RETRIES") or DEFAULT_MAX_RETRIES))
    except ValueError:
        return DEFAULT_MAX_RETRIES


def _as_dict(profile: Any) -> dict[str, Any]:
    if profile is None:
        return {}
    if hasattr(profile, "model_dump"):
        return profile.model_dump(mode="json")
    return profile if isinstance(profile, dict) else {}


def validate_lesson(lesson: str, profile: Any) -> ValidationReport:
    """Run every rule over a lesson. Never raises.

    A rule that throws is a bug in the rule, and the right response is to lose
    that rule for this run rather than the lesson: the validator sits between a
    finished draft and the teacher, and must not be the reason a run fails.
    """
    if not lesson or not lesson.strip():
        return ValidationReport()

    profile_dict = _as_dict(profile)
    violations: list[Violation] = []

    for rule in ALL_RULES:
        try:
            violations.extend(rule(lesson, profile_dict))
        except Exception:  # noqa: BLE001 — one broken rule must not fail the run
            logger.exception("[validator] rule %s raised and was skipped", rule.__name__)

    report = ValidationReport(violations)
    if report.violations:
        logger.info(
            "[validator] %d blocking, %d warning: %s",
            len(report.blocking),
            len(report.warnings),
            ", ".join(sorted({v.rule for v in report.violations})),
        )
    return report
