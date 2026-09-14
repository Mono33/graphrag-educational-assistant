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
    """On by default: this is a correction, not a feature.

    The flag exists so a run can be compared against the previous behaviour, and
    so it can be switched off if a rule turns out to block good lessons.
    """
    return (os.getenv("AIX_VALIDATOR_ENABLED") or "true").strip().lower() not in (
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
