#!/usr/bin/env python3
"""Measure whether a checker can tell a supported citation from an empty one.

Why this exists
---------------
The citations themselves are sound: measured on 2026-09-21, the Writer only
ever copies back a marker it was handed — it invents nothing. What it does do
is attach an authentic marker to a sentence the passage does not support. Two
real lessons cited a page of Common Core standards for claims about NATO and
about the Soviet collapse, and a page of exercise questions for a speech
printed elsewhere.

Before building a check into the pipeline, this says whether the check works.
The same discipline as ``run_private_rag.py``: the harness comes first, so a
candidate is kept or dropped on evidence rather than on how clever it sounds.

What it measures
----------------
Each case is a (claim, passage) pair with a verdict a correct checker must
reach. Two numbers matter, and they are not symmetric:

``recall on unsupported``
    of the citations that should be flagged, how many are. A miss is what the
    teacher sees.
``false alarms``
    of the citations that are fine, how many are flagged. Every one of these
    costs a revision cycle and teaches the team to ignore the checker — the
    expensive failure, not the cheap one.

Usage::

    python scripts/eval/check_citation_support.py            # the lexical baseline
    python scripts/eval/check_citation_support.py --llm      # ask the model
    python scripts/eval/check_citation_support.py -v         # one line per case
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CASES_FILE = REPO_ROOT / "data" / "eval" / "citations" / "cases.yaml"


@dataclass
class Outcome:
    case_id: str
    expected: str
    got: str
    detail: str = ""

    @property
    def correct(self) -> bool:
        return self.expected == self.got


# ---------------------------------------------------------------------------
# Candidate 1 — words in common
# ---------------------------------------------------------------------------


def lexical_checker(claim: str, passage: str) -> tuple[str, str]:
    """Supported when the claim's words appear in the passage.

    The cheapest thing that could possibly work, measured first so that the
    cost of anything heavier is justified rather than assumed. It reuses the
    tokenizer of the hybrid search, stemming and all.
    """
    from aix.webui.materials.lexical import tokenize

    claim_terms = set(tokenize(claim))
    passage_terms = set(tokenize(passage))
    if not claim_terms:
        return "unsupported", "no terms"

    overlap = len(claim_terms & passage_terms) / len(claim_terms)
    verdict = "supported" if overlap >= 0.30 else "unsupported"
    return verdict, f"overlap {overlap:.0%}"


# ---------------------------------------------------------------------------
# Candidate 2 — ask the model
# ---------------------------------------------------------------------------

async def llm_checker(claim: str, passage: str) -> tuple[str, str]:
    """The judgement the pipeline actually makes, on one pair.

    Calls ``citation_support.judge`` rather than reproducing its prompt. The
    first version of this file had its own copy, and within a day the two had
    drifted: this harness scored 9/9 on a case the pipeline was getting wrong,
    because the pipeline was asking a different question.
    """
    from aix.agent.validation.citation_support import judge
    from aix.core.config import config as app_config

    client = app_config.openai.get_async_client()
    verdicts = await judge([{"claim": claim, "passage": passage}], client)
    if not verdicts:
        return "unsupported", "the judge could not answer"
    return verdicts[0], verdicts[0]


# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    import yaml

    cases = yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))["cases"]
    outcomes: list[Outcome] = []

    for case in cases:
        claim, passage = case["claim"].strip(), case["passage"].strip()
        if args.llm:
            verdict, detail = await llm_checker(claim, passage)
        else:
            verdict, detail = lexical_checker(claim, passage)
        outcomes.append(Outcome(case["id"], case["verdict"], verdict, detail))

    name = "the model" if args.llm else "words in common"
    print(f"# Citation support — {name}, {len(outcomes)} cases\n")

    if args.verbose:
        for o in outcomes:
            mark = "ok " if o.correct else "MISS"
            print(f"  {mark}  {o.case_id:<44} atteso={o.expected:<12} detto={o.got:<12} {o.detail}")
        print()

    should_flag = [o for o in outcomes if o.expected == "unsupported"]
    caught = [o for o in should_flag if o.got == "unsupported"]
    fine = [o for o in outcomes if o.expected == "supported"]
    false_alarms = [o for o in fine if o.got == "unsupported"]

    print(f"- **unsupported caught**: {len(caught)}/{len(should_flag)}")
    print(f"- **false alarms**: {len(false_alarms)}/{len(fine)}"
          f"{' — ' + ', '.join(o.case_id for o in false_alarms) if false_alarms else ''}")
    print(f"- **correct overall**: {sum(o.correct for o in outcomes)}/{len(outcomes)}")

    if false_alarms:
        print("\nA false alarm costs a revision cycle and credibility. Read those first.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", action="store_true", help="ask the model instead of counting words")
    parser.add_argument("--verbose", "-v", action="store_true", help="one line per case")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
