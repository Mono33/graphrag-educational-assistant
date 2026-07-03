"""
Phase 2 educational prompts — reusable LLM templates exposed via MCP.

These prompts let any MCP client (Claude Desktop, Cursor IDE, Lovable apps,
external partners on Streamable HTTP in Phase 5) render two standardised
educational workflows without re-implementing Aix-specific phrasing:

* ``educational-query``      — turns a teacher's free-form topic + optional
                                student profile into a well-formed query
                                ready for ``kg.search`` / ``kg.get_context``.
* ``lesson-plan-request``    — assembles a comprehensive, methodology-aware
                                lesson-plan brief that the agent pipeline
                                (Planner → Retriever → Writer → Critic) can
                                consume directly.

Why prompts (not tools)?
------------------------
Tools take action; prompts shape *thought*. A prompt template lives entirely
on the server side, but the LLM calls happen on the client — so the client
keeps full control over model choice, temperature, and follow-up turns.
This is exactly the layer Anthropic recommends for "reusable workflows":
the server says *"here's how I think you should ask"*, the client decides
*"and here's the model that will answer"*.

Note on type hints
------------------
We deliberately *do not* use ``from __future__ import annotations`` in this
module. FastMCP 3.x (Pydantic 2.11) auto-generates the prompt argument JSON
schema at registration time by introspecting the function signature; under
PEP 563 (``__future__.annotations``) those hints are strings, and Pydantic
fails to resolve ``Optional[str]`` from a string in the prompt-argument
context — leading to DEBUG warnings and missing args in client pickers
(Claude Desktop, Cursor IDE). Eager-evaluated annotations sidestep that.

MCP spec — every prompt argument is a *string*
----------------------------------------------
The Model Context Protocol mandates that ``GetPromptRequestParams.arguments``
is ``dict[str, str]``: clients send every argument as a string regardless of
its semantic type. FastMCP enforces that at the protocol boundary, so a
prompt function MUST declare every arg as ``str`` (or ``Optional[str]``).
Numeric coercion happens *inside* the function — see
``lesson_plan_request`` for the canonical pattern.
"""

import logging
from typing import Optional

from fastmcp import FastMCP
from fastmcp.prompts.prompt import Message

logger = logging.getLogger(__name__)


_VALID_DOMAINS = ("neuro", "udl")


def _normalize_domain(domain: Optional[str]) -> str:
    """Coerce an optional domain string to a known value, defaulting to neuro."""
    if not domain:
        return "neuro"
    candidate = domain.strip().lower()
    return candidate if candidate in _VALID_DOMAINS else "neuro"


def register(mcp: FastMCP) -> None:
    """Register both Phase 2 prompts onto the shared FastMCP instance."""

    # ---- educational-query -----------------------------------------------
    @mcp.prompt(
        name="educational-query",
        description=(
            "Render a well-formed educational query targeting the Aix "
            "Knowledge Graph. Takes a teacher's free-form topic plus an "
            "optional student profile and produces a query string the LLM "
            "should hand to kg.search or kg.get_context. Italian-first by "
            "default; pass a profile for differentiated instruction."
        ),
        tags={"education", "query", "graphrag"},
    )
    def educational_query(
        topic: str,
        student_profile: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[Message]:
        """Educational-query prompt template.

        Args:
            topic: Teacher's topic or open question (e.g. "fotosintesi",
                "strategie per studenti con dislessia").
            student_profile: Optional one-line profile of the target
                students (age range, special-needs cues, language level, etc).
            domain: Knowledge domain — 'neuro' (neuroscience methodologies)
                or 'udl' (Universal Design for Learning). Defaults to 'neuro'.
        """
        domain_norm = _normalize_domain(domain)
        topic_clean = (topic or "").strip()

        domain_blurb = (
            "neuroscience-grounded methodologies (executive function, "
            "motivation, working memory, etc.)."
            if domain_norm == "neuro"
            else "Universal Design for Learning (UDL principles, "
            "accessibility, multiple means of representation)."
        )

        # MCP prompt messages may only use roles 'user' | 'assistant'.
        # We keep the framing concise and inline it at the top of the user
        # message so the consuming model treats it as task instructions.
        body = (
            "## Context\n"
            "You are an expert educational assistant working with the Aix "
            "Knowledge Graph. Your job is to translate the teacher's request "
            f"into a precise, retrieval-friendly query in domain '{domain_norm}': "
            f"{domain_blurb}\n"
            "Reply in Italian unless the teacher's input is in another language. "
            "Keep the final query under 200 characters and optimised for "
            "semantic search (no fluff, no quotation marks).\n\n"
            "## Teacher input\n"
            f"- Topic: {topic_clean or '(empty)'}"
        )
        if student_profile and student_profile.strip():
            body += f"\n- Student profile: {student_profile.strip()}"
        body += (
            "\n\n## Task\n"
            "Produce a single retrieval query I can pass to kg.search or "
            "kg.get_context, then briefly explain (1-2 lines) why this "
            "phrasing should retrieve the most relevant nodes."
        )

        return [Message(content=body, role="user")]

    # ---- lesson-plan-request ---------------------------------------------
    @mcp.prompt(
        name="lesson-plan-request",
        description=(
            "Render a comprehensive lesson-plan brief that the Aix agent "
            "pipeline (Planner → Retriever → Writer → Critic) can consume. "
            "Takes a topic, an optional duration, an optional methodology "
            "and an optional student level. Produces an Italian-first brief "
            "with clear learning objectives, methodology constraints and "
            "assessment expectations."
        ),
        tags={"education", "lesson-plan", "agent"},
    )
    def lesson_plan_request(
        topic: str,
        duration_minutes: str = "60",
        methodology: Optional[str] = None,
        level: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[Message]:
        """Lesson-plan-request prompt template.

        Args:
            topic: Subject of the lesson (e.g. "la respirazione cellulare").
            duration_minutes: Target lesson duration in minutes as a string
                (15..240). String per MCP spec — coerced to int internally
                with safe clamping. Defaults to '60'.
            methodology: Optional methodology hint (e.g. "UDL",
                "active learning", "spaced retrieval"). Free-text.
            level: Optional educational level cue (e.g. "scuola primaria",
                "secondaria di primo grado", "B1 italiano L2").
            domain: 'neuro' or 'udl' (default 'neuro').
        """
        domain_norm = _normalize_domain(domain)
        topic_clean = (topic or "").strip()
        try:
            duration_int = int(str(duration_minutes).strip() or "60")
        except (TypeError, ValueError):
            duration_int = 60
        duration_safe = max(15, min(240, duration_int))

        body_lines: list[str] = [
            "## Ruolo",
            "Sei l'assistente di Briefing del Lesson Planner Aix. NON "
            "scrivi la lezione — produci un brief strutturato che la "
            "pipeline agentica Aix (Planner, Retriever, Writer, Critic) "
            "consumerà come input.",
            f"Dominio: '{domain_norm}'. Lingua di output: italiano. Usa "
            "intestazioni di sezione chiare (## Obiettivi, ## Vincoli, "
            "## Metodologie, ## Valutazione). Sii concreto e conciso.",
            "",
            "## Input docente",
            f"- Argomento: {topic_clean or '(non specificato)'}",
            f"- Durata target: {duration_safe} minuti",
        ]
        if level and level.strip():
            body_lines.append(f"- Livello / target: {level.strip()}")
        if methodology and methodology.strip():
            body_lines.append(f"- Metodologia preferita: {methodology.strip()}")
        body_lines += [
            "",
            "## Compito",
            "Produci un brief strutturato con:",
            "1. **Obiettivi di apprendimento** misurabili (3-5 bullet).",
            "2. **Vincoli e prerequisiti** dello studente.",
            "3. **Metodologie suggerite** (se sai del KG, cita 2-3 "
            "metodologie applicabili — altrimenti chiedi al team Retriever "
            "via kg.search).",
            "4. **Criteri di valutazione formativa** (almeno 2).",
        ]

        return [Message(content="\n".join(body_lines), role="user")]

    _ = (educational_query, lesson_plan_request)
