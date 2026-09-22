"""
Planner Agent

Analyzes teacher queries and creates structured retrieval plans.
Determines what to search in the knowledge graph.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from openai import AsyncOpenAI

from aix.agent.prompts.planner_prompt import (
    PLANNER_FOLLOWUP_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
)
from aix.agent.prompts.profile_renderer import render_classroom_constraints
from aix.core.concurrency import guarded_chat_completion
from aix.core.config import config as app_config
from aix.core.config import extract_response_content

logger = logging.getLogger(__name__)


def _planner_max_tokens() -> int:
    """Output budget for the plan. See the comment at the call site."""
    raw = os.getenv("AIX_PLANNER_MAX_TOKENS")
    try:
        return max(1000, int(raw)) if raw else 4000
    except ValueError:
        return 4000


def _extract_json(content: str) -> dict:
    """Multi-strategy JSON extractor for LLM responses that may wrap JSON in markdown.

    A truncated answer is named as such. «No valid JSON found» sent the next
    reader hunting for markdown fences when the real cause was a plan cut off
    mid-array by the token ceiling (2026-09-21) — the fences were being handled
    correctly all along.
    """
    content = content.strip()

    if content and content.count("{") > content.count("}"):
        raise json.JSONDecodeError(
            "the plan is truncated — more '{' than '}', so the model ran out of "
            "output budget (AIX_PLANNER_MAX_TOKENS)",
            content,
            0,
        )

    # Strategy 1: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip ```json ... ``` fences
    stripped = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: extract first complete {...} block
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in LLM response", content, 0)


# ---------------------------------------------------------------------------
# T6 — formatting the thread context the planner is given on a follow-up turn
# ---------------------------------------------------------------------------

_HISTORY_TURNS = 6
_HISTORY_CHARS = 400
_CORPUS_NAMES = 40


def _format_history(history: Optional[list[dict[str, str]]]) -> str:
    """The last few turns, one per line, each cut to a readable length.

    Only the tail matters for this decision — whether *this* turn needs new
    evidence — and the whole thread would crowd out the query itself in a
    prompt the planner reads top to bottom.
    """
    turns = [t for t in (history or []) if isinstance(t, dict) and t.get("content")]
    if not turns:
        return "(none)"

    lines = []
    for turn in turns[-_HISTORY_TURNS:]:
        content = str(turn["content"]).strip().replace("\n", " ")
        if len(content) > _HISTORY_CHARS:
            content = content[:_HISTORY_CHARS] + "..."
        lines.append(f"- {turn.get('role', 'user')}: {content}")
    return "\n".join(lines)


def _format_corpus(names: Optional[list[str]]) -> str:
    """The names already retrieved, comma separated and capped.

    Capped because a wide first turn can bring back dozens of nodes, and past
    a point the list stops informing the verdict and starts diluting it.
    """
    clean = [str(n).strip() for n in (names or []) if n and str(n).strip()]
    if not clean:
        return "(nothing retrieved yet)"

    shown = clean[:_CORPUS_NAMES]
    suffix = f", and {len(clean) - len(shown)} more" if len(clean) > len(shown) else ""
    return ", ".join(shown) + suffix


def _as_bool(value: Any) -> Optional[bool]:
    """Read the planner's verdict without trusting its JSON typing.

    Returns None for anything unrecognised, which is the same signal as an
    absent key: fall back to comparing concepts rather than guess.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0"):
            return False
    return None


@dataclass
class RetrievalPlan:
    """Structured plan for what to retrieve"""

    query_intent: str  # lesson_creation, definition, comparison, etc.
    key_concepts: list[str]
    search_queries: list[str]
    lesson_type: Optional[str] = None  # Only for lesson/activity intents
    target_grade: Optional[str] = None
    special_needs: Optional[list[str]] = None
    time_constraints: Optional[str] = None
    intent_confidence: str = "MEDIUM"
    reasoning: Optional[str] = None

    # NEW Phase A: Scope detection fields
    scope_status: str = "in_scope"  # in_scope, partial_scope, out_of_scope
    scope_confidence: float = 1.0  # 0.0-1.0 confidence in scope detection
    subject_concepts: Optional[list[str]] = None  # Subject-specific (may need external APIs)
    pedagogy_concepts: Optional[list[str]] = None  # Teaching strategies from KG

    # CORE 2 #10 follow-up — Point (a): LLM-driven response-language detection.
    # The Planner is the canonical L1 detector because it sees the full query
    # and can distinguish sentence structure from isolated terminology. The
    # service layer pre-seeds state["language"] with a statistical L2/L3
    # guess before this node runs, then plan_node OVERRIDES that seed with
    # ``response_language`` when ``language_confidence`` >= MEDIUM. This keeps
    # the writer/critic on the user's actual language even on follow-up
    # queries that don't trigger the brittle stop-word heuristic.
    response_language: Optional[str] = (
        None  # ISO 2-letter ("it", "en", "es", "fr"); None = let caller fall back
    )
    language_confidence: str = "LOW"  # "HIGH" | "MEDIUM" | "LOW"

    # T6 — conditional retrieval. Set only when the caller put the thread's
    # context in front of the planner (see ``PLANNER_FOLLOWUP_TEMPLATE``); a
    # first turn, or a planner that ignored the two keys, leaves this None and
    # the router falls back to comparing concepts.
    requires_new_retrieval: Optional[bool] = None
    retrieval_reason: Optional[str] = None

    @property
    def is_lesson_intent(self) -> bool:
        """Check if this is a lesson/activity creation intent"""
        return self.query_intent in ("lesson_creation", "activity_design")

    @property
    def needs_external_apis(self) -> bool:
        """Check if this plan requires external API calls for subject content"""
        return self.scope_status in ("partial_scope", "out_of_scope")

    @property
    def is_in_scope(self) -> bool:
        """Check if query is fully within Knowledge Graph scope"""
        return self.scope_status == "in_scope"

    @property
    def has_confident_language(self) -> bool:
        """
        Whether the LLM-detected ``response_language`` is trustworthy enough
        to override the seed language passed in by the service layer.

        We accept HIGH and MEDIUM (the Planner sees the entire query AND any
        augmented multi-turn context, so MEDIUM is already more reliable than
        a stop-word heuristic). LOW means the LLM itself wasn't sure, so we
        keep the seed instead of risking a wrong override.
        """
        return self.response_language is not None and self.language_confidence in ("HIGH", "MEDIUM")


#: How much of each attachment the Planner sees. It chooses concepts and search
#: queries, so the opening of a document is enough; the Writer gets the rest.
MATERIAL_DIGEST_PER_FILE = 1500
MATERIAL_DIGEST_TOTAL = 4500


def material_digest(teacher_materials: Optional[str]) -> Optional[str]:
    """One line per attachment: its name and the start of its text.

    Reads the ``### filename`` sections that ``_teacher_upload_context`` builds;
    plain text from an API caller becomes a single section called "Materiale".
    """
    import re

    if not teacher_materials or not teacher_materials.strip():
        return None

    text = teacher_materials.strip()
    if not text.startswith("### "):
        text = "### Materiale\n" + text

    lines = []
    for section in re.split(r"(?m)^### ", text):
        if not section.strip():
            continue
        title, _, body = section.partition("\n")
        body = " ".join(body.split())
        if not body:
            continue
        if len(body) > MATERIAL_DIGEST_PER_FILE:
            body = body[:MATERIAL_DIGEST_PER_FILE].rstrip() + "…"
        lines.append(f"- {title.strip()}: {body}")

    digest = "\n".join(lines)
    return digest[:MATERIAL_DIGEST_TOTAL] if digest else None


class PlannerAgent:
    """
    Planner Agent - First step in the lesson planning pipeline.

    Responsibilities:
    1. Analyze the teacher's natural language query
    2. Identify key educational concepts to search
    3. Create a structured retrieval plan
    4. Determine lesson type, grade level, and constraints
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Planner Agent.

        Args:
            model: OpenAI model to use for planning
        """
        self.model = model
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            self._client = app_config.openai.get_async_client()
        return self._client

    async def plan(
        self,
        query: str,
        domain: str = "neuro",
        language: str = "it",
        pedagogical_intent: Optional[str] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
        existing_corpus: Optional[list[str]] = None,
        educational_profile: Optional[dict[str, Any]] = None,
        teacher_materials: Optional[str] = None,
    ) -> RetrievalPlan:
        """
        Analyze a teacher query and create a retrieval plan.

        Args:
            query: Teacher's natural language query
            domain: Knowledge domain ("neuro" or "udl")
            language: Response language ("it" or "en")
            educational_profile: The class and classroom. Only its constraints
                are used here — devices, LIM, WiFi, furniture, class size — so
                the concepts and queries chosen are ones the room can act on.
                Without a profile the prompt is unchanged.
            conversation_history: Earlier turns of the thread. Passing it marks
                this as a follow-up and asks the planner for a
                ``requires_new_retrieval`` verdict; omitting it keeps the
                prompt byte-identical to a first turn.
            existing_corpus: Names of what the thread already retrieved, which
                is what the verdict is judged against.
            teacher_materials: The teacher's attachments, as the Writer receives
                them (``### filename`` sections). The Planner gets a short digest
                of each, so it plans on the document instead of declaring it
                unavailable and searching the web for "documento allegato".
                Without it the prompt is unchanged.

        Returns:
            RetrievalPlan with structured search parameters
        """
        logger.info(f"[PlannerAgent] Analyzing query: {query[:50]}...")

        client = self._get_client()

        # Format the user prompt
        user_prompt = PLANNER_USER_TEMPLATE.format(query=query, domain=domain, language=language)

        # F4: Inject pedagogical_intent so the planner generates
        # objective-aligned search queries and explicit learning objectives.
        if pedagogical_intent:
            from aix.api.schemas.educational_profile import PEDAGOGICAL_INTENT_BY_CODE

            code, _, detail = pedagogical_intent.partition(": ")
            option = PEDAGOGICAL_INTENT_BY_CODE.get(code.strip())
            if option:
                resolved = option["prompt"]
                if detail.strip():
                    resolved += f" ({detail.strip()})"
            else:
                resolved = pedagogical_intent
            user_prompt = (
                f"Teacher's pedagogical intent: {resolved}\n"
                f"Generate lesson objectives that fulfill this intent before mapping "
                f"concepts to the knowledge graph.\n\n"
            ) + user_prompt

        # Q1 · 1.1: the classroom, as constraints. A Planner that does not know
        # the students have no devices retrieves evidence for device-based
        # activities, and the Writer then has nothing better to write from.
        constraints = render_classroom_constraints(educational_profile)
        if constraints:
            user_prompt = (
                "Classroom constraints (choose concepts and search queries the "
                "classroom can actually act on):\n"
                + "\n".join(f"- {line}" for line in constraints)
                + "\n\n"
                + user_prompt
            )

        # The teacher's attachments. Before this the Planner never saw them —
        # only the Writer did — and on "prepara una lezione a partire dal
        # documento allegato" it reported the document as unavailable and sent
        # that phrase to Wikipedia, YouTube and OpenAlex as a search query.
        digest = material_digest(teacher_materials)
        if digest:
            user_prompt = (
                "Teacher-attached material. The lesson must be built on this content: "
                "take the key concepts and search queries from it. The material IS "
                "available - never treat it as missing, and never use the attachment "
                "itself (e.g. 'documento allegato') as a concept or a search query.\n"
                + digest
                + "\n\n"
                + user_prompt
            )

        # T6: give the planner the thread it is answering inside. Prepended
        # last so it reads first — whether this is a follow-up changes how
        # everything below it should be read.
        is_followup = bool(conversation_history)
        if is_followup:
            user_prompt = (
                PLANNER_FOLLOWUP_TEMPLATE.format(
                    history=_format_history(conversation_history),
                    corpus=_format_corpus(existing_corpus),
                )
                + user_prompt
            )

        try:
            # CORE 2 #11a — JSON parse hardening (2026-05-09):
            # Pass json_mode=True so OpenRouter forwards
            # ``response_format={"type": "json_object"}`` to providers that
            # support it (OpenAI, most OpenRouter routes). build_completion_kwargs
            # automatically skips it for reasoning models (o1/o3/o4/DeepSeek-R1
            # / Claude-thinking) where the API rejects the parameter, so this
            # is byte-identical for those families and a hardening boost for
            # everything else. The PLANNER_SYSTEM_PROMPT already asks for
            # strict JSON so this is purely additive — no prompt change.
            # 4 000, not 2 000. A reasoning model spends part of the budget
            # thinking before it writes a character, so a ceiling that fits the
            # JSON exactly does not fit it at all: on 2026-09-21 the plan came
            # back cut off mid-array, the parse failed, and the run fell through
            # to defaults — intent LOW, no subject concepts, and a lesson whose
            # Planner card showed nothing from the teacher's documents. Same
            # trap as ``AIX_WRITER_MAX_TOKENS`` on 2026-09-16, one agent over.
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.3,
                max_tokens=_planner_max_tokens(),
                json_mode=True,
            )
            response = await guarded_chat_completion(
                client,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **completion_kwargs,
            )

            # Parse JSON response (extract_response_content also logs thinking tokens)
            content = extract_response_content(response, logger)
            plan_data = _extract_json(content)

            # Extract query intent (with fallback for backward compatibility)
            query_intent = plan_data.get("query_intent", "lesson_creation")

            # NEW Phase A: Extract scope detection fields
            scope_status = plan_data.get("scope_status", "in_scope")
            scope_confidence = plan_data.get("scope_confidence", 1.0)
            subject_concepts = plan_data.get("subject_concepts")
            pedagogy_concepts = plan_data.get("pedagogy_concepts")

            # Point (a): Extract LLM-detected response language.
            # ``response_language`` may be None on legacy / older Langfuse
            # prompt versions or when the LLM omits the field — in that case
            # we leave it as None so the caller falls back to the seed
            # language without a noisy warning.
            response_language_raw = plan_data.get("response_language")
            response_language: Optional[str] = None
            if isinstance(response_language_raw, str):
                code = response_language_raw.strip().lower()
                # Normalize common variants (en-US → en, IT → it, etc.)
                if "-" in code:
                    code = code.split("-", 1)[0]
                # Whitelist: only the languages we actually support across
                # writer/critic prompts. Anything outside silently falls
                # back to None (= keep the seed) rather than degrade output.
                if code in {"it", "en", "es", "fr"}:
                    response_language = code
                elif code:  # non-empty but unsupported
                    logger.info(
                        "[PlannerAgent] LLM emitted unsupported response_language=%r — "
                        "ignoring and keeping seed language",
                        response_language_raw,
                    )
            language_confidence = (plan_data.get("language_confidence") or "LOW").upper()
            if language_confidence not in {"HIGH", "MEDIUM", "LOW"}:
                language_confidence = "LOW"

            plan = RetrievalPlan(
                query_intent=query_intent,
                key_concepts=plan_data.get("key_concepts", []),
                search_queries=plan_data.get("search_queries", [query]),
                lesson_type=plan_data.get("lesson_type"),  # Only set for lesson intents
                target_grade=plan_data.get("target_grade"),
                special_needs=plan_data.get("special_needs"),
                time_constraints=plan_data.get("time_constraints"),
                intent_confidence=plan_data.get("intent_confidence", "MEDIUM"),
                reasoning=plan_data.get("reasoning"),
                # NEW Phase A: Scope detection
                scope_status=scope_status,
                scope_confidence=scope_confidence,
                subject_concepts=subject_concepts,
                pedagogy_concepts=pedagogy_concepts,
                # Point (a): Response language detection
                response_language=response_language,
                language_confidence=language_confidence,
                # Only trusted when the question was actually asked. Reading it
                # off a first-turn response would be reading a hallucination.
                requires_new_retrieval=(
                    _as_bool(plan_data.get("requires_new_retrieval")) if is_followup else None
                ),
                retrieval_reason=plan_data.get("retrieval_reason") if is_followup else None,
            )

            # Enhanced logging with scope status
            scope_emoji = {"in_scope": "✅", "partial_scope": "⚠️", "out_of_scope": "❌"}.get(
                scope_status, "❓"
            )
            lang_part = (
                f"lang={plan.response_language}({language_confidence}), "
                if plan.response_language
                else "lang=<none>, "
            )
            logger.info(
                f"[PlannerAgent] Created plan: {lang_part}intent={plan.query_intent}, "
                f"scope={scope_emoji} {scope_status} ({scope_confidence:.0%}), "
                f"{len(plan.search_queries)} queries, "
                f"concepts: {plan.key_concepts[:3]}..."
            )

            if plan.requires_new_retrieval is not None:
                logger.info(
                    "[PlannerAgent] follow-up turn: requires_new_retrieval=%s (%s)",
                    plan.requires_new_retrieval,
                    plan.retrieval_reason or "no reason given",
                )

            if plan.needs_external_apis:
                logger.info(f"[PlannerAgent] 🌐 External APIs needed for: {subject_concepts}")

            return plan

        except json.JSONDecodeError as e:
            # CORE 2 #11a (2026-05-09): structured parse-error log.
            # The marker `event=agent_parse_error agent=planner` is what
            # Langfuse / log-aggregation filters key on so a fallback run
            # can be distinguished from a real one in dashboards (today the
            # fallback was indistinguishable, hence the silent-failure mode
            # that the doc calls out). The fallback *shape* below is
            # intentionally identical to pre-#11a so behaviour is byte-
            # compatible — only the observability fingerprint changes.
            raw_preview = content[:300] if "content" in dir() else "<not set>"
            logger.error(
                "event=agent_parse_error agent=planner err=%s raw_preview=%r",
                e,
                raw_preview,
            )
            return RetrievalPlan(
                query_intent="lesson_creation",
                key_concepts=[],
                search_queries=[query],
                lesson_type="full_lesson",
                intent_confidence="LOW",
                reasoning="Fallback plan due to JSON parsing error",
            )
        except Exception as e:
            logger.error(f"[PlannerAgent] Planning failed: {e}")
            raise

    def plan_sync(self, query: str, domain: str = "neuro", language: str = "it") -> RetrievalPlan:
        """Synchronous version of plan()"""
        import asyncio

        return asyncio.run(self.plan(query, domain, language))
