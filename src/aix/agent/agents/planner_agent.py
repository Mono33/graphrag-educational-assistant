"""
Planner Agent

Analyzes teacher queries and creates structured retrieval plans.
Determines what to search in the knowledge graph.
"""

import json
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from openai import AsyncOpenAI
from aix.core.config import config as app_config, extract_response_content

from aix.agent.prompts.planner_prompt import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE

logger = logging.getLogger(__name__)


def _extract_json(content: str) -> dict:
    """Multi-strategy JSON extractor for LLM responses that may wrap JSON in markdown."""
    content = content.strip()

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


@dataclass
class RetrievalPlan:
    """Structured plan for what to retrieve"""
    query_intent: str  # lesson_creation, definition, comparison, etc.
    key_concepts: List[str]
    search_queries: List[str]
    lesson_type: Optional[str] = None  # Only for lesson/activity intents
    target_grade: Optional[str] = None
    special_needs: Optional[List[str]] = None
    time_constraints: Optional[str] = None
    intent_confidence: str = "MEDIUM"
    reasoning: Optional[str] = None
    
    # NEW Phase A: Scope detection fields
    scope_status: str = "in_scope"  # in_scope, partial_scope, out_of_scope
    scope_confidence: float = 1.0   # 0.0-1.0 confidence in scope detection
    subject_concepts: Optional[List[str]] = None  # Subject-specific (may need external APIs)
    pedagogy_concepts: Optional[List[str]] = None  # Teaching strategies from KG

    # CORE 2 #10 follow-up — Point (a): LLM-driven response-language detection.
    # The Planner is the canonical L1 detector because it sees the full query
    # and can distinguish sentence structure from isolated terminology. The
    # service layer pre-seeds state["language"] with a statistical L2/L3
    # guess before this node runs, then plan_node OVERRIDES that seed with
    # ``response_language`` when ``language_confidence`` >= MEDIUM. This keeps
    # the writer/critic on the user's actual language even on follow-up
    # queries that don't trigger the brittle stop-word heuristic.
    response_language: Optional[str] = None  # ISO 2-letter ("it", "en", "es", "fr"); None = let caller fall back
    language_confidence: str = "LOW"          # "HIGH" | "MEDIUM" | "LOW"
    
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
        return (
            self.response_language is not None
            and self.language_confidence in ("HIGH", "MEDIUM")
        )


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
        language: str = "it"
    ) -> RetrievalPlan:
        """
        Analyze a teacher query and create a retrieval plan.
        
        Args:
            query: Teacher's natural language query
            domain: Knowledge domain ("neuro" or "udl")
            language: Response language ("it" or "en")
            
        Returns:
            RetrievalPlan with structured search parameters
        """
        logger.info(f"[PlannerAgent] Analyzing query: {query[:50]}...")
        
        client = self._get_client()
        
        # Format the user prompt
        user_prompt = PLANNER_USER_TEMPLATE.format(
            query=query,
            domain=domain,
            language=language
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
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.3,
                max_tokens=2000,
                json_mode=True,
            )
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                **completion_kwargs
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
            language_confidence = (
                plan_data.get("language_confidence") or "LOW"
            ).upper()
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
            )
            
            # Enhanced logging with scope status
            scope_emoji = {"in_scope": "✅", "partial_scope": "⚠️", "out_of_scope": "❌"}.get(scope_status, "❓")
            lang_part = (
                f"lang={plan.response_language}({language_confidence}), "
                if plan.response_language else "lang=<none>, "
            )
            logger.info(
                f"[PlannerAgent] Created plan: {lang_part}intent={plan.query_intent}, "
                f"scope={scope_emoji} {scope_status} ({scope_confidence:.0%}), "
                f"{len(plan.search_queries)} queries, "
                f"concepts: {plan.key_concepts[:3]}..."
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
                e, raw_preview,
            )
            return RetrievalPlan(
                query_intent="lesson_creation",
                key_concepts=[],
                search_queries=[query],
                lesson_type="full_lesson",
                intent_confidence="LOW",
                reasoning="Fallback plan due to JSON parsing error"
            )
        except Exception as e:
            logger.error(f"[PlannerAgent] Planning failed: {e}")
            raise
    
    def plan_sync(
        self,
        query: str,
        domain: str = "neuro",
        language: str = "it"
    ) -> RetrievalPlan:
        """Synchronous version of plan()"""
        import asyncio
        return asyncio.run(self.plan(query, domain, language))

