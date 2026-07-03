"""
Critic Agent

Reviews lesson plans for quality and provides feedback.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

# Fast, cheap model for the Critic — it only outputs ~300 tokens of structured
# JSON; using the full LLM_MODEL is overkill and adds 25+ seconds of prefill.
# Defaults to TEXT2CYPHER_MODEL (already google/gemini-2.5-flash in .env.example)
# then falls back to LLM_MODEL so existing deployments stay unbroken.
_CRITIC_MODEL: Optional[str] = (
    os.getenv("AIX_CRITIC_MODEL")
    or os.getenv("TEXT2CYPHER_MODEL")
    or None  # None → build_completion_kwargs uses self.model (LLM_MODEL)
)

# Critic reads back the lesson it just received — truncate to avoid massive
# prefill cost.  Strategy: keep head (objectives + first activities) AND tail
# (differentiation + assessment) so all 5 scoring dimensions have evidence.
_CRITIC_LESSON_MAX_CHARS = int(os.getenv("AIX_CRITIC_LESSON_MAX_CHARS", "3000"))

# Retrieved context sent to the Critic — skip media lists (videos, OER, articles)
# which the Critic never uses; keep only concept names and methodology names.
_CRITIC_CONTEXT_MAX_CHARS = int(os.getenv("AIX_CRITIC_CONTEXT_MAX_CHARS", "1200"))


def _truncate_lesson_for_critic(text: str) -> str:
    """Keep head (2/3) + tail (1/3) so all lesson sections are represented."""
    if len(text) <= _CRITIC_LESSON_MAX_CHARS:
        return text
    head = (_CRITIC_LESSON_MAX_CHARS * 2) // 3
    tail = _CRITIC_LESSON_MAX_CHARS - head
    omitted = len(text) - _CRITIC_LESSON_MAX_CHARS
    return text[:head] + f"\n\n...[{omitted} chars omitted for brevity]...\n\n" + text[-tail:]


from openai import AsyncOpenAI

from aix.agent.agents.retriever_agent import RetrievalResult
from aix.agent.prompts.critic_prompt import (
    get_critic_prompts,
    is_lesson_intent,
)
from aix.core.concurrency import guarded_chat_completion
from aix.core.config import config as app_config
from aix.core.config import extract_response_content

# Optional domain extensions - fails gracefully if not available
try:
    from aix.agent.configs.domain_prompts import get_domain_extension

    DOMAIN_EXTENSIONS_AVAILABLE = True
except ImportError:
    DOMAIN_EXTENSIONS_AVAILABLE = False

    def get_domain_extension(d, a):
        return ""  # Fallback: no extension


logger = logging.getLogger(__name__)


def _extract_json(content: str) -> dict:
    """Multi-strategy JSON extractor for LLM responses that may wrap JSON in markdown."""
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in LLM response", content, 0)


@dataclass
class CritiqueResult:
    """Result of a lesson plan critique"""

    scores: dict[str, int]
    average_score: float
    decision: str  # "APPROVE" or "REVISE"
    strengths: list
    weaknesses: list
    revision_instructions: Optional[str]
    summary: str

    @property
    def approved(self) -> bool:
        return self.decision == "APPROVE"


class CriticAgent:
    """
    Critic Agent - Reviews and evaluates lesson plans.

    Responsibilities:
    1. Evaluate lesson plans on multiple criteria
    2. Verify content is grounded in retrieved evidence
    3. Decide whether to approve or request revision
    4. Provide specific feedback for improvements
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Critic Agent.

        Args:
            model: OpenAI model to use for evaluation
        """
        self.model = model
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            self._client = app_config.openai.get_async_client()
        return self._client

    async def critique(
        self,
        lesson_plan: str,
        teacher_query: str,
        retrieval_result: RetrievalResult,
        revision_count: int = 0,
        max_revisions: int = 2,
        domain: str = "neuro",
        language: str = "it",
        query_intent: str = "lesson_creation",
    ) -> CritiqueResult:
        """
        Critique content and decide whether to approve.

        Args:
            lesson_plan: The content to review (lesson plan or informational content)
            teacher_query: Original teacher request
            retrieval_result: Retrieved context for evidence checking
            revision_count: Current revision iteration
            max_revisions: Maximum allowed revisions
            domain: Knowledge domain
            language: Language of the content
            query_intent: Detected query intent for appropriate evaluation

        Returns:
            CritiqueResult with scores, decision, and feedback
        """
        content_type = (
            "lesson plan" if is_lesson_intent(query_intent) else f"{query_intent} content"
        )

        logger.info(
            f"[CriticAgent] Reviewing {content_type} (revision {revision_count}/{max_revisions})..."
        )

        # Auto-approve if max revisions reached
        if revision_count >= max_revisions:
            logger.info("[CriticAgent] Max revisions reached, auto-approving")
            return CritiqueResult(
                scores={"auto": 4},
                average_score=4.0,
                decision="APPROVE",
                strengths=["Multiple revision cycles completed"],
                weaknesses=[],
                revision_instructions=None,
                summary="Auto-approved after maximum revision cycles",
            )

        client = self._get_client()

        # Get intent-specific prompts
        system_prompt, user_template = get_critic_prompts(query_intent)

        # NEW Phase B: Add domain-specific extensions
        if DOMAIN_EXTENSIONS_AVAILABLE:
            domain_ext = get_domain_extension(domain, "critic")
            if domain_ext:
                system_prompt += domain_ext
                logger.info(f"[CriticAgent] Applied domain extension for '{domain}'")

        # Change #2 — truncate inputs to cut prefill cost.
        # The Critic reads the full lesson plan back (~4 000-8 000 chars) plus
        # the retrieved-context string (media lists, node details) — that can
        # be 6 000+ input tokens, causing 15-20s of TTFT on any model.
        # Fix: head+tail lesson slice keeps all 5 scoring dimensions visible;
        # context without media is enough for Evidence Grounding scoring.
        lesson_for_critic = _truncate_lesson_for_critic(lesson_plan)
        context_text = retrieval_result.to_context_string(include_media=False)
        if len(context_text) > _CRITIC_CONTEXT_MAX_CHARS:
            context_text = context_text[:_CRITIC_CONTEXT_MAX_CHARS] + "\n...[context truncated]"

        logger.debug(
            "[CriticAgent] input sizes: lesson=%d→%d chars, context=%d chars, model=%s",
            len(lesson_plan),
            len(lesson_for_critic),
            len(context_text),
            _CRITIC_MODEL or "LLM_MODEL",
        )

        # Format user prompt (handle both template types)
        if is_lesson_intent(query_intent):
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                lesson_plan=lesson_for_critic,
                retrieved_context=context_text,
                revision_count=revision_count,
                max_revisions=max_revisions,
                domain=domain,
                language="Italian" if language == "it" else "English",
            )
        else:
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                lesson_plan=lesson_for_critic,
                retrieved_context=context_text,
                query_intent=query_intent,
                revision_count=revision_count,
                max_revisions=max_revisions,
                domain=domain,
                language="Italian" if language == "it" else "English",
            )

        try:
            # Change #1 — fast model for the Critic.
            # The Critic outputs ~300 tokens of JSON; using LLM_MODEL (Claude Sonnet)
            # wastes 25-30s on a task a fast model handles in 2-4s.
            # model_override=_CRITIC_MODEL ensures build_completion_kwargs runs all
            # o-series / reasoning-model guards against the actual model, not LLM_MODEL.
            # CORE 2 #11a — json_mode=True to harden JSON parsing (see Planner comment).
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.3,
                max_tokens=2000,
                json_mode=True,
                model_override=_CRITIC_MODEL,
            )
            response = await guarded_chat_completion(
                client,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **completion_kwargs,
            )

            content = extract_response_content(response, logger)
            critique_data = _extract_json(content)

            result = CritiqueResult(
                scores=critique_data.get("scores", {}),
                average_score=critique_data.get("average_score", 3.0),
                decision=critique_data.get("decision", "APPROVE"),
                strengths=critique_data.get("strengths", []),
                weaknesses=critique_data.get("weaknesses", []),
                revision_instructions=critique_data.get("revision_instructions"),
                summary=critique_data.get("summary", ""),
            )

            logger.info(
                f"[CriticAgent] Decision: {result.decision} (score: {result.average_score:.1f})"
            )

            return result

        except json.JSONDecodeError as e:
            # CORE 2 #11a (2026-05-09): structured parse-error log + env-gated
            # behaviour. The legacy default is ``approve`` (today's silent
            # auto-approve with summary "Approved due to parsing error") —
            # kept as default so this change is byte-identical to pre-#11a
            # behaviour. Ops can flip the env to ``revise`` to force the
            # revision loop (exposes the failure to the writer-revise pass)
            # or ``raise`` to hard-fail the run (best for staging once the
            # parse-error rate is observed and proven low under json_mode).
            raw_preview = content[:300] if "content" in dir() else "<not set>"
            mode = (os.getenv("AIX_CRITIC_PARSE_ERROR_BEHAVIOR") or "approve").strip().lower()
            logger.error(
                "event=agent_parse_error agent=critic mode=%s err=%s raw_preview=%r",
                mode,
                e,
                raw_preview,
            )

            if mode == "raise":
                raise

            if mode == "revise":
                # Surface the failure into the revision loop — Writer will
                # see the typed marker in revision_instructions and the
                # next critique cycle gets a fresh, hopefully-parseable run.
                return CritiqueResult(
                    scores={},
                    average_score=2.0,
                    decision="REVISE",
                    strengths=[],
                    weaknesses=["Critic response was unparseable JSON (parse-error fallback)"],
                    revision_instructions=(
                        "[parse_error] The previous critic response was malformed and could "
                        "not be parsed. Please regenerate the lesson — your previous draft is "
                        "kept; this revision pass exists to recover from a transient critic "
                        "JSON error, not because of any specific issue with the lesson content."
                    ),
                    summary="[parse_error] Critic returned malformed JSON; forcing one revision pass for recovery.",
                )

            # Default: legacy "approve" path — unchanged behaviour. The only
            # observable difference from pre-#11a is the log line above.
            return CritiqueResult(
                scores={},
                average_score=3.5,
                decision="APPROVE",
                strengths=[],
                weaknesses=["Could not parse critique response"],
                revision_instructions=None,
                summary="Approved due to parsing error",
            )
        except Exception as e:
            logger.error(f"[CriticAgent] Critique failed: {e}")
            raise

    def critique_sync(
        self,
        lesson_plan: str,
        teacher_query: str,
        retrieval_result: RetrievalResult,
        revision_count: int = 0,
        max_revisions: int = 2,
        domain: str = "neuro",
        language: str = "it",
    ) -> CritiqueResult:
        """Synchronous version of critique()"""
        import asyncio

        return asyncio.run(
            self.critique(
                lesson_plan,
                teacher_query,
                retrieval_result,
                revision_count,
                max_revisions,
                domain,
                language,
            )
        )
