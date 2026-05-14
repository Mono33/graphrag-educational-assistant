"""
Retrieval Grader Agent  —  CORE 2 #9 (Corrective RAG)
======================================================

Why this agent exists
---------------------

Pre-#9, the pipeline trusted the Retriever blindly: whatever GraphRAG
returned was passed into the Writer prompt verbatim. That meant a noisy
query (typos, wrong domain, ambiguous phrasing) would still feed the
Writer, which then dressed up irrelevant content in a plausible-looking
lesson — the silent-failure mode the ClickUp doc flags as
"#9 Retrieval Grading".

This agent grades retrieval quality on two axes:

* **relevance** — do the retrieved concepts actually match the user
  intent + key concepts?
* **rewrite suggestion** — if not, what should we re-search?

Output shape (strict JSON, ``response_format={"type": "json_object"}``):

.. code-block:: json

    {
      "grade": "relevant" | "ambiguous" | "irrelevant",
      "reason": "<1-2 sentence rationale, in English>",
      "rewritten_query": "<better query>" | null
    }

The agent is cheap by design — small token budget (~150 in, ~120 out),
``temperature=0.0`` for stability, and it sees only the *titles* of the
top-N retrieved concepts (NOT the full node payload) so the prompt stays
tiny. Single LLM call per grading pass.

Backward compatibility
----------------------

* The agent is only invoked from the new ``grade_retrieval_node`` which
  itself is only added to the graph topology when
  ``AIX_CORRECTIVE_RAG_ENABLED=true``. With the flag off (the default),
  this module is imported but never called. **Zero** behavioural change.
* Honors :func:`aix.core.config.AIxConfig.openai.build_completion_kwargs`
  so reasoning models (o1/o3/o4/DeepSeek-R1/Claude-thinking) silently
  drop the ``response_format`` kwarg the way Planner/Critic already do.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from aix.core.config import config as app_config, extract_response_content

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts (kept inline; small enough they don't deserve a separate module)
# ---------------------------------------------------------------------------

GRADER_SYSTEM_PROMPT = """\
You are a retrieval-quality grader for an educational Knowledge Graph
about pedagogy and learning sciences. Given a teacher's query and the
list of concept titles a Knowledge Graph retriever just returned, decide
whether those concepts are good enough context to write a lesson on the
query topic.

Output STRICT JSON with three fields:

- "grade": one of "relevant", "ambiguous", "irrelevant"
    * "relevant"   — the retrieved concepts clearly cover the query intent
                     and key concepts. Writer can proceed.
    * "ambiguous"  — partial overlap; some concepts match but core entity
                     of the query is missing or only tangentially related.
    * "irrelevant" — the retrieved concepts do not address the query.

- "reason": a 1-2 sentence English rationale explaining the grade.

- "rewritten_query": when grade is "ambiguous" or "irrelevant", a single
    short improved search query (in the same language as the original
    query) that is more likely to surface the correct concepts. When
    grade is "relevant", set this to null.

Be strict. If in doubt between "relevant" and "ambiguous", choose
"ambiguous". If the retriever returned 0 concepts, the grade is
"irrelevant".
"""


GRADER_USER_TEMPLATE = """\
TEACHER QUERY:
{query}

PLANNER KEY CONCEPTS:
{key_concepts}

PLANNER SEARCH QUERIES:
{search_queries}

RETRIEVED CONCEPT TITLES (top {n_titles}):
{retrieved_titles}

RETRIEVED METHODOLOGIES (top {n_recs}):
{recommendations}

Output ONLY the JSON object, no prose, no code fences.
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

VALID_GRADES = ("relevant", "ambiguous", "irrelevant")


@dataclass
class GraderResult:
    """Structured output of one grading pass."""
    grade: str  # one of VALID_GRADES
    reason: str
    rewritten_query: Optional[str] = None

    @property
    def is_relevant(self) -> bool:
        return self.grade == "relevant"

    @property
    def needs_retry(self) -> bool:
        """Whether the grade indicates a re-retrieval would be valuable."""
        return self.grade in ("ambiguous", "irrelevant")


# ---------------------------------------------------------------------------
# JSON extractor (same multi-strategy as planner/critic — keep them aligned)
# ---------------------------------------------------------------------------

def _extract_json(content: str) -> dict:
    """Multi-strategy JSON extractor for LLM responses that may wrap JSON
    in markdown or add stray prose. Mirrors the helper in planner_agent /
    critic_agent so all three agents have identical fallback semantics."""
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
            return json.loads(content[start: end + 1])
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in grader response", content, 0)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RetrievalGraderAgent:
    """
    Cheap LLM-based grader for the corrective-RAG loop.

    Usage::

        grader = RetrievalGraderAgent()
        result = await grader.grade(
            query="...",
            key_concepts=[...],
            search_queries=[...],
            retrieved_nodes=[...],
            recommendations=[...],
        )
        if result.needs_retry:
            new_q = result.rewritten_query
            ...
    """

    # Soft cap on how many titles we send to the grader prompt. Larger
    # values waste tokens with no accuracy gain on this task.
    MAX_RETRIEVED_TITLES: int = 12
    MAX_RECOMMENDATIONS: int = 6

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the grader.

        ``gpt-4o-mini`` (or any small model) is the right default — this
        is a binary-ish classifier, not an authoring task. Production
        deployments using OpenRouter can override via the standard
        :class:`aix.core.config.AIxConfig` model env var; we keep the
        per-agent default low-cost regardless.
        """
        self.model = model
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = app_config.openai.get_async_client()
        return self._client

    @staticmethod
    def _node_titles(nodes: List[Dict[str, Any]], cap: int) -> List[str]:
        """Best-effort title extraction. Different retriever code paths
        populate ``title`` / ``name`` / ``label`` / ``id``; we walk all of
        them, dedup case-insensitively, and stop at ``cap`` entries."""
        titles: List[str] = []
        seen: set[str] = set()
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            t = n.get("title") or n.get("name") or n.get("label") or n.get("id")
            if not t:
                continue
            t = str(t).strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            titles.append(t)
            if len(titles) >= cap:
                break
        return titles

    @staticmethod
    def _recommendation_names(recs: List[Dict[str, Any]], cap: int) -> List[str]:
        names: List[str] = []
        for r in (recs or [])[:cap]:
            if isinstance(r, dict):
                n = r.get("name") or r.get("title")
                if n:
                    names.append(str(n).strip())
        return names

    async def grade(
        self,
        *,
        query: str,
        key_concepts: Optional[List[str]] = None,
        search_queries: Optional[List[str]] = None,
        retrieved_nodes: Optional[List[Dict[str, Any]]] = None,
        recommendations: Optional[List[Dict[str, Any]]] = None,
    ) -> GraderResult:
        """
        Grade a single retrieval pass.

        Args:
            query: The original (or augmented) teacher query.
            key_concepts: Planner's key concepts (best-effort hint).
            search_queries: Planner's search queries actually run.
            retrieved_nodes: Top-K nodes returned by the retriever
                (only the titles are sent to the LLM).
            recommendations: Methodology recommendations from retriever.

        Returns:
            :class:`GraderResult`. On any unexpected failure (LLM error,
            JSON parse failure, etc.) the agent returns a *safe* fallback
            of ``grade="relevant"`` so the corrective-RAG loop never
            blocks the writer — the worst case under failure is identical
            to the pre-#9 (no-grading) behaviour, which keeps the change
            backward compatible.
        """
        titles = self._node_titles(retrieved_nodes or [], self.MAX_RETRIEVED_TITLES)
        rec_names = self._recommendation_names(recommendations or [], self.MAX_RECOMMENDATIONS)

        # Empty retrieval is mechanically irrelevant — we don't need an
        # LLM round-trip to know that. Saves cost on the worst case.
        if not titles and not rec_names:
            logger.info(
                "[RetrievalGrader] retrieval is empty — short-circuiting to grade=irrelevant"
            )
            return GraderResult(
                grade="irrelevant",
                reason="Retriever returned no concepts; cannot ground the lesson.",
                rewritten_query=None,
            )

        retrieved_block = "\n".join(f"- {t}" for t in titles) or "(none)"
        recs_block = "\n".join(f"- {n}" for n in rec_names) or "(none)"
        kc_block = ", ".join(key_concepts or []) or "(none)"
        sq_block = " | ".join(search_queries or []) or "(none)"

        user_prompt = GRADER_USER_TEMPLATE.format(
            query=query,
            key_concepts=kc_block,
            search_queries=sq_block,
            n_titles=len(titles),
            retrieved_titles=retrieved_block,
            n_recs=len(rec_names),
            recommendations=recs_block,
        )

        client = self._get_client()
        try:
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.0,
                max_tokens=300,
                json_mode=True,
            )
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **completion_kwargs,
            )

            content = extract_response_content(response, logger)
            data = _extract_json(content)
        except json.JSONDecodeError as e:
            # Same observability fingerprint as Planner / Critic — see
            # CORE 2 #11a. The fallback is "relevant" (no-op) so we never
            # block the writer, preserving backward-compat behaviour.
            raw_preview = content[:300] if "content" in dir() else "<not set>"
            logger.error(
                "event=agent_parse_error agent=retrieval_grader err=%s raw_preview=%r",
                e, raw_preview,
            )
            return GraderResult(
                grade="relevant",
                reason="Grader JSON parse failure — falling back to no-op (relevant).",
                rewritten_query=None,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[RetrievalGrader] LLM call failed (%s); defaulting to grade=relevant "
                "to avoid blocking the writer.",
                e,
            )
            return GraderResult(
                grade="relevant",
                reason=f"Grader unavailable ({e.__class__.__name__}); skipping correction.",
                rewritten_query=None,
            )

        grade = str(data.get("grade", "relevant")).strip().lower()
        if grade not in VALID_GRADES:
            logger.warning(
                "[RetrievalGrader] LLM emitted unknown grade=%r — coercing to 'relevant'",
                grade,
            )
            grade = "relevant"

        reason = str(data.get("reason") or "").strip() or "(no rationale provided)"
        rewritten_raw = data.get("rewritten_query")
        rewritten = str(rewritten_raw).strip() if isinstance(rewritten_raw, str) else None
        if rewritten in ("", "null", "None"):
            rewritten = None
        # Don't propagate a rewrite when the grade says we don't need one.
        if grade == "relevant":
            rewritten = None

        result = GraderResult(grade=grade, reason=reason, rewritten_query=rewritten)
        logger.info(
            "[RetrievalGrader] grade=%s reason=%r rewrite=%r",
            result.grade, result.reason[:120], result.rewritten_query,
        )
        return result
