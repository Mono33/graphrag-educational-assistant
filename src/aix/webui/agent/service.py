"""
Agent streaming service — the single seam between ``aix.webui`` and the
LangGraph agent (CORE 2 #6.6 P2).

Why a service module instead of calling the orchestrator from the route:
    1. The route layer should only translate HTTP ↔ events. All knowledge of
       LangGraph's state shape, node names, and revision-loop quirks lives
       here so we can swap the engine (or move it behind an RPC) without
       touching ``lessons/routes.py`` or any template.
    2. ``AgentOrchestrator.create_lesson_plan`` is *atomic* — it ``ainvoke``s
       the compiled graph and returns the final result. P2 needs phase-level
       progress, which means we have to drop down to the same compiled graph
       and use ``astream(..., stream_mode="updates")`` ourselves.
    3. Persisting the run lifecycle (``Lesson.status``, ``lesson_plan_md``,
       ``error_message``, ``teacher_query``) is something the agent doesn't
       know about — it's a webui concern. Doing it here means the route
       handler stays trivial.

Event taxonomy (P2 phase 2 — chat workspace):
    The chat UI is a stack of cards, one per agent step, accumulated in
    order. The service emits a ``StreamEvent`` with a ``kind`` that maps
    1:1 to a Jinja2 partial in ``templates/partials/chat_*.html``:

        kind             card                                emitted when
        ─────────────    ────────────────────────────────    ────────────────────
        "planner"        chat_planner_card.html              after `plan` node
        "retriever"      chat_retriever_card.html            after `retrieve`
                         (also drives the right-side          (single event,
                         media panel via OOB swap)            two render targets)
        "writer_pending" chat_writer_pending.html            right before write
                                                              starts (synthetic;
                                                              writer is the slow
                                                              call so the chat
                                                              must not look stuck)
        "writer"         chat_writer_card.html               after `write` node
                                                              (replaces the
                                                              matching pending
                                                              card via OOB)
        "critic"         chat_critic_card.html               after `critique`
                                                              node
        "done"           chat_lesson_card.html (+ summary)   end of run
        "error"          chat_error.html                     on exception

    LLM token streaming is *not* enabled inside the agent's nodes today
    (planner / writer / critic call ``chat.completions.create`` non-
    streamingly), so phase-level granularity is the realistic ceiling
    without modifying the agents themselves. P2 phase 3 will switch the
    writer agent to ``stream_mode="messages"`` so writer tokens land in
    the writer card live.

Reentrancy / concurrency:
    The compiled graph and its underlying agents are module-level singletons
    inside ``aix.agent.graph.nodes``. That's fine for a single user dev box
    but worth re-evaluating in CORE 6 deploy. The route layer enforces
    one in-flight run per ``lesson.id`` via ``_ACTIVE_RUNS``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical phase metadata (must match the node names registered in
# ``build_lesson_planner_graph`` — see aix/agent/graph/lesson_planner_graph.py).
# Italian labels are user-facing; keep them in sync with the templates.
#
# We keep ``PHASE_ORDER`` / ``PHASE_LABELS`` exported for routes.py and any
# legacy partial that still wants a generic tracker. The chat workspace
# itself doesn't render a separate tracker — the per-agent cards *are* the
# tracker.
# ---------------------------------------------------------------------------

PHASE_ORDER: tuple[str, ...] = ("plan", "retrieve", "write", "critique")

PHASE_LABELS: dict[str, str] = {
    "plan": "Pianificazione della lezione",
    "retrieve": "Recupero contesto dal Knowledge Graph",
    # CORE 2 #9 — Corrective RAG. The ``grade_retrieval`` node is added
    # to the workflow only when ``AIX_CORRECTIVE_RAG_ENABLED=true``; we
    # whitelist it here so its state diffs are applied to ``final_state``
    # in the SSE loop (otherwise the grade fields would never reach the
    # closing meta event). When the flag is OFF, the node is absent from
    # the topology and this label is unused — pre-#9 behaviour preserved.
    "grade_retrieval": "Valutazione qualità del recupero (Corrective RAG)",
    "write": "Scrittura della lezione",
    "critique": "Revisione e valutazione qualità",
}


# Italian labels for the planner's enum outputs. Keep these here (not in
# the template) so the template only needs `{{ payload.intent_label }}`.
_INTENT_LABELS: dict[str, str] = {
    "lesson_creation": "Creazione lezione",
    "activity_design": "Progettazione attività",
    "definition": "Definizione",
    "comparison": "Confronto",
    "explanation": "Spiegazione",
    "recommendation": "Raccomandazione",
    "list": "Elenco",
    "unknown": "Sconosciuto",
}

# Scope status → (label, wa-tag variant). The variant choice is intentional:
# in_scope is "success" (green) because the KG can speak to the topic;
# partial / out_of_scope are "warning" (amber) because the lesson will be
# composed primarily from external sources (Wikipedia, OER).
_SCOPE_LABELS: dict[str, tuple[str, str]] = {
    "in_scope":      ("Nel Knowledge Graph",     "success"),
    "partial_scope": ("Parzialmente nel KG",     "warning"),
    "out_of_scope":  ("Fuori dal KG",            "warning"),
    "unknown":       ("Scope sconosciuto",       "neutral"),
}


# ---------------------------------------------------------------------------
# Event model — what the route layer / SSE rendering consumes
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """
    Normalized event yielded by ``run_agent_stream``.

    Field usage by ``kind``:

        kind=="planner"
            payload = { intent, intent_label, scope, scope_label,
                        scope_variant, scope_confidence,
                        key_concepts: [...], search_queries: [...] }

        kind=="retriever"
            payload = { nodes_count, relationships_count,
                        recommendations_count, media_counts: {videos, articles, oer},
                        media: <full curated_media dict for the side panel>,
                        top_concepts: [...], retrieval_confidence }

        kind=="writer_pending"
            payload = { revision, is_revision, feedback }
            ``revision`` is a 1-based counter so the UI can render unique
            DOM ids per attempt (writer-card-rev1, -rev2, ...). The first
            attempt has is_revision=False; subsequent attempts (entered
            from a critic non-approval) have is_revision=True and may
            carry the critic's ``revision_instructions`` text.

        kind=="writer"
            payload = { revision }
            lesson_plan_md = <draft markdown for THIS revision>
            The route renders this *as* the chat-side writer card and uses
            ``revision`` to OOB-replace the matching writer-pending card.

        kind=="critic"
            payload = { approved, revision_count, max_revisions, score,
                        score_pct, critique, revision_instructions }

        kind=="done"
            lesson_plan_md = <full final markdown>
            meta = { duration_seconds, approved, revision_count,
                     scores, nodes_count, recommendations_count,
                     media_counts, search_queries_count }

        kind=="writer_chunk"
            payload = { token: "<raw text delta from streaming Writer LLM>" }
            Emitted many times during write_node — one per LLM output token.
            The webUI JS appends each token to #writer-stream-{lesson_id}.

        kind=="error"
            error = "<short message, ≤ 480 chars>"
    """

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    lesson_plan_md: Optional[str] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers — query synthesis + payload builders
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Language detection — 3-layer architecture (CORE 2 #10 follow-up, Point (a))
# ---------------------------------------------------------------------------
#
# The OLD detector here was a hard-coded Italian stop-word list:
#
#     _ITALIAN_INDICATORS = {"come", "cosa", "per", "con", ...}
#     "it" if words & _ITALIAN_INDICATORS else "en"
#
# It misclassified perfectly normal queries:
#   * "differenza tra motivazione intrinseca ed estrinseca" → "en"
#     (the connective "tra" wasn't in the set, "intrinseca" looks foreign)
#   * "elenca i tipi di memoria" → "en" (no overlap with the seed words)
#   * "DSA per studenti dislessici" → could go either way
#
# The new architecture has THREE layers:
#   L1 — Planner LLM (canonical, see RetrievalPlan.response_language /
#        plan_node language override). The Planner sees the FULL augmented
#        query (incl. multi-turn context) and emits a confidence-weighted
#        response_language that overrides the seed.
#   L2 — Statistical detector (lingua-language-detector). Pure-Python
#        n-gram + character-trigram model that handles short text, mixed
#        content, and acronyms much better than a stop-word list. Runs
#        BEFORE the Planner to seed state["language"] (the Planner needs a
#        language hint, even if it can later override it).
#   L3 — Default "it". Final safety net. Rationale: the platform's primary
#        user base is Italian teachers; defaulting to "it" minimizes the
#        L1-mismatch risk on edge cases (very short queries, pure
#        terminology, acronyms-only).
#
# This module owns L2 + L3. L1 lives in planner_agent.py + nodes.plan_node.

_DEFAULT_LANGUAGE = "it"
_SUPPORTED_LANGUAGES = {"it", "en", "es", "fr"}

# Lazy-init singleton for lingua. We only instantiate the detector once per
# process because building the n-gram models eats ~50ms and a few MB. The
# bool sentinel ``_lingua_unavailable`` short-circuits subsequent calls when
# the package is missing so we don't pay the import-failure cost on every
# query.
_lingua_detector: Any = None
_lingua_unavailable: bool = False


def _get_lingua_detector() -> Any:
    """
    Return a cached ``lingua.LanguageDetector`` configured for the four
    languages we actively support, or ``None`` if the package is unavailable.

    Building the detector with ``with_low_accuracy_mode()`` skips the heaviest
    n-gram models — perfectly fine for our short-query use case (teacher
    prompts are typically 1-3 sentences) and shaves ~5x off init time.
    """
    global _lingua_detector, _lingua_unavailable

    if _lingua_unavailable:
        return None
    if _lingua_detector is not None:
        return _lingua_detector

    try:
        from lingua import Language, LanguageDetectorBuilder

        languages = [
            Language.ITALIAN,
            Language.ENGLISH,
            Language.SPANISH,
            Language.FRENCH,
        ]
        _lingua_detector = (
            LanguageDetectorBuilder
            .from_languages(*languages)
            .with_low_accuracy_mode()
            .build()
        )
        logger.info(
            "[language] lingua-language-detector initialized (4 languages, "
            "low-accuracy mode)"
        )
        return _lingua_detector
    except Exception as exc:  # ImportError, runtime errors, etc.
        _lingua_unavailable = True
        logger.warning(
            "[language] lingua-language-detector unavailable (%s) — "
            "falling back to default %r. To enable statistical detection: "
            "pip install lingua-language-detector",
            exc, _DEFAULT_LANGUAGE,
        )
        return None


def _detect_language(query: str) -> str:
    """
    Statistical-or-default seed language for the agent (L2 / L3).

    This is the SEED that's passed to the Planner. The Planner then sees
    the actual query text and may OVERRIDE this with its own LLM-driven
    detection (L1) when confident — see ``aix.agent.graph.nodes.plan_node``.

    Returns:
        ISO 2-letter code, one of ``it`` / ``en`` / ``es`` / ``fr``. Falls
        back to ``"it"`` (the platform's primary user language) on:
          * empty / whitespace-only / very short queries (< 4 chars),
          * lingua-language-detector not installed (graceful degradation),
          * detector returning ``None`` (= no confident match).
    """
    text = (query or "").strip()
    if len(text) < 4:
        # Too short to detect reliably (e.g., "DSA", "ADHD", "ok").
        # Keep the platform default rather than risk a wrong override.
        return _DEFAULT_LANGUAGE

    detector = _get_lingua_detector()
    if detector is None:
        return _DEFAULT_LANGUAGE

    try:
        match = detector.detect_language_of(text)
    except Exception as exc:
        # Defensive: detector should be pure-python and side-effect-free,
        # but we don't want a language-detection blip to crash the run.
        logger.warning(
            "[language] lingua detection failed for query (len=%d): %s — "
            "defaulting to %r",
            len(text), exc, _DEFAULT_LANGUAGE,
        )
        return _DEFAULT_LANGUAGE

    if match is None:
        return _DEFAULT_LANGUAGE

    # lingua's Language enum exposes ``iso_code_639_1`` as a typed enum
    # member; we want the lowercase 2-letter string ("it"/"en"/"es"/"fr").
    code = getattr(match.iso_code_639_1, "name", "").lower()
    if code in _SUPPORTED_LANGUAGES:
        return code
    return _DEFAULT_LANGUAGE


def _query_from_lesson(lesson: Any) -> str:
    """
    Build the natural-language teacher query the agent expects when the
    user did *not* supply a free-text query in the lesson form.

    The agent expects "Crea una lezione su X per Y". We synthesize a
    sensible default from the profile fields. The agent then specializes
    via the ``educational_profile`` we also pass through (CORE 1 #2.5).
    """
    profile = lesson.educational_profile_json or {}
    topic = profile.get("specific_topic") or lesson.title or "argomento generale"
    subject = profile.get("subject_area")
    grade = (profile.get("group") or {}).get("grade")

    parts = [f"Crea una lezione su {topic}"]
    if subject:
        parts.append(f"per {subject}")
    if grade:
        # Grade is a code like "PRIMARY_4" — the prompt is friendlier with
        # the raw code than nothing; the educational_profile dict carries
        # the structured value too.
        parts.append(f"(livello: {grade})")

    return " ".join(parts)


def _extract_lesson_plan_md(final_state: Dict[str, Any]) -> str:
    """
    The lesson plan ends up under ``final_lesson_plan`` after a successful
    run, or ``lesson_plan_draft`` if the critic loop bailed out before
    approving. Empty string if neither is present (degenerate case — the
    route layer will surface this as an error).
    """
    return (
        final_state.get("final_lesson_plan")
        or final_state.get("lesson_plan_draft")
        or ""
    )


def _count_media(media: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """
    Reduce the ``curated_media`` dict to a flat tallied summary suitable
    for the retriever card and the final run summary.

    Source shape (see ``retriever_agent.py``):
        curated_media = {
            "videos":         [ {title, url, ...}, ... ],
            "resources":      [ {title, url, ...}, ... ],
            "citations":      [ {title, authors, year, ...}, ... ],
            "open_textbooks": [ {title, source, chapter, ...}, ... ],
            "images":         [ ... ],   # not surfaced in the UI yet
        }

    We bucket these into the three sidebar groups the user actually sees:
        videos    → "Video curati"
        articles  → "Articoli scientifici"   (citations from Semantic Scholar)
        oer       → "Risorse OER & Manuali"  (resources + open_textbooks)
    """
    if not isinstance(media, dict):
        return {"videos": 0, "articles": 0, "oer": 0}

    def _len_of(key: str) -> int:
        value = media.get(key)
        return len(value) if isinstance(value, (list, dict)) else 0

    return {
        "videos":    _len_of("videos"),
        "articles":  _len_of("citations"),
        "oer":       _len_of("resources") + _len_of("open_textbooks"),
        "web":       _len_of("web_links"),
    }


def _teacher_upload_context(lesson: Any) -> Optional[str]:
    """
    Join the text excerpts of the teacher's uploaded files into a single
    plain-text block for ``AgentState.teacher_provided_context``.

    This is *only* used by the Writer prompt (CORE 2 #6.6 P3). The Planner
    and Retriever stay GraphRAG-only — uploads are not ingested into the
    domain Knowledge Graph in this scope.
    """
    raw = getattr(lesson, "uploaded_files_json", None)
    if not raw or not isinstance(raw, list):
        return None
    parts: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ex = (item.get("text_excerpt") or "").strip()
        if not ex:
            continue
        title = (item.get("filename") or "file").strip()
        parts.append(f"### {title}\n{ex}")
    if not parts:
        return None
    joined = "\n\n".join(parts)
    return joined[:48000] if len(joined) > 48000 else joined


# ---------------------------------------------------------------------------
# Multi-turn conversation history (CORE 2 #10.3).
#
# We persist each turn as a pair of ``LessonMessage`` rows (user + assistant)
# alongside the LangGraph checkpointer's msgpack thread store. The pair
# (CQRS pattern) gives us:
#   • a SQL-queryable transcript for the chat-pane render path,
#   • a direct write-after-success hook for the assistant's turn,
#   • a stable, dialect-agnostic source of truth that survives a checkpointer
#     wipe, schema upgrade, or storage backend swap (#15 Postgres).
#
# The agent layer doesn't read these tables directly — the service composes
# a flat ``conversation_history`` list and injects it as an augmented
# ``teacher_query`` so EVERY agent in the pipeline (Planner / Retriever /
# Writer / Critic) sees the prior context without bespoke prompt edits.
# Per-agent prompt-level integration is a follow-up; the service-layer
# augmentation is sufficient for V1 multi-turn UX.
# ---------------------------------------------------------------------------

# Truncation for prior assistant messages when they're folded into the
# augmented query. 2000 chars per assistant turn keeps the prompt bounded
# at ~10 turns × 2000 = 20K chars (≈5K tokens) without #10.4's summary
# buffer kicking in. The summary buffer (#10.4) runs BEFORE this and
# replaces older turns wholesale, so this cap only hits the in-window turns.
_HISTORY_ASSISTANT_EXCERPT_CHARS = 2000

# Truncation cap on user messages when folded into the augmented query.
# Conservative: user turns are usually short ("now adapt for ADHD") but
# we hard-cap at 1500 chars so a teacher who pasted a long doc into the
# chat doesn't blow the prompt size budget.
_HISTORY_USER_EXCERPT_CHARS = 1500


async def _load_conversation_history(
    session: AsyncSession,
    lesson_id: Any,
) -> List[Dict[str, str]]:
    """
    Load the prior-turn ``LessonMessage`` rows (excluding the current
    in-progress turn) and shape them into the
    ``[{"role": "user"|"assistant", "content": str}, ...]`` form expected
    by the augmenter and by ``AgentState.conversation_history``.

    Filter rule: include all rows with ``turn_index`` strictly less than
    the latest turn_index for this lesson. The latest turn_index is the
    user message we're about to run (just persisted by the route layer);
    its assistant reply doesn't exist yet. Earlier turns are guaranteed
    to have BOTH user + assistant rows by construction (assistant write
    is gated on agent success in run_agent_stream below).
    """
    # Imported here to keep this module free of cross-package import
    # cycles during cold-collection (e.g., test runners that import
    # ``aix.webui.agent.service`` without webui's full DB stack).
    from aix.webui.lessons.models import LessonMessage

    latest_turn = await session.scalar(
        select(func.max(LessonMessage.turn_index))
        .where(LessonMessage.lesson_id == lesson_id)
    )
    if not latest_turn or latest_turn < 2:
        # Either no rows (legacy lesson with backfill skipped, or the
        # very first turn) or only the current in-progress turn exists.
        # Either way: no prior context to inject.
        return []

    rows_result = await session.execute(
        select(LessonMessage)
        .where(LessonMessage.lesson_id == lesson_id)
        .where(LessonMessage.turn_index < latest_turn)
        .order_by(LessonMessage.turn_index, LessonMessage.created_at)
    )
    history: List[Dict[str, str]] = []
    for msg in rows_result.scalars().all():
        if msg.role not in ("user", "assistant"):
            # Reserved roles (``system`` for #10.4 summary buffer) are
            # carried via separate AgentState fields, not the inline list.
            continue
        history.append({"role": msg.role, "content": msg.content_md or ""})
    return history


def _augment_query_with_history(
    raw_query: str,
    history: List[Dict[str, str]],
    summary: Optional[str],
    language: str,
) -> str:
    """
    Compose an augmented teacher query that includes prior conversation
    context, a summary of older turns (when present), and the current
    request — in that order.

    Returns ``raw_query`` unchanged when there's no history AND no
    summary (= first turn) so the single-turn path is byte-identical to
    the pre-#10 behaviour. This makes the history-injection feature
    fully backward-compatible for first turns and degraded-mode runs.

    Format (Italian by default; English when ``language == 'en'``):

        ## Conversazione precedente

        ### Sintesi dei turni più vecchi    (only when summary present)
        {summary}

        ### Turno 1 — Docente
        {prior_user_1}

        ### Turno 1 — Risposta dell'assistente
        {prior_assistant_1_excerpt}

        … (one block per prior turn) …

        ## Nuova richiesta del docente
        {raw_query}
    """
    if not history and not (summary and summary.strip()):
        return raw_query

    is_it = (language or "it").lower().startswith("it")

    history_label  = "Conversazione precedente"        if is_it else "Previous conversation"
    summary_label  = "Sintesi dei turni più vecchi"    if is_it else "Summary of older turns"
    turn_label     = "Turno"                            if is_it else "Turn"
    user_label     = "Docente"                          if is_it else "Teacher"
    asst_label     = "Risposta dell'assistente"         if is_it else "Assistant reply"
    request_label  = "Nuova richiesta del docente"     if is_it else "New request from the teacher"

    parts: List[str] = [f"## {history_label}", ""]

    if summary and summary.strip():
        parts.append(f"### {summary_label}")
        parts.append(summary.strip())
        parts.append("")

    # Group consecutive (user, assistant) pairs into turns. We're trusting
    # the upstream invariant that history is well-formed (no orphans), so
    # we walk linearly and bump the turn counter on each user message.
    turn_idx = 0
    for msg in history:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            turn_idx += 1
            excerpt = content[:_HISTORY_USER_EXCERPT_CHARS]
            if len(content) > _HISTORY_USER_EXCERPT_CHARS:
                excerpt = excerpt + "…"
            parts.append(f"### {turn_label} {turn_idx} — {user_label}")
            parts.append(excerpt)
            parts.append("")
        elif role == "assistant":
            excerpt = content[:_HISTORY_ASSISTANT_EXCERPT_CHARS]
            if len(content) > _HISTORY_ASSISTANT_EXCERPT_CHARS:
                excerpt = excerpt + "…"
            parts.append(f"### {turn_label} {turn_idx} — {asst_label}")
            parts.append(excerpt)
            parts.append("")

    parts.append(f"## {request_label}")
    parts.append(raw_query)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Summary-buffer windowing (CORE 2 #10.4).
#
# Long conversations would eventually overflow the model's context window
# (Claude Sonnet 4.6 is generous at 200k, but our augmented prompt also
# carries the educational profile, the retriever's KG context, and the
# Writer's instruction template — call it ~30k of prompt budget). Even
# well within the limit, every extra historical turn pays a per-token cost
# on every subsequent call.
#
# Strategy (keep it simple; revisit if it doesn't fit reality):
#   • Keep the last ``WINDOW_TURNS`` turns verbatim — short-term memory.
#   • Summarise everything older — long-term memory.
#   • The summary lives in ``AgentState.conversation_summary`` and is
#     rendered as a "Sintesi dei turni più vecchi" block in the augmented
#     query (see ``_augment_query_with_history``).
#
# Tunable via the ``AIX_CONVERSATION_WINDOW_TURNS`` env var (default 4 →
# 4 user/assistant pairs ≈ 8 messages ≈ ~10–20K chars retained verbatim).
# Setting it to 0 effectively disables verbatim retention (everything is
# summarised); a high value (e.g. 50) effectively disables windowing.
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW_TURNS = 4


def _window_turns_from_env() -> int:
    """Resolve the configured window size, with a safe default + clamp."""
    try:
        raw = int(os.getenv("AIX_CONVERSATION_WINDOW_TURNS", str(_DEFAULT_WINDOW_TURNS)))
    except (ValueError, TypeError):
        return _DEFAULT_WINDOW_TURNS
    # Clamp: 0 ≤ N ≤ 50. Below 0 makes no sense; above 50 is "effectively
    # disabled" and we'd rather be obvious about it than silently allow
    # 10000.
    return max(0, min(50, raw))


async def _maybe_window_history(
    history: List[Dict[str, str]],
    language: str,
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """
    Apply summary-buffer windowing to ``history`` when it exceeds the
    configured window size.

    Returns ``(summary, retained_history)``:
        • If ``history`` fits within the window → ``(None, history)``
          unchanged. Identical to single-turn behaviour.
        • If ``history`` exceeds the window → summarise the oldest
          turns into a single string, return that summary plus the most
          recent ``WINDOW_TURNS`` turns kept verbatim.

    Failure handling: if the summarisation LLM call fails (network,
    rate limit, parser error), we log + return ``(None, history)`` so
    the run continues with the full untrimmed history. The downstream
    augmenter and Writer will likely succeed with the longer prompt
    even if it's wasteful — better than crashing the run.
    """
    window_turns = _window_turns_from_env()
    window_messages = window_turns * 2  # each turn = user + assistant pair

    if len(history) <= window_messages:
        return None, history

    older = history[:-window_messages] if window_messages else history
    recent = history[-window_messages:] if window_messages else []

    try:
        summary = await _summarise_history(older, language)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[webui.agent] history summarisation failed: %s; "
            "passing full history without windowing", exc,
        )
        return None, history

    if not summary or not summary.strip():
        # LLM returned empty — fall back to no windowing rather than
        # silently dropping context.
        logger.warning(
            "[webui.agent] history summarisation returned empty; "
            "passing full history without windowing"
        )
        return None, history

    logger.info(
        "[webui.agent] windowing applied: summarised %s older messages "
        "into %s chars; kept %s recent messages verbatim",
        len(older), len(summary), len(recent),
    )
    return summary.strip(), recent


async def _summarise_history(
    older_messages: List[Dict[str, str]],
    language: str,
) -> str:
    """
    Concise LLM-driven summary of the oldest portion of the conversation.

    Uses the same OpenAI/OpenRouter client as the agents — no new API key
    or model selection. Temperature is low (0.2) because we want
    deterministic, fact-preserving compression, not creative paraphrase.

    Token budget for the summary itself is capped at 600 tokens via
    ``max_tokens`` so a runaway model can't burn budget on a 5000-token
    summary that defeats the windowing's purpose.
    """
    # Local import to avoid pulling the OpenAI stack into the module's
    # cold import path (test collection, etc.). Mirrors the pattern in
    # ``run_agent_stream``.
    from aix.core.config import config as app_config, extract_response_content

    if not older_messages:
        return ""

    # Format the older messages as a compact transcript. Truncate each
    # individual message at 1200 chars so a single long lesson plan can't
    # blow the summariser's input budget.
    transcript_lines: List[str] = []
    for m in older_messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        excerpt = content[:1200] + ("…" if len(content) > 1200 else "")
        if role == "user":
            transcript_lines.append(f"### Docente\n{excerpt}")
        elif role == "assistant":
            transcript_lines.append(f"### Assistente\n{excerpt}")
        else:
            transcript_lines.append(f"### {role.title()}\n{excerpt}")
    transcript = "\n\n".join(transcript_lines)

    is_it = (language or "it").lower().startswith("it")
    if is_it:
        system_prompt = (
            "Sei un assistente che riassume in modo conciso e fedele "
            "conversazioni didattiche tra un docente e un assistente "
            "educativo. Mantieni TUTTI i fatti rilevanti per continuare "
            "la conversazione: argomento della lezione, livello scolastico, "
            "vincoli temporali, bisogni educativi speciali, scelte già "
            "fatte, decisioni concordate. Usa elenchi puntati (max 8 punti). "
            "Non aggiungere commenti, scrivi solo il riassunto."
        )
        user_prompt = (
            "Riassumi la seguente conversazione (turni più vecchi) in "
            "italiano:\n\n" + transcript
        )
    else:
        system_prompt = (
            "You summarise teacher–assistant educational conversations "
            "concisely and faithfully. Preserve ALL facts relevant to "
            "continuing the conversation: lesson topic, grade level, "
            "time constraints, special educational needs, choices already "
            "made, agreed decisions. Use bullet points (max 8). No "
            "commentary — output only the summary."
        )
        user_prompt = (
            "Summarise the following conversation (older turns) in "
            "English:\n\n" + transcript
        )

    client = app_config.openai.get_async_client()
    completion_kwargs = app_config.openai.build_completion_kwargs(
        temperature=0.2,
        max_tokens=600,
    )
    response = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **completion_kwargs,
    )
    return extract_response_content(response, logger)


async def _persist_assistant_turn(
    session: AsyncSession,
    lesson_id: Any,
    content_md: str,
    meta: Dict[str, Any],
) -> None:
    """
    Append the assistant's response for the current turn to the
    ``lesson_message`` log.

    Turn index discovery:
        We use ``MAX(turn_index)`` for this lesson's user messages. The
        route layer persisted the user message at this turn_index BEFORE
        invoking the service; the assistant's reply shares it. This keeps
        (user, assistant) pairs aligned without explicit threading of
        ``turn_index`` through the service signature.

    Failure handling:
        We log + swallow on persistence failure. The lesson row's
        ``status="complete"`` and ``lesson_plan_md`` writes are the
        load-bearing persistence; the lesson_message row is the
        history-render index, important but not run-critical. Skipping
        it on a transient DB error is preferable to failing the whole
        run.
    """
    from aix.webui.lessons.models import LessonMessage

    try:
        latest_turn = await session.scalar(
            select(func.max(LessonMessage.turn_index))
            .where(LessonMessage.lesson_id == lesson_id)
            .where(LessonMessage.role == "user")
        ) or 1

        session.add(LessonMessage(
            lesson_id=lesson_id,
            role="assistant",
            content_md=content_md,
            turn_index=int(latest_turn),
            agent_kind="writer",
            meta_json=meta or None,
        ))
        await session.commit()
    except Exception:  # noqa: BLE001
        logger.exception(
            "[webui.agent] failed to persist assistant LessonMessage lesson_id=%s; "
            "lesson row state is authoritative",
            lesson_id,
        )
        # Roll back the open transaction so subsequent writes (e.g., the
        # lesson.status flip on a downstream error) don't ride on a
        # poisoned session.
        try:
            await session.rollback()
        except Exception:  # noqa: BLE001
            pass


def _build_planner_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the post-plan state into the planner card's context."""
    intent = state.get("query_intent") or "unknown"
    scope = state.get("scope_status") or "unknown"
    scope_label, scope_variant = _SCOPE_LABELS.get(scope, (scope, "neutral"))
    return {
        "intent": intent,
        "intent_label": _INTENT_LABELS.get(intent, intent),
        "scope": scope,
        "scope_label": scope_label,
        "scope_variant": scope_variant,
        "scope_confidence": state.get("scope_confidence"),
        "key_concepts": list(state.get("key_concepts") or [])[:8],
        "search_queries": list(state.get("search_queries") or [])[:8],
        "lesson_type": state.get("lesson_type"),
        "target_grade": state.get("target_grade"),
    }


def _is_corrective_rag_enabled() -> bool:
    """CORE 2 #9 — match the graph builder's flag check exactly so the
    SSE loop's behaviour stays in sync with the topology actually built
    by ``build_lesson_planner_graph_async``. Imported here lazily to
    avoid a circular import on cold start (service → nodes → service)."""
    from aix.agent.graph.nodes import _corrective_rag_enabled
    return _corrective_rag_enabled()


def _resolve_max_attempts() -> int:
    """Resolve ``AIX_CORRECTIVE_RAG_MAX_ATTEMPTS`` with the same defaults
    and clamping that :func:`aix.agent.graph.nodes._corrective_rag_max_attempts`
    uses. Single source of truth for the SSE layer (consumed by both
    :func:`_grader_will_retry` and :func:`_build_retriever_payload`).
    """
    raw = os.getenv("AIX_CORRECTIVE_RAG_MAX_ATTEMPTS", "2")
    try:
        return max(1, min(int(raw), 4))
    except ValueError:
        return 2


def _grader_will_retry(state: Dict[str, Any]) -> bool:
    """Mirror the routing logic of :func:`should_retry_retrieval` in
    ``nodes.py``. Returns True iff the corrective-RAG router will send
    the run BACK to the retriever instead of forward to the writer.

    Kept as a module-private helper rather than importing the router
    function directly because the router takes ``AgentState`` (TypedDict)
    and we only have the loose ``final_state`` dict here. Logic is
    intentionally simple and well-covered by the agent-side tests.
    """
    grade = state.get("retrieval_grade", "relevant")
    if grade == "relevant":
        return False
    attempts = int(state.get("retrieval_attempts") or 0)
    return attempts < _resolve_max_attempts()


# ---------------------------------------------------------------------------
# CORE 2 #9.UX-5 — domain-aware teacher-friendly retriever copy (CR OFF only).
#
# When Corrective RAG is disabled (the production default in ``.env``), the
# ``grade_retrieval`` node never runs, so the entire ``aix-outcome-callout``
# block in ``chat_retriever_card.html`` is skipped — leaving the retriever
# card with no explainability layer at all. The teacher sees raw counts
# (nodes_count / recommendations_count / media) but no narrative about
# what they MEAN for the lesson.
#
# #9.UX-5 fills that gap with a 3-tier static banner derived purely from
# ``nodes_count``:
#
#   tier="healthy"      ≥5 nodes   sage    "Ricerca completata sulla base {D}: ..."
#   tier="limited"      1-4 nodes  amber   "Copertura parziale ... La lezione sarà integrata ..."
#   tier="out_of_scope" 0 nodes    info    "Questo argomento non è presente nella base {D}: ..."
#
# Domain labels are short/long parallel forms — short for tier 1/2 headlines
# and the footer ("UDL"/"Neuro"), long for tier 0 where the explanatory
# parenthetical ("(pedagogia inclusiva)") improves teacher comprehension.
#
# Scope: CR OFF only. CR ON (``aix-outcome-callout``) is untouched — it
# already provides its own explainability via the four ``retrieval_outcome``
# branches. When CR is re-enabled (post #LAT-7), this same tier logic can
# be lifted into the outcome branches without coordination — they're
# mutually exclusive (the template's outer guards see to that).
# ---------------------------------------------------------------------------

_DOMAIN_LABELS: Dict[str, Dict[str, str]] = {
    "udl":   {"short": "UDL",   "long": "UDL (pedagogia inclusiva)"},
    "neuro": {"short": "Neuro", "long": "Neuro"},
}

# Tier classifier threshold. The 5-node cutoff is empirical: under 5 the
# Writer materially leans on general didactic knowledge to compose the
# lesson, so we owe the teacher a visible "expect lighter KG anchoring"
# signal. Configurable via ``AIX_COVERAGE_HEALTHY_THRESHOLD`` for ops
# tuning without a template touch.
_COVERAGE_HEALTHY_DEFAULT = 5


def _coverage_healthy_threshold() -> int:
    """Resolve the ``nodes_count`` floor for the ``healthy`` tier with a
    safe default. Clamped to a sane range so a typo (``999``) doesn't
    silently turn every lesson into ``limited``."""
    raw = os.getenv("AIX_COVERAGE_HEALTHY_THRESHOLD", str(_COVERAGE_HEALTHY_DEFAULT))
    try:
        return max(1, min(int(raw), 50))
    except ValueError:
        return _COVERAGE_HEALTHY_DEFAULT


def _resolve_domain_labels(domain: Optional[str]) -> Dict[str, str]:
    """Return the ``{short, long}`` label pair for ``domain``.

    Unknown domains fall back to the raw value as both short and long
    forms — so a future ``"stem"`` domain renders ``"stem"`` until we
    register a proper label, never crashing on the KeyError path.
    """
    key = (domain or "").lower().strip()
    if key in _DOMAIN_LABELS:
        return _DOMAIN_LABELS[key]
    safe = key or "il dominio attivo"
    return {"short": safe, "long": safe}


def _classify_coverage_tier(nodes_count: int) -> str:
    """Pure tier classifier — no env reads, no state, no domain logic.

    Returns one of ``"healthy"`` / ``"limited"`` / ``"out_of_scope"``.
    Kept as a free function so the unit test can lock the boundaries
    (0 → out_of_scope, 1..N-1 → limited, ≥N → healthy) without standing
    up a full ``_state()`` fixture.
    """
    if nodes_count <= 0:
        return "out_of_scope"
    if nodes_count < _coverage_healthy_threshold():
        return "limited"
    return "healthy"


# CORE 2 #9.UX-3 — sentinel prefix used by ``grade_retrieval_node`` to
# mark a defensive fallback after the grader LLM threw. The node returns
# ``grade=relevant`` in that case (so the loop never blocks the writer)
# but stamps the reason with this prefix so the UI layer can distinguish
# "real green light" from "we couldn't grade, defaulted green". See
# ``nodes.grade_retrieval_node``'s ``except Exception`` branch.
_GRADER_EXCEPTION_REASON_PREFIX = "Grader exception:"


def _compute_retrieval_outcome(
    state: Dict[str, Any],
    media_counts: Dict[str, int],
) -> str:
    """CORE 2 #9.UX-3 — derive the single ``retrieval_outcome`` token that
    drives the chat card's color, headline, and explanatory copy.

    Four mutually exclusive outcomes:

      ``"success"``               — grade=relevant (or grading didn't run).
                                    Green ✅. Existing behaviour, unchanged.
      ``"adapted_with_hybrid"``   — grade=ambiguous|irrelevant AFTER all
                                    attempts AND external/hybrid resources
                                    populated the gap. Blue ℹ️. The KG was
                                    out-of-scope for the disciplinary
                                    content, but Wikipedia + papers + OER
                                    filled in. The lesson is still useful.
      ``"limited_kg_only"``       — grade=ambiguous|irrelevant AFTER all
                                    attempts AND no external resources
                                    landed. Amber ⚠️. The lesson should
                                    be reviewed manually.
      ``"grader_error"``          — the grader LLM threw and the node
                                    fell back to ``grade=relevant`` with
                                    the sentinel reason prefix. Red ❌.
                                    Only legitimate red — distinct from
                                    "irrelevant" which is a routine
                                    out-of-scope signal, not an error.

    When the corrective-RAG flag is OFF (``grade is None``), this returns
    ``"success"`` for consistency, but the template's outer guard
    (``{% if p.get('retrieval_grade') %}``) skips the row entirely, so
    the rendered card is byte-identical to pre-#9.
    """
    grade = state.get("retrieval_grade")
    reason = (state.get("retrieval_grade_reason") or "")

    if reason.startswith(_GRADER_EXCEPTION_REASON_PREFIX):
        return "grader_error"

    if grade in (None, "relevant"):
        return "success"

    # grade is "ambiguous" or "irrelevant" — choose blue vs amber based
    # on whether the hybrid retrieval path landed any external content.
    external = state.get("external_resources")
    has_external = False
    if isinstance(external, dict):
        # Truthy if any sub-bucket has content (Wikipedia, OER, S2 papers…).
        has_external = any(bool(v) for v in external.values())
    elif external:  # list, scalar — be permissive
        has_external = True

    # Hybrid retrieval lands papers in ``citations`` and OER in
    # ``resources`` / ``open_textbooks`` (= ``oer`` bucket of media_counts).
    # Videos may be KG-curated too, so we DON'T count them as a
    # hybrid signal here — we want a clean "external content arrived"
    # indicator.
    has_hybrid_media = (
        int(media_counts.get("articles") or 0)
        + int(media_counts.get("oer") or 0)
    ) > 0

    if has_external or has_hybrid_media:
        return "adapted_with_hybrid"
    return "limited_kg_only"


def _build_retriever_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the post-retrieve state into the retriever card's context."""
    nodes = state.get("retrieved_nodes") or []
    rels = state.get("retrieved_relationships") or []
    recs = state.get("recommendations") or []
    media = state.get("curated_media") or {}
    media_counts = _count_media(media)

    # Best-effort top-N concept titles. Different code paths populate
    # different keys (``title`` vs ``name`` vs ``id``); we walk them all.
    top_concepts: list[str] = []
    for n in nodes[:5]:
        if not isinstance(n, dict):
            continue
        title = n.get("title") or n.get("name") or n.get("label") or n.get("id")
        if title:
            top_concepts.append(str(title))

    # CORE 2 #9 — Corrective RAG (Retrieval Grading). When the feature
    # flag is off, the grade fields are None and the retriever card
    # template skips the grading row, so the rendered card is identical
    # to pre-#9. When ON, we surface a 1-row "Grading" line (grade icon +
    # rationale + attempts) on the existing retriever card without adding
    # a new card or stream-event kind. Adding fields to an existing
    # payload is the smallest possible UI change to keep #9 visible.
    grade = state.get("retrieval_grade")
    grade_label_map = {
        "relevant": "Rilevante",
        "ambiguous": "Ambiguo",
        "irrelevant": "Non rilevante",
    }
    grade_emoji_map = {
        "relevant": "✅",
        "ambiguous": "⚠️",
        "irrelevant": "❌",
    }

    # CORE 2 #9.UX-3 — outcome is the single token the template branches
    # on (success / adapted_with_hybrid / limited_kg_only / grader_error).
    # Computed defensively (always returns one of the four valid values).
    outcome = _compute_retrieval_outcome(state, media_counts)

    # CORE 2 #9.UX-5 — domain-aware coverage tier for the CR-OFF banner.
    # Computed unconditionally (cheap pure functions); the template only
    # renders the banner block when ``retrieval_grade`` is None, so these
    # fields are inert on the CR-ON path. Bundling them with the rest of
    # the payload keeps the SSE event shape stable across flag states.
    domain_raw = state.get("domain")
    domain_labels = _resolve_domain_labels(domain_raw)
    nodes_count = len(nodes)
    coverage_tier = _classify_coverage_tier(nodes_count)
    media_total = (
        int(media_counts.get("videos") or 0)
        + int(media_counts.get("articles") or 0)
        + int(media_counts.get("oer") or 0)
    )

    return {
        "nodes_count": nodes_count,
        "relationships_count": len(rels),
        "recommendations_count": len(recs),
        "media_counts": media_counts,
        "media_total": media_total,
        "media": media,  # full payload for the right sidebar
        "top_concepts": top_concepts,
        "retrieval_confidence": state.get("retrieval_confidence"),
        # Corrective-RAG fields. All None when feature flag is off; the
        # template's ``{% if payload.retrieval_grade %}`` guard keeps the
        # markup unchanged in that case.
        "retrieval_grade": grade,
        "retrieval_grade_label": grade_label_map.get(grade) if grade else None,
        "retrieval_grade_emoji": grade_emoji_map.get(grade) if grade else None,
        "retrieval_grade_reason": (state.get("retrieval_grade_reason") or "").strip() or None,
        "retrieval_attempts": state.get("retrieval_attempts"),
        # CORE 2 #9.UX-2 — paired with ``retrieval_attempts`` so the chat
        # card can render a ``Tentativi: N/M`` badge. Always populated
        # (resolved from env at request time) so the template doesn't
        # need to know about env vars.
        "retrieval_attempts_max": _resolve_max_attempts(),
        # CORE 2 #9.UX-3 — outcome token; see ``_compute_retrieval_outcome``.
        "retrieval_outcome": outcome,
        "retrieval_warning": bool(state.get("retrieval_warning")),
        "retrieval_rewritten_query": state.get("retrieval_rewritten_query"),
        # CORE 2 #9.UX-5 — domain-aware fields for the CR-OFF banner and
        # the (domain-aware) bottom footer label. ``domain`` is the raw
        # value from state; the two label forms are pre-rendered so the
        # template never sees a dictionary lookup or a hardcoded label.
        "domain": (domain_raw or "").lower() or None,
        "domain_label_short": domain_labels["short"],
        "domain_label_long":  domain_labels["long"],
        "coverage_tier": coverage_tier,
    }


def _build_critic_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the post-critique state into the critic card's context."""
    score = state.get("critique_score")
    score_pct: Optional[int] = None
    if isinstance(score, (int, float)):
        raw = float(score)
        # The critic agent's ``average_score`` is on a 1–5 scale (see
        # ``CritiqueResult.average_score`` and ``critic_prompt.py``).
        # Convert to 0–100% for display, clamped.
        if raw <= 5.0:
            score_pct = min(100, max(0, int(round(raw / 5.0 * 100))))
        else:
            score_pct = min(100, int(round(raw)))

    return {
        "approved": bool(state.get("approved", False)),
        "revision_count": int(state.get("revision_count", 0)),
        "max_revisions": int(state.get("max_revisions", 1)),
        "score": score,
        "score_pct": score_pct,
        "critique": (state.get("critique") or "").strip(),
        "revision_instructions": (state.get("revision_instructions") or "").strip(),
    }


def _extract_meta(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """Run summary used by the final lesson card and the run-complete banner."""
    media = final_state.get("curated_media") or {}
    return {
        "approved": bool(final_state.get("approved", False)),
        "revision_count": int(final_state.get("revision_count", 0)),
        "scores": (final_state.get("final_metadata") or {}).get("scores"),
        "nodes_count": len(final_state.get("retrieved_nodes") or []),
        "recommendations_count": len(final_state.get("recommendations") or []),
        "media_counts": _count_media(media),
        "search_queries_count": len(final_state.get("search_queries") or []),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_agent_stream(
    lesson: Any,
    session: AsyncSession,
) -> AsyncIterator[StreamEvent]:
    """
    Run the LangGraph lesson-planner pipeline for ``lesson`` and yield
    chat-card events as the graph progresses.

    Lifecycle side-effects (persisted via ``session``):
        on entry         : lesson.status = "running",
                           lesson.teacher_query persisted (if not already)
        on success       : lesson.status = "complete",
                           lesson_plan_md = <markdown>
        on exception     : lesson.status = "error",
                           error_message = <short msg>

    Yield order for a 0-revision run (happy path):
        planner → retriever → writer_pending → writer → critic → done

    Yield order for a 1-revision run:
        planner → retriever
                → writer_pending(rev=1)  → writer(rev=1)  → critic
                → writer_pending(rev=2)  → writer(rev=2)  → critic
                → done

    The route layer maps each event 1:1 to an SSE message; the partial
    template named ``partials/chat_<kind>_card.html`` (or ``chat_writer_*``)
    renders the card. The ``writer_pending`` event lets the UI show a
    "Sto scrivendo…" placeholder during the (slow, non-streaming) writer
    call; the matching ``writer`` event then replaces that placeholder
    in-place via an OOB swap on the unique ``writer-card-rev{N}`` id.
    """
    started_at = time.monotonic()
    final_state: Dict[str, Any] = {}
    write_revision_idx = 0  # 1-based counter once first writer_pending fires

    # ── Setup phase. Anything that can blow up here (import errors, OpenAI
    # client init, missing env vars, …) MUST be caught so we never leave the
    # lesson in ``status="running"`` with no streaming events. Without this
    # guard the chat appears frozen for the user (P3 regression).
    try:
        # Lazy imports: keep the heavy LangGraph + OpenAI stack out of the
        # import path of cold modules (test collection, etc.). The
        # orchestrator already does ``load_dotenv()`` at import — fine to
        # incur once per process.
        from aix.agent.graph.state import create_initial_state
        from aix.agent.orchestrator import AgentOrchestrator

        profile_dict = lesson.educational_profile_json or None

        # Use the persisted teacher_query if the user supplied one in the
        # form; otherwise synthesize from the profile and *write it back*
        # so the user's first chat bubble survives reloads.
        if getattr(lesson, "teacher_query", None) and lesson.teacher_query.strip():
            raw_query = lesson.teacher_query.strip()
        else:
            raw_query = _query_from_lesson(lesson)
            lesson.teacher_query = raw_query

        # ── Multi-turn history loading (#10.3) ───────────────────────
        # Load all PRIOR turns from the lesson_messages CQRS log and shape
        # them into the AgentState.conversation_history form. On the very
        # first turn this returns []; on follow-ups it carries the past
        # exchanges (post-backfill for legacy lessons).
        full_history = await _load_conversation_history(session, lesson.id)

        # ── Summary-buffer windowing (#10.4) ─────────────────────────
        # If the conversation has more than WINDOW_TURNS prior turns,
        # summarise the older ones and keep only the recent window
        # verbatim. Short conversations short-circuit cleanly with
        # ``(None, full_history)`` so single-turn / few-turn behaviour
        # is identical to pre-#10.4. Failures fall back to the full
        # untrimmed history with a logged warning.
        conversation_summary, conversation_history = await _maybe_window_history(
            history=full_history,
            language=_detect_language(raw_query),
        )

        # Compose the augmented query. When history is empty AND no summary
        # is set (= first turn), this returns raw_query unchanged — single
        # turn behaviour is byte-identical to pre-#10.
        query = _augment_query_with_history(
            raw_query=raw_query,
            history=conversation_history,
            summary=conversation_summary,
            language=_detect_language(raw_query),
        )

        # Domain comes from the form ("neuro" / "udl" — captured at submit
        # time in P1). Language is inferred from the teacher's RAW query
        # (not the history-augmented one) so the prompt block we just
        # appended doesn't accidentally override the teacher's true
        # language preference.
        orchestrator = AgentOrchestrator(
            domain=lesson.domain or "neuro",
            language=_detect_language(raw_query),
        )
        pipeline = orchestrator._get_pipeline()  # noqa: SLF001 — intentional seam

        # CORE 2 #10.2 — compile the graph with the AsyncSqliteSaver
        # checkpointer when available. Falls back to the no-checkpointer
        # path on graceful degradation (langgraph-checkpoint-sqlite missing,
        # disk write error, etc.) so single-turn behaviour is preserved
        # in degraded environments.
        graph = await pipeline._get_graph_async()  # noqa: SLF001

        # Thread config — required by LangGraph whenever a checkpointer
        # is attached. ``str(lesson.id)`` is the canonical thread_id so
        # follow-up turns on the same lesson share state (multi-turn —
        # see #10.3 which adds the user-facing follow_up / regenerate
        # / new modes on top of this plumbing).
        from aix.agent.graph.checkpointer import thread_config
        run_config = thread_config(str(lesson.id))

        teacher_ctx = _teacher_upload_context(lesson)

        initial_state = create_initial_state(
            query=query,
            domain=pipeline.domain,
            language=pipeline.language,
            session_id=str(lesson.id),
            max_revisions=pipeline.max_revisions,
            educational_profile=profile_dict,
            teacher_provided_context=teacher_ctx,
            conversation_history=conversation_history or None,
            conversation_summary=conversation_summary,
            # CORE 2 #12b.3 — preserve the un-augmented current turn so
            # plan_node can apply user-vs-history precedence on duration
            # (and any other profile-vs-history conflicts in future).
            raw_user_turn=raw_query,
        )

        # CORE 2 #9.UX-5 hotfix — pre-seed final_state with initial_state so
        # static fields (notably ``domain``) that no node ever overwrites are
        # visible to the post-stream payload builders. LangGraph state_diff
        # chunks only carry fields the graph actually mutated, so without this
        # seed ``final_state.get("domain")`` returns None and the coverage
        # banner falls back to the generic "il dominio attivo" label.
        final_state.update(initial_state)

        logger.info(
            "[webui.agent] starting run lesson_id=%s domain=%s query=%r "
            "uploads=%s thread_id=%s history_turns=%s",
            lesson.id, lesson.domain, raw_query[:80],
            len(getattr(lesson, "uploaded_files_json", None) or []),
            run_config["configurable"]["thread_id"],
            # Number of completed prior turns (= half of history length,
            # since each turn contributes a user + assistant pair).
            len(conversation_history) // 2,
        )

        # ── Mark RUNNING ─────────────────────────────────────────────
        lesson.status = "running"
        lesson.error_message = None
        await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[webui.agent] setup FAILED lesson_id=%s", lesson.id,
        )
        msg = str(exc) or exc.__class__.__name__
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")
        try:
            lesson.status = "error"
            lesson.error_message = short_msg
            await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception(
                "[webui.agent] failed to persist setup error for lesson_id=%s",
                lesson.id,
            )
        yield StreamEvent(kind="error", error=short_msg)
        return

    # ── Run the graph ───────────────────────────────────────────────────────
    from aix.agent.graph import write_stream as _write_stream  # lazy — avoids circular at module level
    _write_stream.register(str(lesson.id))
    try:
        async for chunk in graph.astream(
            initial_state, config=run_config, stream_mode="updates",
        ):
            for node_name, state_diff in chunk.items():
                if node_name not in PHASE_LABELS:
                    continue
                if isinstance(state_diff, dict):
                    final_state.update(state_diff)

                if node_name == "plan":
                    yield StreamEvent(
                        kind="planner",
                        payload=_build_planner_payload(final_state),
                    )
                    yield StreamEvent(kind="retriever_pending", payload={})

                elif node_name == "retrieve":
                    # CORE 2 #9.UX-1 — symmetric with stream_agent_events.
                    if not _is_corrective_rag_enabled():
                        yield StreamEvent(
                            kind="retriever",
                            payload=_build_retriever_payload(final_state),
                        )
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": False,
                                "feedback": "",
                            },
                        )

                elif node_name == "grade_retrieval":
                    # CORE 2 #9.UX-2 — emit once on final attempt.
                    if not _grader_will_retry(final_state):
                        yield StreamEvent(
                            kind="retriever",
                            payload=_build_retriever_payload(final_state),
                        )
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": False,
                                "feedback": "",
                            },
                        )

                elif node_name == "write":
                    yield StreamEvent(
                        kind="writer",
                        payload={"revision": write_revision_idx},
                        lesson_plan_md=final_state.get("lesson_plan_draft") or "",
                    )
                    yield StreamEvent(
                        kind="critic_pending",
                        payload={"write_revision": write_revision_idx},
                    )

                elif node_name == "critique":
                    _critic_pl = _build_critic_payload(final_state)
                    _critic_pl["write_revision"] = write_revision_idx
                    yield StreamEvent(
                        kind="critic",
                        payload=_critic_pl,
                    )
                    approved = final_state.get("approved", False)
                    rev_count = int(final_state.get("revision_count", 0))
                    max_rev = int(final_state.get("max_revisions", 1))
                    if (not approved) and rev_count < max_rev:
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": True,
                                "feedback": (
                                    final_state.get("revision_instructions") or ""
                                ).strip(),
                            },
                        )

        # ── Run finished cleanly ─────────────────────────────────────────────
        elapsed = time.monotonic() - started_at
        lesson_plan_md = _extract_lesson_plan_md(final_state)
        meta = _extract_meta(final_state)
        meta["duration_seconds"] = round(elapsed, 1)

        if not lesson_plan_md.strip():
            raise RuntimeError(
                "L'agente ha terminato senza produrre una lezione "
                "(stato finale vuoto)."
            )

        lesson.status = "complete"
        lesson.lesson_plan_md = lesson_plan_md
        await session.commit()

        await _persist_assistant_turn(
            session=session,
            lesson_id=lesson.id,
            content_md=lesson_plan_md,
            meta=meta,
        )

        logger.info(
            "[webui.agent] run complete lesson_id=%s duration=%.1fs "
            "approved=%s revisions=%s thread_id=%s",
            lesson.id, elapsed, meta["approved"], meta["revision_count"],
            run_config["configurable"]["thread_id"],
        )

        yield StreamEvent(
            kind="done",
            lesson_plan_md=lesson_plan_md,
            meta=meta,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[webui.agent] run FAILED lesson_id=%s after %.1fs",
            lesson.id, time.monotonic() - started_at,
        )
        msg = str(exc) or exc.__class__.__name__
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")
        try:
            lesson.status = "error"
            lesson.error_message = short_msg
            await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception(
                "[webui.agent] failed to persist error state for lesson_id=%s",
                lesson.id,
            )
        yield StreamEvent(kind="error", error=short_msg)
    finally:
        _write_stream.deregister(str(lesson.id))


# ---------------------------------------------------------------------------
# Public DB-less helper — used by the JSON+SSE API at /api/v1/agent/*
#
# This is the same engine ``run_agent_stream`` drives, with two changes:
#   1. No ``Lesson`` row, no ``AsyncSession``, no DB writes.
#   2. Inputs come from the validated Pydantic request, not a persisted row.
#
# Why a separate helper instead of refactoring ``run_agent_stream``:
#   - ``run_agent_stream`` is on the hot path of the webui chat workspace
#     and has been smoke-tested through P2 → P3. Touching it would put that
#     surface at risk for the sake of code dedup.
#   - Both helpers share the *real* logic — the LangGraph ``astream`` loop
#     and the ``_build_*_payload`` shapers — by direct call. The only
#     duplication is the ~25-line dispatch loop, which is the cheapest
#     possible price for backward compatibility.
#   - When CORE 6 deploy is ready, both can be collapsed into a single
#     ``aix.agent.streaming`` module that lives under the agent layer
#     proper. That refactor is tracked under #7's "future improvements"
#     bullet, not in scope here.
#
# CORE 2 #7 — see docs/product/ClickUp_Agentic_GraphRAG_Update.md.
# ---------------------------------------------------------------------------


async def stream_agent_events(
    *,
    query: str,
    domain: str = "neuro",
    language: str = "it",
    session_id: Optional[str] = None,
    educational_profile: Optional[Dict[str, Any]] = None,
    teacher_provided_context: Optional[str] = None,
    max_revisions: Optional[int] = None,
) -> AsyncIterator[StreamEvent]:
    """
    Run the LangGraph lesson-planner pipeline for an *ad-hoc* request and
    yield ``StreamEvent`` objects in the same order ``run_agent_stream``
    does.

    This helper has no DB persistence: callers are responsible for any
    state they want to keep. It exists to back the public JSON+SSE API
    where the request is one-shot and the contract is the *event stream*
    itself, not a persisted ``Lesson`` row.

    Parameters mirror :func:`aix.agent.graph.state.create_initial_state`,
    which is the single source of truth for the agent's input shape.

    Yields the same ``kind`` taxonomy as :func:`run_agent_stream`:

        planner → retriever → writer_pending → writer → critic → done

    On any setup or runtime exception, yields exactly one ``error`` event
    and returns. The caller never sees an exception cross the generator
    boundary; failures are domain data.
    """
    started_at = time.monotonic()
    final_state: Dict[str, Any] = {}
    write_revision_idx = 0

    # ── Setup. Mirrors the guard in run_agent_stream so a missing API
    # key / import error / etc. surfaces as a clean ``error`` event
    # instead of a 500 from the route layer.
    try:
        import uuid as _uuid

        from aix.agent.graph.checkpointer import thread_config
        from aix.agent.graph.state import create_initial_state
        from aix.agent.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator(domain=domain, language=language)
        pipeline = orchestrator._get_pipeline()  # noqa: SLF001 — same seam as webui

        # CORE 2 #10.2 — checkpointed graph. Public-API callers that pass
        # a stable ``session_id`` get multi-turn memory across requests;
        # callers that don't get an ephemeral per-call thread (functionally
        # identical to the pre-#10 behaviour — single-turn, no shared state).
        graph = await pipeline._get_graph_async()  # noqa: SLF001
        effective_thread_id = session_id or f"ephemeral-{_uuid.uuid4()}"
        run_config = thread_config(effective_thread_id)

        effective_max_revisions = (
            max_revisions if max_revisions is not None else pipeline.max_revisions
        )

        initial_state = create_initial_state(
            query=query,
            domain=pipeline.domain,
            language=pipeline.language,
            session_id=session_id,
            max_revisions=effective_max_revisions,
            educational_profile=educational_profile,
            teacher_provided_context=teacher_provided_context,
            # CORE 2 #12b.3 — public-API callers don't augment with history
            # at this layer (the caller may pre-augment, but the contract
            # treats ``query`` as the current turn), so ``query`` is also
            # the raw user turn here. Passing it explicitly keeps plan_node's
            # precedence rule consistent across both entry points.
            raw_user_turn=query,
        )

        # CORE 2 #9.UX-5 hotfix — pre-seed final_state with initial_state so
        # static fields (notably ``domain``) that no node ever overwrites are
        # visible to the post-stream payload builders. See run_agent_stream
        # for the full rationale.
        final_state.update(initial_state)

        logger.info(
            "[api.agent] starting run session_id=%s thread_id=%s domain=%s query=%r "
            "max_revisions=%s profile=%s teacher_ctx_chars=%s",
            session_id, effective_thread_id, domain, query[:80],
            effective_max_revisions,
            "yes" if educational_profile else "no",
            len(teacher_provided_context or ""),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[api.agent] setup FAILED session_id=%s", session_id)
        msg = str(exc) or exc.__class__.__name__
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")
        yield StreamEvent(kind="error", error=short_msg)
        return

    try:
        async for chunk in graph.astream(
            initial_state, config=run_config, stream_mode="updates",
        ):
            for node_name, state_diff in chunk.items():
                if node_name not in PHASE_LABELS:
                    continue
                if isinstance(state_diff, dict):
                    final_state.update(state_diff)

                if node_name == "plan":
                    yield StreamEvent(
                        kind="planner",
                        payload=_build_planner_payload(final_state),
                    )

                elif node_name == "retrieve":
                    # CORE 2 #9.UX-1 — see first SSE loop above for the
                    # symmetric reasoning. When corrective-RAG is ON we
                    # defer the retriever emit to the ``grade_retrieval``
                    # branch so the chat shows ONE card per turn. When
                    # OFF, behaviour is unchanged from pre-#9.
                    if not _is_corrective_rag_enabled():
                        yield StreamEvent(
                            kind="retriever",
                            payload=_build_retriever_payload(final_state),
                        )
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": False,
                                "feedback": "",
                            },
                        )

                elif node_name == "grade_retrieval":
                    # CORE 2 #9.UX-2 — see the symmetric branch in
                    # ``run_agent_stream`` above for the full rationale.
                    # Gate the emit on ``_grader_will_retry`` so a turn
                    # with N attempts produces ONE retriever card with
                    # ``retrieval_attempts == N`` rather than N cards.
                    if not _grader_will_retry(final_state):
                        yield StreamEvent(
                            kind="retriever",
                            payload=_build_retriever_payload(final_state),
                        )
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": False,
                                "feedback": "",
                            },
                        )

                elif node_name == "write":
                    yield StreamEvent(
                        kind="writer",
                        payload={"revision": write_revision_idx},
                        lesson_plan_md=final_state.get("lesson_plan_draft") or "",
                    )

                elif node_name == "critique":
                    yield StreamEvent(
                        kind="critic",
                        payload=_build_critic_payload(final_state),
                    )
                    approved = final_state.get("approved", False)
                    rev_count = int(final_state.get("revision_count", 0))
                    max_rev = int(final_state.get("max_revisions", 1))
                    if (not approved) and rev_count < max_rev:
                        write_revision_idx += 1
                        yield StreamEvent(
                            kind="writer_pending",
                            payload={
                                "revision": write_revision_idx,
                                "is_revision": True,
                                "feedback": (final_state.get("revision_instructions") or "").strip(),
                            },
                        )

        # ── Done ────────────────────────────────────────────────────────
        elapsed = time.monotonic() - started_at
        lesson_plan_md = _extract_lesson_plan_md(final_state)
        meta = _extract_meta(final_state)
        meta["duration_seconds"] = round(elapsed, 1)

        if not lesson_plan_md.strip():
            raise RuntimeError(
                "L'agente ha terminato senza produrre una lezione "
                "(stato finale vuoto)."
            )

        logger.info(
            "[api.agent] run complete session_id=%s thread_id=%s duration=%.1fs "
            "approved=%s revisions=%s",
            session_id, effective_thread_id, elapsed,
            meta["approved"], meta["revision_count"],
        )

        yield StreamEvent(
            kind="done",
            lesson_plan_md=lesson_plan_md,
            meta=meta,
        )

    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[api.agent] run FAILED session_id=%s after %.1fs",
            session_id, time.monotonic() - started_at,
        )
        msg = str(exc) or exc.__class__.__name__
        short_msg = msg[:480] + ("…" if len(msg) > 480 else "")
        yield StreamEvent(kind="error", error=short_msg)
