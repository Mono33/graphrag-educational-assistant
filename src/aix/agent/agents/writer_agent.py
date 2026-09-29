"""
Writer Agent

Generates educational content adapted to the detected query intent.
Supports lesson plans, definitions, comparisons, explanations, recommendations, and lists.

Enhanced with curated media integration (Phase 2) for multimodal content support.
"""

import logging
import os
import re
from typing import Any, Optional

from openai import AsyncOpenAI

from aix.agent.prompts.today import today_note
from aix.agent.agents.planner_agent import RetrievalPlan
from aix.agent.agents.retriever_agent import RetrievalResult
from aix.agent.prompts.profile_renderer import render_profile_section
from aix.agent.prompts.writer_prompt import (
    WRITER_MATERIAL_MAPS_BLOCK,
    WRITER_MATERIAL_PASSAGES_APPENDIX,
    WRITER_REVISION_TEMPLATE,
    WRITER_TEACHER_UPLOADS_APPENDIX,
    WRITER_USER_TEMPLATE_HYBRID,
    WRITER_USER_TEMPLATE_RECOMMENDATION,
    format_external_resources,
    get_writer_prompts,
)
from aix.core.concurrency import guarded_chat_completion, llm_slot
from aix.core.config import config as app_config
from aix.core.config import extract_response_content

# Optional domain extensions - fails gracefully if not available
try:
    from aix.agent.configs.domain_prompts import get_domain_extension

    DOMAIN_EXTENSIONS_AVAILABLE = True
except ImportError:
    DOMAIN_EXTENSIONS_AVAILABLE = False

    def get_domain_extension(domain: str, agent: str) -> str:  # noqa: ARG001
        return ""


logger = logging.getLogger(__name__)

# Maximum output tokens for a single writer LLM call.
# At ~3 chars/token (Italian markdown), 3 500 tokens ≈ 10 500-12 000 characters.
# When the LLM hits this ceiling, the writer automatically makes ONE
# continuation call (see _continue_truncated) so the user never sees a
# mid-sentence cut. Override with AIX_WRITER_MAX_TOKENS env var.
_WRITER_MAX_TOKENS = int(os.getenv("AIX_WRITER_MAX_TOKENS", "8000"))

# Floor for the retry after a reasoning model spends its whole budget thinking
# and returns nothing. Retrying under the same ceiling — which is what happened
# on 2026-09-16 with AIX_WRITER_MAX_TOKENS=2000 — asks the model to do again
# exactly what it just could not do: two runs, four minutes, zero characters.
_WRITER_MIN_CONTENT_TOKENS = int(os.getenv("AIX_WRITER_MIN_CONTENT_TOKENS", "4000"))

#: §8.7 — the private material gets its own slice of the prompt, so a long
#: document cannot push out the system instructions, the classroom profile or
#: the Knowledge Graph evidence. ~12k characters is a few pages of quotations:
#: enough to ground a lesson, far from the budget that produced zero output on
#: 2026-09-16 when two whole attachments were pasted in.
_MATERIAL_BUDGET_CHARS = int(os.getenv("AIX_PRIVATE_RAG_BUDGET_CHARS", "12000"))

#: When the documents have maps, the passages come from two searches -- the
#: request, and the parts of the document it points to -- and the documents are
#: the primary source rather than an appendix: they get more room. The maps
#: themselves are bounded apart, by ``_MAPS_BUDGET_CHARS``.
_DOCUMENT_BUDGET_CHARS = int(os.getenv("AIX_PRIVATE_RAG_DOCUMENT_BUDGET_CHARS", "20000"))
_MAPS_BUDGET_CHARS = 6000


#: A document name is text the teacher chose, and it travels into a prompt.
#: Kept to one line and a sane length: a file called "ignora le istruzioni
#: precedenti.pdf" is a filename, not a turn of phrase the model should weigh.
_MAX_LABEL_CHARS = 60


def _safe_label(raw: Any) -> str:
    """One line, bounded, no control characters. See :data:`_MAX_LABEL_CHARS`."""
    label = " ".join(str(raw or "").split())
    return label[: _MAX_LABEL_CHARS - 1] + "…" if len(label) > _MAX_LABEL_CHARS else label


def render_material_passages(
    items: Optional[list[dict[str, Any]]], *, budget_chars: int = _MATERIAL_BUDGET_CHARS
) -> str:
    """The retrieved passages as the Writer sees them, grouped by document.

    Best first — they arrive sorted by score — and cut at the budget rather than
    truncated mid-passage, because half a quotation is something the model will
    complete from memory.

    The marker is the one the model must copy back (§8.6). OCR text says so, so
    a date read from a scan is not quoted as if it had been typed (§7.8).

    **Grouped under the document's name, and that is not cosmetic.** Until
    2026-09-21 the passages arrived as a flat list under opaque UUIDs, so a model
    handed five paragraphs of an Italian handout and three of an English one
    could not tell there were two documents at all. It wrote the Italian lesson
    from the Italian source and never cited the other — and the teacher, who had
    attached two files and saw one in the sources, reasonably concluded the
    second had been ignored. Naming them costs a line each.

    Documents appear in the order of their best passage, so the strongest source
    is still read first.
    """
    if not items:
        return ""

    usable = [
        item
        for item in items
        if str(item.get("text") or "").strip() and str(item.get("citation") or "").strip()
    ]
    if not usable:
        return ""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in usable:
        grouped.setdefault(str(item.get("source_id") or ""), []).append(item)

    several = len(grouped) > 1
    blocks: list[str] = []
    used = 0

    for number, passages in enumerate(grouped.values(), start=1):
        if several:
            label = _safe_label(passages[0].get("source_label")) or f"documento {number}"
            heading = f"### {label}"
            if used + len(heading) > budget_chars:
                break
            blocks.append(heading)
            used += len(heading)

        for item in passages:
            header = str(item.get("citation")).strip()
            if item.get("part"):
                header += f" · parte: {_safe_label(item.get('part'))}"
            if item.get("ocr"):
                header += " · testo letto con OCR"
            block = f"{header}\n«{str(item.get('text')).strip()}»"
            if used + len(block) > budget_chars:
                return "\n\n".join(blocks)
            blocks.append(block)
            used += len(block)

    return "\n\n".join(blocks)


def render_document_maps(
    maps: Optional[list[dict[str, Any]]], *, budget_chars: int = _MAPS_BUDGET_CHARS
) -> str:
    """The documents' maps as the Writer reads them: what each is about, its
    themes, and its parts with their pages. Cut at the budget, part by part."""
    if not maps:
        return ""
    lines: list[str] = []
    used = 0
    for number, document_map in enumerate(maps, start=1):
        label = _safe_label(document_map.get("source_label")) or f"documento {number}"
        head = [f"**{label}**"]
        if document_map.get("summary"):
            head.append(f"Di che cosa tratta: {document_map['summary']}")
        if document_map.get("themes"):
            head.append("Temi: " + ", ".join(str(t) for t in document_map["themes"]))
        sections = document_map.get("sections") or []
        if len(sections) > 1:
            head.append("Parti:")
        for line in head:
            if used + len(line) > budget_chars:
                return "\n".join(lines)
            lines.append(line)
            used += len(line)
        if len(sections) <= 1:
            continue
        for section in sections:
            pages = section.get("pages")
            where = f" (pp. {pages[0]}–{pages[1]})" if pages and pages[0] != pages[1] else (
                f" (p. {pages[0]})" if pages else ""
            )
            line = f"- {section.get('title', '')}{where}: {section.get('summary', '')}"
            if used + len(line) > budget_chars:
                return "\n".join(lines)
            lines.append(line)
            used += len(line)
    return "\n".join(lines)


def teacher_material_appendix(
    material_context_items: Optional[list[dict[str, Any]]],
    teacher_provided_context: Optional[str],
    *,
    phase: str,
    material_maps: Optional[list[dict[str, Any]]] = None,
) -> str:
    """The teacher's material for the prompt: retrieved passages, or the upload.

    One place for the choice ``write()`` and ``revise()`` both make, and one log
    line saying which way it went. Without that line a run with the private RAG
    on looks identical in the log to a run that silently fell back to pasting the
    whole document — and telling those apart is the first thing to check when a
    lesson ignores its sources.
    """
    maps = render_document_maps(material_maps)
    passages = render_material_passages(
        material_context_items,
        budget_chars=_DOCUMENT_BUDGET_CHARS if maps else _MATERIAL_BUDGET_CHARS,
    )
    if passages or maps:
        logger.info(
            "[WriterAgent] %s: %d retrieved passage(s), %d chars, %d map(s) — the document"
            " is not pasted",
            phase,
            passages.count("[[material:"),
            len(passages),
            len(material_maps or []) if maps else 0,
        )
        return WRITER_MATERIAL_PASSAGES_APPENDIX.format(
            material_maps=WRITER_MATERIAL_MAPS_BLOCK.format(maps=maps) if maps else "",
            material_passages=passages
            or "(Nessun passaggio dei documenti è pertinente a questa richiesta.)",
        )

    if teacher_provided_context and teacher_provided_context.strip():
        logger.info(
            "[WriterAgent] %s: no retrieved passages — using the uploaded excerpt (%d chars)",
            phase,
            len(teacher_provided_context),
        )
        return WRITER_TEACHER_UPLOADS_APPENDIX.format(
            teacher_provided_context=teacher_provided_context.strip()
        )

    return ""


# Max number of automatic continuation calls when the writer output is cut
# by finish_reason="length". 1 = at most 2 total LLM calls per write/revise.
_WRITER_MAX_CONTINUATIONS = int(os.getenv("AIX_WRITER_MAX_CONTINUATIONS", "1"))

# Continuation instruction appended to the conversation when the assistant's
# previous turn was cut by finish_reason=length. The previous assistant turn
# (already in the message history) ends mid-token; we ask the model to pick
# up exactly where it stopped without repeating anything.
_CONTINUATION_INSTRUCTION = (
    "Your previous response was cut off mid-content because you hit the token limit. "
    "Continue writing EXACTLY from where you stopped — do not repeat any content, "
    "do not start a new heading, do not add a transition. Pick up the sentence or "
    "list item that was cut and finish the lesson concisely, ending with the "
    "`*Fonti:*` footer line."
)


class WriterAgent:
    """
    Writer Agent - Generates educational content adapted to query intent.

    Responsibilities:
    1. Detect query intent from the plan
    2. Select appropriate output format and template
    3. Generate content grounded in retrieved evidence
    4. Handle revision requests from CriticAgent
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Writer Agent.

        Args:
            model: OpenAI model to use for generation
        """
        self.model = model
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            self._client = app_config.openai.get_async_client()
        return self._client

    async def _stream_completion(
        self,
        client: AsyncOpenAI,
        messages: list,
        completion_kwargs: dict,
        token_bus,
    ) -> tuple:
        """Run one streaming completion, push tokens to the bus, and return
        (content, finish_reason). Shared by the initial write call and any
        auto-continuation calls so both feed the same UI stream.
        """
        content_parts: list[str] = []
        last_finish_reason: Optional[str] = None
        think_chunks_seen = 0
        # CORE 6 #32 — hold a global LLM slot across the ENTIRE streaming call
        # (open + token consumption), since the request is in-flight the whole
        # time. Releasing after ``create()`` would under-count the slowest LLM
        # call we make. The slot is released when the stream is fully drained.
        async with llm_slot():
            stream = await client.chat.completions.create(
                messages=messages,
                stream=True,
                **completion_kwargs,
            )
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                # Thinking tokens — OpenRouter may expose them in several places
                # depending on the SDK version and model family. Check all variants
                # defensively so an unexpected structure never kills the content stream.
                if token_bus:
                    try:
                        # Variant A: reasoning_details list (OpenRouter standard for R1 + Claude)
                        rd_list = (
                            getattr(choice.delta, "reasoning_details", None)
                            or (getattr(choice.delta, "model_extra", None) or {}).get(
                                "reasoning_details"
                            )
                            or []
                        )
                        for rd in rd_list:
                            text = (
                                rd.get("text", "")
                                if isinstance(rd, dict)
                                else getattr(rd, "text", "")
                            )
                            if text:
                                token_bus.put_nowait(("think", text))
                                think_chunks_seen += 1
                        # Variant B: reasoning_content string (some OpenRouter models)
                        rc = getattr(choice.delta, "reasoning_content", None) or (
                            getattr(choice.delta, "model_extra", None) or {}
                        ).get("reasoning_content")
                        if rc:
                            token_bus.put_nowait(("think", rc))
                            think_chunks_seen += 1
                    except Exception as _think_exc:
                        logger.debug(
                            "[WriterAgent] thinking-token extraction skipped: %s", _think_exc
                        )
                # Lesson content tokens
                delta = choice.delta.content or ""
                if delta:
                    content_parts.append(delta)
                    if token_bus:
                        token_bus.put_nowait(("content", delta))
                if choice.finish_reason:
                    last_finish_reason = choice.finish_reason
        if think_chunks_seen:
            logger.info("[WriterAgent] Streamed %d thinking chunks to bus", think_chunks_seen)
        return "".join(content_parts), last_finish_reason

    async def write(
        self,
        teacher_query: str,
        plan: RetrievalPlan,
        retrieval_result: RetrievalResult,
        language: str = "it",
        curated_media: Optional[dict[str, Any]] = None,  # Phase 2: Optional media
        external_resources: Optional[dict[str, Any]] = None,  # Phase A: External resources
        domain: str = "neuro",  # Phase B: Domain for extensions
        teacher_provided_context: Optional[str] = None,  # WebUI #6.6 P3: chat uploads
        material_context_items: Optional[list[dict[str, Any]]] = None,  # PR 6: §8.2
        material_maps: Optional[list[dict[str, Any]]] = None,  # what the documents hold
        educational_profile: Optional[dict[str, Any]] = None,  # Teacher's lesson profile
        pedagogical_intent: Optional[str] = None,  # "{code}" or "{code}: {detail}"
        refinement_instruction: Optional[str] = None,  # SAM refinement instruction
        token_bus=None,  # Optional asyncio.Queue for live token streaming to the webUI
        # CORE 2 #9 — Corrective RAG. When the grader exhausted its retry
        # budget without reaching ``relevant``, the node sets
        # ``state.retrieval_warning=True`` and we receive it here. The
        # Writer then prepends a short "low-confidence caveat" instruction
        # to the user prompt so the lesson carries an explicit flag for
        # the teacher rather than silently authoring on weak evidence.
        # ``None`` / ``False`` = no caveat, output identical to pre-#9.
        retrieval_warning: Optional[bool] = None,
        retrieval_grade_reason: Optional[str] = None,
    ) -> str:
        """
        Generate content based on the detected intent and retrieved context.

        Args:
            teacher_query: Original teacher request
            plan: Retrieval plan with intent and lesson requirements
            retrieval_result: Retrieved knowledge from GraphRAG
            language: Output language ("it" or "en")
            curated_media: Optional curated media from sidecar JSON (Phase 2)
            external_resources: Optional external resources for hybrid mode (Phase A)
            teacher_provided_context: Optional plain-text appendix from files
                attached by the teacher in the WebUI chat (CORE 2 #6.6 P3).
                These do NOT enter the Knowledge Graph and do NOT influence
                the Planner / Retriever — they are appended only to this
                Writer prompt as supplementary context.

        Returns:
            Generated content as markdown string
        """
        # Get the query intent and scope from the plan
        intent = getattr(plan, "query_intent", "lesson_creation")
        scope_status = getattr(plan, "scope_status", "in_scope")

        # Check for media/resources availability
        has_media = bool(curated_media and any(curated_media.values()))
        has_external = bool(external_resources and any(external_resources.values()))

        # Determine if this is hybrid mode
        is_hybrid = scope_status in ("partial_scope", "out_of_scope") and has_external

        scope_emoji = {"in_scope": "✅", "partial_scope": "⚠️", "out_of_scope": "❌"}.get(
            scope_status, "❓"
        )
        logger.info(
            f"[WriterAgent] Generating content: intent={intent}, scope={scope_emoji} {scope_status}, "
            f"media={has_media}, hybrid={is_hybrid}"
        )

        client = self._get_client()

        # Get intent-specific prompts (with scope awareness for hybrid mode)
        system_prompt, user_template = get_writer_prompts(intent, scope_status)

        # Format retrieved nodes for prompt
        nodes_text = self._format_nodes(retrieval_result.nodes)
        recommendations_text = self._format_recommendations(retrieval_result.recommendations)
        key_concepts_text = ", ".join(plan.key_concepts) if plan.key_concepts else "Not specified"

        # NEW Phase 2: Format curated media if available
        media_text = self._format_media(curated_media) if has_media else ""

        # The whole profile, with codes resolved to the teacher's labels and the
        # classroom restated as constraints. This used to be built here from five
        # fields, so the model never saw the device policy, the LIM, the WiFi, the
        # furniture or the class size — and read the BES codes as acronyms it
        # expanded from memory. See ``profile_renderer`` for what changed and why.
        edu_profile_section = render_profile_section(educational_profile)

        # Resolve pedagogical_intent code → human label + prompt instruction.
        # Stored as "{code}" or "{code}: {detail}". If the code matches a
        # predefined option, inject its resolved label and prompt instruction.
        if pedagogical_intent:
            from aix.api.schemas.educational_profile import PEDAGOGICAL_INTENT_BY_CODE

            code, _, detail = pedagogical_intent.partition(": ")
            code = code.strip()
            detail = detail.strip()
            option = PEDAGOGICAL_INTENT_BY_CODE.get(code)
            if option:
                resolved_label = option["label"]
                resolved_prompt = option["prompt"]
                intent_line = f"Obiettivo pedagogico: {resolved_label}"
                if detail:
                    intent_line += f" — {detail}"
                edu_profile_section += (
                    f"\n## Pedagogical Intent\n"
                    f"- {intent_line}\n"
                    f"- Instruction: {resolved_prompt}" + (f" ({detail})" if detail else "") + "\n"
                )
            else:
                edu_profile_section += f"\n## Pedagogical Intent\n- {pedagogical_intent}\n"

        # NEW Phase A: Handle HYBRID mode (out-of-scope with external resources)
        if is_hybrid and intent in ("lesson_creation", "activity_design"):
            # is_hybrid requires has_external which requires external_resources to be non-None
            assert external_resources is not None
            wikipedia_content, papers_content, oer_content = format_external_resources(
                external_resources
            )

            user_prompt = WRITER_USER_TEMPLATE_HYBRID.format(
                teacher_query=teacher_query,
                educational_profile_section=edu_profile_section,
                wikipedia_content=wikipedia_content,
                papers_content=papers_content,
                oer_content=oer_content,
                recommendations=recommendations_text,
                retrieved_nodes=nodes_text,
                lesson_type=plan.lesson_type or "full_lesson",
                target_grade=plan.target_grade or "Not specified",
                time_constraints=plan.time_constraints or "Not specified",
                language="Italian" if language == "it" else "English",
            )

            logger.info(
                "[WriterAgent] Using HYBRID mode prompt with external resources (OER: %d textbooks)",
                len(external_resources.get("oer_textbooks", [])),
            )

        # Standard mode: Format user prompt based on intent
        elif intent in ("lesson_creation", "activity_design"):
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                educational_profile_section=edu_profile_section,
                key_concepts=key_concepts_text,
                recommendations=recommendations_text,
                retrieved_nodes=nodes_text,
                lesson_type=plan.lesson_type or "full_lesson",
                target_grade=plan.target_grade or "Not specified",
                time_constraints=plan.time_constraints or "Not specified",
                special_needs=", ".join(plan.special_needs) if plan.special_needs else "None",
                language="Italian" if language == "it" else "English",
            )
        elif intent == "recommendation":
            user_prompt = WRITER_USER_TEMPLATE_RECOMMENDATION.format(
                teacher_query=teacher_query,
                key_concepts=key_concepts_text,
                recommendations=recommendations_text,
                retrieved_nodes=nodes_text,
                special_needs=", ".join(plan.special_needs) if plan.special_needs else "None",
                language="Italian" if language == "it" else "English",
            )
        else:
            # definition, comparison, explanation, list
            user_prompt = user_template.format(
                teacher_query=teacher_query,
                key_concepts=key_concepts_text,
                retrieved_nodes=nodes_text,
                language="Italian" if language == "it" else "English",
            )

        # SAM refinement instruction — prepend so the model's first frame
        # is "improve the previous lesson in this specific direction".
        if refinement_instruction and refinement_instruction.strip():
            user_prompt = (
                f"## Improvement requested by the teacher\n"
                f"{refinement_instruction.strip()}\n\n"
                f"Apply the improvement above to the lesson you generate below. "
                f"Produce a complete revised lesson (not just the changed parts).\n\n"
            ) + user_prompt

        # Fase 2 · PR 6 — the retrieved passages replace the pasted document.
        # Both would be the same text twice: once chosen for this request, once
        # entire. The fallback is deliberate (§5.4 hybrid): when retrieval finds
        # nothing — still being indexed, or the flag is off — the teacher's
        # upload still reaches the Writer exactly as it did before.
        user_prompt += today_note()
        user_prompt += teacher_material_appendix(
            material_context_items,
            teacher_provided_context,
            phase="write",
            material_maps=material_maps,
        )

        # NEW Phase 2: Append media context if available
        if media_text:
            user_prompt += media_text

        # CORE 2 #9 — Corrective RAG low-confidence caveat (additive).
        # Only fires when ``retrieval_warning=True`` (i.e., the grader's
        # retry budget was exhausted with grade != "relevant"). The
        # snippet asks the writer to add a single short note at the top of
        # the lesson telling the teacher that the KG match was weak — it
        # does NOT change the rest of the prompt. With the corrective-RAG
        # feature flag off this branch is dead code (the flag never
        # populates ``retrieval_warning``).
        if retrieval_warning:
            reason_clause = ""
            if retrieval_grade_reason:
                reason_clause = f' (grader rationale: "{retrieval_grade_reason.strip()}")'
            if language == "it":
                user_prompt += (
                    "\n\n## ⚠️ Avviso di bassa confidenza sul Knowledge Graph"
                    f"{reason_clause}\n"
                    "Il recupero dal Knowledge Graph non ha trovato concetti pienamente "
                    "allineati alla query. Procedi comunque a comporre la lezione, ma "
                    "aggiungi UNA breve nota ESPLICITA in cima al documento (1-2 righe, in "
                    "italiano, formato callout/blockquote) che avverte il docente che il KG "
                    "è risultato a bassa confidenza per questo argomento e che le fonti "
                    "richiamate potrebbero essere parziali. Non inventare contenuti che non "
                    "siano supportati dai concetti recuperati."
                )
            else:
                user_prompt += (
                    "\n\n## ⚠️ Knowledge-Graph low-confidence notice"
                    f"{reason_clause}\n"
                    "The Knowledge-Graph retrieval did not surface concepts that fully "
                    "match the query. Proceed to draft the lesson, but ADD a single "
                    "short note at the very top of the document (1-2 lines, blockquote/"
                    "callout) telling the teacher that the KG was low-confidence on this "
                    "topic and that the cited sources may be partial. Do not invent "
                    "content that the retrieved concepts do not support."
                )
            logger.info(
                "[WriterAgent] retrieval_warning=True → appended low-confidence caveat to user prompt"
            )

        # Format system prompt with language
        system_prompt = system_prompt.replace(
            "{language}", "Italian" if language == "it" else "English"
        )

        # NEW Phase B: Add domain-specific extensions
        if DOMAIN_EXTENSIONS_AVAILABLE:
            domain_ext = get_domain_extension(domain, "writer")
            if domain_ext:
                system_prompt += domain_ext
                logger.info(f"[WriterAgent] Applied domain extension for '{domain}'")

        # Hard override appended LAST so it takes precedence over any domain
        # extension (e.g. Langfuse udl.writer_prompt) that might instruct the
        # LLM to annotate with "> 🔗 UDL Principle:" or "> ⚠️ Why it matters:".
        # These annotations waste tokens and cause truncation.
        system_prompt += (
            "\n\n**FINAL CONSTRAINT (overrides all above)**: "
            "Do NOT add any inline annotation markers in the lesson body — "
            "this includes `> 🔗 ...`, `> ⚠️ ...`, `[✅ ...]`, `[📌 ...]`, "
            "or any similar bracketed/blockquote source tags. "
            "All source citations go exclusively in the `*Fonti:*` footer."
        )

        try:
            # max_tokens=8000: Italian markdown lessons with structured sections
            # (I DO / WE DO / YOU DO / Conclusione / Riferimenti) routinely run
            # 12-20K characters. With Anthropic's tokenizer averaging ~3 chars
            # per token on heavily-formatted markdown, a 4K cap was clipping
            # full lessons mid-sentence (observed: 11292-char output stopping
            # at "Cosa la r"). 8K covers ~24K chars with comfortable headroom
            # while staying well below Claude Sonnet 4.6's 64K output ceiling.
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.7,
                max_tokens=_WRITER_MAX_TOKENS,
            )
            messages: list[Any] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if token_bus is not None:
                # Streaming path: forward each token to the webUI bus so the
                # SSE endpoint can display the lesson as it's being written.
                content, last_finish_reason = await self._stream_completion(
                    client,
                    messages,
                    completion_kwargs,
                    token_bus,
                )
                # Safety net: reasoning model exhausted the token budget on
                # thinking and produced zero content. Retry once with thinking
                # disabled so the model generates the lesson without CoT overhead.
                if not content:
                    retry_max_tokens = max(_WRITER_MAX_TOKENS, _WRITER_MIN_CONTENT_TOKENS)
                    logger.warning(
                        "[WriterAgent] Zero-content response after streaming "
                        "(reasoning model exhausted the %d-token budget). Retrying "
                        "with include_reasoning=False and max_tokens=%d.",
                        _WRITER_MAX_TOKENS,
                        retry_max_tokens,
                    )
                    fallback_kwargs = app_config.openai.build_completion_kwargs(
                        temperature=0.7,
                        max_tokens=retry_max_tokens,
                        include_reasoning=False,
                    )
                    content, last_finish_reason = await self._stream_completion(
                        client,
                        messages,
                        fallback_kwargs,
                        token_bus,
                    )
                # Auto-continue if the LLM hit max_tokens — keeps streaming
                # tokens to the same bus so the UI shows uninterrupted output.
                continues = 0
                while (
                    last_finish_reason in ("length", "max_tokens")
                    and continues < _WRITER_MAX_CONTINUATIONS
                    and len(content) > 50  # skip retry when thinking consumed all tokens
                ):
                    continues += 1
                    logger.warning(
                        "[WriterAgent] Streaming truncated at %d chars "
                        "(finish_reason=%s) — auto-continuing (attempt %d/%d)",
                        len(content),
                        last_finish_reason,
                        continues,
                        _WRITER_MAX_CONTINUATIONS,
                    )
                    cont_messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _CONTINUATION_INSTRUCTION},
                    ]
                    extra, last_finish_reason = await self._stream_completion(
                        client,
                        cont_messages,
                        completion_kwargs,
                        token_bus,
                    )
                    content += extra
            else:
                # Non-streaming path (revisions, non-webUI callers).
                response = await guarded_chat_completion(
                    client,
                    messages=messages,
                    **completion_kwargs,
                )
                content = extract_response_content(response, logger)

                finish_reason = None
                try:
                    finish_reason = response.choices[0].finish_reason
                except (AttributeError, IndexError):
                    pass

                # Auto-continue when the model hit max_tokens.
                continues = 0
                while (
                    finish_reason in ("length", "max_tokens")
                    and continues < _WRITER_MAX_CONTINUATIONS
                ):
                    continues += 1
                    logger.warning(
                        "[WriterAgent] Non-streaming output truncated at %d chars "
                        "(finish_reason=%s) — auto-continuing (attempt %d/%d)",
                        len(content),
                        finish_reason,
                        continues,
                        _WRITER_MAX_CONTINUATIONS,
                    )
                    cont_messages = messages + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _CONTINUATION_INSTRUCTION},
                    ]
                    cont_response = await guarded_chat_completion(
                        client,
                        messages=cont_messages,
                        **completion_kwargs,
                    )
                    content += extract_response_content(cont_response, logger)
                    try:
                        finish_reason = cont_response.choices[0].finish_reason
                    except (AttributeError, IndexError):
                        finish_reason = None

            # Strip any stray inline source markers the LLM may insert despite
            # the prompt prohibition — e.g. [✅ Da Knowledge Graph — ...] or
            # [📌 Da fonte esterna]. These consume tokens without adding value.
            content = re.sub(r"\[(?:✅|📌)[^\]]{0,200}\]", "", content)
            content = re.sub(r"\s*>[ \t]*(?:🔗|⚠️)[^\n]*", "", content)

            logger.info(f"[WriterAgent] Generated {intent} content ({len(content)} characters)")

            return content

        except Exception as e:
            logger.error(f"[WriterAgent] Generation failed: {e}")
            raise

    async def revise(
        self,
        current_draft: str,
        critique: str,
        revision_instructions: str,
        language: str = "it",
        intent: str = "lesson_creation",
        teacher_provided_context: Optional[str] = None,
        material_context_items: Optional[list[dict[str, Any]]] = None,
        material_maps: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """
        Revise content based on critic feedback.

        Args:
            current_draft: Current content draft
            critique: Critic's feedback
            revision_instructions: Specific revision instructions
            language: Output language
            intent: Query intent for selecting the right system prompt
            teacher_provided_context: The same uploaded-material appendix
                ``write()`` receives. Without it a revision is produced from
                the draft and the critique alone, so any constraint the
                material carried — a syllabus, a class limitation, a passage
                the lesson had to quote — silently stops applying the moment
                the Critic asks for a change. Teachers experience that as the
                assistant forgetting their file.

        Returns:
            Revised content
        """
        logger.info(f"[WriterAgent] Revising {intent} content based on feedback...")

        client = self._get_client()

        user_prompt = WRITER_REVISION_TEMPLATE.format(
            current_draft=current_draft,
            critique=critique,
            revision_instructions=revision_instructions,
        )

        # Same appendix ``write()`` uses, so both paths tell the model the same
        # thing about the teacher's material rather than describing it twice in
        # two ways. A revision that lost the material would quietly rewrite the
        # lesson away from the document it was grounded in — the defect fixed in
        # `3ca2e0b`, and the passages must not reintroduce it.
        user_prompt += today_note()
        user_prompt += teacher_material_appendix(
            material_context_items,
            teacher_provided_context,
            phase="revise",
            material_maps=material_maps,
        )

        # Get intent-specific system prompt for consistent formatting
        system_prompt, _ = get_writer_prompts(intent)
        system_prompt = system_prompt.replace(
            "{language}", "Italian" if language == "it" else "English"
        )

        try:
            # max_tokens=8000: same rationale as ``write()`` — revisions can
            # be just as long as initial drafts (especially "expand activity X"
            # critiques that GROW the content). Keep the two paths in sync so
            # we don't truncate the revised version and pass quality regression.
            completion_kwargs = app_config.openai.build_completion_kwargs(
                temperature=0.5,
                max_tokens=_WRITER_MAX_TOKENS,
            )
            messages: list[Any] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = await guarded_chat_completion(
                client,
                messages=messages,
                **completion_kwargs,
            )

            revised_content = extract_response_content(response, logger)

            finish_reason = None
            try:
                finish_reason = response.choices[0].finish_reason
            except (AttributeError, IndexError):
                pass

            # Auto-continue when revision hits the token ceiling.
            continues = 0
            while (
                finish_reason in ("length", "max_tokens") and continues < _WRITER_MAX_CONTINUATIONS
            ):
                continues += 1
                logger.warning(
                    "[WriterAgent] Revision truncated at %d chars "
                    "(finish_reason=%s) — auto-continuing (attempt %d/%d)",
                    len(revised_content),
                    finish_reason,
                    continues,
                    _WRITER_MAX_CONTINUATIONS,
                )
                cont_messages = messages + [
                    {"role": "assistant", "content": revised_content},
                    {"role": "user", "content": _CONTINUATION_INSTRUCTION},
                ]
                cont_response = await guarded_chat_completion(
                    client,
                    messages=cont_messages,
                    **completion_kwargs,
                )
                revised_content += extract_response_content(cont_response, logger)
                try:
                    finish_reason = cont_response.choices[0].finish_reason
                except (AttributeError, IndexError):
                    finish_reason = None

            revised_content = re.sub(r"\[(?:✅|📌)[^\]]{0,200}\]", "", revised_content)
            revised_content = re.sub(r"\s*>[ \t]*(?:🔗|⚠️)[^\n]*", "", revised_content)

            logger.info(f"[WriterAgent] Revised content ({len(revised_content)} characters)")

            return revised_content

        except Exception as e:
            logger.error(f"[WriterAgent] Revision failed: {e}")
            raise

    def _format_nodes(self, nodes: list, max_nodes: int = 20) -> str:
        """Format retrieved nodes for the prompt"""
        if not nodes:
            return "No nodes retrieved"

        lines = []
        for node in nodes[:max_nodes]:
            name = node.get("name", "Unknown")
            labels = node.get("labels", [])
            desc = node.get("description", "")[:150]

            label_str = f" [{', '.join(labels)}]" if labels else ""
            lines.append(f"- **{name}**{label_str}")
            if desc:
                lines.append(f"  {desc}")

        return "\n".join(lines)

    def _format_recommendations(self, recommendations: list, max_recs: int = 10) -> str:
        """Format recommendations for the prompt"""
        if not recommendations:
            return "No specific recommendations"

        lines = []
        for rec in recommendations[:max_recs]:
            name = rec.get("name", "Unknown")
            confidence = rec.get("confidence", "MEDIUM")
            desc = rec.get("description", "")[:200]

            lines.append(f"### {name} (Confidence: {confidence})")
            if desc:
                lines.append(desc)
            lines.append("")

        return "\n".join(lines)

    def _format_media(self, curated_media: Optional[dict[str, Any]]) -> str:
        """
        Format curated media for inclusion in the prompt (Phase 2).

        This provides the LLM with available media resources to reference
        in the generated content.

        Args:
            curated_media: Dict with videos, resources, citations

        Returns:
            Formatted media context string (empty if no media)
        """
        if not curated_media:
            return ""

        lines = [
            "\n\n## 📚 Available Educational Media",
            "Use these curated resources to enrich your response. Include relevant links and citations.",
        ]

        # Videos
        videos = curated_media.get("videos", [])
        if videos:
            lines.append("\n### 🎥 Video Educativi Disponibili")
            for v in videos[:5]:
                title = v.get("title", "Video")
                url = v.get("url")
                search_query = v.get("search_query", "")
                duration = v.get("duration_hint", "")

                if url:
                    lines.append(f"- [{title}]({url})" + (f" ({duration})" if duration else ""))
                else:
                    lines.append(f'- **{title}** - Cerca su YouTube: "{search_query}"')

        # Images/Diagrams (descriptions for potential generation)
        images = curated_media.get("images", [])
        if images:
            lines.append("\n### 🖼️ Diagrammi/Immagini Suggeriti")
            lines.append("Puoi suggerire questi diagrammi nel contenuto:")
            for img in images[:3]:
                desc = img.get("description", "")
                img_type = img.get("type", "diagram")
                lines.append(f"- [{img_type.title()}] {desc}")

        # External Resources
        resources = curated_media.get("resources", [])
        if resources:
            lines.append("\n### 🔗 Risorse Educative")
            for r in resources[:5]:
                title = r.get("title", "Resource")
                url = r.get("url") or r.get("suggested_url")
                res_type = r.get("type", "educational")

                if url:
                    lines.append(f"- [{title}]({url}) ({res_type})")
                else:
                    lines.append(f"- {title} ({res_type})")

        # Academic Citations
        citations = curated_media.get("citations", [])
        if citations:
            lines.append("\n### 📖 Riferimenti Scientifici")
            lines.append("Includi questi riferimenti nella sezione Fonti:")
            for c in citations[:3]:
                authors = c.get("authors", [])
                authors_str = ", ".join(authors[:2])
                if len(authors) > 2:
                    authors_str += " et al."
                year = c.get("year", "")
                title = c.get("title", "")
                journal = c.get("journal", "")
                doi = c.get("doi")

                cite_line = f"- {authors_str}"
                if year:
                    cite_line += f" ({year})"
                cite_line += f". *{title}*"
                if journal:
                    cite_line += f". {journal}"
                if doi:
                    cite_line += f" [DOI: {doi}]"
                lines.append(cite_line)

        # Instruction for the LLM
        lines.append(
            "\n**Istruzioni:** Incorpora i link video e risorse pertinenti nel contenuto. "
        )
        lines.append("Cita i riferimenti scientifici nella sezione Fonti alla fine.")

        return "\n".join(lines)

    def write_sync(
        self,
        teacher_query: str,
        plan: RetrievalPlan,
        retrieval_result: RetrievalResult,
        language: str = "it",
        curated_media: Optional[dict[str, Any]] = None,
        external_resources: Optional[dict[str, Any]] = None,
        domain: str = "neuro",
    ) -> str:
        """Synchronous version of write()"""
        import asyncio

        return asyncio.run(
            self.write(
                teacher_query,
                plan,
                retrieval_result,
                language,
                curated_media,
                external_resources,
                domain,
            )
        )
