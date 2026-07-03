# [AI TEAM] - AGENTIC GRAPHRAG — Updated Task Description

**Last Updated:** May 13, 2026 AM *(latest — **#9.UX-5 hotfix + Tier 0 wording revision + #9.UX-7 polish tracked**: (i) pre-seed `final_state` with `initial_state` in both SSE loops so the static `domain` field survives the LangGraph state-diff merge and the banner/footer render "UDL" / "Neuro" instead of the "il dominio attivo" fallback (one-line fix × 2 loops in `service.py`); (ii) Tier 0 banner copy revised to *"La lezione si baserà su conoscenze didattiche generali del nostro assistente e fonti esterne verificate ed integrate"* — the hybrid retrieval path is in fact decoupled from `AIX_CORRECTIVE_RAG_ENABLED` and runs whenever the Planner sets `external_apis_needed=True`, so the promise is structurally true (live 2026-05-13 fotosintesi smoke confirmed `external_resources` populated with CR OFF); (iii) new `#9.UX-7 media-panel re-ranking for out-of-scope queries` tracked as future polish — the right-hand media tiles surface curated KG-side pedagogy on out-of-scope queries (e.g. fotosintesi on `udl`) while the writer correctly cites external photosynthesis sources inline; cosmetic mismatch, no impact on lesson correctness.)* *(Earlier — May 13, 2026 AM — **#9.UX-5 teacher-friendly coverage banner + domain-aware footer LANDED** (CR OFF only). When `AIX_CORRECTIVE_RAG_ENABLED=false` (production default), the retriever card now shows a 3-tier domain-aware coverage banner: sage "Ricerca completata" (≥5 KG nodes), amber "Copertura parziale" (1-4 nodes), slate-blue "Questo argomento non è presente nella base {domain}" (0 nodes). Footer replaced from internal `"Retriever · GraphRAG + curated media"` to `"Fonte: Knowledge graph della didattica {domain} + risorse multimediali curate"`. Domain labels: UDL → "UDL (pedagogia inclusiva)" / Neuro → "Neuro". Threshold tunable via `AIX_COVERAGE_HEALTHY_THRESHOLD` (default 5, clamped 1-50). 4 new unit tests (15 total). CR-ON branches byte-identical to pre-#9.UX-5. See #9.UX-5 entry under Subtask 9 for full details.)* *(Earlier — May 12, 2026 PM — CORE 3 + CORE 3.5 — **Subtask 17.c + #LAT-7 added** *(doc-only)*. The 2026-05-12 PM live smoke of *"crea una lezione su disturbi da deficit di attenzione"* (domain `udl`, 45 min profile) with `AIX_CORRECTIVE_RAG_ENABLED=false` (set after the 2026-05-11 PM ADHD smoke hit 514s due to a false-positive `ambiguous` grade triggering a 215s wasted retry) dropped the total wall-clock to **199.8s** — confirming the CR retry was the single largest cost — but the **single retrieval pass still took ~102s** for 11 sub-queries. Five new independent hotspots were surfaced that are NOT covered by #17.b / #LAT-2/3 (which are corrective-RAG-loop-specific): **(R6)** language-detect false-Italian on English planner queries → wastes ~25-30s/pass on no-op OpenRouter "translation" round-trips; **(R7)** negative-embedding cache miss → 12× repeated `Learner Variability` embedding lookups for ~3-5s wasted; **(R2)** profile-enrichment dedup → `['ADHD', 'Scienze', 'DSA', 'ADHD']` re-issues the planner's `ADHD` query; **(R3)** within-pass query dedup; **(R1)** grader sees raw retrieval dump instead of filtered recommendations — root cause of the 2026-05-11 PM false-positive retry, must land before re-enabling CR in production. Two changes in this doc update: **(A)** new **Subtask 17.c — Single-Pass Retriever Efficiency** in CORE 3 captures all five hotspots with sized acceptance criteria, env-gated R1 default-OFF for backward compat; **(B)** new **`#LAT-7`** row in CORE 3.5 promotes #17.c into the wave, with the latency-budget snapshot updated to show both the 2026-05-11 PM (CR ON, 514s) and 2026-05-12 PM (CR OFF, 199.8s) live observations side-by-side. Also: **`#LAT-2` scope reduced** to OpenAlex/yt-dlp only since Angelo's 2026-05-12 AM media-pool merge eliminates live Semantic Scholar calls (verified `kg_*_media_pool.json` ships pre-generated citations). Combined effect of #LAT-7 alone: ~102s single pass → ~55-65s, independent of any retry, streaming, or fan-out improvements. See "Subtask 17.c" and updated "CORE 3.5 — Latency & Performance Wave (Proposed)" sections for full reasoning + acceptance criteria + per-fix latency-win estimates.)* *(Earlier — May 11, 2026 PM — CORE 2 — **#9.UX-4 logged as DEFERRED** *(doc-only, no code change)*. A live smoke of *"crea una lezione sulla motivazione intrinseca"* on the `neuro` domain (2026-05-11 PM) surfaced two display-only bugs in the corrective-RAG card: **(B1)** `_compute_retrieval_outcome` treats KG-curated articles + OER as proof that "hybrid retrieval kicked in", so any *ambiguous/irrelevant* turn with curated media is mis-classified as BLUE `adapted_with_hybrid` instead of AMBER `limited_kg_only`; **(B2)** the BLUE branch in `chat_retriever_card.html` hardcodes `"pedagogia inclusiva (UDL)"` in its copy, which is wrong on the neuro domain. Neither affects the agent graph, persisted state, or any non-display surface — and the grader itself fired correctly on neuro (the fix is in the display layer, not the corrective-RAG node). **Bundled with P5.4 (Workspace re-skin)** because the second fix touches `chat_retriever_card.html`, which is on the re-skin queue anyway. See "Subtask 9 → #9.UX-4" for the full diagnosis, proposed one-line fix for B1, two copy options for B2, and acceptance criteria.)* *(Earlier — CORE 3 — **Latency story consolidated** *(doc-only, no code change)*. The 2026-05-10 PM live smoke of corrective-RAG with `AIX_CORRECTIVE_RAG_ENABLED=true` exposed four concrete latency hotspots that were not yet ticketed anywhere: *(1)* Semantic Scholar 429 backoff (~30s/attempt × 2 = ~60s wasted on hybrid runs), *(2)* corrective-RAG retry overhead (~33-65s per extra attempt), *(3)* duplicate/trivial rewrites that spend the full retry budget on near-identical queries, *(4)* writer LLM cold-block (~97-110s — already tracked under #6.6 P2 phase 3 but never sized end-to-end). Two changes landed in this doc: **(A)** new **Subtask 17.b — Production Latency Hotspots** in CORE 3 captures the four observed hotspots as concrete, sized acceptance items (default-OFF mitigations: Semantic Scholar circuit breaker, per-run external-API cache, skip-trivial-rewrite guard, `phase_durations_ms` telemetry); **(B)** new **CORE 3.5 — Latency & Performance Wave (Proposed)** consolidates the existing latency items (CORE 3 #17 semantic cache, #6.6 P2 phase 3 writer streaming, #18 model A/B latency) and the new #17.b mitigations into a single coherent wave with a measured-vs-target latency-budget table (observed P50: ~150s KG-covered / ~310s out-of-scope → target P50: <90s / <120s post-wave). Status PROPOSED — no ClickUp tickets created yet; promotion gated on **#11b** (Langfuse traces) being live so we measure before we cut. Writer-streaming explicitly *promoted* from "DEFERRED" to "PROPOSED FOR CORE 3.5 #LAT-1" with a back-pointer in the #6.6 P2 phase 3 row. See "Subtask 17.b" + "CORE 3.5 — Latency & Performance Wave (Proposed)" for the full breakdown.)* *(Earlier — CORE 2 — **#9.UX-2 + #9.UX-3 + rewrite-UX LANDED** — second live smoke (turn for "fotosintesi clorofiliana") exposed two follow-up issues despite #9.UX-1: (1) when the grader requested a retry, the chat thread STILL showed two cards because `grade_retrieval` runs once per attempt and emitted on each iteration; (2) the irrelevant grade rendered with red ✗ "NON RILEVANTE" styling even when the hybrid retrieval gracefully filled the gap with Wikipedia + papers + OER — looked like an error to the teacher, but no error occurred. **#9.UX-2** (single card across N attempts): the `grade_retrieval` SSE branch now defers its emit until `_grader_will_retry` returns False, so a turn with N attempts produces ONE retriever card with `retrieval_attempts == N` (rendered as "Tentativi: N/M" badge, only when N>1 to keep the happy path clean). **#9.UX-3** (outcome-driven explainability) — derives a single `retrieval_outcome` token (success/adapted_with_hybrid/limited_kg_only/grader_error) that drives the card's color, headline, and copy. The "fotosintesi" case now renders BLUE ℹ️ "Adattamento riuscito" with a structured "what came from KG (pedagogy) + what came from external sources (disciplinary content)" breakdown — explainability, not error. RED is reserved for genuine grader exceptions only. **Bonus rewrite UX**: the grader's rewritten query is surfaced as a labelled inline line ("Query riformulata al N° tentativo: ...") instead of a buried small text. 11 new unit tests in `tests/unit/test_retriever_payload_outcome.py` lock the four outcomes + Tentativi badge data shape + `_grader_will_retry` invariant. Backward-compat: flag-OFF and grade=relevant paths render byte-identically to pre-#9.UX-3. See "Subtask 9 → #9.UX-2", "→ #9.UX-3", and "→ Rewrite UX" for full reasoning + outcome matrix.)* *(Earlier today — CORE 2 — **#9.UX-1 single-retriever-card-per-turn fix LANDED** — first live smoke of corrective-RAG (turn 8 of `lesson_id=fed4810b-…`) showed the Retriever Agent card rendering twice in the chat thread: once with raw stats from the `retrieve` node (~13:16:31), then again ~5s later with the green "VALUTAZIONE RECUPERO · RILEVANTE" row from the `grade_retrieval` node (~13:16:36). Root cause was purely in the SSE streaming layer — both SSE loops in `service.py` emitted `kind="retriever"` twice per turn when corrective-RAG was ON, and `#chat-cards` uses `hx-swap=beforeend` (append, not replace). Fix (Option A): when `AIX_CORRECTIVE_RAG_ENABLED=true`, the `retrieve` branch defers its emit to `grade_retrieval`, which becomes the sole emitter; flag-OFF path unchanged. One retriever card per turn under both flag states. Backward-compatible. See "Subtask 9 → #9.UX-1" for full reasoning + code citation.* **#12b.3 Duration Precedence Fix LANDED** — surfaced in the post-#9/#11a smoke (turn 7 of a follow-up) where the lesson rendered "Durata: 45 minuti" while the educational profile sidebar said 60 min. Root cause: in multi-turn, the service-layer-augmented `teacher_query` carries the conversation history, the Planner faithfully extracts any duration mentioned anywhere in the blob, and the pre-#12b.3 reconciliation in `plan_node` only filled in the profile when the Planner extracted *nothing* — so a history-leaked "45 min" silently outranked the teacher's just-set 60 min sidebar. Fix: new optional `AgentState.raw_user_turn` carries the un-augmented current turn; service layer populates it; `plan_node` now applies a 3-tier precedence — explicit duration in current turn wins → else profile is authoritative → else fall through to whatever the Planner extracted. Backward-compat for legacy callers that don't populate `raw_user_turn` (sniffer falls back to `teacher_query`, which on a first-turn IS the raw turn). 13 unit tests added covering the full 5-row matrix + sniffer edge cases (Italian/English/hours phrasing, stray-digit false-positives like "UDL 2.0") + 2 backward-compat tests for legacy callers. `AIX_CORRECTIVE_RAG_ENABLED=true` and `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS=2` activated in `.env` to exercise #9 in the next live runs.)*

**Previous update — May 10, 2026 AM:** *(CORE 2 — **#9 Corrective RAG (Retrieval Grading)** and **#11a JSON Parse Hardening** code-complete and merged; default-OFF behind `AIX_CORRECTIVE_RAG_ENABLED` and default-`approve` for `AIX_CRITIC_PARSE_ERROR_BEHAVIOR`, so runtime is byte-identical to pre-#9/#11a until ops flips the flags; unit tests green for #11a (5-run no-fallback smoke + json_mode forwarding + revise opt-in), full integration / live-LLM smoke still TODO. CORE 3 — **Infra readiness DONE**: runtime SSL fix (Plan B — `pip-system-certs` patches Python's SSL to trust the Windows OS cert store, unblocked PyPI + OpenRouter today) and Python 3.13 upgrade path (Plan A — automation script `scripts/setup/upgrade_to_python313.ps1` ready, pre-flight green for all 8 critical wheels: gensim 4.4.0 / node2vec 0.5.0 / numpy 2.4.4 / scipy 1.17.1 / pandas 3.0.2 / scikit-learn 1.8.0 / lingua-language-detector 2.2.0 / reportlab 4.5.0 — execution gated until next maintenance window). Also added: `src/aix/core/connectivity_probe.py` (one-shot startup probe distinguishing TLS / DNS / 401 / timeout — solves the "Connection error." mystery) and 4 new env flags documented in `.env.example` (`AIX_CORRECTIVE_RAG_ENABLED`, `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS`, `AIX_CRITIC_PARSE_ERROR_BEHAVIOR`, `AIX_LLM_PROBE_ENABLED`).)*

**Previous update — May 9, 2026:** *(CORE 4 — `#15` split into `#15.a` (PostgresSaver migration — original scope, unchanged), `#15.b` (Conversation-Memory Hardening — production-readiness without new UX, promoted from the Point A investigation: persist `LessonMessage.checkpoint_id`, startup checkpointer-loaded smoke + INFO log, expose `AIX_CONVERSATION_WINDOW_TURNS` in admin, memory-usage telemetry to Langfuse, first-turn augmentation dedup), and `#15.c` (Conversation-Memory UX V2 — DEFERRED design-only: time-travel "Rigenera", "Modifica e rigenera", "Branca da qui", CLI session REPL — not shipped until pilot signal). Also: **CORE 2 #12b.2** cosmetic follow-up landed — added one sentence to both Writer system prompts to stop the LLM echoing the Mandatory-Constraints language as a parenthetical after the **Durata** line.)*

**Previous update — May 1, 2026:** *(CORE 2 #10 — Conversation Memory — phases 10.1-10.4 LANDED end-to-end: chat_input OOB-swap (no more dead-end UX), `AsyncSqliteSaver` checkpointer with graceful degradation, `lesson_messages` CQRS table + multi-turn `/run` with auto-detected `mode={new, follow_up}` and backfill for pre-#10.3 lessons, service-layer history augmentation, turn-based summary-buffer windowing. Manual `pip install langgraph-checkpoint-sqlite>=2.0,<3.0` required after pull; `Base.metadata.create_all` auto-creates the new table. Time-travel regenerate + CLI session REPL deferred to V2; #10.5 PostgresSaver migration tracked under CORE 4 #15 (1-line saver-class swap, byte-identical code path).)*

**Earlier update — April 27, 2026:** *CORE 5 #20 — MCP Tool Servers — ✅ DONE via Option A: 7 of 7 phases landed. Phase 7 closed with `MCP_Setup.md` Production-deployment + Live-integration-follow-up sections; ClickUp #20 flipped to ✅ DONE. The MCP server is regression-locked by a 19-test pytest suite, fully documented for stdio + Streamable HTTP onboarding, and ready for Angelo's manual GUI-client smoke (Cursor / Claude / Inspector — ~30 min, no code changes required).*
**Copy-paste the sections below into ClickUp**

> 📦 **Repo layout note:** Since `phase-3c-complete` (April 25, 2026), every importable module lives under `src/aix/`. References below to `src/aix/agent/...`, `src/aix/api/...`, `src/aix/domains/...`, etc. are the canonical post-reorg locations. See `docs/product/REPO_REORG_MIGRATION_GUIDE.md` for a one-page cheat sheet on the new layout. The implementation plan, dependencies, and effort estimates below are unchanged by the reorg.

---

## TASK DESCRIPTION (copy into ClickUp main description)

---

### [AI TEAM] - AGENTIC GRAPHRAG

**Objective:**
Extend the existing GraphRAG system with a multi-agent pipeline for intelligent, automated educational content generation (lesson plans, activities, definitions, etc.) using LangGraph. Deploy as a production-ready service with API endpoint, streaming, memory, guardrails, and observability.

---

### ✅ Completed Work

**1. Core Agentic Pipeline Implementation**
- Built a 4-agent orchestration system using LangGraph (Plan → Retrieve → Write → Critique)
- "Planner Agent": Analyzes teacher queries, extracts key concepts, generates search strategies
- "Retriever Agent": Executes multiple GraphRAG searches against Neo4j Knowledge Graph
- "Writer Agent": Generates educational content grounded in retrieved KG data
- "Critic Agent": Reviews content quality with scoring mechanism (1-5 scale), triggers revisions if needed
- Revision loop with configurable max iterations (default: 2 cycles)

**2. Intelligent Intent Detection**
- Planner Agent classifies queries into 7+ categories:
  lesson_creation, activity_design, assessment, unit_plan,
  definition, comparison, explanation, recommendation, list
- Scope detection: in_scope, partial_scope, out_of_scope relative to KG
- Writer & Critic agents adapt prompts and evaluation criteria based on detected intent
- No longer forces full lesson plan for simple queries (definitions, comparisons, etc.)

---

### 🔄 In Progress

**3. Streamlit UI Integration** *(code exists, needs end-to-end validation)*
- Added "mode toggle" in sidebar: "GraphRAG" vs "Agent (Lesson Planner)"
- Full backward compatibility with existing GraphRAG mode
- Agent mode displays: generated content, approval status, revision count, scores, KG sources
- Download options: TXT, MD, PDF
- **Remaining:** End-to-end validation, UX polish, media display testing

**4. Upsell Conversion Feature** *(plumbing exists, needs validation)*
- key_concepts and curated_media passed through AgentState for upsell buttons
- Intent-aware upsell suggestions in Streamlit
- **Remaining:** End-to-end testing of upsell flow, verify auto-formulation of follow-up queries

**5. Quality Assurance System (Critic Agent)** *(code exists, needs domain enrichment + testing)*
- Critic Agent scores content on multiple criteria: Structure, Evidence Grounding, Pedagogical Soundness
- Auto revision loop (max 2 cycles): if average score < 3.5 or any criterion < 2
- Intent-specific evaluation criteria (lesson vs definition vs comparison)
- Auto-approve after max_revisions to prevent infinite loops
- **Remaining:** UDL evaluation criteria thin (~25 lines vs Neuro ~85 lines), Critic disconnected from rich domain configs, end-to-end revision loop not validated

---

### 📝 Code Written But Not Validated

**Media Enrichment Pipeline** *(7 files, ~3,700 lines — not end-to-end tested)*
- MediaLookup: Loads curated media from sidecar JSON (`kg_{domain}_media_mapping.json`)
- ExternalMediaAPI: Real-time search across YouTube, Wikipedia, Semantic Scholar, OER sources (1,054 lines)
- MermaidGenerator: LLM generates Mermaid.js diagrams, rendered via mermaid.ink (510 lines)
- ImageGenerator: DALL-E 3 educational diagram generation with disk cache (442 lines)
- DiagramFactory: Routes diagram requests to Mermaid, DALL-E, or Canva (371 lines)
- ResourceLookup: Static educational resource filtering by topic/audience (447 lines)
- **Issues:** UDL has no media JSON, Neuro only 20/695 concepts mapped, `diagram_factory.py` has DALL-E method name bug, External APIs not tested with real keys

**Neuro Media Mapping** *(partial — 20/695 concepts = 2.9% coverage)*
- `scripts/ml/generate_media_mapping.py`: Script to generate curated media JSON from KG concepts via GPT-4o
- Supports batched async processing with rate limiting
- Only 20 concepts mapped due to `--limit` during initial test run

**Domain Prompt Extensions** *(Neuro full, UDL skeleton)*
- Neuro domain: Full prompt extensions (~85 lines) covering neurodidactic principles
- UDL domain: Basic prompt extensions (~25 lines) — needs enrichment
- Domain-aware prompt routing for Writer and Critic agents
- **Critical gap:** Agent Writer/Critic completely disconnected from rich `src/aix/domains/udl_domain.py` (200+ lines) and `src/aix/domains/neuro_domain.py` configs. Design doc with 3 options exists, none implemented.

**CLI Test Harness**
- `apps/cli/run_agent.py`: Interactive CLI for testing agent pipeline with custom queries (Phase 3B: was `test_agent.py` at repo root)
- `--query` flag for single-run mode, interactive loop for exploration

---

### 📁 Files Modified/Created (30 files under `src/aix/agent/`)

**Core Pipeline:**
- `src/aix/agent/orchestrator.py` | Main entry point, simplified API (356 lines)
- `src/aix/agent/graph/lesson_planner_graph.py` | LangGraph state machine (186 lines)
- `src/aix/agent/graph/nodes.py` | Pipeline node definitions (386 lines)
- `src/aix/agent/graph/state.py` | State types + enums (213 lines)

**Agents:**
- `src/aix/agent/agents/planner_agent.py` | Query analysis + intent + scope detection (189 lines)
- `src/aix/agent/agents/retriever_agent.py` | Multi-search + media + external APIs (619 lines)
- `src/aix/agent/agents/writer_agent.py` | Adaptive content generation (415 lines)
- `src/aix/agent/agents/critic_agent.py` | Review + quality scoring (224 lines)
- `src/aix/agent/agents/graph_updater_agent.py` | Phase 3 placeholder (96 lines)

**Tools:**
- `src/aix/agent/tools/graphrag_tool.py` | GraphRAG wrapper for agents (258 lines)
- `src/aix/agent/tools/curriculum_tool.py` | Phase 3 placeholder (180 lines)

**Prompts:**
- `src/aix/agent/prompts/planner_prompt.py` | Planner system + user prompts (295 lines)
- `src/aix/agent/prompts/writer_prompt.py` | Intent-specific writer prompts (725 lines)
- `src/aix/agent/prompts/critic_prompt.py` | Evaluation criteria prompts (245 lines)
- `src/aix/agent/prompts/templates/lesson_template.txt` | Italian lesson template (94 lines — not imported)
- `src/aix/agent/configs/domain_prompts.py` | Neuro/UDL prompt extensions (232 lines)

**Media Layer:**
- `src/aix/agent/media/external_apis.py` | YouTube, Wikipedia, Semantic Scholar, OER (1,054 lines)
- `src/aix/agent/media/media_lookup.py` | Sidecar JSON media loading (419 lines)
- `src/aix/agent/media/mermaid_generator.py` | LLM → Mermaid diagrams (510 lines)
- `src/aix/agent/media/image_generator.py` | DALL-E 3 diagrams (442 lines)
- `src/aix/agent/media/diagram_factory.py` | Routing to Mermaid/DALL-E/Canva (371 lines)
- `src/aix/agent/media/resource_lookup.py` | Static resource lookup (447 lines)
- `src/aix/agent/media/canva_generator.py` | Phase 5 placeholder (238 lines)

**Other (entry points & data, NOT under `src/aix/`):**
- `apps/streamlit/main.py` | UI with mode toggle + agent mode (Phase 3B: moved out of repo root)
- `apps/cli/run_agent.py` | CLI test harness (Phase 3B: was `test_agent.py` at repo root)
- `scripts/ml/generate_media_mapping.py` | Media mapping generator (488 lines)
- `data/media/kg_neuro_media_mapping.json` | Curated media for 20 Neuro concepts (Phase 3B: moved from repo root)

**Total: ~7,500+ lines | 24 working files | 4 stubs/placeholders | 2 known bugs**

---

### 🔧 Technical Stack

- **LangGraph** for multi-agent orchestration
- **OpenAI GPT-4o** for LLM reasoning (upgrade to GPT-5.x planned — see doc)
- **Neo4j** Knowledge Graph (KG_NEURO + KG_UDL data)
- **Streamlit** for visual UI
- **AsyncIO** for concurrent operations
- **aiohttp** for external API calls
- **FastAPI** for REST API (GraphRAG mode only — Agent endpoint pending)

---

## SUBTASKS LIST (create these in ClickUp)

Subtasks are organized by **CORE epic** (matching the existing ClickUp `[AI TEAM] - AGENTIC GRAPHRAG CORE N` task structure) and ordered by dependency within each core: if subtask B depends on subtask A, then A comes first. Within the same core, independent subtasks can be done in parallel.

---

### CORE 0 — Legacy / Pre-existing Work (update status in ClickUp)

This is what already lives in the original `[AI TEAM] - AGENTIC GRAPHRAG CORE 0` ClickUp task. Two items are DONE, two remain IN PROGRESS. **E5 has been moved to CORE 1** because its remaining work (UDL Critic criteria + e2e revision loop validation) only becomes possible after Subtask #2 lands.

| # | Subtask Name | Assignee | Status | Notes |
|---|---|---|---|---|
| E1 | **Core Agentic Pipeline Implementation** | LM | ✅ DONE | 4 agents, LangGraph, revision loop |
| E2 | **Intelligent Intent Detection** | LM | ✅ DONE | 7+ intents, scope detection |
| E3 | **Streamlit UI Integration** | AG, LM | 🔵 IN PROGRESS | Code exists, needs e2e validation + polish |
| E4 | **Upsell Conversion Feature** | LM, AG | 🔵 IN PROGRESS | Plumbing exists, needs e2e validation |
| ~~E5~~ | ~~Quality Assurance System~~ | ➡️ **Moved to CORE 1** | — | Renamed and re-scoped. Depends on #2 (Agent ↔ Domain Config) + #2.5 (Educational Profile Schema). |

---

### CORE 1 — Agentic Foundations (Target: February 2026)

**Theme:** Bug-fix wave + domain enrichment + per-request profile + media data.
**Principle:** Unblock everything else. Mostly zero-dep work that finishes the foundation laid in CORE 0.
**Deliverable:** *"Agent pipeline runs end-to-end with rich domain prompts, full media coverage for both domains, and per-class personalization input."*

```
Dependency graph:

  #1   Bug Fixes (DALL-E, duplicate tool, template)        ← no deps, do first (quick wins)
  #2   Agent ↔ Domain Config Integration                    ← no deps
  #2.5 Educational Profile Schema Integration               ← no deps (port from fem/enhanced-variables-extraction)
  E5   Quality Assurance System (Critic UDL + e2e)          ← depends on #2 + #2.5
  #3   UDL Media Mapping (fix script + generate)            ← no deps
  #4   Neuro Media Mapping Expansion                        ← no deps
  #5   Validate External APIs e2e                           ← no deps
  #6   Validate Media Layer e2e                             ← depends on #1, #3, #4, #5
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 1 | **Bug Fixes: DALL-E method + duplicate CurriculumTool + unused template** | LM | 🟠 High | 1h | None | ✅ DONE |
| 2 | **Agent ↔ Domain Config Integration** | LM | 🔴 Urgent | 1-4h | None | ✅ DONE (Option 2 only; Option 3 deferred) |
| **2.5** | **Educational Profile Schema Integration** ⭐ NEW | LM (port) / AG (review) | 🔴 Urgent | 3-4h | None | ✅ **DONE — port phase** (schema landed, wired through `ContextRequest` + `AgentState`; deeper integration items remain on #2) |
| **E5** | **Quality Assurance System** (re-scoped, was in CORE 0) | AG, LM | 🟠 High | 2h | #2, #2.5 | TODO |
| 3 | **UDL Media Mapping — Fix Script + Generate JSON** | LM | 🟠 High | 2h | None | ✅ DONE |
| 4 | **Neuro Media Mapping — Full Expansion (695 concepts)** | LM/AG | 🟠 High | 1h | None | ✅ DONE |
| 5 | **Validate External APIs end-to-end** | LM | 🟠 High | 2h | None | ✅ DONE |
| 6 | **Validate Media Layer end-to-end** | LM | 🟠 High | 2h | #1, #3, #4, #5 | ✅ DONE |

**CORE 1 total effort:** ~14-18h (~2-3 days)

---

### CORE 2 — Production-Readiness: API + Safety + Observability (Target: March 2026)

**Theme:** Make the agent usable by the frontend and safe in production.
**Principle:** Everything Lovable / FEM main app / future frontend needs to integrate.
**Deliverable:** *"Production-ready agent API consumable by an external frontend, with streaming, safety, observability, and corrective retrieval."*

```
Dependency graph:

  #6.5 Frontend Platform Evaluation & Decision (Spike)        ← no deps  ✅ DONE
  #6.6 Path C Webui Skeleton + Agent In-Process Integration   ← depends on #6.5 + CORE 1
  #7   FastAPI JSON+SSE Agent Endpoint (public contract)      ← depends on #6.6 (extracts service layer)
  #8   Guardrails: Input/Output Validation                    ← depends on #6.6 (input schema known from real UI usage)
  #11  Observability (Agent JSON Parse Hardening + Tracing)   ← 11a depends on #6.6; 11b depends on 11a; #11 must precede CORE 3 #18
  #9   Corrective RAG (Retrieval Grading)                     ← no deps
  #10  Conversation Memory (LangGraph Checkpointer)           ← no deps (unblocks CORE 4 #15.a, #15.b, #19)
  #12  SSE Streaming Hardening (backpressure + reconnect)     ← depends on #6.6 (already partial), finalised here
  #12b Educational Profile → Writer Prompt Adherence          ← depends on #6.6 (profile form exists)
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| **6.5** | **Frontend Platform Evaluation & Decision (Research Spike)** | LM (+ Diego, Simone input) | 🔴 Urgent | 4-6h | None | ✅ DONE — see `docs/architecture/Frontend_Platform_Evaluation.md` |
| **6.6** | **Path C Webui Skeleton + Agent In-Process Integration** ⭐ NEW | LM | 🔴 Urgent | 8-10h | #6.5, CORE 1 | 🟡 IN PROGRESS *(P0–P3 ✅ DONE 2026-04-26; P4/P5/P6 TODO — see phase table in Subtask 6.6 section below)* |
| 7 | **FastAPI JSON+SSE Agent Endpoint (public contract)** | LM | 🔴 Urgent | 4-6h | #6.6 | ✅ DONE *(2026-04-26)* — `POST /api/v1/agent/run` + `POST /api/v1/agent/stream` mounted; JWT Bearer transport added in parallel to the cookie backend (zero webui regression); Minimal / Rich examples dropdown in Swagger UI mirrors `/api/v1/context`; 7 contract tests green; OpenAPI inventory strictly additive |
| 8 | **Guardrails: Input/Output Validation** | LM | 🔴 Urgent | 3-5h | #6.6 | TODO |
| 11 | **Observability (Agent JSON Parse Hardening + LangSmith/Langfuse)** | LM, AG | 🟠 High | 4-6h | #6.6 (for 11a repro) | 🟡 IN PROGRESS — **11a** ✅ DONE *(code) 2026-05-10* (`json_mode=True` forwarded on Planner + Critic via `build_completion_kwargs`; structured `event=agent_parse_error` log with `agent` + `raw_response_preview`; Critic fallback now configurable via `AIX_CRITIC_PARSE_ERROR_BEHAVIOR={approve|revise|raise}`, default `approve` = legacy behaviour; Planner keeps backward-compatible degraded `RetrievalPlan`; **unit tests green** — `tests/unit/test_json_parse_hardening.py` covers json_mode forwarding + 5-run no-fallback smoke + revise opt-in; live-LLM smoke against OpenRouter still TODO). **11b** (Langfuse) foundation **LANDED** by AG (Langfuse client + prompt management — `aix.domains.langfuse_prompts`, `scripts/ops/seed_langfuse_prompts.py`, `langfuse>=2.0.0`); 11b dashboard verification + cost tracking + sanity check still TODO. |
| 9 | **Corrective RAG (Retrieval Grading)** | LM | 🔴 Urgent | 3-4h | None | ✅ DONE *(code) — 2026-05-10* — `RetrievalGraderAgent` (cheap LLM, json_mode, fallback `grade="relevant"`); new `grade_retrieval_node` + `should_retry_retrieval` router in `nodes.py`; topology change in `lesson_planner_graph.py` gated by `AIX_CORRECTIVE_RAG_ENABLED` (default `false` → byte-identical to pre-#9); `AgentState` extended with 5 fields (`retrieval_grade`, `retrieval_grade_reason`, `retrieval_attempts`, `retrieval_rewritten_query`, `retrieval_warning`), all initialised to `None`; writer prompt adds a low-confidence caveat *only* when `retrieval_warning=True`; SSE `retriever` payload + `chat_retriever_card.html` render a "Valutazione recupero" row when grading ran; retry budget tunable via `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS` (default 2, clamped 1-4). **Not yet exercised end-to-end** — flag stays OFF until #11b traces are in place to measure quality delta. **#9.UX-1 / UX-2 / UX-3 / Rewrite-UX follow-ups** ✅ DONE *(2026-05-10 PM)* — single retriever card per turn (UX-1: defer `retrieve` emit when corrective-RAG ON; UX-2: defer `grade_retrieval` emit on intermediate retry attempts; UX-2 also surfaces a `Tentativi: N/M` badge when N>1); outcome-driven explainability messaging (UX-3: 4 outcomes — success/adapted_with_hybrid/limited_kg_only/grader_error — drive color + headline + copy; RED reserved for genuine grader exceptions only); Rewrite UX (grader's reformulated query surfaced as a labelled inline line "Query riformulata al N° tentativo: …"). 11 unit tests in `tests/unit/test_retriever_payload_outcome.py` lock the four outcomes + Tentativi badge data + `_grader_will_retry` invariant. See "Subtask 9 → #9.UX-1/UX-2/UX-3/Rewrite-UX". **#9.UX-5 teacher-friendly coverage banner + domain-aware footer (CR OFF only)** ✅ DONE *(2026-05-13 AM)* — 3-tier coverage banner when `AIX_CORRECTIVE_RAG_ENABLED=false` (sage ≥5 / amber 1-4 / info 0 nodes); domain-aware footer `"Fonte: Knowledge graph della didattica {domain} + risorse multimediali curate"`; 4 new unit tests (15 total). See "Subtask 9 → #9.UX-5". |
| 10 | **Conversation Memory (LangGraph Checkpointer + Multi-Turn Chat)** | LM | 🔴 Urgent | 5-7d | None | 🟢 LANDED 10.1-10.4 *(2026-05-01; regenerate via time-travel + CLI session REPL deferred to V2 → tracked under CORE 4 #15.c; #10.5 PostgresSaver migration tracked under CORE 4 #15.a; no-UX server-side hardening promoted to CORE 4 #15.b on 2026-05-09)* |
| 12 | **SSE Streaming Hardening (backpressure + reconnect)** | LM | 🟠 High | 2-3h | #6.6 | TODO |
| 12b | **Educational Profile → Writer Prompt Adherence (Duration + Profile in Hybrid)** | LM | 🔴 Urgent | 1-2h | #6.6 | ✅ DONE *(2026-05-09; #12b.3 follow-up landed 2026-05-10 PM)* — **#12b.3 multi-turn duration precedence fix *(2026-05-10 PM)*:** after #9/#11a smoke on turn 7 of a follow-up chat, the lesson rendered "Durata: 45 minuti" while the educational-profile sidebar showed 60 min. Root cause was in the interaction between **#10 (conversation history augmentation)** and **#12b.1 (educational-profile duration fill-in)** — the service layer's `_augment_query_with_history` glues prior turns into `teacher_query`, the Planner extracts any duration mentioned anywhere in that blob (its prompt instructs *"Time Constraints: Duration if mentioned"*), and the pre-#12b.3 reconciliation in `plan_node` only filled in the profile when the Planner extracted *nothing*. Net: a previous turn's "45 min" silently outranked the teacher's just-set 60 min sidebar. **Fix (3 changes, fully backward-compatible):** 1️⃣ new optional `AgentState.raw_user_turn` field (TypedDict total=False, default `None`) carrying the un-augmented current turn; 2️⃣ both service entry points (`run_agent_stream` for the webui, `stream_agent_events` for the public JSON+SSE API) populate it via the new `create_initial_state(raw_user_turn=…)` kwarg; 3️⃣ `plan_node` now applies a 3-tier precedence — *explicit duration in current turn wins → else profile is authoritative (overrides any Planner extraction from history) → else fall through to whatever the Planner extracted (no profile to defer to)*. New module-level helper `_current_turn_mentions_duration(state)` uses a word-boundary-anchored regex (`\b\d+\s*(?:min(?:ut[oi]|utes?)?|h(?:rs?)?|hours?|ore?)\b`) so stray digits in concept names ("UDL 2.0", "WCAG 2.1") never false-positive. Sniffer falls back to `teacher_query` when `raw_user_turn` is absent — keeps legacy callers byte-identical (on a first-turn run with no history, `teacher_query` IS the raw turn). **Behaviour matrix:** `(profile=60, current=∅, history=∅) → 60` ✅; `(profile=60, current=∅, history="45") → 60 ✅` ← the bug; `(profile=60, current="30", history="45") → 30 ✅` (current wins); `(profile=∅, current=∅, history="45") → 45` (no profile to defer to); `(profile=∅, current="30", history=∅) → 30`. **13 unit tests** in `tests/unit/test_plan_node_profile_wins_over_history.py` cover the full matrix + sniffer edge cases (IT/EN, hours, stray-digit false-positives) + 2 backward-compat tests for legacy callers that omit `raw_user_turn`. **Files changed:** `src/aix/agent/graph/state.py` (+`raw_user_turn` field + kwarg in `create_initial_state`), `src/aix/agent/graph/nodes.py` (+sniffer helper, replaced single-branch fill-in with 3-tier precedence), `src/aix/webui/agent/service.py` (×2 — both `create_initial_state` callsites pass `raw_user_turn`), new `tests/unit/test_plan_node_profile_wins_over_history.py`. **Zero regression risk:** the only behaviour change happens when (a) the educational profile has `time_available_minutes` set AND (b) the current raw turn has no duration AND (c) the Planner extracted a duration (necessarily from history). All other paths take the same branch they took pre-#12b.3. **#12b.2 cosmetic follow-up *(2026-05-09 PM)*:** after #12b.1 the LLM started paraphrasing the new "Mandatory Constraints" prompt section into the user-facing output as a parenthetical (e.g. *"Durata: 60 minuti (vincolo rigido — somma esatta di tutte le fasi)"*, *"Durata: 45 minuti (come da profilo docente — vincolo rigido)"*). The constraint was being respected correctly — only the literal echo was unwanted. Added one sentence to both `WRITER_SYSTEM_PROMPT_LESSON` and `WRITER_SYSTEM_PROMPT_HYBRID` Mandatory-Constraints blocks: *"Apply the Duration silently. Do NOT echo this constraint as parenthetical or explanatory text in the rendered lesson."* with the two observed phrasings cited as negative examples. Zero risk of weakening enforcement (the HARD-constraint language above the new line is unchanged). **Bug:** lesson_duration from the educat **#12b.2 cosmetic follow-up *(2026-05-09 PM)*:** after #12b.1 the LLM started paraphrasing the new "Mandatory Constraints" prompt section into the user-facing output as a parenthetical (e.g. *"Durata: 60 minuti (vincolo rigido — somma esatta di tutte le fasi)"*, *"Durata: 45 minuti (come da profilo docente — vincolo rigido)"*). The constraint was being respected correctly — only the literal echo was unwanted. Added one sentence to both `WRITER_SYSTEM_PROMPT_LESSON` and `WRITER_SYSTEM_PROMPT_HYBRID` Mandatory-Constraints blocks: *"Apply the Duration silently. Do NOT echo this constraint as parenthetical or explanatory text in the rendered lesson."* with the two observed phrasings cited as negative examples. Zero risk of weakening enforcement (the HARD-constraint language above the new line is unchanged). **Bug:** lesson_duration from the educational profile sidebar was ignored in both paths: (a) standard path had misaligned signals (Profile Duration vs Requirements Time Constraints = "Not specified"); (b) hybrid template (`WRITER_USER_TEMPLATE_HYBRID`) lacked the `{educational_profile_section}` placeholder entirely. **Fix (3 changes, backward-compatible):** 1️⃣ `plan_node` now fills `plan.time_constraints` from the educational profile when the Planner doesn't extract one from free-text (nodes.py); 2️⃣ HYBRID template now includes `{educational_profile_section}` and receives it from `writer_agent.py`; 3️⃣ Both writer system prompts (`LESSON` + `HYBRID`) now have a "Mandatory Constraints" section making Duration a HARD requirement for the LLM. **Zero breakage:** guard checks `if not plan.time_constraints` so explicit query durations still win; empty profiles produce an empty string placeholder. **#12b.1 follow-up fix *(2026-05-09 PM)*:** First pass still produced 90-min lessons (3 user tests: ADHD 60min → 90min, fotosintesi 45min → 90min, Storia 60min → 90min). Root cause was a **key mismatch**: the canonical schema field is `time_available_minutes` (top-level) but `nodes.py:144` and `writer_agent.py:147` were both reading `ep.get("lesson_duration")` — which returns `None`, so #12b's plan-node back-fill never fired and the writer's profile section never rendered the Duration line, leaving the Mandatory-Constraints paragraph (Fix 3) with nothing to bind to. While here, also corrected `writer_agent.py:142` which read top-level `ep.get("disabilities")` instead of nested `ep["group"]["disabilities"]` (so "Learner needs:" never rendered either; disability flow only reached the retriever's enriched search queries, not the writer prompt). Both keys now read from the canonical paths with legacy keys as defensive fallbacks (`ep.get("time_available_minutes") or ep.get("lesson_duration")`). |

**CORE 2 total effort:** ~6-9 working days *(was ~4-5.5d; #10 grew from 3-5h to 5-7d on 2026-05-01 after #6.6 P2 phase 2 + post-completion smoke surfaced the full multi-turn scope: chat_input OOB fix + `AsyncSqliteSaver` checkpointer + `lesson_messages` CQRS table + multi-turn `/run` semantics + summary-buffer windowing. Already DONE: ~6h on #6.5, ~10h on #6.6 P2 phase 2, ~5h on #7. **#11 status (2026-05-01):** 11a (JSON parse silent-fallthrough fix) still TODO — a defensive `_extract_json` multi-strategy helper does live in both `planner_agent.py` and `critic_agent.py`, but the real fix (typed `AgentError` event on parse failure instead of silent fallback to hardcoded defaults like Critic's `"Approved due to parsing error"` + `response_format={"type":"json_object"}` + 5-run smoke test) is not yet in. 11b foundation **LANDED by AG** (Langfuse client + prompt management — `aix.domains.langfuse_prompts`, `scripts/ops/seed_langfuse_prompts.py`, `langfuse>=2.0.0`); tracing-dashboard verification + cost tracking + sanity-check pass against real (non-fallback) Critic critiques still TODO.)*

---

### CORE 3 — Quality & Cost: Advanced RAG Techniques (Target: April 2026)

**Theme:** Push answer quality higher and cost lower.
**Principle:** All depend on CORE 2 being live so we can measure improvements via traces.
**Deliverable:** *"Higher-quality, faster, cheaper, fully-cited answers."*

> **Infra readiness — ✅ DONE *(2026-05-10)* — Runtime SSL fix + Python 3.13 upgrade path.**
> Pre-requisite for everything in this wave: every CORE 3 ticket exercises live LLM traffic, and we hit a hard environment block on 2026-05-10 — Python 3.11.0 on Windows shipped with OpenSSL 1.1.1q (EOL Sep 2023) and an outdated CA bundle, which broke TLS to OpenRouter, OpenAI, GlitchTip, and PyPI with `[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate`. Two mitigations landed on the same day:
> * **Plan B (immediate, applied):** `python -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org pip-system-certs` — patches Python's SSL to trust the Windows OS certificate store. Connectivity probe now returns `status=200` against OpenRouter.
> * **Plan A (script ready, execution gated):** `scripts/setup/upgrade_to_python313.ps1` — automation for upgrading the project to Python 3.13.x (which ships with a modern OpenSSL 3.x and a current CA bundle). Creates `venv-313` side-by-side with the existing `venv` (zero-risk, reversible), reinstalls `requirements*.txt`, runs unit tests, and runs the connectivity probe. **Pre-flight all-green** for the 8 critical wheels: `gensim 4.4.0` (cp313 wheel), `node2vec 0.5.0` (py3-none-any), `numpy 2.4.4` (cp313 wheel), `scipy 1.17.1` (cp313 wheel), `pandas 3.0.2` (cp313 wheel), `scikit-learn 1.8.0` (cp313 wheel), `lingua-language-detector 2.2.0` (cp313 wheel), `reportlab 4.5.0` (py3-none-any) — no Visual Studio Build Tools detour needed. Execution deferred to next maintenance window.
> * **Diagnostic plumbing landed alongside:** `src/aix/core/connectivity_probe.py` runs a one-shot `GET /models` against the configured `LLM` base_url at FastAPI startup and emits ONE actionable log line that distinguishes TLS failure / DNS / connect-timeout / read-timeout / 401 / 403 / 4xx / 5xx — replacing the catch-all "Connection error." that hid three different failure modes inside the OpenAI SDK's retry layer. Default-on; flip `AIX_LLM_PROBE_ENABLED=false` to disable.

```
Dependency graph:

  #13   Query Decomposition (Multi-Hop Reasoning)             ← needs CORE 2 #11 (traces)
  #14   Citation Grounding & Source Attribution                ← needs CORE 2 #11
  #17   Semantic Caching (GraphRAG + Agent layers)             ← needs CORE 2 #11
  #17.b Production Latency Hotspots (Live-Run Findings)        ← needs CORE 2 #11b (NEW)
  #18   Model Upgrade & Evaluation phase (Claude/GPT-5.x/Gemini A/B)  ← needs CORE 2 #11
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 13 | **Query Decomposition (Multi-Hop Reasoning)** | LM | 🟠 High | 4-5h | #11 | TODO |
| 14 | **Citation Grounding & Source Attribution** | LM | 🟡 Medium | 3h | #11 | TODO |
| 17 | **Semantic Caching (GraphRAG + Agent layers)** | LM | 🟡 Medium | 4-8h | #11 | TODO |
| 17.b | **Production Latency Hotspots — Live-Run Findings** | LM | 🟠 High | 4-6h | #11b | TODO *(NEW 2026-05-10 PM — promoted from corrective-RAG live smoke; Semantic Scholar 429 + retry overhead + trivial-rewrite guard + phase-duration telemetry; cross-listed under proposed CORE 3.5 #LAT-2/3)* |
| 18 | **Model Upgrade & Evaluation Phase (A/B test Claude / GPT-5.x / Gemini)** | LM | 🟡 Medium | 4-6h | #11 | TODO |

**CORE 3 total effort:** ~19-28h (~2.5-3.5 days) *(was ~15-22h; +4-6h for #17.b)*

---

### CORE 3.5 — Latency & Performance Wave (Proposed) (Target: April-May 2026, post-#11b)

**Theme:** Make the agent feel fast.
**Principle:** All depend on **#11b (Langfuse traces)** being live — *we measure before we cut*. Two of the candidate subtasks already exist elsewhere in this doc and are **promoted** into this wave rather than re-invented:
- **#6.6 P2 phase 3 (Writer Token Streaming) — DEFERRED** *(see L658 phase table; the §6.6 acceptance bullet "sees Writer tokens streaming")* — first user-perceived latency item ever recorded; promoted to **`#LAT-1`** below.
- **CORE 3 #17 (Semantic Caching, Layer 1+2) — TODO** — cross-listed in CORE 3 *and* CORE 3.5 since cache hits are simultaneously a quality and a latency win. Not double-counted in effort totals.
- **CORE 3 #17.b (Production Latency Hotspots) — TODO (NEW)** — the four cheap mitigations promoted into `#LAT-2` and `#LAT-3` below.
- **CORE 3 #17.c (Single-Pass Retriever Efficiency) — TODO (NEW 2026-05-12 PM)** — five guards (R1, R2, R3, R6, R7) from the post-Corrective-RAG-OFF ADHD smoke, promoted into `#LAT-7` below.

**Status:** **PROPOSED — design-only, no ClickUp tickets created yet.** Same convention as **CORE 4 #15.c** (deferred design-only) and **CORE 6** (future placeholder). Captured here so the four NEW items live next to their siblings rather than scattered across follow-up notes. Promotion to ClickUp gated on **#11b** traces being live (we can't size the wins until we can read them off a Langfuse dashboard).

**Deliverable:** *"Lessons feel responsive: P50 end-to-end < 90s for KG-covered queries, P50 < 120s for hybrid (out-of-scope) queries, with progressive UX during the wait. P95 capped at < 180s / < 240s respectively."*

```
Dependency graph:

  #LAT-1  Writer Token Streaming (= #6.6 P2 phase 3)              ← needs #11b
  #LAT-2  External-API Resilience (OpenAlex + yt-dlp + cache)     ← needs #11b
  #LAT-3  Corrective-RAG Retry Hardening (skip trivial rewrites)  ← needs #11b, #9
  #LAT-4  Parallel Retrieval Fan-out (KG + external concurrent)   ← needs #11b
  #LAT-5  Semantic Cache Layer 1+2 (= CORE 3 #17, cross-listed)   ← needs #11, #11b
  #LAT-6  End-to-End Latency Budget Dashboard (Langfuse view)     ← needs #11b
  #LAT-7  Single-Pass Retriever Efficiency (= #17.c)               ← needs #11b
```

| # | Subtask Name | Origin | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|---|
| LAT-1 | **Writer Token Streaming** | promoted from #6.6 P2 phase 3 | LM | 🟠 High | 3-5h | #11b | DEFERRED → **PROPOSED FOR CORE 3.5** |
| LAT-2 | **External-API Resilience** *(OpenAlex + yt-dlp circuit breaker + per-run cache; ~~Semantic Scholar 429~~ replaced 2026-05-12 by Angelo's verified media-pool merge)* | promoted from #17.b | LM | 🟡 Medium | 1-2h | #11b | NEW *(2026-05-10 PM — **scope reduced 2026-05-12 PM**; Semantic Scholar hotspot now obsolete since media pool is pre-generated)* |
| LAT-3 | **Corrective-RAG Retry Hardening** *(skip trivial rewrites + Levenshtein guard + smart-retry: re-run only the rewritten query, reuse first-pass results)* | promoted from #17.b + R4 (2026-05-12) | LM | 🟡 Medium | 1-2h | #11b, #9 | NEW *(2026-05-10 PM, smart-retry sibling added 2026-05-12 PM)* |
| LAT-4 | **Parallel Retrieval Fan-out** *(both axes: (a) the 8-11 KG sub-queries within a single pass via `asyncio.gather`; (b) KG + external concurrent)* | NEW | LM | 🟠 High | 2-3h | #11b | NEW *(2026-05-10 PM, in-pass-fanout sibling added 2026-05-12 PM after ADHD smoke showed 11 sequential queries)* |
| LAT-5 | **Semantic Cache Layer 1+2** | cross-listed from CORE 3 #17 | LM | 🟡 Medium | 4-8h | #11 | TODO *(see CORE 3 #17 — not double-counted in CORE 3.5 effort)* |
| LAT-6 | **End-to-End Latency Budget Dashboard** *(Langfuse view: P50/P95 per phase per query type)* | NEW | LM | 🟡 Medium | 1-2h | #11b | NEW *(2026-05-10 PM)* |
| LAT-7 | **Single-Pass Retriever Efficiency** *(language-detect bypass for English queries + profile-enrichment dedup + within-pass query dedup + negative-embedding cache + grader sees filtered recs)* | promoted from #17.c | LM | 🟠 High | 3-4h | #11b | NEW *(2026-05-12 PM — surfaced by the post-Corrective-RAG-OFF ADHD smoke: 11 sequential queries, 4 wasted no-op translations of English queries, 12× repeated `Learner Variability` embedding miss, duplicate `ADHD` in `['ADHD', 'Scienze', 'DSA', 'ADHD']`)* |

**CORE 3.5 total effort:** ~12-19h (~2-2.5 days) *(excluding #LAT-5 which is counted under CORE 3 #17 to avoid double-counting; including #LAT-5 the total is ~16-27h ≈ 2.5-3.5 days)*. Two of the seven items (#LAT-1, #LAT-5) are already in the doc; #LAT-2/#LAT-3/#LAT-4/#LAT-6 were added 2026-05-10 PM; #LAT-7 was added 2026-05-12 PM after the post-Corrective-RAG-OFF smoke revealed that even with retries disabled, a single retriever pass still costs ~100s due to issues that are *independent* of the corrective-RAG loop.

**Latency budget snapshot — updated 2026-05-12 PM (post-Corrective-RAG-OFF ADHD smoke)**

| Phase | KG-covered query *(ADHD, CR ON, 2 attempts)* | KG-covered query *(ADHD, **CR OFF**, 1 attempt — 2026-05-12)* | Out-of-scope *(fotosintesi, CR ON, 2 attempts)* | Owned by |
|---|---|---|---|---|
| Plan | ~5-10s | ~10s | ~5s | — *(low; no action)* |
| Retrieve attempt 1 | ~93s *(8 KG queries, 4 false-IT translations, 12× negative embed)* | **~102s** *(11 KG queries — 8 planner + 3 profile incl. 1 dup; same wasted translations & embeds)* | **~30s** | **`#LAT-2`** *(was Semantic Scholar 429 — now obsolete)*, **`#LAT-4` (in-pass parallel fan-out)**, **`#LAT-7` (R6 translate bypass + R2/R3 dedup + R7 neg-embed cache)** |
| Grade attempt 1 | ~5s | n/a *(CR OFF)* | ~3-5s | — *(cheap LLM, OK)* |
| Retrieve attempt 2 | **~215s** *(grader false-positive `ambiguous` → full re-run, no smart-retry)* | — | **~30s** | **`#LAT-3` (R4 smart-retry: only re-run the rewritten query)** |
| Grade attempt 2 | ~5s | — | ~3-5s | — |
| **Write** | ~90s | ~60s | **~97-109s** | **`#LAT-1`** *(single largest fixed cost — block-rendered, no streaming)* |
| Critic | ~68s | ~28s | ~15-20s | — *(acceptable; revisits if observability flags it)* |
| **Total observed** | **~514s** (2026-05-11 PM live) | **~199.8s** (2026-05-12 PM live) | **~310s** | |
| **Target P50 (post-CORE-3.5)** | **< 90s** | **< 70s** | **< 120s** | |

**Three single-biggest wins from this table:**
1. **`#LAT-1` Writer streaming** — same total time, dramatically different UX. ~60-100s of "blank-card" waiting becomes progressive content rendering. Highest *perceived*-latency win even though it doesn't reduce the wall-clock number.
2. **`#LAT-4` In-pass parallel fan-out** — running the 8-11 KG sub-queries concurrently with `asyncio.gather` reasonably halves a 100s retrieve pass to ~50s on the happy path, and is the only fix here that scales linearly with the number of planner queries (so the bigger the plan, the bigger the win).
3. **`#LAT-7` Single-pass retriever efficiency** — the *cheapest* win: a regex-guarded language-detect bypass (R6) alone saves ~25-30s/run on the ADHD-style English-planner case, plus another ~3-5s from negative-embedding cache (R7) and dedup (R2+R3). Total: ~35-40s saved per pass, ~1 day of dev work.

**Note on the Semantic Scholar row:** the original 2026-05-10 PM budget had Semantic Scholar 429 as the top out-of-scope hotspot (~60s wasted). After Angelo's media-pool merge (2026-05-12 AM), live Semantic Scholar calls are gone — the verified `kg_*_media_pool.json` ships pre-generated citations. The `#LAT-2` ticket scope is reduced to the remaining live external sources (OpenAlex used by `external_apis.py` hybrid path + yt-dlp metadata in the verification pipeline). The out-of-scope budget column above already reflects the post-merge reality.

**Why a separate wave (and not just more CORE 3 subtasks):**
- CORE 3 is the *quality* wave — its theme is "answer is more correct, more grounded, cheaper". Latency is a different cross-cutting concern that touches *every* subtask in CORE 1-4 (not just retrieval and not just writer). Putting the latency story behind a single CORE 3.5 banner keeps the dependencies, measurement methodology (#11b), and budget table colocated.
- **CORE 3.5 is intentionally numbered between 3 and 4** rather than renumbering CORE 4-6, so existing ClickUp epic IDs stay byte-identical. Same convention as **#15.c** (numerical suffix for added scope) and the `phase 1 / phase 2 / phase 3` split inside #6.6 P2.
- Two of the six items already have homes elsewhere; the wave is mostly about **organising and sequencing**, not creating bulk new effort. Total NEW work is ~4-7h (#LAT-2 + #LAT-3 + #LAT-4 + #LAT-6 minus what's already in #17.b).

**What's explicitly OUT of scope for CORE 3.5:**
- Multi-region deployment / CDN edge functions — that lives in **CORE 6 (#24-#31 deployment placeholder)**.
- LLM cost optimisation via cheaper grader models — that lives in **CORE 3 #18 (Model Upgrade & A/B)**.
- Conversation history truncation tweaks — already covered by **CORE 2 #10.4** (summary-buffer windowing) and **CORE 4 #15.b.5** (first-turn augmentation dedup).

---

### CORE 4 — Personalization: Memory & Human Loop (Target: May 2026)

**Theme:** The agent remembers and learns.
**Principle:** All depend on the Checkpointer in CORE 2 (#10).
**Deliverable:** *"Agent personalizes content per teacher and supports interactive editing mid-generation."*

```
Dependency graph:

  #15.a State Checkpointing (PostgresSaver upgrade)         ← depends on CORE 2 #10
  #15.b Conversation-Memory Hardening (no new UX)            ← depends on CORE 2 #10
  #15.c Conversation-Memory UX V2 (time-travel + branching)  ← depends on #15.a + #15.b — DEFERRED, design only
  #16   Long-Term Memory (Teacher Profiles)                  ← depends on #15.a
  #19   Human-in-the-Loop Interrupts                         ← depends on #15.a
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 15.a | **State Checkpointing (PostgresSaver upgrade)** | LM | 🟡 Medium | 5-8h | #10 | ✅ DONE locally *(2026-05-16; restored after repo sync on 2026-06-06)* |
| 15.b | **Conversation-Memory Hardening — Production-readiness without new UX** | LM | 🟠 High | 2-3h | #10 | TODO *(promoted from Point A analysis 2026-05-09)* |
| 15.c | **Conversation-Memory UX V2 — Time-travel "Rigenera" + Branching + Edit-and-rerun** | LM | 🔵 Low | 1-2d | #15.a, #15.b | DEFERRED *(design-only for now; no UI work until product signal justifies it)* |
| 16 | **Long-Term Memory (Teacher Profiles)** | LM | 🟡 Medium | 6-8h | #15.a | TODO |
| 19 | **Human-in-the-Loop Interrupts** | LM | 🔵 Low | 2h | #15.a | TODO |

**CORE 4 total effort:** ~17-23h core (~2.5-3 days for #15.a + #15.b + #16 + #19); #15.c is intentionally deferred (design recorded, no shipping commitment). *#15 was split on 2026-05-09 into 15.a (PostgresSaver — original scope) + 15.b (server-side memory hardening, no UX changes — promoted from the Point A investigation) + 15.c (UX V2 — deferred until pilot signal). Both 15.a and 15.b are landable independently and in any order.*

---

### CORE 5 — Strategic / Extension Layer (Target: June+ 2026)

**Theme:** Future-facing differentiators.
**Principle:** No internal dependencies — these are the longest tickets and the most experimental. Can be parallelized across multiple devs (LM, AG, Filippo). Follow 2026 Agentic best practices.
**Deliverable:** *"Strategic differentiators (MCP, KG self-update, curriculum alignment, slide generation)."*

```
Dependency graph:

  #20 MCP Tool Servers                                      ← no deps
  #21 Graph Updater Agent (Phase 3)                         ← no deps
  #22 Curriculum Tool — Italian Standards (Phase 3)         ← no deps
  #23 Canva Integration                                     ← no deps
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 20 | **MCP Tool Servers** | LM | 🔵 Low | 2-3 days | None | ✅ DONE (Phases 1-7 LANDED — Option A path; full surface reachable over stdio + Streamable HTTP behind JWT Bearer; 19-test regression suite green; live GUI-client smokes queued as a separate 30-min follow-up) |
| 21 | **Graph Updater Agent (Phase 3)** | LM | 🔵 Low | 2-3 days | None | TODO |
| 22 | **Curriculum Tool — Italian Standards (Phase 3)** | LM/AG | 🔵 Low | 2-3 days | None | TODO |
| 23 | **Canva Integration** | AG | 🔵 Low | 1-2 days | None | TODO |

**CORE 5 total effort:** ~7-11 days

---

### CORE 6 — Deployment & Frontend Production (Future Placeholder)

**Theme:** Take the agent live as a real product.
**Principle:** This Core is intentionally **not yet ticketed in ClickUp**. With #6.5 ✅ DONE (Path C — Mirror Stack), the deployment shape is now known: Docker Compose on Hetzner/Coolify, single uvicorn process serving `/api/v1/*` (JSON) and `/webui/*` (HTML+SSE), Postgres + Redis sidecars. Final tickets will be created once CORE 1–5 lands and we know which integration shape with AixLearning native (iframe / template port / JSON-only — see §6.5 of `docs/architecture/Frontend_Platform_Evaluation.md`) the FEM platform team prefers.
**Deliverable:** *"Agentic GraphRAG deployed as a publicly-accessible product, with onboarding flow, beta pilot with real teachers, operational runbooks, and a documented embed path into AixLearning native."*

```
Suggested subtasks (to be detailed after CORE 1-5):

  #24 Production Docker Compose stack (FastAPI + Postgres + Redis + Caddy/Traefik TLS)
  #25 CI/CD pipeline (GitHub Actions → container registry → Hetzner/Coolify deploy)
  #26 Production observability dashboard (Langfuse self-hosted or Grafana + Loki)
  #27 Load testing + capacity planning (Locust / k6)
  #28 User onboarding flow (signup, EducationalProfile setup, first lesson)
  #29 Beta teacher pilot (5-10 schools, structured feedback collection)
  #30 Operational runbooks (incident response, key rotation, scaling playbook)
  #31 AixLearning embed handoff (choose iframe / template port / JSON-only — see §6.5 ADR)
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 24-31 | **(Detailed after CORE 1-5 ships)** | TBD | TBD | ~3-4 weeks total | CORE 1-5, #6.6, #7, #12 | PLANNED |

**CORE 6 status:** Placeholder. Tickets to be created after CORE 1-5 ships and the agent has been validated end-to-end via the Path C webui.

---

## SUBTASK DESCRIPTIONS (for ClickUp subtask details)

---

### Subtask 1: Bug Fixes (DALL-E method + duplicate CurriculumTool + unused template)
**Priority:** 🟠 High | **Effort:** 1h | **Assignee:** LM

**Description:**
Three quick bug fixes / tech debt cleanups to do before validation work.

**Acceptance Criteria:**
- [ ] Fix `diagram_factory.py`: change `generate_diagram()` → `generate_educational_diagram()` (runtime crash if DALL-E branch used)
- [ ] Remove duplicate `CurriculumTool` class from inside `tools/graphrag_tool.py` (lines 239-257) — keep only the one in `tools/curriculum_tool.py`
- [ ] Decide: wire `lesson_template.txt` into Writer agent OR delete the unused file

**Depends on:** None

---

### Subtask 2: Agent ↔ Domain Config Integration
**Priority:** 🔴 Urgent | **Effort:** 1-4h | **Assignee:** LM

**Description:**
Connect Agent mode Writer/Critic to the rich domain configs in `src/aix/domains/udl_domain.py` (200+ lines) and `src/aix/domains/neuro_domain.py`. Design doc: `docs/architecture/Agent_Domain_Prompt_Integration.md`.

**Implementation status (current branch):**
- **Option 2 — DONE:** `get_domain_extension()` in `src/aix/agent/configs/domain_prompts.py` dynamically loads `get_system_prompt()` from the domain registry for the **Writer** agent (with fallback to the previous static extensions if import fails).
- **Option 3 — NOT done:** No `get_lesson_plan_template()` on `base_config.py` yet; Writer still uses the shared lesson-plan shell (not Neuro I Do/We Do/You Do vs UDL 3-principle structure as first-class templates).
- **Critic:** Still uses the **static** Writer/Critic extension blocks in `domain_prompts.py` (not the full `domains/*.py` critic text). A future step can mirror the Writer pattern or add domain-config methods as in the design doc.

**Migration path (original plan):**
- Step 1 — Option 2 (quick win, ~1h): Modify `get_domain_extension()` in `src/aix/agent/configs/domain_prompts.py` to dynamically load `get_system_prompt()` from `src/aix/domains/`. Writer keeps its lesson plan format. 1 file changed. **← Implemented.**
- Step 2 — Option 3 (clean architecture, ~4h): Add `get_lesson_plan_template()` to `src/aix/domains/base_config.py`, implement domain-specific lesson structures (Neuro: I Do/We Do/You Do; UDL: 3-Principle framework). **← Deferred.**

**Acceptance Criteria:**
- [x] Writer agent receives rich domain expertise from `src/aix/domains/` configs via Option 2 (`get_system_prompt()`), with static fallback
- [x] Critic agent has domain-specific evaluation criteria (static blocks in `domain_prompts.py`; not yet synced from full `domains/*.py` like Writer)
- [ ] Critic agent loads rich criteria from `src/aix/domains/` (optional follow-up — aligns with E5 depth goals)
- [x] Backward compatible (graceful fallback if `aix.domains` import fails)
- [ ] Test both Neuro and UDL domains via `python apps/cli/run_agent.py` (recommended confirmation)
- [ ] Existing Subtask E5 (Quality Assurance) fully benefits from enriched Critic criteria (depends on critic/domain depth + #2.5)

**Depends on:** None
**Pairs with:** #2.5 (Educational Profile Schema) — both should land in the same release for full effect
**Unblocks:** E5 (Quality Assurance System — completing Critic UDL evaluation)
**Reference:** `docs/architecture/Agent_Domain_Prompt_Integration.md`

---

### Subtask 2.5: Educational Profile Schema Integration ⭐ NEW
**Priority:** 🔴 Urgent | **Effort:** 3-4h | **Assignee:** LM (port phase complete) / AG (reviewer + integration phase) | **Status:** ✅ **PORT PHASE DONE — INTEGRATION PHASE ROLLS UP TO #2**

**Description:**
Port the per-request `EducationalProfile` schema from the existing `Angelo` branch of `FEM-modena/graphrag-aixlearning` into `feature/openrouter` (and later `main`). This gives every API request — both GraphRAG mode (`/api/v1/context`) and the future Agent mode (`/api/v1/agent/lesson`) — a structured payload describing the class (size, grade, BES disabilities, attributes, features) and the classroom environment (LIM, WiFi, furniture mobility, BYOD policy). All field names map 1:1 to the AixLearning production models (`party.models.Party`, `classroom.models.Classroom`), so Lovable, the future Vercel/Next.js frontend, and the main FEM platform can pass them through without any field translation.

**What was ported** (from `FEM-modena/graphrag-aixlearning@Angelo:api/schemas/educational_profile.py` and `agent/graph/state.py`):
- 6 enums: `GradeLevel`, `DisabilityType` (10 BES types: DSA, ADHD, DOP, DF, DCGL/M/S, DLDS, PD, SA), `ClassFeature`, `StudentAttribute`, `FornitureMobility`, `OwnDevicePolicy`
- 3 Pydantic models: `EducationalGroup`, `ClassroomEnvironment`, `EducationalProfile`
- 6 Italian label dicts (`GRADE_LABELS`, `DISABILITY_LABELS`, `CLASS_FEATURE_LABELS`, `STUDENT_ATTR_LABELS`, `FORNITURE_MOBILITY_LABELS`, `OWN_DEVICE_LABELS`) — used by `#6.6` to render the lesson form natively in Italian.

**Source path → destination path** (so Angelo can review the diff cleanly):
- `api/schemas/educational_profile.py`  →  `src/aix/api/schemas/educational_profile.py`  *(re-encoded to clean UTF-8 — original had cp437/UTF-8 mojibake on Italian accents like `Università`, `Difficoltà`, `Sì`)*
- `agent/graph/state.py`  →  merged into `src/aix/agent/graph/state.py` (added `educational_profile` to `AgentState` + `create_initial_state`; pre-existing fields untouched)

**Why it matters:** Without this, the rich domain prompts from #2 have nothing to specialize against — every lesson gets generic adaptations because the agent doesn't know it's a 25-student class with 2 ADHD + 1 DSA in a non-mobile room with no LIM. Subtasks #2 and #2.5 are the *input → processing* pair that together unlock real personalization.

**Acceptance Criteria:**
- [x] Port the schema (6 enums + 3 Pydantic models) into `src/aix/api/schemas/educational_profile.py`  ← **DONE** (port phase, by LM)
- [x] Add optional `educational_profile: Optional[EducationalProfile] = None` field to `ContextRequest` (GraphRAG, in `src/aix/api/schemas/models.py`)  ← **DONE — frozen DEV contract preserved (field is optional, omitted in legacy requests = `None`)**. The future `AgentRequest` (#7) reuses the same shape.
- [x] Propagate profile through `AgentState` (`src/aix/agent/graph/state.py`) and `create_initial_state(...)`  ← **DONE**
- [x] Thread profile through `LessonPlannerPipeline.run(...)` and `AgentOrchestrator.create_lesson_plan(...)` — both accept either a Pydantic `EducationalProfile` or a dict, normalized to dict for LangGraph state serialization  ← **DONE**
- [x] Backward compatible: every field is `Optional`; missing profile falls back to current generic behavior  ← **DONE — verified by Pydantic smoke test (`ContextRequest(query=...)` still parses, `educational_profile=None`)**
- [ ] Inject profile context into Writer / Planner / Critic prompts via the domain extension layer (combines with #2)  ← **rolls up to #2**
- [ ] Use profile in `MethodologyRanker` to boost methodologies matching disabilities present (e.g., if `ADHD` in profile, boost ADHD-tagged strategies)  ← **rolls up to #2**
- [ ] Document profile fields in `docs/api/Explainability_API_Guide_for_Frontend.md` so Simone knows exactly what to send  ← *small follow-up doc edit, not blocking*
- [ ] Test: same query with and without profile produces measurably different recommendations  ← **depends on the prompt-injection / ranker changes in #2**

**Decision recorded** *(2026-04-26)*: the port phase was executed by LM ahead of #6.6 P1, because P1's lesson form auto-renders from this Pydantic schema — stubbing it would have created throwaway form code. The remaining integration items (prompt injection + ranker boost + comparison test) are intrinsically part of #2 (Agent ↔ Domain Config Integration) and remain in AG's queue. AG is the reviewer of the port PR.

**Depends on:** None (port phase complete)
**Pairs with:** #2 (Agent ↔ Domain Config Integration — owns the prompt-injection / ranker work)
**Unblocks:** **#6.6 P1** (lesson form), E5 (Quality Assurance — Critic can now evaluate profile-aware adaptations), #7 (Agent endpoint accepts richer payload), #16 (Long-Term Memory will store last-used `EducationalProfile` per teacher)
**Reference branch ported from:** `FEM-modena/graphrag-aixlearning` branch `Angelo` — source files: `api/schemas/educational_profile.py`, `agent/graph/state.py`

---

### Subtask E5: Quality Assurance System — Critic UDL Criteria + e2e Revision Loop Validation
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** AG, LM
**Note:** Originally subtask **E5 in CORE 0** ("in progress"). Re-scoped and moved to CORE 1 because the remaining work is unblocked by Subtask #2 (Agent ↔ Domain Config) and Subtask #2.5 (Educational Profile Schema).

**Description:**
The Critic Agent already exists with multi-criteria scoring (Structure, Evidence Grounding, Pedagogical Soundness) and an automatic revision loop (max 2 cycles, triggers if average < 3.5 or any criterion < 2). Two pieces are still missing for production-quality evaluation: (1) the UDL evaluation criteria are thin (~25 lines) compared to Neuro (~85 lines), and (2) the end-to-end revision loop has never been exercised against real teacher queries with the rich domain configs and the `EducationalProfile`.

**Acceptance Criteria:**
- [ ] After #2 lands: confirm Critic auto-loads UDL evaluation criteria from `src/aix/domains/udl_domain.py` via the dynamic `get_domain_extension()` path
- [ ] After #2.5 lands: confirm Critic penalizes lesson plans that ignore the `EducationalProfile` (e.g., ignoring stated disabilities, exceeding class capacity)
- [ ] Run `python apps/cli/run_agent.py` with 5 representative UDL queries and 5 Neuro queries; verify revision loop fires when expected
- [ ] Document the Critic's scoring rubric for both domains in `docs/architecture/Agentic_GraphRAG_Architecture_Analysis.md`
- [ ] Auto-approve safety net (after `max_revisions=2`) still works to prevent infinite loops

**Depends on:** #2 (Agent ↔ Domain Config Integration), #2.5 (Educational Profile Schema Integration)
**Cross-reference:** `FUTURE_FIXES.md` **#6 (Integration Test Coverage)** — Angelo's 6-scenario matrix (normal ADHD, out-of-scope, low-confidence, `include_explainability=false`, concept graph cap, post-Neo4j MITIGATED_BY) should be adopted verbatim as the E5 acceptance test suite.

---

### Subtask 3: UDL Media Mapping — Fix Script + Generate JSON
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** LM

**Description:**
Fix `scripts/ml/generate_media_mapping.py` for UDL support and generate `data/media/kg_udl_media_mapping.json`. Currently 3 issues block UDL: wrong KG path, Neuro-specific system prompt, Neuro-specific priority categories.

**Acceptance Criteria:**
- [ ] Fix KG path resolution for UDL (`data/kg/udl/kg_udl_neo4j.json` — Phase 3B layout)
- [ ] Create UDL-specific system prompt (CAST framework, inclusive education, variability profiles, UDL-specific OER sources)
- [ ] Add UDL priority categories (`Adhd`, `AutismSpectrum`, `Dyslexia`, `UdlPrinciple`, `Barrier`, `MitigationStrategy`, etc.)
- [ ] Create UDL-specific user prompt template
- [ ] Test with `--limit 5` before full run
- [ ] Generate `data/media/kg_udl_media_mapping.json` (~763 concepts)

**Depends on:** None
**Reference:** `docs/architecture/Media_Mapping_and_Model_Upgrade_Analysis.md` — Part A, Sections A5-A7

---

### Subtask 4: Neuro Media Mapping — Full Expansion (695 concepts)
**Priority:** 🟠 High | **Effort:** 1h | **Assignee:** LM/AG

**Description:**
Run `scripts/ml/generate_media_mapping.py` for all ~695 Neuro concepts. Currently only 20 are mapped (2.9% coverage) due to `--limit 20` during initial test run. Cost: ~$5-8. Time: ~40 min.

**Acceptance Criteria:**
- [ ] Run: `python scripts/ml/generate_media_mapping.py --domain neuro --batch-size 10`
- [ ] Verify output covers ~695 concepts
- [ ] Spot-check 10 entries for quality (videos, citations, OER)
- [ ] Backup existing 20-concept JSON before overwriting

**Depends on:** None
**Reference:** `docs/architecture/Media_Mapping_and_Model_Upgrade_Analysis.md` — Part A, Section A4

---

### Subtask 5: Validate External APIs end-to-end
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** LM

**Description:**
The ExternalMediaAPI (YouTube, Wikipedia, Semantic Scholar, OER) has 1,054 lines of code but has never been tested with real API keys in a live environment. Validate each integration works.

**Acceptance Criteria:**
- [x] YouTube search: verified fallback URL mode (no `YOUTUBE_API_KEY` — returns search-page link)
- [x] Wikipedia search: verified — title, summary, URL returned for "metacognition"
- [x] Semantic Scholar: verified — works on free tier with aggressive 429 backoff; API key recommended
- [x] OER search (DOAB, Open Textbook Library, BC Campus): verified — **fixed DOAB metadata parser** (was treating list-of-dicts as dict; returned 0 results before fix, now returns 2)
- [x] Document which API keys are required vs optional in `.env` — added `YOUTUBE_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` sections to `.env.example` (both optional, commented out, with signup instructions)

**Bugs fixed during validation:**
- `external_apis.py` `_get_doab_field` / `_get_doab_authors`: DOAB API returns `metadata` as `List[{key, value}]`, not `{value: [...]}`. Fixed to handle both shapes.
- `__init__.py` eager import of `external_apis` caused `RuntimeWarning` when running `python -m aix.agent.media.external_apis`. Fixed with lazy `__getattr__` loading.

**Depends on:** None

---

### Subtask 6: Validate Media Layer end-to-end
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** LM

**Description:**
The full media layer (MediaLookup + ExternalMediaAPI + MermaidGenerator + ImageGenerator + DiagramFactory) must be validated as a whole. Individual components may work but the integration through RetrieverAgent → WriterAgent has not been tested.

**Acceptance Criteria:**
- [x] MediaLookup loads `data/media/kg_neuro_media_mapping.json` correctly — 688 concepts
- [x] MediaLookup loads `data/media/kg_udl_media_mapping.json` correctly — 756 concepts
- [x] MermaidGenerator produces valid diagram URLs — SVG renders HTTP 200 via mermaid.ink
- [x] ImageGenerator: `OPENAI_API_KEY` present, `generate_educational_diagram` method verified (no live DALL-E call — costs $0.04/image; init + method existence confirmed)
- [x] DiagramFactory routes correctly to Mermaid and DALL-E — factory initializes, `GeneratorType.MERMAID` / `DALLE` dispatch confirmed
- [x] RetrieverAgent integration: `_get_media_lookup()` loads both Neuro (688) and UDL (756); `get_combined_media(["working memory","metacognition"])` returns `has_content=True, videos=4, resources=4`

**Bug fixed during validation:**
- `media_lookup.py` default path: `Path(__file__).parent.parent.parent` resolved to `src/aix/` (3 parents from `agent/media/`), but JSON files live at `<repo_root>/data/media/`. Changed to 5 parents (`src/aix/agent/media/ → repo root`). **This means curated media was silently not loading since the Phase 3C reorg.**

**Depends on:** #1 (bug fixes), #3 (UDL media JSON), #4 (Neuro media expansion), #5 (external API validation)

---

### Subtask 6.5: Frontend Platform Evaluation & Decision (Research Spike) — ✅ DONE
**Status:** ✅ DONE (2026-04-26) | **Priority:** 🔴 Urgent | **Effort spent:** ~5h | **Assignee:** LM (with Diego, Simone, Filippo input pending review)

**Deliverables:**
- 📄 **Full evaluation + ADR**: [`docs/architecture/Frontend_Platform_Evaluation.md`](../architecture/Frontend_Platform_Evaluation.md) (12-criterion comparison, ADR-0001, three integration shapes for the eventual AixLearning embed)
- 🖼️ **UI mockup of the recommended path**: `assets/graphrag_frontend_mockup_path_c.png` (Italian-first teacher UI: educational-profile sidebar, streaming Writer card, tool-call card, Critic revision card, multimedia panel)

**Decision (ADR-0001 — Proposed):**
- **Primary:** Build the frontend on **Path C — "Mirror Stack": FastAPI + Jinja2 + htmx 2 + WebAwesome + Tailwind + sse-starlette**. Same paradigm and component library as the AixLearning native app (Django + htmx + WebAwesome + Mercure), so the eventual embed becomes a Jinja → Django syntax port instead of a rewrite.
- **Rejected for now:** Path A (Vercel + Next.js + Vercel AI SDK) — paradigm mismatch with FEM's stack, skill mismatch with the team, ~10× higher TCO, future-embed cost is a full rewrite.
- **Deferred:** Path B (immediate native AixLearning embed) — correct destination but premature; couples our release cadence to the AixLearning platform team during product discovery.
- **Working hypothesis updated:** the original hypothesis was Vercel + Next.js + Vercel AI SDK. The deep investigation of the AixLearning private repo (Python 45.5 % + HTML 34.9 % + htmx + WebAwesome + Mercure + Bun + Docker Compose) made it clear that mirroring that stack is dramatically lower-risk and lower-cost.

**Three integration shapes for the eventual embed (deferred to end of CORE 6):** iframe / template port / JSON-only. Documented in §6.5 of the eval doc. Path C preserves all three; Path A would have preserved only iframe.

**Original evaluation criteria, candidates, and full reasoning are preserved below for the historical record. The ADR consequences live in the linked doc.**

**Candidates to evaluate:**

| Platform | Type | Strengths | Weaknesses |
|---|---|---|---|
| **Vercel + Next.js + Vercel AI SDK** ⭐ working hypothesis | Full-stack React framework | Native SSE / WebSocket streaming, mature LangChain.js / LangGraph.js / Vercel AI SDK ecosystem, full UI control, `vercel deploy` zero-config, edge functions for low latency, easy Auth.js integration | Requires frontend dev capacity (or AI-paired coding); higher upfront effort than no-code |
| **Lovable** | AI-assisted no-code | Already in use by Simone, fastest iteration cycle, AI-generated UI from prompts | Vendor lock-in, limited streaming-agent support, unclear self-hosting / data residency story |
| **Streamlit Cloud** | Python-native data app | We already have a working Streamlit app — zero new code, deploy in minutes | Not production-grade UX, single-user feel, limited customization for explainability views |
| **Retool / internal tools** | Low-code admin builder | Quick CRUD-style admin panels for FEM ops team | Not student/teacher-facing; designed for internal dashboards |
| **Bubble** | Visual no-code | Visual editor like Lovable, mature ecosystem | Less AI-native than Lovable; slower iteration for AI features |
| **AixLearning native integration** | Embed inside the existing FEM main app | `EducationalProfile` already maps 1:1 to `party.models` and `classroom.models`, single sign-on, unified billing/auth | Requires coordination with main FEM app team and longer release cycle; couples our timeline to theirs |

**Evaluation criteria (score each candidate 1-5):**
- [ ] **Streaming support** — handles SSE / WebSocket out of the box for #12?
- [ ] **Auth & multi-tenancy** — supports SSO with FEM AixLearning users?
- [ ] **Time to first deployment** — days vs weeks to publish a working URL
- [ ] **Customization ceiling** — can we build the full explainability UI Simone designed (concept_graph viz, methodology cards with `explainability_phrase`, `context_warning` banners)?
- [ ] **Vendor lock-in risk** — can we self-host / migrate later if pricing changes?
- [ ] **Total cost of ownership** — license + dev time + ongoing maintenance over 12 months
- [ ] **Integration with `EducationalProfile` (#2.5)** — easy form generation from Pydantic schema?
- [ ] **AixLearning compatibility** — can it sit alongside (or eventually inside) the existing FEM platform?
- [ ] **File upload / lesson PDF support** — do teachers need to upload material for personalization?

**Acceptance Criteria:**
- [ ] Comparison matrix doc `docs/Frontend_Platform_Evaluation.md` with all 6 candidates scored against the 9 criteria
- [ ] Clear recommendation (1 primary + 1 fallback) with justification
- [ ] **Architecture Decision Record (ADR)** documenting the choice, alternatives considered, and consequences
- [ ] Stakeholder review 
- [ ] API contract sketch for #7 (request/response shape, auth header, streaming format) informed by the chosen platform — this becomes the input to #7
- [ ] Decision documented in `docs/ClickUp_Agentic_GraphRAG_Update.md` and `CHANGELOG.md`

**Depends on:** None (can start immediately, ideally in parallel with CORE 1)
**Blocks:** #6.6 (Path C webui — chosen stack), #7 (FastAPI Agent Endpoint — auth + payload contract), #12 (SSE Streaming — protocol choice), #8 (Guardrails — auth-related rules)
**Unblocks:** CORE 6 (Deployment & Frontend Production — three integration shapes documented; final shape chosen at end of CORE 6)

---

### Subtask 6.6: Path C Webui Skeleton + Agent In-Process Integration ⭐ NEW
**Priority:** 🔴 Urgent | **Effort:** 8-10h | **Assignee:** LM

**Description:**
Stand up the recommended Path C frontend (FastAPI + Jinja2 + htmx 2 + WebAwesome + Tailwind, per ADR-0001 in `docs/architecture/Frontend_Platform_Evaluation.md`) as a **new internal-facing webui that becomes the canonical end-to-end test surface for the Agent pipeline**, retiring `apps/streamlit/main.py` for that role.

The agent's **first HTTP-callable surface is HTML+SSE, not JSON+SSE**. We do this on purpose: it lets the API contract for #7 (public JSON+SSE endpoint) be designed *after* we see what the UI actually needs, instead of in a vacuum. The risk of building the wrong API drops to near-zero.

A new pure-Python **service module** (`src/aix/agent/service.py`) will wrap the existing LangGraph agent with a clean async interface. The webui (#6.6) calls this service in-process. The public JSON API (#7) wraps the same service over HTTP. Both consumers share one tested path through the agent.

**Why this ordering (vs. the original "API first") is correct here:**
- **One team builds both surfaces** (LM + AG). UI-first is the right pattern when the same person designs the contract and consumes it.
- **Real usage reveals contract requirements** — `EducationalProfile` defaults, partial-result framing, tool-approval payload shape, error semantics — instead of guessing.
- **Streamlit's UX limits hide bugs** (especially around tool-approval and Critic revisions). Path C surfaces them immediately.
- **#12 (SSE streaming) becomes a hardening task** instead of a from-scratch build, because #6.6 already ships SSE for the only consumer that exists at that point (the webui).

**Architecture:**

```
src/aix/
├── agent/
│   ├── service.py            ← NEW: pure-Python async API around the LangGraph agent
│   │                          run_agent(query, profile, *, on_event) → AsyncIterator[Event]
│   └── graph/                  (existing LangGraph nodes — unchanged)
├── api/                        (existing — JSON endpoints; #7 will wrap service.py later)
└── webui/                    ← NEW
    ├── __init__.py
    ├── routes.py             (HTML + SSE routes; in-process call into agent.service)
    ├── deps.py               (auth, current_user, EducationalProfile loader)
    ├── streaming.py          (sse-starlette helpers; converts agent events → HTML fragments)
    └── templates/
        ├── _base.html        (loads WebAwesome via CDN, Tailwind, htmx 2, Alpine.js)
        ├── partials/
        │   ├── chat_message.html
        │   ├── tool_event.html
        │   ├── critic_revision.html
        │   └── lesson_card.html
        ├── pages/
        │   ├── home.html
        │   ├── new_lesson.html
        │   └── lesson_detail.html
        └── forms/
            ├── educational_profile.html
            └── upload_pdf.html
webui_static/                 ← NEW
├── tailwind.css              (Tailwind CLI output)
└── alpine_islands/           (tiny JS islands for theme toggle, mobile nav)
```

**Phased delivery status (revised 2026-04-26):**

`#6.6` is delivered in **seven internal phases** (P0 → P6 — P0 through P3 are done as of 2026-04-26; P4 + P5 + P6 remain). The per-phase scope and status are auditable below; the original single-shot acceptance list is preserved further down for reference. Phase split mirrors `docs/architecture/Frontend_Platform_Evaluation.md` §7.

| Phase | Scope | Status | Notes |
|---|---|---|---|
| **P0 — Skeleton** | webui package, `_base.html`, dummy `/webui/` route, mounted in `aix.api.main` | ✅ DONE | Verified single-uvicorn dual-served process. |
| **P1 — Auth + lesson form** | FastAPI-Users (JWT-in-HttpOnly-cookie), register / login / logout, `/webui/lesson/new` form rendering the full `EducationalProfile` schema, persistence to SQLite | ✅ DONE | Smoke-tested by user 2026-04-26. |
| **P2 phase 1 — SSE plumbing** | `aix.webui.agent.service.run_agent_stream`, `POST /lesson/{id}/run`, `GET /lesson/{id}/stream`, phase-tracker partial, persisted-result replay, reconnect-loop fix via `outerHTML`-on-terminal-event | ✅ DONE | Verified 2026-04-26. Phase-level granularity only; per-tool / per-critic cards deferred to phase 2. |
| **P2 phase 2 — Chat workspace** | 3-pane layout on `/webui/lesson/{id}` (profile sidebar / chat / media sidebar), per-agent cards (Planner, Retriever, Writer, Critic), final lesson card, free-text query as **active chat input** on draft state (moved off `/webui/lesson/new` after smoke-test feedback), `teacher_query` persisted on `Lesson`, OOB-swap media panel, **inline profile editing** in the left sidebar, disabled multi-turn input with #10 tooltip | ✅ DONE | Smoke-tested by user 2026-04-26 on KG-covered query (`motivazione intrinseca` → 15 nodes / 30 relations / 15 media items / 143s end-to-end). Closes the §6.6 acceptance bullets *"submits a query"*, *"tool-call card"*, *"Critic revision card"*. Mockup ref: `docs/architecture/mockup front end.png`. **Two pre-existing agent-layer issues surfaced during smoke**: Planner & Critic silently fall through to hardcoded defaults when OpenRouter returns empty bodies (`"Failed to parse JSON response: Expecting value: line 1 column 1 (char 0)"` in both `planner_agent.py` and `critic_agent.py`). Tracked under the new **#11 phase 11a** (see below). |
| **P2 phase 3 — Token streaming** | Switch `aix.webui.agent.service` to `graph.astream(..., stream_mode="messages")` so writer tokens stream into the writer card live; phase tracker becomes a small secondary signal above the chat | DEFERRED → **PROPOSED FOR CORE 3.5 `#LAT-1`** *(2026-05-10 PM)* | Explicit deferral 2026-04-26: the §6.6 acceptance bullet *"sees Writer tokens streaming"* is the only one phase 3 unlocks; current phase-2 UX (writer-pending placeholder + final card) is acceptable for the dev/test e2e surface. Pick up after #11 (observability) so we can measure the user-perceived latency improvement. **Promoted into the proposed CORE 3.5 — Latency & Performance Wave on 2026-05-10 PM** as `#LAT-1`, alongside #17.b (live-run hotspots: Semantic Scholar 429 + retry overhead) and CORE 3 #17 (semantic cache, cross-listed). The 2026-05-10 PM corrective-RAG live smoke quantified the writer cold-block at **~97-110s** end-to-end — the single largest fixed cost in the latency budget table (see CORE 3.5). Streaming doesn't change wall-clock time but it transforms 100s of blank-card waiting into 100s of progressive content rendering — highest perceived-latency win in the wave. |
| **P3 — Chat attachments (uploads-only)** | **Scope:** add file uploads as Writer-only context, keep the rest of the P2 live-streaming flow untouched. Paperclip in the chat input uploads PDF/TXT/Markdown via `POST /lesson/{id}/upload` (reused `partials/chat_attachments.html` chips, with `DELETE /upload/{file_id}` to remove). Manifest persisted on `Lesson.uploaded_files_json` (files on disk under `data/webui/uploads/{lesson_id}/`). At run time, joined excerpts feed `AgentState.teacher_provided_context` → Writer prompt appendix only (no KG ingest, no Planner gating). `POST /run` flips `draft → running` and the SSE pane drives planner → retriever → writer → critic live, exactly like P2. | ✅ DONE *(2026-04-26)* | Smoke-tested. **Late fix same day**: the icon-only `<wa-button>` wrappers around the paperclip / Mermaid / disabled-paper-plane icons collapsed to ~0 width because WebAwesome 3.x only triggers the square icon-button render path when the slotted `<wa-icon>` carries a `label="..."` attribute (per [webawesome.com/docs/components/button](https://webawesome.com/docs/components/button/) → "Icon Buttons"). Patched by moving the screen-reader text from `aria-label` on the button to `label` on the icon in `partials/chat_input.html`. |
| **P4 — Lesson library + history** | Lesson list page + search + PDF export, per `Frontend_Platform_Evaluation.md` §7. Replaces today's marketing-style `/webui/` home with a teacher-facing library of past lessons (status chips, filter by status / subject / date, paginated by `created_at DESC` over `Lesson.owner_id`), adds `DELETE /webui/lesson/{id}` (with confirm), and ships server-side HTML→PDF export at `GET /webui/lesson/{id}/export.pdf` via **WeasyPrint** (reuses the `chat_lesson_card` template under a print stylesheet). Uses the columns already on `Lesson` (`teacher_query`, `subject`, `topic`, `status`, `created_at`); no schema migration needed. | TODO (~2d) | **Soft prerequisite: #11a (Agent JSON Parse Hardening)** — without it, every library row claims *"✓ approvata dal Critico"* even when Critic silently fell through to "Approved due to parsing error", which corrupts the trust signal of the library and the exported PDFs. Order recommended: 11a → P4. |
| **P5 — Polish + Italian copy + Tailwind CLI** | Polish pass before any production deploy. Four sub-deliverables: *(a)* Italian-copy sweep (status chips → "completata" / "in esecuzione" / "errore", `it-IT` date formatting, empty-state strings, missing aria-labels); *(b)* a11y audit (color contrast on slate-on-slate cards, keyboard nav through chat input + attachments + send, `aria-live="polite"` on SSE stream container, focus management after htmx swaps); *(c)* mobile breakpoints (3-pane → `<wa-drawer>` collapse on `<md`, full-viewport chat, iOS-Safari keyboard handling); *(d)* **Tailwind CDN → Tailwind CLI** with a real `npm run build:css` writing to `src/aix/webui/static/css/app.css`, replacing the current `cdn.tailwindcss.com` script that prints a "do not use in production" console warning and ships ~3 MB of utilities instead of a tree-shaken ~30 KB bundle. Mounts `StaticFiles` for `/webui/static/`. | TODO (~2d) | Open question deferred to Simone (Italian-copy ownership — see `Frontend_Platform_Evaluation.md` §9 Q4). Tailwind CLI is the only sub-task with non-trivial infra (Node toolchain in dev container). a11y + mobile are the only remaining pieces gating an actual production deploy. |
| **P6 — Hetzner deploy** | Docker Compose with FastAPI app + Postgres (replaces dev SQLite at `data/webui/webui.db`) + Caddy/Traefik TLS termination. Optional Mercure hub if we need fan-out for later collaborative features. Per `Frontend_Platform_Evaluation.md` §7. | TODO (~1d) | Last #6.6 phase before P7 (embed handoff to AixLearning at end of CORE 6). Triggers the SQLite → Postgres migration of `Lesson` + `User` tables; that migration is straightforward but should land with #11 observability already enabled so we can monitor the cutover. |

**Why the split:** P2 phase 1 wired the streaming plumbing end-to-end but kept a phase-tracker UI rather than the per-agent-card chat workspace from the original §6.6 architecture sketch. The phased split records that gap and closes it in phase 2 + phase 3 without re-opening P1 / P0 work.

**Multi-turn input parking:** the persistent *"Chiedi una modifica…"* input rendered at the bottom of the chat (per the mockup) is **disabled** in P2 phase 2 with a `<wa-tooltip>` pointing to **#10 — Conversation Memory (LangGraph Checkpointer)**. Multi-turn refinement is a real feature gated on #10's checkpointer, not a UI-only toggle.

**Acceptance Criteria (original — preserved for reference; concretely satisfied by the phase table above):**
- [x] **Service layer:** `src/aix/webui/agent/service.py` with `run_agent_stream(lesson, session) -> AsyncIterator[StreamEvent]`. Wraps the existing `lesson_planner_graph`. No HTTP, no UI knowledge. Unit tests deferred to "Tests for webui routes + agent.service" task. *(Note: ended up under `aix.webui.agent.service` rather than `aix.agent.service` since the wrapper is webui-specific lifecycle glue, not pure agent logic; #7 will lift it if needed.)*
- [x] **WebUI package:** `src/aix/webui/` with `__init__.py`, `routes.py`, `auth/` (replaces `deps.py`), `agent/service.py` (replaces top-level `streaming.py`), and the `templates/` tree above.
- [x] **Routes:**
  - `GET  /webui/`                       → home + recent lessons
  - `GET  /webui/lesson/new`             → educational profile form (mirrors CORE 1 #2.5 schema)
  - `POST /webui/lesson`                 → creates session, redirects to detail page
  - `GET  /webui/lesson/{id}`            → detail page with SSE-connected stream container
  - `POST /webui/lesson/{id}/run`        → ⭐ added in P2 phase 1: triggers agent run, returns the chat pane fragment
  - `GET  /webui/lesson/{id}/stream`     → SSE channel emitting Writer tokens, tool start/end, Critic revisions, completion
- [x] **Asset pipeline:** `_base.html` loads htmx 2 + WebAwesome + Tailwind via CDN (Phase P0); Tailwind CLI added in Phase P5. No Node bundler required.
- [x] **Auth:** FastAPI-Users with JWT cookie (HTTP-only, SameSite=Lax). JWT shape compatible with future AixLearning SSO (`iss`, `sub`, `email`, `domain`).
- [x] **End-to-end manual test** *(2026-04-26)*: teacher logs in, fills the educational profile, submits a query directly in the chat input on `/webui/lesson/{id}`, sees Planner + Retriever + Writer + Critic cards stream into the conversation in order, and the final lesson card with full markdown content. The retriever card and right-sidebar media panel populate from the KG (`motivazione intrinseca` → 15 nodes / 30 relations / 8 strategies / 15 media items). Inline profile editing in the left sidebar verified end-to-end. ⚠️ Caveats explicitly accepted: *(a)* writer-token streaming is deferred to **P2 phase 3** (writer card currently pops in fully formed after the writer LLM call returns); *(b)* Planner & Critic intermittently fall back to hardcoded defaults due to OpenRouter empty-body responses — tracked under **#11 phase 11a**.
- [x] **Streamlit retired** as the agent e2e test surface *(2026-04-26)*. `apps/streamlit/main.py::render_agent_mode()` displays a `st.warning(...)` banner pointing to `http://127.0.0.1:8765/webui/` and listing the chat-workspace features. The GraphRAG admin mode in the same Streamlit app is intentionally untouched — it remains the read-only KG inspector and is **not** considered retired.
- [x] **Registered in `src/aix/api/main.py`:** `app.include_router(webui_router, prefix="/webui")`. Single uvicorn process serves both the existing `/api/v1/*` JSON endpoints and the new `/webui/*` HTML endpoints.
- [x] **No public JSON contract for the agent yet** — that lands in #7. The webui's HTML routes are explicitly *internal* (no documented contract, no CORS exposure).

**Out of scope (explicit):**
- Public JSON+SSE contract → #7
- Hardened SSE (backpressure, reconnection, `Last-Event-ID`) → #12
- Input/output guardrails → #8
- Multi-turn refinement chat (active "Chiedi una modifica…" input) → #10 (Conversation Memory)
- Embed into AixLearning → CORE 6 (P7)

**Depends on:** #6.5 (ADR — chosen stack), CORE 1 (`EducationalProfile` schema, validated agent loop, validated media layer)
**Blocks:** #7 (extracts and stabilises the service layer), #8 (input schema validated against real UI usage), #12 (hardens streaming first delivered here)

---

### Subtask 7: FastAPI JSON+SSE Agent Endpoint (public contract) — ✅ DONE *(2026-04-26)*
**Priority:** 🔴 Urgent | **Effort:** 4-6h *(actual: ~5h)* | **Assignee:** LM

**Description:**
Wrap the **already-existing service layer** (`src/aix/webui/agent/service.py`, built in #6.6) in a documented public JSON+SSE contract at `/api/v1/agent/*`. This is the contract any *external* consumer will use: AixLearning embed (CORE 6), mobile app (future), Postman / curl, partner integrations.

Because the service layer was already exercised end-to-end by the Path C webui in #6.6, the design risk here was near-zero — we froze the shape that already works.

**Acceptance Criteria:**
- [x] New route file `src/aix/api/routes/agent.py` exposing:
  - `POST /api/v1/agent/run` (sync — returns final lesson plan + explainability fields, drains the SSE stream and assembles `AgentRunResponse`)
  - `POST /api/v1/agent/stream` (SSE — same event taxonomy as the webui, but as JSON-encoded SSE events instead of HTML fragments)
- [x] Pydantic request schema (`AgentRunRequest`) accepts: `query`, `domain`, `language`, `session_id`, **`educational_profile`** (CORE 1 #2.5), `teacher_provided_context` (P3 paperclip uploads), `max_revisions`
- [x] Pydantic response schema (`AgentRunResponse`) returns: lesson plan markdown + metadata (intent, scope, evidence count, retrieval attempts), `CriticScores`, `approved` flag, planner/retriever **explainability fields** matching `/api/v1/context` patterns, `MediaCounts`
- [x] Both endpoints delegate to `aix.webui.agent.service.stream_agent_events` — a **DB-less sibling** of `run_agent_stream` — so the public API never touches the webui SQLite. Zero agent logic in the route handler.
- [x] Auth middleware: JWT Bearer **and** cookie work in parallel (`auth_backend` cookie + new `bearer_backend`); both backends register the same `current_active_user` dependency, so swapping transports is a header change for the consumer, no code change in the route.
- [x] CORS made env-driven: `WEBUI_CORS_ALLOW_ORIGINS` (default `*` for dev) so prod (Hetzner #P6) can lock to a single origin without code edits.
- [x] OpenAPI spec auto-generated at `/openapi.json` and rendered at `/docs`. **UX upgrade:** Swagger UI shows a `Minimal` / `Rich` examples dropdown for both `/agent/run` and `/agent/stream` (driven by `Body(..., openapi_examples=...)`), mirroring `/api/v1/context`'s pattern. The `Rich` example exercises every optional `EducationalProfile` field + `teacher_provided_context`.
- [x] Integration tests in `tests/api/test_agent_routes.py` (7 cases, all green): 401 on `/run` without auth, 401 on `/stream` without auth, 422 on bad payload, sync JSON happy path, 502 on agent-pipeline error, SSE event stream emission, **OpenAPI inventory strictly additive vs. `data/diagnostic/openapi_before_p7.txt` baseline**. Auth fixture spins a real fastapi-users registration flow + JWT issuance; the agent runtime is mocked at the `stream_agent_events` boundary so the suite is contract-only and fast (~39s).
- [x] Webui **NOT** migrated to call the public API over HTTP — by deliberate decision, the webui still uses in-process `run_agent_stream` for zero latency and no double serialisation. Both code paths are now routed through the same upstream `AgentOrchestrator`.

**Files touched:**
- *NEW:* `src/aix/api/schemas/agent.py` — `AgentRunRequest` / `AgentRunResponse` / 7-variant `AgentStreamEvent` discriminated union
- *NEW:* `src/aix/api/routes/agent.py` — `POST /run`, `POST /stream`, plus `_AGENT_REQUEST_OPENAPI_EXAMPLES` (Minimal/Rich dropdown)
- *NEW:* `tests/api/test_agent_routes.py` + `tests/api/__init__.py`
- *NEW:* `data/diagnostic/openapi_before_p7.txt` (OpenAPI inventory baseline) + `scripts/diagnostic/list_openapi_paths.py` (helper)
- *Edited:* `src/aix/webui/agent/service.py` — added `stream_agent_events()` DB-less helper; `run_agent_stream()` left untouched
- *Edited:* `src/aix/webui/auth/backend.py` — added `BearerTransport` + `bearer_backend` alongside the existing `CookieTransport` + `auth_backend`
- *Edited:* `src/aix/webui/auth/dependencies.py` + `__init__.py` — registered both backends with the same `FastAPIUsers` instance
- *Edited:* `src/aix/api/main.py` — mounted `agent_router` at `/api/v1` and `fastapi_users.get_auth_router(bearer_backend)` at `/auth/jwt`, both wrapped in `try/except` so a future regression there cannot kill the GraphRAG mode startup. CORS middleware moved to env-driven config.
- *Edited:* `src/aix/api/routes/context.py` — docstring "forthcoming /agent endpoint" sentence updated to point at the now-live routes (no behaviour change).

**Depends on:** #6.6 (service layer + validated agent UX), #2.5 (Educational Profile schema)
**Blocks:** Future AixLearning embed (CORE 6), any non-browser API consumer (Postman / curl / mobile / MCP server in #20)

**Implementation notes & lessons learned:**
1. **Service-layer split was the cheapest change with the highest payoff** — extracting a DB-less `stream_agent_events()` from `run_agent_stream()` (instead of a flag on the existing function) means the webui's SQLite path is not even importable from the public API call graph. Cleaner blast-radius if the public API ever needs a different persistence story.
2. **JWT Bearer + cookie can coexist trivially in fastapi-users** by listing both backends in the same `FastAPIUsers([...])` constructor. The `current_active_user` dependency just tries each transport and accepts the first that resolves. No conditional middleware, no per-route flag.
3. **Swagger UI dropdown UX must use route-level `Body(..., openapi_examples=...)`, not schema-level `json_schema_extra={"examples": ...}`** — the latter renders as raw JSON in the editable body and is widely cited as a Swagger UI footgun. The Schema tab still benefits from a single canonical `json_schema_extra={"example": ...}` for clients that consume the bare schema (Postman / openapi-generator).
4. **OpenAPI strictly-additive regression test** (`test_openapi_inventory_strictly_additive`) is cheap and high-signal: snapshot the live spec to a flat-file baseline before the change, fail the suite if any path disappears or renames. This caught nothing this round but is the single best safety net for any future "lift-and-shift" tasks (#20 MCP, CORE 6 embed).
5. **`try/except` around router mounts in `main.py`** — both `agent_router` and `bearer_backend` mounts are wrapped, so a future import error in either path drops the new feature but **never** prevents the legacy GraphRAG mode from booting. Mirrors the same pattern used for the webui mount.
6. **One bash-vs-PowerShell pitfall surfaced in dev:** `pytest ... | tail -n 40` failed because `tail` is not a PowerShell cmdlet (rejected the entire pipeline before pytest ran). Use `Select-Object -Last 40` on Windows, or just drop the pipe — pytest's `-q` summary is already short. Captured here so future contributors don't burn 2 minutes on it.
7. **`/api/v1/context` docstring reference cleanup** — the previous "forthcoming `/api/v1/agent/...`" note was updated in the same change so the GraphRAG-mode docs stay self-consistent. *(Minor but: docs that say "forthcoming" tend to age into lies if not bumped at delivery.)*

---

### Subtask 8: Guardrails: Input/Output Validation
**Priority:** 🔴 Urgent | **Effort:** 3-5h | **Assignee:** LM

**Description:**
Add safety guardrails for an educational system. Zero guardrails currently exist — no prompt injection detection, no output validation, no PII protection. Validation runs **inside the service layer** (`src/aix/agent/service.py`, built in #6.6) so both the webui and the public JSON API enforce identical rules with one implementation.

**Acceptance Criteria:**
- [ ] Input: prompt injection detection (regex patterns for "ignore all previous instructions" etc.)
- [ ] Input: query length limits
- [ ] Input: language detection (accept only Italian/English)
- [ ] Input: `educational_profile` schema validated server-side via Pydantic (the field shapes that the webui exercises in #6.6 are now hard requirements)
- [ ] Output: Pydantic schema validation (lesson plan has required sections)
- [ ] Output: content safety check (OpenAI Moderation API — free)
- [ ] Output: PII detection (no student personal data leakage)
- [ ] Failures surface as typed `AgentError` events on the SSE channel so the webui can render them as `<wa-callout variant="danger">` cards

**Depends on:** #6.6 (`agent.service` is the single insertion point; input schema is exercised by the real UI before this task starts)

---

### Subtask 9: Corrective RAG (Retrieval Grading) — ✅ DONE (code) *(2026-05-10)*
**Priority:** 🔴 Urgent | **Effort:** 3-4h | **Assignee:** LM | **Status:** ✅ Code-complete; flag-OFF default; live-traffic exercise pending #11b

**Description:**
Add a retrieval quality grading step between Retriever and Writer in the LangGraph. If retrieved KG data is poor or irrelevant, rewrite the query and re-retrieve (max 2 attempts by default) instead of generating from bad context. Research shows ~30% of RAG errors trace to poor retrieval.

**Acceptance Criteria:**
- [x] New `grade_retrieval_node` in LangGraph between Retrieve and Write *(`src/aix/agent/graph/nodes.py`)*
- [x] Lightweight LLM grading call (~200-400 tokens) — cheap model (`gpt-4o-mini`) with `response_format={"type":"json_object"}`; fail-open fallback `grade="relevant"` so the loop never blocks the run
- [x] Query rewriting + re-retrieval loop (max attempts tunable via `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS`, default 2, clamped 1-4)
- [x] `retrieval_attempts` counter in `AgentState` (+ 4 sibling fields: `retrieval_grade`, `retrieval_grade_reason`, `retrieval_rewritten_query`, `retrieval_warning`)
- [x] Logging for grading decisions and rewrites — structured INFO/DEBUG events from `nodes.grade_retrieval_node` and `should_retry_retrieval`
- [x] **Backward-compat:** topology only mutates when `AIX_CORRECTIVE_RAG_ENABLED=true`; default OFF → byte-identical to pre-#9 (`retrieve → write`)
- [x] **Writer awareness:** `WriterAgent.write` accepts `retrieval_warning` + `retrieval_grade_reason`; surfaces a low-confidence caveat in IT/EN *only* when warning is set
- [x] **UI surface:** SSE `retriever` payload extended with grade fields; `chat_retriever_card.html` renders a "Valutazione recupero" row (emoji + label + reason + attempt count) when grading ran
- [ ] **Live-LLM end-to-end smoke** — flip flag, run 5 lessons (3 happy-path + 2 with deliberately weak queries), confirm at least one retry fires and lesson includes the caveat; deferred until #11b Langfuse traces are wired so we can quantify the quality delta

**Files changed (landed 2026-05-10):**
- *NEW:* `src/aix/agent/agents/retrieval_grader_agent.py` — `RetrievalGraderAgent` + `GraderResult` + multi-strategy `_extract_json` helper
- *Edited:* `src/aix/agent/graph/state.py` — 5 new optional `retrieval_*` fields, all initialised to `None` in `create_initial_state()`
- *Edited:* `src/aix/agent/graph/nodes.py` — `grade_retrieval_node`, `should_retry_retrieval` router, `_corrective_rag_enabled()` + `_corrective_rag_max_attempts()` helpers; `write_node` threads `retrieval_warning` + `retrieval_grade_reason` into the writer
- *Edited:* `src/aix/agent/graph/lesson_planner_graph.py` — conditional topology behind `AIX_CORRECTIVE_RAG_ENABLED`
- *Edited:* `src/aix/agent/agents/writer_agent.py` — accepts new kwargs; prepends low-confidence caveat when warning is set
- *Edited:* `src/aix/webui/agent/service.py` — `_build_retriever_payload` exposes new fields; `grade_retrieval` added to `PHASE_LABELS`; `_grader_will_retry()` mirrors the router so `writer_pending` is only emitted when the grader will continue
- *Edited:* `src/aix/webui/templates/partials/chat_retriever_card.html` — "Valutazione recupero" row gated on `p.get('retrieval_grade')`
- *Edited:* `.env.example` + `.env` — documented `AIX_CORRECTIVE_RAG_ENABLED` (default `false`) and `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS` (default `2`)

**How to enable for testing:**
```
# in .env
AIX_CORRECTIVE_RAG_ENABLED=true
AIX_CORRECTIVE_RAG_MAX_ATTEMPTS=2
```
Restart the API. With the flag OFF, the graph compiles and runs exactly as before.

**Depends on:** None
**Unblocks:** Quantitative quality measurement under #11b (need traces to compare grade=relevant vs grade=irrelevant runs)

#### #9.UX-1 — single retriever card per turn *(2026-05-10 PM)* — ✅ DONE
**Why:** the first post-#9 live smoke (turn 8 of `lesson_id=fed4810b-…`) rendered the **Retriever Agent card twice** in the chat thread — once with raw stats only, then ~5s later again with the green "VALUTAZIONE RECUPERO · RILEVANTE" row. The agent is fine; the duplication is purely a streaming-layer side-effect of how `grade_retrieval` was wired into the SSE stream.

**Root cause:** two issues compounded in `aix.webui.agent.service` —
1. Both SSE loops (`run_agent_stream` for the WebUI and `stream_agent_events` for `/api/v1/.../events`) emitted `kind="retriever"` **twice per turn** when corrective-RAG was ON: once after the `retrieve` node finished (no grade fields yet), and again after the `grade_retrieval` node finished (with the grading verdict populated).
2. The chat workspace container `#chat-cards` is declared with `hx-swap="beforeend"` (see `partials/chat_pane.html`) — i.e. every SSE `card` event APPENDS a fresh card; it never replaces. That's the correct behaviour for the chat thread (each agent gets its own card, in order) but it meant our second emit produced a second visible card instead of updating the first.

**Fix (Option A — minimal, default-safe):** keep `grade_retrieval` as the sole emitter when corrective-RAG is ON, and let the `retrieve` branch keep its pre-#9 emit when the flag is OFF. Two symmetric edits in `service.py`:

```python
elif node_name == "retrieve":
    # CORE 2 #9.UX-1 — when corrective-RAG is ENABLED, the
    # ``grade_retrieval`` node will run next and become the
    # sole emitter of ``kind=retriever`` (with the grading
    # verdict already populated in the payload). We skip
    # the emit here to avoid producing TWO retriever cards
    # per turn, because ``#chat-cards`` uses
    # ``hx-swap=beforeend`` (i.e. append, not replace).
    if not _is_corrective_rag_enabled():
        yield StreamEvent(kind="retriever", payload=_build_retriever_payload(final_state))
        write_revision_idx += 1
        yield StreamEvent(kind="writer_pending", payload={...})
```

The `grade_retrieval` branch is unchanged — it already emits exactly one `kind="retriever"` (with grade fields) and one `kind="writer_pending"` (when the router decided not to retry). Net effect:

| Mode | `retrieve` branch | `grade_retrieval` branch | Cards per turn |
|---|---|---|---|
| Corrective-RAG **OFF** (default) | emits retriever + writer_pending *(unchanged from pre-#9)* | not in topology | **1 retriever** |
| Corrective-RAG **ON** | silent | emits retriever + writer_pending | **1 retriever** |

**Trade-off:** with the flag ON, the retriever card appears ~3-5s later (the grader's wall time on `gpt-4o-mini` via OpenRouter), but it appears **once** with the grade verdict already populated — a cleaner "full picture in one shot" UX vs. the prior "stats now, verdict 5s later" pop-in.

**Backward compatibility:** flag-OFF byte-identical to pre-#9; flag-ON now byte-identical to flag-OFF in event count (one retriever per turn) — only the payload of that one event differs (gains the grade fields). No template, no htmx, no schema changes.

**Files touched:**
- `src/aix/webui/agent/service.py` — gated emit in both `run_agent_stream` (~L1164) and `stream_agent_events` (~L1438).

**Verification:** lints clean. Live re-run will confirm one Retriever card per turn under both flag states.

#### #9.UX-2 — single retriever card across N retry attempts *(2026-05-10 PM, later)* — ✅ DONE
**Why:** the second post-#9 live smoke (turn for "fotosintesi clorofiliana") rendered the **Retriever Agent card twice** in the chat thread *despite* #9.UX-1. Different root cause: this query was out-of-scope for the KG, so the grader returned `irrelevant` on attempt 1 (rewrite proposed) and `irrelevant` again on attempt 2 (max attempts hit, fall through to writer). The `grade_retrieval` LangGraph node ran TWICE for that turn — once per attempt — and our SSE branch emitted `kind="retriever"` on each iteration. `hx-swap=beforeend` on `#chat-cards` then stacked them.

**Root cause:** #9.UX-1 only addressed the `retrieve` vs. `grade_retrieval` duplicate. It assumed `grade_retrieval` runs ONCE per turn. That holds when the grader returns `relevant` on attempt 1 (the common happy path, which is what the first smoke exercised), but breaks down whenever the corrective-RAG retry loop fires.

**Fix:** add a second gate to the `grade_retrieval` SSE branch — only emit the retriever card on the FINAL iteration (when the router decided NOT to retry), exactly mirroring the gate already used for `writer_pending`. Both events deferred together until the loop terminates. The retriever card therefore appears once per turn with the FINAL `retrieval_attempts` count already settled (e.g. `Tentativi: 2/2`).

```python
elif node_name == "grade_retrieval":
    # CORE 2 #9 + #9.UX-1 + #9.UX-2 — gate retriever emit on
    # ``_grader_will_retry`` so a turn with N attempts produces ONE
    # retriever card, not N. Both events (retriever + writer_pending)
    # are deferred together so the chat card surfaces with the
    # final attempts count already populated.
    if not _grader_will_retry(final_state):
        yield StreamEvent(kind="retriever", payload=_build_retriever_payload(final_state))
        write_revision_idx += 1
        yield StreamEvent(kind="writer_pending", payload={...})
```

**Tentativi badge ("N/M"):** the chat card now shows a small `Tentativi: N/M` badge in the grading row's header — but only when N>1. Progressive disclosure: the common happy-path case (one attempt → relevant) stays visually clean; the badge surfaces only when the retry loop actually fired. Driven by the new `retrieval_attempts_max` payload field which `_resolve_max_attempts()` reads from `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS` with the same `1 ≤ N ≤ 4` clamp the graph uses.

**Trade-off:** with corrective-RAG ON and a retry firing, the retriever card now appears later (~10-15s end-to-end on Claude Sonnet 4.6 via OpenRouter, vs ~5s on the happy path) — that's the cost of waiting for the loop to terminate. Acceptable given the alternatives (intermediate "thinking" pulse = new event kind + new template = larger surface for V1.

**Backward compatibility:** flag-OFF byte-identical to pre-#9; flag-ON happy path identical to #9.UX-1 (one card with the green RILEVANTE row); flag-ON retry path now produces ONE card instead of N.

**Files touched:**
- `src/aix/webui/agent/service.py` — gated emit in both SSE loops; new `_resolve_max_attempts()` helper extracted from the existing `_grader_will_retry` env-resolution code (single source of truth); `retrieval_attempts_max` field added to `_build_retriever_payload`.
- `src/aix/webui/templates/partials/chat_retriever_card.html` — `Tentativi: N/M` badge in the grading row header, gated on `attempts > 1`.

#### #9.UX-3 — outcome-driven explainability messaging *(2026-05-10 PM, later)* — ✅ DONE
**Why:** the same "fotosintesi clorofiliana" turn that surfaced #9.UX-2 also surfaced a UX/copy issue. The grader correctly returned `grade=irrelevant` (the KG is specialised in pedagogy + UDL, not in disciplinary photosynthesis content), the writer correctly composed a useful lesson by integrating Wikipedia + papers + OER from the hybrid retrieval path, and the run completed cleanly — yet the chat card rendered with **red ✗ "NON RILEVANTE"** styling, looking like an error to the teacher. Two semantic problems in one:
1. The teacher reads red as "something broke", not "the KG didn't have the disciplinary content but I went and got it from external sources for you" — which is what actually happened.
2. The grader's `irrelevant` verdict refers to the KG specifically, not to the lesson outcome. Mapping it 1:1 to a UI color collapses two very different states ("KG out-of-scope, hybrid filled the gap" vs. "KG sparse + nothing else found") into the same red treatment.

**Fix:** introduce a single `retrieval_outcome` token computed by `service._compute_retrieval_outcome(state, media_counts)` that drives the card's color, headline, and explanatory copy. Four mutually exclusive outcomes:

| Outcome | Color | Headline | Trigger | Body copy |
|---|---|---|---|---|
| `success` | 🟢 emerald | "Valutazione recupero · Rilevante" | grade=relevant *(or grading didn't run)* | grader's own `retrieval_grade_reason` verbatim — unchanged from pre-#9.UX-3 |
| `adapted_with_hybrid` | 🔵 sky | "Valutazione recupero · **Adattamento riuscito**" | grade=ambiguous\|irrelevant AND hybrid retrieval landed (`external_resources` populated OR `media_counts.articles + .oer > 0`) | structured "what came from KG (pedagogy + learner profile) + what came from external sources (disciplinary content)" breakdown with concrete counts; grader's reason kept as small "Dettaglio del valutatore" italic line |
| `limited_kg_only` | 🟡 amber | "Valutazione recupero · Copertura limitata" | grade=ambiguous\|irrelevant AND no hybrid resources landed | explicit "verifica manualmente il contenuto disciplinare prima dell'uso in classe" warning |
| `grader_error` | 🔴 rose | "Valutazione recupero · Errore valutatore" | grader LLM threw and `grade_retrieval_node` returned `grade=relevant` with the sentinel reason `"Grader exception: <ClassName>"` | "Il valutatore non ha potuto giudicare … considera di rigenerare se il risultato non è soddisfacente" |

**Key invariants:**
- **In-scope queries (ADHD, UDL, DSA…) are unchanged.** They land on `grade=relevant` on attempt 1 → green ✅ "RILEVANTE" with the grader's own reason — bit-for-bit identical to the screenshot from the live ADHD smoke. The outcome token doesn't change rendering for the success path; it just expresses the existing branch as a token instead of a `grade`-string switch.
- **RED is reserved for genuine grader exceptions only.** A grader-detected "irrelevant" is no longer red — it's BLUE (gracefully adapted) or AMBER (genuinely sparse), depending on whether the hybrid path landed external content. This is the outcome-driven part of the reframe: the color reflects the LESSON-LEVEL outcome, not the KG-level grade.
- The detection of `grader_error` works through the sentinel reason prefix `"Grader exception:"` — `grade_retrieval_node` already stamps that prefix in its `except Exception:` branch. The grade itself is fail-open `"relevant"` so the writer's downstream behaviour is unchanged (no caveat injected); only the chat-card outcome reflects the error.

**Lifted from the green ADHD card** (the 2026-05-10 live smoke screenshot of the in-scope path):
- *Dimension-by-dimension structure*: the green card reads "directly cover the core pedagogical and neurological dimensions of the query: ADHD characteristics … inclusive teaching strategies … self-regulation approaches". The new BLUE copy mirrors that structure, separating "what came from KG" from "what came from external sources" with the same axis-by-axis style.
- *Concrete counts inline*: the green card cites concept names so the teacher can verify retrieval at a glance. The new BLUE copy cites concrete counts (`{nodes_count} concetti, {recommendations_count} strategie, {articles} articoli scientifici, {oer} OER & manuali`) for the same verification benefit.
- *Trust verdict conclusion*: the green card ends with "sufficient context to write a complete lesson on attention deficit disorders". The new BLUE copy ends with "→ La lezione integra pedagogia inclusiva (KG) e contenuto disciplinare (risorse esterne curate). Le fonti esterne sono citate inline nel piano per permetterti di verificarle." — same role: actionable trust signal for the teacher.

**Backward compatibility:**
- When corrective-RAG is OFF, `retrieval_grade` is `None` and the template's outer `{% if p.retrieval_grade %}` guard skips the entire block — pre-#9 cards render bit-for-bit identically.
- When corrective-RAG is ON and grade=relevant, the rendered green card is bit-for-bit identical to pre-#9.UX-3 (the `success` outcome reproduces the prior styling and copy).
- The new outcomes (blue / amber / red) only render when grading is on AND grade != relevant — net-new UX surface, no regression risk to the existing in-scope rendering.

**Files touched:**
- `src/aix/webui/agent/service.py` — new `_compute_retrieval_outcome(state, media_counts)` pure function (4-way branch + sentinel detection); new `_GRADER_EXCEPTION_REASON_PREFIX` constant (single source of truth, mirrors the prefix used in `nodes.grade_retrieval_node`); `retrieval_outcome` field added to `_build_retriever_payload`.
- `src/aix/webui/templates/partials/chat_retriever_card.html` — full rewrite of the grading-row block: outcome → palette/icon/headline/body branching, plus the rewrite-UX line and the success/blue conclusion line.

#### #9 — Rewrite UX (Bonus) — surfacing the grader's reformulated query *(2026-05-10 PM, later)* — ✅ DONE
**Why:** the corrective-RAG retry path computes a rewritten search query when it requests a retry (`retrieval_rewritten_query`), but pre-#9.UX-3 the field was rendered as a small unlabelled `<code>` line at the bottom of the grading row. Teachers couldn't tell *what attempt* it belonged to or *why* it mattered.

**Fix:** the rewrite is now surfaced as a labelled inline line *only when populated* (i.e. only when a retry actually happened):

```
Query riformulata al 2° tentativo:  fotosintesi clorofiliana UDL scaffolding DSA ADHD scuola secondaria strategie inclusive
```

The "al N° tentativo" qualifier appears only when `attempts > 1` (consistent with the `Tentativi: N/M` badge's progressive-disclosure rule). On the happy path (one attempt → relevant), nothing renders. The rewrite line lives inside ALL outcome variants (green / blue / amber / red) — i.e. it's outcome-agnostic explainability. This is meaningful trust signal: the teacher sees that the system "thinks about its own queries" rather than just hitting the KG once and giving up.

**Files touched:**
- `src/aix/webui/templates/partials/chat_retriever_card.html` — labelled inline `Query riformulata al N° tentativo:` line, gated on `p.retrieval_rewritten_query`.

#### #9.UX-2 + #9.UX-3 + Rewrite UX — combined outcome matrix
| User query type | Grade | Hybrid? | Color | Headline | Body | Tentativi badge | Rewrite line |
|---|---|---|---|---|---|---|---|
| In-scope (ADHD, UDL, DSA…) | `relevant` (attempt 1) | irrelevant | 🟢 green | "Rilevante" | grader's own reason verbatim | hidden | hidden |
| In-scope, ambiguous keywords | `relevant` (attempt 2 after rewrite) | maybe | 🟢 green | "Rilevante" | grader's own reason | "Tentativi: 2/2" | "Query riformulata al 2° tentativo: …" |
| Out-of-scope (fotosintesi, Galileo…) | `irrelevant` (max attempts) | yes | 🔵 blue | "Adattamento riuscito" | structured KG vs external breakdown + grader's reason as Dettaglio | "Tentativi: 2/2" | "Query riformulata al 2° tentativo: …" |
| True KG gap (rare) | `irrelevant` (max attempts) | no | 🟡 amber | "Copertura limitata" | "verifica manualmente …" warning + grader's reason | "Tentativi: 2/2" | shown if rewrite was proposed |
| Grader LLM failed | `relevant` (fail-open) + sentinel reason | n/a | 🔴 rose | "Errore valutatore" | "Il valutatore non ha potuto giudicare … considera di rigenerare" | hidden (attempts=1) | hidden |
| Corrective-RAG OFF | n/a | n/a | n/a | (entire block hidden by template's outer guard) | — | — | — |

**Verification:**
- 11 unit tests in `tests/unit/test_retriever_payload_outcome.py` (`pytest -q tests/unit/test_retriever_payload_outcome.py` ⇒ all green): one per canonical outcome (success / adapted_with_hybrid / limited_kg_only / grader_error), one for the `_compute_retrieval_outcome` pure function called directly with all five branches, one for env-driven `retrieval_attempts_max` resolution, one for clamp behaviour on garbage env values, one for the `_grader_will_retry` invariant matching `should_retry_retrieval`'s logic, plus two backward-compat / shape-stability tests.
- Lints clean on all three touched files (`service.py`, `chat_retriever_card.html`, doc).
- Live re-run pending — will confirm: (a) the "fotosintesi" turn renders BLUE instead of red; (b) the ADHD turn renders GREEN unchanged; (c) one card per turn under both flag states even with retries.

---

#### #9.UX-4 — outcome misclassification (curated media ↔ hybrid) + domain-blind copy *(deferred — 2026-05-11 PM)*

**Status:** 🟡 **TODO — DEFERRED.** Discovered during a live smoke of *"crea una lezione sulla motivazione intrinseca"* on the **neuro** domain (2026-05-11 PM). Two pre-existing #9.UX-3 bugs were uncovered, both display-only. **No code change in this entry** — captured here so we can pick it up after the P5 workspace re-skin (the second fix touches `chat_retriever_card.html`, which gets re-skinned in P5.4 anyway, so we bundle them).

**Why both look identical to "neuro is missing from #9"** *(it isn't)*: corrective-RAG runs unconditionally on every domain — the grader log line `[RetrievalGrader] grade=ambiguous reason='Some core concepts of intrinsic motivation are present (Autonomy, Self-Determination, Competence) …'` for that smoke proves the node fired with the neuro state. What's broken is **outcome routing + copy**, not the grader itself.

##### Bug B1 — `_compute_retrieval_outcome` treats KG-curated media as a "hybrid retrieval" signal

**Symptom:** `motivazione intrinseca` on `neuro` triggered `grade=ambiguous`, `max_attempts=2` reached, **no hybrid retrieval** (writer log: `media=True, hybrid=False`, `external_resources=None`). The card should have rendered AMBER ("Copertura limitata"). Instead it rendered BLUE ("Adattamento riuscito").

**Root cause** — `src/aix/webui/agent/service.py` `_compute_retrieval_outcome()`:
```python
has_hybrid_media = (
    int(media_counts.get("articles") or 0)
    + int(media_counts.get("oer") or 0)
) > 0
if has_external or has_hybrid_media:
    return "adapted_with_hybrid"
return "limited_kg_only"
```
`media_counts.articles` + `media_counts.oer` count entries from the **KG-curated media DB** (curated catalogue baked next to KG concepts — videos, articles, OER from a curated table), NOT from the hybrid path (Wikipedia / Semantic Scholar / external resources fetched at runtime). The retriever's `Found curated media: 5 videos, 5 resources, 3 citations, 3 textbooks` log line is curated-side, but the function treats `3+3=6` as proof that hybrid kicked in, so any *ambiguous/irrelevant* turn that has curated media (= almost every turn) is mis-flagged as `adapted_with_hybrid`.

The Writer agent already has the canonical "did hybrid actually run" signal — it reads `external_resources` and prints `hybrid=True/False` accordingly. We should mirror that.

**Fix:** drop the `has_hybrid_media` proxy entirely; decide blue-vs-amber **purely** on `state.get("external_resources")` (the same field Writer reads). One-line change:
```python
if has_external:
    return "adapted_with_hybrid"
return "limited_kg_only"
```
Acceptance:
- [ ] `motivazione intrinseca` / `neuro` smoke renders AMBER ("Copertura limitata — verifica manualmente") with grader's reason quoted underneath
- [ ] `fotosintesi clorofiliana` / `udl` smoke (the original #9.UX-3 reference case) still renders BLUE ("Adattamento riuscito") — Wikipedia + papers populate `external_resources`, so `has_external=True`
- [ ] One unit test added in `tests/unit/test_retriever_payload_outcome.py`: *"ambiguous grade + curated media + empty external_resources → `limited_kg_only`"* (today this test would fail because the production code returns `adapted_with_hybrid`)
- [ ] Existing 11 tests still green (none of them combined `has_external=False` with curated `articles>0`, so the regression is genuinely uncovered)

##### Bug B2 — `chat_retriever_card.html` hardcodes "(UDL)" copy in the `adapted_with_hybrid` branch

**Symptom:** even when B1 is fixed and the BLUE outcome is legitimate (e.g. *"fotosintesi"* on `neuro` where the KG genuinely doesn't have the disciplinary content and hybrid does fill it in), the body copy reads:
> *"Il Knowledge Graph è specializzato in pedagogia inclusiva (UDL) e variabilità dell'apprendimento, non nel contenuto disciplinare specifico …"*

The `(UDL)` parenthetical was written assuming the only "out-of-scope" KG was UDL. On `neuro` the parenthetical is wrong: the neuro KG is neuroscience-grounded pedagogy, not UDL.

**Fix options** (pick at implementation time):
- **(a) Domain-aware copy** — add `domain: state.get("domain")` to `_build_retriever_payload`, then template branches:
  ```jinja
  {% if p.get('domain') == 'udl' %}
    pedagogia inclusiva (UDL) e variabilità dell'apprendimento
  {% else %}
    pedagogia neuroscientifica e variabilità dell'apprendimento
  {% endif %}
  ```
- **(b) Neutral copy** — drop the parenthetical entirely: *"Il Knowledge Graph è specializzato in pedagogia, non nel contenuto disciplinare specifico …"* Works for both domains, ships in 1 minute, slightly less explanatory.

Recommendation: **(a)** — keeps the explainability rich and is a 4-line template diff.

Acceptance:
- [ ] BLUE card on `udl` says "pedagogia inclusiva (UDL)"; BLUE card on `neuro` says "pedagogia neuroscientifica" *(or whatever the team agrees reads best in Italian)*
- [ ] No copy regression on `success` / `limited_kg_only` / `grader_error` outcomes (domain only feeds the `adapted_with_hybrid` branch)
- [ ] One unit/snapshot test asserts the domain-aware string appears for each domain × outcome combination that uses it

##### Why we defer

- **B1 is one line + one test**, but it's safer to land it in the same pass as B2 because the AMBER copy ("Copertura limitata") will become much more visible after B1 (today the bug hides AMBER behind a wrong BLUE).
- **B2 lives in `chat_retriever_card.html`**, which is on the P5.4 (Workspace) re-skin queue. We're rewriting that file anyway when we apply the warm-academic brand to the chat cards, so bundling B1+B2 with P5.4 avoids touching the partial twice.
- **Neither bug affects the agent graph, persisted state, or any other surface** — they are display-only on a card most teachers never reach (corrective-RAG default is `false` in `.env.example`; only `.env` flips it on).

##### Separate model-level observation (not a code bug, parked for later)

The same smoke also surfaced that the grader's *reason* claimed *"key elements like curiosity, flow state, mastery orientation, internal locus of control are missing"* — but the retriever logs prove those exact nodes WERE retrieved across the 8 planner sub-queries. The grader is judging from a top-N snapshot of the merged retrieval set, not the full 88-node fan-out, so it can be overly strict on broad in-scope queries. This is a **grader prompt / grader input** tuning question, not an outcome-routing question, and belongs in CORE 3.5 or a future #9.b *(grader recall calibration)* ticket — not in #9.UX-4.

---

#### #9.UX-5 — teacher-friendly coverage banner + domain-aware footer (CR OFF only) *(2026-05-13 AM)* — ✅ DONE

**Why:** with `AIX_CORRECTIVE_RAG_ENABLED=false` as the production default (set on 2026-05-12 PM after the corrective-RAG retry path was too slow on KG-covered queries — see CORE 3.5 `#LAT-7`), the retriever chat card lost ALL teacher-facing explainability: the entire `{% if p.retrieval_grade %}` block (#9.UX-3's outcome-driven copy) was hidden, leaving only raw counts ("9 nodes · 8 strategies · 5 videos · 8 articles"). For an in-scope query like *"ADHD"* the teacher had no way to tell *"this is on-topic, the system is confident"* from *"this is empty, the system will hallucinate"*. For an out-of-scope query like *"fotosintesi clorofiliana"* on `neuro`, same problem in reverse — no signal that hybrid retrieval was about to fill the gap with verified external sources.

**Design choice — unified message system across both flag states:** rather than building two parallel UX pipelines (one for CR ON, one for CR OFF), we treat **coverage tier** as the canonical signal (`nodes_count` derived from `state["retrieved_concepts"]`) and gate only on `retrieval_grade is None`. Concretely:
- **CR ON path (already shipped in #9.UX-3) is untouched.** The four outcome-driven cards (green / blue / amber / red) still render exactly as before.
- **CR OFF path (new in #9.UX-5)** renders a 3-tier coverage banner derived from `nodes_count` alone — no grader involvement.

**The three tiers (CR OFF only):**

| Tier | `nodes_count` | Color | Headline | Body copy |
|---|---|---|---|---|
| `healthy` | ≥ 5 *(tunable via `AIX_COVERAGE_HEALTHY_THRESHOLD`)* | 🟢 sage | "Ricerca completata sulla base **{domain_short}**" | "{recommendations_count} strategie selezionate da {nodes_count} concetti correlati, con {media_total} risorse multimediali già pronte all'uso." |
| `limited` | 1-4 | 🟡 amber | "Copertura parziale per questo argomento nella base **{domain_short}**" | "Trovate {recommendations_count} strategie specifiche. La lezione sarà integrata con strategie didattiche generali per restare completa e coerente." |
| `out_of_scope` | 0 | 🔵 info | "Questo argomento non è presente nella base **{domain_long}**" | "La lezione si baserà su conoscenze didattiche generali del nostro assistente e fonti esterne verificate ed integrate." |

**Why the blue (Tier 0) message promises external sources even with CR OFF:** the hybrid retrieval path (Wikipedia / OpenAlex / OER) is **decoupled from the corrective-RAG flag** — the Planner sets `RetrievalPlan.external_apis_needed=True` whenever the topic is out-of-scope for the KG, and `retrieve_node` honours that signal regardless of `AIX_CORRECTIVE_RAG_ENABLED`. The live 2026-05-13 fotosintesi smoke confirmed `external_resources` was populated even with CR OFF. So the Tier 0 message is structurally true, not aspirational.

**Domain-aware labels** (driven by new `_DOMAIN_LABELS` table in `service.py`):

| Domain | Short label *(headlines)* | Long label *(out-of-scope banner only)* |
|---|---|---|
| `udl` | "UDL" | "UDL (pedagogia inclusiva)" |
| `neuro` | "Neuro" | "Neuro" |
| *(unknown)* | the raw domain key | the raw domain key |
| `None` / empty | "il dominio attivo" *(graceful fallback)* | "il dominio attivo" |

**Footer:** unchanged on the CR-ON path, replaced on the CR-OFF path:
- *Before:* `Retriever · GraphRAG + curated media` *(internal-sounding English)*
- *After:* `Fonte: Knowledge graph della didattica {domain_short} + risorse multimediali curate` *(domain-aware, teacher-facing IT)*

**Threshold tunability:** the `healthy`/`limited` boundary is configurable via `AIX_COVERAGE_HEALTHY_THRESHOLD` (default `5`, clamped `1 ≤ N ≤ 50`). Lowering it to `3` is reasonable for narrow specialised domains where 3 highly-relevant concepts already suffice; raising it to `8-10` for very broad domains where 5 concepts is barely a starting point. Keeps the banner's semantics meaningful across future KGs without a code change.

**Files changed (landed 2026-05-13 AM):**
- *Edited:* `src/aix/webui/agent/service.py` — new `_DOMAIN_LABELS` constant, `_coverage_healthy_threshold()` env helper, `_resolve_domain_labels(domain)` pure function, `_classify_coverage_tier(nodes_count)` pure function; `_build_retriever_payload` now exposes 5 new fields (`domain`, `domain_label_short`, `domain_label_long`, `coverage_tier`, `media_total`). All computations are cheap and unconditional — the template gates rendering on `not p.retrieval_grade`, so CR-ON payloads carry the fields inert.
- *Edited:* `src/aix/webui/static/css/aix-brand.css` — new `.aix-coverage-banner` block + 3 variants (`--healthy` / `--limited` / `--out-of-scope`) reusing existing status-color tokens (`--aix-status-success-soft` / `--aix-status-warning-soft` / `--aix-status-info-soft`). Headline + body sub-elements share the warm-academic typography ramp already used by other agent cards.
- *Edited:* `src/aix/webui/templates/partials/chat_retriever_card.html` — new `{% if not p.get('retrieval_grade') and p.get('coverage_tier') %}` banner block (CR-OFF only) above the existing CR-ON outcome block; footer now renders domain-aware IT label when `domain_label_short` is present, falls back to the internal English label otherwise.
- *Edited:* `tests/unit/test_retriever_payload_outcome.py` — 4 new tests (15 total): coverage-tier classifier boundaries, threshold env-config + clamp, domain-label resolution (UDL / Neuro / unknown / None), end-to-end payload-carries-all-fields integration.

**Verification:**
- ✅ `pytest -q tests/unit/test_retriever_payload_outcome.py` — 15/15 green.
- ✅ `ReadLints` clean on all 4 touched files.
- ✅ Live 2026-05-13 smoke on `udl` domain:
  - `crea una lezione su adhd` *(in-scope, 9 KG nodes, CR OFF)* → 🟢 sage "Ricerca completata sulla base **UDL**" + footer "Fonte: Knowledge graph della didattica **UDL** + risorse multimediali curate". ✅
  - `crea una lezione su sintesi clorofiliana` *(out-of-scope, 0 KG nodes, CR OFF, hybrid landed)* → 🔵 info "Questo argomento non è presente nella base **UDL (pedagogia inclusiva)**" with the unified external-sources promise. ✅

**Backward compatibility:**
- CR-ON path (the entire #9.UX-3 outcome block) is **byte-identical** to pre-#9.UX-5 — the new banner is gated on `not p.get('retrieval_grade')`, so it never renders when grading ran.
- CR-OFF path was previously *no banner at all* (just raw counts); now renders one of three banners. No prior copy / no prior wiring is replaced — purely additive.
- The 5 new payload fields are inert on the CR-ON path (template ignores them).

**Sub-fix — `#9.UX-5 hotfix` — pre-seed `final_state` with `initial_state` so `domain` is non-empty *(2026-05-13 AM, hours after the initial land)*:**

The first live test post-land surfaced a regression: both the banner and the footer showed `"il dominio attivo"` (the graceful fallback) instead of `"UDL"` / `"Neuro"`. Root cause: in both SSE loops (`run_agent_stream` and `stream_agent_events`) `final_state` was initialised as `{}` at function scope, then mutated via `final_state.update(state_diff)` inside the LangGraph `astream(stream_mode="updates")` loop. State-diff chunks **only carry the fields a node actually mutated** — `domain` is a static input that no node ever touches, so it never appeared in any chunk, and `final_state.get("domain")` returned `None`, triggering the `_resolve_domain_labels(None) → "il dominio attivo"` fallback.

**Fix:** one new line in each SSE loop, immediately after the `create_initial_state(...)` call:
```python
final_state.update(initial_state)
```
This seeds `final_state` with the full static input set (including `domain`, `language`, `session_id`, the educational profile dict, the un-augmented `raw_user_turn`, etc.). Subsequent `final_state.update(state_diff)` calls overwrite the dynamic fields (retrieved concepts, plan, lesson_content, …) without ever touching `domain` (no node returns it), so the seeded value flows all the way through to `_build_retriever_payload`. Verified: ADHD smoke now reads `"Ricerca completata sulla base UDL"`. *(Tier 0 message also updated in the same pass: replaced "adattate al profilo educativo della tua classe" closer with "e fonti esterne verificate ed integrate" to match the unified design and reflect that hybrid retrieval is in fact running.)*

---

#### #9.UX-7 — media-panel re-ranking for out-of-scope queries *(future polish — tracked 2026-05-13 AM)*

**Status:** 🟡 **TODO — DEFERRED** *(low priority, post-CORE-3.5)*.

**Why:** the 2026-05-13 fotosintesi smoke surfaced a media-alignment issue **on the out-of-scope path only**. The Tier 0 banner correctly explains *"questo argomento non è presente nella base UDL: la lezione si baserà su conoscenze didattiche generali e fonti esterne verificate ed integrate"*, and the writer-agent's final lesson correctly cites Wikipedia/OpenAlex articles on chlorophyll/photosynthesis inline. **But** the right-hand `media` panel renders 5 curated YouTube videos + 5 curated articles whose subjects are pedagogical (e.g. UDL inclusive-teaching demos, executive-function articles) — pulled from Angelo's curated KG-side media pool — rather than disciplinary content on photosynthesis itself. From the teacher's POV the message says *"I went and found photosynthesis material"* but the media tiles show pedagogy.

**Not a bug in Angelo's media pool** — the pool correctly indexes media against KG concepts (which are pedagogical). The mismatch arises because the **retriever**, on an out-of-scope query, currently surfaces the top-N curated-media tiles regardless of how relevant they are to the query, and the **template** treats all media identically (no "external/hybrid first" ranking).

**Proposed fix at implementation time** (pick one — both are surface-only):
- **(a) Re-rank in `_build_retriever_payload`:** on `coverage_tier="out_of_scope"`, prefer `external_resources` (Wikipedia + OpenAlex hits, which ARE topical) over curated KG media in the media list. Curated media moves below a small "Materiali correlati dal Knowledge Graph" sub-header for transparency.
- **(b) Split the panel:** keep the existing curated tiles but add a new "Fonti esterne integrate" tile group fed from `external_resources` at the top of the media column. Teacher sees both, ordered by topical relevance.

Recommendation: **(a)** — single source of truth (the existing media panel), simpler template diff, matches the unified-banner narrative ("the lesson will integrate verified external sources" → so those should be the prominent media). **(b)** is acceptable if user-testing shows teachers want the curated tiles to remain prominent for cross-pollination ("here's photosynthesis from Wikipedia + here's some UDL teaching strategies for science topics from our base").

**Why we defer:**
- Display-only on a single (out-of-scope) path. The lesson itself (and its citations) is already correct.
- Touches `_build_retriever_payload` *and* `chat_retriever_card.html` *and* possibly the media-panel partial — bigger surface than the in-place banner fix, want to bundle with the next chat-card pass.
- Curated-media-on-out-of-scope is *some* signal even if imperfect (the teacher can use the curated UDL videos as classroom-management context for a disciplinary lesson). Not actively harmful.

**Acceptance (when picked up):**
- [ ] `fotosintesi` / `udl` smoke: media panel renders Wikipedia + OpenAlex articles on photosynthesis **first**, before any curated KG-side media tiles
- [ ] `ADHD` / `udl` smoke *(in-scope, healthy coverage)*: media panel is **unchanged** from today (curated tiles stay prominent — they ARE topical for in-scope queries)
- [ ] No regression to the writer-agent's inline citation behaviour (Writer reads `external_resources` directly — not affected by panel rendering)
- [ ] One unit/snapshot test asserts the ranking branches on `coverage_tier`

---

### Subtask 10: Conversation Memory (LangGraph Checkpointer + Multi-Turn Chat)
**Priority:** 🔴 Urgent | **Effort:** 5-7d *(revised up from 3-5h on 2026-05-01)* | **Assignee:** LM

**Description:**
Add LangGraph checkpointing AND the chat-workspace plumbing for true multi-turn conversation. Today every `POST /webui/lesson/{id}/run` resets `lesson_plan_md` and re-executes the full pipeline from scratch — the teacher cannot say *"ora adattala per ADHD"* or *"rigenera con durata 30 min"* without losing the previous answer. On top of that, after an agent run completes the chat input remains stuck in the disabled `running` state, so even a *new* lesson request requires the teacher to either reload or click *"Nuova lezione"*.

Three independent issues stack on top of each other; this subtask resolves all three.

**Discovered during #6.6 P2 phase 2 + post-completion smoke (2026-04-26 → 2026-05-01):**

| Layer | Problem | Severity |
|---|---|---|
| 1. SSE `done` event in `webui.agent.service._stream_event_to_sse` | Only OOB-swaps the `lesson-card-loading` placeholder; the `chat_input` partial stays at `lesson.status="running"` (disabled) | Bug |
| 2. `GET /webui/lesson/{id}/card-fragment` | Returns only `chat_lesson_card.html`; never re-renders `chat_input.html` | Bug |
| 3. `POST /webui/lesson/{id}/run` semantics | Resets `lesson_plan_md`, no thread/conversation state — every run is a full re-execution. `session_id` exists in `AgentState` but is never used for persistence. | Architecture gap (THIS subtask) |

Layers 1+2 are quick fixes (1-2h total), but until layer 3 is solved (LangGraph checkpointer), *"follow-up"* really means *"full re-run with no memory of the previous answer"*.

**Root cause for layer 3** — `src/aix/agent/graph/lesson_planner_graph.py` line 65:
```python
compiled = workflow.compile()  # ← No checkpointer passed!
```

**Architecture — 4-layer memory model:**

```
L0 — Working memory (one ainvoke()): AgentState — already exists ✅
L1 — Short-term thread memory: LangGraph AsyncSqliteSaver in dev → AsyncPostgresSaver in #15.a
       thread_id = lesson.id; full message history + AgentState snapshots
L2 — App-DB rendered messages: NEW lesson_messages table (UI-facing CQRS view)
       (id, lesson_id, role, content_md, agent_kind, meta_json, created_at)
L3 — Long-term cross-session preferences: deferred to #16 (LangGraph Store API)
```

**Phased delivery:**

| Phase | Scope | Effort | Status | Depends on |
|---|---|---|---|---|
| **10.1 — Chat input OOB fix** | Modify `webui.agent.service._stream_event_to_sse` for `kind="done"` (and `kind="error"`) to also append the re-rendered `chat_input.html` as `hx-swap-oob="outerHTML"` on a stable `#chat-input-wrapper` root. Add an `_oob: bool` flag to the partial to gate the `hx-swap-oob` attribute. Removes the dead-end UX even before multi-turn lands. | 1-2h | ✅ **DONE 2026-05-01** | None |
| **10.2 — `AsyncSqliteSaver` checkpointer** | `pip install langgraph-checkpoint-sqlite`. New `aix.agent.graph.checkpointer.get_checkpointer()` async singleton with graceful degradation if package is missing. New `build_lesson_planner_graph_async()` compiles with checkpointer attached; old sync `build_lesson_planner_graph()` preserved for backward compat. Pass `config={"configurable": {"thread_id": str(lesson.id)}}` on every `astream()` / `ainvoke()` call in `webui.agent.service` and `aix.api.routes.agent`. DB at `data/agent_threads.db`, override via `LANGGRAPH_CHECKPOINTER_URL`. | 4-6h | ✅ **DONE 2026-05-01** | None |
| **10.3 — `lesson_messages` table + multi-turn `/run`** | New SQLAlchemy model `LessonMessage(id, lesson_id, role, content_md, turn_index, agent_kind, meta_json, checkpoint_id, created_at)` with compound index on `(lesson_id, turn_index)`; auto-created by `Base.metadata.create_all` (Alembic stays in its planned home at #6.6 P5/CORE 6). `POST /run` auto-detects `mode` from `lesson.status` (`draft` → `new`, `complete`/`error` → `follow_up`); persists user `LessonMessage` on entry, assistant `LessonMessage` on `kind="done"`. Backward-compat **backfill on first follow-up** for pre-#10.3 lessons (turn 1 imported from `lesson.teacher_query`/`lesson.lesson_plan_md`). Service layer loads prior turns and injects them as a service-layer-augmented `teacher_query` (Italian/English-aware) so EVERY agent in the pipeline sees prior context without bespoke prompt edits. New `chat_history.html` partial renders the full transcript (user bubbles + per-turn lesson cards); `chat_conversation.html` falls back to the legacy single-bubble layout when no `lesson_message` rows exist. Chat input copy refreshed: "Continua la conversazione…" / "Invia" instead of "Nuova". **Time-travel regenerate deferred** to a follow-up — V1 multi-turn delivers `mode={new, follow_up}` with auto-detection. | 2-3d | ✅ **DONE 2026-05-01** *(regenerate via time-travel deferred to V2)* | 10.2 |
| **10.4 — Memory window strategy** | Turn-based summary-buffer pattern: when conversation has more than `AIX_CONVERSATION_WINDOW_TURNS` (default 4) prior turns, summarise everything older via a cheap LLM call (temperature 0.2, max_tokens 600, language-aware) into `AgentState.conversation_summary`; keep the recent window verbatim in `conversation_history`. Both flow into the augmented query — summary first, then recent turns, then current request. Failures fall back to no windowing with a logged warning so the run never crashes on a transient LLM error. Tunable via env var. | 1d | ✅ **DONE 2026-05-01** | 10.3 |
| **10.5 — `AsyncPostgresSaver` migration** | Rolled into **CORE 4 #15.a**. Runtime backend selection now uses `AsyncPostgresSaver(pool)` when `LANGGRAPH_DATABASE_URL` is a Postgres URL and `AsyncSqliteSaver` for zero-config dev. Same schema shape, same graph/API path, just durable + multi-instance in production. | rolls up to #15.a | ✅ DONE locally *(2026-05-16; restored 2026-06-06)* | #10.3 |

**Acceptance Criteria:**
- [x] **10.1:** `chat_input` re-enables automatically after the SSE stream closes; no page reload required; verified across `complete` and `error` terminal states *(stable `#chat-input-wrapper` root + `_oob` flag in partial; `_stream_event_to_sse` appends OOB-rendered partial on `done`/`error`)*
- [x] **10.2:** Graph compiled with `AsyncSqliteSaver` (lazy singleton, `data/agent_threads.db`); `thread_id` passed on every `astream`/`ainvoke`; graceful degradation if `langgraph-checkpoint-sqlite` is missing — single-turn behaviour preserved with a logged warning
- [x] **10.3:** New `lesson_message` table; `/run` auto-detects `mode={new, follow_up}` from `lesson.status`; backfill on first follow-up imports turn 1 from legacy `lesson.teacher_query`/`lesson.lesson_plan_md`; service-layer augmented query exposes prior turns to all agents (Planner sees the conversation context, not just the latest user query)
- [ ] **10.3:** *"Rigenera"* button branches from the previous user turn (LangGraph native time-travel via `get_state_history()` + `aupdate_state(checkpoint_id=...)`), not a full re-run *(deferred to V2 — V1 ships `mode={new, follow_up}`; regenerate today is a re-run on the same thread, not time-travel)*
- [x] **10.4:** Long-conversation windowing wired (default 4 turns retained verbatim, older turns summarised via LLM, summary lives in `AgentState.conversation_summary`); `AIX_CONVERSATION_WINDOW_TURNS` env-tunable; failure path falls back to full history with logged warning
- [x] **10.4:** No regression in single-turn workflows — first-turn path is byte-identical to pre-#10 (history empty → augmenter returns `raw_query` unchanged → windowing short-circuits with `(None, [])`)
- [ ] Update `apps/cli/run_agent.py` to support session mode (interactive REPL with shared `thread_id`) *(deferred — `LessonPlannerPipeline.run()` already passes `thread_id` config when `session_id` is provided; interactive REPL is a CLI ergonomics ticket)*
- [x] Public API endpoint (#7) accepts and passes `session_id` / `thread_id` — `stream_agent_events` builds `thread_config(session_id or "ephemeral-<uuid>")` and threads it through `astream`

**Investigation note — "is LangGraph checkpointer the best practice?":** Yes, definitively, for any system already built on LangGraph (which this is — `langgraph==0.5.4` in `requirements.txt`). The checkpointer abstraction is the canonical answer used at production scale by Replit, Klarna, Uber, and Anthropic for stateful agents. Time-travel, branching, human-in-the-loop interrupts (#19), and concurrent-read safety are all built-in. **Alternatives considered and rejected:** *(a)* manual `lesson_messages` history fed to the LLM as context — 3× the code, no time-travel, more bugs; *(b)* Redis-only sessions — loses time-travel, in-memory data loss on restart; *(c)* Mem0 / Zep / Letta dedicated memory frameworks — overkill for V1, useful for L3 long-term memory under #16; *(d)* OpenAI Assistants API threads — vendor lock-in (we use OpenRouter). Langfuse (added separately by AG for prompt management) is observability, not state — complementary, not a substitute.

**Investigation note — "will it work with PostgreSQL in production?":** Yes, by design. LangGraph ships multiple checkpointer backends with **identical APIs**: `MemorySaver` (tests), `AsyncSqliteSaver` (dev — this subtask), `AsyncPostgresSaver` (prod — #15.a), `RedisSaver` (community). Migration is a 1-line swap of the saver class; everything else (graph compile, `astream`, `aget_state`, `get_state_history`, `aupdate_state`) stays byte-identical. The Postgres backend uses `psycopg` 3 with connection pooling, creates 3 idempotent tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes` — all with `JSONB` columns), and is battle-tested at billion-event scale.

**Why L2 (`lesson_messages` table) on top of L1 (checkpointer) — production CQRS pattern:** The checkpointer stores the *agent's* view (msgpack-serialized `AgentState` snapshots — fast for the agent, opaque for everything else). The UI needs the *user's* view (timestamped messages with rendered markdown, attachment chips, agent badges, scores). Reconstructing the UI view from msgpack on every page load is slow, version-fragile, and SQL-unfriendly (no `LIKE '%fotosintesi%'` search, no analytics joins, no PDF export queries). We keep both in sync via the SSE event emitter — the same code path that pushes events to the browser also writes a `lesson_messages` row. Standard pattern, well-understood, scales to millions of conversations.

**Files actually changed (landed 2026-05-01):**
- *NEW:* `src/aix/agent/graph/checkpointer.py` — async-singleton `get_checkpointer()` with graceful degradation; `thread_config()` helper; `LANGGRAPH_CHECKPOINTER_URL` override
- *Edited:* `src/aix/agent/graph/lesson_planner_graph.py` — added `build_lesson_planner_graph_async()` (with checkpointer) alongside the original sync builder (preserved for backward compat); `LessonPlannerPipeline._get_graph_async()`; `run()` now passes `thread_config()` to `ainvoke`
- *Edited:* `src/aix/agent/graph/state.py` — `AgentState.conversation_history` + `conversation_summary` fields; `create_initial_state` accepts both as kwargs (default `None` → fully backward compatible)
- *Edited:* `src/aix/webui/agent/service.py` — `_load_conversation_history` (CQRS reader), `_augment_query_with_history` (Italian/English-aware service-layer prompt augmentation), `_persist_assistant_turn` (writer of the assistant `LessonMessage`), `_maybe_window_history` + `_summarise_history` (#10.4 turn-based summary buffer); both `run_agent_stream` and `stream_agent_events` thread `thread_id` config and the windowing pipeline
- *Edited:* `src/aix/webui/templates/partials/chat_input.html` — stable `#chat-input-wrapper` root with conditional `hx-swap-oob`; updated copy for follow-up state ("Continua la conversazione…" / "Invia" with paper-plane icon)
- *NEW:* `src/aix/webui/templates/partials/chat_history.html` — chronological transcript renderer (user bubble + per-turn lesson card with `data-turn-index`)
- *Edited:* `src/aix/webui/templates/partials/chat_conversation.html` — multi-turn rendering with legacy single-bubble fallback when `messages` is empty; lesson card duplication guard on `complete` state
- *Edited:* `src/aix/webui/lessons/routes.py` — `/run` mode auto-detection + user-message persistence + backfill for pre-#10.3 lessons; `_load_chat_messages` helper; `lesson_show` and `/run` template responses pass `messages` context; OOB chat_input on `done`/`error` SSE events
- *Edited:* `src/aix/webui/lessons/models.py` — `LessonMessage` model (UUID PK, `lesson_id` FK with `CASCADE`, `role`, `content_md` `Text`, `turn_index`, `agent_kind`, `meta_json`, `checkpoint_id`, `created_at`); compound index on `(lesson_id, turn_index)`; `Lesson.messages` relationship with `cascade="all, delete-orphan"`
- *Edited:* `requirements.txt` — added `langgraph-checkpoint-sqlite>=2.0,<3.0`

**Deferred to V2 (intentional cuts, recorded for future work):**
- `apps/cli/run_agent.py` interactive REPL mode reusing `thread_id` (the underlying pipeline already supports it via `LessonPlannerPipeline.run(session_id=...)`; the wrapper is CLI ergonomics work)
- *"Rigenera"* button using LangGraph native time-travel (`get_state_history()` + `aupdate_state()`) — V1 ships `mode={new, follow_up}` only; today's "Rigenera" replays on the same thread rather than branching from a past checkpoint
- Per-agent prompt-level integration of `conversation_history` field — V1 uses service-layer augmented query, which is sufficient for the user-visible UX and avoids regression risk on the existing prompts
- Alembic migrations — the existing `Base.metadata.create_all` + dev-hotpatch pattern in `aix.webui.db.init_db()` handles the new table cleanly; Alembic stays in its planned home at #6.6 P5 / CORE 6 where real schema migrations land

**Manual install + smoke required after pulling this change:**
```
pip install "langgraph-checkpoint-sqlite>=2.0,<3.0"
# Restart the FastAPI app — Base.metadata.create_all auto-creates lesson_message
# Smoke: open a complete lesson, type a follow-up like "ora adattala per ADHD" → input stays active, history shows both turns
```
- *Edited:* `requirements.txt` (`langgraph-checkpoint-sqlite>=2.0`)

**Depends on:** None
**Blocks:** **#15.a (PostgresSaver upgrade)**, **#15.b (Conversation-Memory Hardening — no UX)**, #16 (Long-Term Memory), #19 (Human-in-the-Loop Interrupts)

---

### Subtask 11: Observability (Agent JSON Parse Hardening + LangSmith / Langfuse Integration)
**Priority:** 🟠 High | **Effort:** 4-6h | **Assignee:** LM

**Description:**
Two phases delivered as one ticket because they share the same goal — *"give us real, trustworthy traces of what the agent did"* — and because phase 11a is a hard prerequisite for phase 11b: tracing a pipeline whose Planner and Critic silently fall through to hardcoded defaults produces traces that look successful but contain no real classification or critique. Phase 11a fixes the silent fall-through; phase 11b lights up the dashboard against an agent that's actually answering for itself.

**Discovered during #6.6 P2 phase 2 smoke testing (2026-04-26):** every recorded run shows the same pattern in the logs from `planner_agent.py` and `critic_agent.py`:

```
HTTP Request: POST openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"
[PlannerAgent] Failed to parse JSON response: Expecting value: line 1 column 1 (char 0)
[CriticAgent]  Failed to parse JSON response: Expecting value: line 1 column 1 (char 0)
```

OpenRouter returns 200 OK with an empty (or non-JSON) body for both agents' completion calls. Each agent has a hardcoded fallback (`intent=lesson_creation, confidence=LOW, scope=in_scope 100%` for the Planner; `approved=True, average_score=3.5, critique="Approved due to parsing error"` for the Critic). The fallback masks the failure — the run completes "successfully" — but the Planner card and Critic card in the webui display fallback values, not real model output. This means the Critic is currently a **no-op approval gate**, and we can't trust its score.

**Phase 11a — Agent JSON Parse Hardening** ✅ **DONE (code) — 2026-05-10** (must land before 11b):
- [x] Defensive multi-strategy `_extract_json` helper already lives in both `planner_agent.py` and `critic_agent.py` (strips markdown fences, finds outermost `{...}`, retries on substring)
- [x] Both agents' OpenRouter calls now pass `response_format={"type": "json_object"}` via `app_config.openai.build_completion_kwargs(json_mode=True)` (Gemini 2.0 Flash + GPT-4o + Claude all honour this; legacy fallback in `_extract_json` covers models that don't)
- [x] On real parse failure both agents now emit a structured log line `event=agent_parse_error agent={planner|critic} raw_response_preview=…` so the failure is observable in CloudWatch / GlitchTip / future Langfuse traces (no longer silently masked by the fallback dict)
- [x] **Critic fallback is now configurable** via `AIX_CRITIC_PARSE_ERROR_BEHAVIOR`:
  - `approve` (default — legacy behaviour, lesson ships with `decision=APPROVE` + `summary="Approved due to parsing error"`); chosen as default to preserve byte-identical behaviour for everyone who hasn't yet flipped 11a's downstream defaults
  - `revise` — forces one revision pass with a `[parse_error]` marker in `revision_instructions` (recommended once parse-error rate is verified to be near zero with json_mode on)
  - `raise` — re-raises the `JSONDecodeError`; intended for staging where every parse failure should be investigated
- [x] **Planner keeps backward-compatible degraded plan** on parse failure (intent=`lesson_creation`, confidence=`LOW`, scope=`in_scope` 100%) — now coupled with the structured log so the silent path is at least loud in observability
- [x] **5-run smoke test** lives in `tests/unit/test_json_parse_hardening.py` — asserts that with valid JSON responses **0 of 5 runs** trigger the parse-error fallback (the canonical regression guard for the literal string "Approved due to parsing error")
- [x] **Unit tests cover:** `json_mode=True` is forwarded to OpenAI for both agents; default Critic fallback (`approve`) preserves legacy behaviour; opt-in `revise` mode forces revision; Planner's degraded plan is unchanged on parse failure
- [ ] **Live-LLM smoke against real OpenRouter** (run 5 lessons end-to-end and grep logs for `agent_parse_error` to confirm json_mode actually drops the rate to ~0) — **deferred** until #11b Langfuse traces are wired so we can measure the rate directly from a dashboard rather than ad-hoc grep
- [ ] *(Carried — not yet wired)* Translate the parse-error event into a `kind="error"` SSE in `webui.agent.service.run_agent_stream` so the UI can render `<wa-callout variant="warning">`. Currently the structured log fires but the SSE stream still ships the legacy fallback values to the user. Tracked as a follow-up because changing the SSE kind has UX implications (lesson disappears) that need product sign-off.

**Files changed for 11a (landed 2026-05-10):**
- *Edited:* `src/aix/agent/agents/planner_agent.py` — `json_mode=True` on the completion kwargs; structured `event=agent_parse_error agent=planner` log on `JSONDecodeError`; degraded `RetrievalPlan` fallback preserved
- *Edited:* `src/aix/agent/agents/critic_agent.py` — `json_mode=True`; structured log; new `AIX_CRITIC_PARSE_ERROR_BEHAVIOR` env-gated branching (`approve` / `revise` / `raise`)
- *NEW:* `tests/unit/test_json_parse_hardening.py` — mock-based suite (json_mode forwarding × Planner/Critic; default-fallback regression × Planner/Critic; revise opt-in; 5-run no-fallback smoke)
- *Edited:* `.env.example` + `.env` — documented `AIX_CRITIC_PARSE_ERROR_BEHAVIOR` (default `approve` = legacy behaviour)

**Phase 11b — Tracing dashboard** (depends on 11a):
- [x] **Foundation LANDED by AG (2026-04-2X)** — Langfuse Python SDK pinned in `requirements.txt` (`langfuse>=2.0.0`); Langfuse-backed prompt management lives in `src/aix/domains/langfuse_prompts.py` with `scripts/ops/seed_langfuse_prompts.py` to seed Langfuse with the canonical prompts; `docs/prompts/langfuse_prompts_reference.md` documents the prompt registry; integration referenced in `src/aix/domains/{base_config,neuro_domain,udl_domain}.py`. Note: this is the **prompt-management** half of Langfuse; the **tracing-dashboard** half (below) is still TODO.
- [ ] Add `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` (or Langfuse equivalent — `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`) to `.env` and `.env.example`
- [ ] Verify trace tree appears in dashboard for the full pipeline (planner → retriever → writer → critic), with one node per agent step
- [ ] Verify cost tracking (tokens per node, $ per request) and latency breakdown
- [ ] **Sanity check:** the 5 smoke-test runs from 11a now show *real* planner classifications and *real* critic critiques in the trace payloads — not the fallback defaults
- [ ] Document setup in `README.md` and `docs/runbooks/GraphRAG_Data_Pipeline_Guide.md`

**Why merged into one ticket and not split:** keeping 11a inside #11 (rather than spinning it off as a separate subtask) avoids double-counting in CORE 2 effort estimates, and makes it impossible to ship Langfuse before the agents are honest — the dependency is structural, not just temporal.

**Depends on:** LangSmith API key (free tier: 5K traces/month) OR self-hosted Langfuse instance. Phase 11a depends on **#6.6** (the empty-body issue is only visible in real runs, which #6.6 enables)
**Unblocks:** CORE 3 (#13, #14, #17, #18 all need traces for measurement; all four also need a Critic that produces real critique data — without 11a, #14 *Citation Grounding* in particular would build on a hallucinated approval signal)

---

### Subtask 12: SSE Streaming Hardening (backpressure + reconnect)
**Priority:** 🟠 High | **Effort:** 2-3h | **Assignee:** LM

**Description:**
SSE streaming itself is **already delivered** by #6.6 (HTML fragments over `text/event-stream` to the htmx webui) and exposed in JSON form by #7. This task is the *hardening* pass: make the stream production-grade for unstable mobile networks, slow proxies, and long lessons that exceed connection idle timeouts.

Protocol decision (SSE) is final per ADR-0001. WebSockets and NDJSON were considered and rejected (uniformly worse fit for one-way agent → browser flow on htmx + Mercure-style infra).

**Acceptance Criteria:**
- [ ] **Heartbeats:** server emits `: heartbeat\n\n` comments every 15s to keep proxies (Caddy/Nginx/Traefik) from closing idle connections
- [ ] **`Last-Event-ID` resume:** every event has a monotonically increasing `id:` field; on reconnect, the server replays missed events from a Redis-backed buffer (TTL 10 min)
- [ ] **Backpressure:** `asyncio.Queue(maxsize=N)` per session; if the client lags, the server drops *token* events (not tool/critic events) and emits a `truncated` event marker
- [ ] **Proxy headers:** `X-Accel-Buffering: no`, `Cache-Control: no-cache`, `Connection: keep-alive` on every SSE response
- [ ] **Reconnection UX:** htmx `sse-ext` handles automatic reconnect natively; verify the page recovers gracefully after a forced 30s offline window
- [ ] **Optional Mercure path:** if/when AixLearning embed (CORE 6) needs cross-app notifications, document the publish-to-Mercure-hub bridge (the FEM `aixlearning` repo already runs Mercure in production per their *"init mercure infra"* commit)
- [ ] **Doc:** event taxonomy documented in `docs/api/agent_sse_events.md` (event types, payload shapes, ordering guarantees)

**Depends on:** #6.6 (initial SSE implementation), #7 (JSON SSE variant)
**Notes vs. previous draft:** scope narrowed from "build SSE from scratch" to "harden the SSE that #6.6 ships"; effort revised 4-6h → 2-3h.

---

### Subtask 13: Query Decomposition (Multi-Hop Reasoning)
**Priority:** 🟠 High | **Effort:** 4-5h | **Assignee:** LM

**Description:**
Add query decomposition for complex multi-faceted queries. Currently the Planner generates flat, independent search queries. Should generate a DAG of sub-queries with dependencies, where later queries use results from earlier ones (e.g., "find strategies that address BOTH ADHD and dyslexia" depends on first retrieving each separately).

**Acceptance Criteria:**
- [ ] Planner outputs sub-queries with `depends_on` field
- [ ] Retriever executes independent queries in parallel, dependent queries sequentially
- [ ] Cross-referencing logic for queries that combine multiple results
- [ ] Test with complex query: "Lezione che integri strategie ADHD e dislessia con principi UDL"
- [ ] Measure quality improvement vs flat queries via #11 traces

**Depends on:** #11 (Observability — needed to measure quality lift)

---

### Subtask 14: Citation Grounding & Source Attribution
**Priority:** 🟡 Medium | **Effort:** 3h | **Assignee:** LM

**Description:**
Add source attribution to Writer output (each strategy cites which KG node it came from) and hallucination scoring in Critic (semantic similarity between output and retrieved context). Currently Writer can hallucinate strategies not in KG and nobody catches it. Pairs naturally with the explainability layer already in `feature/openrouter` — frontend can render a "source" badge on each lesson element.

**Acceptance Criteria:**
- [ ] Writer prompt requires `[Source: KG]` or `[Source: External]` markers
- [ ] Critic penalizes unsourced claims in scoring
- [ ] Post-generation grounding score (cosine similarity output vs context)
- [ ] Flag responses with grounding score < 0.5
- [ ] Surface citations in API response so frontend can highlight them

**Depends on:** #11 (Observability — for grounding score telemetry)

---

### Subtask 15.a: State Checkpointing (`AsyncSqliteSaver` → `AsyncPostgresSaver` migration)
**Priority:** 🟡 Medium | **Effort:** 5-8h | **Assignee:** LM
**Note:** First subtask of **CORE 4 — Personalization**. Depends on CORE 2 #10 already being live (`AsyncSqliteSaver`-backed, dev). Triggered alongside the broader app-wide SQLite → PostgreSQL migration in #6.6 P6 (Hetzner deploy). *Originally tracked as #15; renamed to 15.a on 2026-05-09 when #15 was split into 15.a (this — backend migration), 15.b (no-UX memory hardening), and 15.c (deferred memory UX V2). See 15.b/15.c below; nothing in this subtask's description, acceptance criteria, or dependencies has changed — only the numbering.*

**Status update 2026-06-06:** ✅ DONE locally and restored after the repo-sync conflict/reset. `src/aix/agent/graph/checkpointer.py` now selects `AsyncPostgresSaver` when `LANGGRAPH_DATABASE_URL` / `LANGGRAPH_CHECKPOINTER_URL` is a Postgres URL and falls back to `AsyncSqliteSaver` for local dev. The production deployment plan records the original 2026-05-16 smoke against Postgres 16 (`setup()` table bootstrap + `aput`/`aget_tuple` round-trip); the code was rewritten/restored to that contract after the SQLite-only file reappeared during the Angelo branch sync.

**Description:**
Swap `AsyncSqliteSaver` (dev — set up in CORE 2 #10) for `AsyncPostgresSaver` for production-grade durable conversation memory across pod restarts, multi-instance deployments, and concurrent-request safety.

The migration is intentionally backend-agnostic because LangGraph's `BaseCheckpointSaver` abstraction exposes the same `astream` / `aget_state` / `get_state_history` / `aupdate_state` API on both backends. The app now uses an env-driven selector rather than a hardcoded saver class: Postgres in production, SQLite in dev. The data shape (3 tables: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` — all `JSONB` columns) is created idempotently via `await saver.setup()`.

**Why this is "Medium" priority and not blocking:** the dev `AsyncSqliteSaver` set up in #10 is already durable (writes to `data/agent_threads.db` on disk); it just doesn't support multi-instance horizontal scaling. The migration becomes urgent only when we deploy multiple uvicorn replicas or hit SQLite write-lock contention under load — neither is a near-term concern.

**Acceptance Criteria:**
- [x] `langgraph-checkpoint-postgres` (`>=2.0`) installed / locked.
- [x] PostgreSQL connection configured through separate `LANGGRAPH_DATABASE_URL` env var; `LANGGRAPH_CHECKPOINTER_URL` remains as a legacy/dev override.
- [x] `psycopg_pool.AsyncConnectionPool` used for `AsyncPostgresSaver` in production; max pool size tunable via `LANGGRAPH_PG_POOL_MAX` (default 20).
- [x] Runtime selector uses `AsyncPostgresSaver(pool)` in production and `AsyncSqliteSaver` in dev, preserving the same graph/API path.
- [x] Idempotent table setup (`await saver.setup()`) runs during lazy checkpointer initialisation.
- [x] Local Postgres 16 smoke recorded in the production deployment plan: table bootstrap + `aput`/`aget_tuple` round-trip.
- [x] Historical SQLite checkpoint migration decision recorded as intentionally skipped for the internal pilot; production starts with empty conversation memory and empty WebUI data.
- [ ] Final Linux VM / CD smoke: verify Postgres backend selected from deployed `LANGGRAPH_DATABASE_URL`.
- [ ] Crash recovery test: kill process mid-pipeline, restart, verify resume from last node via `aget_state(config)` and SSE replay of unfinished events.
- [ ] Concurrent-request stress test: 20 parallel runs across 5 different threads, no row-lock contention.
- [ ] OpenAPI strictly-additive regression baseline still green (no public-API surface change).

**Depends on:** PostgreSQL database (provisioned by ops in #6.6 P6 / CORE 6), **#10 (Conversation Memory — `AsyncSqliteSaver` must be live first)**
**Unblocks:** #16 (Long-Term Memory — needs PostgresSaver-backed thread store + LangGraph Store API for L3 cross-session preferences), #19 (Human-in-the-Loop — needs durable interrupt state across pod restarts)

---

### Subtask 15.b: Conversation-Memory Hardening — Production-readiness without new UX
**Priority:** 🟠 High | **Effort:** 2-3h | **Assignee:** LM
**Promoted from the 2026-05-09 Point A investigation** ("can we improve conversation memory without adding new UX buttons?"). Recorded here so that *both* paths — the no-UX hardening (this 15.b) and the UX-driven V2 features (15.c below) — live under a single CORE 4 home.

**Why this exists as a separate ticket:** CORE 2 #10 shipped phases 10.1–10.4 (LANDED 2026-05-01) — the four-layer memory model is in: AgentState (L0), `AsyncSqliteSaver` checkpointer (L1), `lesson_messages` CQRS table (L2), summary-buffer windowing (L4 windowing within L1+L2). What's deferred to 10's V2 (time-travel "Rigenera", branching, edit-and-rerun, CLI REPL) all require **new UX surfaces**. The product decision on 2026-05-09 was *"hold UX flat for V1 — don't add buttons until pilot signal"*. That decision left a small, valuable set of **server-only** improvements that make today's two-surface model ("Continua la conversazione…" + "Rigenera") production-grade without changing anything teachers see. Those land here.

**Description:**
Stabilise CORE 2 #10's memory plumbing so it's production-grade for the V1 UX contract (current "Continua la conversazione…" follow-up input + "Rigenera" full-rerun button). Pure backend hardening — zero changes to the rendered UI, zero new buttons, zero schema migrations. Forward-compat: lays the data groundwork for 15.c (UX V2) so it becomes a pure UI ticket later, with no additional checkpointer or CQRS work needed.

**Acceptance Criteria:**
- [ ] **15.b.1 — Persist `LessonMessage.checkpoint_id` on every assistant turn.** Column already exists (CORE 2 #10.3); today it's always `NULL`. Read it from `AgentState` (or from `aget_state(thread_config)` on the compiled async graph) at the moment we write the assistant `LessonMessage` row in `webui/agent/service.py::_persist_assistant_turn`. Pure forward-compat for 15.c — without this, time-travel "Rigenera" needs a one-off backfill migration when it lands.
- [ ] **15.b.2 — Verify `[checkpointer] AsyncSqliteSaver ready` log fires on app startup.** The codepath in `aix.agent.graph.checkpointer.get_checkpointer()` already gracefully degrades to `None` (no checkpointer, single-turn behaviour) if `langgraph-checkpoint-sqlite` is missing. Add a startup smoke test that asserts the saver loaded, plus a single explicit log line at INFO level so ops can see it in stdout. Today the warning fires on degradation but there's no positive-path confirmation — leaves us blind in production.
- [ ] **15.b.3 — Expose `AIX_CONVERSATION_WINDOW_TURNS` (default 4) in admin settings.** Already env-tunable today via `webui.agent.service._maybe_window_history`; add it to the admin settings UI as a read-only-or-editable knob (depending on auth tier). Lets ops tune cost/quality on real traffic without redeploying.
- [ ] **15.b.4 — Memory-usage telemetry into Langfuse (when #11b is live).** In `_maybe_window_history` and `_summarise_history`, emit a small structured payload per turn: `{turns_kept_verbatim, turns_summarised, summary_token_count, augmented_query_token_count}`. Light up the existing Langfuse trace span (#11b) — no new instrumentation framework. Lets us spot when summaries grow stale or when augmented queries balloon past sensible limits.
- [ ] **15.b.5 — Dedup augmented prompt on first turn (no-history short-circuit).** In `_augment_query_with_history`, if `conversation_history` is empty AND `conversation_summary` is None, return the raw query verbatim — skip the "previous turns:" wrapper template entirely. Today the augmenter already handles empty history, but it still emits the wrapper string with empty content (a few wasted tokens per first-turn run). Tiny token saving, zero risk.
- [ ] No regression on existing CORE 2 #10 tests; first-turn path stays byte-identical.
- [ ] No new UX surfaces introduced — no new buttons, no new partials, no template changes outside the admin settings page in 15.b.3.

**Out of scope (intentionally — these belong in 15.c):**
- Time-travel "Rigenera" (changes "Rigenera" button behaviour)
- "Modifica e rigenera" button on user bubbles
- "Branca da qui" branching action
- CLI interactive REPL session mode
- Any new template, partial, or button visible to teachers

**Files expected to change:**
- *Edited:* `src/aix/webui/agent/service.py` — `_persist_assistant_turn` (15.b.1), `_maybe_window_history` + `_summarise_history` (15.b.4), `_augment_query_with_history` (15.b.5)
- *Edited:* `src/aix/agent/graph/checkpointer.py` — explicit success-path INFO log (15.b.2)
- *Edited (small):* admin settings template + handler — `AIX_CONVERSATION_WINDOW_TURNS` knob (15.b.3)
- *NEW (small):* startup smoke test — `tests/integration/test_checkpointer_startup.py` (15.b.2)

**Why "🟠 High" and not "🔴 Urgent":** the V1 UX (current "Continua la conversazione…" + "Rigenera") already works end-to-end. This is hardening for an external pilot, not a blocker for an internal demo.

**Depends on:** **#10 (Conversation Memory — must be live, which it is)**. *Independent of 15.a* — can land before, after, or in parallel with the PostgresSaver migration. The two share no code paths beyond reading from the same `LessonMessage` table.

---

### Subtask 15.c: Conversation-Memory UX V2 — Time-travel "Rigenera" + Branching + Edit-and-rerun
**Priority:** 🔵 Low | **Effort:** 1-2d *(rough estimate; revisit when activated)* | **Assignee:** LM
**Status:** **DEFERRED — design-only.** No shipping commitment until pilot signal justifies it. Documented here so the design lives next to the rest of CORE 4 instead of in scattered slack threads.

**Promoted from the 2026-05-09 Point A investigation** ("with new UX, what would it look like?"). Captures the buttons-and-partials side that was deliberately *not* built on 2026-05-09 because the product decision was "hold UX flat for V1". When/if that decision flips, this is the canonical implementation plan.

**Description:**
The full V2 conversation-memory UX, layered on top of #10's checkpointer (CORE 2) + 15.a's PostgresSaver (CORE 4) + 15.b's hardening (no-UX). Adds three teacher-facing surfaces that turn the LangGraph time-travel API into product features.

**Proposed surfaces (each independently shippable behind a feature flag):**

| ID | Surface | Description | UX touch | Implementation handle |
|---|---|---|---|---|
| **15.c.1** | **"Modifica e rigenera"** on user bubbles | Common ChatGPT pattern. Click on a past user message → opens an inline editor → re-runs with the edited query but keeps full prior history. *Does not change "Rigenera" button behaviour.* | New button on user bubbles; new POST handler | Service-layer rewrite of the user `LessonMessage` for that turn, then `astream` from that turn |
| **15.c.2** | **"Branca da qui"** action on past turns | Branching without time-travel. Clones the lesson (`new lesson_id`), copies messages 1..N from the source, lets the user diverge. Existing original conversation untouched. | New action button on each past turn | `INSERT lesson` + `INSERT lesson_messages` from source, then redirect to the new `lesson_id` |
| **15.c.3** | **Time-travel "Rigenera"** (replaces today's full-rerun "Rigenera") | LangGraph native time-travel via `aget_state_history()` + `aupdate_state(checkpoint_id=...)`. Branches from the previous user turn instead of replaying the whole thread. | "Rigenera" button **changes behaviour** (not a new button — *this is the only one that changes existing UX semantics, and it's why we deferred*) | Already implemented in LangGraph's `BaseCheckpointSaver`; needs the `checkpoint_id` from 15.b.1 to be present on every `LessonMessage` |
| **15.c.4** | **CLI session REPL** (`apps/cli/run_agent.py --session`) | Interactive REPL with shared `thread_id` for non-frontend users (Angelo's evaluation harness, MCP debugging). Already supported by the underlying `LessonPlannerPipeline.run(session_id=...)`; this is purely CLI ergonomics. | No frontend change — CLI only | Wrap existing `pipeline.run()` in a `while`-loop that prompts and reuses `thread_id` |

**Why deferred (record of the 2026-05-09 product decision):**
1. **No pilot signal yet** that teachers want branching. ChatGPT shipped without branching for two years; usage data is what should drive this, not engineering hunch.
2. **15.c.3 changes existing UX semantics.** Today's "Rigenera" is a full re-run on the same thread — predictable, idempotent, well-understood. Replacing it with time-travel makes two visually identical buttons behave differently across history depths, which is harder to explain to teachers than to engineers.
3. **15.b is enough** to make conversation memory production-grade for V1. 15.c is *quality-of-life*, not *correctness*.

**Pre-requisites for activation (already in flight or deferred):**
- ✅ #10 (CORE 2) — Checkpointer + CQRS + windowing — LANDED 2026-05-01
- ⏳ 15.b — `checkpoint_id` populated on every `LessonMessage` (required for 15.c.3 specifically, nice-to-have for 15.c.1 and 15.c.2)
- ⏳ 15.a — PostgresSaver migration recommended (multi-instance branching/replay would otherwise hit SQLite write-lock contention under any real concurrency)

**Acceptance criteria:** *(deferred — to be defined at activation; the 4 surfaces above are independently shippable, not a single monolithic ticket)*

**Depends on:** **15.a (recommended)**, **15.b (required for 15.c.3 specifically)**, #12 (SSE streaming hardening — for 15.c.3's branched-replay event stream)
**Activation trigger:** explicit product decision driven by pilot/teacher feedback — not by engineering schedule. Re-open this ticket when that signal lands.

---

### Subtask 16: Long-Term Memory (Teacher Profiles)
**Priority:** 🟡 Medium | **Effort:** 6-8h | **Assignee:** LM

**Description:**
Persistent teacher profiles across sessions: preferred grade level, class composition (student variabilities), preferred activity duration, past query topics. Uses LangGraph Store (separate from Checkpointer — different mechanism).

**Acceptance Criteria:**
- [ ] Teacher profile data model
- [ ] PostgresStore or InMemoryStore implementation
- [ ] Auto-extraction of preferences from queries
- [ ] Profile data injected into Planner context
- [ ] Test: query twice with same teacher_id, verify preferences remembered

**Depends on:** #15.a (PostgresSaver)

---

### Subtask 17: Semantic Caching (GraphRAG + Agent layers)
**Priority:** 🟡 Medium | **Effort:** 4-8h | **Assignee:** LM

**Description:**
Cache responses for semantically similar queries. Production data shows 47% of queries are semantically similar but differently worded. Two-layer strategy: cache Cypher results (Layer 1, TTL 1h) + cache full lesson plans (Layer 2, TTL 24h).

**Acceptance Criteria:**
- [ ] Layer 1: GraphRAG query cache (embedding-based key, TTL: 1h)
- [ ] Layer 2: Agent response cache (embedding-based key, TTL: 24h)
- [ ] Semantic similarity matching threshold (configurable)
- [ ] Cache hit/miss logging for observability
- [ ] Latency reduction measurable (target: 65% for cache hits)

**Depends on:** #11 (Observability — to measure cache hit/miss ratio and latency wins). Redis optional for production.

---

### Subtask 17.b: Production Latency Hotspots — Live-Run Findings *(NEW 2026-05-10 PM)*
**Priority:** 🟠 High | **Effort:** 4-6h | **Assignee:** LM | **Status:** TODO *(soft-gated on #11b for measurement)*

**Description:**
Captures the four concrete latency contributors observed during the 2026-05-10 PM corrective-RAG live smokes (turn for *"fotosintesi clorofiliana"* with `AIX_CORRECTIVE_RAG_ENABLED=true` and `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS=2`). These are the **cheap, scoped mitigations** that can land independently of the larger CORE 3 #17 (semantic cache) work and the deferred #6.6 P2 phase 3 (writer streaming). Promoted into the proposed **CORE 3.5 — Latency & Performance Wave** for the consolidated longer-term plan.

**Observed hotspots (from 2026-05-10 PM smoke):**

| # | Hotspot | Observed cost | Where time goes |
|---|---|---|---|
| 1 | **Semantic Scholar 429 backoff** | ~30s/attempt × 2 attempts = **~60s wasted** | `src/aix/agent/media/external_apis.py` retries with exponential backoff; logs show *"WARNING - [Semantic Scholar] Rate limited (429)"* repeated ~5× per retrieve call. Backoff is correct (politeness) but unbounded per-run. |
| 2 | **Corrective-RAG retry overhead** | **~33-65s per extra attempt** | Each retry pays the full retrieve+grade pass, *including* hotspot #1 above. Acceptable when the rewrite produces a meaningfully different query, wasteful when it doesn't. |
| 3 | **Duplicate / trivial rewrites** | full retry budget on near-identical query | The grader sometimes returns a `retrieval_rewritten_query` that is semantically near-identical to the original (e.g. punctuation change, same key terms reordered). Today we still spend the full retry budget on it. |
| 4 | **Writer LLM cold-block** | **~97-110s** (single largest fixed cost) | Listed for completeness. Already tracked as **#6.6 P2 phase 3 (Writer Token Streaming, DEFERRED)** — *promoted into proposed CORE 3.5 as #LAT-1*. Same total time, but streaming changes 100s of blank-card waiting into 100s of progressive content rendering. |

**Acceptance Criteria (#17.b own scope — items 1, 2, 3, telemetry):**
- [ ] **Semantic Scholar circuit breaker:** if a single retrieve call hits 429 N≥2 times, mark Semantic Scholar `unavailable` for the rest of *this lesson run* (in-process flag on `AgentState.unavailable_external_sources: set[str]`); on subsequent retries skip the source instead of re-paying the backoff. Log `event=external_source_disabled source=semantic_scholar reason=rate_limit_repeated`.
- [ ] **Per-run external-API cache:** within a single agent run, if attempt N issues the *same* `query` (after lowercase + trim normalisation) to the same external source as attempt N-1, return the cached result instead of re-calling. Cache lives on `AgentState`, dies with the run — no Redis dep.
- [ ] **Skip-trivial-rewrite guard in `should_retry_retrieval`:** if `retrieval_rewritten_query` (lowercased, whitespace-collapsed, punctuation-stripped) has Levenshtein distance ≤ 3 from the original `teacher_query`, do NOT retry — emit the irrelevant grade with the existing **#9.UX-3 `limited_kg_only`** outcome instead. Add INFO log `event=corrective_rag_skip_trivial_rewrite original=… rewrite=… distance=…`.
- [ ] **Phase-duration telemetry:** `_compute_retrieval_outcome()` payload extended with `phase_durations_ms: dict[str, int]` keyed by node name (`plan`, `retrieve`, `grade_retrieval`, `write`, `critic`) so #11b traces (and the future Langfuse latency dashboard in CORE 3.5 #LAT-6) can chart end-to-end where time goes per turn. Already partially captured in service-layer event timestamps; this just promotes it to a first-class payload field.
- [ ] **Writer streaming cross-reference:** add a one-liner under #6.6 P2 phase 3 pointing back to this subtask + CORE 3.5 #LAT-1 (covered by Edit 4 below).
- [ ] **Backward compat:** all four mitigations must be additive — flag-OFF behaviour (`AIX_CORRECTIVE_RAG_ENABLED=false`) byte-identical to today; flag-ON behaviour without the new guards is byte-identical to current #9.UX-2/UX-3 output (the guards only *reduce* work, never add it).

**Files likely touched:**
- `src/aix/agent/media/external_apis.py` — circuit-breaker + per-run cache hooks (new helpers reading `AgentState`-passed dict)
- `src/aix/agent/graph/state.py` — new optional `unavailable_external_sources: set[str]` + `external_api_cache: dict[str, Any]` (both default `None` for back-compat)
- `src/aix/agent/graph/nodes.py` — `should_retry_retrieval` guard (skip trivial rewrites); `retrieve_node` threads cache + circuit-breaker state
- `src/aix/webui/agent/service.py` — `phase_durations_ms` in retriever payload (timing already captured at SSE-event level — promote to payload)
- `tests/unit/test_retrieval_latency_guards.py` — NEW: covers circuit breaker after N 429s, cache hit on repeat query, trivial-rewrite skip with distance ≤ 3, distance > 3 still retries

**Latency win (estimated):**
- Hotspot 1 (Semantic Scholar): ~30-60s saved on every multi-attempt run with hybrid sources.
- Hotspot 2+3 combined (skip trivial rewrites): ~33-65s saved on ~30-50% of corrective-RAG retries (rough estimate — needs #11b traces to confirm).
- **Combined effect on out-of-scope queries:** ~310s observed → ~200-220s expected (roughly 30% reduction), without touching the writer.

**Depends on:** #11b (Observability — required to measure before optimizing). Cheap to ship without #11b for the *correctness* of the guards (unit tests cover that), but the *latency win* can only be quantified post-#11b.

**Cross-reference:** Promoted into the proposed **CORE 3.5 — Latency & Performance Wave** as `#LAT-2` (external-API resilience), `#LAT-3` (retry hardening), partial of `#LAT-6` (latency-budget dashboard). Hotspot #4 (writer cold-block) is `#LAT-1` and is owned by the existing #6.6 P2 phase 3 ticket — listed here only so the latency budget reads end-to-end.

---

### Subtask 17.c: Single-Pass Retriever Efficiency — Post-Corrective-RAG-OFF Findings *(NEW 2026-05-12 PM)*
**Priority:** 🟠 High | **Effort:** 3-4h | **Assignee:** LM | **Status:** TODO *(soft-gated on #11b for measurement; the four guards are landable independently)*

**Description:**
Captures the latency contributors observed during the **2026-05-12 PM** live smoke of *"crea una lezione su disturbi da deficit di attenzione"* on domain `udl`, with `AIX_CORRECTIVE_RAG_ENABLED=false` (set after the 2026-05-11 PM ADHD smoke hit 514s due to a false-positive grader retry). With retries disabled, the run dropped to **199.8s** — confirming the corrective-RAG retry was the single largest cost — but the **single retrieval pass still took ~102s**, revealing a *second* family of hotspots that are completely independent of the corrective-RAG loop. These are the **cheap, scoped guards** that make a single pass fast regardless of whether Corrective RAG is on or off.

**Observed hotspots (from 2026-05-12 PM smoke, retrieve = 102s for 11 sub-queries):**

| # | Hotspot ID | Observed cost in 102s pass | Where time goes |
|---|---|---|---|
| R1 | **Grader input signal** | (only matters when CR is ON) | Today `RetrievalGraderAgent` evaluates the *raw* retrieval dump (~30k tokens, mostly noise from rejected nodes) instead of the *filtered* recommendations (~5-10 KG-validated strategies). This produced the false-positive `ambiguous` grade in the 2026-05-11 PM ADHD run that triggered the 215s wasted retry. **Independent guard:** when CR re-enabled, this is the highest-precision fix. |
| R2 | **Profile-enrichment dedup** | ~3-10s wasted on duplicate query | `[Node: Retrieve] Profile enrichment added 4 terms: ['ADHD', 'Scienze', 'DSA', 'ADHD']` — `ADHD` appears twice because `specific_topic` was already in the planner's 8 queries and is naively re-appended. Same issue would happen for any topic that appears in both `specific_topic` and the planner's `key_concepts`. |
| R3 | **Within-pass query dedup** | ~3-10s wasted *(sibling of R2)* | Beyond profile enrichment, the planner itself can emit two semantically-identical queries (e.g. "teaching strategies for ADHD students" + "UDL universal design for learning ADHD" both fan out to the same `MATCH (a:Adhd)-[:SUGGESTS]->...` Cypher). After normalisation, dedup before execution. |
| R6 | **Language-detect false-Italian on English queries** | **~25-30s wasted per pass** *(biggest single contributor)* | `multilingual_text2cypher` runs Italian dictionary coverage on the planner's English queries (e.g. *"teaching strategies for ADHD students"*, *"cognitive load management attention deficit"*). Dictionary coverage is 0-25% → falls below the 50% threshold → falls back to OpenRouter "translation" which returns essentially the same English string after a 1-10s LLM round-trip. Logs show this happened on 4 of 8 planner queries; one of them (`cognitive load management attention deficit`) cost 37s alone. |
| R7 | **Negative-embedding cache miss within a pass** | ~3-5s wasted per pass | Lines 152-169 of the 2026-05-12 log show 12 sequential `POST /api/v1/embeddings` calls for `'Learner Variability'`, each returning *"No similar concepts found"*. The retriever is iterating over the 12 ADHD-related nodes and re-embedding `Learner Variability` each time without remembering it failed the first time. Pure cache-miss inefficiency. |

**Acceptance Criteria (#17.c own scope — R1, R2, R3, R6, R7):**
- [ ] **R6 — Language-detect bypass for English queries:** add a fast pre-check in `multilingual_text2cypher.detect_language()`: if the query is ≥80% ASCII letters with no Italian-specific characters (`àèìòùé`) AND contains at least one common English educational keyword (`strategies, learning, classroom, students, teaching, design, attention, working memory, executive, regulation, instruction, framework`), short-circuit to `language=en` and skip the dictionary-coverage + OpenRouter-translation path entirely. **Acceptance:** unit test asserts 4 planner-style queries from the 2026-05-12 log no longer trigger `[Translation] Coverage X% < 50%`. Backward-compat: Italian queries unaffected.
- [ ] **R7 — Per-pass negative-embedding cache:** in `graph_retriever.HybridGraphRetriever`, add a `_seen_unresolved: set[str]` instance field that lives for one retrieval call; before issuing an embedding lookup, check membership. If hit, return empty result without HTTP call. Reset on each top-level `retrieve()` invocation so memory doesn't leak across runs. **Acceptance:** unit test injects a fake embedder returning empty, asserts only 1 HTTP call is made for 12 lookups of the same term.
- [ ] **R2 — Profile-enrichment dedup:** in `nodes.py::retrieve_node`, convert the enrichment list to a `set()` *after* the planner's queries are already in the search list, so terms the planner already covered aren't re-issued. **Acceptance:** unit test asserts `['ADHD', 'Scienze', 'DSA', 'ADHD']` from a planner that already searches `'ADHD'` collapses to `['Scienze', 'DSA']` for enrichment.
- [ ] **R3 — Within-pass query dedup:** in `retriever_agent.RetrieverAgent.run()`, before the search loop, normalise each query (lowercase, whitespace-collapsed, punctuation-stripped) and skip exact duplicates. **Acceptance:** unit test asserts a search list with 2 normalisation-equivalent queries executes only 1 of them.
- [ ] **R1 — Grader sees filtered recommendations (gated, default-OFF):** add an optional `AIX_CORRECTIVE_RAG_GRADER_INPUT={raw,filtered}` env flag (default `raw` = current behaviour for full backward-compat). When `filtered`, pass `state["recommendations"][:N]` + their count + a one-line domain hint to the grader instead of the raw `state["retrieval_results"]`. **Acceptance:** unit test asserts grader on the 2026-05-11 PM ADHD payload (`recommendations=52`) returns `relevant` under `filtered` and `ambiguous` under `raw` (i.e. reproduces today's false-positive on the same input). **Promotion gate:** flip default to `filtered` only after #11b traces confirm zero relevant→ambiguous flips on a sample of ≥50 production runs.
- [ ] **Backward compat:** R6, R7, R2, R3 are all *additive* (they only reduce work, never add it); flag-OFF flag-ON behaviour unchanged. R1 is *gated behind a new env flag* defaulting to today's behaviour; no surprise regressions.

**Files likely touched:**
- `src/aix/retrieval/multilingual_text2cypher.py` — R6 fast-path English detector before dictionary coverage
- `src/aix/retrieval/graph_retriever.py` — R7 `_seen_unresolved` set in `HybridGraphRetriever.__init__` + `retrieve()` reset
- `src/aix/agent/graph/nodes.py` — R2 enrichment dedup against existing search list
- `src/aix/agent/agents/retriever_agent.py` — R3 within-pass normalisation+dedup
- `src/aix/agent/agents/retrieval_grader_agent.py` — R1 optional `grader_input` mode reading from `state["recommendations"]` instead of `state["retrieval_results"]`
- `tests/unit/test_retriever_efficiency_guards.py` — NEW: covers all five guards above + the false-Italian regression baseline

**Latency win (estimated, per retrieval pass):**
- R6 (English bypass): **~25-30s saved** per pass (4-7 of 8-11 planner queries are English in typical Italian-teacher runs).
- R7 (neg-embed cache): ~3-5s saved per pass.
- R2 + R3 (dedup): ~5-15s saved per pass.
- R1 (grader input): not a latency win directly — *prevents* the 200s+ wasted retry that the false-positive `ambiguous` grade triggered on 2026-05-11 PM. Pays for itself the first time CR is flipped back on.
- **Combined effect on a single pass:** ~102s observed → ~55-65s expected (roughly 40% reduction), independent of any retry logic, writer streaming, or parallel fan-out. Adds super-linearly with `#LAT-4` (parallel fan-out — half of 60s is 30s).

**Depends on:** #11b (Observability — measure before optimising). All five guards are correctness-testable without #11b; the actual latency wins can only be measured post-#11b.

**Cross-reference:** Promoted into proposed **CORE 3.5 — Latency & Performance Wave** as `#LAT-7`. R1's grader-input fix is the precondition for safely re-enabling `AIX_CORRECTIVE_RAG_ENABLED=true` in production (the 2026-05-11 PM ADHD smoke showed that without R1, CR triggers a 200s+ false-positive retry on well-covered topics — exactly the case Corrective RAG is *supposed* to handle gracefully). #LAT-3 (smart-retry, R4) and #LAT-4 (in-pass parallel fan-out, R5) are independent siblings that compound multiplicatively with #LAT-7.

---

### Subtask 18: Model Upgrade & Evaluation Phase (Claude / GPT-5.x / Gemini A/B)
**Priority:** 🟡 Medium | **Effort:** 4-6h | **Assignee:** LM
**Note:** The **OpenRouter migration is already done** in `feature/openrouter` (commits `98f7b3f` and following). This subtask is therefore now an **A/B evaluation phase**, not a migration: comparing model families on the same workload to pick the optimal mix per component.

**Description:**
With OpenRouter live, switching models is a one-line change in `.env`. This subtask runs structured A/B evaluations across model families to lock in the optimal model per component: e.g., `anthropic/claude-sonnet-4-6` (current default) vs `openai/gpt-5.x` vs `google/gemini-2.5-pro` for the Writer; `google/gemini-2.0-flash` (current default) vs `openai/gpt-5-nano` for Text2Cypher and translation; reasoning models (`openai/o4-mini`, `deepseek/deepseek-r1`) for the Critic. Evaluation is driven by traces from #11.

**Acceptance Criteria:**
- [ ] Define a benchmark set: 10 representative queries per domain (Neuro + UDL), with ground-truth lesson plans where available
- [ ] Run each query through the pipeline with at least 3 model combinations
- [ ] Capture quality (Critic scores), latency (#11 traces), and cost (#11 token counts)
- [ ] Recommendation matrix: best model per component (Writer, Critic, Planner, Text2Cypher, translation, embeddings)
- [ ] Update default `LLM_MODEL`, `TEXT2CYPHER_MODEL`, `EMBEDDING_MODEL` in `.env.example` with justification
- [ ] Document findings in `docs/Media_Mapping_and_Model_Upgrade_Analysis.md` Part B (update existing analysis with real numbers)

**Depends on:** #11 (Observability — traces are required for measurement)
**Reference:** `docs/Media_Mapping_and_Model_Upgrade_Analysis.md` — Part B (theoretical analysis to be replaced with empirical results)
**Cross-reference:** `FUTURE_FIXES.md` **#4 (LLM Model Selection — Ad Hoc Tests Required)** — Angelo's candidate lists align directly: #4a Text2Cypher (`gemini-2.0-flash`, `claude-haiku-4-5`, `gpt-4o-mini`, `deepseek-chat`) and #4b Lesson Generation (`claude-sonnet-4-6`, `gpt-4o`, `gemini-2.0-flash-thinking`, `deepseek-r1`) should seed the A/B benchmark set.

---

### Subtask 19: Human-in-the-Loop Interrupts
**Priority:** 🔵 Low | **Effort:** 2h | **Assignee:** LM
**Note:** Last subtask of **CORE 4 — Personalization**.

**Description:**
Add LangGraph `interrupt()` at key decision points: after Planner (out_of_scope confirmation), after Retriever (low results warning), after Critic (major revision approval). This lets a teacher take control mid-generation rather than waiting for a finished plan they then have to discard.

**Acceptance Criteria:**
- [ ] `interrupt()` after Planner if `scope_status == "out_of_scope"` — frontend shows confirmation modal
- [ ] `interrupt()` after Retriever if `len(retrieved_nodes) < 3` — frontend warns of weak retrieval
- [ ] `interrupt()` after Critic if `revision_count >= max_revisions` and avg score < 3.5 — frontend asks teacher to approve / edit
- [ ] Streaming endpoint (#12) emits interrupt events
- [ ] Resume mechanism documented for the Path C webui (#6.6) — htmx `<wa-dialog>` opened via SSE event, teacher submits decision via `hx-post`, server resumes the LangGraph thread

**Depends on:** #15.a (PostgresSaver — required for pause/resume state persistence), #12 (SSE Streaming — for emitting interrupt events to the frontend)

---

### Subtask 20: MCP Tool Servers
**Priority:** 🔵 Low | **Effort:** 2-3 days | **Assignee:** LM | **Status:** ✅ DONE (Option A path — 7 of 7 phases LANDED; live GUI-client smokes deferred to a 30-min follow-up documented in `docs/integrations/MCP_Setup.md` § "Live integration follow-up")

**Description:**
Wrap GraphRAG, Media, and Agent tools as a single MCP (Model Context Protocol) server so external AI clients (Claude Desktop, Cursor IDE, Lovable apps, partner LangGraph agents) can access the Knowledge Graph and the full lesson-planning pipeline via the standardized protocol. Hybrid architecture: shared `FastMCP` instance reachable over **stdio** (local clients) and **Streamable HTTP** (remote clients, mounted inside the existing FastAPI app at `/mcp/` behind JWT Bearer auth).

**Architecture chosen (Q1 — APPROVED):** Option C — hybrid stdio + mounted Streamable HTTP via FastMCP 3.x.
**Tool surface (Q2 — APPROVED):** All production-ready tools listed below.
**Phase delivery (Q3 — APPROVED):** Step-by-step phases, smoke-tested individually before moving forward.

**Acceptance Criteria:**
- [x] **Phase 1** — 4 `kg.*` tools (search, get_context, list_concepts, get_schema) + `stdio_main.py` entry + `mcp_smoke.py` + Cursor / Claude Desktop config snippets in `docs/integrations/MCP_Setup.md`. ✅ Smoke verified end-to-end.
- [x] **Phase 2** — 4 resources (`kg://schema`, `kg://concepts/{domain}`, `methodology://list`, `media://stats`) + 2 prompts (`educational-query`, `lesson-plan-request`). ✅ Smoke verified (list + read all resources, list + render both prompts).
- [x] **Phase 3** — 5 `media.*` tools (`lookup_curated`, `search_youtube`, `search_academic`, `search_oer`, `generate_diagram`). ✅ Smoke verified (5/5 tools, lookup_curated 5/6 matched across both domains, live YouTube fallback + Semantic Scholar rate-limit handling proven).
- [x] **Phase 4** — 1 `agent.run_lesson_plan` tool wrapping the same `stream_agent_events` helper that backs `POST /api/v1/agent/run` (CORE 2 #7). MCP progress notifications stream Planner → Retriever → Writer → Critic phases. ✅ Smoke verified end-to-end against live agent pipeline (~106s run, 10,828-char lesson, 9 KG nodes, full structured response).
- [x] **Phase 5** — Streamable HTTP transport mounted at `/mcp/` inside `aix.api.main` with `JWTVerifier` (HS256, audience `fastapi-users:auth`, shared `WEBUI_AUTH_SECRET`). Lifespan combined via `AsyncExitStack` so the FastMCP `StreamableHTTPSessionManager` starts/stops with the parent app. ✅ Smoke verified end-to-end (`mcp_smoke.py --phase5-verify`): healthy → unauth `POST /mcp/` returns 401 → login mints token → `Client(url, auth=token)` lists 10 tools → `kg.list_concepts` returns 5 concepts over HTTP. Full Streamable HTTP session lifecycle (POST/GET/DELETE) confirmed working.
- [x] **Phase 6** — `tests/mcp_server/` integration suite (19 tests across 5 files) + OpenAPI strictly-additive regression baseline (`data/diagnostic/openapi_before_p20.txt`). ✅ All 19 tests PASS in ~64s. Coverage: tool/resource/prompt inventory locked (10/3/2 + 1 templated), JWT Bearer auth gate (401 unauth / 401 wrong-secret / 2xx valid), cheap KG tools (`kg.list_concepts` + `kg.get_schema` for both domains, validation), `agent.run_lesson_plan` contract with mocked `stream_agent_events` (happy path + validation + error propagation), and a strict-additive REST-surface guard ensuring `/api/v1/*`, `/auth/*`, `/webui/*` routes never disappear. The suite runs entirely in-process (no live uvicorn, no live Neo4j, no LLM calls).
- [x] **Phase 7** — Final docs polish + ClickUp #20 DONE entry. ✅ Landed via Option A: `MCP_Setup.md` updated with a new **Production deployment notes** section (TLS, secret hygiene, session affinity, cold-start budget, OAuth 2.1 follow-up) and a **Live integration follow-up** section that queues the manual Cursor IDE / Claude Desktop / MCP Inspector smokes as a 30-minute hand-off task. ClickUp #20 flipped to ✅ DONE.

**What landed (6 of 7 phases):**

| Surface | Count | Names |
|---|---|---|
| Tools | 10 | `kg.search`, `kg.get_context`, `kg.list_concepts`, `kg.get_schema`, `media.lookup_curated`, `media.search_youtube`, `media.search_academic`, `media.search_oer`, `media.generate_diagram`, `agent.run_lesson_plan` |
| Resources | 4 | `kg://schema`, `kg://concepts/{domain}`, `methodology://list`, `media://stats` |
| Prompts | 2 | `educational-query`, `lesson-plan-request` |
| Transports | 2 | `stdio` (local — Claude Desktop / Cursor IDE) + `Streamable HTTP` mounted at `/mcp/` (remote — JWT Bearer auth) |
| Regression tests | 19 | `test_mcp_surface` (5) + `test_mcp_http_auth` (4) + `test_mcp_kg_tools` (5) + `test_mcp_agent_tool_contract` (3) + `test_mcp_openapi_regression` (2) — all PASS in ~64s |

**Files added / changed:**
- `src/aix/mcp/__init__.py`, `server.py`, `stdio_main.py`, `http_app.py`
- `src/aix/mcp/tools/kg_tools.py`, `media_tools.py`, `agent_tools.py`
- `src/aix/mcp/resources/kg_resources.py`
- `src/aix/mcp/prompts/educational_prompts.py`
- `scripts/diagnostic/mcp_smoke.py` (extended for Phases 2-5 verify modes)
- `scripts/diagnostic/probe_mcp_endpoint.py`, `inspect_mcp_mount.py`, `capture_openapi_baseline.py`
- `tests/mcp_server/__init__.py`, `conftest.py`, `test_mcp_surface.py`, `test_mcp_http_auth.py`, `test_mcp_kg_tools.py`, `test_mcp_agent_tool_contract.py`, `test_mcp_openapi_regression.py`
- `data/diagnostic/openapi_before_p20.txt` (baseline for the strict-additive guard)
- `docs/integrations/MCP_Setup.md` (canonical setup guide)
- `requirements.txt` (pinned `fastmcp>=3.0.0,<4.0.0`)
- `src/aix/api/main.py` (lifespan-combined `/mcp/` mount; circular-import fix `sys.path` → `src/`)

**Key engineering notes / lessons learned:**
1. **Circular import (Phase 5)** — `sys.path.insert(0, parent.parent)` in `aix.api.main` was making our internal `aix.mcp` package resolvable as plain `mcp`, colliding with the official Anthropic `mcp` SDK that `fastmcp` imports during its own logging setup. Fixed by inserting `src/` (project source root) instead of `src/aix/`. Also moved `_mcp_http_app = build_mcp_http_app()` from module scope into the FastAPI `lifespan` so all `aix.api` modules are fully loaded before MCP boots.
2. **Test-package shadowing (Phase 6)** — pytest auto-adds `tests/` to `sys.path`, so a directory named `tests/mcp/` shadows the third-party `mcp` SDK and breaks every fastmcp import inside the test process (and silently disables the `/mcp/` mount). Renamed to `tests/mcp_server/` and documented the convention in its `__init__.py`.
3. **FastMCP prompt API quirks (Phase 2)** — (a) `from __future__ import annotations` breaks Pydantic 2.11 prompt-arg schema generation for `Optional[str]`; remove it from prompt modules. (b) `mcp.list_resources()` only lists *static* resources; templates require `mcp.list_resource_templates()`. (c) MCP spec mandates `arguments: dict[str, str]` — `int` args must be declared as `str` and parsed inside the function. (d) `mcp.get_prompt()` returns the *definition*; use `mcp.render_prompt(name, args)` for actual rendering. (e) Prompt functions must return `list[Message]` (from `fastmcp.prompts.prompt`), and the `'system'` role is not allowed — inline system context into a single `'user'` message.
4. **Auth alignment** — the `JWTVerifier` reuses `WEBUI_AUTH_SECRET` directly with HS256 + audience `fastapi-users:auth`, so a single `POST /auth/jwt/login` token works on both `/api/v1/agent/*` (Bearer backend) and `/mcp/`. Zero token duplication, zero second login flow.
5. **Backward compat** — every phase landed strictly additive: no breaking change to `/api/v1/*`, `/webui/*`, `/auth/jwt/*`, or the existing cookie auth. Verified via OpenAPI before/after diffs, end-to-end smoke on each phase, and the `test_p20_strictly_additive` regression test (which now runs on every CI invocation).
6. **Pre-existing agent issue surfaced (Phase 4)** — Planner/Critic occasionally hit `JSON parse failure` from OpenRouter empty bodies. The MCP layer correctly propagates the failure; underlying fix is tracked under #11a (Agent JSON Parse Hardening).

**Depends on:** None (additive, doesn't replace existing code)
**Reference:** `docs/architecture/Agentic_GraphRAG_Architecture_Analysis.md` — Section 3 · `docs/integrations/MCP_Setup.md` — full client-onboarding guide

---

### Subtask 21: Graph Updater Agent (Phase 3)
**Priority:** 🔵 Low | **Effort:** 2-3 days | **Assignee:** LM

**Description:**
Implement the Graph Updater Agent to extract new concepts from generated lesson plans and propose Knowledge Graph additions. Currently a stub returning empty results (`src/aix/agent/agents/graph_updater_agent.py`). Critical for keeping the KG fresh as teachers use the system — without it, the KG ossifies.

**Acceptance Criteria:**
- [ ] LLM extracts candidate concepts from approved lesson plans (e.g., new strategies mentioned)
- [ ] Diff against existing KG (deduplication via embeddings)
- [ ] Human review queue (proposals never auto-merge)
- [ ] CLI / admin UI for accepting/rejecting proposals
- [ ] Audit log of all KG changes

**Depends on:** Research on safe KG update strategies, admin UI for review queue

---

### Subtask 22: Curriculum Tool — Italian Standards (Phase 3)
**Priority:** 🔵 Low | **Effort:** 2-3 days | **Assignee:** LM/AG

**Description:**
Implement curriculum standards lookup: Italian National Curriculum (Indicazioni Nazionali), European Qualifications Framework, regional school authority standards. Currently a stub returning placeholder data (`src/aix/agent/tools/curriculum_tool.py`). Lets the agent verify each lesson aligns with required learning outcomes for the teacher's grade level.

**Acceptance Criteria:**
- [ ] Source: Italian Ministry of Education API or scraped/cached dataset
- [ ] Tool exposes `lookup_curriculum(grade, subject, learning_outcome)` to the agent
- [ ] Writer prompt augmented to include relevant curriculum codes
- [ ] Critic checks lesson plan against required outcomes for the grade

**Depends on:** Curriculum data source (API or static dataset), legal review of data sourcing

---

### Subtask 23: Canva Integration
**Priority:** 🔵 Low | **Effort:** 1-2 days | **Assignee:** AG

**Description:**
Implement Canva Connect API for professional template-based slide-deck and worksheet generation. Currently a stub returning "coming soon" (`src/aix/agent/media/canva_generator.py`). Lets teachers export a generated lesson plan as a polished slide deck or worksheet ready to use in class.

**Acceptance Criteria:**
- [ ] Canva Connect API integration
- [ ] Template library mapped to lesson sections (warm-up, I Do, We Do, You Do, assessment)
- [ ] Generates editable Canva design from lesson plan content
- [ ] Returns shareable Canva URL in API response

**Depends on:** Canva API key + template library + Canva developer account approval

---

## REFERENCE DOCUMENTS

| Document | Content |
|---|---|
| `docs/architecture/Agentic_GraphRAG_Architecture_Analysis.md` | Full architecture analysis, best practices gaps (10 items), MCP analysis, memory analysis (3 types), priority roadmap |
| `docs/architecture/Media_Mapping_and_Model_Upgrade_Analysis.md` | Media mapping script analysis (3 UDL blocking bugs) + GPT-5.x upgrade strategy (tiered recommendation) |
| `docs/architecture/Agent_Domain_Prompt_Integration.md` | 3 options for connecting Agent prompts to domain configs. Option 2 (quick win) → Option 3 (clean architecture) |
| `docs/runbooks/GraphRAG_Data_Pipeline_Guide.md` | Data pipeline documentation for data ingestion and validation |
| `docs/architecture/Agentic_GraphRAG_BestPractices_Validation.md` | **Companion doc** — validates CORE 1–6 plan against 2026 Agentic RAG best practices: subtask coverage audit, tech stack validation (17/17), architecture pattern check, 3 minor gap recommendations |
| `docs/product/REPO_REORG_MIGRATION_GUIDE.md` | **Onboarding cheat-sheet** — one-page guide explaining the `src/aix/` layout (Phase 3C), old→new path mappings, new entry points, and how to fix old-import branches with the rewrite script |

---

## SUMMARY

### Subtask counts per Core

| Core | Theme | Subtasks | Target |
|---|---|---|---|
| **CORE 0** | Legacy / Pre-existing (E1-E4) | 4 (2 DONE, 2 IN PROGRESS) | — (in progress) |
| **CORE 1** | Agentic Foundations | 8 (#1, #2, **#2.5**, **E5**, #3, #4, #5, #6) | Feb 2026 |
| **CORE 2** | Production-Readiness (API + Safety + Observability) | 8 (**#6.5** ✅, **#6.6**, #7, #8, #11, #9, #10, #12) | Mar 2026 |
| **CORE 3** | Quality & Cost (Advanced RAG) | 5 (#13, #14, #17, **#17.b** ⭐ NEW, #18) | Apr 2026 |
| **CORE 3.5** | Latency & Performance Wave (Proposed) | 6 (#LAT-1 thru #LAT-6 — 2 cross-listed, 4 NEW) | Apr-May 2026 *(PROPOSED — design only)* |
| **CORE 4** | Personalization (Memory & Human Loop) | 5 (#15.a, #15.b, #15.c [deferred], #16, #19) | May 2026 |
| **CORE 5** | Strategic / Extension Layer | 4 (#20, #21, #22, #23) | Jun+ 2026 |
| **CORE 6** | Deployment & Frontend Production (placeholder) | ~8 (#24-#31, includes AixLearning embed handoff) | TBD |
| | | | |
| **Total ticketed subtasks** | | **32** (5 in ClickUp today + 23 new + 3 newly added: #2.5, #6.5, #6.6 + **#17.b ⭐ NEW**) | |
| **Future placeholder subtasks** | | **~13** (~7 CORE 6 + 6 CORE 3.5 — design-only until #11b unlocks measurement; 2 CORE 3.5 items are cross-listed from existing tickets so net new is 4) | |

### Effort estimates

| Core | Effort | In days |
|---|---|---|
| CORE 1 | ~14-18h | ~2-3 days |
| CORE 2 | ~31-45h (incl. 4-6h ✅ DONE on #6.5, ~10h ✅ DONE on #6.6 P2 phase 2) | ~4-5.5 days |
| CORE 3 | ~19-28h *(was ~15-22h; +4-6h for #17.b)* | ~2.5-3.5 days |
| CORE 3.5 *(PROPOSED — design-only)* | ~9-15h NEW *(excluding #LAT-5 cross-listed under CORE 3 #17)* | ~1.5-2 days |
| CORE 4 | ~13-15h | ~2 days |
| CORE 5 | ~7-11 days | ~7-11 days |
| CORE 6 | TBD (~3-4 weeks once scoped) | ~3-4 weeks |
| **CORE 1-5 total** *(incl. CORE 3.5 if promoted)* | **~84-115h + 7-11 days** | **~18-26 working days** |

### Key changes vs previous Tier 1/2/3 layout

1. **Tier-based grouping → Core-based grouping** (matches the existing ClickUp epic structure CORE 0, CORE 1, CORE 2, ...).
2. **E5 moved out of CORE 0 → CORE 1** because its remaining work (UDL Critic + e2e revision validation) is unblocked only by #2 + #2.5.
3. **#2.5 added to CORE 1** — Educational Profile Schema port from `fem/enhanced-variables-extraction`.
4. **#6.5 added as the first subtask of CORE 2** — Frontend Platform Evaluation. ✅ DONE (2026-04-26). **Decision: Path C — Mirror Stack** (FastAPI + Jinja2 + htmx 2 + WebAwesome + Tailwind + sse-starlette). The original working hypothesis (Vercel + Next.js + Vercel AI SDK) was rejected after deep investigation of the AixLearning private repo revealed it runs Python + htmx + WebAwesome + Mercure + Bun + Docker Compose — i.e. the same paradigm Path C now mirrors. Full ADR in `docs/architecture/Frontend_Platform_Evaluation.md`.
4.1. **#6.6 inserted into CORE 2** ⭐ NEW — *Path C Webui Skeleton + Agent In-Process Integration*. Builds the recommended stack as the canonical end-to-end test surface for the agent, retiring Streamlit for that role. Crucially, the agent's **first HTTP-callable surface is HTML+SSE (in #6.6), not JSON+SSE** — the public JSON contract (#7) is then designed against a UI that already works, eliminating contract design risk. A new pure-Python service module `src/aix/agent/service.py` is the single insertion point both consumers wrap.
4.2. **#7, #8, #12 rescoped** as a consequence of #6.6:
   - **#7** is now "expose the existing service layer over public JSON+SSE" (4-6h, was 3h).
   - **#8** moves its dependency from "None" to "#6.6", so guardrails enforce a schema that the real UI has already exercised.
   - **#12** is reframed from "build SSE from scratch" to "harden the SSE delivered in #6.6" (2-3h, was 4-6h).
4.3. **Three integration shapes for the eventual AixLearning embed** documented in §6.5 of the eval doc (iframe / template port / JSON-only). Path C explicitly preserves all three. The embed itself is deferred to **end of CORE 6** and is not on the CORE 1-5 critical path.
5. **CORE 2 expanded** to absorb #11 (Observability) and #10 (Conversation Memory) from old Tier 1, and now includes the #6.5 → #6.6 → #7 → #8 → #12 streaming/API chain.
6. **CORE 3 redefined** as the "quality & cost" wave that depends on #11 traces being live.
6.1. **CORE 3 #17.b inserted + CORE 3.5 wave proposed** *(2026-05-10 PM)* — the 2026-05-10 PM corrective-RAG live smoke (with `AIX_CORRECTIVE_RAG_ENABLED=true` for the first time) was the first run that *exercised* multi-attempt retrieval against external sources end-to-end, and it surfaced four concrete latency hotspots that had never been measured before: *(a)* Semantic Scholar 429 backoff (~30s/attempt × 2 = ~60s wasted on hybrid runs), *(b)* corrective-RAG retry overhead (~33-65s per extra attempt), *(c)* duplicate / trivial rewrites that spend the full retry budget on near-identical queries, *(d)* writer LLM cold-block (~97-110s — already tracked under #6.6 P2 phase 3 but never sized end-to-end). Two doc-only changes captured the finding: **(A) `Subtask 17.b` added to CORE 3** as the cheap-mitigations ticket (Semantic Scholar circuit breaker, per-run external-API cache, skip-trivial-rewrite guard, `phase_durations_ms` telemetry — all backward-compat-additive); **(B) `CORE 3.5 — Latency & Performance Wave (Proposed)`** added between CORE 3 and CORE 4 as the consolidated forward-looking wave. CORE 3.5 promotes #6.6 P2 phase 3 (Writer Token Streaming, was DEFERRED) to `#LAT-1`, cross-lists CORE 3 #17 (semantic cache) as `#LAT-5` (no double-count), and adds four NEW items (`#LAT-2` external-API resilience, `#LAT-3` retry hardening, `#LAT-4` parallel retrieval fan-out, `#LAT-6` end-to-end latency-budget Langfuse dashboard). **Status: PROPOSED — design only**, same convention as **#15.c** (deferred design-only) and **CORE 6** (future placeholder); promotion to ClickUp gated on **#11b** Langfuse traces being live so we *measure before we cut*. Observed P50: ~150s KG-covered / ~310s out-of-scope → target P50 post-CORE-3.5: <90s / <120s. Two single-biggest wins: `#LAT-1` (writer streaming — perceived-latency win, no wall-clock change) + `#LAT-2` (Semantic Scholar 429 — ~60s wall-clock saved on hybrid runs).
7. **CORE 4 redefined** as the "personalization" wave (all checkpointer-dependent).
8. **CORE 5 unchanged in scope** — still the strategic/experimental future bucket.
9. **CORE 6 added as a future placeholder** — deployment shape now known (Path C → Docker Compose on Hetzner/Coolify); ticket creation deferred to end of CORE 1-5. Embed shape (iframe / template port / JSON-only) decided at the *end* of CORE 6 in coordination with the AixLearning platform team.
10. **#6.6 P2 phase 2 closed** *(2026-04-26)* — the Path C webui chat workspace ships with the 3-pane layout (profile sidebar / chat / media), per-agent cards (Planner → Retriever → Writer → Critic), the user's first query as an active chat input on the draft state, inline profile editing, and OOB-swap media panel. End-to-end smoke verified on a KG-covered query (`motivazione intrinseca` → 15 nodes / 30 relations / 15 media items). Streamlit retired for agent e2e: `apps/streamlit/main.py::render_agent_mode()` now displays a banner pointing to `http://127.0.0.1:8765/webui/`. The GraphRAG admin mode in the same Streamlit app is intentionally untouched.
10.1. **#6.6 P2 phase 3 (writer-token streaming) explicitly deferred** post-#11 so we can measure the user-perceived latency improvement in the trace dashboard rather than ship it blind.
10.2. **#11 expanded into 11a + 11b** as a direct consequence of #6.6 P2 phase 2 smoke testing: every captured run shows Planner and Critic silently falling through to hardcoded defaults because OpenRouter returns 200 OK with an empty / non-JSON body. The fallbacks mask the failure (run completes "successfully", Critic always approves with the literal critique text "Approved due to parsing error"), so the Critic is currently a no-op approval gate. **11a (Agent JSON Parse Hardening)** must precede **11b (Langfuse/LangSmith dashboard)** — tracing a no-op Critic produces traces that look healthy but contain no real data, which would in turn corrupt #14 (Citation Grounding) and #18 (Model Eval) in CORE 3. #11 effort revised 2h → 4-6h.
10.3. **#6.6 P3 rescoped to "uploads-only"** *(2026-04-26)* — first P3 attempt added a planner-preview approval gate (`POST /run` → `awaiting_approval` → `POST /run/approve` starting LangGraph at `retrieve`). Smoke testing surfaced two regressions: *(a)* the teacher lost the live planner→retriever→writer streaming UX they had in P2, and *(b)* a runtime `TypeError` from a stale `create_initial_state` signature stalled the run at "running" with no SSE cards. **Rolled back same-day** to a minimal P3: keep P2's live streaming flow intact and only add the chat-attachment uploads (paperclip in `chat_input.html` → `POST /lesson/{id}/upload` → `partials/chat_attachments.html` chips → `Lesson.uploaded_files_json` → `AgentState.teacher_provided_context` → Writer prompt appendix). LangGraph entry stays at `plan`; the planner-snapshot column was dropped from the model. `pypdf` pinned in `requirements.txt`. The `aix.webui.agent.service.run_agent_stream` setup phase is now wrapped in a try/except that persists `status="error"` and emits an `error` event so a transient setup failure can never leave the lesson stuck at `running`.
10.4. **#6.6 P3 paperclip-icon visibility fix** *(2026-04-26, three attempts; root cause confirmed via server-side HTML diagnostic)* — second smoke after the P3 rollback showed the chat input rendering with no paperclip / Mermaid icon buttons. *(Attempt 1)* applied the WebAwesome-doc pattern of moving the screen-reader text from `aria-label` on `<wa-button>` to `label` on the inner `<wa-icon>` (per [webawesome.com/docs/components/button](https://webawesome.com/docs/components/button/) → "Icon Buttons"). User re-tested and the icons were **still invisible**, so attempt 1 was insufficient. *(Attempt 2)* replaced `<wa-button>` with plain HTML `<button>` styled via Tailwind (`w-8 h-8 inline-flex …`) for icon-only cases, keeping `<wa-icon>` inside; user re-tested and **icons were STILL invisible** despite the plain button being immune to web-component sizing quirks. *(Diagnostic — landed)* wrote `scripts/diagnostic/inspect_chat_input.py` which logs in via `/auth/login` (form fields `email` + `password`, **NOT** the OAuth2 `username` default — fastapi-users was customised to match the visible label), fetches the rendered HTML server-side, and runs structural checks. Result: **all the markup IS in the response (paperclip `<button>`, `<wa-icon>`, hidden `<input type="file">`, `<wa-tooltip>` wrappers, send button)** — bug is 100% client-side. Cross-checked in **InPrivate / extension-free mode**: paperclip still missing → ruled out browser-extension interference. *(Attempt 3 — landed, root-cause fix)* every invisible button shared one pattern: **wrapped in `<wa-tooltip>`**; the visible "Invia" `<wa-button>` was the only one without a tooltip wrapper. The CDN bundle we load (`webawesome@3.5.0` from `ka-f.webawesome.com`) does not register the `<wa-tooltip>` custom element, so the global FOUC-prevention rule `:not(:defined) { visibility: hidden }` in `_base.html` keeps the entire tooltip subtree (including the slotted plain `<button>`) permanently invisible. Verifiable in DevTools console: `customElements.get('wa-tooltip')` returns `undefined`. **Fix:** dropped all six `<wa-tooltip>` wrappers from `partials/chat_input.html`, moved the same Italian copy into native HTML `title="..."` attributes on the buttons / textarea. Native `title` is universal, screen-reader-friendly, has zero JS dependency, and survives the missing custom element. **Codebase rules documented inline in `chat_input.html`**: *(rule 1)* icon-only buttons use plain `<button>` + `<wa-icon>`, never `<wa-button>`; *(rule 2)* never wrap form controls in `<wa-tooltip>` — use `title=` instead, and revisit fancier styled tooltips in P5 polish only after confirming a working `<wa-tooltip>` registration in the bundle then. The diagnostic script (`scripts/diagnostic/inspect_chat_input.py`) is left in the repo so the same isolation playbook (server-side markup capture → InPrivate test → DevTools `customElements.get`) can be re-run on any future "element renders in HTML but not in browser" bug in P4 / P5.
10.5. **#6.6 P0 → P3 closed; P4 + P5 + P6 enumerated** *(2026-04-26)* — the phase table at the top of #6.6 is now the canonical status board: P0 (skeleton), P1 (auth + form), P2 phase 1 + 2 (SSE + chat workspace), and P3 (chat attachments) all ✅ DONE; P2 phase 3 (token streaming) DEFERRED behind #11; **P4 (Lesson library + history + PDF export, ~2d)**, **P5 (Italian copy + a11y + mobile + Tailwind CLI, ~2d)**, and **P6 (Hetzner deploy via Docker Compose, ~1d)** TODO. Recommended order is `#11a → P4 → P5 → P6`, because the lesson library and PDF export both surface a *"✓ approvata dal Critico"* signal that today is a no-op fallback pending #11a (Agent JSON Parse Hardening). Detail per phase in the table above.

12. **#10 scope expanded + #15 migration story locked in** *(2026-05-01)* — Following the post-#6.6 chat-workspace smoke, **#10 (Conversation Memory)** was rescoped from a 3-5h checkpointer-only ticket to a 5-7d phased delivery: *(10.1)* `chat_input` OOB-swap on SSE close to remove the dead-end UX, *(10.2)* `AsyncSqliteSaver` checkpointer wiring, *(10.3)* new `lesson_messages` CQRS table + multi-turn `/run` semantics with `mode={follow_up,regenerate,new}` (regenerate uses LangGraph native time-travel from the previous user turn via `get_state_history()` + `aupdate_state()`), and *(10.4)* summary-buffer windowing for long conversations. **#15 (CORE 4)** clarified as the matching `AsyncSqliteSaver → AsyncPostgresSaver` migration — a 1-line saver-class swap because LangGraph's `BaseCheckpointSaver` abstraction is backend-agnostic; same `lesson_messages` CQRS view (L2), same code path, just durable + multi-instance. Confirmed best-practice via 2026 production-agent surveys (Replit, Klarna, Uber, Anthropic all use LangGraph checkpointers for stateful agents). Alternatives rejected: manual message history fed to LLM as context (3× code, no time-travel), Redis-only sessions (no time-travel, in-memory data loss), Mem0/Zep/Letta dedicated memory frameworks (deferred to V2 long-term memory under #16), OpenAI Assistants API threads (vendor lock-in vs. OpenRouter). Langfuse (added separately by AG for prompt management — `aix.domains.langfuse_prompts` + `scripts/ops/seed_langfuse_prompts.py`) confirmed as observability/prompt-management layer, not state — complementary, not a substitute.

13. **#10 phases 10.1-10.4 LANDED** *(2026-05-01)* — Conversation Memory delivered end-to-end across the four planned phases; the dead-end-input UX bug and the inability to do follow-up turns are both fixed.

  **10.1 (chat_input OOB-swap, ~1.5h):** `chat_input.html` wrapped in a stable `<div id="chat-input-wrapper">` carrying conditional `hx-swap-oob="outerHTML"` when an `_oob` flag is set. `_stream_event_to_sse` for both `kind="done"` and `kind="error"` now appends the OOB-rendered partial to the SSE payload; htmx routes the OOB fragment to `#chat-input-wrapper` while the placeholder lesson card / error card still flows through the normal `beforeend` swap on `#chat-cards`. By the time the SSE event fires, `lesson.status` has already been mutated to `complete` / `error` by `run_agent_stream` (verified contract), so the partial renders the active follow-up state in the OOB slot. Zero JS changes — `chat_pane.html`'s `htmx:sseClose` handler still only swaps the `lesson-card-loading` placeholder via `/card-fragment`. Backward compat: full-page reloads on `complete`/`error` lessons render the active input via the regular include path (`_oob=False` default).

  **10.2 (AsyncSqliteSaver checkpointer, ~5h):** Added `langgraph-checkpoint-sqlite>=2.0,<3.0` to `requirements.txt`. New `aix.agent.graph.checkpointer` module exposes a process-singleton `get_checkpointer()` (asyncio.Lock-guarded first-access init, `LANGGRAPH_CHECKPOINTER_URL` env override, default `data/agent_threads.db` mirroring the webui SQLite convention) plus a `thread_config(thread_id)` helper. **Graceful degradation by design** — if the package is missing, init logs a clear warning and returns `None`; the graph compiles without a checkpointer and single-turn behaviour is preserved (no crash on the test runners or fresh checkouts before `pip install`). New async builder `build_lesson_planner_graph_async()` compiles with the saver attached; the original sync `build_lesson_planner_graph()` is preserved for the legacy CLI test harness. `LessonPlannerPipeline` got a sibling `_get_graph_async()` method; `run()` switches to it and passes `thread_config(session_id or "ephemeral-<uuid>")`. Both `run_agent_stream` (webui SSE) and `stream_agent_events` (public API) thread the config through `astream`. **Lifespan-managed pooling deferred to #15** — the `AsyncSqliteSaver` connection is intentionally held for the process lifetime via the lazy singleton; explicit setup/teardown via FastAPI lifespan is the right home for the Postgres pool, not the SQLite connection.

  **10.3 (`lesson_messages` table + multi-turn `/run`, ~2d):** New `LessonMessage` SQLAlchemy model with `(id, lesson_id FK CASCADE, role, content_md Text, turn_index, agent_kind, meta_json JSON, checkpoint_id, created_at)` plus a compound index on `(lesson_id, turn_index)` for the chat pane's primary query. `Lesson.messages` reverse relationship with `cascade="all, delete-orphan"` and `order_by` on `(turn_index, created_at)`. Auto-created by `Base.metadata.create_all` on next startup — **Alembic intentionally NOT introduced in this scope** (the existing dev-hotpatch pattern in `aix.webui.db.init_db()` handles new-table creation cleanly; Alembic stays in its planned home at #6.6 P5/CORE 6 where ALTER TABLE migrations land). `POST /run` auto-detects `mode` from `lesson.status` (`draft` → `new`, `complete`/`error` → `follow_up`), persists a user `LessonMessage` at `MAX(turn_index)+1`, flips status to `running`, returns the chat conversation partial. **Backward-compat backfill on first follow-up** — pre-#10.3 lessons have no `lesson_message` rows but DO have `lesson.teacher_query` + `lesson.lesson_plan_md`; the route detects this (count probe) and synthesises turn 1 (user + assistant) from the legacy fields with `meta_json={"backfilled": true}` so the chat history stays contiguous. Service layer's `run_agent_stream` adds three new helpers: `_load_conversation_history` (CQRS reader filtered by `turn_index < latest`), `_augment_query_with_history` (Italian/English-aware service-layer prompt augmentation that prepends prior turns + summary to the teacher's raw query — every agent in the pipeline sees the context without bespoke prompt edits), and `_persist_assistant_turn` (writes the assistant `LessonMessage` after the agent completes successfully, with the same `turn_index` as the user message — failures are logged + swallowed because the lesson row's `status="complete"` is the load-bearing persistence). New `chat_history.html` partial renders the chronological transcript (user bubble + per-turn lesson card with `data-turn-index`); `chat_conversation.html` includes it when `messages` is non-empty and falls back to the legacy single-bubble layout otherwise (zero regression for old lessons). `chat_input.html` follow-up state copy refreshed: "Continua la conversazione…" placeholder, "Invia" button with `paper-plane` icon (replacing the misleading "Nuova" / `rotate` semantics that implied a fresh start). **Time-travel-based regenerate intentionally deferred to V2** — V1 ships `mode={new, follow_up}` only; today's "Rigenera" button still replays on the same thread, which is acceptable continuity. The full time-travel branch (`get_state_history()` + `aupdate_state(checkpoint_id=...)`) requires careful UI work (which checkpoint? show-all-branches?) that doesn't gate the user-visible follow-up UX.

  **10.4 (summary-buffer windowing, ~6h):** Turn-based windowing in `_maybe_window_history` — when conversation has more than `AIX_CONVERSATION_WINDOW_TURNS` (env-tunable, default 4 turns = 8 messages) prior turns, summarise everything older via a low-temperature LLM call (`temperature=0.2`, `max_tokens=600`, language-aware Italian/English system prompts) and stash the summary in `AgentState.conversation_summary`. The augmenter renders the summary as a "Sintesi dei turni più vecchi" block before the verbatim recent window, then the current request. **Failure mode is non-destructive** — summarisation errors fall back to `(None, full_history)` with a logged warning so the run continues with the larger prompt rather than crashing on a transient API hiccup; same for empty-summary results. Single-turn / few-turn behaviour is byte-identical to pre-#10.4 because the early-return on `len(history) ≤ window_messages` short-circuits before any LLM call. The summariser uses the same OpenAI/OpenRouter client as the agents (no new API key, no model selection drift). 1200-char per-message truncation in the summary-input transcript prevents a long lesson plan from blowing the summariser's input budget.

  **Manual install required after pulling:** `pip install "langgraph-checkpoint-sqlite>=2.0,<3.0"` then restart the FastAPI app. `Base.metadata.create_all` auto-creates the new `lesson_message` table. **Smoke recipe:** open a complete lesson → type a follow-up like *"ora adattala per studenti con DSA"* → input stays active (no reload) → after generation completes, the new turn appears below the prior lesson card in the chat history; reload the page and the full transcript is preserved. Verify checkpointer DB at `data/agent_threads.db` exists and is non-empty after the second turn.

  **What's NOT done (recorded as V2):** *(a)* `apps/cli/run_agent.py` interactive REPL with shared `thread_id` — the underlying pipeline already supports it via `LessonPlannerPipeline.run(session_id=...)`, just needs a CLI wrapper; *(b)* "Rigenera" via LangGraph native time-travel — V1 replays on the same thread, V2 will branch from the previous checkpoint via `get_state_history()` + `aupdate_state()`; *(c)* per-agent prompt-level integration of `conversation_history` field — V1 uses service-layer augmented query which is sufficient for the user-visible UX and avoids regression risk on the existing prompts; *(d)* tests — chat_input OOB regression, multi-turn smoke, windowing trigger should land before this work hits production traffic.

11. **#7 closed — FastAPI JSON+SSE Agent Endpoint shipped** *(2026-04-26)* — public agent contract live at `POST /api/v1/agent/run` and `POST /api/v1/agent/stream`, both protected by `current_active_user` and discoverable in Swagger UI at `/docs`. The Swagger UI **Try it out** panel exposes a `Minimal` / `Rich` examples dropdown for both endpoints — same UX as `/api/v1/context` — driven by route-level `Body(..., openapi_examples=...)`; the `Rich` example exercises every optional `EducationalProfile` field plus `teacher_provided_context`. **Auth:** new `BearerTransport` registered alongside the existing `CookieTransport` in fastapi-users — both transports share the same `current_active_user` dependency, so the webui's cookie flow keeps working unchanged while CLI / Postman / mobile callers send `Authorization: Bearer <jwt>`. The JWT login endpoint is mounted at `POST /auth/jwt/login` (form-encoded `email` + `password`, returns `{access_token, token_type}`). **Service split:** introduced a DB-less `aix.webui.agent.service.stream_agent_events()` helper for the public API; `run_agent_stream()` (the webui-only DB-backed sibling) is byte-untouched, eliminating any chance of public-API traffic mutating the webui SQLite. **Backward compat:** the `test_openapi_inventory_strictly_additive` regression test diffs the live `/openapi.json` against `data/diagnostic/openapi_before_p7.txt` (the pre-#7 baseline captured by `scripts/diagnostic/list_openapi_paths.py`) and fails the suite if any pre-existing path disappears or renames — currently green, so `/api/v1/context`, `/webui/*`, and `/auth/*` are byte-compatible vs. the day-before snapshot. **CORS:** middleware moved from a hard-coded `["*"]` to env-driven `WEBUI_CORS_ALLOW_ORIGINS` (default `*` for dev) so #P6 (Hetzner deploy) can lock origins to a single hostname without code edits. **Tests:** `tests/api/test_agent_routes.py` ships 7 contract tests (auth 401 × 2, payload 422, sync happy path, pipeline-error 502, SSE stream emission, OpenAPI inventory) — agent runtime mocked at the `stream_agent_events` boundary so the suite is fast (~39s) and orthogonal to LLM availability. **Webui NOT migrated** to call the public API over HTTP by deliberate decision — in-process keeps zero latency and avoids double serialisation; both code paths now route through the same upstream `AgentOrchestrator`. **Lessons:** *(a)* JWT Bearer + cookie coexist in fastapi-users by listing both backends in the same `FastAPIUsers([...])` constructor — `current_active_user` accepts the first transport that resolves, no per-route flag needed; *(b)* Swagger UI dropdown UX requires route-level `Body(..., openapi_examples=...)` — schema-level `json_schema_extra={"examples": ...}` (plural) is a known Swagger UI footgun that leaks the wrapper object into the editable body; the singular `json_schema_extra={"example": ...}` is fine for the *Schema* tab; *(c)* PowerShell on Windows has no `tail` cmdlet — use `Select-Object -Last N` or just drop the pipe (rejected pipeline blocks pytest from running at all, looks like a hang); *(d)* `try/except` around router mounts in `main.py` (used here for both `agent_router` and `bearer_backend`) means any future regression in the public API path can never prevent the legacy GraphRAG mode from booting — same pattern as the existing webui mount.
