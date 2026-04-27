# [AI TEAM] - AGENTIC GRAPHRAG — Updated Task Description

**Last Updated:** April 27, 2026 *(CORE 5 #20 — MCP Tool Servers — ✅ DONE via Option A: 7 of 7 phases landed. Phase 7 closed with `MCP_Setup.md` Production-deployment + Live-integration-follow-up sections; ClickUp #20 flipped to ✅ DONE. The MCP server is regression-locked by a 19-test pytest suite, fully documented for stdio + Streamable HTTP onboarding, and ready for Angelo's manual GUI-client smoke (Cursor / Claude / Inspector — ~30 min, no code changes required).)*
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
  #10  Conversation Memory (LangGraph Checkpointer)           ← no deps (unblocks CORE 4 #15, #19)
  #12  SSE Streaming Hardening (backpressure + reconnect)     ← depends on #6.6 (already partial), finalised here
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| **6.5** | **Frontend Platform Evaluation & Decision (Research Spike)** | LM (+ Diego, Simone input) | 🔴 Urgent | 4-6h | None | ✅ DONE — see `docs/architecture/Frontend_Platform_Evaluation.md` |
| **6.6** | **Path C Webui Skeleton + Agent In-Process Integration** ⭐ NEW | LM | 🔴 Urgent | 8-10h | #6.5, CORE 1 | TODO |
| 7 | **FastAPI JSON+SSE Agent Endpoint (public contract)** | LM | 🔴 Urgent | 4-6h | #6.6 | ✅ DONE *(2026-04-26)* — `POST /api/v1/agent/run` + `POST /api/v1/agent/stream` mounted; JWT Bearer transport added in parallel to the cookie backend (zero webui regression); Minimal / Rich examples dropdown in Swagger UI mirrors `/api/v1/context`; 7 contract tests green; OpenAPI inventory strictly additive |
| 8 | **Guardrails: Input/Output Validation** | LM | 🔴 Urgent | 3-5h | #6.6 | TODO |
| 11 | **Observability (Agent JSON Parse Hardening + LangSmith/Langfuse)** | LM | 🟠 High | 4-6h | #6.6 (for 11a repro) | TODO — split into **11a** (JSON parse hardening, must precede 11b) + **11b** (tracing dashboard) |
| 9 | **Corrective RAG (Retrieval Grading)** | LM | 🔴 Urgent | 3-4h | None | TODO |
| 10 | **Conversation Memory (LangGraph Checkpointer)** | LM | 🔴 Urgent | 3-5h | None | TODO |
| 12 | **SSE Streaming Hardening (backpressure + reconnect)** | LM | 🟠 High | 2-3h | #6.6 | TODO |

**CORE 2 total effort:** ~31-45h (~4-5.5 days, of which 4-6h already DONE on #6.5; #11 effort grew from 2h → 4-6h after #6.6 P2 phase 2 surfaced the agent JSON parse silent fallthrough)

---

### CORE 3 — Quality & Cost: Advanced RAG Techniques (Target: April 2026)

**Theme:** Push answer quality higher and cost lower.
**Principle:** All depend on CORE 2 being live so we can measure improvements via traces.
**Deliverable:** *"Higher-quality, faster, cheaper, fully-cited answers."*

```
Dependency graph:

  #13 Query Decomposition (Multi-Hop Reasoning)             ← needs CORE 2 #11 (traces)
  #14 Citation Grounding & Source Attribution                ← needs CORE 2 #11
  #17 Semantic Caching (GraphRAG + Agent layers)             ← needs CORE 2 #11
  #18 Model Upgrade & Evaluation phase (Claude/GPT-5.x/Gemini A/B)  ← needs CORE 2 #11
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 13 | **Query Decomposition (Multi-Hop Reasoning)** | LM | 🟠 High | 4-5h | #11 | TODO |
| 14 | **Citation Grounding & Source Attribution** | LM | 🟡 Medium | 3h | #11 | TODO |
| 17 | **Semantic Caching (GraphRAG + Agent layers)** | LM | 🟡 Medium | 4-8h | #11 | TODO |
| 18 | **Model Upgrade & Evaluation Phase (A/B test Claude / GPT-5.x / Gemini)** | LM | 🟡 Medium | 4-6h | #11 | TODO |

**CORE 3 total effort:** ~15-22h (~2-3 days)

---

### CORE 4 — Personalization: Memory & Human Loop (Target: May 2026)

**Theme:** The agent remembers and learns.
**Principle:** All depend on the Checkpointer in CORE 2 (#10).
**Deliverable:** *"Agent personalizes content per teacher and supports interactive editing mid-generation."*

```
Dependency graph:

  #15 State Checkpointing (PostgresSaver upgrade)           ← depends on CORE 2 #10
  #16 Long-Term Memory (Teacher Profiles)                    ← depends on #15
  #19 Human-in-the-Loop Interrupts                           ← depends on #15
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 15 | **State Checkpointing (PostgresSaver upgrade)** | LM | 🟡 Medium | 5h | #10 | TODO |
| 16 | **Long-Term Memory (Teacher Profiles)** | LM | 🟡 Medium | 6-8h | #15 | TODO |
| 19 | **Human-in-the-Loop Interrupts** | LM | 🔵 Low | 2h | #15 | TODO |

**CORE 4 total effort:** ~13-15h (~2 days)

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
| **P2 phase 3 — Token streaming** | Switch `aix.webui.agent.service` to `graph.astream(..., stream_mode="messages")` so writer tokens stream into the writer card live; phase tracker becomes a small secondary signal above the chat | DEFERRED | Explicit deferral 2026-04-26: the §6.6 acceptance bullet *"sees Writer tokens streaming"* is the only one phase 3 unlocks; current phase-2 UX (writer-pending placeholder + final card) is acceptable for the dev/test e2e surface. Pick up after #11 (observability) so we can measure the user-perceived latency improvement. |
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

### Subtask 9: Corrective RAG (Retrieval Grading)
**Priority:** 🔴 Urgent | **Effort:** 3-4h | **Assignee:** LM

**Description:**
Add a retrieval quality grading step between Retriever and Writer in the LangGraph. If retrieved KG data is poor or irrelevant, rewrite the query and re-retrieve (max 3 attempts) instead of generating from bad context. Research shows 30% of RAG errors trace to poor retrieval.

**Acceptance Criteria:**
- [ ] New `grade_retrieval_node` in LangGraph between Retrieve and Write
- [ ] Lightweight LLM grading call (~200-400 tokens)
- [ ] Query rewriting + re-retrieval loop (max 3 iterations)
- [ ] `retrieval_attempts` counter in AgentState
- [ ] Logging for grading decisions and rewrites

**Depends on:** None

---

### Subtask 10: Conversation Memory (LangGraph Checkpointer)
**Priority:** 🔴 Urgent | **Effort:** 3-5h | **Assignee:** LM

**Description:**
Add LangGraph checkpointing so teachers can refine lesson plans across multiple turns ("add an ADHD activity", "change duration to 45 min"). Currently `session_id` exists in AgentState but is never used for persistence — every query starts from scratch.

Root cause: `src/aix/agent/graph/lesson_planner_graph.py` line 68 compiles without a checkpointer:
`compiled = workflow.compile()  # ← No checkpointer!`

**Acceptance Criteria:**
- [ ] Add `MemorySaver` (dev) or `PostgresSaver` (production) to `src/aix/agent/graph/lesson_planner_graph.py`
- [ ] Pass `thread_id` config at invocation using `session_id`
- [ ] Multi-turn test: query → follow-up modification → verify context preserved
- [ ] Update `apps/cli/run_agent.py` to support session mode
- [ ] Update FastAPI endpoint (#7) to accept and pass session_id

**Depends on:** None
**Blocks:** #16 (Long-Term Memory), #19 (Human-in-the-Loop)

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

**Phase 11a — Agent JSON Parse Hardening** (must land before 11b):
- [ ] Reproduce the empty-body issue with a captured request from `planner_agent.py::analyze_query` against OpenRouter (model `google/gemini-2.0-flash` per current `.env`)
- [ ] Identify root cause from the candidate list: *(a)* empty response body, *(b)* JSON wrapped in markdown code fences (`` ```json … ``` ``), *(c)* preamble text before the JSON object, *(d)* OpenRouter quota / model-availability fallback returning a different shape
- [ ] Add a robust `_extract_json(text: str) -> dict` helper used by both `planner_agent.py` and `critic_agent.py`: strips markdown fences, finds the outermost `{...}` block, retries parsing once on the captured substring before falling through
- [ ] Switch both agents' OpenRouter calls to use `response_format={"type": "json_object"}` where the underlying model supports it (Gemini 2.0 Flash and GPT-4o do; document fallback for models that don't)
- [ ] On *real* parse failure (after `_extract_json` exhausts retries), the agents must emit a typed event (`AgentError(stage="planner_json_parse", raw_response=...)`) rather than silently returning a fallback dict — so the UI can render a `<wa-callout variant="warning">` "L'agente non ha risposto correttamente, riprova" instead of pretending it succeeded
- [ ] Update `webui.agent.service.run_agent_stream` to translate that event into the `error` SSE kind (already wired for other failures)
- [ ] Smoke-test: run 5 lessons; **0 of them** should show the literal string "Approved due to parsing error" in the Critic card

**Phase 11b — Tracing dashboard** (depends on 11a):
- [ ] Add `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` (or Langfuse equivalent) to `.env` and `.env.example`
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

### Subtask 15: State Checkpointing (PostgresSaver upgrade)
**Priority:** 🟡 Medium | **Effort:** 5h | **Assignee:** LM
**Note:** First subtask of **CORE 4 — Personalization**. Depends on CORE 2 #10 already being live.

**Description:**
Replace `MemorySaver` (in-memory, volatile, set up in CORE 2 #10) with `PostgresSaver` for production-grade crash recovery, durable conversation memory across pod restarts, and horizontal scaling.

**Acceptance Criteria:**
- [ ] `pip install langgraph-checkpoint-postgres`
- [ ] PostgreSQL connection configured (separate `LANGGRAPH_DATABASE_URL`)
- [ ] Graph compiled with PostgresSaver in production, MemorySaver in dev
- [ ] Crash recovery test: kill process mid-pipeline, verify resume from last node
- [ ] Production-ready for concurrent requests (connection pooling)

**Depends on:** PostgreSQL database (provisioned by ops), #10 (Conversation Memory — MemorySaver must be working first)
**Unblocks:** #16 (Long-Term Memory), #19 (Human-in-the-Loop)

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

**Depends on:** #15 (PostgresSaver)

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

**Depends on:** #15 (PostgresSaver — required for pause/resume state persistence), #12 (SSE Streaming — for emitting interrupt events to the frontend)

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
| **CORE 3** | Quality & Cost (Advanced RAG) | 4 (#13, #14, #17, #18) | Apr 2026 |
| **CORE 4** | Personalization (Memory & Human Loop) | 3 (#15, #16, #19) | May 2026 |
| **CORE 5** | Strategic / Extension Layer | 4 (#20, #21, #22, #23) | Jun+ 2026 |
| **CORE 6** | Deployment & Frontend Production (placeholder) | ~8 (#24-#31, includes AixLearning embed handoff) | TBD |
| | | | |
| **Total ticketed subtasks** | | **31** (5 in ClickUp today + 23 new + 3 newly added: #2.5, #6.5, #6.6) | |
| **Future placeholder subtasks** | | **~7** (CORE 6, scoped post-#6.5; embed shape selected at end of CORE 6) | |

### Effort estimates

| Core | Effort | In days |
|---|---|---|
| CORE 1 | ~14-18h | ~2-3 days |
| CORE 2 | ~31-45h (incl. 4-6h ✅ DONE on #6.5, ~10h ✅ DONE on #6.6 P2 phase 2) | ~4-5.5 days |
| CORE 3 | ~15-22h | ~2-3 days |
| CORE 4 | ~13-15h | ~2 days |
| CORE 5 | ~7-11 days | ~7-11 days |
| CORE 6 | TBD (~3-4 weeks once scoped) | ~3-4 weeks |
| **CORE 1-5 total** | **~71-96h + 7-11 days** | **~16-23 working days** |

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
7. **CORE 4 redefined** as the "personalization" wave (all checkpointer-dependent).
8. **CORE 5 unchanged in scope** — still the strategic/experimental future bucket.
9. **CORE 6 added as a future placeholder** — deployment shape now known (Path C → Docker Compose on Hetzner/Coolify); ticket creation deferred to end of CORE 1-5. Embed shape (iframe / template port / JSON-only) decided at the *end* of CORE 6 in coordination with the AixLearning platform team.
10. **#6.6 P2 phase 2 closed** *(2026-04-26)* — the Path C webui chat workspace ships with the 3-pane layout (profile sidebar / chat / media), per-agent cards (Planner → Retriever → Writer → Critic), the user's first query as an active chat input on the draft state, inline profile editing, and OOB-swap media panel. End-to-end smoke verified on a KG-covered query (`motivazione intrinseca` → 15 nodes / 30 relations / 15 media items). Streamlit retired for agent e2e: `apps/streamlit/main.py::render_agent_mode()` now displays a banner pointing to `http://127.0.0.1:8765/webui/`. The GraphRAG admin mode in the same Streamlit app is intentionally untouched.
10.1. **#6.6 P2 phase 3 (writer-token streaming) explicitly deferred** post-#11 so we can measure the user-perceived latency improvement in the trace dashboard rather than ship it blind.
10.2. **#11 expanded into 11a + 11b** as a direct consequence of #6.6 P2 phase 2 smoke testing: every captured run shows Planner and Critic silently falling through to hardcoded defaults because OpenRouter returns 200 OK with an empty / non-JSON body. The fallbacks mask the failure (run completes "successfully", Critic always approves with the literal critique text "Approved due to parsing error"), so the Critic is currently a no-op approval gate. **11a (Agent JSON Parse Hardening)** must precede **11b (Langfuse/LangSmith dashboard)** — tracing a no-op Critic produces traces that look healthy but contain no real data, which would in turn corrupt #14 (Citation Grounding) and #18 (Model Eval) in CORE 3. #11 effort revised 2h → 4-6h.
10.3. **#6.6 P3 rescoped to "uploads-only"** *(2026-04-26)* — first P3 attempt added a planner-preview approval gate (`POST /run` → `awaiting_approval` → `POST /run/approve` starting LangGraph at `retrieve`). Smoke testing surfaced two regressions: *(a)* the teacher lost the live planner→retriever→writer streaming UX they had in P2, and *(b)* a runtime `TypeError` from a stale `create_initial_state` signature stalled the run at "running" with no SSE cards. **Rolled back same-day** to a minimal P3: keep P2's live streaming flow intact and only add the chat-attachment uploads (paperclip in `chat_input.html` → `POST /lesson/{id}/upload` → `partials/chat_attachments.html` chips → `Lesson.uploaded_files_json` → `AgentState.teacher_provided_context` → Writer prompt appendix). LangGraph entry stays at `plan`; the planner-snapshot column was dropped from the model. `pypdf` pinned in `requirements.txt`. The `aix.webui.agent.service.run_agent_stream` setup phase is now wrapped in a try/except that persists `status="error"` and emits an `error` event so a transient setup failure can never leave the lesson stuck at `running`.
10.4. **#6.6 P3 paperclip-icon visibility fix** *(2026-04-26, three attempts; root cause confirmed via server-side HTML diagnostic)* — second smoke after the P3 rollback showed the chat input rendering with no paperclip / Mermaid icon buttons. *(Attempt 1)* applied the WebAwesome-doc pattern of moving the screen-reader text from `aria-label` on `<wa-button>` to `label` on the inner `<wa-icon>` (per [webawesome.com/docs/components/button](https://webawesome.com/docs/components/button/) → "Icon Buttons"). User re-tested and the icons were **still invisible**, so attempt 1 was insufficient. *(Attempt 2)* replaced `<wa-button>` with plain HTML `<button>` styled via Tailwind (`w-8 h-8 inline-flex …`) for icon-only cases, keeping `<wa-icon>` inside; user re-tested and **icons were STILL invisible** despite the plain button being immune to web-component sizing quirks. *(Diagnostic — landed)* wrote `scripts/diagnostic/inspect_chat_input.py` which logs in via `/auth/login` (form fields `email` + `password`, **NOT** the OAuth2 `username` default — fastapi-users was customised to match the visible label), fetches the rendered HTML server-side, and runs structural checks. Result: **all the markup IS in the response (paperclip `<button>`, `<wa-icon>`, hidden `<input type="file">`, `<wa-tooltip>` wrappers, send button)** — bug is 100% client-side. Cross-checked in **InPrivate / extension-free mode**: paperclip still missing → ruled out browser-extension interference. *(Attempt 3 — landed, root-cause fix)* every invisible button shared one pattern: **wrapped in `<wa-tooltip>`**; the visible "Invia" `<wa-button>` was the only one without a tooltip wrapper. The CDN bundle we load (`webawesome@3.5.0` from `ka-f.webawesome.com`) does not register the `<wa-tooltip>` custom element, so the global FOUC-prevention rule `:not(:defined) { visibility: hidden }` in `_base.html` keeps the entire tooltip subtree (including the slotted plain `<button>`) permanently invisible. Verifiable in DevTools console: `customElements.get('wa-tooltip')` returns `undefined`. **Fix:** dropped all six `<wa-tooltip>` wrappers from `partials/chat_input.html`, moved the same Italian copy into native HTML `title="..."` attributes on the buttons / textarea. Native `title` is universal, screen-reader-friendly, has zero JS dependency, and survives the missing custom element. **Codebase rules documented inline in `chat_input.html`**: *(rule 1)* icon-only buttons use plain `<button>` + `<wa-icon>`, never `<wa-button>`; *(rule 2)* never wrap form controls in `<wa-tooltip>` — use `title=` instead, and revisit fancier styled tooltips in P5 polish only after confirming a working `<wa-tooltip>` registration in the bundle then. The diagnostic script (`scripts/diagnostic/inspect_chat_input.py`) is left in the repo so the same isolation playbook (server-side markup capture → InPrivate test → DevTools `customElements.get`) can be re-run on any future "element renders in HTML but not in browser" bug in P4 / P5.
10.5. **#6.6 P0 → P3 closed; P4 + P5 + P6 enumerated** *(2026-04-26)* — the phase table at the top of #6.6 is now the canonical status board: P0 (skeleton), P1 (auth + form), P2 phase 1 + 2 (SSE + chat workspace), and P3 (chat attachments) all ✅ DONE; P2 phase 3 (token streaming) DEFERRED behind #11; **P4 (Lesson library + history + PDF export, ~2d)**, **P5 (Italian copy + a11y + mobile + Tailwind CLI, ~2d)**, and **P6 (Hetzner deploy via Docker Compose, ~1d)** TODO. Recommended order is `#11a → P4 → P5 → P6`, because the lesson library and PDF export both surface a *"✓ approvata dal Critico"* signal that today is a no-op fallback pending #11a (Agent JSON Parse Hardening). Detail per phase in the table above.

11. **#7 closed — FastAPI JSON+SSE Agent Endpoint shipped** *(2026-04-26)* — public agent contract live at `POST /api/v1/agent/run` and `POST /api/v1/agent/stream`, both protected by `current_active_user` and discoverable in Swagger UI at `/docs`. The Swagger UI **Try it out** panel exposes a `Minimal` / `Rich` examples dropdown for both endpoints — same UX as `/api/v1/context` — driven by route-level `Body(..., openapi_examples=...)`; the `Rich` example exercises every optional `EducationalProfile` field plus `teacher_provided_context`. **Auth:** new `BearerTransport` registered alongside the existing `CookieTransport` in fastapi-users — both transports share the same `current_active_user` dependency, so the webui's cookie flow keeps working unchanged while CLI / Postman / mobile callers send `Authorization: Bearer <jwt>`. The JWT login endpoint is mounted at `POST /auth/jwt/login` (form-encoded `email` + `password`, returns `{access_token, token_type}`). **Service split:** introduced a DB-less `aix.webui.agent.service.stream_agent_events()` helper for the public API; `run_agent_stream()` (the webui-only DB-backed sibling) is byte-untouched, eliminating any chance of public-API traffic mutating the webui SQLite. **Backward compat:** the `test_openapi_inventory_strictly_additive` regression test diffs the live `/openapi.json` against `data/diagnostic/openapi_before_p7.txt` (the pre-#7 baseline captured by `scripts/diagnostic/list_openapi_paths.py`) and fails the suite if any pre-existing path disappears or renames — currently green, so `/api/v1/context`, `/webui/*`, and `/auth/*` are byte-compatible vs. the day-before snapshot. **CORS:** middleware moved from a hard-coded `["*"]` to env-driven `WEBUI_CORS_ALLOW_ORIGINS` (default `*` for dev) so #P6 (Hetzner deploy) can lock origins to a single hostname without code edits. **Tests:** `tests/api/test_agent_routes.py` ships 7 contract tests (auth 401 × 2, payload 422, sync happy path, pipeline-error 502, SSE stream emission, OpenAPI inventory) — agent runtime mocked at the `stream_agent_events` boundary so the suite is fast (~39s) and orthogonal to LLM availability. **Webui NOT migrated** to call the public API over HTTP by deliberate decision — in-process keeps zero latency and avoids double serialisation; both code paths now route through the same upstream `AgentOrchestrator`. **Lessons:** *(a)* JWT Bearer + cookie coexist in fastapi-users by listing both backends in the same `FastAPIUsers([...])` constructor — `current_active_user` accepts the first transport that resolves, no per-route flag needed; *(b)* Swagger UI dropdown UX requires route-level `Body(..., openapi_examples=...)` — schema-level `json_schema_extra={"examples": ...}` (plural) is a known Swagger UI footgun that leaks the wrapper object into the editable body; the singular `json_schema_extra={"example": ...}` is fine for the *Schema* tab; *(c)* PowerShell on Windows has no `tail` cmdlet — use `Select-Object -Last N` or just drop the pipe (rejected pipeline blocks pytest from running at all, looks like a hang); *(d)* `try/except` around router mounts in `main.py` (used here for both `agent_router` and `bearer_backend`) means any future regression in the public API path can never prevent the legacy GraphRAG mode from booting — same pattern as the existing webui mount.
