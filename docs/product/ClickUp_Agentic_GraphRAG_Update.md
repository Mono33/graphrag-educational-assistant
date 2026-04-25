# [AI TEAM] - AGENTIC GRAPHRAG — Updated Task Description

**Last Updated:** January 27, 2026  
**Copy-paste the sections below into ClickUp**

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
- `generate_media_mapping.py`: Script to generate curated media JSON from KG concepts via GPT-4o
- Supports batched async processing with rate limiting
- Only 20 concepts mapped due to `--limit` during initial test run

**Domain Prompt Extensions** *(Neuro full, UDL skeleton)*
- Neuro domain: Full prompt extensions (~85 lines) covering neurodidactic principles
- UDL domain: Basic prompt extensions (~25 lines) — needs enrichment
- Domain-aware prompt routing for Writer and Critic agents
- **Critical gap:** Agent Writer/Critic completely disconnected from rich `domains/udl_domain.py` (200+ lines) and `domains/neuro_domain.py` configs. Design doc with 3 options exists, none implemented.

**CLI Test Harness**
- `test_agent.py`: Interactive CLI for testing agent pipeline with custom queries
- `--query` flag for single-run mode, interactive loop for exploration

---

### 📁 Files Modified/Created (30 files in agent/)

**Core Pipeline:**
- `agent/orchestrator.py` | Main entry point, simplified API (356 lines)
- `agent/graph/lesson_planner_graph.py` | LangGraph state machine (186 lines)
- `agent/graph/nodes.py` | Pipeline node definitions (386 lines)
- `agent/graph/state.py` | State types + enums (213 lines)

**Agents:**
- `agent/agents/planner_agent.py` | Query analysis + intent + scope detection (189 lines)
- `agent/agents/retriever_agent.py` | Multi-search + media + external APIs (619 lines)
- `agent/agents/writer_agent.py` | Adaptive content generation (415 lines)
- `agent/agents/critic_agent.py` | Review + quality scoring (224 lines)
- `agent/agents/graph_updater_agent.py` | Phase 3 placeholder (96 lines)

**Tools:**
- `agent/tools/graphrag_tool.py` | GraphRAG wrapper for agents (258 lines)
- `agent/tools/curriculum_tool.py` | Phase 3 placeholder (180 lines)

**Prompts:**
- `agent/prompts/planner_prompt.py` | Planner system + user prompts (295 lines)
- `agent/prompts/writer_prompt.py` | Intent-specific writer prompts (725 lines)
- `agent/prompts/critic_prompt.py` | Evaluation criteria prompts (245 lines)
- `agent/prompts/templates/lesson_template.txt` | Italian lesson template (94 lines — not imported)
- `agent/configs/domain_prompts.py` | Neuro/UDL prompt extensions (232 lines)

**Media Layer:**
- `agent/media/external_apis.py` | YouTube, Wikipedia, Semantic Scholar, OER (1,054 lines)
- `agent/media/media_lookup.py` | Sidecar JSON media loading (419 lines)
- `agent/media/mermaid_generator.py` | LLM → Mermaid diagrams (510 lines)
- `agent/media/image_generator.py` | DALL-E 3 diagrams (442 lines)
- `agent/media/diagram_factory.py` | Routing to Mermaid/DALL-E/Canva (371 lines)
- `agent/media/resource_lookup.py` | Static resource lookup (447 lines)
- `agent/media/canva_generator.py` | Phase 5 placeholder (238 lines)

**Other:**
- `streamlit_app.py` | UI with mode toggle + agent mode
- `test_agent.py` | CLI test harness (242 lines)
- `generate_media_mapping.py` | Media mapping generator (488 lines)
- `kg_neuro_media_mapping.json` | Curated media for 20 Neuro concepts

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
| 1 | **Bug Fixes: DALL-E method + duplicate CurriculumTool + unused template** | LM | 🟠 High | 1h | None | TODO |
| 2 | **Agent ↔ Domain Config Integration** | LM | 🔴 Urgent | 1-4h | None | TODO |
| **2.5** | **Educational Profile Schema Integration** ⭐ NEW | AG, LM | 🔴 Urgent | 3-4h | None | TODO |
| **E5** | **Quality Assurance System** (re-scoped, was in CORE 0) | AG, LM | 🟠 High | 2h | #2, #2.5 | TODO |
| 3 | **UDL Media Mapping — Fix Script + Generate JSON** | LM | 🟠 High | 2h | None | TODO |
| 4 | **Neuro Media Mapping — Full Expansion (695 concepts)** | LM/AG | 🟠 High | 1h | None | TODO |
| 5 | **Validate External APIs end-to-end** | LM | 🟠 High | 2h | None | TODO |
| 6 | **Validate Media Layer end-to-end** | LM | 🟠 High | 2h | #1, #3, #4, #5 | TODO |

**CORE 1 total effort:** ~14-18h (~2-3 days)

---

### CORE 2 — Production-Readiness: API + Safety + Observability (Target: March 2026)

**Theme:** Make the agent usable by the frontend and safe in production.
**Principle:** Everything Lovable / FEM main app / future frontend needs to integrate.
**Deliverable:** *"Production-ready agent API consumable by an external frontend, with streaming, safety, observability, and corrective retrieval."*

```
Dependency graph:

  #6.5 Frontend Platform Evaluation & Decision (Spike)      ← no deps, MUST land before #7
  #7   FastAPI Endpoint for Agent Mode                       ← depends on #6.5
  #8   Guardrails: Input/Output Validation                   ← no deps (auth pieces depend on #6.5)
  #11  Observability (LangSmith / Langfuse Integration)      ← no deps (must precede CORE 3 #18)
  #9   Corrective RAG (Retrieval Grading)                    ← no deps
  #10  Conversation Memory (LangGraph Checkpointer)          ← no deps (unblocks CORE 4 #15, #19)
  #12  SSE Streaming to Frontend                             ← depends on #7 + #6.5
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| **6.5** | **Frontend Platform Evaluation & Decision (Research Spike)** ⭐ NEW | LM (+ Diego, Simone input) | 🔴 Urgent | 4-6h | None | TODO |
| 7 | **FastAPI Endpoint for Agent Mode** | LM | 🔴 Urgent | 3h | #6.5 | TODO |
| 8 | **Guardrails: Input/Output Validation** | LM | 🔴 Urgent | 3-5h | None | TODO |
| 11 | **Observability (LangSmith/Langfuse Integration)** | LM | 🟠 High | 2h | None | TODO |
| 9 | **Corrective RAG (Retrieval Grading)** | LM | 🔴 Urgent | 3-4h | None | TODO |
| 10 | **Conversation Memory (LangGraph Checkpointer)** | LM | 🔴 Urgent | 3-5h | None | TODO |
| 12 | **SSE Streaming to Frontend** | LM | 🟠 High | 4-6h | #7, #6.5 | TODO |

**CORE 2 total effort:** ~22-31h (~3-4 days)

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
| 20 | **MCP Tool Servers** | LM | 🔵 Low | 2-3 days | None | TODO |
| 21 | **Graph Updater Agent (Phase 3)** | LM | 🔵 Low | 2-3 days | None | TODO |
| 22 | **Curriculum Tool — Italian Standards (Phase 3)** | LM/AG | 🔵 Low | 2-3 days | None | TODO |
| 23 | **Canva Integration** | AG | 🔵 Low | 1-2 days | None | TODO |

**CORE 5 total effort:** ~7-11 days

---

### CORE 6 — Deployment & Frontend Production (Future Placeholder)

**Theme:** Take the agent live as a real product.
**Principle:** This Core is intentionally **not yet ticketed in ClickUp**. It only becomes well-defined once #6.5 (Frontend Platform Evaluation) lands and the chosen platform shapes the deployment plan. Listed here so the AI team has full visibility of the long-term roadmap.
**Deliverable:** *"Agentic GraphRAG deployed as a publicly-accessible product, with onboarding flow, beta pilot with real teachers, and operational runbooks."*

```
Suggested subtasks (to be detailed after #6.5 decision):

  #24 Frontend production build (informed by #6.5 decision)
  #25 CI/CD pipeline (Docker + container registry + K8s/ECS/Vercel deploy)
  #26 Production observability dashboard (Grafana / DataDog / Vercel Analytics)
  #27 Load testing + capacity planning (Locust / k6)
  #28 User onboarding flow (signup, EducationalProfile setup, first lesson)
  #29 Beta teacher pilot (5-10 schools, structured feedback collection)
  #30 Operational runbooks (incident response, key rotation, scaling playbook)
```

| # | Subtask Name | Assignee | Priority | Est. Effort | Depends On | Status |
|---|---|---|---|---|---|---|
| 24-30 | **(Detailed after CORE 2 #6.5 platform decision)** | TBD | TBD | ~3-4 weeks total | #6.5, #7, #12 | PLANNED |

**CORE 6 status:** Placeholder. Tickets to be created after the #6.5 ADR is finalized.

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
Connect Agent mode Writer/Critic to the rich domain configs in `domains/udl_domain.py` (200+ lines) and `domains/neuro_domain.py`. Currently the Agent pipeline's prompts are completely isolated — the UDL Writer extension is only 25 lines vs 200+ available. Design doc exists with 3 options (`docs/Agent_Domain_Prompt_Integration.md`), none implemented.

**Migration path:**
- Step 1 — Option 2 (quick win, ~1h): Modify `get_domain_extension()` in `domain_prompts.py` to dynamically load `get_system_prompt()` from `domains/`. Writer keeps its lesson plan format. 1 file changed.
- Step 2 — Option 3 (clean architecture, ~4h): Add `get_lesson_plan_template()` to `base_config.py`, implement domain-specific lesson structures (Neuro: I Do/We Do/You Do; UDL: 3-Principle framework).

**Acceptance Criteria:**
- [ ] Writer agent receives rich domain expertise (variability profiles, checkpoints, meta-rules) from `domains/` configs
- [ ] Critic agent gets domain-specific evaluation criteria
- [ ] Backward compatible (graceful fallback if `domains/` import fails)
- [ ] Test both Neuro and UDL domains via `test_agent.py`
- [ ] Existing Subtask E5 (Quality Assurance) benefits from enriched Critic criteria

**Depends on:** None
**Pairs with:** #2.5 (Educational Profile Schema) — both should land in the same release for full effect
**Unblocks:** E5 (Quality Assurance System — completing Critic UDL evaluation)
**Reference:** `docs/Agent_Domain_Prompt_Integration.md`

---

### Subtask 2.5: Educational Profile Schema Integration ⭐ NEW
**Priority:** 🔴 Urgent | **Effort:** 3-4h | **Assignee:** AG (porter, original author) / LM (reviewer)

**Description:**
Port the per-request `EducationalProfile` schema from the existing `fem/enhanced-variables-extraction` branch into `feature/openrouter` (and later `main`). This gives every API request — both GraphRAG mode (`/api/v1/context`) and the future Agent mode (`/api/v1/agent/lesson`) — a structured payload describing the class (size, grade, BES disabilities, attributes, features) and the classroom environment (LIM, WiFi, furniture mobility, BYOD policy). All field names map 1:1 to the AixLearning production models (`party.models.Party`, `classroom.models.Classroom`), so Lovable, the future Vercel/Next.js frontend, and the main FEM platform can pass them through without any field translation.

**What's already coded** (on `fem/enhanced-variables-extraction`, ready to port):
- 6 enums: `GradeLevel`, `DisabilityType` (10 BES types: DSA, ADHD, DOP, DF, DCGL/M/S, DLDS, PD, SA), `ClassFeature`, `StudentAttribute`, `FornitureMobility`, `OwnDevicePolicy`
- 3 Pydantic models: `EducationalGroup`, `ClassroomEnvironment`, `EducationalProfile`
- File: `api/schemas/educational_profile.py`

**Why it matters:** Without this, the rich domain prompts from #2 have nothing to specialize against — every lesson gets generic adaptations because the agent doesn't know it's a 25-student class with 2 ADHD + 1 DSA in a non-mobile room with no LIM. Subtasks #2 and #2.5 are the *input → processing* pair that together unlock real personalization.

**Acceptance Criteria:**
- [ ] Port `api/schemas/educational_profile.py` (6 enums + 3 Pydantic models) into the current branch
- [ ] Add optional `educational_profile: Optional[EducationalProfile] = None` field to `ContextRequest` (GraphRAG) and the future `AgentRequest` (#7)
- [ ] Propagate profile through `AgentState` (`agent/graph/state.py` — pattern already exists on source branch)
- [ ] Inject profile context into Writer / Planner / Critic prompts via the domain extension layer (combines with #2)
- [ ] Use profile in `MethodologyRanker` to boost methodologies matching disabilities present (e.g., if `ADHD` in profile, boost ADHD-tagged strategies)
- [ ] Backward compatible: every field is `Optional`; missing profile falls back to current generic behavior
- [ ] Document profile fields in `docs/Explainability_API_Guide_for_Frontend.md` so Simone knows exactly what to send
- [ ] Test: same query with and without profile produces measurably different recommendations

**Depends on:** None (can start immediately, ideally in parallel with #2)
**Pairs with:** #2 (Agent ↔ Domain Config Integration)
**Unblocks:** E5 (Quality Assurance — Critic can now evaluate profile-aware adaptations), #7 (Agent endpoint accepts richer payload), #16 (Long-Term Memory will store last-used `EducationalProfile` per teacher)
**Reference branch to port from:** `fem/enhanced-variables-extraction` — files: `api/schemas/educational_profile.py`, `agent/graph/state.py`

---

### Subtask E5: Quality Assurance System — Critic UDL Criteria + e2e Revision Loop Validation
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** AG, LM
**Note:** Originally subtask **E5 in CORE 0** ("in progress"). Re-scoped and moved to CORE 1 because the remaining work is unblocked by Subtask #2 (Agent ↔ Domain Config) and Subtask #2.5 (Educational Profile Schema).

**Description:**
The Critic Agent already exists with multi-criteria scoring (Structure, Evidence Grounding, Pedagogical Soundness) and an automatic revision loop (max 2 cycles, triggers if average < 3.5 or any criterion < 2). Two pieces are still missing for production-quality evaluation: (1) the UDL evaluation criteria are thin (~25 lines) compared to Neuro (~85 lines), and (2) the end-to-end revision loop has never been exercised against real teacher queries with the rich domain configs and the `EducationalProfile`.

**Acceptance Criteria:**
- [ ] After #2 lands: confirm Critic auto-loads UDL evaluation criteria from `domains/udl_domain.py` via the dynamic `get_domain_extension()` path
- [ ] After #2.5 lands: confirm Critic penalizes lesson plans that ignore the `EducationalProfile` (e.g., ignoring stated disabilities, exceeding class capacity)
- [ ] Run `test_agent.py` with 5 representative UDL queries and 5 Neuro queries; verify revision loop fires when expected
- [ ] Document the Critic's scoring rubric for both domains in `docs/Agentic_GraphRAG_Architecture_Analysis.md`
- [ ] Auto-approve safety net (after `max_revisions=2`) still works to prevent infinite loops

**Depends on:** #2 (Agent ↔ Domain Config Integration), #2.5 (Educational Profile Schema Integration)
**Cross-reference:** `FUTURE_FIXES.md` **#6 (Integration Test Coverage)** — Angelo's 6-scenario matrix (normal ADHD, out-of-scope, low-confidence, `include_explainability=false`, concept graph cap, post-Neo4j MITIGATED_BY) should be adopted verbatim as the E5 acceptance test suite.

---

### Subtask 3: UDL Media Mapping — Fix Script + Generate JSON
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** LM

**Description:**
Fix `generate_media_mapping.py` for UDL support and generate `kg_udl_media_mapping.json`. Currently 3 issues block UDL: wrong KG path, Neuro-specific system prompt, Neuro-specific priority categories.

**Acceptance Criteria:**
- [ ] Fix KG path resolution for UDL (`UDLdata/kg_udl_neo4j.json`)
- [ ] Create UDL-specific system prompt (CAST framework, inclusive education, variability profiles, UDL-specific OER sources)
- [ ] Add UDL priority categories (`Adhd`, `AutismSpectrum`, `Dyslexia`, `UdlPrinciple`, `Barrier`, `MitigationStrategy`, etc.)
- [ ] Create UDL-specific user prompt template
- [ ] Test with `--limit 5` before full run
- [ ] Generate `kg_udl_media_mapping.json` (~763 concepts)

**Depends on:** None
**Reference:** `docs/Media_Mapping_and_Model_Upgrade_Analysis.md` — Part A, Sections A5-A7

---

### Subtask 4: Neuro Media Mapping — Full Expansion (695 concepts)
**Priority:** 🟠 High | **Effort:** 1h | **Assignee:** LM/AG

**Description:**
Run `generate_media_mapping.py` for all ~695 Neuro concepts. Currently only 20 are mapped (2.9% coverage) due to `--limit 20` during initial test run. Cost: ~$5-8. Time: ~40 min.

**Acceptance Criteria:**
- [ ] Run: `python generate_media_mapping.py --domain neuro --batch-size 10`
- [ ] Verify output covers ~695 concepts
- [ ] Spot-check 10 entries for quality (videos, citations, OER)
- [ ] Backup existing 20-concept JSON before overwriting

**Depends on:** None
**Reference:** `docs/Media_Mapping_and_Model_Upgrade_Analysis.md` — Part A, Section A4

---

### Subtask 5: Validate External APIs end-to-end
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** LM

**Description:**
The ExternalMediaAPI (YouTube, Wikipedia, Semantic Scholar, OER) has 1,054 lines of code but has never been tested with real API keys in a live environment. Validate each integration works.

**Acceptance Criteria:**
- [ ] YouTube search: verify results with/without API key (fallback URL mode)
- [ ] Wikipedia search: verify article retrieval
- [ ] Semantic Scholar: verify paper search and rate limiting
- [ ] OER search (DOAB, Open Textbook Library, BC Campus): verify each source
- [ ] Document which API keys are required vs optional in `.env`

**Depends on:** None

---

### Subtask 6: Validate Media Layer end-to-end
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** LM

**Description:**
The full media layer (MediaLookup + ExternalMediaAPI + MermaidGenerator + ImageGenerator + DiagramFactory) must be validated as a whole. Individual components may work but the integration through RetrieverAgent → WriterAgent has not been tested.

**Acceptance Criteria:**
- [ ] MediaLookup loads `kg_neuro_media_mapping.json` correctly
- [ ] MediaLookup loads `kg_udl_media_mapping.json` correctly (requires #3 done first)
- [ ] MermaidGenerator produces valid diagram URLs
- [ ] ImageGenerator produces DALL-E images (requires OpenAI API)
- [ ] DiagramFactory routes correctly to Mermaid and DALL-E (DALL-E fix from #1 required)
- [ ] Full pipeline test: query → Retriever fetches media → Writer embeds media in lesson plan

**Depends on:** #1 (bug fixes), #3 (UDL media JSON), #4 (Neuro media expansion), #5 (external API validation)

---

### Subtask 6.5: Frontend Platform Evaluation & Decision (Research Spike) ⭐ NEW
**Priority:** 🔴 Urgent | **Effort:** 4-6h research + 2h decision doc | **Assignee:** LM (with Diego, Simone, Filippo input)

**Description:**
Research and decide which frontend platform best fits the Agentic GraphRAG **as a deployable independent product** (not just a Streamlit demo). The decision MUST land before #7 (FastAPI Agent Endpoint) is built, because it directly shapes the API contract: auth scheme, CORS, streaming protocol (SSE vs WebSockets), payload shape, session management, and multi-tenancy. If we build #7 first and then choose a platform, we'll have to revise the contract within weeks.

**Working hypothesis (to be confirmed):** Given the team's stated direction of **deploying as a new, independent platform**, the strongest candidate is **Vercel + Next.js + Vercel AI SDK**. This combination gives us first-class SSE streaming (no infrastructure work for #12), the largest ecosystem of LangChain/LangGraph JavaScript bindings, full UI customization for the explainability views Simone designed, near-zero hosting friction (`vercel deploy`), and an unlimited customization ceiling. Lovable remains the best **POC and iteration tool**, while AixLearning native integration is the best **long-term embed path** (the `EducationalProfile` schema from #2.5 already maps 1:1 to FEM production models — no field translation needed).

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
**Blocks:** #7 (FastAPI Endpoint — auth + payload contract), #12 (SSE Streaming — protocol choice), #8 (Guardrails — auth-related rules)
**Unblocks:** CORE 6 (Deployment & Frontend Production — entirely scoped from this decision)

---

### Subtask 7: FastAPI Endpoint for Agent Mode
**Priority:** 🔴 Urgent | **Effort:** 3h | **Assignee:** LM

**Description:**
Create a FastAPI endpoint `POST /api/v1/agent/lesson` that exposes the Agent pipeline to the chosen frontend (decided in #6.5). Currently only GraphRAG mode has an API endpoint (`/api/v1/context`). The Agent mode is only accessible via Streamlit and CLI.

**Acceptance Criteria:**
- [ ] New route file `api/routes/agent.py`
- [ ] Endpoint accepts: `query`, `domain`, `language`, `session_id`, **`educational_profile`** (from #2.5)
- [ ] Returns: lesson plan content, metadata (intent, scope), scores, approval status, **explainability fields** (matching `/api/v1/context` patterns)
- [ ] Error handling and validation (Pydantic)
- [ ] Auth middleware (scheme determined by #6.5)
- [ ] CORS configuration matching the chosen frontend host (#6.5)
- [ ] Registered in `api/main.py`

**Depends on:** #6.5 (Frontend Platform Evaluation — defines auth + payload contract), #2.5 (Educational Profile schema)
**Blocks:** #12 (SSE Streaming)

---

### Subtask 8: Guardrails: Input/Output Validation
**Priority:** 🔴 Urgent | **Effort:** 3-5h | **Assignee:** LM

**Description:**
Add safety guardrails for an educational system. Zero guardrails currently exist — no prompt injection detection, no output validation, no PII protection.

**Acceptance Criteria:**
- [ ] Input: prompt injection detection (regex patterns for "ignore all previous instructions" etc.)
- [ ] Input: query length limits
- [ ] Input: language detection (accept only Italian/English)
- [ ] Output: Pydantic schema validation (lesson plan has required sections)
- [ ] Output: content safety check (OpenAI Moderation API — free)
- [ ] Output: PII detection (no student personal data leakage)

**Depends on:** None

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

Root cause: `lesson_planner_graph.py` line 68 compiles without a checkpointer:
`compiled = workflow.compile()  # ← No checkpointer!`

**Acceptance Criteria:**
- [ ] Add `MemorySaver` (dev) or `PostgresSaver` (production) to `lesson_planner_graph.py`
- [ ] Pass `thread_id` config at invocation using `session_id`
- [ ] Multi-turn test: query → follow-up modification → verify context preserved
- [ ] Update `test_agent.py` to support session mode
- [ ] Update FastAPI endpoint (#7) to accept and pass session_id

**Depends on:** None
**Blocks:** #16 (Long-Term Memory), #19 (Human-in-the-Loop)

---

### Subtask 11: Observability (LangSmith / Langfuse Integration)
**Priority:** 🟠 High | **Effort:** 2h | **Assignee:** LM

**Description:**
Add structured tracing (LangSmith or Langfuse — Langfuse is open-source/self-hostable) for end-to-end observability: visual trace trees, latency breakdowns, cost per request, evaluation datasets. Currently we only have basic `logging` and GlitchTip for crashes — no structured traces, no cost tracking, no way to compare runs. **This subtask is a hard prerequisite for CORE 3** (you can't A/B test models, grade retrieval quality, or measure caching wins without traces).

**Acceptance Criteria:**
- [ ] Add `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` (or Langfuse equivalent) to `.env`
- [ ] Verify trace appears in dashboard for agent pipeline (planner → retriever → writer → critic)
- [ ] Verify cost tracking (tokens per node, $ per request)
- [ ] Document setup in README and `docs/GraphRAG_Data_Pipeline_Guide.md`

**Depends on:** LangSmith API key (free tier: 5K traces/month) OR self-hosted Langfuse instance
**Unblocks:** CORE 3 (#13, #14, #17, #18 all need traces for measurement)

---

### Subtask 12: SSE Streaming to Frontend
**Priority:** 🟠 High | **Effort:** 4-6h | **Assignee:** LM

**Description:**
Add Server-Sent Events (SSE) streaming so the chosen frontend (per #6.5) shows real-time progress ("Planning...", "Searching KG...", "Writing...") and token-by-token output of the lesson plan. Currently users would wait 20-30 seconds with no feedback. **Protocol choice (SSE vs WebSocket vs NDJSON) is determined by #6.5.**

**Acceptance Criteria:**
- [ ] Streaming endpoint (e.g., `POST /api/v1/agent/lesson/stream` for SSE, or upgrade-style for WebSocket — per #6.5)
- [ ] Step-level events (node start/complete for each agent: planner, retriever, writer, critic)
- [ ] Token-level streaming for Writer output
- [ ] Client-side event spec documented for the chosen frontend (Vercel/Next.js + Vercel AI SDK supports the spec natively if that path is chosen in #6.5)

**Depends on:** #7 (FastAPI Endpoint), #6.5 (Frontend Platform Evaluation — defines protocol)

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
- [ ] Resume mechanism documented for frontend (per #6.5 platform)

**Depends on:** #15 (PostgresSaver — required for pause/resume state persistence), #12 (SSE Streaming — for emitting interrupt events to the frontend)

---

### Subtask 20: MCP Tool Servers
**Priority:** 🔵 Low | **Effort:** 2-3 days | **Assignee:** LM

**Description:**
Wrap GraphRAG and Media tools as MCP (Model Context Protocol) servers so external AI tools (Claude Desktop, Cursor, Lovable) can access the Knowledge Graph via standardized protocol. Analysis and implementation plan in reference doc.

**Acceptance Criteria:**
- [ ] MCP Server: graphrag-tools (search_kg, get_context, list_concepts, get_schema)
- [ ] MCP Server: media-tools (lookup_media, generate_diagram, search_youtube)
- [ ] SSE transport for remote access
- [ ] Test with MCP client

**Depends on:** None (additive, doesn't replace existing code)
**Reference:** `docs/Agentic_GraphRAG_Architecture_Analysis.md` — Section 3

---

### Subtask 21: Graph Updater Agent (Phase 3)
**Priority:** 🔵 Low | **Effort:** 2-3 days | **Assignee:** LM

**Description:**
Implement the Graph Updater Agent to extract new concepts from generated lesson plans and propose Knowledge Graph additions. Currently a stub returning empty results (`agent/agents/graph_updater_agent.py`). Critical for keeping the KG fresh as teachers use the system — without it, the KG ossifies.

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
Implement curriculum standards lookup: Italian National Curriculum (Indicazioni Nazionali), European Qualifications Framework, regional school authority standards. Currently a stub returning placeholder data (`agent/tools/curriculum_tool.py`). Lets the agent verify each lesson aligns with required learning outcomes for the teacher's grade level.

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
Implement Canva Connect API for professional template-based slide-deck and worksheet generation. Currently a stub returning "coming soon" (`agent/media/canva_generator.py`). Lets teachers export a generated lesson plan as a polished slide deck or worksheet ready to use in class.

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
| `docs/Agentic_GraphRAG_Architecture_Analysis.md` | Full architecture analysis, best practices gaps (10 items), MCP analysis, memory analysis (3 types), priority roadmap |
| `docs/Media_Mapping_and_Model_Upgrade_Analysis.md` | Media mapping script analysis (3 UDL blocking bugs) + GPT-5.x upgrade strategy (tiered recommendation) |
| `docs/Agent_Domain_Prompt_Integration.md` | 3 options for connecting Agent prompts to domain configs. Option 2 (quick win) → Option 3 (clean architecture) |
| `docs/GraphRAG_Data_Pipeline_Guide.md` | Data pipeline documentation for data ingestion and validation |
| `docs/Agentic_GraphRAG_BestPractices_Validation.md` | **Companion doc** — validates CORE 1–6 plan against 2026 Agentic RAG best practices: subtask coverage audit, tech stack validation (17/17), architecture pattern check, 3 minor gap recommendations |

---

## SUMMARY

### Subtask counts per Core

| Core | Theme | Subtasks | Target |
|---|---|---|---|
| **CORE 0** | Legacy / Pre-existing (E1-E4) | 4 (2 DONE, 2 IN PROGRESS) | — (in progress) |
| **CORE 1** | Agentic Foundations | 8 (#1, #2, **#2.5**, **E5**, #3, #4, #5, #6) | Feb 2026 |
| **CORE 2** | Production-Readiness (API + Safety + Observability) | 7 (**#6.5**, #7, #8, #11, #9, #10, #12) | Mar 2026 |
| **CORE 3** | Quality & Cost (Advanced RAG) | 4 (#13, #14, #17, #18) | Apr 2026 |
| **CORE 4** | Personalization (Memory & Human Loop) | 3 (#15, #16, #19) | May 2026 |
| **CORE 5** | Strategic / Extension Layer | 4 (#20, #21, #22, #23) | Jun+ 2026 |
| **CORE 6** | Deployment & Frontend Production (placeholder) | ~7 (#24-#30, scoped after #6.5) | TBD |
| | | | |
| **Total ticketed subtasks** | | **30** (5 in ClickUp today + 23 new + 2 newly added: #2.5, #6.5) | |
| **Future placeholder subtasks** | | **~7** (CORE 6, scoped post-#6.5) | |

### Effort estimates

| Core | Effort | In days |
|---|---|---|
| CORE 1 | ~14-18h | ~2-3 days |
| CORE 2 | ~22-31h | ~3-4 days |
| CORE 3 | ~15-22h | ~2-3 days |
| CORE 4 | ~13-15h | ~2 days |
| CORE 5 | ~7-11 days | ~7-11 days |
| CORE 6 | TBD (~3-4 weeks once scoped) | ~3-4 weeks |
| **CORE 1-5 total** | **~64-86h + 7-11 days** | **~16-22 working days** |

### Key changes vs previous Tier 1/2/3 layout

1. **Tier-based grouping → Core-based grouping** (matches the existing ClickUp epic structure CORE 0, CORE 1, CORE 2, ...).
2. **E5 moved out of CORE 0 → CORE 1** because its remaining work (UDL Critic + e2e revision validation) is unblocked only by #2 + #2.5.
3. **#2.5 added to CORE 1** — Educational Profile Schema port from `fem/enhanced-variables-extraction`.
4. **#6.5 added as the first subtask of CORE 2** — Frontend Platform Evaluation (working hypothesis: **Vercel + Next.js + Vercel AI SDK** for new independent platform deployment).
5. **CORE 2 expanded** to absorb #11 (Observability) and #10 (Conversation Memory) from old Tier 1, and now includes #6.5 + the streaming/API endpoint chain.
6. **CORE 3 redefined** as the "quality & cost" wave that depends on #11 traces being live.
7. **CORE 4 redefined** as the "personalization" wave (all checkpointer-dependent).
8. **CORE 5 unchanged in scope** — still the strategic/experimental future bucket.
9. **CORE 6 added as a future placeholder** — scoped after #6.5 ADR.
