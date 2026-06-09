# Agentic GraphRAG — Technical Documentation

**Project:** Agentic GraphRAG — Multi-agent Lesson Planning System
**Owner:** FEM AI Team
**Document type:** Technical reference for engineering, integration, and operations
**Version:** 1.0 (first complete draft)
**Last updated:** May 2026

---

## Table of Contents

1. Introduction
   1.1 Purpose
   1.2 Audience
   1.3 Scope
   1.4 Document conventions
   1.5 Glossary (LangGraph, RAG, KG, SSE, Critic, Planner, etc.)

2. System Overview
   2.1 What the system does
   2.2 Key capabilities
   2.3 High-level architecture diagram
   2.4 Technology stack (FastAPI, LangGraph, Neo4j, PostgreSQL, Caddy, OpenRouter, Langfuse)

3. Architecture
   3.1 Logical architecture (Planner → Retriever → Writer → Critic)
   3.2 Component responsibilities (per `src/aix/*` module)
   3.3 Runtime architecture (FastAPI + LangGraph orchestration + Postgres checkpointer)
   3.4 Data architecture (Neo4j KG schema, Postgres tables, agent state)
   3.5 Sequence diagrams (sync run, streaming run, multi-turn)
   3.6 Deployment architecture (Mode A standalone, Mode B native AixLearning)

4. Repository Structure
   4.1 Top-level layout
   4.2 `src/aix/` package map (api, agent, retrieval, webui, mcp, generation, domains, core)
   4.3 `deploy/` folder
   4.4 `scripts/` folder
   4.5 `tests/` folder
   4.6 `docs/` folder

5. API Reference
   5.1 Public API surface (`/api/v1/*`)
   5.2 `POST /api/v1/agent/run` — synchronous JSON contract
   5.3 `POST /api/v1/agent/stream` — SSE event taxonomy
   5.4 `POST /api/v1/context` — legacy GraphRAG endpoint
   5.5 Authentication (Cookie, Bearer JWT, planned Basic Auth)
   5.6 Error model (HTTP codes, error events, retry semantics)
   5.7 OpenAPI / Swagger / ReDoc references
   5.8 Versioning and stability guarantees

6. Agent Pipeline (Internals)
   6.1 LangGraph state machine
   6.2 Planner agent
   6.3 Retriever agent
   6.4 Writer agent (incl. token streaming)
   6.5 Critic agent (revision loop)
   6.6 Checkpointer (SQLite dev / Postgres prod)
   6.7 Multi-turn conversation memory
   6.8 Educational profile schema
   6.9 Domain extensions (UDL / Neuro)

7. Retrieval Layer
   7.1 Knowledge Graph (Neo4j) overview
   7.2 GraphRAG retrieval strategies
   7.3 Hybrid retrieval (vector + graph + external sources)
   7.4 External APIs (Wikipedia, OpenAlex, YouTube, DuckDuckGo)
   7.5 Coverage classification

8. Configuration & Environment
   8.1 `.env` reference (categorized)
   8.2 Local development configuration
   8.3 Production configuration (`.env.prod`)
   8.4 Feature flags (corrective RAG, thinking effort, etc.)
   8.5 Secret management

9. Local Development
   9.1 Prerequisites (Python 3.11/3.12, Docker, Neo4j)
   9.2 Initial setup (venv, requirements.txt)
   9.3 Running the API locally
   9.4 Running the WebUI
   9.5 Running smoke tests
   9.6 Code style (Ruff format/lint, replaces Black + isort + flake8)

10. Testing Strategy
    10.1 Test pyramid (unit / integration / API / contract)
    10.2 Running tests
    10.3 Test data fixtures
    10.4 Adding new tests

11. Deployment
    11.1 Production stack (Docker Compose: app + Postgres + Caddy)
    11.2 Dockerfile (slim Bookworm, non-root, healthcheck)
    11.3 `requirements.lock.txt` (Linux/Python 3.12, hash-pinned)
    11.4 Caddy reverse proxy + auto TLS
    11.5 First deploy procedure
    11.6 Upgrade flow
    11.7 Backup & restore procedures
    11.8 Rollback procedure

12. Observability
    12.1 Langfuse traces (per agent + full pipeline)
    12.2 GlitchTip / Sentry error monitoring
    12.3 Health-check endpoints
    12.4 Structured logging
    12.5 Metrics dashboards

13. Security
    13.1 Authentication model (FastAPI-Users, JWT, planned Basic Auth)
    13.2 Authorization
    13.3 CORS policy
    13.4 Secrets handling
    13.5 Rate limiting
    13.6 Network isolation (internal Docker network)
    13.7 EU AI Act marking (`X-AI-Generated`, Markdown comment, exports)

14. Integration Patterns
    14.1 Mode A — Standalone WebUI
    14.2 Mode B — Native AixLearning integration
    14.3 Mode coexistence rules
    14.4 Reference wrapper (`AgenticGraphRagService` pattern)

15. Performance & SLOs
    15.1 Latency budget per phase
    15.2 Streaming first-event target
    15.3 Cost per interaction
    15.4 Capacity assumptions

16. Operational Runbook
    16.1 Common incidents
    16.2 Debugging guide
    16.3 Restart / redeploy
    16.4 Database maintenance

17. Roadmap & Known Limitations
    17.1 Deferred items (CORE 3 / 4 / 5)
    17.2 LangGraph 1.x upgrade
    17.3 RS256 / multi-issuer JWT migration
    17.4 Frontend evolution

18. Appendices
    A. Environment variables reference (full table)
    B. Glossary
    C. Cross-references to internal docs
    D. Changelog

---

## 1. Introduction

### 1.1 Purpose

This document is the **technical reference** for the Agentic GraphRAG system. It describes the architecture, code organization, public API, deployment model, and operational procedures of the platform. It is intended to allow an engineer who has never seen the project to:

- understand the system at a high level in a few minutes,
- locate the right module or endpoint in the repository,
- integrate the system from another service (e.g. AixLearning),
- deploy, operate, and troubleshoot the system in production,
- extend the agent pipeline, the retrieval layer, or the WebUI safely.

This document is **not** a marketing piece or an executive overview. For that perspective, see the companion document `docs/release/Functional_Documentation.md`, which describes the system from a product, pedagogical, and business standpoint.

### 1.2 Audience

This document is written for several engineering roles. Different sections are more relevant to different readers, but the document is intended to be readable end-to-end.

- **Backend engineers** who will modify, extend, or debug the agent pipeline, retrieval layer, FastAPI surface, or WebUI.
- **Integrators / DEV team** of partner platforms (in particular the AixLearning DEV team) who will call the public API from another service.
- **DevOps / Operations** engineers responsible for the production deployment, observability, backups, and incident response.
- **Security and compliance engineers** who need to verify authentication, secret handling, network isolation, and EU AI Act / GDPR controls.
- **New team members** onboarding to the AI Team who need a single entry point into the codebase.

Readers are expected to be familiar with Python, async I/O, FastAPI, Docker, REST/SSE, and basic graph database concepts. No prior knowledge of LangChain, LangGraph, or this specific repository is assumed.

### 1.3 Scope

This document covers:

- the **architecture** of the Agentic GraphRAG system, including the multi-agent pipeline, retrieval layer, WebUI, and public API;
- the **repository layout**, with a map of the `src/aix/*` packages and supporting folders;
- the **public API contract** (`/api/v1/*`), including request/response schemas, authentication, SSE event taxonomy, and stability guarantees;
- the **agent pipeline internals** (Planner, Retriever, Writer, Critic), including state machine, checkpointer, and educational profile;
- the **retrieval layer** (Neo4j Knowledge Graph, hybrid retrieval, external APIs);
- **local development** (environment setup, running the API and WebUI, tests, Ruff-based code style);
- **deployment** of the standalone production pilot (Docker Compose, Caddy, Postgres, backups, rollback);
- **observability and security** (Langfuse, GlitchTip, health checks, CORS, secrets, network isolation);
- **integration patterns** for the two supported deployment modes (Mode A standalone, Mode B native AixLearning);
- **performance targets, SLOs, and operational runbooks** at the level required for a pilot deployment;
- the **roadmap** of explicitly deferred items and known limitations.

This document does **not** cover:

- the pedagogical model in depth (covered by the functional documentation);
- AixLearning-internal Django code (covered by the AixLearning team's own documentation);
- the regulatory framework in depth — that is covered by `docs/product/Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md`;
- the internal pilot deployment plan with timing and ownership — that is covered by `docs/product/Internal_Production_Deployment_Plan.md`.

The document focuses on the current state of the codebase plus the deployment artifacts already shipped in `deploy/`.

### 1.4 Document conventions

The document uses the following conventions to remain unambiguous and easy to scan.

- **Code references** use backticks for module paths (e.g. `src/aix/agent/orchestrator.py`), function/class names (e.g. `stream_agent_events`), endpoints (e.g. `POST /api/v1/agent/stream`), environment variables (e.g. `LANGGRAPH_DATABASE_URL`), and shell commands.
- **Code blocks** use fenced syntax. Configuration examples use environment-variable syntax (`KEY=value`); shell examples are explicitly marked when they are PowerShell vs. Bash.
- **API examples** follow the JSON shapes defined under `src/aix/api/schemas/*` and the OpenAPI spec served at `/openapi.json`. Where possible, examples mirror the Swagger UI examples ("minimal" and "rich").
- **Diagrams** are kept simple and ASCII-friendly where they help comprehension; larger architecture diagrams live as PNGs under `docs/mockups/`.
- **Status callouts** use plain prefixes: `Note:`, `Warning:`, `Limitation:`, and `Recommendation:`. They are not stylized boxes, to keep the document portable.
- **Stability promises** are stated explicitly. Anything not explicitly marked stable should be considered subject to change, and should not be relied on by external integrators.
- **Cross-references** to other internal documents use repository-relative paths (e.g. `docs/product/Dev_Handoff_AgenticGraphRAG_Integration.md`) so they remain valid when the documentation is read offline or in a different rendering tool.

### 1.5 Glossary

Common terms used throughout this document are listed here. A more exhaustive glossary is provided in Appendix B.

- **Agentic GraphRAG** — the overall system described by this document: a multi-agent pipeline that produces lesson plans grounded in a Knowledge Graph and supplemented by external sources.
- **Agent** — a discrete unit of reasoning in the pipeline (Planner, Retriever, Writer, Critic), implemented as a node in a LangGraph state machine.
- **LangGraph** — a state-machine framework for orchestrating multi-step LLM workflows, used to compose and run the agent pipeline.
- **GraphRAG** — Retrieval-Augmented Generation where the retrieval layer is built on top of a Knowledge Graph rather than (only) a vector store.
- **Knowledge Graph (KG)** — the Neo4j-backed graph of pedagogical concepts and relationships (UDL and Neuroscience domains) used for grounded retrieval.
- **Educational profile** — the structured description of a class, classroom, and time/subject context used by the agents to specialize their output.
- **Checkpointer** — the LangGraph component that persists conversation state (multi-turn memory) to a backing store; SQLite in development, PostgreSQL in production.
- **Critic loop** — the revision cycle in which the Critic agent evaluates the Writer's output and may request a revision.
- **SSE (Server-Sent Events)** — the streaming protocol used by `POST /api/v1/agent/stream` to emit per-phase agent events to clients.
- **WebUI** — the internal teacher-facing web interface served by the same FastAPI process at `/webui/*`, used for the standalone internal pilot.
- **Mode A / Mode B** — the two supported deployment modes (standalone WebUI vs. native AixLearning integration); see §3.6 and §14.
- **AixLearning** — the partner Django platform that integrates the Agentic GraphRAG service in Mode B.
- **Langfuse / GlitchTip** — the third-party observability tools used for tracing and error monitoring.
- **Caddy** — the reverse proxy used in production, terminating TLS via Let's Encrypt and forwarding traffic to the FastAPI container.

---

## 2. System Overview

### 2.1 What the system does

The Agentic GraphRAG system turns a teacher's natural-language request (in Italian or English) into a structured, pedagogically-grounded lesson plan. A request such as *"Crea una lezione di 45 minuti sulla fotosintesi adattata a una classe con 2 studenti DSA"* is processed by a multi-agent pipeline that:

1. **understands** the request (intent + scope detection),
2. **retrieves** grounded knowledge from a Neo4j Knowledge Graph and, when needed, verified external sources,
3. **writes** a complete lesson plan specialized to the class profile,
4. **reviews** the result against quality criteria and revises it if necessary.

The system runs two complementary modes from a **single FastAPI process**:

- a **standalone teacher WebUI** (`/webui/*`) used by the internal FEM pilot, and
- a **public JSON + SSE API** (`/api/v1/agent/*`) consumed by non-browser clients (the AixLearning backend, Postman/curl, future apps).

Both modes drive the **same agent pipeline** and the **same retrieval layer**.

### 2.2 Key capabilities

- **Multi-agent orchestration** (Planner → Retriever → Writer → Critic) on a LangGraph state machine.
- **Intent detection** across 7 query types (lesson creation, activity design, definition, comparison, explanation, recommendation, list).
- **Scope detection** relative to the Knowledge Graph (`in_scope` / `partial_scope` / `out_of_scope`).
- **Hybrid retrieval**: Neo4j graph traversal + Node2Vec/semantic embeddings + curated media + verified external sources (Wikipedia, OpenAlex, OER, YouTube).
- **Educational profile specialization**: every request can carry a structured class/classroom profile (grade, BES/DSA, resources, time budget).
- **Multi-turn conversation memory** via a LangGraph checkpointer (SQLite in dev, PostgreSQL in production), with summary-buffer windowing for long threads.
- **Streaming**: per-phase Server-Sent Events for incremental UI; live writer-token streaming in the WebUI.
- **Quality control**: a Critic agent scores the draft and can trigger a bounded revision loop.
- **Optional Corrective RAG**: a retrieval-grading loop (off by default, flag-gated).
- **MCP tool servers**: the KG, media, and agent capabilities are also exposed via Model Context Protocol (stdio + Streamable HTTP).
- **Observability**: Langfuse tracing + GlitchTip/Sentry error monitoring + a startup connectivity probe.

### 2.3 High-level architecture diagram

```
                         ┌────────────────────────────────────────────┐
   Browser (teacher)     │                FastAPI process              │
   ───────────────►──────┤  (single uvicorn app — aix.api.main:app)    │
   /webui/*  (HTML+SSE)   │                                            │
                          │   /webui/*        Teacher WebUI (htmx)      │
   Non-browser client     │   /api/v1/context Legacy GraphRAG API       │
   ───────────────►──────┤   /api/v1/agent/* Agent API (JSON + SSE)    │
   /api/v1/agent/* (JSON) │   /mcp/           MCP Streamable HTTP        │
                          │   /auth/jwt/*     JWT login                  │
                          │   /docs /openapi.json  Swagger / OpenAPI    │
                          │                                            │
                          │   ┌──────────────── Agent pipeline ──────┐ │
                          │   │ Planner → Retriever → Writer → Critic │ │
                          │   │            (LangGraph)        ↑   ↓   │ │
                          │   │                               └revise┘ │ │
                          │   └───────┬───────────────┬──────────────┘ │
                          └───────────┼───────────────┼────────────────┘
                                      │               │
                              ┌───────▼──────┐  ┌─────▼─────────────┐
                              │   Neo4j KG   │  │ PostgreSQL (prod) │
                              │ (UDL/Neuro)  │  │ webui + langgraph │
                              └──────────────┘  └───────────────────┘
                                      │
                              ┌───────▼──────────────────────────────┐
                              │ External sources / LLM provider       │
                              │ OpenRouter (LLM), Wikipedia, OpenAlex, │
                              │ OER, YouTube                          │
                              └───────────────────────────────────────┘
```

### 2.4 Technology stack

| Layer | Technology |
|---|---|
| API + serving | FastAPI (single uvicorn process), `sse-starlette` |
| Agent orchestration | LangChain + LangGraph |
| Knowledge Graph | Neo4j (Aura / FEM-managed instance) |
| Embeddings | Node2Vec (graph) + OpenAI-compatible text embeddings (hybrid) |
| LLM provider | OpenRouter (default `anthropic/claude-sonnet-4-6`); OpenAI as fallback |
| State / memory | LangGraph checkpointer — SQLite (dev) / PostgreSQL (prod) |
| WebUI persistence | SQLAlchemy async — SQLite (dev) / PostgreSQL (prod) |
| WebUI frontend | Jinja2 + htmx 2 + WebAwesome + Tailwind + Alpine.js |
| Auth | FastAPI-Users (cookie + Bearer JWT, HS256) |
| Tool protocol | FastMCP 3.x (stdio + Streamable HTTP) |
| Reverse proxy / TLS | Caddy 2 (Let's Encrypt) |
| Observability | Langfuse (tracing), GlitchTip/Sentry (errors) |
| Packaging / tooling | `pyproject.toml` (src layout), Ruff (lint+format), mypy, pytest |

---

## 3. Architecture

### 3.1 Logical architecture (Planner → Retriever → Writer → Critic)

The pipeline is a LangGraph `StateGraph` whose nodes are the four agents. A shared `AgentState` (TypedDict) flows through every node; each node reads the fields it needs and writes its outputs back.

```
plan ──► retrieve ──► write ──► critique ──► [revise | finish]
                                   ▲              │
                                   └──────────────┘   (bounded revision loop)
```

- **Planner** (`plan`) — classifies intent + scope, extracts key concepts, and produces the search queries.
- **Retriever** (`retrieve`) — runs GraphRAG searches against Neo4j, attaches curated media, and (for out-of-scope topics) verified external sources.
- **Writer** (`write`) — generates the lesson plan, specialized by domain prompt + educational profile + optional teacher-provided context.
- **Critic** (`critique`) — scores the draft; on non-approval (and within `max_revisions`) routes back to `write` with revision instructions.

When `AIX_CORRECTIVE_RAG_ENABLED=true`, an extra `grade_retrieval` node is inserted between `retrieve` and `write`, with a bounded retry edge back to `retrieve` (see §6.6).

### 3.2 Component responsibilities (per `src/aix/*` module)

| Package | Responsibility |
|---|---|
| `aix.core` | Shared configuration (`config.py`), connectivity probe, cross-cutting utilities |
| `aix.retrieval` | GraphRAG retrieval: Text2Cypher, hybrid graph retriever, context builder, query metrics |
| `aix.generation` | LLM response generation for the legacy GraphRAG path (`llm_chain.py`) |
| `aix.agent` | Agentic pipeline: orchestrator, LangGraph graph/nodes/state, the 4 agents, prompts, media, tools, domain prompt configs |
| `aix.api` | FastAPI app (`main.py`), routes (`context`, `agent`), Pydantic schemas, helper client |
| `aix.webui` | Teacher WebUI: auth, lessons (CRUD/uploads), agent streaming service, Jinja2 templates, DB |
| `aix.mcp` | MCP tool servers: composition root, stdio entry, Streamable HTTP factory, tools/resources/prompts |
| `aix.domains` | Domain configs and prompt knowledge for UDL and Neuroscience |

### 3.3 Runtime architecture

A single uvicorn process (`aix.api.main:app`) hosts every surface. At import/startup it:

- applies a Windows event-loop policy shim when Postgres is configured (psycopg async requires the selector loop on Windows; Linux production is unaffected);
- optionally initializes Sentry/GlitchTip when `SENTRY_DSN` is set;
- builds the MCP Streamable HTTP sub-app (guarded — a build failure cannot block `/api/v1`);
- on `lifespan` startup: verifies Neo4j connectivity, checks domain configs, runs the optional LLM connectivity probe, and sets up the LangGraph checkpointer;
- mounts routers: `/api/v1/context`, `/api/v1/agent/*`, `/webui/*`, `/auth/jwt/*`, `/static`, and `/mcp/`.

The agent pipeline is driven by `aix.webui.agent.service`, which owns the LangGraph `astream` loop and the translation of state diffs into normalized `StreamEvent`s. The compiled graph and its agents are module-level singletons (acceptable for the pilot; revisit for high concurrency — see §15/§17).

### 3.4 Data architecture

- **Neo4j Knowledge Graph** — pedagogical concepts, methodologies, strategies, and their relationships, in two domains (`neuro`, `udl`). Read-only at request time. Source dumps live under `data/kg/{neuro,udl}/`.
- **PostgreSQL (production)** — a single instance backing two logical concerns:
  - **WebUI DB** (`WEBUI_DATABASE_URL`): users, lessons, lesson messages (multi-turn transcript).
  - **LangGraph checkpointer** (`LANGGRAPH_DATABASE_URL`): three tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) storing per-thread agent state.
  - In development both default to SQLite files.
- **Agent state** — the in-flight `AgentState` TypedDict (see §6.1), persisted per `thread_id` by the checkpointer to enable multi-turn follow-ups.
- **Artifacts** — Node2Vec embeddings and the OpenAI embeddings cache under `artifacts/` (mounted as a Docker volume in production so a rebuild doesn't trigger a full re-embed).

### 3.5 Sequence diagrams

**Synchronous run (`POST /api/v1/agent/run`):**

```
Client → /api/v1/agent/run (JSON)
  → auth (cookie or Bearer JWT)
  → stream_agent_events(...) drained to completion
      plan → retrieve → write → critique [→ revise]
  → assemble AgentRunResponse { lesson_plan_md, meta, planner, retriever }
  → 200 JSON   (or 502 if the pipeline errored)
```

**Streaming run (`POST /api/v1/agent/stream`):**

```
Client → /api/v1/agent/stream (JSON body)  →  text/event-stream
  event: planner   data: {...}
  event: retriever data: {...}
  event: writer_pending
  event: writer    data: {...}  lesson_plan_md: "<draft>"
  event: critic    data: {...}
  (… revision loop may repeat writer_pending/writer/critic …)
  event: done      lesson_plan_md: "<final>"  data(meta): {...}
  # heartbeat ping every 15s keeps proxies from closing the idle connection
```

### 3.6 Deployment architecture (Mode A vs Mode B)

- **Mode A — Standalone internal pilot.** Browser → Caddy (80/443, TLS) → `app:8765`. The app serves `/webui/*` and `/api/v1/*`. PostgreSQL is private to the Docker network. This is the FEM internal pilot at `https://agente.aiforlearning.digital`.
- **Mode B — Native AixLearning integration.** The AixLearning Django backend/worker calls the agent service over the internal Docker network (`http://graphrag-api:8765/api/v1/agent/...`). AixLearning owns its own UX and data; the agent service owns only its own state.

Full treatment is in §14 and in `docs/product/Dev_Handoff_AgenticGraphRAG_Integration.md`.

---

## 4. Repository Structure

### 4.1 Top-level layout

The project uses the modern **src layout**: all importable code lives under `src/aix/`, exposed as the `aix.*` package via `pip install -e .`.

```
graphaixlearning/
├── src/aix/            # All importable source (import as aix.*)
├── apps/               # User-facing entry points (NOT importable libs)
│   ├── streamlit/      #   Legacy Streamlit demo (retirement banner)
│   └── cli/run_agent.py#   Interactive agent testing CLI
├── scripts/            # Operational & data-prep scripts
│   ├── ingest/ audit/ data_prep/ ml/ diagnostic/ ops/ media_pool/
├── data/               # KG dumps, media mappings, reference, reports
│   └── kg/{neuro,udl}/ #   Knowledge graph core dumps
├── artifacts/          # ML artifacts: node2vec/, embeddings_cache/
├── tests/              # unit / integration / api / mcp_server suites
├── deploy/             # Production stack (compose, Caddyfile, .env.prod.example, scripts)
├── docs/               # Documentation (api, architecture, integrations, product, release, …)
├── Dockerfile          # Container build (api target, non-root, healthcheck)
├── requirements.txt    # Runtime deps (single source of truth)
├── requirements.lock.txt # Hash-pinned lockfile for prod (Python 3.12 / Linux)
├── pyproject.toml      # Build, deps, pytest, ruff, mypy
└── Makefile            # make test / api / streamlit / agent
```

### 4.2 `src/aix/` package map

```
src/aix/
├── core/
│   ├── config.py             # `from aix.core.config import config` — Neo4j/LLM/embeddings config
│   └── connectivity_probe.py # one-shot startup LLM-endpoint probe (TLS/DNS/401/timeout)
├── retrieval/                # GraphRAG retrieval layer
│   ├── text2cypher.py            # NL → Cypher conversion (+ self-repair)
│   ├── multilingual_text2cypher.py # IT/EN translation wrapper around text2cypher
│   ├── graph_retriever.py        # hybrid graph + vector retrieval (Node2Vec/OpenAI)
│   ├── context_builder.py        # raw graph data → structured educational context
│   └── query_metrics.py          # retrieval telemetry
├── generation/
│   └── llm_chain.py          # legacy GraphRAG response generation
├── agent/                    # Agentic GraphRAG (multi-agent pipeline)
│   ├── orchestrator.py       # `from aix.agent import AgentOrchestrator` — clean entry point
│   ├── agents/               # planner_agent, retriever_agent, writer_agent, critic_agent
│   ├── graph/                # LangGraph: state.py, nodes.py, lesson_planner_graph.py,
│   │                         #   checkpointer.py, write_stream.py
│   ├── prompts/              # intent-specific prompts (planner/writer/critic)
│   ├── media/                # media lookup, diagram/image generation, resource lookup
│   ├── tools/                # GraphRAG + curriculum tool wrappers for agents
│   └── configs/              # domain prompt extensions (domain_prompts.py)
├── api/                      # FastAPI service
│   ├── main.py               # uvicorn aix.api.main:app — mounts every surface
│   ├── routes/context.py     # /api/v1/context (legacy GraphRAG)
│   ├── routes/agent.py       # /api/v1/agent/run + /stream (Agent API)
│   ├── schemas/              # Pydantic models (agent.py, models.py, educational_profile.py)
│   └── graphrag_client.py    # helper client for the DEV team
├── webui/                    # Teacher WebUI (htmx + WebAwesome)
│   ├── auth/                 # FastAPI-Users: manager, backend, dependencies, models, routes
│   ├── lessons/              # lesson CRUD, uploads, schemas, models, display
│   ├── agent/service.py      # run_agent_stream + stream_agent_events (the engine seam)
│   ├── templates/            # Jinja2: _base.html, pages/, partials/
│   ├── routes.py             # /webui/* handlers
│   └── db.py                 # SQLAlchemy async engine (+ aiosqlite dev default)
├── mcp/                      # MCP Tool Servers (FastMCP 3.x)
│   ├── server.py             # build_mcp_server() composition root
│   ├── stdio_main.py         # stdio entry (Claude Desktop / Cursor IDE)
│   ├── http_app.py           # Streamable HTTP factory (JWT Bearer at /mcp/)
│   ├── tools/ resources/ prompts/
└── domains/                  # Domain configs (udl_domain.py, neuro_domain.py, base_config.py)
```

### 4.3 `deploy/` folder

Production stack and runbook (Wave 1 of the internal deployment plan):

```
deploy/
├── docker-compose.prod.yml  # app + postgres(16) + caddy(2); app/postgres internal-only
├── Caddyfile                # reverse proxy + auto Let's-Encrypt TLS (parameterized by $AIX_DOMAIN)
├── .env.prod.example        # production env template (copy → .env.prod, chmod 600)
├── scripts/                 # backup_postgres / restore_postgres / backup_caddy
└── README.md                # first-deploy, backups, rollback, log inspection runbook
```

### 4.4 `scripts/` folder

Operational and data-prep tooling, grouped by purpose: `ingest/` (Neo4j import/export), `audit/` (KG label checks), `data_prep/` (cleaning/merging), `ml/` (Node2Vec training, media mapping generation), `diagnostic/` (MCP smoke tests, OpenAPI baseline capture), `ops/` (preflight, migrations), `media_pool/` (offline media pool generation via LM Studio).

### 4.5 `tests/` folder

`unit/` (pure, no external services), `integration/` (Neo4j/LLM — marked `@pytest.mark.integration`), `api/` (agent API contract tests), `mcp_server/` (19-test MCP regression suite), plus `conftest.py` shared fixtures. Pytest is configured in `pyproject.toml` with markers `integration`, `slow`, `unit` and `asyncio_mode = "auto"`.

### 4.6 `docs/` folder

`api/` (integration guides), `architecture/` (frontend evaluation/ADRs, model analyses), `integrations/` (MCP setup), `product/` (deployment plan, dev handoff, ClickUp tracker, regulatory alignment), `release/` (this document + the functional documentation), plus reports and runbooks.

## 5. API Reference

### 5.1 API surface overview

All HTTP surfaces are served by the single FastAPI app (`aix.api.main:app`). The OpenAPI spec is published at `/openapi.json`, with interactive docs at `/docs` (Swagger UI) and `/redoc`.

| Path | Method | Purpose | Auth |
|---|---|---|---|
| `/api/v1/health` | GET | Liveness/readiness probe | none |
| `/api/v1/context` | POST | Legacy GraphRAG context (single-shot retrieval) | service auth |
| `/api/v1/agent/run` | POST | Run the agent, return final lesson plan (sync JSON) | cookie or Bearer JWT |
| `/api/v1/agent/stream` | POST | Run the agent, stream phases as SSE JSON | cookie or Bearer JWT |
| `/auth/jwt/login` | POST | Mint a Bearer JWT (FastAPI-Users) | credentials |
| `/webui/*` | GET/POST | Teacher WebUI (HTML + htmx + SSE) | cookie session |
| `/mcp/` | (MCP) | MCP Streamable HTTP transport | Bearer JWT |
| `/docs`, `/redoc`, `/openapi.json` | GET | API documentation | none |

The two surfaces the integration team consumes are **`/api/v1/agent/run`** and **`/api/v1/agent/stream`**. They are *additive* — they do not alter `/api/v1/context`, `/webui/*`, or `/auth/*`.

### 5.2 Authentication

Both agent endpoints depend on `current_active_user` (FastAPI-Users), which accepts **either**:

- the **WebUI session cookie** (used by browser clients), or
- an **`Authorization: Bearer <jwt>`** header (used by API/integration clients).

Tokens are HS256, signed with `WEBUI_AUTH_SECRET` (shared by cookie and Bearer backends). To obtain a Bearer token programmatically:

```bash
curl -X POST https://agente.aiforlearning.digital/auth/jwt/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=teacher@example.org&password=•••••"
# → { "access_token": "<jwt>", "token_type": "bearer" }
```

Note: the legacy `/api/v1/context` endpoint uses a separate service-to-service auth scheme (HTTP Basic), as documented in the integration guide; it is not affected by the FastAPI-Users layer.

### 5.3 `POST /api/v1/agent/run` — synchronous JSON

Drains the full **Planner → Retriever → Writer → Critic** pipeline and returns a single response. Typical run: **60–120 s** (the Writer LLM call dominates). For incremental UI use `/stream` instead.

**Request body** (`AgentRunRequest`):

| Field | Type | Req. | Notes |
|---|---|---|---|
| `query` | string (3–2000) | ✅ | Teacher request in natural language |
| `domain` | `"neuro"` \| `"udl"` | ✅ | Knowledge graph domain |
| `language` | `"it"` \| `"en"` | — | Output language (default `it`) |
| `session_id` | string (≤128) | — | Correlation/thread id; UUID4 generated if omitted |
| `educational_profile` | `EducationalProfile` | — | Class/classroom context (CORE 1 #2.5) |
| `teacher_provided_context` | string (≤48000) | — | Joined text from uploads; Writer-only, **not** ingested into the KG |
| `max_revisions` | int (0–4) | — | Critic loop cap; `null` → `AIX_MAX_REVISIONS` (default 1); `0` disables |

Minimal request:

```json
{ "query": "Crea una lezione sull'attenzione", "domain": "neuro" }
```

**Response** (`AgentRunResponse`):

```json
{
  "lesson_plan_md": "# Lezione: ...",
  "meta": {
    "duration_seconds": 73.2,
    "approved": true,
    "revision_count": 0,
    "scores": { "average_score": 4.4 },
    "nodes_count": 14,
    "recommendations_count": 5,
    "media_counts": { "videos": 3, "articles": 2, "oer": 4 },
    "search_queries_count": 4
  },
  "planner": {
    "intent": "lesson_creation", "intent_label": "Creazione lezione",
    "scope": "in_scope", "scope_label": "Nel Knowledge Graph",
    "key_concepts": ["attenzione", "memoria di lavoro"],
    "search_queries": ["strategie attenzione DSA", "..."]
  },
  "retriever": {
    "nodes_count": 14, "relationships_count": 31, "recommendations_count": 5,
    "top_concepts": ["Attenzione sostenuta", "Self-regulation"],
    "retrieval_confidence": "HIGH"
  }
}
```

**Status codes:** `200` success · `401` missing/invalid auth · `422` body validation error · `502` agent pipeline runtime error (LLM failure, KG unreachable — potentially retryable).

### 5.4 `POST /api/v1/agent/stream` — Server-Sent Events

Same request body as `/run`. Returns `text/event-stream`. Each frame carries an `event:` line (the `kind`) and a JSON `data:` payload. A heartbeat comment is sent every **15 s** so proxies/load balancers don't drop the idle connection during the slow Writer call.

```bash
curl -N -X POST https://agente.aiforlearning.digital/api/v1/agent/stream \
  -H "Authorization: Bearer <jwt>" -H "Content-Type: application/json" \
  -d '{"query":"Crea una lezione sull'\''attenzione","domain":"neuro"}'
```

Wire shape per frame:

```
event: planner
data: {"kind":"planner","data":{...},"lesson_plan_md":null,"error":null}
```

> Note: Swagger UI's "Try it out" renders the whole stream as one blob. For live event-by-event inspection use `curl -N`, Postman, or Bruno.

### 5.5 SSE event taxonomy

The public stream (`stream_agent_events`) emits the 7 event kinds frozen by the `AgentStreamEvent` union in `src/aix/api/schemas/agent.py`. Every frame has the same outer envelope: `{ kind, data, lesson_plan_md, error }`.

| `kind` | When | Key `data` fields |
|---|---|---|
| `planner` | after `plan` | `intent`, `intent_label`, `scope`, `scope_label`, `scope_confidence`, `key_concepts[]`, `search_queries[]` |
| `retriever` | after `retrieve` | `nodes_count`, `relationships_count`, `recommendations_count`, `media_counts{videos,articles,oer}`, `media{}`, `top_concepts[]`, `retrieval_confidence`, coverage tier |
| `writer_pending` | before a write attempt | `revision`, `is_revision`, `feedback` |
| `writer` | after `write` | `revision`; `lesson_plan_md` = draft for this revision |
| `critic` | after `critique` | `approved`, `revision_count`, `max_revisions`, `score`, `score_pct`, `critique`, `revision_instructions` |
| `done` | end of run | (envelope) `lesson_plan_md` = final; `data`/meta = run summary |
| `error` | on failure | `error` = short message (≤480 chars) |

Happy-path order (0 revisions): `planner → retriever → writer_pending → writer → critic → done`.
With one revision: `planner → retriever → writer_pending → writer → critic → writer_pending → writer → critic → done`.

Implementation note: clients should `switch` on `kind` and ignore unknown kinds (forward-compatible). The browser WebUI uses additional internal-only kinds (`retriever_pending`, `critic_pending`, `writer_chunk` for live token streaming) that are **not** part of the public union; non-browser clients will not receive them from `/api/v1/agent/stream`.

### 5.6 Educational profile schema

`educational_profile` reuses the same `EducationalProfile` model the WebUI form serializes (`src/aix/api/schemas/educational_profile.py`), so a profile is interchangeable across both surfaces. It carries:

- **`group`** — class context: `title`, `students_number`, `grade` (e.g. `SECONDARIA_II_GRADO`), `disabilities` (e.g. `ADHD`, `DSA`), `class_features`, `student_attributes`.
- **`classroom`** — physical context: `title`, `forniture_mobility`, `has_lim`, `has_wifi`, `has_suite`, `pc_station`, `own_device`.
- **top-level** — `time_available_minutes`, `subject_area`, `specific_topic`.

All fields are optional; omitting the profile makes the agent fall back to generic prompts. The Swagger UI exposes two named examples — **minimal** and **rich** — driven by `openapi_examples` in `routes/agent.py`.

### 5.7 Legacy GraphRAG context endpoint

`POST /api/v1/context` (`routes/context.py`) is the original single-shot retrieval endpoint that powered the first AixLearning integration. It returns structured GraphRAG context (concepts, methodologies, recommendations) without running the agent pipeline. It remains supported for backward compatibility; new integrations should prefer the agent endpoints.

### 5.8 Error model

- Agent endpoints raise `502 Bad Gateway` (not `500`) when the *pipeline* fails — communicating that the route itself didn't crash and the request may be retryable.
- The SSE stream never lets an exception cross the generator boundary: failures are emitted as a terminal `error` event (treated as domain data).
- Validation errors (`422`) follow FastAPI's standard `{"detail": [...]}` shape.

---

## 6. Agent Pipeline (Internals)

### 6.1 `AgentState` — the shared contract

`src/aix/agent/graph/state.py` defines `AgentState`, a `TypedDict(total=False)` that flows through every node. It is grouped into input, per-agent outputs, corrective-RAG fields, metadata, and final output. Key fields:

- **Input**: `teacher_query`, `domain`, `language`, `session_id`, `educational_profile`, `teacher_provided_context`, `conversation_history`, `conversation_summary`, `raw_user_turn`.
- **Planner output**: `query_intent`, `lesson_type`, `target_grade`, `key_concepts`, `search_queries`, `scope_status`, `scope_confidence`, `subject_concepts`, `pedagogy_concepts`.
- **Retriever output**: `graphrag_results`, `retrieved_nodes`, `retrieved_relationships`, `recommendations`, `retrieval_confidence`, `curated_media`, `external_resources`.
- **Corrective-RAG** (only when enabled): `retrieval_grade`, `retrieval_grade_reason`, `retrieval_attempts`, `retrieval_rewritten_query`, `retrieval_warning`.
- **Writer output**: `lesson_plan_draft`, `lesson_plan_structured`, `sources_cited`.
- **Critic output**: `critique`, `critique_score`, `approved`, `revision_instructions`.
- **Metadata / final**: `revision_count`, `max_revisions`, `current_step`, `error`, `final_lesson_plan`, `final_metadata`.

`create_initial_state(...)` is the single source of truth for the agent's input shape; both the WebUI service and the public API construct state through it. New nullable fields are designed to be additive so older callers behave identically.

### 6.2 Planner agent

`agent/agents/planner_agent.py` (node `plan`) performs **intent detection** (7 `QueryIntent` types), **scope detection** (`ScopeStatus`: `in_scope` / `partial_scope` / `out_of_scope` / `unknown`), and extracts `key_concepts` + `search_queries`. It also acts as **language layer L1**: it sees the full augmented query and can override the statistical seed language (L2 = `lingua` detector, L3 = default `it`). `plan_node` applies user-vs-history precedence on duration via `raw_user_turn` (the un-augmented current turn).

### 6.3 Retriever agent

`agent/agents/retriever_agent.py` (node `retrieve`) executes the planner's `search_queries` against the GraphRAG retrieval layer (§7), aggregates nodes/relationships/recommendations, and assembles `curated_media` (videos, resources, citations, open_textbooks). For `partial_scope`/`out_of_scope` topics it attaches `external_resources` (Wikipedia, OER, papers) so the Writer can still compose a useful lesson. Uploaded teacher context is **never** sent to the retriever — it is Writer-only.

### 6.4 Writer agent

`agent/agents/writer_agent.py` (node `write`) generates the lesson plan. Its prompt is assembled from the **intent-specific prompt** (`agent/prompts/`), the **domain prompt extension** (`agent/configs/`), the **educational profile**, the **retrieved context + media**, optional **teacher-provided context**, and any **conversation history/summary**. Output length is bounded by `AIX_WRITER_MAX_TOKENS` (default 3500) with up to `AIX_WRITER_MAX_CONTINUATIONS` automatic continuations when the model hits `finish_reason="length"`. In the WebUI, writer tokens stream live (`writer_chunk`); the public API delivers the writer output as a single `writer` event per revision.

### 6.5 Critic agent and the revision loop

`agent/agents/critic_agent.py` (node `critique`) scores the draft on multiple criteria (average on a 1–5 scale) and returns an approve/revise decision. `should_continue_to_revision` routes the run:

- `approved == true` → `finish` (END);
- `not approved` **and** `revision_count < max_revisions` → `revise` (back to `write`) with `revision_instructions`;
- otherwise → `finish`.

`max_revisions` defaults to `AIX_MAX_REVISIONS` (1). Robustness flags: `AIX_CRITIC_PARSE_ERROR_BEHAVIOR` (`approve`/`revise`/`raise`) controls behavior on unparseable critic JSON; `AIX_CRITIC_MODEL` (default a fast model) keeps the ~300-token classification cheap; `AIX_CRITIC_LESSON_MAX_CHARS` / `AIX_CRITIC_CONTEXT_MAX_CHARS` cap prefill to reduce latency.

### 6.6 Corrective RAG (optional)

Gated by `AIX_CORRECTIVE_RAG_ENABLED` (default **off**). When on, `_build_workflow()` inserts a `grade_retrieval` node after `retrieve`:

```
plan → retrieve → grade_retrieval ─[continue]→ write → critique → [revise|finish]
                        │
                        └─[retry]→ retrieve   (bounded by AIX_CORRECTIVE_RAG_MAX_ATTEMPTS, default 2)
```

A grader LLM classifies the retrieval as `relevant` / `ambiguous` / `irrelevant`. Non-relevant grades trigger a bounded retry with a rewritten query; after the attempt budget is exhausted, the run falls through to the Writer with `retrieval_warning=true` so the lesson carries an explicit low-confidence caveat. When the flag is off, the topology is byte-identical to the pre-feature pipeline.

### 6.7 LangGraph orchestration

`agent/graph/lesson_planner_graph.py` builds the `StateGraph`. Two compile paths share one topology (`_build_workflow`):

- `build_lesson_planner_graph()` (sync, **no** checkpointer) — for legacy/ephemeral callers.
- `build_lesson_planner_graph_async()` (async, **with** the checkpointer when available) — used by the WebUI, the public API, and MCP.

`LessonPlannerPipeline` wraps the compiled graph; `AgentOrchestrator` (`agent/orchestrator.py`) is the public, ergonomic entry point (`create_lesson_plan(...)`). The execution engine for streaming is `aix.webui.agent.service` (`run_agent_stream` for the WebUI/DB path, `stream_agent_events` for the DB-less API path) — both drive `graph.astream(..., stream_mode="updates")` and translate state diffs into `StreamEvent`s.

### 6.8 Multi-turn memory & checkpointing

`agent/graph/checkpointer.py` resolves the checkpointer from `LANGGRAPH_DATABASE_URL`: `AsyncPostgresSaver` for `postgresql[+driver]://…`, `AsyncSqliteSaver` for SQLite (the dev default `data/agent_threads.db`). Every streaming run passes `thread_config(thread_id)`; the WebUI uses `str(lesson.id)` so follow-up turns share state. For long threads, the service applies **summary-buffer windowing** (`AIX_CONVERSATION_WINDOW_TURNS`, default 4): the most recent turns are kept verbatim and older turns are LLM-summarized into `conversation_summary`. The WebUI also persists a SQL transcript (`lesson_message` rows) as a dialect-agnostic source of truth that survives a checkpointer wipe.

### 6.9 Prompt system

`agent/prompts/` holds intent-specific prompt builders (planner/writer/critic); `agent/configs/domain_prompts.py` holds domain-specific extensions (UDL vs Neuro tone, terminology, and constraints). Reference prompt texts are mirrored under `docs/prompts_reference/`. This separation lets the same agent code specialize per domain without branching logic.

---

## 7. Retrieval Layer

### 7.1 GraphRAG approach

Retrieval is **graph-first**: the system grounds answers in a curated Neo4j Knowledge Graph rather than relying solely on a vector store. A teacher query is converted to Cypher, executed against Neo4j, optionally expanded via semantic similarity, and structured into an educational context object the agents consume. This is what makes outputs *evidence-based* and auditable.

### 7.2 Text2Cypher (multilingual)

`retrieval/text2cypher.py` converts natural language to Cypher; `retrieval/multilingual_text2cypher.py` adds Italian→English translation so Italian queries match the (partly English) graph vocabulary. Generated Cypher is optionally validated and self-repaired before execution. Tuning: `TEXT2CYPHER_MODEL` (a fast/cheap model, default `google/gemini-2.5-flash`), `TEXT2CYPHER_MAX_QUERY_LENGTH`, `TEXT2CYPHER_DEFAULT_LIMIT`, `TEXT2CYPHER_ENABLE_VALIDATION`, `TEXT2CYPHER_ENABLE_EXECUTION` (set false for dry-run).

### 7.3 Hybrid graph retriever (Neo4j + Node2Vec / embeddings)

`retrieval/graph_retriever.py` combines three signals:

1. **Direct graph traversal** — the generated Cypher.
2. **Semantic search** — embedding similarity over graph nodes.
3. **Neighbor expansion** — pulling related concepts for completeness.

Embedding modes (`EMBEDDING_MODE`):

| Mode | Behavior |
|---|---|
| `node2vec` | Pre-trained graph embeddings only; fast, no API call; language-blind |
| `hybrid_semantic` | `EMBEDDING_NODE2VEC_WEIGHT` (default 0.4) Node2Vec + (0.6) text embeddings; **recommended for production** (handles Italian, synonyms, paraphrase) |
| `openai_only` | Pure semantic embeddings (use when a domain has no Node2Vec model) |

Relevant settings: `EMBEDDING_MODEL` (default `openai/text-embedding-3-small`, 1536-dim), `EMBEDDING_SEMANTIC_THRESHOLD` (default 0.7), `EMBEDDINGS_CACHE_DIR`, `NODE2VEC_MODEL_DIR`. **Warning:** changing `EMBEDDING_MODEL` requires deleting `artifacts/embeddings_cache/` so node and query vectors stay dimension-compatible.

### 7.4 Context builder

`retrieval/context_builder.py` turns raw graph results into a structured educational context: methodology recommendations with confidence levels, concept groupings, and a student-profile view. This structured object — not raw rows — is what the agents reason over, keeping prompts compact and consistent.

### 7.5 Embeddings & artifacts

Pre-trained artifacts ship in the repo and are mounted as a Docker volume in production:

- `artifacts/node2vec/{domain}_node2vec_embeddings.npz` — graph embeddings (128-dim; walk length 80, 200 walks).
- `artifacts/embeddings_cache/{domain}_openai_embeddings.json` — cached text embeddings.

Retraining: `python scripts/ml/train_node2vec.py {neuro|udl}`. First-time hybrid setup precomputes text embeddings via `python -m aix.retrieval.graph_retriever --precompute {domain}`.

### 7.6 External sources (hybrid / out-of-scope)

When the Planner marks a topic `partial_scope`/`out_of_scope`, the Retriever supplements the KG with verified external sources via `agent/media/` (and `agent/tools/`): Wikipedia, OpenAlex/Semantic Scholar (academic citations), OER repositories, and YouTube (via `yt-dlp` + oEmbed, no API quota). `SEMANTIC_SCHOLAR_API_KEY` is optional and only raises rate limits. These sources fill the `external_resources` / media buckets so the lesson remains useful even outside the KG's core coverage, while the UI clearly signals reduced KG anchoring (coverage tiers).

### 7.7 Knowledge Graph schema (UDL & Neuro domains)

Two domains are served from the same Neo4j instance:

- **`neuro`** — neuroscience-of-learning concepts, methodologies, and strategies.
- **`udl`** — Universal Design for Learning (inclusive pedagogy).

Nodes represent educational concepts/methodologies/strategies; relationships encode pedagogical links (e.g. `BELONGS_TO`, `ADDRESSES`, `SUPPORTS`). Domain selection is per-request (`domain` field). Source dumps and the contract format live under `data/kg/{neuro,udl}/` and `data/reference/JSON_reference.json`; ingestion is handled by `scripts/ingest/data_ingestion_neo4j.py`.

## 8. Configuration & Environment

### 8.1 Configuration model

Configuration is environment-driven. `src/aix/core/config.py` loads variables via `python-dotenv` (`load_dotenv()`) and exposes typed config objects (`Neo4jConfig`, `OpenAIConfig`, …) plus a singleton `config`. There is **no** secret in source control: `.env` (dev) and `deploy/.env.prod` (prod) are git-ignored. Templates are committed: `.env.example` (full dev surface) and `deploy/.env.prod.example` (production subset).

The `OpenAIConfig` is provider-agnostic (OpenRouter or OpenAI) and detects reasoning/"thinking" models (Claude Sonnet/Opus 4.x, o-series, DeepSeek R1) to adjust API parameters automatically (`max_completion_tokens` vs `max_tokens`, dropping `temperature`, requesting reasoning tokens). This is why switching models rarely requires code changes.

### 8.2 Environment variables reference

**Core services**

| Variable | Default | Purpose |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection (`bolt+s://` for TLS/Aura) |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / — | Neo4j credentials |
| `NEO4J_ENCRYPTED` | `1` | TLS toggle |
| `OPENROUTER_API_KEY` | — | LLM provider key (preferred) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | LLM endpoint |
| `OPENAI_API_KEY` | — | Fallback if OpenRouter unset |
| `LLM_MODEL` | `anthropic/claude-sonnet-4-6` | Primary lesson-generation model |

**Retrieval / embeddings**

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODE` | `hybrid_semantic` | `node2vec` / `hybrid_semantic` / `openai_only` |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Text embedding model (cache-coupled) |
| `EMBEDDING_NODE2VEC_WEIGHT` | `0.4` | Hybrid weight (Node2Vec share) |
| `EMBEDDING_SEMANTIC_THRESHOLD` | `0.7` | Min semantic similarity |
| `NODE2VEC_MODEL_DIR` | `./artifacts/node2vec` | Node2Vec artifacts |
| `TEXT2CYPHER_MODEL` | `google/gemini-2.5-flash` | Cypher generation + translation |

**WebUI / auth / persistence**

| Variable | Default | Purpose |
|---|---|---|
| `WEBUI_AUTH_SECRET` | dev fallback (warns) | HS256 signing secret (cookie + Bearer). **Must** be random in prod |
| `WEBUI_DATABASE_URL` | SQLite `data/webui/webui.db` | Users/lessons/messages store |
| `WEBUI_TOKEN_LIFETIME_SECONDS` | `86400` | Session lifetime |
| `WEBUI_COOKIE_SECURE` | `0` | Set `1` behind HTTPS |
| `WEBUI_CORS_ALLOW_ORIGINS` | — | Allowed origins (Mode B / cross-origin) |
| `LANGGRAPH_DATABASE_URL` | SQLite `data/agent_threads.db` | LangGraph checkpointer store |

**Agent tuning flags** (all safe-defaulted via `os.getenv` — see `.env.example` for the annotated list)

| Variable | Default | Purpose |
|---|---|---|
| `AIX_MAX_REVISIONS` | `1` | Writer→Critic revision cycles (0–4) |
| `AIX_WRITER_MAX_TOKENS` | `3500` | Writer output ceiling |
| `AIX_WRITER_MAX_CONTINUATIONS` | `1` | Auto-continue on length cutoff |
| `AIX_THINKING_EFFORT` | `low` | Reasoning-token budget (`low`/`medium`/`high`) |
| `AIX_CRITIC_MODEL` | (→ `TEXT2CYPHER_MODEL`) | Critic model (fast/cheap) |
| `AIX_CRITIC_PARSE_ERROR_BEHAVIOR` | `approve` | On unparseable critic JSON (`approve`/`revise`/`raise`) |
| `AIX_CORRECTIVE_RAG_ENABLED` | `false` | Enable retrieval-grading loop |
| `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS` | `2` | Retry budget when CR on (1–4) |
| `AIX_CONVERSATION_WINDOW_TURNS` | `4` | Verbatim turns before summary buffering |
| `AIX_LLM_PROBE_ENABLED` | `true` | Startup LLM connectivity probe |

**Observability / ops**

| Variable | Default | Purpose |
|---|---|---|
| `SENTRY_DSN` | — | GlitchTip/Sentry DSN (empty = disabled) |
| `ENVIRONMENT` | `production` | Issue label (`production`/`staging`/`development`) |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | — | LLM tracing |

**Deployment (Caddy / Postgres / TLS)**

| Variable | Purpose |
|---|---|
| `AIX_DOMAIN` | Public hostname Caddy serves (e.g. `agente.aiforlearning.digital`) |
| `AIX_TLS_EMAIL` | Let's Encrypt registration/expiry email |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Postgres credentials (chosen by us; compose creates the DB on first boot) |
| `GIT_SHA` | Build arg stamped into `CODE_VERSION` for traceability |

### 8.3 Secrets management

- Secrets live only in `.env` (dev) / `deploy/.env.prod` (prod), never in git. On the VM, `chmod 600 deploy/.env.prod`.
- `WEBUI_AUTH_SECRET` is generated with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. If unset, a dev-only fallback is used and a warning is logged — acceptable locally, unacceptable in production.
- `POSTGRES_*` are operator-chosen (not provided by a third party); `docker-compose.prod.yml` derives both `WEBUI_DATABASE_URL` and `LANGGRAPH_DATABASE_URL` from them so there is a single source of truth.
- LLM and Neo4j keys are provided by their respective providers (OpenRouter dashboard, Neo4j Aura/FEM).

### 8.4 Configuration profiles (dev / staging / prod)

- **Dev** — SQLite everywhere, HTTP localhost, `WEBUI_COOKIE_SECURE=0`, optional observability, `.env`.
- **Staging** — Postgres, HTTPS via Caddy, `ENVIRONMENT=staging`, observability on, stricter critic behavior (`AIX_CRITIC_PARSE_ERROR_BEHAVIOR=raise`) for investigation.
- **Prod** — Postgres, HTTPS, `WEBUI_COOKIE_SECURE=1`, random `WEBUI_AUTH_SECRET`, `LOG_LEVEL=INFO`, observability on, `deploy/.env.prod`.

---

## 9. Local Development

### 9.1 Prerequisites

- **Python 3.11** recommended for local dev (project supports ≥3.10; production Docker pins 3.12).
- A reachable **Neo4j** (local or the shared Aura/FEM instance).
- An **OpenRouter** (or OpenAI) API key.
- Git, and on Windows a PowerShell shell (the repo is developed cross-platform).

### 9.2 Installation

```bash
git clone <repo-url> && cd graphaixlearning
python -m venv venv && . venv/Scripts/activate      # PowerShell: venv\Scripts\Activate.ps1
pip install -e ".[dev]"                              # editable install + dev extras (pytest, ruff, mypy)
cp .env.example .env                                 # then fill credentials
```

`pip install -e .` registers the `aix` package so `aix.*` imports resolve from any working directory. The `[dev]` extra adds the test/lint toolchain used by `make test` / `make lint` / CI.

### 9.3 Running locally

The canonical command for end-to-end local testing (serves every surface on one port):

```bash
python -m uvicorn aix.api.main:app --host 127.0.0.1 --port 8765 --log-level info
```

This single process serves `/docs`, `/webui/`, `/api/v1/context`, `/api/v1/agent/run`, `/api/v1/agent/stream`, `/mcp/`, and `/auth/jwt/login`. Add `--reload` for autoreload during development. Other entry points:

```bash
python apps/cli/run_agent.py                  # interactive agent CLI (or: make agent)
python apps/cli/run_agent.py --query "Crea una lezione sulla memoria" --domain neuro
streamlit run apps/streamlit/main.py          # legacy demo (retirement banner)
python -m aix.mcp.stdio_main                  # MCP over stdio (Claude Desktop / Cursor)
```

### 9.4 Knowledge Graph setup

Pre-baked KG dumps ship in the repo, so no data prep is needed for the default setup:

```bash
python scripts/ingest/data_ingestion_neo4j.py \
    --file data/kg/neuro/kg_neuro_neo4j.json \
    --password YOUR_NEO4J_PASSWORD --clear        # swap in kg_udl_neo4j.json for UDL
```

Node2Vec artifacts also ship pre-trained; retrain only if you change the graph: `python scripts/ml/train_node2vec.py {neuro|udl}`.

### 9.5 Common development workflows

- **Add/refresh dependencies** — edit `requirements.txt` (human source of truth), then regenerate the hashed lockfile for the production target: `uv pip compile requirements.txt -o requirements.lock.txt` (Python 3.12 / Linux resolution). Commit both.
- **Lint & format** — `ruff format .` then `ruff check . --fix` (see §10.4). Run via the venv: `python -m ruff …`.
- **Type-check** — `mypy` (lenient by default; configured in `pyproject.toml`).
- **Smoke-test MCP** — `python scripts/diagnostic/mcp_smoke.py` (in-process, no uvicorn).
- **Sync with remote** — `git fetch`, inspect with `git log HEAD..origin/<branch>`, then `git pull --ff-only`; stash local edits with `git stash push -u` if needed.

### 9.6 IDE setup notes

- The `aix` package resolves only after `pip install -e .`; point your interpreter at the project venv.
- Ruff is the single formatter/linter (Black-compatible, line length 100, double quotes). Configure your editor's "format on save" to use Ruff to avoid diff churn.
- On Windows, when Postgres is configured the app applies a selector-event-loop policy at import; this is a no-op on Linux and for SQLite dev.

---

## 10. Testing Strategy

### 10.1 Test layout & markers

Tests live under `tests/` and are configured in `pyproject.toml`:

- `tests/unit/` — pure unit tests, no external services (`@pytest.mark.unit`).
- `tests/integration/` — hit Neo4j / LLM APIs (`@pytest.mark.integration`).
- `tests/api/` — Agent API contract tests (request/response + SSE taxonomy round-trips).
- `tests/mcp_server/` — MCP regression suite (19 tests).

Pytest defaults: `-v --tb=short --strict-markers`, `asyncio_mode = "auto"` (async tests need no explicit decorator). Markers: `integration`, `slow`, `unit`.

### 10.2 Running tests

```bash
pytest tests/ -v                       # all suites (or: make test)
pytest tests/integration/ -v -m integration   # only integration (needs Neo4j / keys)
pytest tests/mcp_server/ -v            # MCP regression suite
pytest tests/api/ -v                   # agent API contract
```

### 10.3 What is covered vs. what to validate manually

- **Well covered**: API contract (schemas, status codes, SSE `kind` taxonomy), MCP tools/resources/prompts, agent routing logic (revision loop, corrective-RAG routing), retrieval helpers.
- **Validate manually / end-to-end** (no full automated coverage yet): the media pipeline enrichment, the upsell flow, and full multi-turn WebUI runs — exercise via `apps/cli/run_agent.py` and the WebUI before a release. See §17 for the known-gaps backlog.

### 10.4 Linting & formatting (Ruff)

Ruff is the single tool for both **linting** (replacing flake8/isort/bugbear/pyupgrade) and **formatting** (Black-compatible). Config in `pyproject.toml`: line length 100, target `py310`, double-quote style, rule sets `E/F/W/I/B/UP`. `E501` (line length) is delegated to the formatter; `B008` (FastAPI `Depends()`) and `UP007` are intentionally ignored. Per-directory ignores relax rules for `tests/`, `apps/`, and `scripts/`.

```bash
python -m ruff format .            # apply formatting
python -m ruff format --check .    # verify formatting (CI)
python -m ruff check . --fix       # lint + safe autofixes
python -m ruff check . --statistics
```

Formatting and import-sorting fixes are behavior-preserving; only review autofixes from rule families that can change semantics (e.g. some `B`/`UP` rewrites) before committing.

### 10.5 CI expectations

CI is expected to run `ruff format --check .`, `ruff check .`, and the non-integration test suite on each push/PR. Integration tests that require live Neo4j/LLM keys are run selectively (they are marked and can be excluded in environments without credentials). The OpenAPI baseline under `data/diagnostic/` serves as a regression guard against unintended contract changes.

---

## 11. Deployment

### 11.1 Container image

`Dockerfile` is a multi-stage build targeting `api`, hardened for production:

- Base `python:3.12-slim-bookworm` with the `it_IT.UTF-8` locale installed (Italian output correctness).
- Dependencies installed from the **hashed lockfile** with `uv pip install --require-hashes -r requirements.lock.txt` (reproducible builds), then `pip install --no-deps -e .`.
- Runs as a **non-root** user (`aix`, uid/gid 10001); `artifacts/` and `data/` are writable for caches.
- Exposes `8765`; `PORT` env drives the bind. `CODE_VERSION` is stamped from the `GIT_SHA` build arg.
- Built-in `HEALTHCHECK` curls `/api/v1/health` (interval 30 s, 45 s start period).

Local-equivalent run command (outside Docker): `python -m uvicorn aix.api.main:app --host 127.0.0.1 --port 8765`.

### 11.2 Production stack (`deploy/docker-compose.prod.yml`)

Three services on a single host:

| Service | Image | Exposure | Role |
|---|---|---|---|
| `app` | built from `Dockerfile` (`api`) | internal only | FastAPI — `/api/v1/*` + `/webui/*` |
| `postgres` | `postgres:16-alpine` | internal only | Backs both DB URLs (webui + checkpointer) |
| `caddy` | `caddy:2-alpine` | `80`, `443`, `443/udp` | Reverse proxy + auto-HTTPS |

Key properties:

- **Neo4j is not in the compose** — production reuses the external Aura/FEM instance via `NEO4J_URI`.
- Both DB URLs are composed from the same `POSTGRES_*` credentials in the compose `environment` block — one source of truth. The webui uses the `asyncpg` driver (SQLAlchemy); LangGraph's `AsyncPostgresSaver` uses the libpq scheme directly.
- `app` waits for `postgres` to be healthy (`depends_on … condition: service_healthy`); tables auto-create on first boot.
- Volumes: `pg_data` (DB), `app_artifacts` (embeddings/Node2Vec — survives rebuilds), `caddy_data`/`caddy_config` (ACME certs + state).
- Run: `docker compose -f docker-compose.prod.yml --env-file .env.prod up -d`.

### 11.3 Reverse proxy & TLS (Caddy)

`deploy/Caddyfile` is parameterized by `$AIX_DOMAIN` and `$AIX_TLS_EMAIL`. Caddy terminates TLS (Let's Encrypt, auto-renew), serves HTTP/3 (QUIC) on UDP/443, and reverse-proxies all traffic to `app:8765`. SSE endpoints work transparently (Caddy streams responses; the app's 15 s heartbeat keeps idle connections open). `AIX_TLS_EMAIL` is the ACME registration/expiry-notice address — required for unattended certificate issuance.

### 11.4 Database & checkpointer migration to PostgreSQL

Wave 1 migrates both stateful stores from SQLite (dev) to PostgreSQL (prod):

- **WebUI DB** (`WEBUI_DATABASE_URL`) — users, lessons, lesson messages.
- **LangGraph checkpointer** (`LANGGRAPH_DATABASE_URL`) — `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`.

Both point at the same Postgres instance (separate logical concerns, distinct table sets). No manual schema step is required for the pilot: tables auto-create on first boot (`init_db()` for the webui; the saver `setup()` for the checkpointer). Backups are handled by `deploy/scripts/backup_postgres.*`.

### 11.5 Deployment modes (Mode A standalone / Mode B native)

- **Mode A (standalone pilot)** — the compose stack above; teachers use `/webui/*` at `https://agente.aiforlearning.digital`. This is the current Wave 1/2 target.
- **Mode B (native AixLearning)** — the agent service runs as an internal service; the AixLearning Django backend calls `/api/v1/agent/*` over the private network and owns its own UX/data. Mode B does not require Caddy/WebUI exposure. See §14.

The hostname strategy (transition from the legacy `graph.aiforlearning.digital` and the relationship between API access and WebUI access) is tracked in `docs/product/Internal_Production_Deployment_Plan.md`.

### 11.6 CI/CD pipeline

The FEM environment already runs continuous deployment: a merge commit pushed to the GitHub repository triggers a build + redeploy of the GraphRAG instance on the managed VM (Debian; networking, Docker, and CD pre-configured per FEM's template). Practically, shipping a new version is a **merge to the deployment branch**. The build uses the committed `requirements.lock.txt` for reproducibility; `GIT_SHA` is stamped into `CODE_VERSION` for traceability across logs and observability tools.

### 11.7 Rollback & recovery

- **Rollback** — redeploy a previous image/commit (CD), or `docker compose up -d` against the prior tag; `CODE_VERSION` identifies the running build.
- **Data recovery** — restore Postgres from `deploy/scripts/restore_postgres.*`; Caddy certs/state from `backup_caddy`. Artifacts in `app_artifacts` are regenerable (re-embed) if lost.
- **Failure isolation** — the MCP sub-app is built defensively so a failure there cannot block `/api/v1`; the agent stream surfaces pipeline failures as `error`/`502` rather than crashing the process. The runbook in `deploy/README.md` covers first deploy, backups, rollback, and log inspection.

## 12. Observability

### 12.1 LLM tracing (Langfuse)

When `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (+ optional `LANGFUSE_HOST`) are set, agent LLM calls are traced to Langfuse, giving per-agent and full-pipeline visibility: prompts, model, token usage, latency, and the planner/retriever/critic outputs. The `session_id` carried through `AgentState` (and accepted by the API) is the correlation key for stitching a multi-step run together. Tracing is **opt-in** — with the keys unset, the system runs identically without tracing.

### 12.2 Error monitoring (GlitchTip / Sentry)

When `SENTRY_DSN` is set, the app initializes the Sentry SDK at startup (GlitchTip is Sentry-compatible). Unhandled exceptions and slow requests are reported, labeled by `ENVIRONMENT` (`production`/`staging`/`development`). Performance tracing is sampled at 20% (`traces_sample_rate=0.2`, set in `api/main.py`) — enough to surface slow endpoints without overwhelming the dashboard. Leave `SENTRY_DSN` empty to disable entirely.

### 12.3 Health checks

`GET /api/v1/health` returns a `HealthResponse` reporting `status` (`healthy`/`degraded`), `neo4j_connected`, and `version` (the package version; the build also stamps `CODE_VERSION` from `GIT_SHA`). It verifies Neo4j connectivity and domain configs, making it suitable as both a container `HEALTHCHECK` (Dockerfile + compose) and an external uptime probe. A `degraded` status (Neo4j unreachable) still returns HTTP 200 with the flag set, so orchestrators can distinguish "process up but dependency down" from "process down".

### 12.4 Connectivity probe

At startup (`AIX_LLM_PROBE_ENABLED=true`, default), `src/aix/core/connectivity_probe.py` issues a one-shot `GET /models` against the configured LLM `base_url` and emits **one** actionable log line that distinguishes TLS certificate failures, DNS/connection errors, read timeouts, 401/403 auth failures, and 4xx/5xx upstream errors. This resolves the common "Connection error." ambiguity that otherwise hides three different failure modes inside the OpenAI SDK retry layer. Disable in air-gapped/test environments.

### 12.5 Structured logging

Logging verbosity is set by `LOG_LEVEL` (default `INFO`). Log lines use consistent, greppable prefixes per subsystem (e.g. `[api.agent]`, `[webui.agent]`, `[LessonPlannerGraph]`, `[language]`) and include the `session_id`/`thread_id`, duration, approval, and revision count on run completion. Use `DEBUG` locally to surface Cypher queries, embedding scores, and reasoning tokens; keep `INFO`/`WARNING` in production.

---

## 13. Security

### 13.1 Authentication

User-facing and API auth is handled by **FastAPI-Users**. Two backends share `WEBUI_AUTH_SECRET` (HS256): a **cookie** backend (browser/WebUI sessions) and a **Bearer JWT** backend (API/integration clients). The agent endpoints accept either via the `current_active_user` dependency. Token lifetime is `WEBUI_TOKEN_LIFETIME_SECONDS` (default 24 h). The legacy `/api/v1/context` endpoint uses a separate HTTP Basic service-to-service scheme for the existing AixLearning integration.

### 13.2 Authorization

For the pilot, authorization is coarse-grained: any active, authenticated user may call the agent endpoints. There is no per-tenant or per-role gating yet — multi-tenant authorization (and the RS256/multi-issuer JWT work in §17.3) is deferred. WebUI users only see and operate on their own lessons (ownership is enforced at the data layer).

### 13.3 CORS policy

CORS is configured via `WEBUI_CORS_ALLOW_ORIGINS` (`api/main.py`). Default is `*` (convenient for local dev and the same-origin pilot); in production with cross-origin clients (Mode B browser callers) it should be set to a comma-separated allow-list. `allow_credentials=True` is set so cookie auth works cross-origin when an explicit origin list is provided.

### 13.4 Secrets handling

See §8.3. In summary: secrets are environment-only (`.env` / `deploy/.env.prod`, git-ignored, `chmod 600` on the VM); `WEBUI_AUTH_SECRET` must be a strong random value in any non-local environment; database URLs are derived from a single set of `POSTGRES_*` credentials; LLM/Neo4j keys come from their providers.

### 13.5 Rate limiting

There is no application-level rate limiter in the pilot. The natural throttle is the LLM provider's own rate limits (OpenRouter), and Caddy can enforce connection-level limits at the edge if required. Application-level per-user rate limiting is a candidate for a later wave if abuse becomes a concern.

### 13.6 Network isolation

In production only Caddy is published (host ports 80/443). The `app` and `postgres` services have **no** `ports:` mapping — they are reachable only on the internal Docker network. Postgres is therefore never exposed to the host or the internet; the app is reachable only through Caddy (TLS-terminated). Neo4j is an external managed instance reached over `bolt+s://` (TLS). The MCP sub-app mount is guarded so a failure there cannot affect `/api/v1`.

### 13.7 AI transparency & EU AI Act alignment

The system is designed to make AI involvement explicit, in line with the project's regulatory analysis (`docs/product/Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md`):

- **In-product transparency (implemented):** the WebUI shows per-phase explainability (planner intent/scope, retriever coverage tiers, critic score) and an explicit banner when a lesson is composed with reduced KG coverage or from external sources ("La lezione è generata con queste limitazioni…"). Outputs are always presented as AI-generated drafts for a human educator to review.
- **Human-in-the-loop:** the teacher is the decision-maker; the system produces drafts, not authoritative content. The Critic loop and coverage signals support, but do not replace, human judgment.
- **Machine-readable AI marking (planned):** a response-level marker (e.g. an `X-AI-Generated` header and/or an embedded Markdown comment on exported lesson plans) is tracked as a compliance enhancement so downstream consumers and exported artifacts can be programmatically identified as AI-assisted. This is not yet enforced in code and should be added before any external/public-facing rollout.

Refer to the regulatory document for the full mapping to EU AI Act and UNI/PdR obligations; this section captures only the technical controls.

---

## 14. Integration Patterns

### 14.1 Mode A — Standalone WebUI

The agent service runs the full stack (§11.2) and exposes the teacher WebUI at `https://<AIX_DOMAIN>/webui/`. Teachers authenticate (cookie session), create lessons with an educational profile, optionally upload context files, and watch the agent stream its phases live. This is the **internal FEM pilot** mode and requires no work from the AixLearning DEV team beyond infrastructure (VM + DNS + CD).

### 14.2 Mode B — Native AixLearning integration

The AixLearning Django platform integrates the agent service the same way it already integrated the legacy GraphRAG `/api/v1/context` endpoint:

1. The teacher uses AixLearning's existing UI to request a lesson.
2. AixLearning's **Dramatiq worker** detects a UDL/NEURO `plan_type` and routes the request to a new **`AgenticGraphRagService`** — a sibling of the existing `GraphRagService` that already calls `/api/v1/context`.
3. `AgenticGraphRagService` calls `POST /api/v1/agent/run` (or `/stream`) over the internal Docker network with a service Bearer token, passing the teacher's query + educational profile.
4. AixLearning persists and renders the returned `lesson_plan_md` in its own UI.

In this mode the agent service owns only its own state (KG, checkpointer, lessons store if used); AixLearning owns the teacher UX and its own data. Caddy/WebUI exposure is optional.

### 14.3 Mode coexistence rules

- The two modes are **not mutually exclusive** — the same running instance can serve `/webui/*` (Mode A) and `/api/v1/agent/*` to AixLearning (Mode B) simultaneously.
- The agent endpoints are **additive and backward-compatible**: the API contract is locked by an automated regression test against an OpenAPI baseline (`data/diagnostic/`), and new fields are always added, never removed or repurposed.
- Authentication differs by caller: browser cookie (Mode A) vs service Bearer token (Mode B). Both resolve to the same `current_active_user` dependency.

### 14.4 Reference wrapper (`AgenticGraphRagService` pattern)

The recommended integration shape on the AixLearning side is a thin service class mirroring the existing `GraphRagService`:

```python
class AgenticGraphRagService:
    def generate_lesson(self, query: str, domain: str, profile: dict) -> str:
        resp = http.post(
            f"{AGENT_BASE_URL}/api/v1/agent/run",
            headers={"Authorization": f"Bearer {SERVICE_JWT}"},
            json={"query": query, "domain": domain, "educational_profile": profile},
            timeout=240,  # the sync run can take 60–120s; allow headroom
        )
        resp.raise_for_status()
        return resp.json()["lesson_plan_md"]
```

For incremental UX, swap `/run` for `/stream` and consume SSE frames, switching on `kind`. The full contract, payload examples, and SSE taxonomy the DEV team needs are in `docs/product/Dev_Handoff_AgenticGraphRAG_Integration.md` and `docs/product/Dev_Technical_Integration_Guide.md`.

---

## 15. Performance & SLOs

### 15.1 Latency budget per phase

A typical KG-covered run (default config, 0–1 revisions) distributes roughly as:

| Phase | Typical time | Notes |
|---|---|---|
| Planner | ~2–5 s | Intent + scope + query extraction |
| Retriever | ~3–8 s | Cypher + hybrid search (+ external sources if out-of-scope) |
| Writer | ~25–40 s | Dominant cost; bounded by `AIX_WRITER_MAX_TOKENS` (+ continuations) |
| Critic | ~2–5 s | Fast model (`AIX_CRITIC_MODEL`), capped prefill |

The Writer is the long pole; thinking-effort and token caps (`AIX_THINKING_EFFORT`, `AIX_WRITER_MAX_TOKENS`) are the primary latency levers.

### 15.2 Streaming first-event target

For `/stream`, the first `planner` event should reach the client within a few seconds (target **< 5 s**), giving immediate UI feedback while the slow Writer runs. The 15 s SSE heartbeat prevents proxy timeouts during the Writer call.

### 15.3 End-to-end SLO targets (pilot)

- Full pipeline (sync `/run`): **< 180 s** for KG-covered topics; **< 240 s** for out-of-scope topics that trigger external retrieval. Integration clients should set timeouts with headroom (≈240 s).
- Availability target for the pilot: **~99.5%** (single-host stack; not HA).

### 15.4 Cost & capacity assumptions

- Cost per interaction is dominated by the Writer LLM call (and thinking tokens). The Critic and Text2Cypher use cheaper fast models to keep per-run cost low.
- The pilot is sized for a single host (2 vCPU / 4 GB RAM / ~50 GB disk, Debian) with the agent graph/agents as process-level singletons and one in-flight run per lesson. This is adequate for internal pilot concurrency; horizontal scaling and a shared-state refactor are deferred (§17). Embedding caches persist in a volume so restarts/rebuilds avoid re-embedding.

---

## 16. Operational Runbook

### 16.1 Common incidents

| Symptom | Likely cause | First action |
|---|---|---|
| `/api/v1/health` reports `degraded` | Neo4j unreachable / bad `NEO4J_*` | Check `NEO4J_URI` + network to Aura/FEM; inspect app logs |
| Runs fail with `502` / `error` events | LLM provider down / bad key / rate limit | Check connectivity-probe log line; verify `OPENROUTER_API_KEY` |
| "Connection error." at startup | TLS/DNS/auth to LLM endpoint | Read the single probe log line (it names the exact failure mode) |
| Slow first event on `/stream` | Cold start / Neo4j latency | Confirm warm process; check Planner/Retriever timings in logs |
| WebUI login fails after deploy | `WEBUI_AUTH_SECRET` changed/missing | Restore the secret; rotating it invalidates existing sessions |
| Lesson "frozen" in WebUI | Setup exception swallowed into `error` | Check `[webui.agent] setup FAILED` logs; lesson is marked `error` |

### 16.2 Debugging guide

- Set `LOG_LEVEL=DEBUG` to see Cypher, embedding scores, and reasoning tokens.
- Trace a specific run by its `session_id`/`thread_id` across logs and Langfuse.
- Reproduce a pipeline run without the WebUI/DB via `apps/cli/run_agent.py --query "…" --domain neuro`.
- Smoke-test MCP in-process with `python scripts/diagnostic/mcp_smoke.py`.
- Verify the API contract hasn't drifted against the OpenAPI baseline in `data/diagnostic/`.

### 16.3 Restart / redeploy

- **Redeploy (CD):** push a merge commit to the deployment branch — the FEM CD pipeline rebuilds and restarts the stack.
- **Manual restart:** `docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d` (recreates changed services; volumes persist).
- **Single service:** `docker compose … restart app` (or `caddy`). Confirm health via `/api/v1/health` and compose health status.

### 16.4 Database maintenance

- **Backups:** `deploy/scripts/backup_postgres.*` (DB) and `backup_caddy` (certs/state) on a schedule; store off-host.
- **Restore:** `deploy/scripts/restore_postgres.*` into a fresh `postgres` volume; restart `app`.
- **Schema:** tables auto-create on first boot (webui `init_db()`, checkpointer `setup()`). The lesson-message transcript is the durable source of truth and survives a checkpointer wipe; artifacts in the `app_artifacts` volume are regenerable.

---

## 17. Roadmap & Known Limitations

### 17.1 Deferred items

Items intentionally out of scope for the current pilot (tracked in `docs/product/ClickUp_Agentic_GraphRAG_Update.md`):

- Lesson library + history + PDF export; Italian copy polish, accessibility, mobile breakpoints, Tailwind CLI build.
- Full automated end-to-end coverage of the media pipeline and the upsell flow (currently validate manually — §10.3).
- UDL content parity (media-mapping enrichment) and enriched UDL Critic criteria; Critic integration with domain-configs mirroring the Writer's pattern.
- Application-level rate limiting and per-tenant authorization.
- Decision on `lesson_template.txt` (wire into the Writer or remove).

### 17.2 Concurrency & scaling

The compiled graph and agents are module-level singletons and the WebUI enforces one in-flight run per lesson. For higher concurrency, the streaming engine should be consolidated into a single `aix.agent.streaming` module and the singleton assumption revisited (worker pool / per-request graph). Tracked as a deploy follow-up.

### 17.3 RS256 / multi-issuer JWT migration

V1 uses HS256 with a shared `WEBUI_AUTH_SECRET`. A migration to **RS256** with multi-issuer support is planned for multi-tenant / multi-issuer scenarios (e.g. AixLearning minting its own tokens), but is **not required for V1**.

### 17.4 Frontend evolution

The standalone WebUI (Jinja2 + htmx + WebAwesome + Tailwind + Alpine.js + SSE) is intentionally lightweight and server-driven, which suits the agentic, streaming UX. A deeper analysis of frontend options for an agentic architecture (including comparison with the native AixLearning Django/Mercure stack) is captured in `docs/product/WebUI_Frontend_Comparison_Agentic_UX.md`.

### 17.5 Other known limitations

- Single-host, non-HA pilot (availability ~99.5%, §15.3).
- Coarse authorization (§13.2) and no app-level rate limiting (§13.5).
- Multilingual generation is tuned for Italian (primary) with English support; other languages are detected but not first-class.

---

## 18. Appendices

### Appendix A — Environment variables reference (full table)

The full, annotated list lives in `.env.example` (development surface) and `deploy/.env.prod.example` (production subset). The categorized summary tables in §8.2 are the canonical in-document reference; treat the `*.example` files as the authoritative source for defaults and inline guidance, since they are versioned alongside the code.

### Appendix B — Glossary

Extends the §1.5 glossary:

- **Node2Vec** — graph-embedding algorithm producing structural vectors for KG nodes.
- **Text2Cypher** — the NL→Cypher conversion layer (with IT/EN translation).
- **Corrective RAG** — optional retrieval-grading loop that retries retrieval when grounding is weak.
- **Educational profile** — structured class/classroom/time/subject context attached to a request.
- **Coverage tier** — UI signal (`healthy`/`limited`/`out_of_scope`) derived from KG node count.
- **Checkpointer** — LangGraph component persisting per-thread agent state (SQLite dev / Postgres prod).
- **Thread / `thread_id`** — the conversation key enabling multi-turn memory.
- **MCP** — Model Context Protocol; exposes tools/resources/prompts to MCP clients.
- **Mode A / Mode B** — standalone WebUI vs native AixLearning integration.
- **Langfuse / GlitchTip** — tracing and error-monitoring backends.

### Appendix C — Cross-references to internal documents

| Document | Purpose |
|---|---|
| `docs/release/Functional_Documentation.md` | Product/pedagogical companion to this technical reference |
| `docs/product/Dev_Handoff_AgenticGraphRAG_Integration.md` | DEV integration handoff (Mode B contract, examples) |
| `docs/product/Dev_Technical_Integration_Guide.md` | Detailed API/integration technical guide |
| `docs/product/Internal_Production_Deployment_Plan.md` | Wave 1/2 deployment plan, timing, ownership, hostnames |
| `docs/product/Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md` | EU AI Act / UNI alignment |
| `docs/product/WebUI_Frontend_Comparison_Agentic_UX.md` | Frontend/UX analysis for agentic architecture |
| `docs/product/ClickUp_Agentic_GraphRAG_Update.md` | Backlog / deferred-item tracker |
| `deploy/README.md` | First-deploy, backup, rollback, log-inspection runbook |
| `README.md` | Repository quick start and feature overview |

### Appendix D — Changelog

| Version | Date | Notes |
|---|---|---|
| 0.1 | May 2026 | Initial structure + TOC + §1 Introduction |
| 1.0 | May 2026 | First complete draft: §2–§18 written and grounded in the codebase |

---

*End of document.*
