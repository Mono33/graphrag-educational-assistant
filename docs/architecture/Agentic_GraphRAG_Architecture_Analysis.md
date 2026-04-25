# Agentic GraphRAG: Architecture Analysis, Best Practices & Roadmap

**Author:** Louis Mono — AI Team Lead  
**Date:** January 27, 2026  
**Project:** AIxLearning — Agentic GraphRAG Pipeline  
**Status:** Architecture Review & Strategic Planning Document  

---

## Table of Contents

1. [Pipeline Inventory — What We Built](#1-pipeline-inventory--what-we-built)
2. [Best Practices Gap Analysis](#2-best-practices-gap-analysis)
   - [2.1 Corrective RAG (Retrieval Quality Grading)](#21-corrective-rag-retrieval-quality-grading)
   - [2.2 Query Decomposition (Multi-Hop Reasoning)](#22-query-decomposition-multi-hop-reasoning)
   - [2.3 Memory Architecture](#23-memory-architecture)
   - [2.4 Observability & Tracing](#24-observability--tracing)
   - [2.5 Guardrails (Input/Output Validation)](#25-guardrails-inputoutput-validation)
   - [2.6 Streaming (SSE) to Frontend](#26-streaming-sse-to-frontend)
   - [2.7 Citation Grounding & Hallucination Scoring](#27-citation-grounding--hallucination-scoring)
   - [2.8 Semantic Caching](#28-semantic-caching)
   - [2.9 State Checkpointing (PostgresSaver)](#29-state-checkpointing-postgressaver)
   - [2.10 Human-in-the-Loop Interrupts](#210-human-in-the-loop-interrupts)
3. [MCP (Model Context Protocol) Analysis](#3-mcp-model-context-protocol-analysis)
   - [3.1 What MCP Is](#31-what-mcp-is)
   - [3.2 What Our Pipeline Uses Instead](#32-what-our-pipeline-uses-instead)
   - [3.3 What MCP Would Give Us](#33-what-mcp-would-give-us)
   - [3.4 MCP Implementation Plan](#34-mcp-implementation-plan)
   - [3.5 MCP Decision Matrix](#35-mcp-decision-matrix)
4. [Known Bugs & Technical Debt](#4-known-bugs--technical-debt)
5. [Consolidated Priority Roadmap](#5-consolidated-priority-roadmap)
6. [Appendix: Full File Inventory](#6-appendix-full-file-inventory)

---

# 1. Pipeline Inventory — What We Built

The Agentic GraphRAG pipeline is a **multi-agent lesson planning system** built with LangGraph, comprising **30 files** and **~7,500+ lines** of code across `agent/`.

## Architecture Overview

```
Teacher Query (Italian/English)
        │
        ▼
┌─────────────────────────────────────────────────┐
│              AgentOrchestrator                   │
│              (orchestrator.py)                   │
│                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐     │
│  │ Planner  │→ │ Retriever │→ │  Writer  │     │
│  │  Agent   │  │   Agent   │  │  Agent   │     │
│  └──────────┘  └───────────┘  └──────────┘     │
│       │              │              │            │
│       │              │              ▼            │
│       │              │        ┌──────────┐      │
│       │              │        │  Critic  │      │
│       │              │        │  Agent   │      │
│       │              │        └──────────┘      │
│       │              │          │       │        │
│       │              │      Approve   Revise     │
│       │              │          │       │        │
│       │              │          ▼       └──→ Writer
│       │              │     Final Output          │
│       │              │                           │
│  Intent Detection    │                           │
│  Scope Detection     │                           │
│  JSON Plan           │                           │
│                      │                           │
│              ┌───────┴────────┐                  │
│              │ GraphRAG Tool  │                  │
│              │ (bridge to     │                  │
│              │  Text2Cypher + │                  │
│              │  ContextBuilder│                  │
│              └───────┬────────┘                  │
│                      │                           │
│              ┌───────┴────────┐                  │
│              │  Media Layer   │                  │
│              │  MediaLookup   │                  │
│              │  ExternalAPIs  │                  │
│              │  Mermaid/DALLE │                  │
│              └────────────────┘                  │
└─────────────────────────────────────────────────┘
```

## What Works

| Component | File | Lines | Status |
|---|---|---|---|
| Orchestrator (entry point) | `orchestrator.py` | 356 | **Working** |
| LangGraph state machine | `graph/lesson_planner_graph.py` | 186 | **Working** |
| State management | `graph/state.py` | 213 | **Working** |
| Node wiring | `graph/nodes.py` | 386 | **Working** |
| Planner Agent (intent + scope detection) | `agents/planner_agent.py` | 189 | **Working** |
| Retriever Agent (KG + media + external) | `agents/retriever_agent.py` | 619 | **Working** |
| Writer Agent (intent-specific generation) | `agents/writer_agent.py` | 415 | **Working** |
| Critic Agent (quality evaluation + revision) | `agents/critic_agent.py` | 224 | **Working** |
| GraphRAG Tool (bridge to existing pipeline) | `tools/graphrag_tool.py` | 258 | **Working** |
| Media Lookup (sidecar JSON) | `media/media_lookup.py` | 419 | **Working** |
| External APIs (YouTube, Wikipedia, etc.) | `media/external_apis.py` | 1054 | **Working** |
| Mermaid diagram generator | `media/mermaid_generator.py` | 510 | **Working** |
| DALL-E image generator | `media/image_generator.py` | 442 | **Working** |
| Resource lookup | `media/resource_lookup.py` | 447 | **Working** |
| Prompts (Planner/Writer/Critic) | `prompts/*.py` | 1,265 | **Working** |
| Domain extensions (Neuro/UDL) | `configs/domain_prompts.py` | 232 | **Working** (UDL thin) |
| Streamlit UI (Agent mode tab) | `streamlit_app.py` | — | **Working** |
| CLI test harness | `test_agent.py` | 242 | **Working** |

## What's Placeholder (Phase 3)

| Component | File | Lines | Status |
|---|---|---|---|
| Graph Updater Agent | `agents/graph_updater_agent.py` | 96 | **Stub** — returns empty |
| Curriculum Tool | `tools/curriculum_tool.py` | 180 | **Stub** — returns placeholder |
| Duplicate CurriculumTool | Inside `tools/graphrag_tool.py` | 18 | **Stub** — unused duplicate |
| Canva Generator | `media/canva_generator.py` | 238 | **Stub** — "coming soon" |
| Lesson Template | `prompts/templates/lesson_template.txt` | 94 | Present but **not imported** |

## What's Missing (Critical Gaps from Previous Analysis)

| Gap | Impact | Effort | Reference |
|---|---|---|---|
| No FastAPI endpoint for Agent mode | Frontend can't call agent pipeline | ~3h | `api/routes/` only has GraphRAG |
| UDL domain extension is a skeleton | Writer lacks UDL pedagogical depth | ~2h | `configs/domain_prompts.py` |
| No `kg_udl_media_mapping.json` | Zero curated media for UDL domain | ~1h | See Media Mapping doc |
| Agent prompts disconnected from `domains/` | Rich configs invisible to Agent | ~4h | See Prompt Integration doc |
| Neuro media mapping only 20/695 concepts | 97% concepts have no curated media | ~1h | Run script without `--limit` |

---

# 2. Best Practices Gap Analysis

Based on a comprehensive review of 2026 state-of-the-art Agentic RAG literature, production multi-agent systems, and LangGraph best practices, here are the patterns our pipeline is missing.

## Scorecard

| Best Practice | Our Status | Priority |
|---|---|---|
| Multi-agent pipeline (Plan→Retrieve→Write→Critique) | **DONE** | — |
| Scope detection (in/out of scope) | **DONE** | — |
| Revision loop with max iterations | **DONE** | — |
| Hybrid retrieval (KG + external APIs) | **DONE** | — |
| Media enrichment (videos, diagrams, OER) | **DONE** | — |
| Intent-specific prompts | **DONE** | — |
| Domain-aware configuration | **DONE** | — |
| External API rate limiting + fallbacks | **DONE** | — |
| Corrective RAG (retrieval grading) | **MISSING** | HIGH |
| Query decomposition (multi-hop) | **MISSING** | HIGH |
| Memory (conversation + long-term) | **MISSING** | HIGH |
| Observability & tracing | **MISSING** | HIGH |
| Guardrails (input/output validation) | **MISSING** | HIGH |
| Streaming (SSE) to frontend | **MISSING** | HIGH |
| Citation grounding & hallucination scoring | **MISSING** | MEDIUM |
| Semantic caching | **MISSING** | MEDIUM |
| State checkpointing (PostgresSaver) | **MISSING** | MEDIUM |
| Human-in-the-loop interrupts | **MISSING** | LOW |
| MCP tool servers | **MISSING** | LOW (for now) |

---

## 2.1 Corrective RAG (Retrieval Quality Grading)

### What It Is

After the Retriever agent gets results from the Knowledge Graph, a lightweight LLM call **grades each retrieved document/node for relevance** before passing to the Writer. If retrieval quality is poor, it triggers **query rewriting and re-retrieval** instead of blindly generating from bad context.

Research shows **30% of basic RAG responses contain factual errors traceable to poor retrieval** (AI Workflow Lab, 2026). The grading step costs only ~200-400 tokens per call but prevents the much more expensive Writer from generating off-topic content.

### Current State

Our Retriever calls `GraphRAGTool.search()` and passes whatever it gets directly to the Writer. The Critic evaluates the *final output* quality, but by then the Writer has already generated a full lesson plan from potentially irrelevant data.

### What's Missing

```
Current flow:
  Planner → Retriever → Writer → Critic → [Revise Writer]
                                              ↑ too late to fix bad retrieval

Best practice (Corrective RAG / CRAG):
  Planner → Retriever → GRADE RETRIEVAL → Writer → Critic
                ↑              │
                └── Rewrite ←──┘  (if relevance < threshold)
```

Three established patterns to consider:

**Corrective RAG (CRAG):** A retrieval evaluator grades documents for relevance, then routes: proceed if relevant, filter if partially relevant, or rewrite query and re-retrieve if all irrelevant.

**Self-RAG:** Reflection tokens embedded in generation evaluate: (a) whether to retrieve, (b) if passages are relevant, (c) if responses are supported, (d) if responses are useful.

**Adaptive RAG:** Uses query metadata filtering and automatic re-queries when confidence is low.

### Implementation Approach

Add a `RetrievalGraderNode` between Retrieve and Write in the LangGraph:

```python
# Conceptual implementation
async def grade_retrieval_node(state: AgentState) -> AgentState:
    """Grade retrieved context quality, rewrite query if needed."""
    retriever_output = state["retriever_output"]
    
    grading_prompt = f"""
    Query: {state['input']}
    Retrieved context: {retriever_output['context_summary']}
    
    Grade the relevance of the retrieved context (1-10).
    If score < 6, suggest a rewritten query.
    Output JSON: {{"score": N, "rewritten_query": "..." or null}}
    """
    
    result = await llm.agenerate(grading_prompt)
    
    if result.score < 6 and state.get("retrieval_attempts", 0) < 3:
        # Rewrite and re-retrieve
        state["input"] = result.rewritten_query
        state["retrieval_attempts"] += 1
        return state  # Route back to Retriever
    
    return state  # Proceed to Writer
```

### Impact & Effort

- **Impact:** Very High — prevents the most common failure mode (bad retrieval → bad output)
- **Effort:** 3-4 hours
- **Cost per request:** +200-400 tokens (~$0.001 with GPT-5 nano)

---

## 2.2 Query Decomposition (Multi-Hop Reasoning)

### What It Is

When a teacher asks a complex, multi-faceted question, the system decomposes it into sequential sub-queries where later queries depend on the results of earlier ones.

**Example:**
> *"Crea una lezione che integri strategie per ADHD e dislessia usando i principi UDL di Rappresentazione"*

This requires:
1. "What are ADHD learning challenges?" → KG query
2. "What are dyslexia learning challenges?" → KG query
3. "What are UDL Representation strategies?" → KG query
4. "Which strategies address both ADHD and dyslexia challenges?" → cross-reference results from 1-3

### Current State

The Planner generates `search_queries` (a list of independent strings). The Retriever runs each through `GraphRAGTool.search()` independently and merges results. This is basic **multi-query** but not true **decomposition with dependency tracking**.

### What's Missing

A decomposition strategy where the Planner outputs a **DAG of sub-queries** with dependencies:

```python
# Current: flat independent queries
{
    "search_queries": [
        "ADHD learning challenges",
        "dyslexia learning challenges",
        "UDL representation strategies"
    ]
}

# Best practice: DAG with dependencies
{
    "sub_queries": [
        {"id": "q1", "query": "ADHD learning challenges", "depends_on": []},
        {"id": "q2", "query": "dyslexia learning challenges", "depends_on": []},
        {"id": "q3", "query": "UDL Representation strategies", "depends_on": []},
        {"id": "q4", "query": "strategies addressing both ADHD and dyslexia",
         "depends_on": ["q1", "q2"],
         "context_from": "cross-reference challenges from q1 and q2"}
    ]
}
```

The Retriever would execute q1, q2, q3 in parallel, then execute q4 using the results of q1+q2 as additional context.

### Impact & Effort

- **Impact:** High — educational queries are inherently multi-faceted (multiple student profiles + multiple pedagogical frameworks)
- **Effort:** 4-5 hours (Planner prompt update + Retriever sequential execution logic)
- **Research context:** Even GPT-5 achieves only 22.6% accuracy on hard multi-hop reasoning tasks (AgenticRAGTracer benchmark, Feb 2026)

---

## 2.3 Memory Architecture

### What It Is

Memory allows the agent to maintain context across turns, sessions, and users. There are three distinct types:

### 2.3.1 Short-Term Memory (Within a Single Request)

**Purpose:** Maintain context during the pipeline execution (Planner → Retriever → Writer → Critic → Revision).

**Our status: Partially implemented.** The `AgentState` TypedDict serves as short-term memory. When the Critic sends a revision request, the Writer has access to the previous output and critique via the state.

**Gap:** No within-node message history. Each LLM call within an agent (e.g., if the Writer needs multiple generation steps) is stateless — no `messages=[]` accumulation within a single agent's execution.

### 2.3.2 Conversation Memory (Within a Session — Multiple Turns)

**Purpose:** Allow multi-turn refinement of a lesson plan:

```
Turn 1: "Crea una lezione sulla memoria di lavoro"
         → System generates a full lesson plan

Turn 2: "Aggiungi un'attività per studenti con ADHD"
         → System modifies the EXISTING plan (doesn't start from scratch)

Turn 3: "Cambia la durata a 45 minuti"
         → System adjusts timing in the accumulated plan
```

**Our status: NOT implemented.**

The `session_id` field exists in `AgentState` (line 54 of `state.py`) and flows through the entire pipeline:

```
orchestrator.create_lesson_plan(query, session_id)
  → pipeline.run(query, session_id)
    → create_initial_state(query, session_id=session_id)
      → state["session_id"] = session_id
```

But **it is never used for any actual persistence or retrieval**. It is purely decorative.

The root cause is in `lesson_planner_graph.py` line 68:

```python
compiled = workflow.compile()  # ← No checkpointer!
```

Without a checkpointer, LangGraph has no memory between invocations. Every query is treated as a brand new, independent request.

**Fix:**

```python
from langgraph.checkpoint.memory import MemorySaver  # Dev
# from langgraph.checkpoint.postgres import PostgresSaver  # Production

checkpointer = MemorySaver()
compiled = workflow.compile(checkpointer=checkpointer)

# Then at invocation:
config = {"configurable": {"thread_id": session_id}}
result = await graph.ainvoke(initial_state, config)
```

**Impact:** Without conversation memory, the Agent mode is fundamentally a **single-shot tool**. Teachers cannot iteratively refine lesson plans — they must re-describe everything from scratch each time.

**Effort:** 2-3 hours for `MemorySaver` (dev), 4-5 hours for `PostgresSaver` (production).

### 2.3.3 Long-Term Memory (Across Sessions — User Profiles)

**Purpose:** Persistent knowledge about a specific teacher that survives across separate sessions:

- *"This teacher always works with middle school students"*
- *"This teacher's class has 3 ADHD students and 1 with dyslexia"*
- *"This teacher prefers activities under 20 minutes"*
- *"Last month, this teacher generated 5 lesson plans on executive functions"*

**Our status: NOT implemented.** No user profiles, no preference learning, no cross-session persistence.

**Architecture:**

```
LangGraph distinguishes:

  Checkpointer = per-thread conversation history (session memory)
  Store         = cross-thread persistent facts (user profiles)

  These are SEPARATE systems. Mixing them up is the #1 architecture mistake.
```

**Implementation options:**

| Option | Persistence | Effort | Best For |
|---|---|---|---|
| `InMemoryStore` | Volatile (lost on restart) | 2h | Dev/testing |
| `PostgresStore` | Durable | 4h | Production |
| **Mem0** (open source) | Durable + auto-extraction | 6h | Advanced (auto-learns from conversations) |

**Why it matters for us specifically:** Our system serves Italian teachers working with specific student variabilities (ADHD, autism, dyslexia). If a teacher repeatedly queries about the same student profiles, the system should learn this and auto-configure. Currently, the teacher must specify "ho studenti con ADHD" in every single query.

**Effort:** 6-8 hours for a basic teacher profile store.

### 2.3.4 Memory Summary

| Memory Type | Scope | LangGraph Mechanism | Our Status | Priority |
|---|---|---|---|---|
| **Short-Term** | Within request | `AgentState` TypedDict | **Partial** | LOW (works for revision loop) |
| **Conversation** | Within session | `Checkpointer` + `thread_id` | **NOT IMPLEMENTED** | **HIGH** |
| **Long-Term** | Across sessions | `Store` (Postgres/Memory) | **NOT IMPLEMENTED** | MEDIUM |

---

## 2.4 Observability & Tracing

### What It Is

End-to-end tracing of every agent step: which LLM was called, what prompt was sent, how many tokens were used, what the latency was, and whether the output was correct. An "agent black box recorder."

### Current State

- Basic `logging.info/warning/error` throughout the pipeline
- GlitchTip for crash monitoring (errors and exceptions)
- **No structured traces** — cannot visualize a single request flowing through Planner → Retriever → Writer → Critic
- **No cost tracking** — no idea how many tokens or dollars each request costs
- **No evaluation datasets** — no systematic way to compare quality across runs

### What's Missing

**Option A: LangSmith (managed service)**
- Visual trace trees per request
- Latency breakdowns per node
- Cost per run (automatic token counting)
- Evaluation datasets and A/B comparison
- Free tier: 5,000 traces/month
- Setup: Add env vars `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`
- Effort: ~2 hours

**Option B: LangFuse (open source, self-hosted)**
- Same concept, no vendor lock-in
- Self-hosted on Docker
- Effort: ~4 hours

**Option C: Custom observability**
- Structured logging with request IDs
- Token counting middleware
- Dashboard (Grafana or similar)
- Effort: ~8 hours

### Recommendation

Start with **LangSmith** (2-hour setup, free tier sufficient for our volume). Migrate to LangFuse if data sovereignty becomes a concern.

### Impact & Effort

- **Impact:** High — essential for debugging, cost monitoring, and demonstrating quality to Direction
- **Effort:** 2 hours (LangSmith) / 4 hours (LangFuse)

---

## 2.5 Guardrails (Input/Output Validation)

### What It Is

Multi-layer safety checks that prevent prompt injection, validate output structure, and ensure educational content quality.

### Current State

**Zero guardrails.** No input validation, no output validation, no content moderation. The Critic agent partially serves as a quality check, but it's an LLM opinion, not structured validation.

### What's Missing

**Input guardrails (before Planner):**

| Check | Purpose | Implementation |
|---|---|---|
| Prompt injection detection | Block "ignore all previous instructions" attacks | Regex patterns + lightweight classifier |
| Query length/complexity limits | Prevent abuse | Simple length check |
| Language detection | Reject non-Italian/English | `langdetect` library or LLM |
| Educational topic validation | Reject non-educational queries | Planner already does scope detection (partial) |

**Output guardrails (after Writer, before return):**

| Check | Purpose | Implementation |
|---|---|---|
| Structural validation | Ensure lesson plan has all required sections | Pydantic schema validation |
| Content safety | No inappropriate content for education | OpenAI Moderation API (free) |
| Factual grounding | Output supported by retrieved KG data? | Semantic similarity scoring |
| PII detection | No student personal data leakage | Regex + `presidio-analyzer` |

**Indirect prompt injection (the hidden threat):**

A critical concern for RAG systems: malicious instructions can be embedded in the KG data itself (or in external API results). When the Retriever fetches this data, the Writer processes the embedded instructions as legitimate content. This bypasses all input-level defenses.

Mitigation: Scan retrieved content before passing to Writer, use instruction hierarchy (system prompt > retrieved data > user input).

### Impact & Effort

- **Impact:** High — especially critical for an educational system serving teachers. One bad output damages trust.
- **Effort:** 3 hours (basic input validation + Pydantic output). 5 hours (+ content safety + PII).

---

## 2.6 Streaming (SSE) to Frontend

### What It Is

Real-time progress updates to the frontend as the agent pipeline executes, with token-by-token streaming of the final output.

### Current State

- **Streamlit:** Simulates progress with `st.status()` steps — not true streaming
- **FastAPI:** No agent endpoint exists at all. The GraphRAG endpoint (`/api/v1/context`) returns a single response after completion.
- **Lovable frontend:** Would experience a 20-30 second blank wait with no feedback

### What's Missing

Two types of streaming are needed:

**Step streaming (progress updates):**
```
data: {"type": "step_start", "node": "planner", "message": "Analyzing query..."}
data: {"type": "step_complete", "node": "planner", "message": "Intent: lesson_creation"}
data: {"type": "step_start", "node": "retriever", "message": "Searching Knowledge Graph..."}
data: {"type": "step_complete", "node": "retriever", "message": "Found 12 relevant concepts"}
data: {"type": "step_start", "node": "writer", "message": "Generating lesson plan..."}
```

**Token streaming (Writer output):**
```
data: {"type": "token", "content": "# "}
data: {"type": "token", "content": "Lezione"}
data: {"type": "token", "content": ": "}
data: {"type": "token", "content": "La "}
data: {"type": "token", "content": "Memoria "}
...
```

### Implementation Approach

```python
# FastAPI SSE endpoint
from fastapi.responses import StreamingResponse

@router.post("/api/v1/agent/lesson")
async def create_lesson_stream(request: LessonRequest):
    async def event_stream():
        async for event in pipeline.astream_events(query, version="v2"):
            if event["event"] == "on_chain_start":
                yield f"data: {json.dumps({'type': 'step', ...})}\n\n"
            elif event["event"] == "on_chat_model_stream":
                yield f"data: {json.dumps({'type': 'token', ...})}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### Impact & Effort

- **Impact:** High — critical for UX. A 20-30 second wait with no feedback is unacceptable for production.
- **Effort:** 4 hours (SSE endpoint + node progress). 6 hours (+ token streaming).

---

## 2.7 Citation Grounding & Hallucination Scoring

### What It Is

Every claim in the Writer's output should be traceable to a specific KG node or retrieved source. Post-generation scoring checks how well the output is "grounded" in actual retrieved data vs. the LLM's parametric knowledge.

### Current State

The Writer receives KG data and *should* use it, but there's no verification. The Critic evaluates overall quality but doesn't check individual claims against source data. If the Writer halluccinates a strategy that doesn't exist in the KG, nobody catches it.

### What's Missing

**Source attribution:** Each strategy recommendation should cite which KG node it came from:

```markdown
## Strategie Raccomandate

1. **Scaffolding Graduato** [Fonte: KG → Scaffolding → SUGGESTS → Adhd]
   - Implementazione: ...

2. **Attività Multisensoriali** [Fonte: KG → MultisensoryActivities → SUGGESTS → Adhd]
   - Implementazione: ...
```

**Hallucination scoring:** Post-generation semantic similarity between the output text and the retrieved context. Low similarity = Writer is hallucinating:

```python
grounding_score = cosine_similarity(
    embed(writer_output),
    embed(retrieved_context)
)
# If grounding_score < 0.5 → flag for review
```

**Citation enforcement:** Modify the Writer prompt to require `[Source: KG]` or `[Source: External]` markers. The Critic prompt should penalize unsourced claims.

Research shows citation-grounded RAG improves grounded response accuracy by **22%** compared to baseline (CiteMind, 2026).

### Impact & Effort

- **Impact:** Medium-High — important for credibility, especially when demonstrating to domain experts
- **Effort:** 3 hours (citation enforcement in Writer/Critic prompts + basic similarity scoring)

---

## 2.8 Semantic Caching

### What It Is

Cache LLM responses for semantically similar queries. When two teachers ask equivalent questions with different wording, return the cached response instead of running the full pipeline.

### Current State

**No caching at any level.** Every query runs the full Planner → Retriever → Writer → Critic pipeline, even if the same question was asked 5 minutes ago.

### What's Missing

Production data shows (TechBuddies.io, Jan 2026):
- Only 18% of queries are exact duplicates
- But **47% are semantically similar** (different wording, same intent)
- Semantic caching reduces LLM costs by **30-70%**
- Reduces average latency by **65%**

For our pipeline (which takes ~20-30 seconds per agent request), this could mean instant responses for repeated topics.

### Implementation Options

| Option | Persistence | Latency | Effort | Best For |
|---|---|---|---|---|
| In-memory FAISS + dict | Volatile | ~0.5ms | 3h | Dev/testing |
| Redis + embeddings | Durable | ~2ms | 6h | Production |
| Two-layer (GraphRAG cache + Agent cache) | Durable | ~2ms | 8h | Full optimization |

**Two-layer caching strategy:**

```
Layer 1: GraphRAG Cache (cache Cypher query → KG results)
  Key: semantic hash of query + domain
  TTL: 1 hour (KG data rarely changes)
  Saves: Text2Cypher + Neo4j round trip (~5-10 seconds)

Layer 2: Agent Cache (cache full lesson plans)
  Key: semantic hash of query + domain + language
  TTL: 24 hours (lesson plans are stable)
  Saves: Full pipeline execution (~20-30 seconds)
```

### Impact & Effort

- **Impact:** Medium-High — significant cost and latency reduction
- **Effort:** 3 hours (basic in-memory) / 6-8 hours (Redis production)

---

## 2.9 State Checkpointing (PostgresSaver)

### What It Is

Persisting the LangGraph state to a database at each node transition, enabling crash recovery, resumption, and the foundation for conversation memory (Section 2.3.2).

### Current State

The LangGraph runs entirely in memory. If the server crashes mid-pipeline (e.g., between Writer and Critic), the entire request is lost and must be restarted.

The graph compiles without any checkpointer:

```python
compiled = workflow.compile()  # No checkpointer → no persistence
```

### What's Missing

Replace the bare `compile()` with a production checkpointer:

```python
# Development:
from langgraph.checkpoint.memory import MemorySaver
compiled = workflow.compile(checkpointer=MemorySaver())

# Production:
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(POSTGRES_URI)
compiled = workflow.compile(checkpointer=checkpointer)
```

This enables:
- **Crash recovery:** Resume from the last successful node
- **Conversation memory:** Required for multi-turn refinement (Section 2.3.2)
- **Debugging:** Inspect the state at any point in a failed pipeline
- **Horizontal scaling:** Multiple server instances can resume any request

### Dependency

`pip install langgraph-checkpoint-postgres` + PostgreSQL database.

Our project already uses Neo4j. Adding PostgreSQL for state persistence is a separate concern — Neo4j is for the Knowledge Graph, PostgreSQL is for operational state.

### Impact & Effort

- **Impact:** Medium — critical for production reliability and prerequisite for conversation memory
- **Effort:** 3 hours (MemorySaver for dev) / 5 hours (PostgresSaver for production)

---

## 2.10 Human-in-the-Loop Interrupts

### What It Is

Pausing the pipeline at specific points for human review/approval before continuing.

### Current State

The pipeline runs autonomously end-to-end with no human intervention points.

### What's Missing

LangGraph's `interrupt()` function enables pause/resume patterns:

**Use cases for our pipeline:**

| Interrupt Point | Trigger | Teacher Interaction |
|---|---|---|
| After Planner (scope detection) | Query is `out_of_scope` | "This topic isn't in our KG. Search external sources?" |
| After Retriever (low results) | < 3 relevant nodes found | "Limited KG data found. Proceed with general knowledge?" |
| After Critic (revision) | Critic requests major revision | "The system wants to revise significantly. Accept current or revise?" |

**Requires:** State checkpointing (Section 2.9) — the pipeline state must persist between pause and resume.

### Impact & Effort

- **Impact:** Low — nice-to-have for production, not essential for initial deployment
- **Effort:** 2 hours (but requires Section 2.9 as prerequisite)

---

# 3. MCP (Model Context Protocol) Analysis

## 3.1 What MCP Is

MCP (Model Context Protocol) is Anthropic's open standard for connecting LLMs to external tools and data sources. It defines a client-server architecture where:

- An **MCP Server** exposes tools (functions), resources (data), and prompts via a standardized JSON-RPC protocol
- An **MCP Client** (an LLM host like Claude Desktop, Cursor, or a custom app) discovers and calls those tools dynamically

Think of it as a **"USB-C for AI"** — a universal plug-and-play protocol so any LLM application can connect to any tool without custom integration code.

**Key MCP concepts:**

| Concept | What It Does | Example |
|---|---|---|
| **Tool** | A function the LLM can call | `search_knowledge_graph(query, domain)` |
| **Resource** | Read-only data the LLM can access | KG schema, domain configurations |
| **Prompt** | Reusable prompt templates | Educational query template |
| **Transport** | Communication channel | `stdio` (local), `SSE` (remote) |

## 3.2 What Our Pipeline Uses Instead

Our pipeline uses **direct Python imports and function calls** — no protocol, no tool discovery, no external access:

```
AgentOrchestrator (Python)
    │
    ├── PlannerAgent   → OpenAI API (direct HTTP)
    ├── RetrieverAgent → GraphRAGTool (Python import)
    │                    ├── Text2Cypher (Python import)
    │                    └── ContextBuilder (Python import)
    ├── WriterAgent    → OpenAI API (direct HTTP)
    ├── CriticAgent    → OpenAI API (direct HTTP)
    │
    └── Media Layer
        ├── MediaLookup (JSON file read)
        ├── ExternalMediaAPI (HTTP via aiohttp)
        ├── MermaidGenerator (OpenAI + mermaid.ink HTTP)
        └── ImageGenerator (OpenAI DALL-E HTTP)
```

Everything is **tightly coupled** via Python imports. This works for a single-service deployment but creates limitations.

**Confirmed:** Zero MCP references exist anywhere in the `graphaixlearning` codebase (searched for `MCP`, `mcp`, `Model Context Protocol`, `tool_server`, `stdio_server`, `McpServer` — no matches).

## 3.3 What MCP Would Give Us

### MCP Server Architecture (Potential)

```
Any MCP Client
├── Claude Desktop (teachers exploring KG directly)
├── Cursor IDE (developers querying KG during development)
├── Lovable Frontend (via MCP client SDK)
├── Custom LangGraph Agent (our own pipeline)
└── Third-party educational tools
    │
    ▼
┌───────────────────────────────────────────────────┐
│  MCP Server: "graphrag-tools"                     │
│                                                    │
│  Tools:                                            │
│  ├── search_knowledge_graph(query, domain, lang)   │
│  ├── get_educational_context(query, domain)         │
│  ├── list_domain_concepts(domain, category)         │
│  └── get_kg_schema(domain)                          │
│                                                    │
│  Resources:                                        │
│  ├── kg://neuro/schema                              │
│  ├── kg://udl/schema                                │
│  ├── kg://neuro/concepts                            │
│  └── kg://udl/concepts                              │
│                                                    │
│  Prompts:                                          │
│  ├── educational-query (Italian teacher template)   │
│  └── lesson-plan-request (structured lesson prompt) │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│  MCP Server: "media-tools"                        │
│                                                    │
│  Tools:                                            │
│  ├── lookup_curated_media(concept, domain)          │
│  ├── generate_diagram(concept, type, style)         │
│  ├── search_youtube(query, max_results)             │
│  ├── search_academic(query, max_results)            │
│  └── search_oer(query, domain)                      │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│  MCP Server: "curriculum-tools" (Phase 3)         │
│                                                    │
│  Tools:                                            │
│  ├── lookup_standards(grade, subject, country)      │
│  ├── validate_lesson(plan, standards)               │
│  └── suggest_objectives(topic, grade)               │
└───────────────────────────────────────────────────┘
```

### Benefits

| Benefit | Description |
|---|---|
| **Lovable integration** | The frontend could call KG tools directly via MCP client SDK, without needing our specific FastAPI endpoint format |
| **Tool discovery** | Clients auto-discover available tools — no hardcoded API contracts |
| **Multi-client access** | Teachers could use Claude Desktop to explore the KG directly, developers could query it from Cursor |
| **Composability** | Other FEM projects could build their own agents that use our KG tools |
| **Standardization** | MCP is becoming the industry standard — future-proofs our architecture |
| **Decoupling** | KG tools become independently deployable services |

### Limitations

| Limitation | Description |
|---|---|
| **Overhead** | Adds a JSON-RPC layer where direct Python calls work fine |
| **Complexity** | More moving parts to deploy and monitor |
| **Maturity** | MCP ecosystem is still young (2024-2026), SDKs evolving rapidly |
| **Latency** | JSON-RPC serialization adds ~1-5ms per call |
| **Single consumer** | Currently only our own pipeline uses these tools |

## 3.4 MCP Implementation Plan

If we decide to implement MCP, the approach would be:

### Phase 1: GraphRAG MCP Server (Effort: 1 day)

Wrap the existing `GraphRAGTool` as an MCP server:

```python
# Conceptual structure
from mcp.server import Server
from mcp.types import Tool, Resource

server = Server("graphrag-tools")

@server.tool("search_knowledge_graph")
async def search_kg(query: str, domain: str = "neuro", language: str = "it"):
    """Search the educational Knowledge Graph."""
    tool = GraphRAGTool()
    result = await tool.search(query, domain)
    return result

@server.resource("kg://{domain}/schema")
async def get_schema(domain: str):
    """Get the KG schema for a domain."""
    config = get_domain_config(domain)
    return config.get_valid_labels()

# Transport: stdio for local, SSE for remote
server.run(transport="sse", port=8001)
```

### Phase 2: Media MCP Server (Effort: 0.5 day)

Wrap `MediaLookup` and `ExternalMediaAPI`:

```python
@server.tool("lookup_curated_media")
async def lookup_media(concept: str, domain: str = "neuro"):
    lookup = MediaLookup(domain=domain)
    return lookup.get_media(concept)

@server.tool("search_youtube")
async def search_youtube(query: str, max_results: int = 3):
    api = ExternalMediaAPI()
    return await api.search_youtube(query, max_results)
```

### Phase 3: Lovable Integration (Effort: 1 day)

Configure the Lovable frontend to connect as an MCP client:

```javascript
// Lovable frontend (conceptual)
import { McpClient } from "@modelcontextprotocol/sdk";

const client = new McpClient("https://our-server.com/mcp");
const tools = await client.listTools();  // Auto-discovery

const result = await client.callTool("search_knowledge_graph", {
    query: "strategie per studenti con ADHD",
    domain: "udl"
});
```

## 3.5 MCP Decision Matrix

| Factor | Assessment |
|---|---|
| **Value for current use** | Low — we have one consumer (our own pipeline) |
| **Value for Lovable frontend** | Medium — could simplify integration, but a REST API also works |
| **Value for multi-client access** | High — if FEM wants Claude Desktop / Cursor / other tools to access KG |
| **Value for composability** | High — other FEM projects could build agents using our KG |
| **Strategic signal** | Strong — shows architectural maturity |
| **Effort** | ~2-3 days total for Phases 1-3 |
| **Risk** | Low — MCP servers coexist alongside direct Python imports (additive, not replacement) |
| **Priority** | **LOW for now** — implement after FastAPI endpoint, memory, and guardrails |

### Recommendation

**Don't implement MCP now.** The immediate priorities are the production-hardening gaps (memory, guardrails, observability, streaming). MCP becomes relevant when:

1. The Lovable frontend needs tool discovery (vs. a fixed REST API contract)
2. FEM wants other AI tools to access the Knowledge Graph
3. The project scales beyond a single-service deployment

When the time comes, MCP implementation is straightforward (~2-3 days) because our tools are already well-encapsulated in `GraphRAGTool`, `MediaLookup`, and `ExternalMediaAPI`.

---

# 4. Known Bugs & Technical Debt

| # | Bug/Debt | Location | Severity | Effort |
|---|---|---|---|---|
| 1 | `diagram_factory.py` calls `ImageGenerator.generate_diagram()` but the actual method is `generate_educational_diagram()` | `media/diagram_factory.py` | **Runtime error** if DALL-E branch is used | 15 min |
| 2 | Duplicate `CurriculumTool` class (defined in both `tools/curriculum_tool.py` AND inside `tools/graphrag_tool.py`) | `tools/` | Low (both are stubs) | 15 min |
| 3 | `lesson_template.txt` exists but is not imported by any Python code | `prompts/templates/` | Low (unused asset) | 30 min to wire or delete |
| 4 | `session_id` flows through entire pipeline but is never used for persistence | `state.py`, `orchestrator.py`, `lesson_planner_graph.py` | **Architectural** (see Memory, Section 2.3) | Part of memory implementation |
| 5 | `domain_prompts.py` UDL section labeled "Placeholder" (~25 lines vs Neuro's ~85) | `configs/domain_prompts.py` | Quality gap for UDL domain | 2h |

---

# 5. Consolidated Priority Roadmap

## Tier 1 — Before Production (Estimated: 3-4 days)

| # | Practice | Impact | Effort | Dependencies |
|---|---|---|---|---|
| 1 | **FastAPI endpoint for Agent mode** | Lovable can't work without it | 3h | None |
| 2 | **Conversation memory** (Checkpointer) | Teachers can't refine lesson plans | 3-5h | None |
| 3 | **Corrective RAG** (retrieval grading) | 30% of responses have retrieval errors | 3-4h | None |
| 4 | **Guardrails** (input/output validation) | Safety for educational system | 3-5h | None |
| 5 | **Observability** (LangSmith) | Can't debug production failures | 2h | LangSmith API key |
| 6 | **Streaming** (SSE to frontend) | 30-second blank wait is unacceptable | 4-6h | #1 (FastAPI endpoint) |
| 7 | **UDL domain enrichment** | Writer lacks UDL pedagogical depth | 2h | None |
| 8 | **UDL media mapping** | Zero curated media for UDL | 1h + script fixes | `generate_media_mapping.py` fixes |

## Tier 2 — After Initial Launch (Estimated: 2-3 days)

| # | Practice | Impact | Effort | Dependencies |
|---|---|---|---|---|
| 9 | **Query decomposition** (multi-hop) | Complex queries fail | 4-5h | None |
| 10 | **Citation grounding** | Hallucinated strategies | 3h | None |
| 11 | **Semantic caching** | Cost + latency for repeated queries | 4-8h | Redis (optional) |
| 12 | **State checkpointing** (PostgresSaver) | Crash recovery, production memory | 5h | PostgreSQL |
| 13 | **Long-term memory** (user profiles) | Teacher preference learning | 6-8h | #12 |
| 14 | **Agent prompts ↔ domains/ integration** | Rich domain configs reach Agent | 4h | None |
| 15 | **Neuro media mapping expansion** | 97% concepts unmapped | 1h | Run existing script |

## Tier 3 — Strategic / Future (Estimated: 5-7 days)

| # | Practice | Impact | Effort | Dependencies |
|---|---|---|---|---|
| 16 | **MCP tool servers** | External tool integration | 2-3 days | None |
| 17 | **Human-in-the-loop** interrupts | User control over decisions | 2h | #12 |
| 18 | **Graph Updater Agent** (Phase 3) | Auto-learn new concepts | 2-3 days | Research needed |
| 19 | **Curriculum Tool** (Phase 3) | Italian curriculum standards | 2-3 days | Data source needed |
| 20 | **Cost monitoring** | Per-request token tracking | 3h | #5 (LangSmith does this) |
| 21 | **Canva integration** | Professional diagrams | 1-2 days | Canva API key |

## Visual Roadmap

```
JANUARY-FEBRUARY 2026                    MARCH 2026                    APRIL+ 2026
─────────────────────                    ──────────                    ───────────
TIER 1: Production Ready                 TIER 2: Enhancement           TIER 3: Strategic
                                          
[1] FastAPI Agent endpoint               [9]  Query decomposition      [16] MCP servers
[2] Conversation memory                  [10] Citation grounding       [17] Human-in-loop
[3] Corrective RAG                       [11] Semantic caching         [18] Graph Updater
[4] Guardrails                           [12] PostgresSaver            [19] Curriculum Tool
[5] Observability (LangSmith)            [13] Long-term memory         [20] Cost monitoring
[6] SSE streaming                        [14] Prompt integration       [21] Canva
[7] UDL domain enrichment               [15] Neuro media expansion
[8] UDL media mapping

◄──── ~3-4 days ────►                    ◄──── ~2-3 days ────►         ◄── ~5-7 days ──►
```

---

# 6. Appendix: Full File Inventory

## `agent/` Directory Tree (30 files)

```
agent/
├── __init__.py                          (27 lines)   Working — re-exports
├── orchestrator.py                      (356 lines)  Working — entry point
│
├── agents/
│   ├── __init__.py                      (23 lines)   Working — re-exports
│   ├── planner_agent.py                 (189 lines)  Working — intent + scope
│   ├── retriever_agent.py               (619 lines)  Working — KG + media + APIs
│   ├── writer_agent.py                  (415 lines)  Working — intent-specific gen
│   ├── critic_agent.py                  (224 lines)  Working — quality eval
│   └── graph_updater_agent.py           (96 lines)   STUB — Phase 3
│
├── configs/
│   ├── __init__.py                      (22 lines)   Working — re-exports
│   └── domain_prompts.py               (232 lines)  Working — Neuro full, UDL thin
│
├── graph/
│   ├── __init__.py                      (14 lines)   Working — re-exports
│   ├── lesson_planner_graph.py          (186 lines)  Working — LangGraph machine
│   ├── nodes.py                         (386 lines)  Working — node functions
│   └── state.py                         (213 lines)  Working — state types
│
├── media/
│   ├── __init__.py                      (92 lines)   Working — package surface
│   ├── canva_generator.py              (238 lines)  STUB — "coming soon"
│   ├── diagram_factory.py              (371 lines)  Partial — DALL-E method bug
│   ├── external_apis.py                (1054 lines) Working — YouTube, Wiki, etc.
│   ├── image_generator.py              (442 lines)  Working — DALL-E + cache
│   ├── media_lookup.py                 (419 lines)  Working — sidecar JSON
│   ├── mermaid_generator.py            (510 lines)  Working — LLM → Mermaid
│   └── resource_lookup.py             (447 lines)  Working — static resources
│
├── prompts/
│   ├── __init__.py                      (16 lines)   Working — re-exports
│   ├── planner_prompt.py               (295 lines)  Working — system + user
│   ├── writer_prompt.py                (725 lines)  Working — intent-specific
│   ├── critic_prompt.py                (245 lines)  Working — evaluation criteria
│   └── templates/
│       └── lesson_template.txt          (94 lines)   NOT IMPORTED by code
│
└── tools/
    ├── __init__.py                      (11 lines)   Working — exports GraphRAGTool
    ├── graphrag_tool.py                (258 lines)  Working — bridge to pipeline
    └── curriculum_tool.py              (180 lines)  STUB — Phase 3
```

**Total: ~7,500+ lines of code | 24 working files | 4 stubs | 2 bugs**

---

*Document generated January 27, 2026. Based on comprehensive codebase investigation, 2026 Agentic RAG literature review, and LangGraph production best practices analysis.*
