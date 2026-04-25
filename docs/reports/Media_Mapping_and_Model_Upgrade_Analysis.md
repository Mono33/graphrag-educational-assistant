# Media Mapping & Model Upgrade Strategy

**Author:** Louis Mono — AI Team Lead  
**Date:** January 27, 2026  
**Project:** AIxLearning — GraphRAG Pipeline  
**Status:** Analysis & Decision Document  

---

## Table of Contents

1. [Part A: Media Mapping Generator Analysis](#part-a-media-mapping-generator-analysis)
   - [A1. What the Script Does](#a1-what-the-script-does)
   - [A2. Architecture & Data Flow](#a2-architecture--data-flow)
   - [A3. CLI Parameters](#a3-cli-parameters)
   - [A4. Current Neuro Coverage](#a4-current-neuro-coverage)
   - [A5. Issues Blocking UDL Support](#a5-issues-blocking-udl-support)
   - [A6. Cost & Time Estimates](#a6-cost--time-estimates)
   - [A7. Recommended Fixes for UDL](#a7-recommended-fixes-for-udl)
2. [Part B: Model Upgrade Strategy (GPT-4o → GPT-5.x)](#part-b-model-upgrade-strategy-gpt-4o--gpt-5x)
   - [B1. Current Model Usage in the Codebase](#b1-current-model-usage-in-the-codebase)
   - [B2. GPT-5 Family Overview](#b2-gpt-5-family-overview)
   - [B3. Detailed Model Comparison](#b3-detailed-model-comparison)
   - [B4. Performance Benchmarks](#b4-performance-benchmarks)
   - [B5. Technical Compatibility Analysis](#b5-technical-compatibility-analysis)
   - [B6. Tiered Upgrade Strategy](#b6-tiered-upgrade-strategy)
   - [B7. Risks & Mitigations](#b7-risks--mitigations)
   - [B8. Execution Plan](#b8-execution-plan)
   - [B9. Final Recommendation](#b9-final-recommendation)

---

# Part A: Media Mapping Generator Analysis

## A1. What the Script Does

`generate_media_mapping.py` generates a **sidecar JSON file** (`kg_{domain}_media_mapping.json`) that enriches Knowledge Graph concepts with curated educational media. It is used exclusively by the **Agent mode** (not GraphRAG mode).

**Processing pipeline:**

1. **Read** the KG JSON file (e.g., `kg_neuro_neo4j.json`)
2. **Extract** unique concept nodes (deduplicated by name)
3. **Prioritize** concepts (core educational concepts first)
4. **Send each concept** to GPT-4o via OpenAI API
5. **GPT-4o generates** structured JSON recommendations per concept
6. **Aggregate** results into a single sidecar JSON file

**For each concept, the LLM generates:**

| Category | Items per Concept | Example |
|---|---|---|
| Educational Videos | 2-3 | YouTube search queries, CrashCourse, TED-Ed |
| Diagrams/Images | 1-2 | Descriptions for educational diagrams |
| External Resources | 2-3 | Wikipedia, Simply Psychology, Verywell Mind |
| Academic Citations | 2-3 | Seminal papers with DOIs |
| Open Textbooks (OER) | 1-2 | OpenStax, DOAB, Pressbooks chapters |

## A2. Architecture & Data Flow

```
KG JSON File (kg_neuro_neo4j.json)
        │
        ▼
extract_unique_concepts()  ──→  List of concept dicts
        │
        ▼
prioritize_concepts()      ──→  Sorted by educational importance
        │
        ▼
process_concepts_batch()   ──→  Batched async API calls
        │                        (batch_size concurrent)
        ▼                        (1s pause between batches)
generate_media_for_concept()──→  GPT-4o with JSON mode
        │
        ▼
kg_{domain}_media_mapping.json  ──→  Used by agent/media/media_lookup.py
                                      ──→  Loaded by RetrieverAgent
                                      ──→  Injected into WriterAgent context
```

**Who consumes the output:**
- `agent/media/media_lookup.py` — `MediaLookup` class loads the JSON
- `agent/agents/retriever_agent.py` — calls `MediaLookup.get_media()` for concepts
- `agent/agents/writer_agent.py` — embeds curated media into lesson plans

**Not used by:** GraphRAG mode / FastAPI (`/api/v1/context`).

## A3. CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `--domain` | `neuro` | Knowledge domain (`neuro` or `udl`) |
| `--batch-size` | `5` | Concurrent API calls per batch |
| `--limit` | `None` | Cap the number of concepts (for testing/budget control) |
| `--output` | `kg_{domain}_media_mapping.json` | Custom output path |
| `--model` | `gpt-4o` | OpenAI model to use |

**Batch size guidance:**
- `--batch-size 5`: Safe for Tier 1 API accounts (500 RPM)
- `--batch-size 10`: Usable for Tier 2+ (5000 RPM)
- `--batch-size 20`: Maximum practical (diminishing returns, rate limit risk)

## A4. Current Neuro Coverage

The existing `kg_neuro_media_mapping.json` was generated on **January 16, 2026** and covers:

| Metric | Value |
|---|---|
| Concepts mapped | **20 out of ~695** (2.9% coverage) |
| Model used | `gpt-4o` |
| Generation time | 71.5 seconds |
| Total videos | 47 |
| Total images | 39 |
| Total resources | 46 |
| Total citations | 42 |
| Total OER textbooks | 40 |

**Implication:** The script was likely run with `--limit 20` or similar for initial testing. The vast majority of Neuro KG concepts (~675) have **no media mapping**.

**Concepts covered:** The 20 concepts are the highest-priority ones (Selective Attention, Working Memory, Executive Functions, etc.) due to `prioritize_concepts()` sorting.

## A5. Issues Blocking UDL Support

Three critical issues prevent running `python scripts/ml/generate_media_mapping.py --domain udl`:

### Issue 1: Wrong KG File Path

```python
# Current code (line 387):
kg_path = base_path / f"kg_{args.domain}_neo4j.json"
# This resolves to: graphaixlearning/kg_udl_neo4j.json

# But the actual UDL file is at:
# graphaixlearning/UDLdata/kg_udl_neo4j.json
```

**Fix:** Add a domain-to-path mapping or move the file.

### Issue 2: Neuro-Specific System Prompt

The `MEDIA_GENERATOR_SYSTEM_PROMPT` (lines 53-145) is entirely neuroscience-focused:

```
"You are an expert educational content curator specializing in
 **cognitive neuroscience, educational psychology, and evidence-based learning**."
```

It references:
- "Cognitive processes (attention, memory, executive functions)"
- "Neuroscience of education (neuroplasticity, brain development)"
- CrashCourse, TED-Ed channels (relevant for neuro, not UDL)
- "OpenStax Psychology" as OER source (Psychology, not UDL/Inclusive Education)

**Fix:** Create a UDL-specific system prompt referencing:
- Universal Design for Learning (CAST framework)
- Inclusive education, differentiated instruction
- Variability profiles (ADHD, autism, dyslexia, etc.)
- Relevant channels (e.g., CAST UDL, Understood.org, IRIS Center)
- UDL-specific OER (CAST guidelines, IRIS modules, etc.)

### Issue 3: Neuro-Specific Priority Categories

`prioritize_concepts()` (lines 313-354) uses hardcoded neuroscience categories:

```python
priority_categories = [
    'attention types', 'memory systems', 'executive functions',
    'cognitive processes', 'learning processes', ...
]
priority_labels = [
    'Attention', 'Memory', 'ExecutiveFunctions', 'Metacognition', ...
]
```

These don't match UDL KG labels (`Adhd`, `AutismSpectrum`, `Dyslexia`, `UdlPrinciple`, `Barrier`, `MitigationStrategy`, etc.)

**Fix:** Add UDL-specific priority categories or load them from `udl_domain.py`.

## A6. Cost & Time Estimates

Based on the Neuro run (20 concepts in 71.5s ≈ 3.6s/concept):

| Task | Concepts | Est. Time | Est. Cost (GPT-4o) | Est. Cost (GPT-5.4 mini) |
|---|---|---|---|---|
| Remaining Neuro | ~675 | ~40 min | ~$5-8 | ~$2-3 |
| Full UDL domain | ~763 | ~46 min | ~$6-9 | ~$2-4 |
| Full both domains | ~1,438 | ~86 min | ~$11-17 | ~$4-7 |

**Per concept cost breakdown** (GPT-4o):
- Input: ~800 tokens × $2.50/M = $0.002
- Output: ~1,200 tokens × $10.00/M = $0.012
- Total: ~$0.014/concept

## A7. Recommended Fixes for UDL

**Effort estimate:** ~2 hours

| Fix | Priority | Effort |
|---|---|---|
| Add domain-aware KG path resolution | Critical | 15 min |
| Create UDL system prompt | Critical | 45 min |
| Add UDL priority categories | Moderate | 30 min |
| Create UDL user prompt template | Moderate | 20 min |
| Test with `--limit 5` | Critical | 10 min |

---

# Part B: Model Upgrade Strategy (GPT-4o → GPT-5.x)

## B1. Current Model Usage in the Codebase

The pipeline uses OpenAI models in **6 distinct places**, each with different quality requirements:

| # | Component | File | Current Model | How Set | Quality Need |
|---|---|---|---|---|---|
| 1 | Text2Cypher | `text2cypher.py` | Via `config.openai.model` | `.env` → `OPENAI_MODEL` | **HIGH** (code gen) |
| 2 | LLM Response Chain | `llm_chain.py` | Via `config.openai.model` | `.env` → `OPENAI_MODEL` | **HIGH** (writing) |
| 3 | Italian Translation | `multilingual_text2cypher.py` | `gpt-4o-mini` | **Hardcoded** (line 412) | **LOW** (translation) |
| 4 | Agent Planner | `agent/agents/planner_agent.py` | `gpt-4o` | **Hardcoded** default | **VERY HIGH** (reasoning) |
| 5 | Agent Writer | `agent/agents/writer_agent.py` | `gpt-4o` | **Hardcoded** default | **VERY HIGH** (content) |
| 6 | Agent Critic | `agent/agents/critic_agent.py` | `gpt-4o` | **Hardcoded** default | **HIGH** (evaluation) |
| 7 | Media Mapping | `generate_media_mapping.py` | `gpt-4o` | CLI `--model` arg | **MEDIUM** (batch) |

**Config architecture:**
- `config.py` defines default `gpt-3.5-turbo-instruct` (line 29)
- Overridden at runtime by `OPENAI_MODEL` env var (line 105)
- Changing `.env` upgrades components #1 and #2 immediately
- Components #3-7 require code changes (hardcoded model strings)

## B2. GPT-5 Family Overview

As of January 2026, the complete GPT-5 family available via API:

### Flagship Models

| Model | Release | Context | Input $/M | Output $/M | Speed | Key Feature |
|---|---|---|---|---|---|---|
| **GPT-5.4** | Mar 5, 2026 | **1.05M** in / 128K out | $2.50 | $15.00 | 74 tok/s | Frontier. Native computer use, agentic |
| **GPT-5.4 pro** | Mar 2026 | 1.05M | $30.00 | $180.00 | Slow | Maximum reasoning |
| **GPT-5.2** | Dec 11, 2025 | 400K (272K usable) | $1.75 | $14.00 | ~80 tok/s | Deep reasoning, 90%+ ARC-AGI |
| **GPT-5.2 pro** | Dec 2025 | 400K | $15.00 | $60.00 | Slow | Extended reasoning |
| **GPT-5.1** | Nov 13, 2025 | 400K (272K usable) | $1.25 | $10.00 | Adaptive | Adaptive reasoning (2-3x faster on simple) |
| **GPT-5.0** | Aug 7, 2025 | 400K / 128K out | $1.25 | $10.00 | ~90 tok/s | <1% hallucinations, first GPT-5 |
| **GPT-4o** *(current)* | May 2024 | 128K / 4K out | $2.50 | $10.00 | 131 tok/s | General purpose, fast |

### Budget Models

| Model | Release | Context | Input $/M | Output $/M | Speed | Key Feature |
|---|---|---|---|---|---|---|
| **GPT-5.4 mini** | Mar 17, 2026 | 400K / 128K out | $0.75 | $4.50 | Fast | 54.4% SWE-Bench Pro |
| **GPT-5.4 nano** | Mar 17, 2026 | 400K / 128K out | $0.20 | $1.25 | Fastest | Classification, data extraction |
| **GPT-5 mini** | 2025 | 400K | $0.25 | $2.00 | Fast | Budget general |
| **GPT-5 nano** | 2025 | 400K | $0.05 | $0.40 | Fastest | Cheapest available |
| **GPT-5.3 Instant** | Mar 3, 2026 | 400K | ~$0.30 | ~$1.20 | Very fast | 26.8% fewer hallucinations vs 5.2 |

### Reasoning Models (o-series)

| Model | Input $/M | Output $/M | Key Feature |
|---|---|---|---|
| **o4-mini** | $1.10 | $4.40 | Best reasoning-per-dollar |
| **o3** | $2.00 | $8.00 | Strong reasoning |

## B3. Detailed Model Comparison

### Cost Comparison vs GPT-4o (per 1M tokens)

```
MODEL              INPUT    OUTPUT   vs GPT-4o OUTPUT
─────────────────────────────────────────────────────
GPT-4o (current)   $2.50    $10.00   baseline
GPT-5.0            $1.25    $10.00   same output, 50% cheaper input
GPT-5.1            $1.25    $10.00   same output, 50% cheaper input
GPT-5.2            $1.75    $14.00   +40% more expensive
GPT-5.4            $2.50    $15.00   +50% more expensive
GPT-5.4 mini       $0.75     $4.50   -55% cheaper
GPT-5.4 nano       $0.20     $1.25   -87.5% cheaper
GPT-5 mini         $0.25     $2.00   -80% cheaper
GPT-5 nano         $0.05     $0.40   -96% cheaper
GPT-5.3 Instant    $0.30     $1.20   -88% cheaper
```

### Context Window Comparison

```
MODEL              CONTEXT IN     MAX OUTPUT
────────────────────────────────────────────
GPT-4o             128K           4,096       ← current limitation
GPT-5.0            400K           128K
GPT-5.1            400K (272K)    128K
GPT-5.2            400K (272K)    128K
GPT-5.4            1.05M          128K        ← 8x larger than GPT-4o
GPT-5.4 mini       400K           128K
GPT-5.4 nano       400K           128K
```

**This matters for us:** Our `formatted_prompt_section` can be very large (10K-30K tokens for rich KG contexts). GPT-4o's 4,096 output token limit is a real constraint for the Agent Writer generating full lesson plans. GPT-5.x models support **128K output tokens** — a 31x increase.

### Quality Benchmarks Side-by-Side

| Benchmark | GPT-4o | GPT-5.0 | GPT-5.1 | GPT-5.2 | GPT-5.4 | GPT-5.4 mini |
|---|---|---|---|---|---|---|
| SWE-Bench Pro | ~35% | 74.9%* | — | — | **57.7%** | 54.4% |
| GPQA Diamond | ~76% | — | — | — | **93.0%** | 88.0% |
| AIME 2025 | ~60% | 94.6% | — | **100%** | **100%** | — |
| GDPval (professional) | — | — | — | 70.9% | **83.0%** | — |
| Hallucinations | ~3-5% | **<1%** | — | — | ~varies | — |
| Terminal-Bench 2.0 | — | — | — | — | 75.1% | 60.0% |
| Toolathlon (tool use) | — | — | — | — | **54.6%** | 42.9% |

*SWE-Bench Verified (different benchmark variant)

## B4. Performance Analysis for Our Use Cases

### Use Case 1: Text2Cypher (NL → Cypher Query Generation)

**Current model:** `config.openai.model` (likely `gpt-4o` from `.env`)

This is **code generation** — the LLM must produce syntactically correct Cypher queries from Italian educational questions. Quality here means fewer malformed queries, fewer fallbacks, better retrieval.

| Candidate | Quality | Cost vs GPT-4o | Verdict |
|---|---|---|---|
| GPT-5.4 | Best code gen, 93% GPQA | +50% output | Overkill for Cypher |
| GPT-5.4 mini | 88% GPQA, 54.4% SWE | **-55% cheaper** | **Best value** |
| GPT-5.1 | Adaptive — fast on simple queries | Same cost | Good alternative |
| GPT-5.0 | <1% hallucinations | Same cost | Reliable |
| GPT-5 nano | 82.8% GPQA | -96% cheaper | Too low for code gen |

**Recommendation: GPT-5.4 mini** — 88% GPQA (code reasoning), 54.4% SWE-Bench, 55% cheaper.

### Use Case 2: Educational Response Writing (LLM Chain)

**Current model:** `config.openai.model` (likely `gpt-4o` from `.env`)

The final user-facing response. Quality = pedagogical accuracy, coherent Italian output, proper UDL/Neuro principle application.

| Candidate | Quality | Cost vs GPT-4o | Verdict |
|---|---|---|---|
| GPT-5.4 | Best reasoning + quality | +50% output | **Best quality** |
| GPT-5.1 | Adaptive, good quality | Same cost | Good balance |
| GPT-5.0 | <1% hallucinations | Same cost | Most reliable |
| GPT-5.4 mini | Good but not flagship | -55% cheaper | Acceptable |

**Recommendation: GPT-5.4** — this is the product-facing output, quality matters most.

### Use Case 3: Italian Translation

**Current model:** `gpt-4o-mini` (hardcoded)

Simple translation from Italian to English for query processing. Doesn't need intelligence.

| Candidate | Quality | Cost | Verdict |
|---|---|---|---|
| GPT-5 nano | Sufficient | **-96% cheaper** | **Best choice** |
| GPT-5.4 nano | Sufficient | -88% cheaper | Also good |
| GPT-5.3 Instant | Good | -88% cheaper | Good |

**Recommendation: GPT-5 nano** — translation is trivial, save maximum cost.

### Use Case 4: Agent Planner (JSON Structured Planning)

**Current model:** `gpt-4o` (hardcoded)

Must produce valid JSON with `QueryIntent`, `ScopeStatus`, and `RetrievalPlan`. Needs reliable structured output.

| Candidate | Quality | Cost vs GPT-4o | Verdict |
|---|---|---|---|
| GPT-5.4 | Best tool use (54.6% Toolathlon) | +50% output | Best but expensive |
| GPT-5.4 mini | Good tool use (42.9%) | -55% cheaper | **Best value** |
| GPT-5.1 | Adaptive reasoning | Same cost | Good |

**Recommendation: GPT-5.4 mini** — structured JSON output is its strength, much cheaper.

### Use Case 5: Agent Writer (Lesson Plan Generation)

**Current model:** `gpt-4o` (hardcoded)

The most quality-sensitive component. Generates full educational lesson plans with pedagogical depth, multi-section structure, UDL/Neuro principles.

| Candidate | Quality | Cost vs GPT-4o | 128K Output | Verdict |
|---|---|---|---|---|
| GPT-5.4 | Best overall quality | +50% output | Yes | **Best choice** |
| GPT-5.2 | Deep reasoning | +40% output | Yes | Good alternative |
| GPT-5.1 | Adaptive | Same cost | Yes | Budget option |
| GPT-5.4 mini | Good but gaps at long context | -55% cheaper | Yes | Risk of quality drop |

**Recommendation: GPT-5.4** — lesson plans are the Agent mode's core product. The 128K output limit (vs 4K for GPT-4o) is transformative for rich lesson plans.

### Use Case 6: Agent Critic (Quality Evaluation)

**Current model:** `gpt-4o` (hardcoded)

Must evaluate content quality with structured scores and approve/reject decisions.

| Candidate | Quality | Cost vs GPT-4o | Verdict |
|---|---|---|---|
| GPT-5.4 mini | 88% GPQA | -55% cheaper | **Best value** |
| GPT-5.1 | Adaptive | Same cost | Good alternative |

**Recommendation: GPT-5.4 mini** — evaluation is structured scoring, mini excels here.

### Use Case 7: Media Mapping Generation

**Current model:** `gpt-4o` (CLI arg)

Offline batch processing. Quality matters for recommendations but latency doesn't.

| Candidate | Quality | Cost Impact | Verdict |
|---|---|---|---|
| GPT-5.4 mini | Good quality | ~60% savings | **Best value** |
| GPT-5.0 | <1% hallucinations | Same cost | Most reliable |
| GPT-5 mini | Acceptable | ~80% savings | Budget option |

**Recommendation: GPT-5.4 mini** — best quality/cost ratio for batch work. Consider GPT-5.0 if hallucination accuracy is critical for citations.

## B5. Technical Compatibility Analysis

### API Compatibility

| Feature | GPT-4o | GPT-5.x | Status |
|---|---|---|---|
| Chat Completions API | Yes | Yes | **Compatible** |
| JSON mode (`response_format`) | Yes | Yes | **Compatible** |
| Structured Outputs (strict JSON) | Yes | Yes | **Compatible** — all GPT-5 variants |
| Function calling | Yes | Yes | **Compatible** |
| Temperature control | Yes | Yes | **Compatible** |
| Async client (`AsyncOpenAI`) | Yes | Yes | **Compatible** |
| Prompt caching | Partial | Yes | **Improvement** — automatic cost savings |

### LangChain Compatibility — CRITICAL WARNING

Our pipeline uses LangChain with `ChatOpenAI`:

```
requirements.txt:
  langchain>=0.1.0
  langchain-openai>=0.0.5
  langchain-community>=0.0.10
  langchain-core>=0.1.0
```

**Known issue:** GPT-5.4+ models were not recognized by LangChain's `_model_prefers_responses_api` function, causing `tool_choice.function` errors. This was fixed in:
- **PR #35594** (merged March 9, 2026)
- **PR #35643** (closed March 9, 2026)

**Required action:**
1. Upgrade `langchain-openai` to a version ≥ the March 2026 fix
2. This will cascade to upgrading `langchain`, `langchain-core`, `langchain-community`
3. **Risk:** LangChain API breaking changes between our old version and current

**For GPT-5.0 / GPT-5.1:** These models are older and should work with current LangChain versions without the GPT-5.4 fix. This is an advantage if we want to avoid a LangChain upgrade.

### OpenAI Python SDK Compatibility

Our code uses `openai.AsyncOpenAI` (in `generate_media_mapping.py`) and `ChatOpenAI` from `langchain-openai` (in the main pipeline). Both support GPT-5 model strings — it's just a model name change.

### Code Changes Required

**Minimal changes (model strings only):**

| File | Change | Lines |
|---|---|---|
| `.env` | `OPENAI_MODEL=gpt-5.4` | 1 line |
| `multilingual_text2cypher.py` | `model="gpt-5-nano"` | Line 412 |
| `agent/agents/planner_agent.py` | `def __init__(self, model: str = "gpt-5.4-mini"):` | Line 66 |
| `agent/agents/writer_agent.py` | `def __init__(self, model: str = "gpt-5.4"):` | Line 52 |
| `agent/agents/critic_agent.py` | `def __init__(self, model: str = "gpt-5.4-mini"):` | Line 60 |
| `generate_media_mapping.py` | `'--model', default='gpt-5.4-mini'` | Line 379 |

**Dependency changes:**

| Package | Current | Required | Risk |
|---|---|---|---|
| `langchain-openai` | `>=0.0.5` | `>=0.3.x` (post-March 2026) | **HIGH** — API changes |
| `langchain` | `>=0.1.0` | `>=0.3.x` | **HIGH** — API changes |
| `langchain-core` | `>=0.1.0` | `>=0.3.x` | **MEDIUM** |
| `langchain-community` | `>=0.0.10` | `>=0.3.x` | **MEDIUM** |
| `openai` | (current) | Latest | **LOW** |

## B6. Tiered Upgrade Strategy

### Option A: Conservative — GPT-5.0/5.1 (No LangChain Upgrade)

Use GPT-5.0 or GPT-5.1 as drop-in replacements. These are older GPT-5 models that likely work with the current LangChain version.

| Component | Model | Cost Change |
|---|---|---|
| Text2Cypher | GPT-5.1 | -50% input, same output |
| LLM Chain | GPT-5.1 | Same |
| Translation | GPT-5 nano | -96% |
| Agent Planner | GPT-5.1 | Same |
| Agent Writer | GPT-5.1 | Same |
| Agent Critic | GPT-5.1 | Same |
| Media Mapping | GPT-5.1 | Same |

**Pros:** No LangChain upgrade needed, <1% hallucinations (5.0), adaptive reasoning (5.1), 400K context (vs 128K), 128K output (vs 4K).  
**Cons:** Not the flagship model, fewer benchmark improvements vs 5.4.

### Option B: Aggressive — GPT-5.4 Tiered (Requires LangChain Upgrade)

Mix GPT-5.4 flagship for quality-critical tasks, GPT-5.4 mini for structured tasks, GPT-5 nano for simple tasks.

| Component | Model | Cost Change |
|---|---|---|
| Text2Cypher | GPT-5.4 mini | -55% |
| LLM Chain | GPT-5.4 | +50% output |
| Translation | GPT-5 nano | -96% |
| Agent Planner | GPT-5.4 mini | -55% |
| Agent Writer | **GPT-5.4** | +50% output |
| Agent Critic | GPT-5.4 mini | -55% |
| Media Mapping | GPT-5.4 mini | -55% |

**Pros:** Best possible quality where it matters, massive cost savings on structured tasks, 1M context for flagship.  
**Cons:** Requires LangChain upgrade (risk), +50% cost on output-heavy tasks.

### Option C: Hybrid — GPT-5.1 + selective GPT-5.4 (Recommended)

Use GPT-5.1 for the GraphRAG pipeline (no LangChain upgrade risk), upgrade to GPT-5.4 only for the Agent pipeline (which uses `openai` directly via `ChatOpenAI` constructor).

| Component | Model | Cost Change | LangChain Upgrade? |
|---|---|---|---|
| Text2Cypher | GPT-5.1 | -50% input | No |
| LLM Chain | GPT-5.1 | Same | No |
| Translation | GPT-5 nano | -96% | No (uses `openai` directly) |
| Agent Planner | GPT-5.4 mini | -55% | Yes (but isolated) |
| Agent Writer | **GPT-5.4** | +50% output | Yes (but isolated) |
| Agent Critic | GPT-5.4 mini | -55% | Yes (but isolated) |
| Media Mapping | GPT-5.4 mini | -55% | No (uses `openai` directly) |

**Pros:** Minimal risk to production GraphRAG, best quality for Agent mode, massive savings on translation and media generation.  
**Cons:** Two different model tiers to maintain, partial LangChain upgrade.

## B7. Risks & Mitigations

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| **LangChain breaking changes** on upgrade | HIGH | Medium | Test in dev branch first. Pin exact version. Option C avoids this for GraphRAG |
| **Cypher generation regression** (different model = different Cypher style) | MEDIUM | Medium | Run evaluation protocol Phase 0 with both models. Compare Cypher accuracy on 15 test queries |
| **Response style change** (GPT-5 may produce differently formatted Italian) | MEDIUM | High | Validate with domain experts (Carla, Paola). Adjust prompts if needed |
| **Latency increase** (GPT-5.4 is 1.8x slower than GPT-4o) | LOW | Certain | Only affects Writer (user-facing). Mini/nano are faster. Users expect quality, not speed |
| **Prompt sensitivity** (GPT-5 may interpret few-shot examples differently) | MEDIUM | Medium | Test all domain-specific prompts. May need to adjust `udl_domain.py` and `neuro_domain.py` examples |
| **Cost spike** if misconfigured | LOW | Low | Tiered approach actually saves money. Set billing alerts |
| **Long context collapse** (GPT-5.4 mini degrades at 64K-256K tokens) | MEDIUM | Low | Only affects mini. Our prompts are typically <30K tokens |
| **Hallucination in citations** (media mapping) | MEDIUM | Medium | Use GPT-5.0 for media mapping (lowest hallucination rate) |

## B8. Execution Plan

### Phase 1: Preparation (1 day)

- [ ] Create a `feature/model-upgrade` branch
- [ ] Audit current `.env` to confirm `OPENAI_MODEL` value
- [ ] Document current response baselines (save 10 Neuro + 10 UDL responses with GPT-4o)

### Phase 2: Low-Risk Quick Wins (30 min)

- [ ] Change `multilingual_text2cypher.py` line 412: `model="gpt-5-nano"`
- [ ] Change `generate_media_mapping.py` default: `default='gpt-5.4-mini'`
- [ ] Test Italian translation with 5 queries
- [ ] Test media mapping with `--limit 3`

### Phase 3: GraphRAG Pipeline Upgrade (2 hours)

- [ ] Change `.env`: `OPENAI_MODEL=gpt-5.1`
- [ ] Run 15 UDL test queries, compare Cypher quality
- [ ] Run 10 Neuro test queries, compare Cypher quality
- [ ] Compare response quality with baseline (Phase 1)
- [ ] If GPT-5.1 works well → keep. If issues → try GPT-5.0

### Phase 4: Agent Pipeline Upgrade (3 hours)

- [ ] Upgrade `langchain-openai` in `requirements.txt`
- [ ] Test LangChain imports and basic functionality
- [ ] Change Planner: `model="gpt-5.4-mini"`
- [ ] Change Writer: `model="gpt-5.4"`
- [ ] Change Critic: `model="gpt-5.4-mini"`
- [ ] Run `python apps/cli/run_agent.py` with 3 queries per domain
- [ ] Compare lesson plan quality with GPT-4o baseline

### Phase 5: Validation & Production (1 day)

- [ ] Run full evaluation protocol (Phase 0 from Protocollo)
- [ ] Domain expert review of 5 responses per domain
- [ ] Merge to main
- [ ] Monitor GlitchTip for new errors
- [ ] Compare costs after 1 week

## B9. Final Recommendation

### Best Overall Strategy: Option C (Hybrid)

**For the GraphRAG pipeline (production, lower risk):**

> **GPT-5.1** — Drop-in replacement for GPT-4o. Same output cost, 50% cheaper input, adaptive reasoning (faster on simple queries), 400K context window, 128K output tokens. No LangChain upgrade required.

**For the Agent pipeline (newer, can tolerate changes):**

> **GPT-5.4** for the Writer agent (quality-critical), **GPT-5.4 mini** for Planner and Critic (structured output tasks). Requires LangChain upgrade but the Agent pipeline is not yet in production.

**For utility tasks:**

> **GPT-5 nano** for Italian translation ($0.05/M input — 96% savings). **GPT-5.4 mini** for media mapping generation.

### Projected Monthly Cost Impact

Assuming ~1,000 queries/day, ~50 agent requests/day:

| Cost Component | Current (GPT-4o) | After Upgrade | Change |
|---|---|---|---|
| GraphRAG queries | ~$X | ~$0.7X (GPT-5.1 cheaper input) | **-30%** |
| Translation | ~$Y | ~$0.04Y (GPT-5 nano) | **-96%** |
| Agent Writer | ~$Z | ~$1.5Z (GPT-5.4 better but costlier) | +50% |
| Agent Planner/Critic | ~$W | ~$0.45W (GPT-5.4 mini) | **-55%** |
| **Net effect** | Baseline | | **~20-30% savings** with better quality |

### Why Not Just GPT-5.4 Everywhere?

1. **LangChain compatibility risk** — GPT-5.4 requires upgrading LangChain, which can break the production GraphRAG pipeline
2. **Cost** — GPT-5.4 output is 50% more expensive; not justified for Cypher generation or translation
3. **Speed** — GPT-5.4 is 1.8x slower than GPT-4o; for real-time GraphRAG queries, latency matters
4. **GPT-5.1 is already a massive upgrade** — 400K context (3x GPT-4o), 128K output (31x GPT-4o), adaptive speed, same cost

### Why Not Just GPT-5.0 Everywhere?

GPT-5.0's <1% hallucination rate is excellent, but:
1. GPT-5.1's adaptive reasoning is faster on simple queries (most of our Text2Cypher calls)
2. GPT-5.1 is the same price as GPT-5.0
3. GPT-5.1 includes all GPT-5.0 improvements plus adaptive speed

**GPT-5.0 is recommended only for `generate_media_mapping.py`** where hallucination accuracy matters most (generating academic citations with DOIs).

---

## Appendix: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMMENDED MODEL MAP                     │
├──────────────────────┬──────────────────┬────────────────────┤
│ Component            │ Model            │ Monthly Savings    │
├──────────────────────┼──────────────────┼────────────────────┤
│ Text2Cypher          │ gpt-5.1          │ -30% (cheaper in)  │
│ LLM Response Chain   │ gpt-5.1          │ -30% (cheaper in)  │
│ Italian Translation  │ gpt-5-nano       │ -96%               │
│ Agent Planner        │ gpt-5.4-mini     │ -55%               │
│ Agent Writer         │ gpt-5.4          │ +50% (worth it)    │
│ Agent Critic         │ gpt-5.4-mini     │ -55%               │
│ Media Mapping        │ gpt-5.4-mini     │ -55%               │
│ Media Mapping (cit.) │ gpt-5.0          │ same (low halluc.) │
└──────────────────────┴──────────────────┴────────────────────┘

Decision date: ____________
Approved by:   ____________
```

---

*Document generated January 27, 2026. Pricing and benchmarks based on publicly available data as of this date. Verify current pricing at https://openai.com/api/pricing/ before implementation.*
