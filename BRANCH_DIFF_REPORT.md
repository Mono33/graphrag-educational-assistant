# Branch Diff Report: feature/openrouter vs origin/main

**Generated**: 2026-04-17  
**Branch A**: `feature/openrouter` (local, HEAD: e0a2c6d + uncommitted edits)  
**Branch B**: `origin/main` (remote, HEAD: 16fe3bb)

---

## Executive Summary

1. **OpenRouter Migration** (`98f7b3f`): Migrated from OpenAI-only to OpenAI-compatible API via OpenRouter, adding reasoning model support (o1, o3, DeepSeek R1, Claude with thinking). Config system now has `is_reasoning_model()` and `build_completion_kwargs()` to handle API compatibility.

2. **Explainability Framework** (`ce63fe5` + `e0a2c6d`): Comprehensive per-methodology and response-level explainability. Replaces simpler `ExplainabilityDetail` with `MethodologyExplainability` (graph path + scoring breakdown), adds `ExplainabilitySummary` (retrieval phases + KG stats + coverage summary), and `ConceptGraph` for visualization.

3. **Schema Caching & Performance** (`25fa3c1`): Added `_warm_schema()` in API lifespan to pre-populate Text2CypherConverter caches in background threads. Avoids 60+ Neo4j schema-extraction queries on first request.

4. **UDL Domain Expansion** (`52b07dc`): Massively expanded `udl_domain.py` system prompt from ~40 lines to 300+ lines. Added detailed reference tables for digital tools, cognitive processes, learner variability mappings, and class profile decision rules.

5. **API Response Enrichment** (context.py refactor): Complete refactor of context route to build rich explainability data, concept graphs, and domain-aware prompt contexts. Response now includes `explainability_summary`, `concept_graph`, `domain_prompt_context`, and `context_warning` (opt-in via `include_explainability=True`).

**Merge risk**: HIGH in `api/routes/context.py` (700+ line rewrite), `api/schemas/models.py` (schema reorganization), and `config.py` (API client abstraction). Moderate in `domains/udl_domain.py` (system prompt expansion).

---

## File-by-File Analysis

### 1. `config.py` — LLM Model Configuration

| Aspect | origin/main | feature/openrouter | Change |
|--------|------------|-------------------|--------|
| Model default | `gpt-3.5-turbo-instruct` | `openai/gpt-4o` | CHANGED |
| API base URL | Hardcoded OpenAI | `https://openrouter.ai/api/v1` | CHANGED |
| `get_client()` | Does not exist | Returns `OpenAI(api_key, base_url)` | ADDED |
| `get_async_client()` | Does not exist | Returns `AsyncOpenAI(api_key, base_url)` | ADDED |
| `is_reasoning_model()` | Does not exist | Returns True for o1/o3/o4/DeepSeek R1/thinking models | ADDED |
| `build_completion_kwargs()` | Does not exist | Builds model-compatible kwargs | ADDED |
| `Text2CypherConfig.model` | No field | `"google/gemini-2.0-flash"` | ADDED |
| Embedding field name | `openai_embedding_model` | `embedding_model` | RENAMED |
| Env var priority | `OPENAI_API_KEY`, `OPENAI_MODEL` | Prefers `OPENROUTER_API_KEY`, falls back to `OPENAI_API_KEY` | CHANGED |
| `extract_response_content()` | Does not exist | New utility to extract text + log reasoning tokens | ADDED |
| `LLM_MODEL` env var | Not supported | Sets `openai.model` | ADDED |
| `TEXT2CYPHER_MODEL` env var | Not supported | Sets `text2cypher.model`, falls back to `LLM_MODEL` | ADDED |

**Key detail**: `build_completion_kwargs()` auto-adjusts API calls — o-series models use `max_completion_tokens` instead of `max_tokens` and don't support `temperature`.

---

### 2. `api/schemas/models.py` — JSON Response Schema

#### Model Renaming / Reorganization

| Concept | origin/main name | feature/openrouter name |
|---------|-----------------|------------------------|
| Graph path | `GraphPathInfo` | `GraphPath` |
| Per-method explainability | `ExplainabilityDetail` | `MethodologyExplainability` |
| Retrieval phase | `RetrievalPhaseInfo` | `RetrievalPhase` |
| KG stats | `KnowledgeGraphStats` | `KGStats` |
| Concept graph | Not present | `ConceptGraph` |

#### New Models in feature/openrouter Only

- **`ScoringBreakdown`**: `base_score`, `semantic_score`, `vector_similarity`, `domain_boost`, `final_rank_score`
- **`ConceptGraphNode`**: `id`, `label`, `score`, `hop_distance`
- **`ConceptGraphEdge`**: `source`, `target`, `relation`
- **`ConceptGraph`**: `nodes` (max 20), `edges` (max 30) — D3.js/vis.js ready

#### MethodologyInfo Additions (feature/openrouter)
```python
explainability_name: Optional[str]   # Italian UI label (e.g., "Raccomandazione diretta dal Knowledge Graph")
explainability_phrase: Optional[str] # Human-readable Italian sentence
explainability: Optional[MethodologyExplainability]  # Full provenance data
```

#### ContextResponse Additions (feature/openrouter)
```python
formatted_prompt_section: Optional[str]         # Pre-formatted Markdown for prompt injection
domain_prompt_context: Optional[DomainPromptContext]  # System prompt + template + KG data
explainability_summary: Optional[ExplainabilitySummary]  # Response-level KG stats
concept_graph: Optional[ConceptGraph]           # Nodes/edges for visualization
context_warning: Optional[str]                  # Italian warning for low-quality results
```

#### `base_score` difference
- **origin/main**: Variable by source — description says `"graph=1.0, structural=0.8, vector=0.6, semantic=0.5"`
- **feature/openrouter**: Always `0.5` (per `JSON_reference.json` approved spec)

---

### 3. `api/schemas/__init__.py` — Module Exports

**origin/main exports**: `ExplainabilityDetail`, `ExplainabilitySummary`, `GraphPathInfo`, `RetrievalPhaseInfo`, `KnowledgeGraphStats`

**feature/openrouter exports**: `GraphPath`, `ScoringBreakdown`, `MethodologyExplainability`, `RetrievalPhase`, `KGStats`, `ExplainabilitySummary`, `ConceptGraphNode`, `ConceptGraphEdge`, `ConceptGraph`

**Breaking**: Any code importing old names (`ExplainabilityDetail`, `GraphPathInfo`, etc.) will fail.

---

### 4. `api/main.py` — Startup & Schema Warmup

**Added in feature/openrouter** only:

```python
def _warm_schema(domain: str) -> None:
    """Pre-populate Text2CypherConverter schema cache. Avoids 60+ Neo4j queries on first request."""
    ...

# In lifespan():
for domain in ["udl", "neuro"]:
    loop.run_in_executor(None, _warm_schema, domain)
logger.info("🔥 Schema cache warm-up started for: udl, neuro")
```

origin/main has no warmup — first request always pays the schema-extraction cost.

---

### 5. `api/routes/context.py` — Main Context Endpoint

#### New helper functions in feature/openrouter

| Function | Purpose |
|----------|---------|
| `_build_methodology_explainability()` | Per-methodology explainability: hop-distance logic, Italian names/phrases, scoring |
| `_build_explainability_summary()` | Response-level: retrieval phases, KG stats, pre-cap totals, coverage sentence |
| `_build_concept_graph()` | Top-scored nodes + real relationships for D3.js visualization |
| `_build_context_warning()` | Italian warning when KG lacks data or confidence is low |
| `_format_student_profile()` | Format student profile for prompt section |
| `_format_methodologies()` | Convert context_builder methodologies to API MethodologyInfo |
| `_get_domain_title()` | Domain-aware title (Neuroscientifico, UDL, etc.) |
| `_format_prompt_section()` | Full Markdown-formatted prompt context (Italian/English) |
| `_build_domain_prompt_context()` | DomainPromptContext with system prompt + template + KG data |

#### context_warning logic comparison

**origin/main** (`_build_context_warning` function):
```python
def _build_context_warning(kg_data_available, overall_confidence, methodologies_count):
    if not kg_data_available or methodologies_count == 0:
        return "Attenzione: il Knowledge Graph non contiene dati specifici..."
    if overall_confidence in ('very_low', 'low'):
        return "Nota: i risultati hanno una confidenza limitata..."
    return None
```

**feature/openrouter** (same function, identical implementation — adopted from origin/main):
```python
def _build_context_warning(kg_data_available, overall_confidence, methodologies_count):
    if not kg_data_available or methodologies_count == 0:
        return "Attenzione: il Knowledge Graph non contiene dati specifici..."
    if overall_confidence in ("very_low", "low"):
        return "Nota: i risultati hanno una confidenza limitata..."
    return None
```

**Key structural difference**: In feature/openrouter, this function is called **outside** the `include_explainability` block, so the warning fires even when `include_explainability=False`. In origin/main it was computed after the response body but before the return.

#### `kg_data_available` ordering
- **origin/main**: Computed just before building the response
- **feature/openrouter**: Computed **before** the explainability block so it's available to the warning logic

#### explainability_phrase confidence inclusion
Both branches include `f"Confidenza: {conf_it}."` at the end of every phrase — **identical behavior**.

#### pre-cap vs post-cap totals (feature/openrouter only)
- `total_nodes_retrieved` uses `graph_count + semantic_count` from metadata (e.g., 56)
- NOT `len(nodes)` post-cap (e.g., 15)
- origin/main does not have this distinction

#### graph_coverage language
- **origin/main**: Italian — "Questa risposta ha utilizzato X concetti..."
- **feature/openrouter**: English — "This response used X concepts from the Knowledge Graph..."

#### Concept graph (feature/openrouter only)
- Top-20 nodes by rank_score, normalized score 0-1
- Real Neo4j edges where at least one endpoint is in top-20 (no synthetic VECTOR_SIMILAR edges)
- Auto-adds "context nodes" for missing edge endpoints
- Prevents dangling edges when max_nodes cap is reached

---

### 6. `graph_retriever.py` — Hybrid Retrieval Engine

#### New Class: `SemanticEmbedder` (feature/openrouter only)

```python
class SemanticEmbedder:
    """OpenAI-based semantic embedder for hybrid retrieval.
    - Uses OpenRouter or OpenAI text-embedding-3-small
    - Caches embeddings to disk (JSON)
    - Batch embedding support (100 texts/batch)
    - Lazy-loads OpenRouter client via config.openai.get_client()
    """
```

#### Embedding modes (feature/openrouter)
- `"node2vec"`: Graph structure only (backward compatible with origin/main)
- `"hybrid_semantic"`: Node2Vec (40%) + OpenAI embeddings (60%)
- `"openai_only"`: Pure semantic embeddings

#### hop_distance fix (feature/openrouter only)
CASE 4 Cypher parsing: nodes with pre-existing `rel_type`/`source_node` are correctly assigned `hop_distance=1` (structural_neighbor). origin/main unconditionally assigns `hop_distance=0`.

#### Edge direction fix (feature/openrouter only)
`_get_educational_neighbors` uses `startNode(r).name` to detect actual Neo4j arrow direction and populates `triple_source_name`/`triple_target_name` for correct edge direction in concept graph and triples.

---

### 7. `context_builder.py` — Educational Context Assembly

#### Dynamic balancing (feature/openrouter only)
- `_detect_query_intent()`: Detects comparison keywords ("differenza", "vs", etc.)
- `_apply_dynamic_balancing()`: Balances nodes across label groups
- `_interleave_nodes_by_label()`: Ensures mixed representation in primary/supporting split

#### `EducationalContext.metadata` (feature/openrouter only)
New field stores pre-cap totals (`graph_count`, `semantic_count`), timing info, and embedding mode — used by `_build_explainability_summary()`.

---

### 8. `domains/udl_domain.py` — UDL System Prompt

**origin/main**: ~40 lines (basic role + 3 UDL principles)

**feature/openrouter**: ~300+ lines adding:
- TAG-CLOUD with priority keywords
- Methodology ↔ Tool mappings (Cooperative Learning → Padlet, Gamification → Kahoot, etc.)
- Catalog of 40+ digital tools organized by function
- Tool ↔ Cognitive Process mappings (Memoria/Recall → Kahoot, Metacognizione → Notion, etc.)
- Per-condition learner variability (ADHD, Autism, Dyslexia, Dyscalculia, Gifted, Sensory) with SUGGERITO/NON SUGGERITO
- Class profile decision rules (by size ≤15, ≤20, ≤30; by characteristics: motivated, disruptive, gifted)
- 12 methodology selection rules ("Se la classe ha uno studente con ADHD → suggerisci game-based learning")

---

### 9. `generate_media_mapping.py` — Media Generation Script

| Aspect | origin/main | feature/openrouter |
|--------|------------|-------------------|
| Default model | Hardcoded `"gpt-4o"` | `_DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o")` |
| API key | `OPENAI_API_KEY` only | `OPENROUTER_API_KEY` → fallback `OPENAI_API_KEY` |
| Client | `AsyncOpenAI(api_key=api_key)` | `AsyncOpenAI(api_key=api_key, base_url=openrouter_url if use_openrouter else None)` |

---

### 10. `multilingual_text2cypher.py`, `text2cypher.py`, `llm_chain.py`, agent files

All updated to use `config.openai.get_client()` / `get_async_client()` and `build_completion_kwargs()` instead of direct OpenAI SDK calls. Reasoning model support transparent via `is_reasoning_model()`.

---

## Schema / JSON Response Differences

### ContextResponse fields

| Field | origin/main | feature/openrouter | Notes |
|-------|------------|-------------------|-------|
| `success` | ✓ | ✓ | Unchanged |
| `query_info` | ✓ | ✓ | Unchanged |
| `context` | ✓ | ✓ | Unchanged |
| `raw_nodes` | ✓ | ✓ | Unchanged — identical `RawNode` structure |
| `metrics` | ✓ | ✓ | Unchanged |
| `formatted_prompt_section` | ✓ | ✓ | Both have it |
| `domain_prompt_context` | ✓ | ✓ | Both have it |
| `explainability_summary` | Simplified | Detailed | REWRITTEN — per-phase breakdown |
| `concept_graph` | ✗ | ✓ | NEW in feature/openrouter |
| `context_warning` | ✓ | ✓ | Both have it — same logic |
| `error` | ✓ | ✓ | Unchanged |

### Explainability structure comparison

**origin/main** `ExplainabilitySummary`:
```json
{
  "embedding_mode": "node2vec",
  "retrieval_phases": {
    "graph_traversal": {"nodes_found": 10, "time_ms": 150},
    "semantic_search": {"nodes_found": 5, "time_ms": 200},
    "fusion_ranking": {"nodes_found": 15, "time_ms": 50}
  },
  "knowledge_graph_stats": {
    "total_nodes_retrieved": 15,
    "total_relationships": 29,
    "direct_hits": 5,
    "structural_neighbors": 5,
    "semantic_matches": 5,
    "label_distribution": {"TeachingPractices": 8}
  },
  "graph_coverage": "Questa risposta ha utilizzato 15 concetti..."
}
```

**feature/openrouter** `ExplainabilitySummary`:
```json
{
  "embedding_mode": "hybrid_semantic",
  "retrieval_phases": {
    "graph_traversal": {"nodes_found": 47, "time_ms": 150},
    "semantic_search": {"nodes_found": 9, "time_ms": 200},
    "fusion_ranking": {"nodes_found": 56, "time_ms": 50}
  },
  "knowledge_graph_stats": {
    "total_nodes_retrieved": 56,
    "total_relationships": 29,
    "direct_hits": 0,
    "structural_neighbors": 15,
    "semantic_matches": 0,
    "label_distribution": {"TeachingPractices": 8}
  },
  "graph_coverage": "This response used 56 concepts from the Knowledge Graph, producing 10 methodology recommendations..."
}
```

Key differences: pre-cap totals (56 vs 15), English graph_coverage, structural_neighbors correctly counted.

---

## Architecture Differences

### 1. LLM Provider Abstraction

| | origin/main | feature/openrouter |
|-|------------|-------------------|
| Provider | OpenAI only | OpenRouter (any provider) |
| Reasoning models | No support | Automatic detection + compatible kwargs |
| Model switching | Restart required | Env var at runtime |
| Default model | `gpt-3.5-turbo-instruct` | `openai/gpt-4o` via OpenRouter |

### 2. Hybrid Retrieval

| | origin/main | feature/openrouter |
|-|------------|-------------------|
| Mode | node2vec only | node2vec / hybrid_semantic / openai_only |
| Semantic embeddings | None | OpenRouter text-embedding-3-small |
| Embedding cache | None | Disk cache (`models/embeddings_cache/`) |
| Score fusion | N/A | `0.4 × node2vec + 0.6 × semantic` |

### 3. Explainability Depth

| | origin/main | feature/openrouter |
|-|------------|-------------------|
| Per-methodology | name + phrase | name + phrase + hop_distance + graph_path + scoring_breakdown + reasoning |
| Response-level | basic stats | retrieval phases + pre-cap KG stats + label_distribution + concept_graph |
| Visualization | None | ConceptGraph (D3.js/vis.js ready) |
| Scoring | N/A | base_score + semantic + vector + domain_boost + final_rank_score |

### 4. context_warning Firing Point

| | origin/main | feature/openrouter |
|-|------------|-------------------|
| Fires when `include_explainability=False` | Yes (computed after explainability block) | Yes (moved outside block in latest edits) |
| Conditions | `not kg_data_available` OR `low/very_low confidence` | Same (adopted from origin/main) |
| Messages | Italian, 2 distinct messages | Identical Italian messages |

### 5. CASE 4 hop_distance

| | origin/main | feature/openrouter |
|-|------------|-------------------|
| CASE 4 nodes | All `hop_distance=0` (bug) | `hop_distance=1` (structural_neighbor) |
| Edge direction | Not detected | `startNode(r).name` comparison |

---

## Risk Assessment

### HIGH RISK — api/routes/context.py
- feature/openrouter: ~950 lines vs origin/main: ~500 lines
- Complete rewrite — manual merge required if origin/main has critical bug fixes not in feature/openrouter
- **Recommendation**: Use feature/openrouter as base; audit origin/main for unique fixes

### HIGH RISK — api/schemas/models.py
- Model names changed (`ExplainabilityDetail` → `MethodologyExplainability`, etc.)
- Old code importing these names will fail
- **Recommendation**: Use feature/openrouter; update all import statements

### HIGH RISK — api/schemas/__init__.py
- Old export names removed
- **Recommendation**: Use feature/openrouter version

### MODERATE RISK — config.py
- `openai_embedding_model` renamed to `embedding_model`
- **Recommendation**: Search and replace in all callers

### MODERATE RISK — domains/udl_domain.py
- Massive expansion — possible divergence if origin/main applied independent fixes
- **Recommendation**: Use feature/openrouter content; diff against origin/main commit 52b07dc

### LOW RISK — All other files
- Config alignment changes only, backward compatible

---

## Summary Table

| File | Δ Lines | Change Type | Risk | API Backward Compat |
|------|---------|-------------|------|---------------------|
| `config.py` | +119 | API client abstraction | MODERATE | BROKEN (env var, field rename) |
| `api/schemas/models.py` | +222 | Schema reorganization | HIGH | OK (fields Optional) |
| `api/schemas/__init__.py` | +29 | Export reorganization | HIGH | BROKEN (import names) |
| `api/main.py` | +40 | Schema warmup | LOW | OK |
| `api/routes/context.py` | +669 | Complete refactor | HIGH | OK (all new fields Optional) |
| `graph_retriever.py` | +237 | Semantic embedder + hop fix | LOW | OK |
| `context_builder.py` | +97 | Smart ranking + metadata | LOW | OK |
| `domains/udl_domain.py` | +400 | System prompt expansion | MODERATE | OK |
| `multilingual_text2cypher.py` | +73 | Config alignment | LOW | OK |
| `text2cypher.py` | +61 | Config alignment | LOW | OK |
| `generate_media_mapping.py` | +23 | OpenRouter migration | LOW | OK |
| agent/* + streamlit | ~90 | Config alignment | LOW | OK |
| **TOTAL** | **~2060** | **Multi-layer feature release** | **HIGH** | **REQUIRES_MANUAL_MERGE** |

---

## Key Takeaways

**feature/openrouter is a strict superset of origin/main in terms of API response data** — all existing fields are preserved. New fields are all `Optional` and default to `null`.

**The main reconciliation needed** is schema naming: the two branches developed parallel implementations of explainability with different class names. The feature/openrouter names should be canonical (they match `JSON_reference.json`).

**No functional regression** exists in feature/openrouter relative to origin/main for the 5 concerns reviewed:
1. `context_warning` — identical logic, now also fires when `include_explainability=False`
2. `raw_nodes` metadata — identical structure
3. Schema warmup — added only in feature/openrouter (improvement)
4. `LLM_MODEL`/`TEXT2CYPHER_MODEL` split — added only in feature/openrouter (improvement)
5. `generate_media_mapping.py` — now migrated to OpenRouter in feature/openrouter
