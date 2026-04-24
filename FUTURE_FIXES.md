# Future Fixes & Technical Debt

## 1. Neo4j Data Quality — MITIGATED_BY Arrow Direction

**Priority**: HIGH  
**Files**: Neo4j source JSON (KG ingestion data)

`MITIGATED_BY` is stored with inverted arrows in Neo4j:
- **Current (wrong)**: `(Mindfulness)-[:MITIGATED_BY]->(Difficulty stopping automatic responses)`
- **Expected**: `(Difficulty stopping automatic responses)-[:MITIGATED_BY]->(Mindfulness)`

The passive voice means the source should be the *thing being mitigated* (the challenge), pointing to the mitigator (the strategy). The code (`startNode(r)` detection) is correct — fix requires identifying all MITIGATED_BY entries in the source JSON, reversing their `from`/`to` node assignments, and re-ingesting.

---

## 2. Missing "Altered reward sensitivity" Edge in Concept Graph

**Priority**: MEDIUM  
**Files**: `graph_retriever.py` — `_extract_triples()`, CASE 4 parsing

`Differentiated Instruction` has `graph_path.source_node = "Altered reward sensitivity"` in its explainability output, but the corresponding edge `Altered reward sensitivity -[SUGGESTS]-> Differentiated Instruction` is absent from `concept_graph.edges`. The other 3 ADHD barrier context nodes appear correctly. Only 18 of 20 max nodes are used, so it is not a capacity issue.

**Investigation needed**: Add debug logging in `_extract_triples()` to verify `triple_source_name` is set correctly for this node after `_expand_neighbors` processes the CASE 4 row for "Altered reward sensitivity".

---

## 3. Schema Model Name Reconciliation (feature/openrouter vs origin/main)

**Priority**: MEDIUM — required before merge  
**Files**: `api/schemas/models.py`, `api/schemas/__init__.py`, all callers

Two parallel implementations with different class names:

| Concept | origin/main (16fe3bb) | feature/openrouter (canonical) |
|---------|----------------------|-------------------------------|
| Graph path | `GraphPathInfo` | `GraphPath` |
| Per-method explainability | `ExplainabilityDetail` | `MethodologyExplainability` |
| Retrieval phase | `RetrievalPhaseInfo` | `RetrievalPhase` |
| KG stats | `KnowledgeGraphStats` | `KGStats` |

feature/openrouter names are canonical (match `JSON_reference.json`). Any frontend code built against origin/main names must be updated before merge.

---

## 4. LLM Model Selection — Ad Hoc Tests Required

**Priority**: HIGH  
Two separate LLM roles need independent optimization.

### 4a. Text2Cypher Model

Current default: `google/gemini-2.0-flash`. Evaluation criteria: Cypher correctness, schema adherence (correct label/relationship names), Italian→English domain term translation, speed.

**Test suite**: 20–30 Italian queries spanning neuro and UDL domains. Include CASE 4 barrier queries, comparison queries, and out-of-scope queries.  
**Candidates**: `google/gemini-2.0-flash`, `anthropic/claude-haiku-4-5`, `openai/gpt-4o-mini`, `deepseek/deepseek-chat`.

### 4b. Lesson Plan Generation Model

Current default: `openai/gpt-4o`. Evaluation criteria: Italian fluency, 4-phase structure compliance, UDL/neuro domain fidelity, specificity of classroom activities.

**Test suite**: 10–15 teacher scenarios (ADHD, Autism, Dyslexia, mixed class). Rubric: structure compliance, specificity of tool suggestions, alignment with KG context, Italian quality.  
**Candidates**: `anthropic/claude-sonnet-4-6`, `openai/gpt-4o`, `google/gemini-2.0-flash-thinking`, `deepseek/deepseek-r1`.

**Note**: Reasoning models (R1, Claude extended thinking) may outperform on lesson plan generation due to multi-step pedagogical reasoning but carry higher latency and cost — test separately and report cost/quality tradeoff.

---

## 5. `graph_path.source_node` Semantic Clarification

**Priority**: LOW  
**Files**: `api/routes/context.py` — `_build_methodology_explainability()`

`graph_path.source_node` stores the **expansion origin** (retrieval traversal start), not necessarily the Neo4j arrow start. For SUGGESTS this coincides; for MITIGATED_BY it may not. After fixing the Neo4j data (Fix #1), verify whether `source_node` in `graph_path` should switch to `triple_source_name` for semantic correctness in the UI.

---

## 6. Integration Test Coverage

**Priority**: MEDIUM — required before merging feature/openrouter → main

| Scenario | Expected outcome |
|----------|-----------------|
| Normal ADHD query (neuro domain) | `context_warning = null` |
| Out-of-scope query ("Qual è il meteo?") | Warning condition 1 fires |
| Low-confidence result | Warning condition 2 fires |
| `include_explainability=false` | `context_warning` still appears in response |
| Concept graph max cap | No dangling edges, ≤20 nodes, ≤30 edges |
| MITIGATED_BY edges (post Neo4j fix) | Correct direction in `concept_graph.edges` |
