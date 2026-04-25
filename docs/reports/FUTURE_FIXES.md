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

---

## 7. UDL KG Coverage Gap — Framework Taxonomy Not in Graph

**Priority**: MEDIUM  
**Files**: Neo4j UDL source JSON (UDL ingestion data)

The UDL framework's own structural taxonomy (3 Principles → Guidelines → Checkpoints) is not stored as graph nodes. It exists only as prose in the system prompt. Queries that reference UDL principles by name produce Cypher with labels like `Principle`, `Guideline`, `Checkpoint` and relationships like `ALIGNS_TO`, `MENTIONS` — none of which exist in the UDL KG schema — and return 0 results with a transparent fallback.

**Observed in diagnostic test (2026-04-24):**
- Query: `"Come posso offrire molteplici modalità di rappresentazione dei contenuti"`
- Generated Cypher: `MATCH (p:Principle {domain: "udl", name: "Representation"})<-[:ALIGNS_TO]-(g:Guideline {domain: "udl"})`
- Result: 0 nodes, `context_warning` fires correctly

**Fix**: Design and ingest a Principle→Guideline→Checkpoint node hierarchy into the UDL Neo4j KG. Suggested relationships: `(Guideline)-[:BELONGS_TO]->(Principle)`, `(Checkpoint)-[:BELONGS_TO]->(Guideline)`. Each Checkpoint should link to the teaching strategies and barriers already in the KG (e.g. `(Checkpoint)-[:ADDRESSED_BY]->(TeachingStrategy)`). This would allow framework queries to traverse into existing strategy/barrier nodes rather than hitting 0 results.

**Note**: Until this is fixed, the transparent fallback (context_warning + 0 results) is the correct behavior — it does not silently return wrong results.

---

## 8. UDL Data Quality — `Public Error Correction` Inverted Relationship Type (Dyslexia)

**Priority**: HIGH  
**Files**: Neo4j UDL source JSON (Dyslexia ingestion data)

`Public error correction` is stored with an inverted relationship type in the Dyslexia KG:
- **Current (wrong)**: `(Risk of reduced self-efficacy in academics:Dyslexia) -[:SUGGESTS]-> (Public error correction)`
- **Expected**: `(Risk of reduced self-efficacy in academics:Dyslexia) -[:NO_SUGGESTS]-> (Public error correction)`

The system prompt explicitly lists this strategy under `NON SUGGERITO` for the self-efficacy risk barrier. However, `context_builder` accepts any `SUGGESTS`-linked node as a positive methodology recommendation, so the bug causes an actively harmful strategy (public correction of dyslexic students' errors) to surface as a recommended teaching approach.

This is the same class of inverted-relationship error as Fix #1 (MITIGATED_BY in neuro domain).

**Observed in diagnostic test (2026-04-24):** Query A UDL (`"Strategie per studenti con dislessia"`) — `Public error correction` appeared in methodology results linked via `SUGGESTS` from `Risk of reduced self-efficacy in academics`.

**Fix**: In the Dyslexia source JSON, change the relationship type from `SUGGESTS` to `NO_SUGGESTS` for `Public error correction`. Re-ingest. Also audit all other Dyslexia barrier nodes for similarly inverted NO_SUGGESTS strategies (check `Unguided independent reading`, `Peer learning` — these were correctly stored as NO_SUGGESTS in the same query run, but a systematic audit is warranted).
