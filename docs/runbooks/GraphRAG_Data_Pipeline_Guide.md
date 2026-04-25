# AIxLearning GraphRAG — Data Pipeline & Validation Guide

**Author:** AI Team (Louis, Angelo)
**Last Updated:** April 2026
**Purpose:** Standard procedure for ingesting new domain data into the GraphRAG Knowledge Graph, and validating end-to-end pipeline functionality.

> This guide documents the full process executed for UDL (Universal Design for Learning) and applicable to any future domain (e.g., Game-Based Learning).

---

## Architecture Overview

```
Raw Excel (domain experts)
    │
    ▼
[Step 0] Backup & Clean existing data ──► backup JSON
    │
    ▼
[Step 1] Data Cleaning ──► cleaned Excel
    │
    ▼
[Step 2] Transformation ──► Neo4j-ready JSON (nodes + relationships)
    │
    ▼
[Step 3] Neo4j Ingestion ──► nodes + relationships in graph DB
    │
    ▼
[Step 4] Embedding Training ──► Node2Vec model + OpenAI embeddings cache
    │
    ▼
[Step 5] Domain Configuration ──► domains/[domain]_domain.py
    │
    ▼
[Step 6] Pipeline Integration ──► text2cypher, graph_retriever, multilingual updates
    │
    ▼
[Step 7] End-to-End Validation ──► test queries, API tests, Streamlit verification
```

**Steps 0–4** are standardized and use shared scripts.
**Steps 5–7** are domain-specific and require manual analysis of the new graph.

---

## PART 1 — Standard Data Pipeline (Steps 0–4)

These steps are repeatable for any new domain. The scripts are shared.

---

### Step 0 — Backup & Clean Existing Data (if necessary)

**When:** Only needed if the domain already has data in Neo4j (i.e., replacing old data, not first-time ingestion).

**Script:** Create a temporary `_step0_clean_old_[domain].py` (delete after use).

**Actions:**
1. Export all existing nodes: `MATCH (n {domain: "[domain]"}) RETURN n, labels(n)`
2. Export all existing relationships: `MATCH (a {domain: "[domain]"})-[r]->(b) RETURN a.id, type(r), b.id, properties(r)`
3. Save backup to `[Domain]data/[domain]_backup_[timestamp].json`
4. Delete old data: `MATCH (n {domain: "[domain]"}) DETACH DELETE n`
5. Verify: `MATCH (n {domain: "[domain]"}) RETURN count(n)` should return 0

**Output:** `[Domain]data/[domain]_backup_[timestamp].json`

**UDL example:** Backed up 104 old nodes + 67 relationships, then deleted them.

---

### Step 1 — Data Cleaning

**Script:** `clean_[domain]_data.py` (one per domain, kept in repo for reproducibility)

**Input:** Raw Excel from domain experts (8-column structure)

| Column | Description |
|--------|------------|
| Category A | Category of source concept |
| Concept A | Source concept type/label |
| Value A | Source concept name |
| Relationship | Relationship type (e.g., SUGGESTS, MITIGATED_BY) |
| Value B | Target concept name |
| Concept B | Target concept type/label |
| Category B | Category of target concept |
| Description | Textual description of the relationship |

**Actions (3 phases):**

1. **Phase 1 — SCAN:** Identify all data quality issues
   - Whitespace (leading/trailing spaces)
   - Typos (e.g., `ASSOCIETS_TO` → `ASSOCIATES_TO`, `Knowdlege` → `Knowledge`)
   - Case inconsistencies (e.g., `cooperative Learning` → `Cooperative Learning`)
   - Null values in Description column

2. **Phase 2 — FIX:** Apply corrections
   - Strip whitespace from all object columns
   - Apply typo fixes (defined in `RELATIONSHIP_FIXES`, `CONCEPT_A_FIXES`, etc.)
   - Apply case fixes
   - Auto-generate missing descriptions from related values

3. **Phase 3 — VERIFY:** Confirm all fixes applied, no remaining issues

**Output:** `[Domain]data/KG_[DOMAIN]_FINAL.xlsx`

**UDL example:** Fixed 320 issues across 799 rows.

**How to run:**
```bash
python clean_udl_data.py
# Runs all 3 phases automatically
# Output: UDLdata/KG_UDL_FINAL.xlsx
```

---

### Step 2 — Transformation (Excel → JSON)

**Script:** `transform_team_data.py` (shared across ALL domains)

**Input:** Cleaned Excel from Step 1

**Actions:**
1. Reads the 8-column Excel
2. Extracts unique nodes with labels (from Concept A/B columns) and properties
3. Extracts relationships with types and properties
4. Adds `domain: "[domain]"` property to all nodes
5. Generates PascalCase labels using `sanitize_label()` function

**Output:** `[Domain]data/kg_[domain]_neo4j.json` — contains `{ "nodes": [...], "relationships": [...] }`

**How to run:**
```bash
python transform_team_data.py --team udl --input UDLdata/KG_UDL_FINAL.xlsx
# Output: UDLdata/kg_udl_neo4j.json
```

**UDL example:** Generated 763 nodes + 799 relationships.

**Verification:** Open the JSON and check:
- Node count matches expected
- Labels are PascalCase (e.g., `CooperativeLearning`, not `cooperative learning`)
- All nodes have `domain: "udl"` property
- Relationship types are UPPER_SNAKE_CASE (e.g., `SUGGESTS`, `MITIGATED_BY`)

---

### Step 3 — Neo4j Ingestion

**Script:** `data_ingestion_neo4j.py` (shared across ALL domains)

**Input:** JSON from Step 2

**Actions:**
1. Connects to Neo4j using provided credentials
2. For each node: `MERGE` on `id` property (avoids duplicates)
3. For each relationship: `MERGE` between source and target nodes
4. Reports counts of created/matched nodes and relationships

**How to run:**
```bash
python data_ingestion_neo4j.py \
  --uri "bolt+s://graph.aiforlearning.digital:7687" \
  --user neo4j \
  --password "YOUR_PASSWORD" \
  --file UDLdata/kg_udl_neo4j.json \
  --domain udl
```

**Important:** This script does NOT read from `.env` — all arguments must be passed explicitly on the command line.

**UDL example:** Imported 763 nodes + 799 relationships in ~126 seconds.

**Verification queries (run in Neo4j Browser or via Cypher):**
```cypher
-- Count nodes
MATCH (n {domain: "udl"}) RETURN count(n)

-- Count relationships
MATCH (a {domain: "udl"})-[r]->(b) RETURN count(r)

-- Check label distribution
MATCH (n {domain: "udl"}) RETURN labels(n) as label, count(*) as cnt ORDER BY cnt DESC LIMIT 20

-- Check relationship type distribution
MATCH (a {domain: "udl"})-[r]->(b) RETURN type(r) as rel_type, count(*) as cnt ORDER BY cnt DESC
```

---

### Step 4 — Embedding Retraining

Two embedding systems must be retrained after ingestion:

#### 4a — Node2Vec (structural embeddings)

**Script:** `train_node2vec.py`

**What it does:** Trains graph embeddings that capture structural relationships between nodes (which nodes are connected, how, and through what paths).

**How to run:**
```bash
python train_node2vec.py udl
```

**Output:** `models/udl_node2vec/` directory containing:
- `udl_node2vec_model.pkl` — trained model
- `udl_node2vec_embeddings.npz` — embedding vectors
- `udl_node2vec_config.json` — training configuration

**UDL example:** 763 nodes, 761 edges, 5 epochs.

#### 4b — OpenAI Semantic Embeddings

**Script:** `graph_retriever.py --precompute`

**What it does:** Generates OpenAI `text-embedding-3-small` embeddings for all node names, enabling semantic similarity search.

**How to run:**
```bash
python graph_retriever.py --precompute udl
```

**Output:** `models/embeddings_cache/udl_openai_embeddings.json`

**UDL example:** 763 embeddings generated.

**Note:** This step requires a valid `OPENAI_API_KEY` in `.env` and makes API calls (cost: ~$0.01 for 763 nodes).

---

## PART 2 — Domain Configuration (Step 5)

This is the **variable** part — specific to each domain's graph structure.

---

### Step 5a — Create/Update Domain Config File

**File:** `domains/[domain]_domain.py`

**Reference:** Use `domains/neuro_domain.py` (1,239 lines) as the template.

**Base class:** `domains/base_config.py` — defines all `@abstractmethod` that must be implemented.

**Process:**
1. **Analyze the new graph** — extract label counts, relationship type counts, connectivity patterns
2. **Implement all 19 methods** based on the analysis

| Method | What to configure | How to derive it |
|--------|------------------|------------------|
| `_get_name` | Domain identifier (e.g., `"udl"`) | Fixed |
| `_get_display_name` | Display name for UI | Fixed |
| `_get_icon` | Emoji icon | Fixed |
| `get_node2vec_weights` | Label importance weights | Formula: `1.0 + log10(node_count)*0.3 + log10(connectivity+1)*0.2` |
| `get_valid_methodology_labels` | Whitelist of ALL valid Neo4j labels | Extract from graph: `MATCH (n {domain:"X"}) RETURN DISTINCT labels(n)` |
| `get_label_category_map` | Label → Italian display category | Manual mapping by domain expert or AI |
| `get_retrieval_boosts` | Label and relationship boosts for ranking | Based on conceptual importance + frequency |
| `get_similarity_threshold` | Cosine similarity minimum for semantic search | 0.70–0.80 depending on label diversity |
| `get_few_shot_examples` | Example Cypher queries for Text2Cypher | Write 15–25 examples using ACTUAL labels from the graph |
| `get_cypher_patterns` | Named query pattern templates | 8–12 patterns covering main query types |
| `repair_cypher_query` | Cypher error correction rules | Fix common LLM mistakes: wrong labels, case issues, UNION mismatches |
| `get_italian_terms` | Italian → English search term mapping | Map Italian educational terms to actual node names |
| `get_query_context` | Context hint for LLM | One-line domain description |
| `get_system_prompt` | Main LLM system prompt | Domain expert knowledge, structured as role + rules |
| `get_response_template` | Response structure template | 6–10 section template for response formatting |
| `get_methodology_categories` | Educational methodology definitions | 6–10 categories with descriptions + adaptations |
| `get_special_needs_mapping` | Learner variability profiles | 4–8 profiles with characteristics + recommendations |
| `get_educational_context_type` | Context type identifier | `"inclusive_education"`, `"neuroscience"`, etc. |

**Critical rule:** Every value must come from the actual graph data. No invented labels, no guessed relationships. Run Cypher queries to extract real data.

---

### Step 5b — Validate Domain Config

Run validation checks:

1. **Label existence:** Every label in `get_valid_methodology_labels()` must exist in Neo4j
2. **Few-shot accuracy:** Every label and relationship type in `get_few_shot_examples()` must exist in the graph
3. **Italian terms:** Every English value in `get_italian_terms()` must match an actual node name
4. **Node2Vec coverage:** The top 50 labels by frequency should all have weights in `get_node2vec_weights()`

```cypher
-- Get all labels for validation
MATCH (n {domain: "udl"}) 
UNWIND labels(n) as label 
RETURN DISTINCT label ORDER BY label

-- Get all relationship types
MATCH (a {domain: "udl"})-[r]->(b) 
RETURN DISTINCT type(r) as rel_type ORDER BY rel_type
```

---

## PART 3 — Pipeline Integration (Step 6)

After the domain config is complete, several pipeline files need to be updated to use the new domain config instead of hardcoded values.

---

### Step 6 — Update Pipeline Files

These files contain hardcoded old labels and patterns that must be updated when a new domain is ingested:

| File | What needs updating | Priority |
|------|-------------------|----------|
| `text2cypher.py` | UDL query patterns (lines 217-227), system prompt rules (line 173), repair logic (`_repair_udl_query`), backup few-shot examples, SEN synonyms, contamination labels | CRITICAL |
| `graph_retriever.py` | `expansion_labels` whitelist (lines 462-546) — must include all new domain labels | CRITICAL |
| `multilingual_text2cypher.py` | Italian term dictionaries (lines 50-209) — must match new node names; ideally refactor to call `domain_config.get_italian_terms()` | CRITICAL |
| `api/routes/context.py` | Fallback text when no KG data (lines 126-148) — currently Neuro-biased | MODERATE |
| `context_builder.py` | Hardcoded old methodology categories and special needs mappings (lines 90-273) — dead code if domain config works, but should be cleaned | LOW |

**Long-term goal:** Refactor these files to read ALL domain-specific values from the domain config, eliminating hardcoded labels entirely. This would make adding a new domain a single-file operation (`domains/[domain]_domain.py`).

---

## PART 4 — End-to-End Validation (Step 7)

---

### Step 7a — Build Test Queries

Create 10–15 queries per domain covering:

| Category | Count | What it tests | Example (UDL) |
|----------|-------|--------------|---------------|
| Single concept | 2 | Basic node retrieval | "Cos'è il Cooperative Learning?" |
| Strategy for variability | 2-3 | SUGGESTS relationships | "Quali strategie aiutano studenti con ADHD?" |
| Negative knowledge | 2 | NO_SUGGESTS / IMPAIRS | "Quali approcci sono sconsigliati per la dislessia?" |
| Multi-hop | 2-3 | Path traversal | "Quali strumenti digitali supportano lo scaffolding per l'autismo?" |
| Relational | 2 | MITIGATED_BY, LEADS, etc. | "Come si mitiga una barriera sensoriale?" |
| Out-of-domain | 1-2 | Fallback handling | "Come si calcola un integrale?" |

### Step 7b — API Testing

Start the API locally and test each query:

```bash
# Start API
uvicorn api.main:app --reload --port 8000

# Test query
curl -X POST http://localhost:8000/api/v1/context \
  -H "Content-Type: application/json" \
  -d '{"query": "Quali strategie per studenti con ADHD?", "domain": "udl", "language": "it"}'
```

**For each query, verify:**
- [ ] `cypher_query` is valid and uses correct labels
- [ ] `total_nodes` > 0 (for in-domain queries)
- [ ] `primary_methodologies` contains relevant results
- [ ] `confidence_level` is appropriate (high/medium for in-domain, low for out-of-domain)
- [ ] `processing_time_ms` is reasonable (< 10s)
- [ ] Response text follows the domain's response template structure

### Step 7c — Streamlit Visual Validation (Optional)

```bash
streamlit run streamlit_app.py
```

- Select UDL domain
- Run test queries visually
- Verify methodology cards, confidence display, response formatting

---

## File Reference

| File | Purpose | Shared/Domain-specific |
|------|---------|----------------------|
| `clean_[domain]_data.py` | Data cleaning (Step 1) | Domain-specific (one per domain) |
| `transform_team_data.py` | Excel → JSON transformation (Step 2) | Shared |
| `data_ingestion_neo4j.py` | JSON → Neo4j ingestion (Step 3) | Shared |
| `train_node2vec.py` | Node2Vec embedding training (Step 4a) | Shared |
| `graph_retriever.py --precompute` | OpenAI embedding precomputation (Step 4b) | Shared |
| `domains/[domain]_domain.py` | Domain configuration (Step 5) | Domain-specific (one per domain) |
| `domains/base_config.py` | Abstract base class for domain configs | Shared |
| `domains/__init__.py` | Domain registry and loading | Shared |
| `text2cypher.py` | Natural language → Cypher query generation | Shared (but has hardcoded sections per domain) |
| `multilingual_text2cypher.py` | Italian query translation | Shared (but has hardcoded sections per domain) |
| `graph_retriever.py` | Hybrid retrieval (Node2Vec + semantic) | Shared (but has hardcoded expansion labels) |
| `context_builder.py` | Raw nodes → educational context | Shared |
| `llm_chain.py` | Prompt assembly + LLM call | Shared (clean — reads from domain config) |
| `api/routes/context.py` | API endpoint | Shared |
| `config.py` | Environment configuration | Shared |

---

## UDL Ingestion Log (March 2026)

| Step | Date | Result |
|------|------|--------|
| Step 0: Backup + clean | Mar 13, 2026 | 104 old nodes + 67 rels backed up and deleted |
| Step 1: Data cleaning | Mar 13, 2026 | 320 fixes across 799 rows → `KG_UDL_FINAL.xlsx` |
| Step 2: Transformation | Mar 13, 2026 | 763 nodes + 799 relationships → `kg_udl_neo4j.json` |
| Step 3: Neo4j ingestion | Mar 13, 2026 | All 763 nodes + 799 rels imported (126s) |
| Step 4a: Node2Vec | Mar 13, 2026 | 763 nodes, 761 edges, 5 epochs |
| Step 4b: OpenAI embeddings | Mar 13, 2026 | 763 embeddings precomputed |
| Step 5a: Domain config | Mar 2026 | `domains/udl_domain.py` — 1,250 lines, 19 methods |
| Step 5b: Validation | Mar 2026 | All labels, examples, terms verified |
| Step 6: Pipeline updates | Pending | `text2cypher.py`, `graph_retriever.py`, `multilingual_text2cypher.py` |
| Step 7: E2E validation | Pending | Test queries + API testing |

---

## Graph Statistics

### UDL Domain (March 2026)
- **Nodes:** 763
- **Unique labels:** 271
- **Relationships:** 799
- **Relationship types:** 35
- **Top labels:** Checkpoint (55), AnalogicalTool (36), DigitalTool (29), EducationalApproach (26), BehavioralManifestations (26)
- **Top relationships:** MITIGATED_BY (178), SUGGESTS (148), ASSOCIATES_TO (75), SUPPORTS_BY (73), MENTIONS (55)

### Neuro Domain
- **Nodes:** ~600+
- **Relationships:** 678
- **Relationship types:** SUPPORTS, FACILITATES, INCREASES, DECREASES, REDUCES, IMPAIRS, ENABLES, ENHANCES, LIMITS, AFFECTS, etc.
- **No NO_SUGGESTS relationships** (uses IMPAIRS/REDUCES/LIMITS instead)
