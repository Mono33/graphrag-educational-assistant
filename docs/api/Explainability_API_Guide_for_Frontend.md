# GraphRAG Explainability API — Frontend Integration Guide

**For:** Simone (Frontend / Lovable)
**From:** AI Team (Louis, Angelo)
**Date:** April 2026
**API Version:** 1.1.0+

---

## Overview

The GraphRAG API now includes **explainability data** in every response. This tells the frontend exactly:

- **WHERE** each recommendation came from (which KG node and relationship)
- **HOW** it was found (direct query, graph neighbor, semantic similarity)
- **WHY** it's ranked where it is (scoring breakdown)

All explainability data is **optional** and **additive** — the existing fields (`name`, `category`, `relevance_score`, `confidence`, etc.) are unchanged. You can adopt explainability incrementally.

---

## Quick-Use Fields for Teachers (NEW)

These fields are designed for **immediate display** in the frontend UI — simple Italian text that teachers can understand without any technical background. They complement the detailed `explainability` object described below.

### Per Methodology: `explainability_name` + `explainability_phrase`

**Location:** `context.primary_methodologies[]` and `context.supporting_methodologies[]`

| Field | Type | Description |
|---|---|---|
| `explainability_name` | `string \| null` | Short Italian label (e.g., _"Raccomandazione diretta dal Knowledge Graph"_). Use as a badge/tag. |
| `explainability_phrase` | `string \| null` | Full Italian sentence explaining WHY this methodology is relevant, including confidence. Display as tooltip or subtitle. |

**Example:**
```json
{
  "name": "Mindfulness",
  "confidence": "very_high",
  "explainability_name": "Strategia collegata nel Knowledge Graph",
  "explainability_phrase": "Questa strategia è collegata a «ADHD» che mitiga «Mindfulness» nel Knowledge Graph. Confidenza: molto alta.",
  "explainability": { "..." }
}
```

**Possible values for `explainability_name`:**

| Retrieval stage | Label shown |
|---|---|
| `direct_query` | _Raccomandazione diretta dal Knowledge Graph_ |
| `structural_neighbor` | _Strategia collegata nel Knowledge Graph_ |
| `vector_neighbor` | _Strategia simile individuata dall'AI_ |
| `semantic_search` | _Suggerimento basato su similarità semantica_ |
| `keyword_semantic` | _Trovato per corrispondenza tematica_ |
| fallback | _Basato su conoscenza pedagogica generale_ |

### Top-Level: `context_warning`

**Location:** root of the response, alongside `explainability_summary`

| Field | Type | Description |
|---|---|---|
| `context_warning` | `string \| null` | Italian warning when the query is too vague or the KG lacks specific data. **`null` when results are solid.** |

**When it appears:**
- **No KG data found** → _"Attenzione: il Knowledge Graph non contiene dati specifici per questa richiesta…"_
- **Low confidence** → _"Nota: i risultati hanno una confidenza limitata…"_
- **Solid results** → `null` (field absent or null — nothing to warn about)

**Frontend suggestion:** Show a yellow/orange banner at the top of the results when `context_warning !== null`.

---

## What Changed in the JSON Response

### 1. Each Methodology Now Has an `explainability` Object

**Location:** `context.primary_methodologies[].explainability` and `context.supporting_methodologies[].explainability`

```json
{
  "name": "Difficulty stopping automatic responses",
  "category": "Sfida di Apprendimento",
  "relevance_score": 0.9,
  "evidence_type": "direct_relationship",
  "confidence": "very_high",
  "explainability_name": "Strategia collegata nel Knowledge Graph",
  "explainability_phrase": "Questa strategia è collegata a «Mindfulness» che mitiga «Difficulty stopping automatic responses» nel Knowledge Graph. Confidenza: molto alta.",
  "implementation_guidance": "...",
  "classroom_applications": ["..."],
  "special_considerations": ["..."],

  "explainability": {
    "retrieval_method": "structural_neighbor",
    "hop_distance": 1,
    "graph_path": {
      "source_node": "Mindfulness",
      "source_label": "EmotionRegulationStrategies",
      "relationship": "MITIGATED_BY",
      "target_node": "Difficulty stopping automatic responses",
      "target_label": "LearningChallenge"
    },
    "scoring_breakdown": {
      "base_score": 0.5,
      "semantic_score": null,
      "vector_similarity": null,
      "domain_boost": 1.7,
      "final_rank_score": 1.36
    },
    "reasoning": "Found as structural graph neighbor: Mindfulness -[MITIGATED_BY]-> Difficulty stopping automatic responses (1 hop)"
  }
}
```

### 2. Top-Level `explainability_summary` Object

**Location:** root level of the response, alongside `context`, `metrics`, etc.

```json
{
  "explainability_summary": {
    "embedding_mode": "hybrid_semantic",
    "retrieval_phases": {
      "graph_traversal": { "nodes_found": 48, "time_ms": 4856 },
      "semantic_search": { "nodes_found": 9,  "time_ms": 188 },
      "fusion_ranking":  { "nodes_found": 57, "time_ms": 1 }
    },
    "knowledge_graph_stats": {
      "total_nodes_retrieved": 57,
      "total_relationships": 28,
      "direct_hits": 8,
      "structural_neighbors": 2,
      "semantic_matches": 0,
      "label_distribution": {
        "InstructionalStrategy": 7,
        "EducationalApproach": 10,
        "LearningMethodology": 9,
        "LearningChallenge": 6,
        ...
      }
    },
    "graph_coverage": "This response used 57 concepts from the Knowledge Graph, producing 10 methodology recommendations. 8 found through direct KG relationships (high confidence). 2 through graph neighbor expansion."
  },
  "context_warning": null
}
```

> **Note:** `context_warning` is `null` here because the KG returned plenty of relevant data. When the query is too vague (e.g., "dimmi qualcosa sull'insegnamento"), you'll see an Italian message like:
> `"Attenzione: il Knowledge Graph non contiene dati specifici per questa richiesta. Le raccomandazioni si basano su principi pedagogici generali. Per risultati più mirati, prova a specificare il tipo di studenti o le difficoltà specifiche."`

---

## Field Reference

### Per-Methodology: Quick-Use Fields (Teacher-facing)

| Field | Type | Description | Example | UI Suggestion |
|-------|------|-------------|---------|---------------|
| `explainability_name` | string or null | Short Italian label for the retrieval type | `"Raccomandazione diretta dal Knowledge Graph"` | Badge/tag on each card |
| `explainability_phrase` | string or null | Full Italian sentence explaining WHY this result is relevant, with confidence | `"Questa strategia è collegata a «ADHD» che mitiga «Mindfulness» nel Knowledge Graph. Confidenza: molto alta."` | Tooltip, subtitle, or info panel |

### Per-Methodology: `explainability` (Technical Detail)

| Field | Type | Description | Example | UI Suggestion |
|-------|------|-------------|---------|---------------|
| `retrieval_method` | string | How this was found | `"direct_query"`, `"structural_neighbor"`, `"vector_neighbor"`, `"semantic_search"`, `"keyword_semantic"` | Badge/tag on each card |
| `hop_distance` | int | Graph distance from query | `0` (direct), `1` (neighbor), `2` (semantic) | Color-coded indicator |
| `graph_path` | object or null | KG relationship path | See below | Mini graph visualization |
| `graph_path.source_node` | string | Origin node name | `"Mindfulness"` | Left side of path |
| `graph_path.source_label` | string | Origin node type | `"EmotionRegulationStrategies"` | Label/tag |
| `graph_path.relationship` | string | KG relationship type | `"MITIGATED_BY"`, `"SUGGESTS"`, `"INFLUENCES"` | Arrow label |
| `graph_path.target_node` | string | This methodology's name | `"Difficulty stopping..."` | Right side of path |
| `graph_path.target_label` | string | This methodology's KG type | `"LearningChallenge"` | Label/tag |
| `scoring_breakdown.base_score` | float | Base score by retrieval type | `1.0` (direct), `0.8` (structural), `0.6` (vector), `0.5` (semantic) | Score bar segment |
| `scoring_breakdown.semantic_score` | float or null | OpenAI embedding similarity | `0.82` or `null` | Score bar segment |
| `scoring_breakdown.vector_similarity` | float or null | Node2Vec graph similarity | `0.74` or `null` | Score bar segment |
| `scoring_breakdown.domain_boost` | float | Domain priority multiplier | `1.7` | Multiplier badge |
| `scoring_breakdown.final_rank_score` | float | Final combined score | `1.36` | Overall score display |
| `reasoning` | string | Human-readable explanation | `"Found as structural graph neighbor: Mindfulness -[MITIGATED_BY]-> Difficulty stopping automatic responses (1 hop)"` | **Display as-is** |

### Top-Level: `explainability_summary`

| Field | Type | Description | UI Suggestion |
|-------|------|-------------|---------------|
| `embedding_mode` | string | Active AI retrieval mode | Info badge: "Hybrid Semantic" |
| `retrieval_phases.graph_traversal.nodes_found` | int | Nodes from KG queries | Stats banner |
| `retrieval_phases.graph_traversal.time_ms` | int | KG query time | Performance indicator |
| `retrieval_phases.semantic_search.nodes_found` | int | Nodes from AI similarity | Stats banner |
| `retrieval_phases.semantic_search.time_ms` | int | AI search time | Performance indicator |
| `retrieval_phases.fusion_ranking.nodes_found` | int | Total after merge | Stats banner |
| `knowledge_graph_stats.direct_hits` | int | High-confidence KG matches | Pie chart segment |
| `knowledge_graph_stats.structural_neighbors` | int | Graph neighbor discoveries | Pie chart segment |
| `knowledge_graph_stats.semantic_matches` | int | AI-discovered matches | Pie chart segment |
| `knowledge_graph_stats.label_distribution` | object | Node types distribution | Tag cloud or bar chart |
| `graph_coverage` | string | Human-readable summary | **Display as-is** in a banner |

### Top-Level: `context_warning`

| Field | Type | Description | UI Suggestion |
|-------|------|-------------|---------------|
| `context_warning` | string or null | Italian warning when the KG lacks data or confidence is low. **`null` when everything is fine.** | Yellow/orange banner at the top of results. Only render when `!== null`. |

---

## Recommended UI Components

### 0. Context Warning Banner (when `context_warning !== null`)

```
┌──────────────────────────────────────────────────────────┐
│  ⚠️  Attenzione: il Knowledge Graph non contiene dati     │
│      specifici per questa richiesta. Le raccomandazioni   │
│      si basano su principi pedagogici generali.           │
│      Per risultati più mirati, prova a specificare il     │
│      tipo di studenti o le difficoltà specifiche.         │
└──────────────────────────────────────────────────────────┘
```

### 1. Methodology Card — With Teacher-Facing Explainability

Each card shows the `explainability_name` as a badge and `explainability_phrase` as a subtitle. The detailed `explainability` object powers an expandable section:

```
┌──────────────────────────────────────────────────────────┐
│  📚 Difficulty stopping automatic responses               │
│  Category: Sfida di Apprendimento                         │
│  Confidence: ●●●●● VERY HIGH                             │
│                                                           │
│  [Strategia collegata nel Knowledge Graph]  ← badge       │
│                                                           │
│  "Questa strategia è collegata a «Mindfulness» che        │
│   mitiga «Difficulty stopping automatic responses»        │
│   nel Knowledge Graph. Confidenza: molto alta."           │
│                                              ↑ phrase     │
│                                                           │
│  ▸ Dettaglio tecnico                                      │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ 🔗 structural_neighbor (1 hop)                        │ │
│  │ Mindfulness ──[MITIGATED_BY]──▶ Difficulty...         │ │
│  │ Score: base=0.5 × domain_boost=1.7 = 1.36             │ │
│  └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 2. Retrieval Method Badge

Use `retrieval_method` for a colored badge:

| Value | Display | Color |
|-------|---------|-------|
| `direct_query` | "Direct KG Match" | Green |
| `structural_neighbor` | "Graph Neighbor" | Blue |
| `vector_neighbor` | "Vector Similar" | Purple |
| `semantic_search` | "AI Semantic Match" | Orange |
| `keyword_semantic` | "Keyword Match" | Gray |

### 3. Hop Distance Indicator

Use `hop_distance` for a visual proximity indicator:

| Value | Display | Meaning |
|-------|---------|---------|
| `0` | ● Direct | Found by the KG query directly |
| `1` | ●● 1 Hop | Found through a neighbor in the graph |
| `2` | ●●● 2 Hops | Found via AI embedding similarity |

### 4. Graph Path Visualization (when `graph_path` is not null)

Display as a simple flow:

```
[Mindfulness] ──MITIGATED_BY──▶ [Difficulty stopping automatic responses]
```

When `graph_path` is `null`, show: "Direct match from Knowledge Graph query"

### 5. Global Stats Banner

Use `explainability_summary` for a top-of-page banner:

```
┌─────────────────────────────────────────────────────────────────┐
│  🧠 Knowledge Graph Coverage                                    │
│                                                                  │
│  57 concepts retrieved │ 28 relationships │ 10 recommendations  │
│  8 direct KG hits │ 2 graph neighbors │ 0 semantic matches      │
│                                                                  │
│  Mode: Hybrid Semantic │ Graph: 4.8s │ Semantic: 0.2s           │
│                                                                  │
│  "This response used 57 concepts from the Knowledge Graph,      │
│   producing 10 methodology recommendations. 8 found through     │
│   direct KG relationships (high confidence). 2 through graph    │
│   neighbor expansion."                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 6. Label Distribution (Optional)

Use `knowledge_graph_stats.label_distribution` for a tag cloud or small bar chart showing which types of knowledge were used:

```
EducationalApproach (10) │ LearningMethodology (9) │ InstructionalStrategy (7) │ ...
```

---

## How to Access the Data (Code Examples)

### JavaScript/TypeScript

```typescript
// Parse the API response
const response = await fetch('/api/v1/context', { method: 'POST', body: ... });
const data = await response.json();

// 1. Context warning — show banner if present
if (data.context_warning) {
  showWarningBanner(data.context_warning);
}

// 2. Per-methodology: quick-use Italian fields
data.context.primary_methodologies.forEach(method => {
  console.log(method.explainability_name);   // Badge text (Italian)
  console.log(method.explainability_phrase);  // Tooltip/subtitle (Italian)

  // 3. Detailed explainability (expandable section)
  const ex = method.explainability;
  if (ex) {
    console.log(ex.reasoning);          // Human-readable explanation (English)
    console.log(ex.retrieval_method);   // "direct_query" | "structural_neighbor" | ...
    console.log(ex.hop_distance);       // 0, 1, or 2

    if (ex.graph_path) {
      console.log(`${ex.graph_path.source_node} -[${ex.graph_path.relationship}]-> ${ex.graph_path.target_node}`);
    }

    console.log(ex.scoring_breakdown.final_rank_score);
    console.log(ex.scoring_breakdown.domain_boost);
  }
});

// 4. Global summary
const summary = data.explainability_summary;
console.log(summary.graph_coverage);                           // Display as banner text
console.log(summary.knowledge_graph_stats.direct_hits);        // For pie chart
console.log(summary.knowledge_graph_stats.label_distribution); // For tag cloud
console.log(summary.retrieval_phases.graph_traversal.time_ms); // Performance display
```

### React Component (Conceptual)

```tsx
function ResultsPage({ data }) {
  return (
    <>
      {data.context_warning && (
        <Alert variant="warning">{data.context_warning}</Alert>
      )}
      {data.context.primary_methodologies.map(m => (
        <MethodologyCard key={m.name} method={m} />
      ))}
    </>
  );
}

function MethodologyCard({ method }) {
  const ex = method.explainability;

  return (
    <Card>
      <h3>{method.name}</h3>

      {/* Teacher-facing Italian badge */}
      {method.explainability_name && (
        <Badge color="blue">{method.explainability_name}</Badge>
      )}
      <ConfidenceMeter value={method.confidence} />

      {/* Teacher-facing Italian explanation */}
      {method.explainability_phrase && (
        <p className="subtitle">{method.explainability_phrase}</p>
      )}

      {/* Technical details (expandable) */}
      {ex && (
        <Expandable title="Dettaglio tecnico">
          <p>{ex.reasoning}</p>
          {ex.graph_path && (
            <GraphPath
              from={ex.graph_path.source_node}
              relationship={ex.graph_path.relationship}
              to={ex.graph_path.target_node}
            />
          )}
          <ScoreBreakdown data={ex.scoring_breakdown} />
        </Expandable>
      )}
    </Card>
  );
}
```

---

## Important Notes

1. **`explainability_name` and `explainability_phrase` are always present** — These are the simplest fields to use. They're in Italian and designed for teachers. Use them as badge + subtitle on every methodology card.

2. **`context_warning` is `null` when everything is fine** — Only render the warning banner when `context_warning !== null`. This is the signal that the KG did not have strong data for this specific query.

3. **`explainability` can be `null`** — Always check before accessing. In rare edge cases (fallback paths), the detailed technical object may not be present. The `explainability_name` and `explainability_phrase` fields will still work.

4. **`graph_path` can be `null`** even when `explainability` exists — This happens for `direct_query` results (hop_distance=0). These came directly from the Cypher query, not through graph traversal. Display the `reasoning` text instead.

5. **`semantic_score` and `vector_similarity` can be `null`** — They're only populated when that specific retrieval channel was used. Null means "this signal was not used for this methodology."

6. **The `reasoning` field is always present and human-readable** — This is the easiest English-language field. You can display it as-is without any parsing.

7. **The `graph_coverage` field is always present** — This is a single sentence summarizing the entire retrieval. Perfect for a banner or tooltip.

8. **All these fields are backward compatible** — If you don't use them, nothing breaks. The existing fields (`name`, `category`, `relevance_score`, etc.) work exactly as before.

---

## Quick Start Checklist for Simone

**Fastest wins (start here):**
- [ ] Show `context_warning` as a yellow banner at the top when it's not `null`
- [ ] Show `method.explainability_name` as a badge/tag on each methodology card
- [ ] Show `method.explainability_phrase` as a subtitle or tooltip on each methodology card

**Next level:**
- [ ] Show `explainability_summary.graph_coverage` as a top-of-page summary banner
- [ ] Add a colored badge based on `method.explainability.retrieval_method`
- [ ] Parse `method.explainability.reasoning` and display it in an expandable section
- [ ] (Optional) Add expandable scoring breakdown with `scoring_breakdown` data
- [ ] (Optional) Show graph path visualization when `graph_path` is not null
- [ ] (Optional) Add a pie chart using `knowledge_graph_stats.direct_hits` / `structural_neighbors` / `semantic_matches`
- [ ] (Optional) Show label distribution as tags using `knowledge_graph_stats.label_distribution`
