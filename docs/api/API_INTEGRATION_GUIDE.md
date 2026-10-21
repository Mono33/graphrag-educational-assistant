# 🚀 GraphRAG Educational API - Integration Guide

This document explains how to use the FastAPI module that provides educational context from the GraphRAG knowledge graph.

> **Scope:** only `POST /api/v1/context`, for tools that put Knowledge Graph
> context into their own prompts. **Not needed for the chat**: for the
> conversational agent (documents, sessions, turns, streaming, exports) start
> from [`QUICKSTART.md`](QUICKSTART.md). The explanation fields of a `/context`
> response are in [`Explainability_API_Guide_for_Frontend.md`](Explainability_API_Guide_for_Frontend.md).

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Access](#access)
3. [API Endpoints](#api-endpoints)
4. [Request & Response Format](#request--response-format)
5. [Integration Options](#integration-options)
6. [Code Examples](#code-examples)

---

## Overview

The GraphRAG Educational API provides structured educational context from a neuroscience and pedagogy knowledge graph. It's designed to be integrated with external systems (like the FEM AixLearning agent) to enhance AI-powered educational assistants.

### What It Does

```
Teacher Query → [Translation] → [Cypher Generation] → [Neo4j Retrieval] 
             → [Context Building] → [Structured Response]
```

### Key Features

- 🌍 **Multilingual**: Accepts Italian/English queries
- 🧠 **Neuroscience Domain**: Teaching practices, metacognition, cognitive strategies
- 📚 **UDL Domain**: Universal Design for Learning principles
- 🎯 **Structured Output**: Ready for prompt injection
- 📊 **Metrics**: Retrieval statistics and confidence levels

---

## Access

| | |
| --- | --- |
| Staging | `https://agente.aiforlearning.digital/api/v1` |
| Auth | HTTP Basic: the instance's client name and password, the same as for the chat API ([`QUICKSTART.md`](QUICKSTART.md)) |

Without credentials `/context` answers `401` (since 2026-10-20). `GET /context/domains`
and `GET /health` stay open. The contract of every route is in
[`openapi.json`](openapi.json).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/context` | POST | Get educational context for a query |
| `/api/v1/context/domains` | GET | List available knowledge domains |
| `/api/v1/health` | GET | Health check with dependency status |

---

## Request & Response Format

### POST `/api/v1/context`

#### Request Body

```json
{
  "query": "Come posso introdurre strategie metacognitive nella mia classe?",
  "domain": "neuro",
  "language": "it",
  "include_raw_nodes": false,
  "max_methodologies": 5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ Yes | - | Educational query (Italian or English) |
| `domain` | string | No | `"neuro"` | `"neuro"`, `"udl"`, or `"all"` |
| `language` | string | No | `"it"` | `"it"` or `"en"` |
| `include_raw_nodes` | boolean | No | `false` | Include raw graph node data |
| `max_methodologies` | integer | No | `5` | Max methodologies to return (1-10) |

#### Response (v1.1.0)

```json
{
  "success": true,
  "query_info": {
    "original_query": "Come posso introdurre strategie metacognitive nella mia classe?",
    "translated_query": "Neuroscience query: How can I introduce metacognitive strategies in my classroom?",
    "detected_language": "it",
    "cypher_query": "MATCH (t:TeachingPractices {domain: \"neuro\"})-[r]-(m:Metacognition {domain: \"neuro\"}) RETURN t, type(r), m LIMIT 10"
  },
  "context": {
    "educational_context_type": "neuroscience",
    "student_profile": "Contesto: neuroscience",
    "primary_methodologies": [
      {
        "name": "Sustained Engagement",
        "category": "Educational Methodology",
        "relevance_score": 0.90,
        "evidence_type": "direct_relationship",
        "implementation_guidance": "Apply Sustained Engagement methodology with appropriate adaptations",
        "classroom_applications": ["Implement Sustained Engagement in classroom context"],
        "special_considerations": ["Adapt based on individual student needs"],
        "confidence": "very_high"
      }
    ],
    "supporting_methodologies": ["..."],
    "evidence_summary": "Recommendations based on general pedagogical principles and domain expertise.",
    "implementation_priority": ["Start with Sustained Engagement (high confidence)", "Pilot with a subset of students first"],
    "confidence_level": "high",
    "fallback_strategies": ["Universal Design for Learning (UDL) principles", "Differentiated instruction approaches"]
  },
  "metrics": {
    "total_nodes": 15,
    "total_relationships": 9,
    "context_relevance": null,
    "processing_time_ms": 28000
  },
  "formatted_prompt_section": "## CONTESTO DAL KNOWLEDGE GRAPH NEUROSCIENTIFICO\n\n**Contesto Educativo:** neuroscience\n...",
  "domain_prompt_context": {
    "domain": "neuro",
    "domain_display_name": "Neuro (Neuroscience)",
    "system_prompt": "# RUOLO\nSei un'Esperta di Neurodidattica e progettazione didattica evidence-based...",
    "response_template": "Schema Lezione: Warm-up → I Do → We Do → You Do → Consolidamento...",
    "kg_context_formatted": "## CONTESTO DAL KNOWLEDGE GRAPH NEUROSCIENTIFICO\n..."
  },
  "error": null
}
```

#### New in v1.1.0: `domain_prompt_context`

| Field | Description |
|-------|-------------|
| `domain` | Domain identifier (e.g., `"neuro"`, `"udl"`) |
| `domain_display_name` | Human-readable name (e.g., `"Neuro (Neuroscience)"`) |
| `system_prompt` | Rich domain system prompt (RUOLO, TAG-CLOUD, PRINCIPI, META-REGOLE) |
| `response_template` | Domain-specific output structure (e.g., I Do/We Do/You Do lesson schema) |
| `kg_context_formatted` | KG data formatted in domain-specific structure |

---

## Integration Options

### Option 1: Simple — Use `formatted_prompt_section` (One Block)

The response includes a domain-aware pre-formatted text section ready for prompt injection as a single block.

```python
import requests

API_URL = "https://agente.aiforlearning.digital/api/v1/context"
AUTH = (CLIENT_NAME, CLIENT_PASSWORD)  # from your server's settings

def get_graphrag_context(user_prompt: str, domain: str = "neuro") -> str:
    """Get educational context from GraphRAG API"""
    try:
        response = requests.post(
            API_URL,
            auth=AUTH,
            json={
                "query": user_prompt,
                "domain": domain,
                "language": "it"
            },
            timeout=30
        )
        if response.ok:
            return response.json().get("formatted_prompt_section", "")
    except Exception:
        pass
    return ""

# Usage in your prompt assembler:
graphrag_block = get_graphrag_context(user_query)
final_prompt = base_prompt + "\n\n" + graphrag_block
```

### Option 2: Granular — Inject at `[usa le info dal GraphRag]` Points

Use the structured `context` data to inject KG content at specific points in the production prompt.

```python
response = requests.post(API_URL, auth=AUTH, json={"query": user_query, "domain": "neuro"})
data = response.json()

if data["success"]:
    ctx = data["context"]
    
    # Build methodology text from raw data
    methods_text = "\n".join(
        f"- {m['name']}: {m['implementation_guidance']}"
        for m in ctx["primary_methodologies"]
    )
    
    # Inject at specific points in the production Neuro prompt
    prompt = base_prompt.replace(
        "Modello I Do – We Do – You Do [usa le info dal GraphRag]",
        f"Modello I Do – We Do – You Do\n{methods_text}"
    )
    
    # Add evidence at another injection point
    prompt = prompt.replace(
        "Motivazione intrinseca ed estrinseca [usa le info dal GraphRag]",
        f"Motivazione intrinseca ed estrinseca\n{ctx['evidence_summary']}"
    )
```

### Option 3: Full Alignment — Use `domain_prompt_context` (v1.1.0)

Access the complete domain expertise: system prompt, lesson schema, and pre-formatted KG context.

```python
response = requests.post(API_URL, auth=AUTH, json={"query": user_query, "domain": "neuro"})
data = response.json()

if data["success"]:
    domain_ctx = data["domain_prompt_context"]
    
    # Rich system prompt (RUOLO, TAG-CLOUD, PRINCIPI, META-REGOLE)
    system_prompt = domain_ctx["system_prompt"]
    
    # Lesson schema template (I Do / We Do / You Do)
    response_template = domain_ctx["response_template"]
    
    # KG data already formatted for this domain
    kg_block = domain_ctx["kg_context_formatted"]
    
    # Compose final prompt
    final_prompt = f"{base_prompt}\n\n{kg_block}"
```

### Note on Session Variables

Environment variables like **number of students**, **class level**, **special needs**, **available tech**, and **lesson duration** are session-specific and should continue to be passed directly by your LLM assistant. The GraphRAG API provides complementary **domain knowledge** (neuroscience concepts, pedagogical strategies, evidence). The LLM combines both to generate contextually adapted responses.

---

## Code Examples

### Python (requests)

```python
import requests

API_URL = "https://agente.aiforlearning.digital/api/v1/context"
AUTH = (CLIENT_NAME, CLIENT_PASSWORD)  # from your server's settings

response = requests.post(
    API_URL,
    auth=AUTH,
    json={
        "query": "Come posso introdurre strategie metacognitive nella mia classe?",
        "domain": "neuro",
        "language": "it"
    }
)

if response.ok:
    data = response.json()
    print(data["formatted_prompt_section"])
else:
    print(f"Error: {response.status_code}")
```

### cURL

```bash
curl -X POST "https://agente.aiforlearning.digital/api/v1/context" \
  -u "$CLIENT_NAME:$CLIENT_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Strategie per studenti con ADHD",
    "domain": "neuro",
    "language": "it"
  }'
```

### JavaScript (fetch)

```javascript
// On your server, never in the browser: the credentials stay there.
const response = await fetch('https://agente.aiforlearning.digital/api/v1/context', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Basic ' + Buffer.from(`${CLIENT_NAME}:${CLIENT_PASSWORD}`).toString('base64'),
  },
  body: JSON.stringify({
    query: 'Come gestire la classe?',
    domain: 'neuro',
    language: 'it'
  })
});

const data = await response.json();
console.log(data.formatted_prompt_section);
```
