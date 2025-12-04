# 🚀 GraphRAG Educational API - Integration Guide

This document explains how to use the FastAPI module that provides educational context from the GraphRAG knowledge graph.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [API Endpoints](#api-endpoints)
4. [Request & Response Format](#request--response-format)
5. [Integration Options](#integration-options)
6. [Code Examples](#code-examples)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)

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

## Quick Start

### 1. Install Dependencies

```bash
cd graphaixlearning
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file with:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Test It

Open browser: http://localhost:8000/docs

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/context` | POST | Get educational context for a query |
| `/api/v1/context/domains` | GET | List available knowledge domains |
| `/api/v1/health` | GET | Health check with dependency status |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

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

#### Response

```json
{
  "success": true,
  "query_info": {
    "original_query": "Come posso introdurre strategie metacognitive nella mia classe?",
    "translated_query": "Neuroscience query: How can I introduce metacognitive strategies in my classroom?",
    "detected_language": "it",
    "cypher_query": "MATCH (t:TeachingPractices {domain: \"neuro\"})-[r]-(m:Metacognition {domain: \"neuro\"}) RETURN t, type(r), m, labels(t) as source_labels, labels(m) as target_labels LIMIT 10"
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
    "supporting_methodologies": [...],
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
  "error": null
}
```

---

## Integration Options

### Option 1: Use `formatted_prompt_section` Directly (Simplest)

The response includes a pre-formatted text section ready for prompt injection.

```python
import requests

def get_graphrag_context(user_prompt: str) -> str:
    """Get educational context from GraphRAG API"""
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/context",
            json={
                "query": user_prompt,
                "domain": "neuro",
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
prompt = your_existing_prompt
prompt += "\n\n" + get_graphrag_context(user_query)
prompt += format_user_prompt(user_query)
```

### Option 2: Use Structured Data (Full Control)

Access individual fields to build your own formatted context.

```python
response = requests.post(API_URL, json={"query": user_query, ...})
data = response.json()

# Access structured data
methodologies = data["context"]["primary_methodologies"]
evidence = data["context"]["evidence_summary"]
confidence = data["context"]["confidence_level"]
priorities = data["context"]["implementation_priority"]

# Build your own format
for method in methodologies:
    print(f"- {method['name']}: {method['implementation_guidance']}")
```

### Option 3: Context-Aware Integration (Best Results)

Enrich the query with student profile information you already have.

```python
def get_graphrag_context(user_prompt: str, student_disabilities: list = None) -> str:
    """Get educational context with student profile enrichment"""
    query = user_prompt
    
    # Enrich query with student context
    if student_disabilities:
        needs = ", ".join(student_disabilities)
        query = f"{user_prompt} (studenti con: {needs})"
    
    response = requests.post(
        "http://localhost:8000/api/v1/context",
        json={"query": query, "domain": "neuro", "language": "it"},
        timeout=30
    )
    
    if response.ok:
        return response.json().get("formatted_prompt_section", "")
    return ""

# Usage:
context = get_graphrag_context(
    "Come spiegare le frazioni?",
    student_disabilities=["ADHD", "Dislessia"]
)
# Returns methodologies tailored for ADHD + Dyslexia students
```

---

## Code Examples

### Python (requests)

```python
import requests

API_URL = "http://localhost:8000/api/v1/context"

response = requests.post(
    API_URL,
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
curl -X POST "http://localhost:8000/api/v1/context" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Strategie per studenti con ADHD",
    "domain": "neuro",
    "language": "it"
  }'
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/api/v1/context', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'Come gestire la classe?',
    domain: 'neuro',
    language: 'it'
  })
});

const data = await response.json();
console.log(data.formatted_prompt_section);
```

---

## Deployment

### Local Development

```bash
uvicorn api.main:app --reload --port 8000
```

### Production (with Gunicorn)

```bash
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t graphrag-api .
docker run -p 8000:8000 --env-file .env graphrag-api
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'api'` | Run from `graphaixlearning/` directory |
| `Neo4j connection failed` | Check `.env` credentials and Neo4j is running |
| `OpenAI API error` | Verify `OPENAI_API_KEY` in `.env` |
| `Timeout on queries` | First query loads Node2Vec model (~5s), subsequent faster |
| `0 relationships returned` | Normal for some queries; check logs for details |

### Checking Health

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "neo4j_connected": true,
  "version": "1.0.0",
  "domain_configs_loaded": ["neuro", "udl"]
}
```

### Viewing Logs

The API logs detailed information:
```
INFO: Processing context request: domain=neuro, query=...
INFO: [P1 Filter] Structural+Vector neighbors: 37 → Filtered: 9
INFO: Context generated successfully in 21059ms
```

---

## API Module Structure

```
graphaixlearning/api/
├── __init__.py              # Package initialization
├── main.py                  # FastAPI app entry point
├── graphrag_client.py       # Helper client for external integration
├── routes/
│   ├── __init__.py
│   └── context.py           # Main /context endpoint
├── schemas/
│   ├── __init__.py
│   └── models.py            # Pydantic request/response models
└── templates/
    └── graphrag_prompt.txt  # Example Jinja2 template
```

---

## Support

For questions or issues:
- Check the Swagger docs: http://localhost:8000/docs
- Review logs for detailed error messages
- Contact the AI Team

---

*Last updated: December 2024*

