# Media Pool Agent

Builds a **verified, rights-tagged media pool** for the GraphAIxLearning platform using a fully local LLM agent (LM Studio). The agent explores the live Neo4j Knowledge Graph to understand concept relationships, then calls real APIs to find and verify educational media.

**No media URLs are hallucinated.** The LLM generates only search intent; APIs return real URLs; every entry is verified before being stored.

---

## How it works

```
LM Studio (local LLM — tool-calling agent)
    │
    ├─→ query_neo4j(cypher)          ← explore concept relationships in the live KG
    │       ↓ rows from Neo4j
    │
    ├─→ search_youtube(query)         ← YouTube Data API v3, verified embeddable only
    │       ↓ real videoIds + embed URLs
    │
    ├─→ search_semantic_scholar(q)    ← Semantic Scholar, open-access + verified DOI
    │       ↓ real papers
    │
    ├─→ search_wikipedia(topic)       ← Wikipedia REST API, non-disambiguation only
    │       ↓ canonical URL
    │
    └─→ save_to_pool(entry)           ← appended to pool JSON, written to disk per concept

Output: data/media/kg_{domain}_media_pool.json
```

The KG-aware approach generates better queries than naive concept searches:
- Instead of `"scaffolding"` → `"scaffolding techniques impaired inhibitory control executive function classroom"`
- Instead of `"metacognition"` → `"metacognition strategies ADHD self-regulated learning"`

---

## Prerequisites

### Required
- **LM Studio** running with a tool-calling capable model  
  Default: `google/gemma-4-26b-a4b` at `http://127.0.0.1:1234`  
  Any 7B+ instruction model with function-calling support works (Llama 3.1, Mistral, Qwen 2.5)
- **Neo4j** connection (uses the same `NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD` from `.env`)
- Python packages already in the project: `openai`, `neo4j`, `requests`, `python-dotenv`

### Optional (recommended)
- **YouTube Data API key** — free, 10,000 units/day at [Google Cloud Console](https://console.cloud.google.com/)  
  Without it: YouTube search is skipped
- **Semantic Scholar API key** — free, raises limit from 100 to 1,000 req/5min  
  [Request here](https://www.semanticscholar.org/product/api#api-key)

### Environment variables

Add to your `.env`:
```
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=google/gemma-4-26b-a4b
YOUTUBE_API_KEY=AIza...your-key
SEMANTIC_SCHOLAR_API_KEY=your-key   # optional
```

---

## Usage

### Step 1 — Preflight check
Verify all connections before starting:
```bash
python scripts/media_pool/00_preflight.py
```

### Step 2 — Run the agent
```bash
# Full run for the neuro domain (700+ concepts, ~3-4 hours)
python scripts/media_pool/01_run_pool_agent.py --domain neuro

# Full run for UDL domain
python scripts/media_pool/01_run_pool_agent.py --domain udl

# Test with 10 concepts first
python scripts/media_pool/01_run_pool_agent.py --domain neuro --limit 10

# Test a single concept
python scripts/media_pool/01_run_pool_agent.py --domain neuro --concept "Metacognition"

# Resume an interrupted run
python scripts/media_pool/01_run_pool_agent.py --domain neuro --resume
```

The script saves progress after each concept. If interrupted, restart with `--resume`.

### Step 3 — Verify the pool
Re-verify all URLs with HEAD requests:
```bash
python scripts/media_pool/02_verify_pool.py --domain neuro

# Remove broken entries
python scripts/media_pool/02_verify_pool.py --domain neuro --fix
```

---

## Output format

Pool file: `data/media/kg_{domain}_media_pool.json`

```json
{
  "domain": "neuro",
  "generated_by": "google/gemma-4-26b-a4b",
  "generated_date": "2026-05-07",
  "entries": {
    "Metacognition": {
      "videos": [
        {
          "title": "Metacognition in the Classroom",
          "video_id": "REAL_VIDEO_ID",
          "url": "https://youtu.be/REAL_VIDEO_ID",
          "embed_url": "https://www.youtube.com/embed/REAL_VIDEO_ID",
          "channel": "Khan Academy",
          "rights_status": "youtube_embed",
          "verified_date": "2026-05-07",
          "language": "en",
          "graph_context": "ADHD -[SUGGESTS]-> Metacognition"
        }
      ],
      "citations": [
        {
          "title": "Metacognitive Strategies and Academic Achievement",
          "authors": ["Flavell, J.H."],
          "year": 1979,
          "doi": "10.1037/0003-066X.34.10.906",
          "doi_url": "https://doi.org/10.1037/0003-066X.34.10.906",
          "rights_status": "open_access_paper",
          "verified_date": "2026-05-07"
        }
      ],
      "wikipedia": {
        "title": "Metacognition",
        "url": "https://en.wikipedia.org/wiki/Metacognition",
        "rights_status": "cc_by_sa",
        "verified_date": "2026-05-07",
        "language": "en"
      }
    }
  }
}
```

### Rights status values

| Value | Source | Display method |
|---|---|---|
| `youtube_embed` | YouTube public video | `<iframe>` embed |
| `youtube_cc` | YouTube Creative Commons video | `<iframe>` embed + CC attribution |
| `open_access_paper` | Open access paper with verified DOI | Link to doi.org |
| `cc_by_sa` | Wikipedia article | Link + quote with attribution |

---

## Files

```
scripts/media_pool/
├── README.md                 — this file
├── 00_preflight.py           — connection checks
├── 01_run_pool_agent.py      — main: LM Studio tool-calling agent
├── 02_verify_pool.py         — URL re-verification (run anytime)
├── schema.py                 — pool entry dataclasses + JSON I/O helpers
└── tools/
    ├── __init__.py
    ├── neo4j_tool.py         — Cypher query execution
    ├── youtube_tool.py       — YouTube Data API v3 search
    ├── scholar_tool.py       — Semantic Scholar search + DOI verification
    └── wikipedia_tool.py     — Wikipedia REST API summary
```

Runtime files (not in git):
```
data/media/kg_neuro_media_pool.json    — generated pool
data/media/kg_udl_media_pool.json      — generated pool
data/media/checkpoint_neuro.json       — resumable run state
data/media/checkpoint_udl.json         — resumable run state
```

---

## Integration with the agent

Once the pool is generated, `media_lookup.py` reads from `kg_{domain}_media_pool.json` at agent startup. The Writer Agent receives verified media alongside the KG context and can embed YouTube videos or cite open-access papers knowing every URL is real.

The old `kg_{domain}_media_mapping.json` files are kept as fallback during the transition period.
