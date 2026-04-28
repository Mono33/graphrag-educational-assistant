# 🎓 GraphRAG Educational Assistant

AI-powered educational assistant that combines Knowledge Graphs (Neo4j) with Retrieval-Augmented Generation (RAG) to provide personalized teaching recommendations in Italian.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Neo4j](https://img.shields.io/badge/neo4j-5.0+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)
![htmx](https://img.shields.io/badge/htmx-2.0-3366CC.svg)
![MCP](https://img.shields.io/badge/MCP-FastMCP_3.x-blueviolet.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple.svg)
![OpenAI](https://img.shields.io/badge/openai-GPT--4o-orange.svg)
![Node2Vec](https://img.shields.io/badge/Node2Vec-Enabled-brightgreen.svg)

---

## 🚀 **Quick Start**

**New to this project? Start here!** → [**QUICKSTART.md**](QUICKSTART.md)

Get up and running in 5 minutes with our step-by-step guide.

---

## 🌟 Features

- **🌍 Multilingual Query Processing**: Handles Italian and English educational queries with intelligent translation
- **🔍 Hybrid Graph Retrieval**: Combines traditional graph traversal with Node2Vec semantic search for superior results
- **🎯 Educational Context Building**: Structures recommendations specifically for teaching methodologies
- **💬 Natural Language Generation**: Produces natural Italian responses tailored for educators
- **📊 Evidence-Based Responses**: All recommendations backed by knowledge graph evidence
- **🤖 Agentic GraphRAG Mode**: Multi-agent pipeline for intelligent lesson plan generation
- **🎯 Intelligent Intent Detection**: Automatically detects 7 query types (lesson, definition, comparison, etc.)
- **✍️ Adaptive Content Generation**: Generates different formats based on query intent
- **🔄 Quality Control Loop**: Critic agent reviews and requests revisions automatically
- **🖥️ Teacher Webui** 🆕: Authenticated three-pane chat workspace (FastAPI + htmx + WebAwesome + Tailwind) with SSE streaming, file uploads, and educational profile editing
- **🔌 Public Agent API** 🆕: `POST /api/v1/agent/run` (JSON) + `POST /api/v1/agent/stream` (SSE) with JWT Bearer auth and Swagger UI examples
- **🧩 MCP Tool Servers** 🆕: 10 tools / 4 resources / 2 prompts exposed via Model Context Protocol (stdio + Streamable HTTP) — works with Claude Desktop, Cursor IDE, MCP Inspector, and any MCP-compatible client

---

## 🏗️ Architecture

### Standard GraphRAG Mode

```
User Query (Italian/English)
    ↓
Text2Cypher Converter (Multilingual)
    ↓
Hybrid Graph Retriever (Neo4j + Node2Vec)
    ├── Direct Graph Traversal
    ├── Semantic Search (Node2Vec)
    └── Neighbor Expansion
    ↓
Context Builder (Structured Educational Context)
    ↓
Response Generator (GPT-4o LLM)
    ↓
Natural Italian Response
```

### 🤖 Agentic GraphRAG Mode (NEW)

Multi-agent pipeline powered by **LangGraph** for intelligent lesson plan generation:

```
User Query (Italian/English)
    ↓
┌────────────────────────────────────────────────────────┐
│                  AGENTIC PIPELINE                       │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                 │
│  │   PLANNER    │ →  │  RETRIEVER   │                 │
│  │ Intent Detection│  │ GraphRAG Search│                │
│  │ Query Analysis │  │ Knowledge Fetch│                │
│  └──────────────┘    └──────────────┘                 │
│         ↓                   ↓                          │
│  ┌──────────────┐    ┌──────────────┐                 │
│  │    WRITER    │ ←→ │   CRITIC     │ ←┐              │
│  │Content Generator│ │Quality Review │  │ Revision    │
│  │Adaptive Format │  │Score & Decide │  │ Loop        │
│  └──────────────┘    └──────────────┘──┘              │
└────────────────────────────────────────────────────────┘
    ↓
Generated Content (Lesson Plan / Definition / Comparison / etc.)
```

#### Query Intent Detection

The Planner Agent automatically classifies queries into 7 intent types:

| Intent | Trigger Examples | Output Format |
|--------|------------------|---------------|
| `lesson_creation` | "Crea una lezione sulla memoria" | Full lesson plan |
| `activity_design` | "Attività di 30 min sulla metacognizione" | Structured activity |
| `definition` | "Cos'è la neuroplasticità?" | Clear definition |
| `comparison` | "Differenza tra memoria procedurale e dichiarativa" | Comparison table |
| `explanation` | "Come funziona l'attenzione selettiva?" | Detailed explanation |
| `recommendation` | "Quali strategie per studenti con ADHD?" | Strategy list |
| `list` | "Elenca i tipi di memoria" | Enumerated list |

### Core Components:

1. **Text2Cypher** (`text2cypher.py`, `multilingual_text2cypher.py`)
   - Converts natural language queries to Cypher
   - Supports Italian and English with intelligent translation
   - Self-repairing Cypher syntax

2. **Graph Retriever** (`graph_retriever.py`)
   - Hybrid retrieval: Graph + Vector search
   - Node2Vec embeddings for semantic similarity
   - Neighbor expansion for comprehensive results

3. **Context Builder** (`context_builder.py`)
   - Structures raw graph data into educational context
   - Methodology recommendations with confidence levels
   - Student profile generation

4. **Response Generator** (`llm_chain.py`)
   - Generates natural Italian responses
   - Evidence-based recommendations
   - Confidence assessment

5. **Streamlit Interface** (`apps/streamlit/main.py`)
   - Interactive web application (legacy demo — retirement banner points to `/webui/`)
   - Real-time pipeline visualization
   - Evidence and comparison views
   - Agent Mode toggle for lesson generation

6. **FastAPI API** (`src/aix/api/`)
   - `POST /api/v1/context` — GraphRAG context endpoint (original)
   - `POST /api/v1/agent/run` — Synchronous agent JSON endpoint 🆕
   - `POST /api/v1/agent/stream` — SSE streaming agent endpoint 🆕
   - JWT Bearer + Cookie auth, Swagger UI at `/docs`
   - See [API_INTEGRATION_GUIDE.md](docs/api/API_INTEGRATION_GUIDE.md) for details

7. **Agentic GraphRAG** (`src/aix/agent/`)
   - **Orchestrator** (`orchestrator.py`): Main entry point, clean API
   - **Planner Agent** (`agents/planner_agent.py`): Query analysis & intent detection
   - **Retriever Agent** (`agents/retriever_agent.py`): GraphRAG knowledge retrieval
   - **Writer Agent** (`agents/writer_agent.py`): Adaptive content generation
   - **Critic Agent** (`agents/critic_agent.py`): Quality review & revision control
   - **LangGraph Pipeline** (`graph/`): State machine orchestration
   - **Intent-Specific Prompts** (`prompts/`): Optimized prompts per query type

8. **Teacher Webui** (`src/aix/webui/`) 🆕
   - Authenticated three-pane chat workspace at `/webui/lesson/{id}`
   - FastAPI-Users (JWT cookies), educational profile, file uploads (PDF/TXT/MD)
   - SSE streaming of granular agent phase cards (Planner → Retriever → Writer → Critic)
   - htmx 2 + WebAwesome 3.x + Tailwind CSS

9. **MCP Tool Servers** (`src/aix/mcp/`) 🆕
   - 10 tools (`kg.*`, `media.*`, `agent.run_lesson_plan`)
   - 4 resources (`kg://schema`, `kg://concepts/{domain}`, `methodology://list`, `media://stats`)
   - 2 prompts (`educational-query`, `lesson-plan-request`)
   - Two transports: **stdio** (Claude Desktop / Cursor IDE) + **Streamable HTTP** at `/mcp/` (JWT Bearer)
   - See [MCP_Setup.md](docs/integrations/MCP_Setup.md) for client onboarding

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Neo4j Database (local or Aura)
- OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mono33/graphrag-educational-assistant.git
   cd graphrag-educational-assistant
   ```

2. **Install dependencies**

   ```bash
   # Editable install — exposes the project as the `graphaixlearning` package.
   # Required after Phase 3A (pyproject.toml) so all imports resolve cleanly.
   pip install -e ".[dev]"
   ```

   The `[dev]` extras add `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`,
   and `mypy` (used by `make test`, `make lint`, and CI). Production
   deployments can use `pip install -e .` (without `[dev]`).

3. **Configure environment**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your credentials
   # - NEO4J_URI: Your Neo4j connection URI
   # - NEO4J_USER: Your Neo4j username
   # - NEO4J_PASSWORD: Your Neo4j password
   # - OPENAI_API_KEY: Your OpenAI API key
   ```

4. **Ingest a Knowledge Graph into Neo4j**

   Pre-baked KG dumps for both domains ship with the repo at
   `data/kg/neuro/kg_neuro_neo4j.json` and `data/kg/udl/kg_udl_neo4j.json`,
   so no data preparation is required for the default setup.

   ```bash
   # Ingest the included Neuro KG (or swap in kg_udl_neo4j.json for UDL)
   python scripts/ingest/data_ingestion_neo4j.py \
       --file data/kg/neuro/kg_neuro_neo4j.json \
       --password YOUR_NEO4J_PASSWORD \
       --clear
   ```

   *Advanced:* if you want to re-build the KG from raw team data,
   use the helpers under `scripts/data_prep/` (e.g.
   `clean_and_compare_neuro_data.py`, `merge_kg_json.py`).

5. **Train Node2Vec model** (optional — pre-trained artifacts ship in `artifacts/node2vec/`)
   ```bash
   python scripts/ml/train_node2vec.py neuro   # or udl
   ```

6. **Run the Streamlit app**
   ```bash
   streamlit run apps/streamlit/main.py     # or: make streamlit
   ```

7. **Access the app**
   - Open your browser to: `http://localhost:8501`
   - Start asking educational questions in Italian! 🇮🇹

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key

# Node2Vec Model Path (Phase 3B: models/ was renamed to artifacts/)
NODE2VEC_MODEL_DIR=./artifacts/node2vec
```

### Neo4j Setup

Your Neo4j database should contain:
- **Nodes**: Educational concepts, methodologies, strategies
- **Relationships**: BELONGS_TO, ADDRESSES, SUPPORTS, etc.
- **Properties**: name, description, category, etc.

See `data_ingestion_neo4j.py` for the ingestion script.

---

## 📊 Embedding Modes

The system supports two embedding modes for semantic search:

### 1. Node2Vec Mode (Default)
Uses graph structure embeddings for semantic search. Best for finding structurally related concepts.

```env
# In .env file (or omit for default)
EMBEDDING_MODE=node2vec
```

### 2. Hybrid Semantic Mode (Node2Vec + OpenAI)
Combines graph structure (Node2Vec) with text semantics (OpenAI embeddings). Best for natural language queries.

```env
# In .env file
EMBEDDING_MODE=hybrid_semantic
EMBEDDING_NODE2VEC_WEIGHT=0.4  # 40% Node2Vec, 60% OpenAI
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**First-time setup for hybrid mode:**
```bash
# Pre-compute OpenAI embeddings (one-time, ~$0.01)
python -m aix.retrieval.graph_retriever --precompute neuro
python -m aix.retrieval.graph_retriever --precompute udl
```

### Comparison

| Mode | Pros | Cons |
|------|------|------|
| `node2vec` | Fast, no API calls, captures domain structure | May miss semantic variations |
| `hybrid_semantic` | Better text understanding, multilingual | Requires OpenAI API, slightly slower |

### Pre-trained Models

Models are stored in `artifacts/`:
- `artifacts/node2vec/{domain}_node2vec_embeddings.npz`: Node2Vec embeddings
- `artifacts/embeddings_cache/{domain}_openai_embeddings.json`: OpenAI embeddings cache

### Retrain Node2Vec:
```bash
python scripts/ml/train_node2vec.py neuro
python scripts/ml/train_node2vec.py udl
```

**Node2Vec Parameters:**
- Dimensions: 128
- Walk Length: 80
- Num Walks: 200
- Workers: 4
- P: 1.0, Q: 1.0

---

## 🎯 Usage Examples

### Example Queries (in Italian):

1. **"Ci sono strategie per studenti con ADHD?"**
   - *Are there strategies for students with ADHD?*

2. **"Metodologie per studenti dello spettro autistico?"**
   - *Methodologies for students on the autism spectrum?*

3. **"Come aiutare studenti senza motivazione?"**
   - *How to help students without motivation?*

4. **"Il mio studente ha ADHD, cosa posso fare?"**
   - *My student has ADHD, what can I do?*

### Example Response:

```
Sì, ci sono diverse strategie efficaci per studenti con ADHD:

1. Flipped Classroom (ALTA confidenza)
   - Permette agli studenti di lavorare al proprio ritmo
   - Riduce la pressione del tempo in classe
   - Favorisce l'apprendimento attivo

2. Cooperative Learning (ALTA confidenza)
   - Sviluppa competenze sociali
   - Migliora l'attenzione attraverso l'interazione
   ...
```

---

## 🤖 Agent Mode Usage (NEW)

### Using Agent Mode in Streamlit

1. Launch the Streamlit app: `streamlit run apps/streamlit/main.py`
2. Toggle **"🤖 Modalità Agente"** in the sidebar
3. Enter your query (lesson request, definition, comparison, etc.)
4. The multi-agent pipeline will:
   - Detect your query intent
   - Search the knowledge graph
   - Generate appropriate content
   - Review and refine the output

### Example Agent Queries:

| Query Type | Example | What You Get |
|------------|---------|--------------|
| **Lesson Creation** | "Crea una lezione sulla motivazione per studenti con ADHD" | Full structured lesson plan |
| **Activity Design** | "Attività di 30 minuti sulla metacognizione" | Detailed activity with timing |
| **Definition** | "Cos'è la neuroplasticità?" | Clear, concise definition |
| **Comparison** | "Qual è la differenza tra memoria a breve e lungo termine?" | Side-by-side comparison |
| **Recommendation** | "Quali strategie per studenti con difficoltà di attenzione?" | Actionable strategy list |

### Using Agent Mode Programmatically

```python
from aix.agent import AgentOrchestrator

# Initialize orchestrator
orchestrator = AgentOrchestrator(
    domain="neuro",      # or "udl"
    language="it",       # or "en"
    max_revisions=2
)

# Generate lesson plan (async)
result = await orchestrator.create_lesson_plan(
    "Crea una lezione sulla motivazione per studenti con ADHD"
)

# Access results
print(result.lesson_plan)       # Generated content
print(result.approved)          # True if critic approved
print(result.scores)            # Quality scores
print(result.query_intent)      # Detected intent type
```

### Testing the Agent Pipeline

```bash
# Interactive testing
python apps/cli/run_agent.py

# With options
python apps/cli/run_agent.py --domain neuro --language it

# Single query mode
python apps/cli/run_agent.py --query "Crea una lezione sulla memoria"

# Test intent detection
python -m pytest tests/integration/test_intent_detection.py
```

---

## 🛠️ Development

### Project Structure

The repository follows the modern Python **src layout**: every importable module lives
under a single top-level package `aix`, exposed by `pip install -e .[dev]`.

```
graphaixlearning/
│
├── src/                            # All importable source code
│   └── aix/                        #   Top-level package (import as `aix.*`)
│       ├── __init__.py
│       ├── core/                   #   Shared config + utilities
│       │   └── config.py           #     `from aix.core.config import config`
│       ├── retrieval/              #   GraphRAG retrieval layer
│       │   ├── graph_retriever.py
│       │   ├── context_builder.py
│       │   ├── text2cypher.py
│       │   ├── multilingual_text2cypher.py
│       │   └── query_metrics.py
│       ├── generation/             #   LLM response generation
│       │   └── llm_chain.py
│       ├── agent/                  #   Agentic GraphRAG (multi-agent pipeline)
│       │   ├── orchestrator.py     #     `from aix.agent import AgentOrchestrator`
│       │   ├── agents/             #     Planner, Retriever, Writer, Critic
│       │   ├── graph/              #     LangGraph state machine
│       │   ├── prompts/            #     Intent-specific prompts
│       │   ├── media/              #     Media lookup + diagram generation
│       │   ├── tools/              #     GraphRAG wrapper for agents
│       │   └── configs/            #     Domain prompt extensions
│       ├── api/                    #   FastAPI service
│       │   ├── main.py             #     uvicorn aix.api.main:app --port 8765
│       │   ├── routes/context.py   #     /api/v1/context (GraphRAG)
│       │   ├── routes/agent.py     #     /api/v1/agent/run + /stream (Agent API) 🆕
│       │   ├── schemas/            #     Pydantic models (agent.py, models.py, etc.)
│       │   └── graphrag_client.py  #     Helper client for DEV team
│       ├── webui/                  #   Teacher webui (htmx + WebAwesome) 🆕
│       │   ├── auth/               #     FastAPI-Users (cookie + Bearer JWT)
│       │   ├── lessons/            #     Lesson CRUD, uploads, schemas
│       │   ├── agent/service.py    #     run_agent_stream + stream_agent_events
│       │   ├── templates/          #     Jinja2: _base, pages/, partials/
│       │   ├── routes.py           #     /webui/* route handlers
│       │   └── db.py               #     SQLAlchemy async + aiosqlite (dev)
│       ├── mcp/                    #   MCP Tool Servers (FastMCP 3.x) 🆕
│       │   ├── server.py           #     build_mcp_server() — composition root
│       │   ├── stdio_main.py       #     stdio entry (Claude Desktop / Cursor)
│       │   ├── http_app.py         #     Streamable HTTP factory (JWT Bearer)
│       │   ├── tools/              #     kg_tools, media_tools, agent_tools
│       │   ├── resources/          #     kg_resources (schema, concepts, etc.)
│       │   └── prompts/            #     educational_prompts
│       └── domains/                #   Domain configs (UDL, Neuro)
│
├── apps/                           # User-facing entry points (NOT importable libs)
│   ├── streamlit/main.py           #   Streamlit interface (legacy — retirement banner)
│   └── cli/run_agent.py            #   Interactive agent testing CLI
│
├── scripts/                        # Operational & data-prep scripts
│   ├── ingest/                     #   Neo4j data import/export
│   ├── audit/                      #   KG auditing + label checks
│   ├── data_prep/                  #   Data cleaning, merging, label fixes
│   ├── ml/                         #   Node2Vec training, media mapping gen
│   ├── diagnostic/                 #   MCP smoke tests, OpenAPI probes 🆕
│   └── ops/                        #   Preflight checks, migration runners
│
├── data/                           # KG data, media mappings, reports
│   ├── kg/{neuro,udl}/             #   Knowledge graph core dumps
│   ├── media/                      #   Media mappings + resource JSONs
│   ├── reference/                  #   API contract (JSON_reference.json)
│   ├── reports/                    #   Audit reports
│   └── diagnostic/                 #   OpenAPI baselines (regression guard) 🆕
│
├── artifacts/                      # ML model artifacts (was models/)
│   ├── node2vec/                   #   Node2Vec {config,embeddings,model}
│   └── embeddings_cache/           #   OpenAI embeddings cache
│
├── tests/                          # Test suite
│   ├── integration/                #   Integration tests (external services)
│   ├── api/                        #   Agent API contract tests 🆕
│   ├── mcp_server/                 #   MCP regression suite (19 tests) 🆕
│   └── conftest.py                 #   Pytest config / shared fixtures
│
├── docs/                           # Documentation
│   ├── api/                        #   API guides
│   ├── architecture/               #   Architecture analyses + frontend eval
│   ├── integrations/               #   MCP_Setup.md (client onboarding) 🆕
│   ├── runbooks/                   #   Quickstart, data pipeline guides
│   ├── reports/                    #   Diff reports, analysis docs
│   ├── product/                    #   ClickUp tracker, handoff docs
│   ├── progress_reports/           #   Templates + generated reports
│   └── prompts_reference/          #   Reference prompt texts
│
├── archive/                        # Deprecated modules (kept for reference)
├── pyproject.toml                  # Build, deps, pytest, ruff, mypy config
├── requirements.txt                # Runtime dependencies
├── Makefile                        # Shortcuts: make test / api / streamlit
├── Dockerfile                      # Container build (uvicorn aix.api.main:app)
├── .env.example                    # Environment template
└── README.md                       # This file
```

**Common entry points** (after `pip install -e .[dev]`):

```bash
# FastAPI server — serves /docs, /webui/, /api/v1/*, /mcp/ on a single port
uvicorn aix.api.main:app --reload --port 8765 --app-dir src

# Streamlit demo UI (legacy — retirement banner points to /webui/)
streamlit run apps/streamlit/main.py                 # or: make streamlit

# Interactive Agent CLI
python apps/cli/run_agent.py                         # or: make agent

# Tests (all suites)
pytest tests/ -v                                     # or: make test

# MCP regression tests only
pytest tests/mcp_server/ -v

# MCP smoke test (in-process, no uvicorn needed)
python scripts/diagnostic/mcp_smoke.py
```

### Testing

```bash
# Run all tests (with pytest via pyproject.toml config)
pytest tests/ -v

# Run only integration tests (requires Neo4j / LLM API keys)
pytest tests/integration/ -v -m integration
```

### Adding New Data

1. **Prepare data**: Format as JSON (see `data/reference/JSON_reference.json`)
2. **Ingest**: Run `python -m scripts.ingest.data_ingestion_neo4j --file your_data.json --clear`
3. **Retrain Node2Vec**: Run `python -m scripts.ml.train_node2vec`
4. **Test**: Launch app and verify results

---

## 📚 Additional Resources

For additional documentation and guides, see the `NOTPUSHED/` folder (local development only).

---

## 🧪 Technologies Used

- **[Neo4j](https://neo4j.com/)**: Graph database for knowledge representation
- **[OpenAI GPT-4o](https://openai.com/)** (via OpenRouter): Language model for response generation
- **[LangChain](https://langchain.com/)**: LLM application framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: Multi-agent orchestration framework
- **[FastAPI](https://fastapi.tiangolo.com/)**: REST API + webui serving (single process)
- **[FastAPI-Users](https://fastapi-users.github.io/)**: Authentication (cookie + JWT Bearer)
- **[htmx 2](https://htmx.org/)**: HTML-over-the-wire for the teacher webui
- **[WebAwesome 3.x](https://www.webawesome.com/)**: Web component library
- **[Tailwind CSS](https://tailwindcss.com/)**: Utility-first CSS framework
- **[sse-starlette](https://github.com/sysid/sse-starlette)**: Server-Sent Events for streaming
- **[FastMCP 3.x](https://github.com/jlowin/fastmcp)**: Model Context Protocol server framework
- **[Node2Vec](https://github.com/eliorc/node2vec)**: Graph embedding for semantic search
- **[Streamlit](https://streamlit.io/)**: Legacy web interface (retirement banner active)

---

## 🎓 Educational Context

This system is designed specifically for **teachers and educators** who need:
- Evidence-based teaching strategies
- Recommendations for students with special needs (ADHD, autism, etc.)
- Personalized methodology suggestions
- Quick access to educational best practices

All responses are:
- ✅ In Italian (primary audience)
- ✅ Based on knowledge graph evidence
- ✅ Structured for classroom implementation
- ✅ Confidence-assessed for reliability

---

## 🚀 Deployment Options

### 1. Local Development (single process — all surfaces)
```bash
uvicorn aix.api.main:app --reload --port 8765 --app-dir src
```
This single uvicorn process serves:
- `/docs` — Swagger UI (GraphRAG + Agent API endpoints)
- `/webui/` — Teacher webui (auth, lesson creation, SSE chat)
- `/api/v1/context` — GraphRAG context endpoint
- `/api/v1/agent/run` + `/api/v1/agent/stream` — Agent API (JSON + SSE)
- `/mcp/` — MCP Streamable HTTP (JWT Bearer auth)
- `/auth/jwt/login` — JWT token minting

### 2. MCP Clients (Claude Desktop / Cursor IDE)
```bash
# stdio transport — no uvicorn needed, clients launch the subprocess directly
python -m aix.mcp.stdio_main
```
See [MCP_Setup.md](docs/integrations/MCP_Setup.md) for client configuration.

### 3. Streamlit (legacy demo)
```bash
streamlit run apps/streamlit/main.py
```

### 4. Cloud Deployment
- Use Neo4j Aura for cloud database
- Docker Compose on Hetzner/Coolify (planned — see `ClickUp_Agentic_GraphRAG_Update.md` #6.6 P6)
- See [API_INTEGRATION_GUIDE.md](docs/api/API_INTEGRATION_GUIDE.md) for REST API docs

---

## 📈 Performance

- **Average Query Time**: 2-4 seconds
- **Retrieval Accuracy**: 85%+ relevant results
- **Node2Vec Coverage**: 109 educational concepts
- **Supported Languages**: Italian (primary), English (queries)

---

## 🤝 Contributing

This is an educational project. If you want to contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support & Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [Your Email/FEM-modena]

---

## 📝 License

[Specify your license here - MIT, Apache 2.0, etc.]

---

## 👥 Authors

- **Louis Mono, Angelo Casali** - Initial development - [Mono33](https://github.com/Mono33)
- **FEM-modena Team** - Educational content and validation

---

## 🙏 Acknowledgments

- Built for educational purposes
- Knowledge graph curated by FEM-modena educational team
- Powered by Neo4j, OpenAI, and open-source technologies

---

## 📊 Future Enhancements

- [x] ~~Integration with existing educational agent~~ ✅ **DONE** (FastAPI module)
- [x] ~~Agentic GraphRAG multi-agent pipeline~~ ✅ **DONE** (LangGraph + 4 agents)
- [x] ~~Intelligent query intent detection~~ ✅ **DONE** (7 intent types)
- [x] ~~Adaptive content generation~~ ✅ **DONE** (Lesson, definition, comparison, etc.)
- [x] ~~Quality control with revision loop~~ ✅ **DONE** (Critic agent + scoring)
- [x] ~~Streaming response generation~~ ✅ **DONE** (SSE via `/api/v1/agent/stream` + webui SSE chat)
- [x] ~~MCP Tool Servers~~ ✅ **DONE** (10 tools / 4 resources / 2 prompts, stdio + Streamable HTTP)
- [x] ~~Teacher webui with auth~~ ✅ **DONE** (FastAPI + htmx + WebAwesome, P0–P3)
- [ ] Lesson library + history + PDF export (#6.6 P4)
- [ ] Italian copy + accessibility + mobile breakpoints + Tailwind CLI (#6.6 P5)
- [ ] Agent JSON parse hardening (#11a)
- [ ] Observability / tracing dashboard (LangSmith/Langfuse, #11b)
- [ ] Multi-language response generation (English, Spanish)
- [ ] Expanded knowledge graph (500+ concepts)
- [ ] Long-term memory for teacher profiles (#16)
- [ ] Conversation memory (LangGraph Checkpointer, #10)
- [ ] Collaborative filtering for recommendations

---

**Made with ❤️ for educators by educators**
