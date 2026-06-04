# Handover Tecnico — Meeting Dev Team

**Data:** 23 maggio 2026  
**Partecipanti:** AI Team (Angelo) + Dev Team AixLearning  
**Obiettivo:** Overview del repo e descrizione del funzionamento nei dettagli  

---

## 1. Cos'è il Sistema — Panoramica in una Pagina

Il sistema è un **pipeline multi-agente** (Agentic GraphRAG) per la generazione di lezioni pedagogicamente fondate. Sostituisce la singola chiamata LLM con 4 agenti specializzati che lavorano in sequenza, grounded su un Knowledge Graph Neo4j con 720+ concetti di neuroscienze e UDL.

**Flusso in una riga:**
```
Teacher query → Planner → Retriever (Neo4j KG) → Writer → Critic → Lesson Plan MD
```

**Stack tecnologico:**
| Componente | Tecnologia |
|-----------|-----------|
| API Server | FastAPI (Python 3.12) |
| Orchestrazione agenti | LangGraph (state-machine) |
| Knowledge Graph | Neo4j (720+ nodi, 745+ relazioni) |
| LLM | OpenRouter (Claude Sonnet 4.6, Gemini Flash) |
| Embedding | text-embedding-3-small (OpenAI via OpenRouter) |
| Multi-turn memory | LangGraph Checkpointer (SQLite dev / Postgres prod) |
| Observability | Langfuse (traces per-agente, prompt versioning) |
| Auth WebUI | fastapi-users (cookie + Bearer JWT) |

**Due modalità di deploy (non in conflitto):**
- **Mode A — Standalone WebUI:** il nostro servizio al dominio `agente.aiforlearning.digital`, Caddy come reverse proxy, WebUI propria
- **Mode B — Integrazione nativa AixLearning:** AixLearning chiama `POST /api/v1/agent/stream` su rete Docker interna; il teacher rimane nella UI AixLearning

---

## 2. Mappa del Repository

```
src/aix/
├── api/
│   ├── main.py              ← Entry point FastAPI, lifespan, startup checks
│   ├── routes/
│   │   ├── agent.py         ← POST /api/v1/agent/run e /stream  ⭐ chiave per l'integrazione
│   │   └── context.py       ← POST /api/v1/context (endpoint legacy KG retrieval)
│   └── schemas/
│       ├── agent.py         ← AgentRunRequest, AgentRunResponse, tutti gli SSE events
│       └── models.py        ← EducationalProfile, Group, Classroom
│
├── agent/
│   ├── agents/
│   │   ├── planner_agent.py     ← Analisi intent, scope detection, search queries
│   │   ├── retriever_agent.py   ← Query Neo4j + fallback Wikipedia/Scholar
│   │   ├── writer_agent.py      ← Generazione lesson plan MD
│   │   ├── critic_agent.py      ← Quality gate (0-1 score, approve/revise)
│   │   └── retrieval_grader_agent.py  ← Corrective RAG (opzionale)
│   ├── graph/
│   │   ├── state.py             ← AgentState TypedDict (stato condiviso tra agenti)
│   │   ├── nodes.py             ← I 5 nodi LangGraph (plan_node, retrieve_node, ...)
│   │   ├── lesson_planner_graph.py  ← Build + compilazione grafo
│   │   └── checkpointer.py      ← AsyncSqliteSaver per multi-turn memory
│   ├── prompts/                 ← System prompt e template per ogni agente
│   └── tools/
│       └── graphrag_tool.py     ← GraphRAGTool (Text2Cypher → Cypher → Neo4j)
│
├── retrieval/
│   ├── multilingual_text2cypher.py  ← Text2Cypher con schema caching
│   └── context_builder.py           ← Costruzione contesto educativo da nodi KG
│
├── webui/
│   ├── agent/service.py     ← stream_agent_events() — il ponte tra API e LangGraph
│   └── auth/
│       ├── dependencies.py  ← current_active_user (cookie + Bearer JWT)
│       └── backend.py       ← Dual-transport auth backend
│
└── domains/
    ├── neuro/              ← Config dominio neuroscienze (system prompt, template)
    ├── udl/                ← Config dominio UDL
    └── langfuse_prompts.py ← Prompt versioning via Langfuse
```

---

## 3. Il Pipeline degli Agenti — Funzionamento Dettagliato

### Topologia Base

```
START → [PLANNER] → [RETRIEVER] → [WRITER] → [CRITIC] → END
                                      ↑           │
                                      └───────────┘  (revision loop, max 2x)
```

### Topologia con Corrective RAG (flag `AIX_CORRECTIVE_RAG_ENABLED=true`)

```
START → [PLANNER] → [RETRIEVER] → [GRADE_RETRIEVAL]
                         ↑               │
                         └──(retry)──────┤ grade="irrelevant"
                                         │ grade="relevant"
                                         ↓
                                    [WRITER] → [CRITIC] → END
```

### Stato Condiviso — `AgentState`

Tutti gli agenti leggono e scrivono su un unico `AgentState` (TypedDict con 150+ campi). È il "contratto interno" del pipeline.

**Campi principali:**
```
INPUT:          teacher_query, domain, language, educational_profile, session_id
PLANNER OUTPUT: plan, query_intent, key_concepts, search_queries, scope_status, scope_confidence
RETRIEVER OUT:  retrieved_nodes, recommendations, retrieval_confidence, curated_media, external_resources
WRITER OUTPUT:  lesson_plan_draft, lesson_plan_structured, sources_cited
CRITIC OUTPUT:  critique_score, approved, revision_instructions, revision_count
FINAL:          final_lesson_plan, final_metadata
```

### I 4 Agenti — Responsabilità

#### 1. Planner Agent (`src/aix/agent/agents/planner_agent.py`)

**Input:** query, domain, language, educational_profile  
**Output in state:** plan, query_intent, key_concepts, search_queries, scope_status

- Analizza il query del teacher per estrarre: intent (`lesson_creation`, `definition`, `comparison`, ...), concetti chiave, query di ricerca per il KG
- Rileva il `scope_status`: `in_scope` (topic nel KG), `partial_scope`, `out_of_scope`
- Rileva la lingua (può override il parametro `language`)
- Considera `time_available_minutes` da `educational_profile` per dimensionare la lezione

**Esempio output (state):**
```json
{
  "query_intent": "lesson_creation",
  "key_concepts": ["metacognizione", "memoria di lavoro", "strategie DSA"],
  "search_queries": ["strategie metacognitive DSA", "memoria lavoro ADHD"],
  "scope_status": "in_scope",
  "scope_confidence": 0.92
}
```

---

#### 2. Retriever Agent (`src/aix/agent/agents/retriever_agent.py`)

**Input da state:** search_queries, domain, scope_status  
**Output in state:** retrieved_nodes, recommendations, retrieval_confidence, curated_media, external_resources

- Esegue le `search_queries` via `GraphRAGTool` → `EnhancedMultilingualText2Cypher` → Cypher → Neo4j
- **Embedding ibrido:** `final_score = 0.4 × node2vec_score + 0.6 × semantic_score`
- Classifica la confidence: `HIGH` / `MEDIUM` / `LOW` in base al numero di nodi trovati e alla loro rilevanza
- Se `scope_status = "out_of_scope"` o `"partial_scope"`: fallback alle fonti ibride
  - Wikipedia (sommari in italiano)
  - Semantic Scholar / OpenAlex (paper accademici)
  - OER Textbooks (CommonLit, OpenStax)
  - DuckDuckGo (ricerca web filtrata per didattica)

**Output media curati:**
```json
{
  "curated_media": {
    "videos": [{"title": "...", "url": "...", "duration_seconds": 480}],
    "articles": [...],
    "oer": [...]
  }
}
```

---

#### 3. Writer Agent (`src/aix/agent/agents/writer_agent.py`)

**Input da state:** retrieved_nodes, recommendations, curated_media, educational_profile, conversation_history, query_intent, revision_instructions (se revisione)  
**Output in state:** lesson_plan_draft, sources_cited

- Seleziona il template Markdown in base a `query_intent` (6 template disponibili)
- Incorpora il profilo educativo: BES, disabilità, risorse aula, tempo disponibile
- Gestisce il multi-turn: `conversation_history` dai turni precedenti + `conversation_summary` (windowing)
- Supporta `teacher_provided_context` (fino a 48k chars da upload PDF/TXT del teacher)
- Token limit handling: auto-continuation se `finish_reason="length"`
- Se è una revisione: usa `revision_instructions` dal Critic per correggere specificamente

---

#### 4. Critic Agent (`src/aix/agent/agents/critic_agent.py`)

**Input da state:** lesson_plan_draft, retrieved_nodes, query_intent  
**Output in state:** critique_score (0-1), approved, revision_instructions

- Valuta la bozza su 5 criteri: allineamento query, solidità pedagogica, evidence-based, adeguatezza età, completezza
- `approved = True` se `critique_score >= threshold` (default: 0.7)
- Se non approvato E `revision_count < max_revisions`: emette `revision_instructions` specifiche → torna al Writer
- Max 2 revisioni per default (`AIX_MAX_REVISIONS=1` → 1 revisione automatica)

---

### Routing Condizionale (LangGraph edges)

```python
# In lesson_planner_graph.py
def should_continue_to_revision(state: AgentState) -> str:
    if state["approved"] or state["revision_count"] >= state["max_revisions"]:
        return "finish"     # → END
    if state.get("error"):
        return "error"      # → END
    return "revise"         # → write_node (loop)
```

---

## 4. Contratto API — Endpoint per l'Integrazione

### `POST /api/v1/agent/stream` — SSE Streaming (raccomandato per V1)

**Request body:**
```json
{
  "query": "Crea una lezione di 45 min sulla fotosintesi per una classe con DSA",
  "domain": "neuro",
  "language": "it",
  "session_id": "uuid-generato-dal-client",
  "educational_profile": {
    "group": {
      "title": "3A Liceo Scientifico",
      "students_number": 25,
      "grade": "SECONDARIA_II_GRADO",
      "disabilities": ["ADHD", "DSA"],
      "class_features": ["MOTIVATA"],
      "student_attributes": ["PUNTI_DI_ECCELLENZA", "PUNTI_DI_CADUTA"]
    },
    "classroom": {
      "title": "Aula 101",
      "forniture_mobility": "PARTIALLY",
      "has_lim": true,
      "has_wifi": true,
      "has_suite": true,
      "pc_station": false,
      "own_device": "BES"
    },
    "time_available_minutes": 45,
    "subject_area": "Scienze",
    "specific_topic": "Fotosintesi"
  },
  "max_revisions": 1
}
```

**Sequenza eventi SSE (discriminati dal campo `kind`):**

```
data: {"kind": "planner", "data": {
  "intent": "lesson_creation",
  "intent_label": "Creazione lezione",
  "scope": "in_scope",
  "scope_label": "Nel Knowledge Graph",
  "scope_confidence": 0.92,
  "key_concepts": ["fotosintesi", "cloroplasti"],
  "search_queries": ["strategie insegnamento fotosintesi DSA", ...]
}, "lesson_plan_md": null, "error": null}

data: {"kind": "retriever", "data": {
  "nodes_count": 23,
  "relationships_count": 47,
  "recommendations_count": 5,
  "media_counts": {"videos": 3, "articles": 2, "oer": 4},
  "media": {"videos": [...], "articles": [...], "oer": [...]},
  "retrieval_confidence": "HIGH"
}, "lesson_plan_md": null, "error": null}

data: {"kind": "writer_pending", "data": {"revision": 1, "is_revision": false}, ...}

data: {"kind": "writer", "data": {"revision": 1},
  "lesson_plan_md": "# Lezione: Fotosintesi\n\n## Obiettivi...",
  "error": null}

data: {"kind": "critic", "data": {
  "score": 0.88,
  "approved": true,
  "feedback": "Strong pedagogical structure...",
  "revision": 1
}, "lesson_plan_md": "# Lezione: Fotosintesi\n\n...", "error": null}

data: {"kind": "done", "data": {}, 
  "lesson_plan_md": "# Lezione: Fotosintesi\n\n...",
  "error": null,
  "meta": {
    "duration_seconds": 73.2,
    "approved": true,
    "revision_count": 0,
    "nodes_count": 23,
    "recommendations_count": 5,
    "media_counts": {"videos": 3, "articles": 2, "oer": 4}
  }}
```

**Mapping eventi → AixLearning (lavoro del DEV):**
| Evento nostro | Evento AixLearning | Note |
|--------------|-------------------|-------|
| `writer` / `done` | `LLMTextDoneEvent` | Il contenuto `lesson_plan_md` è il Markdown finale |
| `planner`, `retriever`, `critic` | `LLMTextDeltaEvent` o badge | V1: opzionale, V2: trust signals in UI |
| `error` | gestione errore | Refund crediti, log Langfuse |

---

### `POST /api/v1/agent/run` — JSON Sincrono (utile per test)

Stessa request. Response unica JSON con tutto il contenuto:
```json
{
  "lesson_plan_md": "# Lezione...",
  "meta": {"duration_seconds": 73, "approved": true, ...},
  "planner": {"intent": "lesson_creation", "key_concepts": [...], ...},
  "retriever": {"nodes_count": 23, "media": {...}, ...}
}
```

Non raccomandato per produzione (timeout a 60-180 secondi), ottimo per smoke test.

---

### `POST /api/v1/context` — Retrieval-only Legacy (invariato)

Endpoint esistente, usato per il vecchio pattern GraphRAG. Continua a funzionare invariato.  
**Nessun auth nel codice FastAPI** — protetto a livello infrastrutturale (Caddy / rete Docker interna).

---

## 5. Autenticazione — Situazione Attuale e Piano per l'Integrazione

### Situazione attuale

| Endpoint | Auth nel codice |
|---------|----------------|
| `POST /api/v1/context` | **Nessuna** (aperto nel codice FastAPI, protetto da infrastruttura/rete) |
| `POST /api/v1/agent/run` | Bearer JWT o cookie (fastapi-users) |
| `POST /api/v1/agent/stream` | Bearer JWT o cookie (fastapi-users) |
| `/webui/*` | Cookie HttpOnly (fastapi-users) |

### Per l'integrazione V1 (Mode B)

**Opzione raccomandata:** aggiungere Basic Auth come metodo alternativo sugli endpoint `/api/v1/agent/*`, usando le stesse credenziali `GRAPH_API_USERNAME` / `GRAPH_API_PWD` già configurate su AixLearning per `/api/v1/context`.

**Implementazione in `src/aix/api/routes/agent.py`:**
```python
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_basic_security = HTTPBasic(auto_error=False)

async def _get_caller(
    basic: Optional[HTTPBasicCredentials] = Depends(_basic_security),
    user: Optional[User] = Depends(optional_current_user),
) -> Union[User, str]:
    """Accetta Bearer JWT (webui), cookie (browser) OPPURE Basic Auth (service-to-service)."""
    if basic is not None:
        expected_user = os.getenv("GRAPH_API_USERNAME", "")
        expected_pwd = os.getenv("GRAPH_API_PWD", "")
        if (
            expected_user
            and secrets.compare_digest(basic.username, expected_user)
            and secrets.compare_digest(basic.password, expected_pwd)
        ):
            return "service_account"  # sentinel, non un User DB
    if user is not None:
        return user
    raise HTTPException(status_code=401, detail="Unauthorized")
```

**Variabili da aggiungere al `.env`:**
```
GRAPH_API_USERNAME=aix-service
GRAPH_API_PWD=<strong-random-secret>
```

**Opzione alternativa (più semplice per V1):** rimuovere del tutto l'auth dai due endpoint agent — come `/api/v1/context` — e affidarsi alla sicurezza della rete Docker interna. Questa è la soluzione zero-code ma va discussa con il team.

---

## 6. Multi-turn Memory — Come Funziona

```
                 ┌─────────────────────────────────────────┐
                 │  LangGraph Checkpointer (L1)             │
                 │  SQLite dev / Postgres prod              │
                 │  Chiave: thread_id = session_id          │
                 │  Salva: snapshot completo AgentState     │
                 └─────────────────────────────────────────┘

                 ┌─────────────────────────────────────────┐
                 │  WebUI DB (L2)                           │
                 │  Tabella lesson_message                  │
                 │  Salva: markdown + metadata per turno    │
                 └─────────────────────────────────────────┘
```

**Per l'integrazione AixLearning:**
- Il `session_id` nel request body diventa il `thread_id` del checkpointer
- AixLearning gestisce già la conversation (passa il proprio ID conversazione)
- Le tabelle LangGraph (`checkpoint`, `checkpoint_blobs`, `checkpoint_versions`) sono gestite automaticamente — non richiedono setup manuale

---

## 7. Docker e Variabili d'Ambiente

### docker-compose.yaml attuale (Mode A standalone)

```yaml
services:
  api:
    image: rg.fr-par.scw.cloud/wonderful-education/graphrag-aixlearning:latest
    container_name: ${API_CONTAINER_NAME}
    ports:
      - "8000:80"
    env_file: .env
    restart: unless-stopped
```

### Aggiunta per Mode B (integrazione AixLearning)

Nel `docker-compose.yml` di AixLearning, aggiungere:
```yaml
services:
  graphrag-api:
    image: rg.fr-par.scw.cloud/wonderful-education/graphrag-aixlearning:latest
    env_file: ./graphrag.env
    networks:
      - fem-internal
    restart: unless-stopped
    # NOT exposed publicly — solo rete interna
```

Variabile da aggiungere su AixLearning:
```
GRAPH_API_ENDPOINT=http://graphrag-api:8765
```

### Variabili d'ambiente essenziali

```bash
# Neo4j (Knowledge Graph)
NEO4J_URI=bolt+s://your-host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
NEO4J_ENCRYPTED=1  # 1 per TLS (Aura), 0 per locale

# LLM
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=anthropic/claude-sonnet-4-6

# Auth WebUI (necessario anche per API JWT)
WEBUI_AUTH_SECRET=<secrets.token_urlsafe(48)>

# Multi-turn memory
LANGGRAPH_CHECKPOINTER_URL=sqlite:///./data/agent_threads.db  # dev
# LANGGRAPH_CHECKPOINTER_URL=postgresql+psycopg://user:pwd@host:5432/db  # prod

# Observability (opzionale ma consigliato)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...

# Per integrazione AixLearning (da aggiungere)
GRAPH_API_USERNAME=aix-service
GRAPH_API_PWD=<strong-random-secret>

# Tuning agenti
AIX_MAX_REVISIONS=1
AIX_CORRECTIVE_RAG_ENABLED=false
AIX_CRITIC_MODEL=google/gemini-2.5-flash
AIX_WRITER_MAX_TOKENS=3500
```

---

## 8. Observability — Langfuse

- **Prompt versioning:** i system prompt degli agenti sono su Langfuse (non nel codice). Modifica → live in 60 secondi senza restart
- **Per-agent traces:** ogni agente emette una trace con modello LLM, token usati, latenza, decisione
- **`meta.trace_id`** nella response `done` può essere passato ad AixLearning per unificare i trace nel dashboard esistente (stessa integrazione Langfuse già in produzione)

---

## 9. I 4 Step per l'Integrazione — Summary per il DEV Team

Seguendo la Sezione D.2 del documento di handoff:

**Step 1 — Nuovo service wrapper** (analogo a `graph_rag_integration.py`):
- Usa lo stesso pattern `httpx.Client` + `TenaciousTransport`
- Target: `POST /api/v1/agent/stream` (invece di `/api/v1/context`)
- Risposta: SSE events (invece di JSON singolo)
- Traduce `writer`/`done` → `LLMTextDoneEvent`; `planner`/`retriever`/`critic` → badge opzionali

**Step 2 — Routing nel Dramatiq worker**:
- `if plan_type in ("UDL", "NEURO"): → AgenticGraphRagService`
- `else: → TextClient (invariato)`
- Una sola condizione `if/else`

**Step 3 — Aggiunta servizio Docker**:
- `graphrag-api` sulla rete interna `fem-internal`
- `GRAPH_API_ENDPOINT=http://graphrag-api:8765`

**Step 4 — Smoke test**:
- Lezione UDL e NEURO in staging
- Verifica trace Langfuse nel dashboard
- Verifica crediti e refund su errore

---

## 10. Agenda Meeting (60 min)

1. **[10 min]** Overview architetturale + i due deploy mode
2. **[15 min]** Walkthrough pipeline: live su Swagger `/docs`, mostrare SSE events con `curl -N`
3. **[15 min]** Contratto API: request body + ogni evento SSE + mapping verso eventi AixLearning
4. **[10 min]** Auth gap + Step 1-4 della sezione D.2 del handoff doc
5. **[10 min]** Q&A + assegnazione task

---

## File Chiave da Condividere con il Team Dev

| File | Perché è rilevante |
|------|-------------------|
| `src/aix/api/schemas/agent.py` | Contratto completo: request + response + tutti gli SSE event types |
| `src/aix/api/routes/agent.py` | Implementazione endpoint (pattern auth, stream structure) |
| `.env.example` | Variabili d'ambiente richieste |
| `docs/api/API_INTEGRATION_GUIDE.md` | Guida esistente per l'integrazione API |

---

*Documento preparato per il meeting del 23 maggio 2026.*
