# Agentic GraphRAG — Documentazione Tecnica

**Progetto:** Agentic GraphRAG — Sistema multi-agente per la pianificazione delle lezioni
**Owner:** FEM AI Team
**Tipo di documento:** Riferimento tecnico per ingegneria, integrazione e operazioni
**Versione:** 1.3
**Ultimo aggiornamento:** Luglio 2026

---

## Storia delle modifiche

| Versione | Data | Descrizione |
|---|---|---|
| 0.1 | Maggio 2026 | Struttura iniziale, indice e §1 Introduzione. |
| 1.0 | Maggio 2026 | Prima bozza completa: §2–§18 scritte e ancorate al codice |
| 1.1 | Giugno 2026 | Aggiunto il livello media dinamici (§7.8) e le variabili `AIX_MEDIA_*` (§8.2); note sulla media-cache (§3.4/§7.5); item di test e runbook per i media live (§10.3/§16.1); riconciliazione YouTube (§7.6); indice riallineato ai titoli reali. |
| 1.2 | Giugno 2026 | storia delle modifiche spostata in testa al documento; alleggerimenti redazionali. |
| 1.3 | Luglio 2026 | Riallineamento al codice corrente: UX docente WebUI a due fasi, profili salvati, intento pedagogico, raffinamento SAM, badge UDL, pannello "Cosa esplorare dopo", export MD/TXT/PDF, default Writer aggiornato, note CSS, registrazione primo utente e concorrenza multi-worker con registro run su database. |

---

## Indice

1. Introduzione
   1.1 Scopo
   1.2 Destinatari
   1.3 Ambito
   1.4 Convenzioni del documento
   1.5 Glossario

2. Panoramica del sistema
   2.1 Cosa fa il sistema
   2.2 Capacità principali
   2.3 Diagramma di architettura ad alto livello
   2.4 Stack tecnologico

3. Architettura
   3.1 Architettura logica (Planner → Retriever → Writer → Critic)
   3.2 Responsabilità dei componenti (per modulo `src/aix/*`)
   3.3 Architettura a runtime
   3.4 Architettura dei dati
   3.5 Diagrammi di sequenza
   3.6 Architettura di deployment (Mode A vs Mode B)

4. Struttura del repository
   4.1 Layout di primo livello
   4.2 Mappa del package `src/aix/`
   4.3 Cartella `deploy/`
   4.4 Cartella `scripts/`
   4.5 Cartella `tests/`
   4.6 Cartella `docs/`

5. Riferimento API
   5.1 Panoramica della superficie API
   5.2 Autenticazione
   5.3 `POST /api/v1/agent/run` — JSON sincrono
   5.4 `POST /api/v1/agent/stream` — Server-Sent Events
   5.5 Tassonomia degli eventi SSE
   5.6 Schema del profilo educativo
   5.7 Endpoint legacy di contesto GraphRAG
   5.8 Modello degli errori

6. Pipeline degli agenti (interni)
   6.1 `AgentState` — il contratto condiviso
   6.2 Agente Planner
   6.3 Agente Retriever
   6.4 Agente Writer
   6.5 Agente Critic e ciclo di revisione
   6.6 Corrective RAG (opzionale)
   6.7 Orchestrazione LangGraph
   6.8 Memoria multi-turno & checkpointing
   6.9 Sistema dei prompt

7. Livello di retrieval
   7.1 Approccio GraphRAG
   7.2 Text2Cypher (multilingue)
   7.3 Retriever ibrido sul grafo (Neo4j + Node2Vec / embeddings)
   7.4 Context builder
   7.5 Embeddings & artifacts
   7.6 Fonti esterne (ibrido / out-of-scope)
   7.7 Schema del Knowledge Graph (domini UDL & Neuro)
   7.8 Livello media dinamici (live media)

8. Configurazione & ambiente
   8.1 Modello di configurazione
   8.2 Riferimento variabili d'ambiente
   8.3 Gestione dei segreti
   8.4 Profili di configurazione (dev / staging / prod)

9. Sviluppo locale
   9.1 Prerequisiti
   9.2 Installazione
   9.3 Esecuzione in locale
   9.4 Setup del Knowledge Graph
   9.5 Flussi di sviluppo comuni
   9.6 Note di configurazione dell'IDE

10. Strategia di testing
    10.1 Layout dei test & marker
    10.2 Esecuzione dei test
    10.3 Cosa è coperto vs cosa validare manualmente
    10.4 Linting & formattazione (Ruff)
    10.5 Aspettative di CI

11. Deployment
    11.1 Immagine container
    11.2 Stack di produzione (`deploy/docker-compose.prod.yml`)
    11.3 Reverse proxy & TLS (Caddy)
    11.4 Migrazione DB & checkpointer a PostgreSQL
    11.5 Modalità di deployment (Mode A standalone / Mode B nativo)
    11.6 Pipeline CI/CD
    11.7 Rollback & recovery

12. Osservabilità
    12.1 Tracing LLM (Langfuse)
    12.2 Monitoraggio errori (GlitchTip / Sentry)
    12.3 Health check
    12.4 Connectivity probe
    12.5 Logging strutturato

13. Sicurezza
    13.1 Autenticazione
    13.2 Autorizzazione
    13.3 Politica CORS
    13.4 Gestione dei segreti
    13.5 Rate limiting
    13.6 Isolamento di rete
    13.7 Trasparenza IA & allineamento EU AI Act

14. Pattern di integrazione
    14.1 Mode A — WebUI standalone
    14.2 Mode B — integrazione nativa AixLearning
    14.3 Regole di coesistenza tra le modalità
    14.4 Wrapper di riferimento (pattern `AgenticGraphRagService`)

15. Performance & SLO
    15.1 Budget di latenza per fase
    15.2 Target del primo evento in streaming
    15.3 Target SLO end-to-end (pilot)
    15.4 Assunzioni di costo & capacità

16. Runbook operativo
    16.1 Incidenti comuni
    16.2 Guida al debugging
    16.3 Restart / redeploy
    16.4 Manutenzione del database

17. Roadmap & limitazioni note
    17.1 Elementi rimandati
    17.2 Concorrenza & scaling
    17.3 Migrazione RS256 / JWT multi-issuer
    17.4 Evoluzione del frontend
    17.5 Altre limitazioni note

18. Appendici
    A. Riferimento variabili d'ambiente (tabella completa)
    B. Glossario
    C. Documenti correlati

---

## 1. Introduzione

### 1.1 Scopo

Questo documento è il **riferimento tecnico** del sistema Agentic GraphRAG. Descrive l'architettura, l'organizzazione del codice, l'API pubblica, il modello di deployment e le procedure operative della piattaforma. È pensato per permettere a un ingegnere che non ha mai visto il progetto di:

- comprendere il sistema ad alto livello in pochi minuti,
- individuare il modulo o l'endpoint giusto nel repository,
- integrare il sistema da un altro servizio (es. AixLearning),
- effettuare deployment, gestione operativa e troubleshooting in produzione,
- estendere in sicurezza la pipeline degli agenti, il livello di retrieval o la WebUI.

Questo documento **non** è materiale di marketing né una panoramica esecutiva. Per quella prospettiva, vedere il documento companion `docs/release/Functional_Documentation.md`, che descrive il sistema dal punto di vista di prodotto, pedagogico e di business.

### 1.2 Destinatari

Il documento è scritto per diversi ruoli di ingegneria. Alcune sezioni sono più rilevanti per certi lettori, ma il documento è pensato per essere leggibile dall'inizio alla fine.

- **Backend engineer** che modificheranno, estenderanno o faranno debugging della pipeline degli agenti, del livello di retrieval, della superficie FastAPI o della WebUI.
- **Integratori / team DEV** di piattaforme partner (in particolare il team DEV di AixLearning) che chiameranno l'API pubblica da un altro servizio.
- **DevOps / Operations** responsabili del deployment di produzione, dell'osservabilità, dei backup e della gestione degli incidenti.
- **Ingegneri di sicurezza e compliance** che devono verificare autenticazione, gestione dei segreti, isolamento di rete e controlli EU AI Act / GDPR.
- **Nuovi membri del team** in onboarding nell'AI Team che hanno bisogno di un singolo punto d'ingresso al codice.

Si assume che il lettore abbia familiarità con Python, I/O asincrono, FastAPI, Docker, REST/SSE e concetti base di database a grafo. Non si assume alcuna conoscenza pregressa di LangChain, LangGraph o di questo specifico repository.

### 1.3 Ambito

Questo documento copre:

- l'**architettura** del sistema Agentic GraphRAG, inclusa la pipeline multi-agente, il livello di retrieval, la WebUI e l'API pubblica;
- il **layout del repository**, con una mappa dei package `src/aix/*` e delle cartelle di supporto;
- il **contratto dell'API pubblica** (`/api/v1/*`), inclusi gli schemi request/response, l'autenticazione, la tassonomia degli eventi SSE e le garanzie di stabilità;
- gli **interni della pipeline degli agenti** (Planner, Retriever, Writer, Critic), inclusi state machine, checkpointer e profilo educativo;
- il **livello di retrieval** (Knowledge Graph Neo4j, retrieval ibrido, API esterne);
- lo **sviluppo locale** (setup dell'ambiente, esecuzione di API e WebUI, test, code style basato su Ruff);
- il **deployment** del pilot di produzione standalone (Docker Compose, Caddy, Postgres, backup, rollback);
- **osservabilità e sicurezza** (Langfuse, GlitchTip, health check, CORS, segreti, isolamento di rete);
- i **pattern di integrazione** per le due modalità di deployment supportate (Mode A standalone, Mode B nativo AixLearning);
- **target di performance, SLO e runbook operativi** al livello richiesto per un deployment pilot;
- la **roadmap** degli elementi esplicitamente rimandati e le limitazioni note.

Questo documento **non** copre:

- il modello pedagogico in profondità (coperto dalla documentazione funzionale);
- il codice Django interno di AixLearning (coperto dalla documentazione del team AixLearning);
- l'analisi normativa approfondita (EU AI Act / UNI 11621-8), mantenuta in un documento separato;
- il piano di deployment del pilot interno con tempistiche e ownership, mantenuto in un documento separato.

Il documento si concentra sullo stato attuale del codice più gli artefatti di deployment già presenti in `deploy/`.

### 1.4 Convenzioni del documento

Il documento usa le seguenti convenzioni per restare non ambiguo e facile da scorrere.

- I **riferimenti al codice** usano i backtick per i percorsi dei moduli (es. `src/aix/agent/orchestrator.py`), nomi di funzioni/classi (es. `stream_agent_events`), endpoint (es. `POST /api/v1/agent/stream`), variabili d'ambiente (es. `LANGGRAPH_DATABASE_URL`) e comandi shell.
- I **blocchi di codice** usano la sintassi fenced. Gli esempi di configurazione usano la sintassi delle variabili d'ambiente (`KEY=value`); gli esempi shell sono marcati esplicitamente quando sono PowerShell vs Bash.
- Gli **esempi API** seguono le forme JSON definite sotto `src/aix/api/schemas/*` e la spec OpenAPI servita su `/openapi.json`. Dove possibile, gli esempi rispecchiano quelli della Swagger UI ("minimal" e "rich").
- I **diagrammi** sono mantenuti semplici e ASCII-friendly dove aiutano la comprensione; i diagrammi di architettura più grandi vivono come PNG sotto `docs/mockups/`.
- I **callout di stato** usano prefissi semplici: `Nota:`, `Attenzione:`, `Limitazione:` e `Raccomandazione:`. Non sono box stilizzati, per mantenere il documento portabile.
- Le **promesse di stabilità** sono dichiarate esplicitamente. Tutto ciò che non è marcato esplicitamente come stabile va considerato soggetto a modifiche e non dovrebbe essere su cui fanno affidamento gli integratori esterni.
- I **riferimenti a file** del repository (codice, script, artefatti di deployment) usano percorsi relativi alla root del repository (es. `src/aix/agent/orchestrator.py`, `deploy/Caddyfile`) così da restare validi anche quando il documento viene letto offline o in un tool di rendering diverso.

### 1.5 Glossario

I termini comuni usati in tutto il documento sono elencati qui. Un glossario più esaustivo è fornito nell'Appendice B.

- **Agentic GraphRAG** — il sistema complessivo descritto da questo documento: una pipeline multi-agente che produce piani di lezione ancorati a un Knowledge Graph e integrati da fonti esterne.
- **Agent** — un'unità discreta di ragionamento nella pipeline (Planner, Retriever, Writer, Critic), implementata come nodo in una state machine LangGraph.
- **LangGraph** — un framework a state machine per orchestrare workflow LLM multi-step, usato per comporre ed eseguire la pipeline degli agenti.
- **GraphRAG** — Retrieval-Augmented Generation in cui il livello di retrieval è costruito su un Knowledge Graph anziché (solo) su un vector store.
- **Knowledge Graph (KG)** — il grafo Neo4j di concetti e relazioni pedagogiche (domini UDL e Neuroscienze) usato per il retrieval ancorato.
- **Profilo educativo** — la descrizione strutturata di una classe, di un'aula e del contesto di tempo/materia usata dagli agenti per specializzare l'output.
- **Checkpointer** — il componente LangGraph che persiste lo stato della conversazione (memoria multi-turno) su uno store di supporto; SQLite in sviluppo, PostgreSQL in produzione.
- **Ciclo Critic** — il ciclo di revisione in cui l'agente Critic valuta l'output del Writer e può richiedere una revisione.
- **SSE (Server-Sent Events)** — il protocollo di streaming usato da `POST /api/v1/agent/stream` per emettere eventi per fase verso i client.
- **WebUI** — l'interfaccia web teacher-facing interna servita dallo stesso processo FastAPI su `/webui/*`, usata per il pilot interno standalone.
- **Mode A / Mode B** — le due modalità di deployment supportate (WebUI standalone vs integrazione nativa AixLearning); vedere §3.6 e §14.
- **AixLearning** — la piattaforma Django partner che integra il servizio Agentic GraphRAG in Mode B.
- **Langfuse / GlitchTip** — i tool di osservabilità di terze parti usati per tracing e monitoraggio errori.
- **Caddy** — il reverse proxy usato in produzione, che termina il TLS via Let's Encrypt e inoltra il traffico al container FastAPI.

---

## 2. Panoramica del sistema

### 2.1 Cosa fa il sistema

Il sistema Agentic GraphRAG trasforma una richiesta in linguaggio naturale di un docente (in italiano o inglese) in un piano di lezione strutturato e pedagogicamente ancorato. Una richiesta come *"Crea una lezione di 45 minuti sulla fotosintesi adattata a una classe con 2 studenti DSA"* viene elaborata da una pipeline multi-agente che:

1. **comprende** la richiesta (rilevamento di intento + scope),
2. **recupera** conoscenza ancorata da un Knowledge Graph Neo4j e, quando serve, da fonti esterne verificate,
3. **scrive** un piano di lezione completo specializzato sul profilo della classe,
4. **revisiona** il risultato rispetto a criteri di qualità e lo rivede se necessario.

Il sistema espone due modalità complementari dalla **stessa applicazione FastAPI**:

- una **WebUI docente standalone** (`/webui/*`) usata dal pilot interno FEM, e
- un'**API pubblica JSON + SSE** (`/api/v1/agent/*`) consumata da client non-browser (il backend AixLearning, Postman/curl, app future).

Entrambe le modalità pilotano la **stessa pipeline di agenti** e lo **stesso livello di retrieval**.

### 2.2 Capacità principali

- **Orchestrazione multi-agente** (Planner → Retriever → Writer → Critic) su una state machine LangGraph.
- **Rilevamento dell'intento** su 7 tipi di query (creazione lezione, design di attività, definizione, confronto, spiegazione, raccomandazione, elenco).
- **Rilevamento dello scope** rispetto al Knowledge Graph (`in_scope` / `partial_scope` / `out_of_scope`).
- **Retrieval ibrido**: traversal del grafo Neo4j + embeddings Node2Vec/semantici + media curati + fonti esterne verificate (Wikipedia, OpenAlex, OER, YouTube).
- **Specializzazione per profilo educativo**: ogni richiesta può portare un profilo strutturato di classe/aula (grado, BES/DSA, risorse, budget di tempo).
- **UX docente guidata**: nella WebUI il primo turno può fermarsi al Planner, mostrare una scheda di selezione dell'intento pedagogico e avviare la pipeline completa solo dopo conferma del docente.
- **Libreria e riuso**: lezioni, storico multi-turno, profili educativi salvati e export MD/TXT/PDF sono gestiti nella WebUI standalone.
- **Memoria conversazionale multi-turno** tramite un checkpointer LangGraph (SQLite in dev, PostgreSQL in produzione), con windowing a summary-buffer per i thread lunghi.
- **Streaming**: Server-Sent Events per fase per una UI incrementale; streaming live dei token del Writer nella WebUI.
- **Controllo qualità**: un agente Critic valuta la bozza e può innescare un ciclo di revisione limitato.
- **Corrective RAG opzionale**: un ciclo di grading del retrieval (off di default, controllato da flag).
- **Server di tool MCP**: le capacità di KG, media e agenti sono esposte anche via Model Context Protocol (stdio + Streamable HTTP).
- **Osservabilità**: tracing Langfuse + monitoraggio errori GlitchTip/Sentry + un connectivity probe all'avvio.

### 2.3 Diagramma di architettura ad alto livello

```
                         ┌────────────────────────────────────────────┐
   Browser (teacher)     │                FastAPI app                  │
   ───────────────►──────┤  (1+ uvicorn worker — aix.api.main:app)     │
   /webui/*  (HTML+SSE)   │                                            │
                          │   /webui/*        Teacher WebUI (htmx)      │
   Non-browser client     │   /api/v1/context Legacy GraphRAG API       │
   ───────────────►──────┤   /api/v1/agent/* Agent API (JSON + SSE)    │
   /api/v1/agent/* (JSON) │   /mcp/           MCP Streamable HTTP        │
                          │   /auth/jwt/*     JWT login                  │
                          │   /docs /openapi.json  Swagger / OpenAPI    │
                          │                                            │
                          │   ┌──────────────── Agent pipeline ──────┐ │
                          │   │ Planner → Retriever → Writer → Critic │ │
                          │   │            (LangGraph)        ↑   ↓   │ │
                          │   │                               └revise┘ │ │
                          │   └───────┬───────────────┬──────────────┘ │
                          └───────────┼───────────────┼────────────────┘
                                      │               │
                              ┌───────▼──────┐  ┌─────▼─────────────┐
                              │   Neo4j KG   │  │ PostgreSQL (prod) │
                              │ (UDL/Neuro)  │  │ webui + langgraph │
                              └──────────────┘  └───────────────────┘
                                      │
                              ┌───────▼──────────────────────────────┐
                              │ External sources / LLM provider       │
                              │ OpenRouter (LLM), Wikipedia, OpenAlex, │
                              │ OER, YouTube                          │
                              └───────────────────────────────────────┘
```

### 2.4 Stack tecnologico

| Livello | Tecnologia |
|---|---|
| API + serving | FastAPI (`aix.api.main:app`, 1+ worker uvicorn), `sse-starlette` |
| Orchestrazione agenti | LangChain + LangGraph |
| Knowledge Graph | Neo4j (istanza Aura / gestita da FEM) |
| Embeddings | Node2Vec (grafo) + text embeddings OpenAI-compatible (ibrido) |
| Provider LLM | OpenRouter (default `anthropic/claude-sonnet-4-6`); OpenAI come fallback |
| Stato / memoria | Checkpointer LangGraph — SQLite (dev) / PostgreSQL (prod) |
| Persistenza WebUI | SQLAlchemy async — SQLite (dev) / PostgreSQL (prod) |
| Frontend WebUI | Jinja2 + htmx 2 + WebAwesome + Tailwind + Alpine.js |
| Auth | FastAPI-Users (cookie + Bearer JWT, HS256) |
| Protocollo tool | FastMCP 3.x (stdio + Streamable HTTP) |
| Reverse proxy / TLS | Caddy 2 (Let's Encrypt) |
| Osservabilità | Langfuse (tracing), GlitchTip/Sentry (errori) |
| Packaging / tooling | `pyproject.toml` (src layout), Ruff (lint+format), mypy, pytest |

---

## 3. Architettura

### 3.1 Architettura logica (Planner → Retriever → Writer → Critic)

La pipeline è uno `StateGraph` LangGraph i cui nodi sono i quattro agenti. Uno `AgentState` condiviso (TypedDict) attraversa ogni nodo; ciascun nodo legge i campi che gli servono e riscrive i propri output.

```
plan ──► retrieve ──► write ──► critique ──► [revise | finish]
                                   ▲              │
                                   └──────────────┘   (bounded revision loop)
```

- **Planner** (`plan`) — classifica intento + scope, estrae i concetti chiave e produce le query di ricerca.
- **Retriever** (`retrieve`) — esegue ricerche GraphRAG su Neo4j, allega i media curati e (per i topic out-of-scope) le fonti esterne verificate.
- **Writer** (`write`) — genera il piano di lezione, specializzato dal prompt di dominio + profilo educativo + eventuale contesto fornito dal docente.
- **Critic** (`critique`) — valuta la bozza; in caso di non approvazione (ed entro `max_revisions`) reinstrada verso `write` con istruzioni di revisione.

Quando `AIX_CORRECTIVE_RAG_ENABLED=true`, un nodo extra `grade_retrieval` viene inserito tra `retrieve` e `write`, con un arco di retry limitato verso `retrieve` (vedere §6.6).

### 3.2 Responsabilità dei componenti (per modulo `src/aix/*`)

| Package | Responsabilità |
|---|---|
| `aix.core` | Configurazione condivisa (`config.py`), connectivity probe, utility trasversali |
| `aix.retrieval` | Retrieval GraphRAG: Text2Cypher, retriever ibrido sul grafo, context builder, metriche di query |
| `aix.generation` | Generazione di risposte LLM per il percorso GraphRAG legacy (`llm_chain.py`) |
| `aix.agent` | Pipeline agentica: orchestrator, graph/nodes/state LangGraph, i 4 agenti, prompt, media, tool, config dei prompt di dominio |
| `aix.api` | App FastAPI (`main.py`), route (`context`, `agent`), schemi Pydantic, helper client |
| `aix.webui` | WebUI docente: auth, lezioni (CRUD/upload), servizio di streaming agenti, template Jinja2, DB |
| `aix.mcp` | Server di tool MCP: composition root, entry stdio, factory Streamable HTTP, tools/resources/prompts |
| `aix.domains` | Config di dominio e conoscenza dei prompt per UDL e Neuroscienze |

### 3.3 Architettura a runtime

La stessa app uvicorn (`aix.api.main:app`) ospita ogni superficie; in produzione può essere eseguita con uno o più worker. All'import/avvio di ciascun worker:

- applica uno shim della event-loop policy di Windows quando Postgres è configurato (psycopg async richiede il selector loop su Windows; la produzione Linux non è interessata);
- inizializza opzionalmente Sentry/GlitchTip quando `SENTRY_DSN` è impostata;
- costruisce la sub-app MCP Streamable HTTP (protetta — un fallimento di build non può bloccare `/api/v1`);
- all'avvio `lifespan`: verifica la connettività Neo4j, controlla le config di dominio, esegue il connectivity probe LLM opzionale e configura il checkpointer LangGraph;
- monta i router: `/api/v1/context`, `/api/v1/agent/*`, `/webui/*`, `/auth/jwt/*`, `/static` e `/mcp/`.

La pipeline degli agenti è pilotata da `aix.webui.agent.service`, che possiede il loop `astream` di LangGraph e la traduzione dei diff di stato in `StreamEvent` normalizzati. Il grafo compilato e i suoi agenti sono singleton a livello di modulo (accettabile per il pilot; da rivalutare per alta concorrenza — vedere §15/§17).

### 3.4 Architettura dei dati

- **Knowledge Graph Neo4j** — concetti pedagogici, metodologie, strategie e loro relazioni, in due domini (`neuro`, `udl`). Sola lettura a runtime. I dump sorgente vivono sotto `data/kg/{neuro,udl}/`.
- **PostgreSQL (produzione)** — una singola istanza che supporta due concerni logici:
  - **DB WebUI** (`WEBUI_DATABASE_URL`): utenti, lezioni, messaggi delle lezioni (trascritto multi-turno).
  - **Checkpointer LangGraph** (`LANGGRAPH_DATABASE_URL`): tre tabelle (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) che memorizzano lo stato per thread.
  - In sviluppo entrambi usano di default file SQLite.
- **Stato dell'agente** — il TypedDict `AgentState` in-flight (vedere §6.1), persistito per `thread_id` dal checkpointer per abilitare i follow-up multi-turno.
- **Artifacts** — embeddings Node2Vec e cache degli embeddings OpenAI sotto `artifacts/` (montati come volume Docker in produzione, così una rebuild non innesca un re-embed completo). La cartella `artifacts/media_cache/` è una cache a runtime del livello media dinamici (diskcache, vedere §7.8): è git-ignored e rigenerabile.

### 3.5 Diagrammi di sequenza

**Run sincrona (`POST /api/v1/agent/run`):**

```
Client → /api/v1/agent/run (JSON)
  → auth (cookie or Bearer JWT)
  → stream_agent_events(...) drained to completion
      plan → retrieve → write → critique [→ revise]
  → assemble AgentRunResponse { lesson_plan_md, meta, planner, retriever }
  → 200 JSON   (or 502 if the pipeline errored)
```

**Run in streaming (`POST /api/v1/agent/stream`):**

```
Client → /api/v1/agent/stream (JSON body)  →  text/event-stream
  event: planner   data: {...}
  event: retriever data: {...}
  event: writer_pending
  event: writer    data: {...}  lesson_plan_md: "<draft>"
  event: critic    data: {...}
  (… revision loop may repeat writer_pending/writer/critic …)
  event: done      lesson_plan_md: "<final>"  data(meta): {...}
  # heartbeat ping every 15s keeps proxies from closing the idle connection
```

### 3.6 Architettura di deployment (Mode A vs Mode B)

- **Mode A — Pilot interno standalone.** Browser → Caddy (80/443, TLS) → `app:8765`. L'app serve `/webui/*` e `/api/v1/*`. PostgreSQL è privato alla rete Docker. È il pilot interno FEM su `https://agente.aiforlearning.digital`.
- **Mode B — Integrazione nativa AixLearning.** Il backend/worker Django di AixLearning chiama il servizio agenti sulla rete Docker interna (`http://graphrag-api:8765/api/v1/agent/...`). AixLearning possiede la propria UX e i propri dati; il servizio agenti possiede solo il proprio stato.

La trattazione completa è in §14.

---

## 4. Struttura del repository

### 4.1 Layout di primo livello

Il progetto usa il moderno **src layout**: tutto il codice importabile vive sotto `src/aix/`, esposto come package `aix.*` tramite `pip install -e .`.

```
<repo-root>/
├── src/aix/            # All importable source (import as aix.*)
├── apps/               # User-facing entry points (NOT importable libs)
│   ├── streamlit/      #   Legacy Streamlit demo (retirement banner)
│   └── cli/run_agent.py#   Interactive agent testing CLI
├── scripts/            # Operational & data-prep scripts
│   ├── ingest/ audit/ data_prep/ ml/ diagnostic/ ops/ media_pool/
├── data/               # KG dumps, media mappings, reference, reports
│   └── kg/{neuro,udl}/ #   Knowledge graph core dumps
├── artifacts/          # ML artifacts: node2vec/, embeddings_cache/
├── tests/              # unit / integration / api / mcp_server suites
├── deploy/             # Production stack (compose, Caddyfile, .env.prod.example, scripts)
├── docs/               # Documentation (api, architecture, integrations, product, release, …)
├── Dockerfile          # Container build (api target, non-root, healthcheck)
├── requirements.txt    # Runtime deps (single source of truth)
├── requirements.lock.txt # Hash-pinned lockfile for prod (Python 3.12 / Linux)
├── pyproject.toml      # Build, deps, pytest, ruff, mypy
└── Makefile            # make test / api / streamlit / agent
```

### 4.2 Mappa del package `src/aix/`

```
src/aix/
├── core/
│   ├── config.py             # `from aix.core.config import config` — Neo4j/LLM/embeddings config
│   └── connectivity_probe.py # one-shot startup LLM-endpoint probe (TLS/DNS/401/timeout)
├── retrieval/                # GraphRAG retrieval layer
│   ├── text2cypher.py            # NL → Cypher conversion (+ self-repair)
│   ├── multilingual_text2cypher.py # IT/EN translation wrapper around text2cypher
│   ├── graph_retriever.py        # hybrid graph + vector retrieval (Node2Vec/OpenAI)
│   ├── context_builder.py        # raw graph data → structured educational context
│   └── query_metrics.py          # retrieval telemetry
├── generation/
│   └── llm_chain.py          # legacy GraphRAG response generation
├── agent/                    # Agentic GraphRAG (multi-agent pipeline)
│   ├── orchestrator.py       # `from aix.agent import AgentOrchestrator` — clean entry point
│   ├── agents/               # planner_agent, retriever_agent, writer_agent, critic_agent
│   ├── graph/                # LangGraph: state.py, nodes.py, lesson_planner_graph.py,
│   │                         #   checkpointer.py, write_stream.py
│   ├── prompts/              # intent-specific prompts (planner/writer/critic)
│   ├── media/                # media lookup, diagram/image generation, live media, resource lookup
│   ├── tools/                # GraphRAG + curriculum tool wrappers for agents
│   └── configs/              # domain prompt extensions (domain_prompts.py)
├── api/                      # FastAPI service
│   ├── main.py               # uvicorn aix.api.main:app — mounts every surface
│   ├── routes/context.py     # /api/v1/context (legacy GraphRAG)
│   ├── routes/agent.py       # /api/v1/agent/run + /stream (Agent API)
│   ├── schemas/              # Pydantic models (agent.py, models.py, educational_profile.py)
│   └── graphrag_client.py    # helper client for the DEV team
├── webui/                    # Teacher WebUI (htmx + WebAwesome)
│   ├── auth/                 # FastAPI-Users: manager, backend, dependencies, models, routes
│   ├── lessons/              # lesson CRUD, uploads, saved profiles, export, display
│   ├── agent/service.py      # run_agent_stream + stream_agent_events (the engine seam)
│   ├── templates/            # Jinja2: _base.html, pages/, partials/
│   ├── routes.py             # /webui/* handlers
│   └── db.py                 # SQLAlchemy async engine (+ aiosqlite dev default)
├── mcp/                      # MCP Tool Servers (FastMCP 3.x)
│   ├── server.py             # build_mcp_server() composition root
│   ├── stdio_main.py         # stdio entry (Claude Desktop / Cursor IDE)
│   ├── http_app.py           # Streamable HTTP factory (JWT Bearer at /mcp/)
│   ├── tools/ resources/ prompts/
└── domains/                  # Domain configs (udl_domain.py, neuro_domain.py, base_config.py)
```

Nota frontend: `src/aix/webui/static/css/aix-brand.css` è la fonte di verità per lo styling teacher-facing. I componenti WebAwesome forniscono i widget base, le classi `.aix-*` possiedono brand, colori, tipografia, card e stati, mentre Tailwind resta limitato al layout (`flex`, `grid`, spacing). Evitare nuovi `style="..."` inline nei template: se un valore visivo ricorre o appartiene al brand, aggiungerlo come token/classe in `aix-brand.css`.

### 4.3 Cartella `deploy/`

Stack di produzione e runbook (Wave 1 del piano di deployment interno):

```
deploy/
├── docker-compose.prod.yml  # app + postgres(16) + caddy(2); app/postgres internal-only
├── Caddyfile                # reverse proxy + auto Let's-Encrypt TLS (parameterized by $AIX_DOMAIN)
├── .env.prod.example        # production env template (copy → .env.prod, chmod 600)
├── scripts/                 # backup_postgres / restore_postgres / backup_caddy
└── README.md                # first-deploy, backups, rollback, log inspection runbook
```

### 4.4 Cartella `scripts/`

Tooling operativo e di data-prep, raggruppato per scopo: `ingest/` (import/export Neo4j), `audit/` (controlli sulle label del KG), `data_prep/` (pulizia/merge), `ml/` (training Node2Vec, generazione media mapping), `diagnostic/` (smoke test MCP, cattura baseline OpenAPI), `ops/` (preflight, migrazioni), `media_pool/` (generazione offline del pool media via LM Studio).

### 4.5 Cartella `tests/`

`unit/` (puri, senza servizi esterni), `integration/` (Neo4j/LLM — marcati `@pytest.mark.integration`), `api/` (test di contratto dell'API agenti), `mcp_server/` (suite di regressione MCP da 19 test), più `conftest.py` con fixture condivise. Pytest è configurato in `pyproject.toml` con i marker `integration`, `slow`, `unit` e `asyncio_mode = "auto"`.

### 4.6 Cartella `docs/`

`api/` (guide di integrazione), `architecture/` (valutazioni frontend/ADR, analisi dei modelli), `integrations/` (setup MCP), `product/` (documenti interni di prodotto e processo), `release/` (questo documento + la documentazione funzionale), più report e runbook.

---

## 5. Riferimento API

### 5.1 Panoramica della superficie API

Tutte le superfici HTTP sono servite dalla singola app FastAPI (`aix.api.main:app`). La spec OpenAPI è pubblicata su `/openapi.json`, con doc interattive su `/docs` (Swagger UI) e `/redoc`.

| Percorso | Metodo | Scopo | Auth |
|---|---|---|---|
| `/api/v1/health` | GET | Probe di liveness/readiness | nessuna |
| `/api/v1/context` | POST | Contesto GraphRAG legacy (retrieval single-shot) | service auth |
| `/api/v1/agent/run` | POST | Esegue l'agente, ritorna il piano di lezione finale (JSON sincrono) | cookie o Bearer JWT |
| `/api/v1/agent/stream` | POST | Esegue l'agente, fa streaming delle fasi come SSE JSON | cookie o Bearer JWT |
| `/auth/jwt/login` | POST | Emette un Bearer JWT (FastAPI-Users) | credenziali |
| `/webui/*` | GET/POST | WebUI docente (HTML + htmx + SSE) | cookie session |
| `/mcp/` | (MCP) | Transport MCP Streamable HTTP | Bearer JWT |
| `/docs`, `/redoc`, `/openapi.json` | GET | Documentazione API | nessuna |

Le due superfici che il team di integrazione consuma sono **`/api/v1/agent/run`** e **`/api/v1/agent/stream`**. Sono *additive* — non alterano `/api/v1/context`, `/webui/*` o `/auth/*`.

### 5.2 Autenticazione

Entrambi gli endpoint dell'agente dipendono da `current_active_user` (FastAPI-Users), che accetta **uno tra**:

- il **cookie di sessione WebUI** (usato dai client browser), oppure
- un header **`Authorization: Bearer <jwt>`** (usato dai client API/integrazione).

I token sono HS256, firmati con `WEBUI_AUTH_SECRET` (condiviso da backend cookie e Bearer). Per ottenere un Bearer token a livello programmatico:

```bash
curl -X POST https://agente.aiforlearning.digital/auth/jwt/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=teacher@example.org&password=•••••"
# → { "access_token": "<jwt>", "token_type": "bearer" }
```

Nota: l'endpoint legacy `/api/v1/context` usa uno schema separato di auth service-to-service (HTTP Basic), come documentato nella guida di integrazione; non è interessato dal layer FastAPI-Users.

### 5.3 `POST /api/v1/agent/run` — JSON sincrono

Esaurisce l'intera pipeline **Planner → Retriever → Writer → Critic** e ritorna una singola risposta. Run tipica: **60–120 s** (domina la chiamata LLM del Writer). Per una UI incrementale usare invece `/stream`.

**Body della richiesta** (`AgentRunRequest`):

| Campo | Tipo | Obbl. | Note |
|---|---|---|---|
| `query` | string (3–2000) | ✅ | Richiesta del docente in linguaggio naturale |
| `domain` | `"neuro"` \| `"udl"` | ✅ | Dominio del knowledge graph |
| `language` | `"it"` \| `"en"` | — | Lingua di output (default `it`) |
| `session_id` | string (≤128) | — | Id di correlazione/thread; UUID4 generato se omesso |
| `educational_profile` | `EducationalProfile` | — | Contesto di classe/aula (CORE 1 #2.5) |
| `teacher_provided_context` | string (≤48000) | — | Testo unito dagli upload; solo Writer, **non** ingerito nel KG |
| `max_revisions` | int (0–4) | — | Cap del ciclo Critic; `null` → `AIX_MAX_REVISIONS` (default 1); `0` disabilita |

Richiesta minimale:

```json
{ "query": "Crea una lezione sull'attenzione", "domain": "neuro" }
```

**Risposta** (`AgentRunResponse`):

```json
{
  "lesson_plan_md": "# Lezione: ...",
  "meta": {
    "duration_seconds": 73.2,
    "approved": true,
    "revision_count": 0,
    "scores": { "average_score": 4.4 },
    "nodes_count": 14,
    "recommendations_count": 5,
    "media_counts": { "videos": 3, "articles": 2, "oer": 4 },
    "search_queries_count": 4
  },
  "planner": {
    "intent": "lesson_creation", "intent_label": "Creazione lezione",
    "scope": "in_scope", "scope_label": "Nel Knowledge Graph",
    "key_concepts": ["attenzione", "memoria di lavoro"],
    "search_queries": ["strategie attenzione DSA", "..."]
  },
  "retriever": {
    "nodes_count": 14, "relationships_count": 31, "recommendations_count": 5,
    "top_concepts": ["Attenzione sostenuta", "Self-regulation"],
    "retrieval_confidence": "HIGH"
  }
}
```

**Codici di stato:** `200` successo · `401` auth mancante/non valida · `422` errore di validazione del body · `502` errore a runtime della pipeline agenti (fallimento LLM, KG irraggiungibile — potenzialmente ritentabile).

### 5.4 `POST /api/v1/agent/stream` — Server-Sent Events

Stesso body di richiesta di `/run`. Ritorna `text/event-stream`. Ogni frame porta una riga `event:` (il `kind`) e un payload JSON `data:`. Un commento di heartbeat è inviato ogni **15 s** così che proxy/load balancer non chiudano la connessione idle durante la lenta chiamata del Writer.

```bash
curl -N -X POST https://agente.aiforlearning.digital/api/v1/agent/stream \
  -H "Authorization: Bearer <jwt>" -H "Content-Type: application/json" \
  -d '{"query":"Crea una lezione sull'\''attenzione","domain":"neuro"}'
```

Forma sul filo per frame:

```
event: planner
data: {"kind":"planner","data":{...},"lesson_plan_md":null,"error":null}
```

> Nota: il "Try it out" della Swagger UI rende l'intero stream come un unico blob. Per l'ispezione evento-per-evento usare `curl -N`, Postman o Bruno.

### 5.5 Tassonomia degli eventi SSE

Lo stream pubblico (`stream_agent_events`) emette i 7 tipi di evento congelati dall'union `AgentStreamEvent` in `src/aix/api/schemas/agent.py`. Ogni frame ha lo stesso envelope esterno: `{ kind, data, lesson_plan_md, error }`.

| `kind` | Quando | Campi `data` principali |
|---|---|---|
| `planner` | dopo `plan` | `intent`, `intent_label`, `scope`, `scope_label`, `scope_confidence`, `key_concepts[]`, `search_queries[]` |
| `retriever` | dopo `retrieve` | `nodes_count`, `relationships_count`, `recommendations_count`, `media_counts{videos,articles,oer}`, `media{}`, `top_concepts[]`, `retrieval_confidence`, coverage tier |
| `writer_pending` | prima di un tentativo di scrittura | `revision`, `is_revision`, `feedback` |
| `writer` | dopo `write` | `revision`; `lesson_plan_md` = bozza per questa revisione |
| `critic` | dopo `critique` | `approved`, `revision_count`, `max_revisions`, `score`, `score_pct`, `critique`, `revision_instructions` |
| `done` | fine della run | (envelope) `lesson_plan_md` = finale; `data`/meta = riepilogo della run |
| `error` | in caso di fallimento | `error` = messaggio breve (≤480 char) |

Ordine happy-path (0 revisioni): `planner → retriever → writer_pending → writer → critic → done`.
Con una revisione: `planner → retriever → writer_pending → writer → critic → writer_pending → writer → critic → done`.

Nota implementativa: i client dovrebbero fare `switch` sul `kind` e ignorare i kind sconosciuti (forward-compatible). La WebUI browser usa kind aggiuntivi interni (`retriever_pending`, `critic_pending`, `writer_chunk` per lo streaming live dei token) che **non** fanno parte dell'union pubblica; i client non-browser non li riceveranno da `/api/v1/agent/stream`.

### 5.6 Schema del profilo educativo

`educational_profile` riusa lo stesso modello `EducationalProfile` che il form della WebUI serializza (`src/aix/api/schemas/educational_profile.py`), così un profilo è interscambiabile tra le due superfici. Porta:

- **`group`** — contesto di classe: `title`, `students_number`, `grade` (es. `SECONDARIA_II_GRADO`), `disabilities` (es. `ADHD`, `DSA`), `class_features`, `student_attributes`.
- **`classroom`** — contesto fisico: `title`, `forniture_mobility`, `has_lim`, `has_wifi`, `has_suite`, `pc_station`, `own_device`.
- **top-level** — `time_available_minutes`, `subject_area`, `specific_topic`.

Tutti i campi sono opzionali; omettere il profilo fa sì che l'agente ricada su prompt generici. La Swagger UI espone due esempi nominati — **minimal** e **rich** — guidati da `openapi_examples` in `routes/agent.py`.

### 5.7 Endpoint legacy di contesto GraphRAG

`POST /api/v1/context` (`routes/context.py`) è l'endpoint di retrieval single-shot originale che alimentava la prima integrazione AixLearning. Ritorna contesto GraphRAG strutturato (concetti, metodologie, raccomandazioni) senza eseguire la pipeline degli agenti. Resta supportato per retrocompatibilità; le nuove integrazioni dovrebbero preferire gli endpoint dell'agente.

### 5.8 Modello degli errori

- Gli endpoint dell'agente sollevano `502 Bad Gateway` (non `500`) quando fallisce la *pipeline* — comunicando che la route stessa non è crashata e che la richiesta può essere ritentabile.
- Lo stream SSE non lascia mai attraversare un'eccezione il confine del generator: i fallimenti sono emessi come evento terminale `error` (trattato come dato di dominio).
- Gli errori di validazione (`422`) seguono la forma standard FastAPI `{"detail": [...]}`.

---

## 6. Pipeline degli agenti (interni)

### 6.1 `AgentState` — il contratto condiviso

`src/aix/agent/graph/state.py` definisce `AgentState`, un `TypedDict(total=False)` che attraversa ogni nodo. È raggruppato in input, output per agente, campi corrective-RAG, metadati e output finale. Campi chiave:

- **Input**: `teacher_query`, `domain`, `language`, `session_id`, `educational_profile`, `pedagogical_intent`, `refinement_instruction`, `teacher_provided_context`, `conversation_history`, `conversation_summary`, `raw_user_turn`.
- **Output Planner**: `query_intent`, `lesson_type`, `target_grade`, `key_concepts`, `search_queries`, `scope_status`, `scope_confidence`, `subject_concepts`, `pedagogy_concepts`.
- **Output Retriever**: `graphrag_results`, `retrieved_nodes`, `retrieved_relationships`, `recommendations`, `retrieval_confidence`, `curated_media`, `external_resources`.
- **Corrective-RAG** (solo quando abilitato): `retrieval_grade`, `retrieval_grade_reason`, `retrieval_attempts`, `retrieval_rewritten_query`, `retrieval_warning`.
- **Output Writer**: `lesson_plan_draft`, `lesson_plan_structured`, `sources_cited`.
- **Output Critic**: `critique`, `critique_score`, `approved`, `revision_instructions`.
- **Metadati / finale**: `revision_count`, `max_revisions`, `current_step`, `error`, `final_lesson_plan`, `final_metadata`.

`create_initial_state(...)` è l'unica fonte di verità per la forma di input dell'agente; sia il servizio WebUI sia l'API pubblica costruiscono lo stato tramite essa. `pedagogical_intent` porta l'obiettivo didattico scelto dal docente (`"{code}"` o `"{code}: {detail}"`), mentre `refinement_instruction` è il comando transitorio prodotto dal pannello SAM quando il docente rigenera una lezione. I nuovi campi nullable sono pensati per essere additivi così che i chiamanti più vecchi si comportino in modo identico.

### 6.2 Agente Planner

`agent/agents/planner_agent.py` (nodo `plan`) esegue il **rilevamento dell'intento** (7 tipi `QueryIntent`), il **rilevamento dello scope** (`ScopeStatus`: `in_scope` / `partial_scope` / `out_of_scope` / `unknown`) ed estrae `key_concepts` + `search_queries`. Agisce anche come **layer linguistico L1**: vede la query aumentata completa e può sovrascrivere il seed linguistico statistico (L2 = detector `lingua`, L3 = default `it`). `plan_node` applica la precedenza utente-vs-storico sulla durata tramite `raw_user_turn` (il turno corrente non aumentato).

### 6.3 Agente Retriever

`agent/agents/retriever_agent.py` (nodo `retrieve`) esegue le `search_queries` del planner sul livello di retrieval GraphRAG (§7), aggrega nodi/relazioni/raccomandazioni e assembla `curated_media` (video, risorse, citazioni, open_textbooks). Per i topic `partial_scope`/`out_of_scope` allega `external_resources` (Wikipedia, OER, paper) così che il Writer possa comunque comporre una lezione utile. Il contesto caricato dal docente **non** viene mai inviato al retriever — è solo per il Writer.

### 6.4 Agente Writer

`agent/agents/writer_agent.py` (nodo `write`) genera il piano di lezione. Il suo prompt è assemblato dal **prompt specifico per intento** (`agent/prompts/`), dall'**estensione di prompt di dominio** (`agent/configs/`), dal **profilo educativo**, dall'eventuale **intento pedagogico confermato dal docente**, dal **contesto recuperato + media**, dall'eventuale **contesto fornito dal docente**, dall'eventuale **istruzione di raffinamento SAM** e da qualsiasi **storico/riassunto della conversazione**. La lunghezza dell'output è limitata da `AIX_WRITER_MAX_TOKENS` (default codice 8000; il template prod può impostare un cap più conservativo, es. 2000) con fino a `AIX_WRITER_MAX_CONTINUATIONS` continuazioni automatiche quando il modello raggiunge `finish_reason="length"`. Nella WebUI i token del writer fanno streaming live (`writer_chunk`); l'API pubblica consegna l'output del writer come singolo evento `writer` per revisione.

### 6.5 Agente Critic e ciclo di revisione

`agent/agents/critic_agent.py` (nodo `critique`) valuta la bozza su più criteri (media su scala 1–5) e ritorna una decisione approva/rivedi. `should_continue_to_revision` instrada la run:

- `approved == true` → `finish` (END);
- `not approved` **e** `revision_count < max_revisions` → `revise` (di nuovo a `write`) con `revision_instructions`;
- altrimenti → `finish`.

`max_revisions` ha default `AIX_MAX_REVISIONS` (1). Flag di robustezza: `AIX_CRITIC_PARSE_ERROR_BEHAVIOR` (`approve`/`revise`/`raise`) controlla il comportamento su JSON del critic non parsabile; `AIX_CRITIC_MODEL` (default un modello veloce) mantiene economica la classificazione da ~300 token; `AIX_CRITIC_LESSON_MAX_CHARS` / `AIX_CRITIC_CONTEXT_MAX_CHARS` limitano il prefill per ridurre la latenza.

### 6.6 Corrective RAG (opzionale)

Controllato da `AIX_CORRECTIVE_RAG_ENABLED` (default **off**). Quando attivo, `_build_workflow()` inserisce un nodo `grade_retrieval` dopo `retrieve`:

```
plan → retrieve → grade_retrieval ─[continue]→ write → critique → [revise|finish]
                        │
                        └─[retry]→ retrieve   (bounded by AIX_CORRECTIVE_RAG_MAX_ATTEMPTS, default 2)
```

Un LLM grader classifica il retrieval come `relevant` / `ambiguous` / `irrelevant`. I grade non rilevanti innescano un retry limitato con una query riscritta; esaurito il budget di tentativi, la run prosegue verso il Writer con `retrieval_warning=true` così che la lezione porti un'esplicita avvertenza di bassa confidenza. Quando il flag è off, la topologia è identica byte-per-byte alla pipeline pre-feature.

### 6.7 Orchestrazione LangGraph

`agent/graph/lesson_planner_graph.py` costruisce lo `StateGraph`. Due percorsi di compile condividono una topologia (`_build_workflow`):

- `build_lesson_planner_graph()` (sync, **senza** checkpointer) — per chiamanti legacy/effimeri.
- `build_lesson_planner_graph_async()` (async, **con** checkpointer quando disponibile) — usato dalla WebUI, dall'API pubblica e da MCP.

`LessonPlannerPipeline` avvolge il grafo compilato; `AgentOrchestrator` (`agent/orchestrator.py`) è il punto d'ingresso pubblico ed ergonomico (`create_lesson_plan(...)`). L'engine di esecuzione per lo streaming è `aix.webui.agent.service` (`run_agent_stream` per il percorso WebUI/DB, `stream_agent_events` per il percorso API senza DB) — entrambi pilotano `graph.astream(..., stream_mode="updates")` e traducono i diff di stato in `StreamEvent`.

### 6.8 Memoria multi-turno & checkpointing

`agent/graph/checkpointer.py` risolve il checkpointer da `LANGGRAPH_DATABASE_URL`: `AsyncPostgresSaver` per `postgresql[+driver]://…`, `AsyncSqliteSaver` per SQLite (il default dev `data/agent_threads.db`). Ogni run in streaming passa `thread_config(thread_id)`; la WebUI usa `str(lesson.id)` così i turni successivi condividono lo stato. Per i thread lunghi, il servizio applica un **windowing a summary-buffer** (`AIX_CONVERSATION_WINDOW_TURNS`, default 4): i turni più recenti sono mantenuti verbatim e quelli più vecchi sono riassunti via LLM in `conversation_summary`. La WebUI persiste anche un trascritto SQL (righe `lesson_message`) come fonte di verità agnostica dal dialetto che sopravvive a un wipe del checkpointer.

### 6.9 Sistema dei prompt

`agent/prompts/` contiene i builder di prompt specifici per intento (planner/writer/critic); `agent/configs/domain_prompts.py` contiene le estensioni specifiche per dominio (tono, terminologia e vincoli UDL vs Neuro). I testi di prompt di riferimento sono rispecchiati sotto `docs/prompts_reference/`. Questa separazione permette allo stesso codice degli agenti di specializzarsi per dominio senza logica condizionale.

---

## 7. Livello di retrieval

### 7.1 Approccio GraphRAG

Il retrieval è **graph-first**: il sistema ancora le risposte a un Knowledge Graph Neo4j curato anziché affidarsi solo a un vector store. Una query del docente viene convertita in Cypher, eseguita su Neo4j, opzionalmente espansa via similarità semantica e strutturata in un oggetto di contesto educativo che gli agenti consumano. È ciò che rende gli output *evidence-based* e auditabili.

### 7.2 Text2Cypher (multilingue)

`retrieval/text2cypher.py` converte il linguaggio naturale in Cypher; `retrieval/multilingual_text2cypher.py` aggiunge la traduzione Italiano→Inglese così che le query italiane corrispondano al vocabolario (in parte inglese) del grafo. Il Cypher generato è opzionalmente validato e auto-riparato prima dell'esecuzione. Tuning: `TEXT2CYPHER_MODEL` (un modello veloce/economico, default `google/gemini-2.5-flash`), `TEXT2CYPHER_MAX_QUERY_LENGTH`, `TEXT2CYPHER_DEFAULT_LIMIT`, `TEXT2CYPHER_ENABLE_VALIDATION`, `TEXT2CYPHER_ENABLE_EXECUTION` (impostare a false per dry-run).

### 7.3 Retriever ibrido sul grafo (Neo4j + Node2Vec / embeddings)

`retrieval/graph_retriever.py` combina tre segnali:

1. **Traversal diretto del grafo** — il Cypher generato.
2. **Ricerca semantica** — similarità di embedding sui nodi del grafo.
3. **Espansione dei vicini** — recupero dei concetti correlati per completezza.

Modalità di embedding (`EMBEDDING_MODE`):

| Modalità | Comportamento |
|---|---|
| `node2vec` | Solo embeddings di grafo pre-addestrati; veloce, nessuna chiamata API; language-blind |
| `hybrid_semantic` | `EMBEDDING_NODE2VEC_WEIGHT` (default 0.4) Node2Vec + (0.6) text embeddings; **raccomandato per la produzione** (gestisce italiano, sinonimi, parafrasi) |
| `openai_only` | Embeddings puramente semantici (da usare quando un dominio non ha modello Node2Vec) |

Impostazioni rilevanti: `EMBEDDING_MODEL` (default `openai/text-embedding-3-small`, 1536-dim), `EMBEDDING_SEMANTIC_THRESHOLD` (default 0.7), `EMBEDDINGS_CACHE_DIR`, `NODE2VEC_MODEL_DIR`. **Attenzione:** cambiare `EMBEDDING_MODEL` richiede di eliminare `artifacts/embeddings_cache/` così che i vettori di nodo e query restino dimensionalmente compatibili.

### 7.4 Context builder

`retrieval/context_builder.py` trasforma i risultati grezzi del grafo in un contesto educativo strutturato: raccomandazioni metodologiche con livelli di confidenza, raggruppamenti di concetti e una vista per profilo studente. Questo oggetto strutturato — non le righe grezze — è ciò su cui gli agenti ragionano, mantenendo i prompt compatti e coerenti.

### 7.5 Embeddings & artifacts

Gli artefatti pre-addestrati sono inclusi nel repo e montati come volume Docker in produzione:

- `artifacts/node2vec/{domain}_node2vec_embeddings.npz` — embeddings di grafo (128-dim; walk length 80, 200 walk).
- `artifacts/embeddings_cache/{domain}_openai_embeddings.json` — text embeddings in cache.

Re-training: `python scripts/ml/train_node2vec.py {neuro|udl}`. Il primo setup ibrido precalcola i text embeddings via `python -m aix.retrieval.graph_retriever --precompute {domain}`.

Nota: la cache del livello media dinamici vive separatamente sotto `artifacts/media_cache/` (vedere §7.8) ed è una cache a runtime git-ignored e rigenerabile, distinta da queste cache di embedding.

### 7.6 Fonti esterne (ibrido / out-of-scope)

Quando il Planner marca un topic come `partial_scope`/`out_of_scope`, il Retriever integra il KG con fonti esterne verificate via `agent/media/` (e `agent/tools/`): Wikipedia, OpenAlex/Semantic Scholar (citazioni accademiche), repository OER e YouTube. Per YouTube il recupero usa la **YouTube Data API** quando `YOUTUBE_API_KEY` è configurata (risultati limitati a `videoEmbeddable=true` e `safeSearch=strict`); senza chiave ricade su URL di ricerca e il pool curato resta il riferimento. `SEMANTIC_SCHOLAR_API_KEY` è opzionale e serve solo ad alzare i rate limit. Queste fonti riempiono i bucket `external_resources` / media così che la lezione resti utile anche fuori dalla copertura centrale del KG, mentre la UI segnala chiaramente l'ancoraggio ridotto al KG (coverage tier).

### 7.7 Schema del Knowledge Graph (domini UDL & Neuro)

Due domini sono serviti dalla stessa istanza Neo4j:

- **`neuro`** — concetti, metodologie e strategie di neuroscienze dell'apprendimento.
- **`udl`** — Universal Design for Learning (didattica inclusiva).

I nodi rappresentano concetti/metodologie/strategie educative; le relazioni codificano legami pedagogici (es. `BELONGS_TO`, `ADDRESSES`, `SUPPORTS`). La selezione del dominio è per richiesta (campo `domain`). I dump sorgente e il formato di contratto vivono sotto `data/kg/{neuro,udl}/` e `data/reference/JSON_reference.json`; l'ingestion è gestita da `scripts/ingest/data_ingestion_neo4j.py`.

### 7.8 Livello media dinamici (live media)

Oltre alle fonti esterne pilotate dal Retriever (§7.6), esiste un **livello media dinamici ("live")** separato, implementato in `src/aix/agent/media/` (`live_media.py`, `media_cache.py`, `media_config.py`, `media_ranker.py`). Recupera media dinamici (paper OpenAlex, voci Wikipedia, video YouTube) in base ai concetti/alla query della lezione e li restituisce nella **stessa forma dict** che il retriever produce per `external_resources`, così il pannello media li mostra senza modifiche ai template.

Proprietà di design (conformi al piano approvato):

- **Off-critical-path.** Nulla qui è chiamato dalla pipeline planner → retriever → writer → critic; è pensato per essere invocato da un worker off-path, quindi la latenza di generazione delle lezioni resta invariata.
- **Flag-gated.** Quando `AIX_MEDIA_LIVE_ENABLED` è false (default), ogni entry point è un no-op che ritorna `{}` — cioè il comportamento odierno.
- **Cache-first.** Il pool di candidati BROAD è letto/scritto da `MediaCache` (Redis in prod / diskcache in dev / null fallback), con chiave sui concetti, così che set di concetti ripetuti siano serviti istantaneamente e la quota esterna sia ammortizzata. La cache di sviluppo vive sotto `artifacts/media_cache/` (vedere §3.4) con TTL di 14 giorni.
- **Bounded & fail-safe.** Un timeout globale per chiamata limita il lavoro, ogni fallimento di sorgente è isolato e qualsiasi errore degrada a `{}` — senza mai sollevare verso il chiamante.
- **Re-ranking semantico (Phase 3).** Quando `AIX_MEDIA_RERANK_ENABLED` è true (default off), gli item live sono riordinati rispetto a query + contenuto della lezione con un punteggio combinato `w_semantic·cosine + w_quality·signal` (`media_ranker.py`); con il flag off gli item mantengono l'ordine di fetch.
- **Deduplica.** Voci Wikipedia deduplicate per URL canonico (concetti sinonimi risolvono alla stessa pagina), paper e video deduplicati per identificatore, per evitare duplicati nel pannello media.

Trasparenza UI: i media **curati** mostrano un badge `✓ Verificato`, quelli **dinamici** un badge `auto`. In ogni caso l'output della lezione resta contenuto generato dall'IA da rivedere da parte del docente (vedere §13.7).

---

## 8. Configurazione & ambiente

### 8.1 Modello di configurazione

La configurazione è guidata dall'ambiente. `src/aix/core/config.py` carica le variabili via `python-dotenv` (`load_dotenv()`) ed espone oggetti di config tipizzati (`Neo4jConfig`, `OpenAIConfig`, …) più un singleton `config`. **Nessun** segreto è nel source control: `.env` (dev) e `deploy/.env.prod` (prod) sono git-ignored. I template sono committati: `.env.example` (superficie dev completa) e `deploy/.env.prod.example` (sottoinsieme di produzione).

`OpenAIConfig` è agnostico rispetto al provider (OpenRouter o OpenAI) e rileva i modelli reasoning/"thinking" (Claude Sonnet/Opus 4.x, serie o, DeepSeek R1) per adattare automaticamente i parametri API (`max_completion_tokens` vs `max_tokens`, eliminando `temperature`, richiedendo reasoning token). È per questo che cambiare modello raramente richiede modifiche al codice.

### 8.2 Riferimento variabili d'ambiente

**Servizi core**

| Variabile | Default | Scopo |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Connessione Neo4j (`bolt+s://` per TLS/Aura) |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / — | Credenziali Neo4j |
| `NEO4J_ENCRYPTED` | `1` | Toggle TLS |
| `OPENROUTER_API_KEY` | — | Chiave del provider LLM (preferito) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint LLM |
| `OPENAI_API_KEY` | — | Fallback se OpenRouter non impostato |
| `LLM_MODEL` | `anthropic/claude-sonnet-4-6` | Modello primario di generazione lezioni |

**Retrieval / embeddings**

| Variabile | Default | Scopo |
|---|---|---|
| `EMBEDDING_MODE` | `hybrid_semantic` | `node2vec` / `hybrid_semantic` / `openai_only` |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Modello di text embedding (accoppiato alla cache) |
| `EMBEDDING_NODE2VEC_WEIGHT` | `0.4` | Peso ibrido (quota Node2Vec) |
| `EMBEDDING_SEMANTIC_THRESHOLD` | `0.7` | Similarità semantica minima |
| `NODE2VEC_MODEL_DIR` | `./artifacts/node2vec` | Artefatti Node2Vec |
| `TEXT2CYPHER_MODEL` | `google/gemini-2.5-flash` | Generazione Cypher + traduzione |

**WebUI / auth / persistenza**

| Variabile | Default | Scopo |
|---|---|---|
| `WEBUI_AUTH_SECRET` | fallback dev (avvisa) | Segreto di firma HS256 (cookie + Bearer). **Deve** essere random in prod |
| `WEBUI_DATABASE_URL` | SQLite `data/webui/webui.db` | Store utenti/lezioni/messaggi |
| `WEBUI_TOKEN_LIFETIME_SECONDS` | `86400` | Durata della sessione |
| `WEBUI_COOKIE_SECURE` | `0` | Impostare `1` dietro HTTPS |
| `WEBUI_CORS_ALLOW_ORIGINS` | — | Origini ammesse (Mode B / cross-origin) |
| `LANGGRAPH_DATABASE_URL` | SQLite `data/agent_threads.db` | Store del checkpointer LangGraph |

**Flag di tuning degli agenti** (tutti con default sicuro via `os.getenv` — vedere `.env.example` per l'elenco annotato)

| Variabile | Default | Scopo |
|---|---|---|
| `AIX_MAX_REVISIONS` | `1` | Cicli di revisione Writer→Critic (0–4) |
| `AIX_WRITER_MAX_TOKENS` | `8000` codice; `2000` nel template prod | Tetto di output del Writer; abbassarlo in produzione limita costo e latenza |
| `AIX_WRITER_MAX_CONTINUATIONS` | `1` | Auto-continuazione su taglio per lunghezza |
| `AIX_THINKING_EFFORT` | `low` | Budget di reasoning token (`low`/`medium`/`high`) |
| `AIX_CRITIC_MODEL` | (→ `TEXT2CYPHER_MODEL`) | Modello Critic (veloce/economico) |
| `AIX_CRITIC_PARSE_ERROR_BEHAVIOR` | `approve` | Su JSON del critic non parsabile (`approve`/`revise`/`raise`) |
| `AIX_CORRECTIVE_RAG_ENABLED` | `false` | Abilita il ciclo di grading del retrieval |
| `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS` | `2` | Budget di retry quando CR è on (1–4) |
| `AIX_CONVERSATION_WINDOW_TURNS` | `4` | Turni verbatim prima del summary buffering |
| `AIX_LLM_PROBE_ENABLED` | `true` | Connectivity probe LLM all'avvio |

**Media dinamici / live** (vedere §7.8; tutti opzionali e con default sicuro — il livello è disattivo finché `AIX_MEDIA_LIVE_ENABLED` non è true)

| Variabile | Default | Scopo |
|---|---|---|
| `AIX_MEDIA_LIVE_ENABLED` | `false` | Master switch del livello media live; off = no-op |
| `AIX_MEDIA_MAX_VIDEOS` | `5` | Cap per bucket — video |
| `AIX_MEDIA_MAX_PAPERS` | `3` | Cap per bucket — paper |
| `AIX_MEDIA_MAX_WEB` | `6` | Cap per bucket — link web |
| `AIX_MEDIA_CACHE_BACKEND` | `diskcache` (`redis` se è impostato un URL Redis) | Backend di cache (`redis`/`diskcache`/`none`) |
| `AIX_MEDIA_CACHE_DIR` | `artifacts/media_cache` | Directory della cache diskcache (dev) |
| `AIX_MEDIA_CACHE_TTL` | `1209600` | TTL della cache in secondi (14 giorni) |
| `AIX_MEDIA_CACHE_NAMESPACE` | `aix:media:v1` | Namespace di chiave versionato (flush indipendente) |
| `AIX_MEDIA_REDIS_URL` / `REDIS_URL` | — | URL Redis condiviso (abilita il backend redis in prod) |
| `AIX_MEDIA_REFRESH_ENABLED` | `false` | Job schedulato di refresh in background (off di default) |
| `AIX_MEDIA_RERANK_ENABLED` | `false` | Re-ranking semantico degli item live (Phase 3) |
| `AIX_MEDIA_RERANK_W_SEMANTIC` | `0.7` | Peso semantico nel punteggio combinato |
| `AIX_MEDIA_RERANK_W_QUALITY` | `0.3` | Peso di qualità nel punteggio combinato |
| `YOUTUBE_API_KEY` | — | Abilita il fetch live di YouTube via Data API (senza chiave: fallback su URL di ricerca) |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Opzionale; alza i rate limit di Semantic Scholar |

**Osservabilità / ops**

| Variabile | Default | Scopo |
|---|---|---|
| `SENTRY_DSN` | — | DSN GlitchTip/Sentry (vuoto = disabilitato) |
| `ENVIRONMENT` | `production` | Label degli issue (`production`/`staging`/`development`) |
| `LOG_LEVEL` | `INFO` | Verbosità dei log |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | — | Tracing LLM |

**Deployment (Caddy / Postgres / TLS)**

| Variabile | Scopo |
|---|---|
| `AIX_DOMAIN` | Hostname pubblico servito da Caddy (es. `agente.aiforlearning.digital`) |
| `AIX_TLS_EMAIL` | Email di registrazione/scadenza Let's Encrypt |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Credenziali Postgres (scelte da noi; compose crea il DB al primo boot) |
| `GIT_SHA` | Build arg impresso in `CODE_VERSION` per tracciabilità |

### 8.3 Gestione dei segreti

- I segreti vivono solo in `.env` (dev) / `deploy/.env.prod` (prod), mai in git. Sulla VM, `chmod 600 deploy/.env.prod`.
- `WEBUI_AUTH_SECRET` è generato con `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Se non impostato, viene usato un fallback solo-dev e viene loggato un warning — accettabile in locale, inaccettabile in produzione.
- I `POSTGRES_*` sono scelti dall'operatore (non forniti da una terza parte); `docker-compose.prod.yml` deriva sia `WEBUI_DATABASE_URL` sia `LANGGRAPH_DATABASE_URL` da essi, così c'è un'unica fonte di verità.
- Le chiavi LLM e Neo4j sono fornite dai rispettivi provider (dashboard OpenRouter, Neo4j Aura/FEM).

### 8.4 Profili di configurazione (dev / staging / prod)

- **Dev** — SQLite ovunque, HTTP localhost, `WEBUI_COOKIE_SECURE=0`, osservabilità opzionale, `.env`.
- **Staging** — Postgres, HTTPS via Caddy, `ENVIRONMENT=staging`, osservabilità on, comportamento critic più severo (`AIX_CRITIC_PARSE_ERROR_BEHAVIOR=raise`) per investigazione.
- **Prod** — Postgres, HTTPS, `WEBUI_COOKIE_SECURE=1`, `WEBUI_AUTH_SECRET` random, `LOG_LEVEL=INFO`, osservabilità on, `deploy/.env.prod`.

---

## 9. Sviluppo locale

### 9.1 Prerequisiti

- **Python 3.11** raccomandato per il dev locale (il progetto supporta ≥3.10; il Docker di produzione fissa 3.12).
- Un **Neo4j** raggiungibile (locale o l'istanza condivisa Aura/FEM).
- Una chiave API **OpenRouter** (o OpenAI).
- Git, e su Windows una shell PowerShell (il repo è sviluppato cross-platform).

### 9.2 Installazione

```bash
git clone <repo-url> && cd <repo-root>
python -m venv venv && . venv/Scripts/activate      # PowerShell: venv\Scripts\Activate.ps1
pip install -e ".[dev]"                              # editable install + dev extras (pytest, ruff, mypy)
cp .env.example .env                                 # then fill credentials
```

`pip install -e .` registra il package `aix` così che gli import `aix.*` si risolvano da qualsiasi working directory. L'extra `[dev]` aggiunge la toolchain di test/lint usata da `make test` / `make lint` / CI.

### 9.3 Esecuzione in locale

Il comando canonico per il test end-to-end locale (serve ogni superficie su una porta):

```bash
python -m uvicorn aix.api.main:app --host 127.0.0.1 --port 8765 --log-level info
```

Questo singolo processo serve `/docs`, `/webui/`, `/api/v1/context`, `/api/v1/agent/run`, `/api/v1/agent/stream`, `/mcp/` e `/auth/jwt/login`. Per usare la WebUI in locale dopo il primo avvio, creare il primo utente browser da `/auth/register` e poi entrare da `/auth/login`; senza un utente registrato le pagine protette di `/webui/` reindirizzano al login. Aggiungere `--reload` per l'autoreload durante lo sviluppo. Altri entry point:

```bash
python apps/cli/run_agent.py                  # interactive agent CLI (or: make agent)
python apps/cli/run_agent.py --query "Crea una lezione sulla memoria" --domain neuro
streamlit run apps/streamlit/main.py          # legacy demo (retirement banner)
python -m aix.mcp.stdio_main                  # MCP over stdio (Claude Desktop / Cursor)
```

### 9.4 Setup del Knowledge Graph

I dump del KG pre-confezionati sono inclusi nel repo, quindi non serve data prep per il setup di default:

```bash
python scripts/ingest/data_ingestion_neo4j.py \
    --file data/kg/neuro/kg_neuro_neo4j.json \
    --password YOUR_NEO4J_PASSWORD --clear        # swap in kg_udl_neo4j.json for UDL
```

Anche gli artefatti Node2Vec sono inclusi pre-addestrati; ri-addestrare solo se si cambia il grafo: `python scripts/ml/train_node2vec.py {neuro|udl}`.

### 9.5 Flussi di sviluppo comuni

- **Aggiungere/aggiornare dipendenze** — modificare `requirements.txt` (fonte di verità umana), poi rigenerare il lockfile con hash per il target di produzione: `uv pip compile requirements.txt -o requirements.lock.txt` (risoluzione Python 3.12 / Linux). Committare entrambi.
- **Lint & format** — `ruff format .` poi `ruff check . --fix` (vedere §10.4). Eseguire via la venv: `python -m ruff …`.
- **Type-check** — `mypy` (permissivo di default; configurato in `pyproject.toml`).
- **Smoke-test MCP** — `python scripts/diagnostic/mcp_smoke.py` (in-process, senza uvicorn).
- **Sync con il remoto** — `git fetch`, ispezionare con `git log HEAD..origin/<branch>`, poi `git pull --ff-only`; mettere da parte le modifiche locali con `git stash push -u` se necessario.

### 9.6 Note di configurazione dell'IDE

- Il package `aix` si risolve solo dopo `pip install -e .`; puntare l'interprete alla venv del progetto.
- Ruff è l'unico formatter/linter (Black-compatible, line length 100, double quotes). Configurare il "format on save" dell'editor per usare Ruff ed evitare churn nei diff.
- Su Windows, quando Postgres è configurato, l'app applica una selector-event-loop policy all'import; è un no-op su Linux e per lo sviluppo SQLite.
- Per i template WebUI, seguire la regola a tre livelli: componenti WebAwesome per i controlli, classi `.aix-*` in `aix-brand.css` per brand/stato/componenti, Tailwind solo per layout. Non aggiungere CSS inline salvo casi eccezionali e locali.

---

## 10. Strategia di testing

### 10.1 Layout dei test & marker

I test vivono sotto `tests/` e sono configurati in `pyproject.toml`:

- `tests/unit/` — unit test puri, senza servizi esterni (`@pytest.mark.unit`).
- `tests/integration/` — colpiscono Neo4j / API LLM (`@pytest.mark.integration`).
- `tests/api/` — test di contratto dell'API agenti (round-trip request/response + tassonomia SSE).
- `tests/mcp_server/` — suite di regressione MCP (19 test).

Default di pytest: `-v --tb=short --strict-markers`, `asyncio_mode = "auto"` (i test async non richiedono decorator esplicito). Marker: `integration`, `slow`, `unit`.

### 10.2 Esecuzione dei test

```bash
pytest tests/ -v                       # all suites (or: make test)
pytest tests/integration/ -v -m integration   # only integration (needs Neo4j / keys)
pytest tests/mcp_server/ -v            # MCP regression suite
pytest tests/api/ -v                   # agent API contract
```

### 10.3 Cosa è coperto vs cosa validare manualmente

- **Ben coperto**: contratto API (schemi, codici di stato, tassonomia `kind` SSE), tool/resources/prompts MCP, logica di routing degli agenti (ciclo di revisione, routing corrective-RAG), helper di retrieval.
- **Da validare manualmente / end-to-end** (nessuna copertura automatica completa ancora): l'arricchimento della pipeline media, il flusso di upsell e le run WebUI multi-turno complete — esercitare via `apps/cli/run_agent.py` e la WebUI prima di una release. Vedere §17 per il backlog dei gap noti.
- **Da validare per il livello media dinamici** (§7.8), quando lo si abilita: deduplica Wikipedia (concetti sinonimi → singola voce), hit/miss della cache `MediaCache`, fallback quando un'API esterna fallisce o è in rate limit, comportamento con `AIX_MEDIA_LIVE_ENABLED` off vs on, re-ranking on/off e rendering del pannello media (badge `auto` vs `✓ Verificato`).

### 10.4 Linting & formattazione (Ruff)

Ruff è l'unico strumento sia per il **linting** (sostituisce flake8/isort/bugbear/pyupgrade) sia per la **formattazione** (Black-compatible). Config in `pyproject.toml`: line length 100, target `py310`, stile double-quote, set di regole `E/F/W/I/B/UP`. `E501` (line length) è delegato al formatter; `B008` (FastAPI `Depends()`) e `UP007` sono ignorati intenzionalmente. Ignore per-directory rilassano le regole per `tests/`, `apps/` e `scripts/`.

```bash
python -m ruff format .            # apply formatting
python -m ruff format --check .    # verify formatting (CI)
python -m ruff check . --fix       # lint + safe autofixes
python -m ruff check . --statistics
```

Le correzioni di formattazione e ordinamento import preservano il comportamento; rivedere solo gli autofix di famiglie di regole che possono cambiare la semantica (es. alcune riscritture `B`/`UP`) prima di committare.

### 10.5 Aspettative di CI

Ci si aspetta che la CI esegua `ruff format --check .`, `ruff check .` e la suite di test non-integration su ogni push/PR. I test di integration che richiedono Neo4j/chiavi LLM live sono eseguiti selettivamente (sono marcati e possono essere esclusi in ambienti senza credenziali). La baseline OpenAPI sotto `data/diagnostic/` funge da guardia di regressione contro modifiche non intenzionali al contratto.

---

## 11. Deployment

### 11.1 Immagine container

`Dockerfile` è una build multi-stage che targetta `api`, hardened per la produzione:

- Base `python:3.12-slim-bookworm` con il locale `it_IT.UTF-8` installato (correttezza dell'output italiano).
- Dipendenze installate dal **lockfile con hash** con `uv pip install --require-hashes -r requirements.lock.txt` (build riproducibili), poi `pip install --no-deps -e .`.
- Esegue come utente **non-root** (`aix`, uid/gid 10001); `artifacts/` e `data/` sono scrivibili per le cache.
- Espone `8765`; l'env `PORT` guida il bind. `CODE_VERSION` è impresso dal build arg `GIT_SHA`.
- `HEALTHCHECK` integrato fa curl di `/api/v1/health` (interval 30 s, start period 45 s).

Comando equivalente locale (fuori da Docker): `python -m uvicorn aix.api.main:app --host 127.0.0.1 --port 8765`.

### 11.2 Stack di produzione (`deploy/docker-compose.prod.yml`)

Tre servizi su un singolo host:

| Servizio | Immagine | Esposizione | Ruolo |
|---|---|---|---|
| `app` | build da `Dockerfile` (`api`) | solo interno | FastAPI — `/api/v1/*` + `/webui/*` |
| `postgres` | `postgres:16-alpine` | solo interno | Supporta entrambi gli URL DB (webui + checkpointer) |
| `caddy` | `caddy:2-alpine` | `80`, `443`, `443/udp` | Reverse proxy + auto-HTTPS |

Proprietà chiave:

- **Neo4j non è nel compose** — la produzione riusa l'istanza esterna Aura/FEM via `NEO4J_URI`.
- Entrambi gli URL DB sono composti dalle stesse credenziali `POSTGRES_*` nel blocco `environment` del compose — un'unica fonte di verità. La webui usa il driver `asyncpg` (SQLAlchemy); `AsyncPostgresSaver` di LangGraph usa direttamente lo schema libpq.
- `app` attende che `postgres` sia healthy (`depends_on … condition: service_healthy`); le tabelle si auto-creano al primo boot.
- Volumi: `pg_data` (DB), `app_artifacts` (embeddings/Node2Vec — sopravvive alle rebuild), `caddy_data`/`caddy_config` (certificati ACME + stato).
- Run: `docker compose -f docker-compose.prod.yml --env-file .env.prod up -d`.

### 11.3 Reverse proxy & TLS (Caddy)

`deploy/Caddyfile` è parametrizzato da `$AIX_DOMAIN` e `$AIX_TLS_EMAIL`. Caddy termina il TLS (Let's Encrypt, auto-renew), serve HTTP/3 (QUIC) su UDP/443 e fa reverse proxy di tutto il traffico verso `app:8765`. Gli endpoint SSE funzionano in modo trasparente (Caddy fa streaming delle risposte; l'heartbeat da 15 s dell'app tiene aperte le connessioni idle). `AIX_TLS_EMAIL` è l'indirizzo di registrazione/avviso scadenza ACME — richiesto per l'emissione non presidiata dei certificati.

### 11.4 Migrazione DB & checkpointer a PostgreSQL

La Wave 1 migra entrambi gli store stateful da SQLite (dev) a PostgreSQL (prod):

- **DB WebUI** (`WEBUI_DATABASE_URL`) — utenti, lezioni, messaggi delle lezioni.
- **Checkpointer LangGraph** (`LANGGRAPH_DATABASE_URL`) — `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`.

Entrambi puntano alla stessa istanza Postgres (concerni logici separati, set di tabelle distinti). Nessuno step di schema manuale è richiesto per il pilot: le tabelle si auto-creano al primo boot (`init_db()` per la webui; `setup()` del saver per il checkpointer). I backup sono gestiti da `deploy/scripts/backup_postgres.*`.

### 11.5 Modalità di deployment (Mode A standalone / Mode B nativo)

- **Mode A (pilot standalone)** — lo stack compose sopra; i docenti usano `/webui/*` su `https://agente.aiforlearning.digital`. È l'attuale target Wave 1/2.
- **Mode B (nativo AixLearning)** — il servizio agenti gira come servizio interno; il backend Django di AixLearning chiama `/api/v1/agent/*` sulla rete privata e possiede la propria UX/dati. Mode B non richiede l'esposizione di Caddy/WebUI. Vedere §14.

La strategia di hostname prevede la transizione dal legacy `graph.aiforlearning.digital` al nuovo `agente.aiforlearning.digital`; la relazione tra accesso API e accesso WebUI è gestita nel piano di deployment interno.

### 11.6 Pipeline CI/CD

L'ambiente FEM esegue già continuous deployment: un merge commit pushato sul repository GitHub innesca una build + redeploy dell'istanza GraphRAG sulla VM gestita (Debian; networking, Docker e CD pre-configurati per il template FEM). In pratica, rilasciare una nuova versione è un **merge sul branch di deployment**. La build usa il `requirements.lock.txt` committato per la riproducibilità; `GIT_SHA` è impresso in `CODE_VERSION` per tracciabilità attraverso log e tool di osservabilità.

### 11.7 Rollback & recovery

- **Rollback** — redeploy di un'immagine/commit precedente (CD), oppure `docker compose up -d` contro il tag precedente; `CODE_VERSION` identifica la build in esecuzione.
- **Recovery dei dati** — ripristinare Postgres da `deploy/scripts/restore_postgres.*`; certificati/stato Caddy da `backup_caddy`. Gli artefatti in `app_artifacts` sono rigenerabili (re-embed) se persi.
- **Isolamento dei fallimenti** — la sub-app MCP è costruita in modo difensivo così che un fallimento lì non possa bloccare `/api/v1`; lo stream agenti fa emergere i fallimenti della pipeline come `error`/`502` anziché crashare il processo. Il runbook in `deploy/README.md` copre primo deploy, backup, rollback e ispezione dei log.

---

## 12. Osservabilità

### 12.1 Tracing LLM (Langfuse)

Quando `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (+ opzionale `LANGFUSE_HOST`) sono impostate, le chiamate LLM degli agenti sono tracciate su Langfuse, dando visibilità per-agente e sull'intera pipeline: prompt, modello, uso dei token, latenza e gli output di planner/retriever/critic. Il `session_id` portato attraverso `AgentState` (e accettato dall'API) è la chiave di correlazione per ricucire una run multi-step. Il tracing è **opt-in** — con le chiavi non impostate, il sistema gira in modo identico senza tracing.

### 12.2 Monitoraggio errori (GlitchTip / Sentry)

Quando `SENTRY_DSN` è impostata, l'app inizializza l'SDK Sentry all'avvio (GlitchTip è Sentry-compatible). Eccezioni non gestite e richieste lente sono riportate, etichettate da `ENVIRONMENT` (`production`/`staging`/`development`). Il performance tracing è campionato al 20% (`traces_sample_rate=0.2`, impostato in `api/main.py`) — sufficiente a far emergere gli endpoint lenti senza sovraccaricare la dashboard. Lasciare `SENTRY_DSN` vuoto per disabilitare del tutto.

### 12.3 Health check

`GET /api/v1/health` ritorna un `HealthResponse` che riporta `status` (`healthy`/`degraded`), `neo4j_connected` e `version` (la versione del package; la build imprime anche `CODE_VERSION` da `GIT_SHA`). Verifica la connettività Neo4j e le config di dominio, rendendolo adatto sia come `HEALTHCHECK` del container (Dockerfile + compose) sia come probe di uptime esterno. Uno stato `degraded` (Neo4j irraggiungibile) ritorna comunque HTTP 200 con il flag impostato, così gli orchestrator possono distinguere "processo su ma dipendenza giù" da "processo giù".

### 12.4 Connectivity probe

All'avvio (`AIX_LLM_PROBE_ENABLED=true`, default), `src/aix/core/connectivity_probe.py` emette un `GET /models` one-shot verso il `base_url` LLM configurato ed emette **una** riga di log azionabile che distingue fallimenti di certificato TLS, errori DNS/connessione, read timeout, fallimenti auth 401/403 ed errori upstream 4xx/5xx. Questo risolve la comune ambiguità "Connection error." che altrimenti nasconde tre diverse modalità di fallimento dentro il retry layer dell'SDK OpenAI. Disabilitare in ambienti air-gapped/di test.

### 12.5 Logging strutturato

La verbosità di logging è impostata da `LOG_LEVEL` (default `INFO`). Le righe di log usano prefissi coerenti e greppabili per sottosistema (es. `[api.agent]`, `[webui.agent]`, `[LessonPlannerGraph]`, `[language]`) e includono `session_id`/`thread_id`, durata, approvazione e revision count al completamento della run. Usare `DEBUG` in locale per far emergere query Cypher, score di embedding e reasoning token; mantenere `INFO`/`WARNING` in produzione.

---

## 13. Sicurezza

### 13.1 Autenticazione

L'auth utente e API è gestita da **FastAPI-Users**. Due backend condividono `WEBUI_AUTH_SECRET` (HS256): un backend **cookie** (sessioni browser/WebUI) e un backend **Bearer JWT** (client API/integrazione). Gli endpoint agente accettano entrambi via la dependency `current_active_user`. La durata del token è `WEBUI_TOKEN_LIFETIME_SECONDS` (default 24 h). L'endpoint legacy `/api/v1/context` usa uno schema HTTP Basic separato service-to-service per l'integrazione AixLearning esistente.

### 13.2 Autorizzazione

Per il pilot, l'autorizzazione è a grana grossa: qualsiasi utente attivo e autenticato può chiamare gli endpoint agente. Non c'è ancora gating per-tenant o per-ruolo — l'autorizzazione multi-tenant (e il lavoro RS256/JWT multi-issuer in §17.3) è rimandata. Gli utenti WebUI vedono e operano solo sulle proprie lezioni (la ownership è applicata a livello dati).

### 13.3 Politica CORS

CORS è configurato via `WEBUI_CORS_ALLOW_ORIGINS` (`api/main.py`). Il default è `*` (comodo per dev locale e per il pilot same-origin); in produzione con client cross-origin (chiamanti browser Mode B) andrebbe impostato a una allow-list separata da virgole. `allow_credentials=True` è impostato così che l'auth a cookie funzioni cross-origin quando è fornita una lista di origini esplicita.

### 13.4 Gestione dei segreti

Vedere §8.3. In sintesi: i segreti sono solo da ambiente (`.env` / `deploy/.env.prod`, git-ignored, `chmod 600` sulla VM); `WEBUI_AUTH_SECRET` deve essere un valore random forte in qualsiasi ambiente non locale; gli URL di database sono derivati da un singolo set di credenziali `POSTGRES_*`; le chiavi LLM/Neo4j vengono dai loro provider.

### 13.5 Rate limiting

Non c'è un rate limiter a livello applicativo nel pilot. Il throttle naturale è dato dai rate limit del provider LLM (OpenRouter), e Caddy può imporre limiti a livello di connessione al bordo se richiesto. Il rate limiting per-utente a livello applicativo è un candidato per una wave successiva se l'abuso diventa una preoccupazione. (Nota: il livello media dinamici di §7.8 ha un proprio rate limiting solo verso le API esterne come Semantic Scholar, non sulle chiamate LLM degli agenti.)

### 13.6 Isolamento di rete

In produzione solo Caddy è pubblicato (porte host 80/443). I servizi `app` e `postgres` **non** hanno mapping `ports:` — sono raggiungibili solo sulla rete Docker interna. Postgres non è quindi mai esposto all'host o a internet; l'app è raggiungibile solo attraverso Caddy (TLS-terminated). Neo4j è un'istanza esterna gestita raggiunta via `bolt+s://` (TLS). Il mount della sub-app MCP è protetto così che un fallimento lì non possa interessare `/api/v1`.

### 13.7 Trasparenza IA & allineamento EU AI Act

Il sistema è progettato per rendere esplicito il coinvolgimento dell'IA, in linea con l'analisi normativa del progetto:

- **Trasparenza in-product (implementata):** la WebUI mostra spiegabilità per fase (intento/scope del planner, coverage tier del retriever, score del critic) e un banner esplicito quando una lezione è composta con copertura KG ridotta o da fonti esterne ("La lezione è generata con queste limitazioni…"). Gli output sono sempre presentati come bozze generate da IA da rivedere da parte di un educatore umano.
- **Human-in-the-loop:** il docente è il decisore; il sistema produce bozze, non contenuti autorevoli. Il ciclo Critic e i segnali di copertura supportano, ma non sostituiscono, il giudizio umano.
- **Marcatura IA machine-readable (pianificata):** un marcatore a livello di risposta (es. un header `X-AI-Generated` e/o un commento Markdown embedded sui piani di lezione esportati) è tracciato come enhancement di compliance così che i consumatori a valle e gli artefatti esportati possano essere identificati programmaticamente come IA-assistiti. Non è ancora imposto nel codice e dovrebbe essere aggiunto prima di qualsiasi rollout esterno/pubblico.

Fare riferimento al documento normativo per la mappatura completa agli obblighi EU AI Act e UNI/PdR; questa sezione cattura solo i controlli tecnici.

---

## 14. Pattern di integrazione

### 14.1 Mode A — WebUI standalone

Il servizio agenti esegue lo stack completo (§11.2) ed espone la WebUI docente su `https://<AIX_DOMAIN>/webui/`. I docenti si autenticano (cookie session), creano lezioni con un profilo educativo, opzionalmente caricano file di contesto e guardano l'agente fare streaming delle sue fasi live. È la modalità **pilot interno FEM** e non richiede lavoro dal team DEV AixLearning oltre all'infrastruttura (VM + DNS + CD).

Il flusso principale della WebUI è server-driven (Jinja2 + htmx + SSE) e oggi include:

1. **Creazione guidata in due fasi.** Sul primo invio di una nuova lezione il backend può eseguire solo il Planner, renderizzare la card Planner e mostrare una card di scelta dell'intento pedagogico. Quando il docente sceglie un chip, il form htmx reinvia `intent_confirmed=1` insieme a `pedagogical_intent_code`/dettaglio; solo allora parte la pipeline completa Planner → Retriever → Writer → Critic. `intent_confirmed` è un dettaglio del contratto WebUI, non un campo dell'API pubblica `/api/v1/agent/run`.
2. **Profili educativi salvati.** Il docente può salvare, ricaricare e cancellare preset di profilo da `/webui/profiles`; i chip precompilano il form di creazione senza legare il profilo a una singola lezione.
3. **Storico e biblioteca lezioni.** Le lezioni e i turni conversazionali sono persistiti in SQL (`lesson`, `lesson_message`) e alimentano la libreria, il dettaglio lezione e i follow-up multi-turno.
4. **Raffinamento SAM.** Nel footer della lezione il pannello "Raffina" offre cinque opzioni (`Semplifica`, `Approfondisci`, `Più attività`, `Adatta alla classe`, `Personalizza`) che diventano `refinement_instruction` per una nuova run sulla stessa lezione.
5. **Segnali pedagogici in UI.** La card finale mostra pill di intento pedagogico, badge UDL quando sono disponibili conteggi sui tre principi e il pannello "Cosa esplorare dopo" con concetti KG adiacenti.
6. **Export.** Ogni lezione finalizzata espone download MD/TXT e una pagina print-friendly per generare PDF tramite il dialog di stampa del browser.

### 14.2 Mode B — integrazione nativa AixLearning

La piattaforma Django AixLearning integra il servizio agenti nello stesso modo in cui ha già integrato l'endpoint GraphRAG legacy `/api/v1/context`:

1. Il docente usa la UI esistente di AixLearning per richiedere una lezione.
2. Il **worker Dramatiq** di AixLearning rileva un `plan_type` UDL/NEURO e instrada la richiesta a un nuovo **`AgenticGraphRagService`** — un fratello dell'esistente `GraphRagService` che già chiama `/api/v1/context`.
3. `AgenticGraphRagService` chiama `POST /api/v1/agent/run` (o `/stream`) sulla rete Docker interna con un service Bearer token, passando la query del docente + il profilo educativo.
4. AixLearning persiste e renderizza il `lesson_plan_md` ritornato nella propria UI.

In questa modalità il servizio agenti possiede solo il proprio stato (KG, checkpointer, store delle lezioni se usato); AixLearning possiede la UX docente e i propri dati. L'esposizione di Caddy/WebUI è opzionale.

### 14.3 Regole di coesistenza tra le modalità

- Le due modalità **non sono mutuamente esclusive** — la stessa istanza in esecuzione può servire `/webui/*` (Mode A) e `/api/v1/agent/*` ad AixLearning (Mode B) simultaneamente.
- Gli endpoint agente sono **additivi e retrocompatibili**: il contratto API è bloccato da un test di regressione automatico contro una baseline OpenAPI (`data/diagnostic/`), e i nuovi campi sono sempre aggiunti, mai rimossi o riadattati.
- L'autenticazione differisce per chiamante: cookie browser (Mode A) vs service Bearer token (Mode B). Entrambi risolvono alla stessa dependency `current_active_user`.

### 14.4 Wrapper di riferimento (pattern `AgenticGraphRagService`)

La forma di integrazione raccomandata lato AixLearning è una classe di servizio sottile che rispecchia l'esistente `GraphRagService`:

```python
class AgenticGraphRagService:
    def generate_lesson(self, query: str, domain: str, profile: dict) -> str:
        resp = http.post(
            f"{AGENT_BASE_URL}/api/v1/agent/run",
            headers={"Authorization": f"Bearer {SERVICE_JWT}"},
            json={"query": query, "domain": domain, "educational_profile": profile},
            timeout=240,  # the sync run can take 60–120s; allow headroom
        )
        resp.raise_for_status()
        return resp.json()["lesson_plan_md"]
```

Per una UX incrementale, sostituire `/run` con `/stream` e consumare i frame SSE, facendo switch sul `kind`. Il contratto completo, gli esempi di payload e la tassonomia SSE di cui il team DEV ha bisogno sono in §5 (Riferimento API) e §5.5 (tassonomia degli eventi SSE).

---

## 15. Performance & SLO

### 15.1 Budget di latenza per fase

Una run tipica coperta dal KG (config di default, 0–1 revisioni) si distribuisce all'incirca così:

| Fase | Tempo tipico | Note |
|---|---|---|
| Planner | ~2–5 s | Intento + scope + estrazione query |
| Retriever | ~3–8 s | Cypher + ricerca ibrida (+ fonti esterne se out-of-scope) |
| Writer | ~25–40 s | Costo dominante; limitato da `AIX_WRITER_MAX_TOKENS` (+ continuazioni) |
| Critic | ~2–5 s | Modello veloce (`AIX_CRITIC_MODEL`), prefill limitato |

Il Writer è il collo di bottiglia; thinking-effort e cap sui token (`AIX_THINKING_EFFORT`, `AIX_WRITER_MAX_TOKENS`) sono le leve di latenza primarie.

### 15.2 Target del primo evento in streaming

Per `/stream`, il primo evento `planner` dovrebbe raggiungere il client entro pochi secondi (target **< 5 s**), dando feedback immediato alla UI mentre gira il lento Writer. L'heartbeat SSE da 15 s previene i timeout dei proxy durante la chiamata del Writer.

### 15.3 Target SLO end-to-end (pilot)

- Pipeline completa (sync `/run`): **< 180 s** per topic coperti dal KG; **< 240 s** per topic out-of-scope che innescano retrieval esterno. I client di integrazione dovrebbero impostare timeout con margine (≈240 s).
- Target di disponibilità per il pilot: **~99.5%** (stack single-host; non HA).

### 15.4 Assunzioni di costo & capacità

- Il costo per interazione è dominato dalla chiamata LLM del Writer (e dai thinking token). Critic e Text2Cypher usano modelli veloci più economici per mantenere basso il costo per run.
- Il pilot è dimensionato per un singolo host (2 vCPU / 4 GB RAM / ~50 GB disco, Debian) con agenti e grafo compilato come singleton per processo. In Postgres, il registro `agent_run` coordina una run in-flight per lezione anche con più worker uvicorn; i cap di concorrenza LLM restano per-processo e vanno dimensionati considerando `workers × cap`. Le cache di embedding persistono in un volume così che restart/rebuild evitino il re-embed.

---

## 16. Runbook operativo

### 16.1 Incidenti comuni

| Sintomo | Causa probabile | Prima azione |
|---|---|---|
| `/api/v1/health` riporta `degraded` | Neo4j irraggiungibile / `NEO4J_*` errati | Controllare `NEO4J_URI` + rete verso Aura/FEM; ispezionare i log dell'app |
| Run falliscono con `502` / eventi `error` | Provider LLM giù / chiave errata / rate limit | Controllare la riga di log del connectivity-probe; verificare `OPENROUTER_API_KEY` |
| "Connection error." all'avvio | TLS/DNS/auth verso l'endpoint LLM | Leggere la singola riga di log del probe (nomina la modalità di fallimento esatta) |
| Primo evento lento su `/stream` | Cold start / latenza Neo4j | Confermare processo caldo; controllare i timing Planner/Retriever nei log |
| Login WebUI fallisce dopo un deploy | `WEBUI_AUTH_SECRET` cambiato/mancante | Ripristinare il segreto; ruotarlo invalida le sessioni esistenti |
| Lezione "congelata" nella WebUI | Eccezione di setup inghiottita in `error` | Controllare i log `[webui.agent] setup FAILED`; la lezione è marcata `error` |
| Media live assenti pur con `AIX_MEDIA_LIVE_ENABLED=true` | Worker off-path non in esecuzione, oppure API esterna giù / in rate limit / `YOUTUBE_API_KEY` mancante | Verificare i log `[live-media]`; il livello degrada in modo silenzioso a `{}` per design (la lezione resta valida) — vedere §7.8 |
| Voci media duplicate / mismatch di conteggio card↔sidebar | Concetti sinonimi o cache vecchia | La deduplica Wikipedia è per URL canonico (§7.8); svuotare la cache `artifacts/media_cache/` o ruotare `AIX_MEDIA_CACHE_NAMESPACE` |

### 16.2 Guida al debugging

- Impostare `LOG_LEVEL=DEBUG` per vedere Cypher, score di embedding e reasoning token.
- Tracciare una run specifica tramite il suo `session_id`/`thread_id` attraverso log e Langfuse.
- Riprodurre una run della pipeline senza WebUI/DB via `apps/cli/run_agent.py --query "…" --domain neuro`.
- Smoke-test di MCP in-process con `python scripts/diagnostic/mcp_smoke.py`.
- Verificare che il contratto API non sia derivato rispetto alla baseline OpenAPI in `data/diagnostic/`.

### 16.3 Restart / redeploy

- **Redeploy (CD):** pushare un merge commit sul branch di deployment — `staging` produce l'immagine con tag `staging`, `production` produce il tag `latest`; la pipeline/ambiente FEM a valle decide pull e riavvio dello stack.
- **Restart manuale:** `docker compose -f deploy/docker-compose.prod.yml --env-file deploy/.env.prod up -d` (ricrea i servizi cambiati; i volumi persistono).
- **Singolo servizio:** `docker compose … restart app` (o `caddy`). Confermare la salute via `/api/v1/health` e lo stato health del compose.

### 16.4 Manutenzione del database

- **Backup:** `deploy/scripts/backup_postgres.*` (DB) e `backup_caddy` (certificati/stato) su schedule; conservare off-host.
- **Restore:** `deploy/scripts/restore_postgres.*` in un volume `postgres` fresco; riavviare `app`.
- **Schema:** le tabelle si auto-creano al primo boot (webui `init_db()`, checkpointer `setup()`). Il trascritto dei messaggi della lezione è la fonte di verità durevole e sopravvive a un wipe del checkpointer; gli artefatti nel volume `app_artifacts` sono rigenerabili.

---

## 17. Roadmap & limitazioni note

### 17.1 Elementi rimandati

Elementi intenzionalmente fuori scope per il pilot attuale (tracciati nel backlog di progetto):

- Rifinitura della copy italiana, accessibilità, breakpoint mobile e build Tailwind CLI dedicata.
- Copertura end-to-end automatica completa della pipeline media e del flusso di upsell (attualmente validati manualmente — §10.3).
- Parità di contenuti UDL (arricchimento del media-mapping) e criteri Critic UDL arricchiti; integrazione del Critic con le domain-config rispecchiando il pattern del Writer.
- Rate limiting condiviso a livello applicativo e autorizzazione per-tenant.
- Decisione su `lesson_template.txt` (collegarlo al Writer o rimuoverlo).

### 17.2 Concorrenza & scaling

Il grafo compilato e gli agenti sono singleton a livello di modulo, quindi ogni worker uvicorn possiede la propria copia in memoria. La WebUI impone una run in-flight per lezione tramite un `RunRegistry`: in SQLite/dev usa il backend in-memory, mentre in Postgres/prod usa la tabella condivisa `agent_run` con heartbeat e recupero delle righe stale. Questo evita doppie generazioni della stessa lezione anche quando due richieste arrivano su worker diversi.

La fase multi-worker ha anche una protezione sulla DDL di startup: `init_db()` serializza `create_all` su Postgres con un advisory lock transazionale, così più worker possono avviarsi insieme su un DB fresco senza gare `DuplicateTable`. Restano da trattare come limiti di scaling: rate limiting realmente condiviso (es. Redis), coda globale delle generazioni, cap globale delle chiamate LLM concorrenti, dashboard operativa dei run attivi e validazione staging/Linux prima di aumentare i worker oltre il profilo pilota.

### 17.3 Migrazione RS256 / JWT multi-issuer

La V1 usa HS256 con un `WEBUI_AUTH_SECRET` condiviso. Una migrazione a **RS256** con supporto multi-issuer è pianificata per scenari multi-tenant / multi-issuer (es. AixLearning che conia i propri token), ma **non è richiesta per la V1**.

### 17.4 Evoluzione del frontend

La WebUI standalone (Jinja2 + htmx + WebAwesome + Tailwind + Alpine.js + SSE) è intenzionalmente leggera e server-driven, il che si adatta alla UX agentica e in streaming. Un'analisi più approfondita delle opzioni di frontend per un'architettura agentica (incluso il confronto con lo stack nativo Django/Mercure di AixLearning) è mantenuta separatamente.

### 17.5 Altre limitazioni note

- Pilot single-host, non-HA (disponibilità ~99.5%, §15.3).
- Autorizzazione grossolana (§13.2) e nessun rate limiting a livello applicativo (§13.5).
- La generazione multilingue è tarata per l'italiano (primario) con supporto inglese; altre lingue sono rilevate ma non first-class.

---

## 18. Appendici

### Appendice A — Riferimento variabili d'ambiente (tabella completa)

L'elenco completo e annotato vive in `.env.example` (superficie di sviluppo) e `deploy/.env.prod.example` (sottoinsieme di produzione). Le tabelle di riepilogo categorizzate in §8.2 sono il riferimento canonico in-documento; trattare i file `*.example` come la fonte autorevole per default e indicazioni inline, dato che sono versionati insieme al codice.

### Appendice B — Glossario

Estende il glossario di §1.5:

- **Node2Vec** — algoritmo di graph-embedding che produce vettori strutturali per i nodi del KG.
- **Text2Cypher** — il layer di conversione NL→Cypher (con traduzione IT/EN).
- **Corrective RAG** — ciclo opzionale di grading del retrieval che ritenta il retrieval quando l'ancoraggio è debole.
- **Profilo educativo** — contesto strutturato di classe/aula/tempo/materia allegato a una richiesta.
- **Intento pedagogico** — obiettivo didattico scelto o confermato dal docente nella WebUI e passato al Writer come vincolo di prompt.
- **SAM / raffinamento guidato** — pannello WebUI che trasforma una richiesta di revisione docente in una `refinement_instruction` per rigenerare la lezione.
- **Run Registry / `agent_run`** — registro delle run in-flight; in produzione usa il database per coordinare più worker o repliche.
- **Coverage tier** — segnale UI (`healthy`/`limited`/`out_of_scope`) derivato dal numero di nodi del KG.
- **Checkpointer** — componente LangGraph che persiste lo stato dell'agente per-thread (SQLite dev / Postgres prod).
- **Thread / `thread_id`** — la chiave di conversazione che abilita la memoria multi-turno.
- **MCP** — Model Context Protocol; espone tool/resources/prompts ai client MCP.
- **Mode A / Mode B** — WebUI standalone vs integrazione nativa AixLearning.
- **Live media layer** — livello media dinamici off-critical-path (§7.8) che arricchisce le risorse via fonti esterne in tempo reale, flag-gated e con cache.
- **Langfuse / GlitchTip** — backend di tracing e monitoraggio errori.

### Appendice C — Documenti correlati

Documenti consegnati insieme a questo riferimento tecnico (parte del repository):

| Documento | Scopo |
|---|---|
| `docs/release/Functional_Documentation.md` | Companion di prodotto/pedagogico a questo riferimento tecnico |
| `deploy/README.md` | Runbook di primo deploy, backup, rollback, ispezione log |
| `README.md` | Quick start del repository e panoramica delle feature |

> La storia delle modifiche del documento è in testa (sezione **Storia delle modifiche**).

---
