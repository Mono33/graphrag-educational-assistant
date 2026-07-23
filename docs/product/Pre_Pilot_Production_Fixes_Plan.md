# Piano pre-pilot: media, Cypher e routing API

**Data:** 15 luglio 2026  
**Obiettivo:** riallineare staging/produzione al comportamento locale prima
dell'onboarding degli utenti pilot.

## 1. Risultato atteso

Al termine del piano:

- la WebUI carica media curati, Node2Vec ed embedding come in locale;
- un Cypher malformato non viene mai eseguito e degrada in modo controllato;
- `/docs` carica correttamente `/openapi.json`;
- staging viene validato prima del passaggio a produzione.

## 2. Piano B — media a zero in produzione

### Causa verificata

`data/` e `artifacts/` sono esclusi dal contesto Docker. Il pool curato
`data/media/kg_{domain}_media_pool.json`, Node2Vec e gli embedding non entrano
quindi nell'immagine. Il Compose cloud-infra effettivamente usato non monta
queste directory da un volume esterno. I sette link visibili in staging sono
risultati DuckDuckGo, non media curati.

### Implementazione

1. Rendere immutabili nell'immagine:
   - `data/media/kg_neuro_media_pool.json`;
   - `data/media/kg_udl_media_pool.json`;
   - `artifacts/node2vec/*_model.pkl`;
   - `artifacts/node2vec/*_embeddings.npz`;
   - `artifacts/embeddings_cache/*_openai_embeddings.json`.
2. Non montare volumi sopra gli artifact statici inclusi nell'immagine. Se si
   desidera persistere la cache runtime, montare soltanto
   `artifacts/media_cache`.
3. Aggiungere al build Docker controlli `test -s` che interrompano il build
   quando un asset obbligatorio manca.
4. Nel runtime reale impostare:
   - `AIX_MEDIA_LIVE_ENABLED=true`;
   - `AIX_MEDIA_RERANK_ENABLED=true`;
   - `YOUTUBE_API_KEY` tramite secret/`.env` sul server, non nel repository.
5. Verificare nei log:
   - `MediaLookup initialized`;
   - `Node2Vec model loaded successfully`;
   - `SemanticEmbedder Loaded ... cached embeddings`.

### File da modificare

| Repository | File | Modifica |
|---|---|---|
| applicazione | `.dockerignore` | Eccezioni ristrette per pool media e artifact statici |
| applicazione | `Dockerfile` | Controllo build-time della presenza degli asset |
| applicazione | `deploy/docker-compose.prod.yml` | Allineare lo stack di riferimento: eventuale volume solo per `artifacts/media_cache` |
| applicazione | `deploy/.env.prod.example` | Placeholder `YOUTUBE_API_KEY` e configurazione cache |
| applicazione | `tests/integration/test_phase1_media.py` | Verifica caricamento pool neuro e UDL |
| cloud-infra | `we-graph-01/neo4j/compose.yaml` | Passaggio variabili runtime; volume cache-only solo se necessario |
| cloud-infra | `we-graph-01/neo4j/.env` | Flag live/rerank; segreti solo sul server/secret store |

## 3. Piano C — Cypher malformato

### Causa verificata

`EXPLAIN` riconosce correttamente `MATCH (c` come non valido e imposta
`metadata.is_valid=false`. Successivamente `_graph_traversal()` ignora però
questo metadato ed esegue nuovamente il Cypher malformato.

### Implementazione

1. Aggiungere una validazione strutturale economica prima di `EXPLAIN`
   (parentesi bilanciate, query completa e clausola `RETURN`).
2. Se `EXPLAIN` fallisce, eseguire al massimo una rigenerazione, fornendo al
   modello l'errore di validazione.
3. Se anche la seconda query è invalida:
   - non inviarla a Neo4j;
   - usare semantic retrieval e/o una query fallback parametrizzata e limitata;
   - registrare un warning strutturato, non un errore applicativo fatale.
4. In `_graph_traversal()` aggiungere il guard definitivo:
   `metadata.is_valid is False` significa nessuna `session.run()` della query.
5. Collegare realmente i flag `TEXT2CYPHER_ENABLE_VALIDATION` e
   `TEXT2CYPHER_ENABLE_EXECUTION`, oggi letti dalla configurazione ma non usati
   dal converter.

### File da modificare

| File | Modifica |
|---|---|
| `src/aix/retrieval/text2cypher.py` | Pre-validazione, retry singolo e fallback |
| `src/aix/retrieval/graph_retriever.py` | Non rieseguire query con `is_valid=false` |
| `src/aix/core/config.py` | Rendere effettivi i flag documentati |
| `tests/unit/test_invalid_cypher_handling.py` (nuovo) | Regressione su `MATCH (c`, retry e mancata esecuzione |

### Criteri di accettazione

- `MATCH (c` non raggiunge mai `session.run()`;
- nessun 500 per Cypher generato male;
- fallback o risultato vuoto controllato;
- evento osservabile come warning con dominio e identificativo richiesta.

## 4. Piano D — `/docs` non funzionante

### Causa verificata

I servizi `api` e `api-staging` leggono lo stesso `.env`, che contiene
`--root-path /fast-api`, ma sono pubblicati con due modelli di routing diversi:

- produzione `api`: `graph.aiforlearning.digital/fast-api`, con StripPrefix;
- staging `api-staging`: `agente.aiforlearning.digital/`, alla root.

Il root path è coerente con il primo servizio, ma non con staging. In staging
Swagger richiede quindi `/fast-api/openapi.json` (404), mentre
`/openapi.json` risponde 200.

### Implementazione

1. Conservare `--root-path /fast-api` per il servizio produzione prefissato.
2. Sovrascrivere nel solo servizio `api-staging`:
   `API_CMD_ARGS=""` finché non è confermato il backend Postgres/multi-worker;
   successivamente usare `API_CMD_ARGS="--workers 2"`.
3. Preferire `.env.production` e `.env.staging` separati, così un parametro di
   routing non viene ereditato accidentalmente da entrambi i servizi.
4. Non modificare `FastAPI(...)`: il codice applicativo è già configurato per
   `/docs`, `/openapi.json`, `/api/v1/*` e `/webui/*` alla root.
5. Ricreare esplicitamente il container dopo la modifica del Compose.
   Watchtower aggiorna l'immagine, ma non applica automaticamente modifiche a
   command, environment o mount del file Compose.
6. Verificare:
   - `/docs` senza errore;
   - `/openapi.json` 200;
   - assenza di `servers: [{"url": "/fast-api"}]`;
   - nessun URL `/fast-api/fast-api/...` nei nuovi eventi.


Staging api-staging
Staging is exposed at the domain root, but inherits:

API_CMD_ARGS=--root-path /fast-api
That is incorrect. It causes Swagger to request /fast-api/openapi.json.

Override it specifically in api-staging:

environment:
  ENVIRONMENT: staging
  API_CMD_ARGS: ""
After Postgres/multi-worker readiness is completed:

API_CMD_ARGS: "--workers 2"
Do not remove the root path globally from the shared .env unless production is also moved away from /fast-api.

A better design is separate environment files:

api:
  env_file: .env.production
api-staging:
  env_file: .env.staging
  

### File da modificare

| Repository | File | Modifica |
|---|---|---|
| cloud-infra | `we-graph-01/neo4j/compose.yaml` | Override `API_CMD_ARGS` solo su `api-staging`; produzione invariata |
| cloud-infra | `we-graph-01/neo4j/.env` | Mantenere il root path della produzione o separare gli env |
| applicazione | `deploy/.env.prod.example` | Documentare routing root vs prefissato e argomenti Uvicorn |
| applicazione | `deploy/README.md` | Documentare root routing e ricreazione dopo cambi Compose |
| applicazione | `tests/api/test_openapi_routing.py` (nuovo) | Regressione `/docs` → `/openapi.json` |

Non è previsto un cambiamento a
`we-graph-01/watchtower/docker-compose.yaml`: Watchtower usa `--label-enable`,
entrambi i servizi API hanno la label richiesta e il polling avviene ogni
300 secondi. `api` segue `:latest`, mentre `api-staging` segue `:staging`.

## 5. Ordine di esecuzione

1. Implementare e testare il guard Cypher.
2. Correggere packaging media/artifact e costruire l'immagine.
3. Aggiornare il Compose cloud-infra (flag media e override root path staging).
4. Ricreare il container staging una volta; non affidarsi al solo Watchtower
   per le modifiche infrastrutturali.
5. Eseguire smoke test ADHD:
   - nodi e relazioni maggiori di zero;
   - media curati maggiori di zero;
   - live media presenti se il flag è attivo;
   - lezione completata;
   - `/docs` operativo.
6. Osservare log e GlitchTip per almeno due esecuzioni concorrenti.
7. Promuovere la stessa immagine validata da staging a produzione.

## 6. Quale endpoint usa la WebUI?

La WebUI non chiama né `/api/v1/context` né `/api/v1/agent/run` via HTTP.

Il browser usa:

```text
POST /webui/lesson/{id}/run
  -> GET /webui/lesson/{id}/stream  (SSE HTML)
  -> run_agent_stream()
  -> AgentOrchestrator / LangGraph
  -> RetrieverAgent
  -> GraphRAGTool
  -> Text2Cypher + Neo4j
```

Le superfici pubbliche hanno scopi diversi:

- `POST /api/v1/context`: endpoint legacy GraphRAG-only; restituisce contesto
  strutturato, non esegue l'intero ciclo Planner → Retriever → Writer → Critic.
- `POST /api/v1/agent/run`: API sincrona per client esterni; usa lo stesso
  motore LangGraph della WebUI, ma senza il flusso HTML e la persistenza WebUI.
- `POST /api/v1/agent/stream`: equivalente esterno SSE/JSON del flusso agentico.
- `/webui/lesson/{id}/run` + `/stream`: integrazione browser interna, con DB,
  autenticazione cookie, rendering HTML e persistenza della lezione.

Di conseguenza, l'evento GlitchTip su `/api/v1/context` proveniva da un altro
client/integratore e non dal percorso della WebUI.

## 7. Configurazione cloud-infra verificata

- `api` usa l'immagine `:latest` ed è pubblicato sotto `/fast-api`;
- `api-staging` usa `:staging` ed è pubblicato alla root del dominio `agente`;
- entrambi sono osservati da Watchtower tramite label;
- il Compose cloud non monta `data/media` o `artifacts`;
- entrambi leggono oggi lo stesso `.env`, con override del solo
  `ENVIRONMENT=staging` sul servizio staging.

Non inserire in questo documento o nel repository i valori segreti del file
`.env`.

## Appendice A — verifiche infrastrutturali aggiuntive

Questi punti sono separati dai fix B/C/D e non sono considerati automaticamente
bloccanti: il Team DEV deve prima confermare come sono integrati sulla VM.

### A.1 Persistenza WebUI e Postgres

Nel Compose condiviso non compaiono un servizio Postgres, `WEBUI_DATABASE_URL`
o `LANGGRAPH_DATABASE_URL`. È possibile che FEM fornisca Postgres con un
servizio esterno o con configurazione iniettata direttamente sulla VM.

Prima del pilot verificare nel container effettivo:

- quale database usa la WebUI;
- dove sono persistiti utenti, lezioni e messaggi;
- quale checkpointer usa LangGraph;
- se il run registry è `memory` oppure database;
- se i dati sopravvivono a un aggiornamento Watchtower.

Se Postgres è già integrato, documentare la sua origine senza riportare
credenziali. Se non lo è, mantenere un solo worker e pianificare la persistenza
prima di abilitare `--workers 2`.

### A.2 Porte pubblicate

Il Compose pubblica `8000:80` e `9000:80`, mentre l'immagine ascolta sulla porta
`8765`. Traefik funziona perché inoltra direttamente a `8765`, ma le porte host
sono fuorvianti. Rimuoverle, se non necessarie, oppure usare `8000:8765` e
`9000:8765`.

### A.3 Sicurezza operativa

Il file `.env` cloud-infra non deve contenere credenziali versionate. Ruotare le
chiavi già esposte, rimuovere i valori dalla cronologia Git e usare secret store
o file server-side non tracciati. Verificare inoltre il casing della variabile
JVM Neo4j (`NEO4J_...`), perché i nomi delle variabili ambiente sono
case-sensitive.




# Another point : treat the attachment case (when attaching a doc in the chat)
short tem: live pdf/doc in the context + get deleted after the session
long term: think about RAG, those docs can make a knowledge base