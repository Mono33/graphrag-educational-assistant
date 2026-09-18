# Variabili d'ambiente

Ogni variabile che il codice legge, il suo default, e cosa succede se non la
imposti. `.env.example` tiene solo quelle che vanno decise; questo è l'elenco
completo.

Generato leggendo il sorgente il 2026-09-16 (98 variabili). Se aggiungi una
variabile al codice, aggiungila qui: `.env.example` conteneva 65 esempi
commentati e ne mancavano 24 che il codice leggeva davvero — una documentazione
che sembra autorevole e non lo è costa più di una assente.

**Convenzione dei booleani:** `1`, `true`, `yes`, `on` accendono; tutto il resto
spegne. Una variabile assente o vuota vale il suo default.

---

## 1. Obbligatorie

| Variabile | Default | Se manca |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | Nessuna chiamata al modello. Ripiego: `OPENAI_API_KEY` |
| `LLM_MODEL` | `anthropic/claude-sonnet-5` | Usa il default del codice |
| `NEO4J_URI` · `NEO4J_USER` · `NEO4J_PASSWORD` | — | Nessun recupero dal Knowledge Graph: lezioni senza fondamento pedagogico |
| `WEBUI_AUTH_SECRET` | **valore di sviluppo nel codice** | L'app parte e logga un avviso. In produzione significa token di sessione firmati con un segreto pubblico |

## 2. Modello e chiamate LLM

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint del provider |
| `OPENAI_API_KEY` | — | Ripiego storico se manca la chiave OpenRouter |
| `AIX_LLM_TIMEOUT_S` | `120` | Timeout di una chiamata |
| `AIX_LLM_MAX_RETRIES` | `1` | Tentativi su errore transitorio |
| `AIX_LLM_PROBE_ENABLED` | `true` | Sonda di connettività all'avvio |
| `AIX_THINKING_EFFORT` | `low` | Sforzo di ragionamento sui modelli che lo espongono |
| `AIX_CRITIC_MODEL` | come `LLM_MODEL` | Modello diverso per il Critic |
| `AIX_MAX_REVISIONS` | `1` | Giri di revisione Critic → Writer |

### Writer

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `AIX_WRITER_MAX_TOKENS` | `8000` | Token in uscita per una lezione. **Troppo basso con un modello di ragionamento = zero caratteri**: a 2000 il budget se n'è andato nel pensiero (2026-09-16) |
| `AIX_WRITER_MIN_CONTENT_TOKENS` | `4000` | Pavimento del secondo tentativo dopo una risposta vuota |
| `AIX_WRITER_MAX_CONTINUATIONS` | `1` | Chiamate di continuazione se la risposta è troncata |
| `AIX_CRITIC_LESSON_MAX_CHARS` | `3000` | Quanta lezione vede il Critic |
| `AIX_CRITIC_CONTEXT_MAX_CHARS` | `1200` | Quanto contesto vede il Critic |
| `AIX_CRITIC_PARSE_ERROR_BEHAVIOR` | `approve` | Cosa fare se la risposta del Critic non si legge |

## 3. Funzioni dietro flag

Tutte spente: §5.2 dice che a flag spente il prodotto è quello di prima.

| Variabile | Default | Cosa accende |
| --- | --- | --- |
| `AIX_UPLOAD_PARSER_ENABLED` | `false` | Parser nuovo sulla graffetta: DOCX, PPTX, immagini, magic bytes, OCR. Spenta: PDF, TXT, MD letti con pypdf |
| `AIX_MATERIALS_ENABLED` | `false` | Storage durevole, tabelle `material*`, coda e worker di ingestione |
| `AIX_PRIVATE_RAG_ENABLED` | `false` | Chunking, embedding e recupero dai documenti della docente. **Il testo esce dalla macchina** verso il provider di embedding |
| `AIX_VALIDATOR_ENABLED` | `false` | Validatore dei vincoli d'aula fra Writer e Critic |
| `AIX_CONDITIONAL_RETRIEVAL_ENABLED` | `false` | Un turno che non porta concetti nuovi riusa il corpus del thread |
| `AIX_CORRECTIVE_RAG_ENABLED` | `false` | Nodo che giudica il recupero e riprova |
| `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS` | `2` | Quante volte può riprovare (limitata fra 1 e 4) |
| `AIX_EXPORT_PDF_ENABLED` | `true` | Bottone PDF (server-side) |
| `AIX_EXPORT_OUTLINE_ENABLED` | `true` | Bottone Scaletta |

### Parsing e OCR (solo con il parser acceso)

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `AIX_PARSE_PDF_ENABLED` · `AIX_PARSE_DOCX_ENABLED` · `AIX_PARSE_PPTX_ENABLED` · `AIX_PARSE_IMAGES_ENABLED` | `true` | Spegnere **un** formato che dà problemi senza spegnere gli altri |
| `AIX_PARSE_OCR_ENABLED` | `true` | OCR sulle pagine senza testo |
| `AIX_OCR_ENABLED` | `true` | Interruttore generale dell'OCR |
| `AIX_OCR_LANGUAGES` | `ita+eng` | Lingue di Tesseract |
| `TESSERACT_CMD` | `tesseract` | Percorso del binario, se non è nel PATH (serve su Windows) |

### RAG privato (solo con `AIX_PRIVATE_RAG_ENABLED`)

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `AIX_PRIVATE_EMBEDDING_MODEL` | `baai/bge-m3` | Modello di embedding |
| `AIX_PRIVATE_EMBEDDING_DIMENSION` | `1024` | **Deve combaciare con la migrazione**: la colonna è `vector(1024)`. Cambiarlo è una migrazione più un re-embedding di tutto |
| `AIX_PRIVATE_RAG_TOP_K` | `8` | Passaggi recuperati per richiesta |
| `AIX_PRIVATE_RAG_BUDGET_CHARS` | `12000` | Quanto prompt spetta al materiale privato (§8.7) |
| `AIX_PRIVATE_RAG_HYBRID` | `true` | Fonde la ricerca per parole con quella vettoriale (§8.5). Spegnerla torna al solo vettoriale: serve a bisezionare una regressione, non è una configurazione da tenere — sui casi d'oro l'ibrida è migliore su tutte le misure |
| `AIX_VALIDATOR_MAX_RETRIES` | `2` | Riscritture che il validatore può chiedere |

## 4. WebUI, database, sicurezza

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `WEBUI_DATABASE_URL` | SQLite in `data/webui/webui.db` | In produzione: `postgresql+asyncpg://…` |
| `WEBUI_DB_ECHO` | `0` | Stampa ogni SQL nel log |
| `AIX_DB_AUTO_MIGRATE` | `true` | Migrazioni all'avvio. Spenta: `alembic upgrade head` a mano |
| `WEBUI_CSRF_ENABLED` | `true` | Protezione CSRF. **Spegnerla è per un rollout, non è una configurazione supportata** |
| `WEBUI_COOKIE_NAME` | `aix_webui_auth` | Nome del cookie di sessione |
| `WEBUI_COOKIE_SECURE` | `0` | Cookie solo su HTTPS: `1` in produzione |
| `WEBUI_TOKEN_LIFETIME_SECONDS` | `86400` | Durata della sessione |
| `WEBUI_CORS_ALLOW_ORIGINS` | `*` | Origini ammesse sull'API |
| `GRAPH_API_USERNAME` · `GRAPH_API_PWD` | vuoti | Basic Auth per i servizi che chiamano `/api/v1` |
| `ENVIRONMENT` | `production` | `development` allenta alcune protezioni |
| `CODE_VERSION` | — | Scritta nell'immagine al build; compare in `/api/v1/health` |
| `SENTRY_DSN` | vuoto | Invio degli errori |
| `LOG_LEVEL` | `INFO` | Verbosità |

## 5. Materiali: storage e coda

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `AIX_STORAGE_PROVIDER` | `local` | `local` o `s3` |
| `AIX_STORAGE_LOCAL_ROOT` | `data/webui/materials` | In produzione è un volume dedicato (`app_materials`) |
| `AIX_STORAGE_S3_ENDPOINT` | — | Obbligatoria con `AIX_STORAGE_PROVIDER=s3`: se manca, errore all'avvio |
| `AIX_STORAGE_S3_BUCKET` | — | Come sopra |
| `AIX_STORAGE_S3_ACCESS_KEY` | — | Credenziali S3 |
| `AIX_STORAGE_S3_SECRET_KEY` | — | Credenziali S3 |
| `AIX_STORAGE_S3_REGION` | — | Regione, se il provider la richiede |
| `AIX_INGESTION_POLL_S` | `3.0` | Intervallo di polling del worker a coda vuota |

## 6. Media e risorse esterne

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `YOUTUBE_API_KEY` | — | Senza: nessun video suggerito |
| `SEMANTIC_SCHOLAR_API_KEY` | — | Senza: quota anonima, più lenta |
| `CANVA_API_KEY` | — | Generatore Canva (sperimentale) |
| `AIX_MEDIA_CACHE_BACKEND` | `diskcache` | `diskcache` o `redis` |
| `AIX_MEDIA_CACHE_DIR` | `artifacts/media_cache` | Cache su disco |
| `AIX_MEDIA_CACHE_NAMESPACE` | `aix:media:v1` | Prefisso delle chiavi |
| `AIX_MEDIA_REDIS_URL` · `REDIS_URL` | — | Redis, se scelto come backend |

## 7. Embedding del Knowledge Graph

Distinti da quelli del RAG privato: qui si tratta di metadati pubblici di video e
paper, là del contenuto delle lezioni della docente. Regole diverse, dichiarate.

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `EMBEDDING_MODE` | `hybrid_semantic` | Come si combinano node2vec e semantico |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Modello |
| `OPENAI_EMBEDDING_MODEL` | come sopra | Alias storico |
| `EMBEDDING_NODE2VEC_WEIGHT` | `0.4` | Peso della parte strutturale |
| `EMBEDDING_SEMANTIC_THRESHOLD` | `0.7` | Soglia di similarità |
| `EMBEDDINGS_CACHE_DIR` | `artifacts/embeddings_cache` | Cache |
| `EMBEDDING_CACHE_EMBEDDINGS` | `true` | Se salvarli |

## 8. Agente: checkpoint e concorrenza

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `LANGGRAPH_CHECKPOINTER_URL` | SQLite locale | Dove vivono i checkpoint del thread |
| `LANGGRAPH_PG_POOL_MAX` | `20` | Pool su PostgreSQL |
| `AIX_RUN_REGISTRY` | `auto` | `memory` o `redis` per il registro dei run |
| `AIX_CONVERSATION_WINDOW_TURNS` | `4` | Turni di storia passati al modello |
| `AIX_COVERAGE_HEALTHY_THRESHOLD` | `5` | Nodi sotto cui il recupero è «povero» |

## 9. MCP e text2cypher

| Variabile | Default | A cosa serve |
| --- | --- | --- |
| `AIX_MCP_REQUIRE_AUTH` | `1` | Autenticazione sul server MCP |
| `AIX_MCP_LOG_LEVEL` · `AIX_MCP_VERSION` | — | Diagnostica |
| `TEXT2CYPHER_MODEL` | vedi `core/config.py` | Modello che genera il Cypher |
| `TEXT2CYPHER_DEFAULT_LIMIT` | vedi `core/config.py` | `LIMIT` di default nelle query generate |
| `TEXT2CYPHER_MAX_QUERY_LENGTH` | vedi `core/config.py` | Lunghezza massima della query generata |
| `TEXT2CYPHER_ENABLE_EXECUTION` | `false` | **Esegue** il Cypher generato |
| `TEXT2CYPHER_ENABLE_VALIDATION` | `true` | Valida prima di eseguire |
| `NEO4J_DATABASE` | `neo4j` | Database dentro l'istanza |
| `NEO4J_USERNAME` | come `NEO4J_USER` | Alias storico, letto in un solo punto (`webui/lessons/routes.py`) |
| `LANGFUSE_HOST` | come `LANGFUSE_BASE_URL` | Alias storico |
| `NEO4J_ENCRYPTED` | `1` | TLS verso Neo4j |
| `LANGFUSE_PUBLIC_KEY` · `LANGFUSE_SECRET_KEY` · `LANGFUSE_BASE_URL` | — | Prompt gestiti fuori dal codice |

---

## Rimosse il 2026-09-16

| Variabile | Perché |
| --- | --- |
| `API_CONTAINER_NAME` · `API_CONTAINER_PORT` · `NODE2VEC_MODEL_DIR` | Nessun file del repository le nomina |
| `LMSTUDIO_BASE_URL` · `LMSTUDIO_MODEL` | Le usano solo due script in `scripts/media_pool/`, mai l'applicazione |
