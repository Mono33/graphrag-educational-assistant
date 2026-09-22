# Strategia di miglioramento — WebUI e pipeline Agentic GraphRAG

**Versione:** 1.1
**Data:** 2026-08-05
**Ultimo aggiornamento:** 2026-08-09
**Autore:** Angelo Casali
**Integrazione pilot/metriche:** Louis Mono
**Fonte:** `milgioramentowebUi_agentic GraphRAG.md` (7 osservazioni di prodotto + analisi tecnica)
**Stato:** proposta — nessuna implementazione avviata

**Documenti collegati (non sostituiti da questo piano):**

- `[Pilot_Registration_Approaches.md](Pilot_Registration_Approaches.md)` — gate di accesso al pilot; per l'onboarding FEM si adotta l'**Approccio B** (`@fem.digital` + OTP email)
- `[AI_Evaluation_Metrics_Strategy.md](AI_Evaluation_Metrics_Strategy.md)` — metriche, golden set, criteri di successo e framework (Ragas/DeepEval/Langfuse)

---



## Indice

1. [Sintesi esecutiva](#1-sintesi-esecutiva)
2. [Ambiente di test locale](#2-ambiente-di-test-locale)
3. [Diagnosi](#3-diagnosi)
4. [Fase 0 — Harness record/replay](#4-fase-0--harness-recordreplay)
5. [Fase 1 — Fondamenta della fattibilità](#5-fase-1--fondamenta-della-fattibilità)
6. [Fase 2 — Risorse: qualità, persistenza, libreria](#6-fase-2--risorse-qualità-persistenza-libreria)
7. [Fase 3 — Documento strutturato a blocchi](#7-fase-3--documento-strutturato-a-blocchi)
8. [Fase 4 — UX, trace, multimodalità](#8-fase-4--ux-trace-multimodalità)
9. [Effort e sequenza](#9-effort-e-sequenza)
10. [Igiene dati](#10-igiene-dati)
11. [Uso di LangGraph](#11-uso-di-langgraph)
12. [Gate pilot — autenticazione Approccio B](#12-gate-pilot--autenticazione-approccio-b)
13. [Integrazione metriche AI e criteri di successo](#13-integrazione-metriche-ai-e-criteri-di-successo)

---



## 1. Sintesi esecutiva

L'innesco di questo documento è il punto 1 della lista di prodotto: **le risorse web recuperate in modalità AUTO finiscono spesso fuori scope didattico**, rimandando a contenuti commerciali che non possono stare in una piattaforma per la scuola.

L'analisi del codice mostra però che quello è il sintomo più visibile di **sette difetti strutturali indipendenti**, e che il problema più grave non riguarda la UI: la pipeline produce lezioni **non eseguibili nell'aula descritta dal profilo**, e il Critic le approva con punteggi alti. La causa non è un LLM disobbediente — è che i vincoli d'aula non gli vengono mai passati.

### I sette difetti


| ID     | Difetto                                                                                                            | Conseguenza visibile                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **D1** | Il Writer riceve un profilo mutilato: manca l'intero blocco `classroom` e il numero di studenti                    | Attività BYOD in aule senza dispositivi; gruppi cooperativi da 8                                  |
| **D2** | Le etichette BES arrivano al modello come codici enum grezzi                                                       | `DCGM` reinterpretato come disprassia, `DLDS` come disturbo del linguaggio invece che italiano L2 |
| **D3** | Nessun controllo deterministico di fattibilità: giudica solo il Critic LLM                                         | Lezioni irrealizzabili approvate al 91%                                                           |
| **D4** | La ricerca web AUTO filtra solo lessicalmente, senza allowlist né soglia semantica                                 | Risorse commerciali e fuori ordine di scuola nel pannello                                         |
| **D5** | I media non vengono persistiti                                                                                     | Il pannello risorse si svuota alla riapertura della lezione                                       |
| **D6** | La lezione è una stringa Markdown senza struttura                                                                  | "Raffina" rigenera tutto e distrugge le parti già approvate                                       |
| **D7** | LangGraph è usato al minimo: il trace mostra segnaposto sintetici, e tre funzionalità native sono riscritte a mano | Spinner invece di progresso reale; token stream rotto in multi-worker; planner eseguito due volte |




### Copertura della lista di prodotto


| #   | Osservazione                                          | Severità          | Intervento                                                                                                  |
| --- | ----------------------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | Risorse multimediali web fuori scope                  | BLOCKING          | [2.1](#21-allowlist--rerank-semantico-per-la-ricerca-web-auto)                                              |
| 2   | Risorse non salvate da nessuna parte                  | BLOCKING          | [2.2](#22-persistenza-dei-media)                                                                            |
| 3   | Manca una sezione "risorse salvate"                   | IMPROVEMENT       | [2.3](#23-libreria-le-mie-risorse)                                                                          |
| 4   | Risposta come blocco unico, non modificabile a pezzi  | IMPROVEMENT       | [3.1](#31-modello-lessondocument) – [3.3](#33-rigenerazione-per-blocco)                                     |
| 5   | UX di salvataggio del profilo educativo               | IMPORTANT         | [4.1](#41-profilo-versionato--ux-di-salvataggio)                                                            |
| 6   | Lezione più interattiva, link e fonti embedded        | IMPROVEMENT       | [1.2](#12-tool-registry) + [3.4](#34-chip-di-tracciabilità-e-onestà-dei-segnali)                            |
| 7   | Artefatti multimodali su piattaforme OSS              | IMPROVEMENT       | [4.3](#43-visualir--renderer-deterministici)                                                                |
| —   | BYOD, acronimi BES, gruppi da 8 *(analisi tecnica)*   | BLOCKING di fatto | [1.1](#11-render-completo-del-profilo--etichette-canoniche) + [1.3](#13-constraintvalidator-deterministico) |
| —   | Trace con segnaposto invece di progresso reale        | IMPROVEMENT       | [11.1](#111-progresso-reale-al-posto-dei-segnaposto)                                                        |
| —   | Registrazione aperta / accesso pilot non controllato  | BLOCKING pilot    | [§12 Approccio B](#12-gate-pilot--autenticazione-approccio-b)                                               |
| —   | Nessun framework di metriche/successo oltre al Critic | BLOCKING misura   | [§13 Metriche AI](#13-integrazione-metriche-ai-e-criteri-di-successo)                                       |




### Da dove partire

Tre binari distinti, da avviare in parallelo dove possibile:

1. **Gate pilot (Approccio B)** — chi può entrare: solo email `@fem.digital` verificate via OTP. Indipendente dalle Fasi 1–4; necessario prima dell'onboarding docenti. Dettaglio in [§12](#12-gate-pilot--autenticazione-approccio-b) e `[Pilot_Registration_Approaches.md](Pilot_Registration_Approaches.md)`.
2. **Intervento 1.1: 6-10 ore** — nessuna dipendenza da Neo4j né da LLM per i test. Da solo sistema buona parte delle lezioni irrealizzabili, prima ancora del ConstraintValidator. È il miglior rapporto valore/costo del piano prodotto.
3. **Telemetria minima + harness Fase 0** — persistere `meta_json` ricco e registrare trace/replay, così i fix di fattibilità diventano misurabili. Dettaglio in [§13](#13-integrazione-metriche-ai-e-criteri-di-successo) e `[AI_Evaluation_Metrics_Strategy.md](AI_Evaluation_Metrics_Strategy.md)`.

**Effort prodotto (questo documento):** 234-342h (31-45 giorni-uomo) sulle Fasi 0–4 + §11.  
**Effort gate auth (Approccio B):** circa 6–10 giorni aggiuntivi, documentati a parte.  
**Effort metriche (backbone MVP):** circa 2–4 settimane aggiuntive per golden set + runner + telemetria, documentate a parte.

Le Fasi 1 e 2 (98-136h) chiudono entrambi i BLOCKING di prodotto e sono ciò che, se un valutatore apre lo strumento, fa la differenza tra "generatore di testo carino" e "sistema che ragiona sui vincoli reali di un'aula". Il gate auth e le metriche non sostituiscono quel lavoro: rendono il pilot **accessibile in modo controllato** e **valutabile**.

---



## 2. Ambiente di test locale

> ⚠️ Il `[docs/runbooks/QUICKSTART.md](../runbooks/QUICKSTART.md)` attuale è **obsoleto**: rimanda a `streamlit run apps/streamlit/main.py`, che è l'app legacy di esplorazione, non il prodotto. Per il percorso WebUI fa fede questa sezione.



### 2.1 Due modi di avviare

**A) Iterazione veloce su UI e template** — SQLite, hot reload, porta 8000:

```powershell
uvicorn aix.api.main:app --reload --port 8000      # equivalente a: make api
```

Poi `http://127.0.0.1:8000/webui/`. È la modalità da usare per il 90% del lavoro: qualsiasi modifica a template, CSS o codice Python ricarica da sola.

**B) Validazione fedele con Postgres e checkpointer multi-turn** — porta 8765:

```powershell
docker compose -f deploy/docker-compose.dev-postgres.yml up -d

$env:WEBUI_DATABASE_URL     = "postgresql+asyncpg://aix:aixdevpass@127.0.0.1:5432/aix"
$env:LANGGRAPH_DATABASE_URL = "postgresql://aix:aixdevpass@127.0.0.1:5432/aix"

python scripts/ops/run_local.py
```



### 2.2 Trappole reali, da conoscere prima

`run_local.py` **non supporta** `--reload`**, ed è deliberato.**
Vedi `[scripts/ops/run_local.py:51-56](../../scripts/ops/run_local.py#L51-L56)`: il reloader di uvicorn possiede i propri processi figli e ne configura l'event loop, reintroducendo esattamente il bug Windows/psycopg che quel launcher esiste per evitare (uvicorn 0.36+ passa una loop factory che su Windows restituisce `ProactorEventLoop`, ignorando la policy globale; psycopg async allora non si connette e il checkpointer muore con `PoolTimeout` a 30s, degradando la memoria multi-turn a single-turn). **Dopo ogni modifica al codice il processo va riavviato a mano.**

`127.0.0.1`**, mai** `localhost`**, nelle URL Postgres.**
Su Windows libpq tenta prima IPv6 `::1`, ma Docker pubblica la porta solo su IPv4. Il checkpointer riscrive `localhost`→`127.0.0.1` come rete di sicurezza, ma impostarlo correttamente tiene veloci entrambi i driver.

**Senza variabili Postgres il WebUI usa SQLite** in `data/webui/webui.db`. Va benissimo per tutta la Fase 1 e la Fase 2.

**Neo4j serve sempre per il retrieval.** Senza connessione il retriever torna vuoto e la lezione degrada a puro LLM. Utile saperlo per non attribuire al codice nuovo un fallimento di connessione:

```powershell
python -c "from aix.core.config import config; print(config.neo4j.uri)"
python scripts/audit/audit_domain_graph.py neuro
```



### 2.3 Matrice "cosa serve per testare cosa"

Evita di montare l'intero stack per verifiche che non lo richiedono.


| Intervento              | Neo4j | LLM (OpenRouter)        | Postgres                         | Rete esterna |
| ----------------------- | ----- | ----------------------- | -------------------------------- | ------------ |
| 1.1 render profilo      | —     | — *(unit su stringa)*   | —                                | —            |
| 1.2 Tool Registry       | —     | —                       | —                                | —            |
| 1.3 ConstraintValidator | —     | — *(nodo puro Python)*  | —                                | —            |
| 1.4 retry mirato        | sì    | sì                      | —                                | —            |
| 2.1 filtro web AUTO     | —     | sì *(embedding rerank)* | —                                | sì *(DDGS)*  |
| 2.2 persistenza media   | —     | —                       | opzionale                        | —            |
| 2.3 libreria risorse    | —     | —                       | opzionale                        | —            |
| 3.1-3.5 blocchi         | sì    | sì                      | opzionale                        | —            |
| 4.1 profilo versionato  | —     | —                       | opzionale                        | —            |
| 4.2 trace collassabile  | —     | — *(con replay Fase 0)* | —                                | —            |
| 4.3 renderer            | —     | — *(golden file)*       | —                                | —            |
| 11.1 progresso reale    | sì    | sì                      | —                                | —            |
| 11.2 interrupt intento  | sì    | sì                      | **sì** *(serve il checkpointer)* | —            |
| 11.3 time-travel        | sì    | sì                      | **sì**                           | —            |
| 11.4 retry/cache policy | —     | — *(errori simulati)*   | —                                | —            |


**La maggior parte della Fase 1 si testa senza toccare Neo4j e senza spendere token.** È esattamente ciò che la rende il posto giusto da cui partire.

### 2.4 Test suite

```powershell
pytest tests/ -v                    # tutto
pytest tests/unit/ -v               # solo unit, veloce, nessuna dipendenza esterna
pytest tests/ -v -m "not integration"
black --check --line-length 120 src/
```

`tests/integration/` colpisce Neo4j e OpenRouter reali. La configurazione pytest è in `pyproject.toml` (`asyncio_mode = "auto"`, marker `integration`/`slow`/`unit`).

---



## 3. Diagnosi

Sette difetti, tutti verificati leggendo il codice.

### D1 — Il Writer riceve un profilo mutilato

`[writer_agent.py:250-270](../../src/aix/agent/agents/writer_agent.py#L250-L270)` costruisce la sezione di profilo che finisce nel prompt:

```python
edu_profile_section = ""
if educational_profile:
    ep = educational_profile
    group = ep.get("group") or {}
    lines = ["\n## Teacher's Educational Profile"]
    if ep.get("specific_topic"):     lines.append(f"- Topic: {ep['specific_topic']}")
    if ep.get("subject_area"):       lines.append(f"- Subject: {ep['subject_area']}")
    if group.get("grade"):           lines.append(f"- Grade level: {group['grade']}")
    disabilities = group.get("disabilities") or ep.get("disabilities")
    if disabilities:
        if isinstance(disabilities, list):
            disabilities = ", ".join(disabilities)
        lines.append(f"- Learner needs: {disabilities}")
    duration = ep.get("time_available_minutes") or ep.get("lesson_duration")
    if duration:                     lines.append(f"- Duration: {duration} minutes")
```

Cinque campi. **Non compare mai il blocco** `classroom` — `own_device` (BYOD), `has_lim`, `has_wifi`, `has_suite`, `pc_station`, `forniture_mobility` — né `group.students_number`, `class_features`, `student_attributes`.

Il modello dati li raccoglie tutti (`[educational_profile.py:176-219](../../src/aix/api/schemas/educational_profile.py#L176-L219)`), il form li chiede, la sidebar li mostra. Semplicemente non arrivano mai al modello.

> **Non è che l'LLM ignori il vincolo BYOD: non l'ha mai visto.** Stessa origine per il jigsaw da 8 studenti — `students_number` non è nel prompt.



### D2 — Le etichette BES arrivano come codici enum grezzi

Stessa funzione, righe 261-265: `", ".join(disabilities)` produce la stringa `ADHD, DCGM, DLDS`.

Il dizionario con le etichette italiane estese **esiste già** — `DISABILITY_LABELS` in `[educational_profile.py:308](../../src/aix/api/schemas/educational_profile.py#L308)` — ma è usato **solo nei template** (`[profile_sidebar.html:117](../../src/aix/webui/templates/partials/profile_sidebar.html#L117)`). Nel percorso agente non entra mai.

Da lì in poi il modello rilegge gli acronimi a memoria: per `DCGM` produce "evitare scrittura manuale prolungata" — che è disprassia, non disabilità cognitiva media; per `DLDS` produce strategie da disturbo del linguaggio invece che da italiano L2. Il Retriever aveva capito bene (le risorse recuperate parlano di *multilingual learners*); è il Writer a rovinare il segnale comprimendo le etichette.

### D3 — Nessun controllo deterministico di fattibilità

Non esiste alcun `*_validator.py` nel codice applicativo; nessuna occorrenza di `violation` o `feasibility`. L'unico giudice è il Critic LLM con rubrica 1-5 (`[critic_agent.py](../../src/aix/agent/agents/critic_agent.py)`), che valuta **qualità pedagogica e fattibilità con lo stesso strumento soggettivo**.

Anche il vincolo di durata è solo testuale — `[writer_prompt.py:56-61](../../src/aix/agent/prompts/writer_prompt.py#L56-L61)` lo dichiara *"HARD constraint that overrides any default heuristic"* — e nulla verifica che i minuti delle attività sommino alla durata richiesta.

> Mescolare giudizio soggettivo e vincoli oggettivi è il motivo per cui esce 91% su una lezione irrealizzabile.



### D4 — La ricerca web AUTO filtra solo lessicalmente

`[retriever_agent.py:563-649](../../src/aix/agent/agents/retriever_agent.py#L563-L649)` — `_fetch_web_links` è **sempre attiva sul percorso critico** (riga 379, ramo `else` incondizionato).

```python
concepts_to_search = (key_concepts or search_queries)[:3]
quoted = f'"{concept}"' if " " in concept else concept
query = f"{quoted} didattica OR insegnamento OR apprendimento"

def _is_relevant(item):
    text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
    return any(tok in text for tok in concept_tokens)   # UN token > 3 caratteri
```

Un risultato passa se titolo o snippet condividono **un solo token di più di 3 caratteri** con i concetti. La blocklist a monte (`[external_apis.py:1115](../../src/aix/agent/media/external_apis.py#L1115)`) elenca 20 domini — Amazon, eBay, social, volantini — e tutto il resto del web commerciale passa indisturbato.

Mancano: allowlist, soglia di rilevanza, filtro di lingua, filtro per ordine di scuola.

**Il ri-ranking semantico esiste già** in `[media_ranker.py](../../src/aix/agent/media/media_ranker.py)` (`rank_live_media`, embedding + qualità, fail-safe), ma è applicato solo al layer live *off-path*, mai ai `web_links`. È infrastruttura già pagata e non utilizzata dove serve di più.

### D5 — I media non vengono persistiti

Dichiarato nei commenti stessi del template (`[media_panel.html:5-12](../../src/aix/webui/templates/partials/media_panel.html#L5-L12)`). Alla riapertura della lezione:

```python
# lessons/routes.py:977
"media": None,  # not persisted yet — empty placeholder on reload
```

`curated_media` vive solo in `AgentState` durante il run. Stesso destino per le metriche: al reload `meta` è un sintetico

```python
# lessons/routes.py:926
meta_for_replay = {"approved": True, "revision_count": 0} if lesson.status == "complete" else {}
```

quindi quasi tutta la striscia di metadati scompare dopo un refresh.

> È esattamente il punto 2 della lista di prodotto, ed è la precondizione tecnica del punto 3.



### D6 — La lezione è una stringa Markdown, senza struttura

`Lesson.lesson_plan_md` è una colonna `Text`. Nello stato dell'agente:

- `lesson_plan_structured` esiste (`[state.py:139](../../src/aix/agent/graph/state.py#L139)`) ma è **sempre inizializzato a** `None` (`[state.py:263](../../src/aix/agent/graph/state.py#L263)`) e nulla lo popola;
- la dataclass `LessonPlanStructure` (`[state.py:294](../../src/aix/agent/graph/state.py#L294)`) — che sarebbe il modello a blocchi — **non è mai istanziata**.

Di conseguenza "Raffina" rilancia l'intera pipeline e il Writer riceve (`[writer_agent.py:355-363](../../src/aix/agent/agents/writer_agent.py#L355-L363)`):

> *"Apply the improvement above to the lesson you generate below. Produce a complete revised lesson (not just the changed parts)."*

Rigenerare un'attività costa l'intero documento in token e **distrugge le parti che il docente aveva già approvato**.

### D7 — LangGraph è usato al minimo, e tre sue funzionalità sono riscritte a mano

Versione installata: **LangGraph 1.1.10** (`requirements.txt` richiede `>=1.0.0,<2.0.0`). La superficie effettivamente utilizzata è:

`StateGraph(AgentState)` · `add_node` · `add_edge` · `add_conditional_edges` · `set_entry_point` · `compile(checkpointer=…)` · `ainvoke` · `astream(stream_mode="updates")` · `AsyncSqliteSaver` / `AsyncPostgresSaver`.

Non usato — e **tutto disponibile in questa versione**, verificato per introspezione: `get_stream_writer`, `interrupt`, `Command`, `Send`, `RetryPolicy`, `CachePolicy`, `aupdate_state`, `get_state_history`, `BaseStore`, subgraph, `durability`. Le uniche occorrenze di quei nomi nel repo sono commenti che ne parlano al futuro.

Tre conseguenze concrete:

**① Il trace mostra segnaposto sintetici, non il progresso reale.**
`stream_mode="updates"` emette **solo quando un nodo termina**. Durante il nodo retriever — Neo4j, Text2Cypher, media lookup, DDGS: facilmente 10-20 secondi — LangGraph non emette nulla. Il vuoto è riempito da eventi inventati dal service layer, come il codice stesso ammette (`[service.py:30-34](../../src/aix/webui/agent/service.py#L30-L34)`):

> `"writer_pending"` … *emitted right before write starts (**synthetic**; writer is the slow call so the chat must not look stuck)*

`retriever_pending`, `writer_pending` e `critic_pending` non vengono dal grafo: sono spinner.

**② Il token stream del writer è un globale di processo — e il sistema sta andando in multi-worker.**
`[write_stream.py](../../src/aix/agent/graph/write_stream.py)` è un `dict[str, asyncio.Queue]` a livello di modulo, con un secondo `EventSource` separato su `/writer-stream` pilotato da un MutationObserver in `_base.html`. È `stream_mode="custom"` riscritto a mano. Ma soprattutto: **un globale di processo non sopravvive al multi-worker.** Se la richiesta SSE `/writer-stream` atterra su un worker diverso da quello che esegue il grafo, il bus non esiste e il typewriter non parte. La tabella `agent_run` e `[run_registry.py](../../src/aix/core/run_registry.py)` esistono proprio per coordinare più worker (CORE 6 #37/#39): questo pezzo è rimasto indietro.

**③ Il flusso a due fasi dell'intento è un flag smuggled, e il planner gira due volte.**
Il primo run su una bozza è planner-only tramite un flag `__planner_only__` infilato dentro `educational_profile_json`, con il commento che ne ammette il motivo (`[lessons/routes.py:1182](../../src/aix/webui/lessons/routes.py#L1182)`): *"so the service layer can pop it without a schema change"*. Alla conferma del chip la pipeline riparte **da capo**, rieseguendo il planner. È esattamente il caso d'uso di `interrupt()` + `Command(resume=…)`.

**Inoltre:** il checkpointer è attivo ma `aupdate_state` e `get_state_history` non vengono mai chiamati. La colonna `checkpoint_id` su `LessonMessage` è già presente, con il commento *"reserved for time-travel branching (regenerate from a past turn)"* (`[lessons/models.py:266-270](../../src/aix/webui/lessons/models.py#L266-L270)`): infrastruttura pagata e mai usata.

> Gli interventi correttivi sono raccolti nella [sezione 11](#11-uso-di-langgraph).

---



## 4. Fase 0 — Harness record/replay

*Prerequisito operativo · 8-12h*

**Sintomo.** Lavorare sulla UI di un sistema agentico significa oggi rigenerare una lezione vera a ogni modifica di template: 30-60 secondi e token a ogni refresh.

**Causa.** Non esiste alcuna registrazione di risposte LLM. `tests/` usa `unittest.mock` ad hoc dentro i singoli test; nessun VCR, nessuna cassetta, nessuna fixture di trace. `tests/integration/` colpisce Neo4j e OpenRouter reali.

**Cosa si cambia.**

- Nuova cartella `tests/fixtures/traces/` con dump JSON di sequenze `StreamEvent` reali, catturabili da `[service.py:1222 run_agent_stream](../../src/aix/webui/agent/service.py#L1222)` (la dataclass è a `[service.py:136](../../src/aix/webui/agent/service.py#L136)`).
- Flag `AIX_REPLAY_TRACE=<path>`: quando è impostato, `run_agent_stream` serve gli eventi dal file invece di eseguire il grafo. **Tutto il resto del percorso resta identico** — SSE, OOB swap, template, `sse-close`.
- Uno script di cattura sul modello di `[capture_openapi_baseline.py](../../scripts/diagnostic/capture_openapi_baseline.py)`, stesso pattern dei baseline OpenAPI già in repo (`data/diagnostic/openapi_*.txt`).

**Verifica in locale.**

```powershell
$env:AIX_REPLAY_TRACE = "tests/fixtures/traces/lesson_byod.json"
uvicorn aix.api.main:app --reload --port 8000
```

Aprire una lezione: il trace deve scorrere in meno di un secondo, con le stesse card e gli stessi swap di un run vero, a costo zero. Rimuovendo la variabile (`Remove-Item Env:AIX_REPLAY_TRACE`) il comportamento torna quello reale.

**Rischi e flag.** Rischio nullo: senza la variabile d'ambiente il codice è inerte. L'unico rischio è che le fixture invecchino rispetto al formato degli eventi — mitigabile con un test che valida le fixture contro la dataclass `StreamEvent`.

> Da fare per prima anche se sembra tempo "non di prodotto": è il moltiplicatore di produttività delle Fasi 3 e 4, dove ogni modifica di template richiede una lezione generata.

**Piazzamento metriche.** L'harness record/replay è anche il ponte verso il golden set descritto in `[AI_Evaluation_Metrics_Strategy.md](AI_Evaluation_Metrics_Strategy.md)` §§6–7: le cassette `tests/fixtures/traces/` diventano casi di regressione UI; i dump di run reali alimentano i record del dataset (query, profilo, retrieval, Critic, media). Non sostituisce il golden set expert-reviewed, ma abbassa il costo di cattura. Vedi [§13](#13-integrazione-metriche-ai-e-criteri-di-successo).

---



## 5. Fase 1 — Fondamenta della fattibilità

*52-70h · risolve i difetti più gravi, e quasi tutto si testa offline*

### 1.1 Render completo del profilo + etichette canoniche

*Risolve D1 e D2 · 6-10h*

**Sintomo.** La lezione richiede Mentimeter da telefono, escape room a coppie su dispositivi personali, Kahoot individuale, registrazione audio su Padlet — in un'aula dove il profilo dichiara che gli studenti **non hanno dispositivi** e c'è una sola LIM. Quattro attività su cinque non si possono fare. In parallelo, gli acronimi BES vengono reinterpretati a caso.

**Causa.** [D1](#d1--il-writer-riceve-un-profilo-mutilato) e [D2](#d2--le-etichette-bes-arrivano-come-codici-enum-grezzi): `[writer_agent.py:250-270](../../src/aix/agent/agents/writer_agent.py#L250-L270)`.

**Cosa si cambia.**

Nuovo file `src/aix/agent/prompts/profile_renderer.py` con una funzione pura:

```python
def render_profile_section(profile: dict) -> str: ...
```

che fa tre cose che oggi non fa nessuno:

1. **Risolve i codici in etichette leggibili**, riusando i dizionari **già presenti** in `[educational_profile.py](../../src/aix/api/schemas/educational_profile.py)`: `DISABILITY_LABELS`, `GRADE_LABELS`, `CLASS_FEATURE_LABELS`, `STUDENT_ATTR_LABELS`, `FORNITURE_MOBILITY_LABELS`, `OWN_DEVICE_LABELS`. **Nessun acronimo raggiunge più il modello.**
2. **Emette una sezione di vincoli d'aula espliciti**, in linguaggio imperativo invece che come coppie chiave-valore:
  ```
   ## Vincoli d'aula (NON negoziabili)
   - Gli studenti NON dispongono di dispositivi personali.
     Nessuna attività può richiederne l'uso.
   - È disponibile una LIM (unico dispositivo di proiezione condiviso).
   - Non è disponibile connessione WiFi per gli studenti.
   - Gli arredi NON sono spostabili: nessuna attività può richiedere
     la riorganizzazione dei banchi.
   - Classe di 25 studenti: i gruppi cooperativi non devono superare i 5 membri.
  ```
3. **Include i campi oggi ignorati**: `students_number`, `class_features`, `student_attributes`.

Applicare la stessa funzione anche al **Planner** (`[planner_agent.py](../../src/aix/agent/agents/planner_agent.py)`), così il vincolo influenza già la scelta dei concetti da recuperare, non solo la scrittura.

**Verifica in locale.**

Prima, senza rete e senza token:

```powershell
pytest tests/unit/test_profile_renderer.py -v
```

Golden string su un profilo di riferimento: BYOD=NO, LIM=sì, WiFi=no, arredi fissi, 25 studenti, BES = ADHD + disabilità cognitiva media + italiano L2, 120 minuti.

Poi end-to-end:

```powershell
uvicorn aix.api.main:app --reload --port 8000
```

Creare una lezione con quel profilo e generarla.

*Cosa deve succedere:* le attività devono essere realizzabili con LIM condivisa e senza rete; le strategie per l'italiano L2 devono essere da apprendimento linguistico, non da disturbo del linguaggio.
*Cosa deve smettere di succedere:* **nessuna occorrenza** di Mentimeter, Kahoot, Padlet, "sul proprio smartphone", "registra un audio"; **nessun acronimo** che non compaia nel profilo (`DCGM`, `DLDS` e simili non devono apparire nel testo).

Controllo rapido sull'output salvato:

```powershell
Select-String -Path data/webui/*.md -Pattern "Mentimeter|Kahoot|Padlet|DCGM|DLDS"
```

**Rischi e flag.** Il prompt si allunga di ~15 righe: trascurabile rispetto a `AIX_WRITER_MAX_TOKENS` (default 2500). Nessun flag necessario — è una correzione, non una feature. Unico effetto collaterale atteso: lezioni più conservative sugli strumenti digitali, che è il comportamento corretto.

> **Questo è il punto di partenza raccomandato.** Sei-dieci ore, zero dipendenze esterne per i test, e da solo sistema buona parte delle lezioni irrealizzabili — prima ancora del validator.

---



### 1.2 Tool Registry

*Abilita il punto 6 e il filtro d'aula · 14-20h*

**Sintomo.** Gli strumenti digitali citati nelle lezioni escono dalla memoria del modello: nomi plausibili, URL a volte allucinati, nessuna verifica che siano gratuiti, in italiano, adatti all'età o compatibili col GDPR.

**Causa.** Non esiste una fonte di verità per gli strumenti. Per le *strategie* il KG c'è; per gli *strumenti* si sta lasciando decidere al modello a memoria.

**Cosa si cambia.**

Nuovo `src/aix/agent/tools/tool_registry.py` + `data/registry/tools.yaml`. Schema minimo per voce:


| Campo                     | Tipo | Uso                                          |
| ------------------------- | ---- | -------------------------------------------- |
| `tool_id`                 | str  | chiave stabile, citata dal modello           |
| `name`, `url`             | str  | resi dal frontend, mai scritti dall'LLM      |
| `category`                | str  | quiz, lavagna, mappa, video, collaborazione… |
| `requires_student_device` | bool | **regola BYOD del validator**                |
| `works_on_lim_only`       | bool | alternativa in aula senza dispositivi        |
| `needs_account`           | bool | attrito e privacy                            |
| `free_tier`               | bool | sostenibilità per la scuola                  |
| `gdpr_note`               | str  | nota di conformità                           |
| `ui_language`             | str  | filtro lingua                                |
| `is_oss`                  | bool | preferenza strategica                        |
| `min_grade`, `max_grade`  | enum | filtro per ordine di scuola                  |


Seed iniziale con gli strumenti che il modello già cita (Mentimeter, Kahoot, Padlet, Wooclap, Genially, Edpuzzle) **più** le alternative OSS auto-ospitabili (H5P, Excalidraw, LearningApps, uMap, Etherpad). Caricamento con lo stesso pattern lazy-singleton di `[media_lookup.py](../../src/aix/agent/media/media_lookup.py)`.

Il retriever filtra il registry sui vincoli d'aula **prima** di passarlo al Writer: il modello riceve solo strumenti utilizzabili in quell'aula. Il frontend rende gli URL dal registry.

> **Il modello sceglie *quali* strumenti, non *scrive* i link.** È così che si chiude il punto 6 della lista (link e riferimenti embedded) senza alcun rischio di allucinazione.

**Verifica in locale.**

```powershell
pytest tests/unit/test_tool_registry.py -v
```

Test di filtro: dato `own_device=NO, has_lim=True, has_wifi=False`, il registry deve restituire solo strumenti con `requires_student_device=False` e utilizzabili offline; Kahoot e Mentimeter devono essere esclusi.

End-to-end: generare due lezioni identiche cambiando solo BYOD (NO → YES) e verificare che l'insieme di strumenti proposti cambi di conseguenza.

**Rischi e flag.** Il registry è un file YAML editabile senza deploy. Rischio principale: un seed troppo povero rende le lezioni ripetitive — mitigabile partendo da ~30 voci e ampliando in base all'uso.

---



### 1.3 ConstraintValidator deterministico

*Risolve D3 · 24-32h*

**Sintomo.** Una lezione irrealizzabile riceve 91% dal Critic. Il docente si fida del punteggio e scopre in aula che non si può fare.

**Causa.** [D3](#d3--nessun-controllo-deterministico-di-fattibilità).

**Cosa si cambia.**

Nuovo package `src/aix/agent/validation/` (`models.py`, `rules.py`, `validator.py`) e un nuovo nodo nel grafo. Topologia attuale (`[lesson_planner_graph.py:79-115](../../src/aix/agent/graph/lesson_planner_graph.py#L79-L115)`): `plan → retrieve → write → critique → [revise → write | finish]`. Nuova:

```
plan → retrieve → write → validate → critique → [revise → write | finish]
                            ↑            │
                            └─ blocking ─┘   (retry mirato, max 2)
```

Il nodo è **Python puro, zero chiamate LLM**: deterministico, istantaneo, gratuito, testabile offline.

```python
class Violation(BaseModel):
    severity: Literal["blocking", "warning"]
    rule: str              # "byod_off", "duration_mismatch", …
    evidence: str          # la frase esatta della lezione
    block_id: str | None   # popolato dalla Fase 3; prima è il titolo di sezione
    remediation: str       # istruzione da iniettare nel retry
```

**Le prime otto regole**, tutte verificabili senza LLM:


| #   | Regola                                                                            | Severità | Fonte del vincolo                      |
| --- | --------------------------------------------------------------------------------- | -------- | -------------------------------------- |
| 1   | `byod_off` — attività che richiede un dispositivo studente con `own_device=NO`    | blocking | Tool Registry + `classroom.own_device` |
| 2   | `no_wifi` — strumento online senza `has_wifi`                                     | blocking | `classroom.has_wifi`                   |
| 3   | `no_lim_no_device` — nessun supporto di proiezione disponibile                    | blocking | `has_lim` + `pc_station`               |
| 4   | `group_size` — gruppi cooperativi > 5 membri                                      | warning  | `group.students_number`                |
| 5   | `duration_mismatch` — somma dei minuti fuori dal ±10%                             | blocking | `time_available_minutes`               |
| 6   | `furniture_static` — richiede riorganizzazione banchi con `forniture_mobility=NO` | blocking | `classroom.forniture_mobility`         |
| 7   | `unknown_tool` — strumento citato assente dal registry                            | warning  | Tool Registry                          |
| 8   | `bes_label_drift` — acronimo BES nel testo non presente nel profilo               | blocking | `group.disabilities`                   |


La regola 5 richiede i minuti per attività: prima della Fase 3 si estraggono con regex dai heading (come già fa `[_parse_lesson_minutes](../../src/aix/agent/agents/retriever_agent.py#L412)` per il `time_constraints`); dopo, si leggono dal modello strutturato. La regola 8 è la **sentinella di regressione su D2**: se qualcuno reintroduce gli acronimi nel prompt, il test fallisce.

**UI.** Nuovo partial `partials/chat_validator_card.html` sullo stile di `[chat_critic_card.html](../../src/aix/webui/templates/partials/chat_critic_card.html)`, nuovo `kind="validator"` nella dataclass `StreamEvent` (`[service.py:136](../../src/aix/webui/agent/service.py#L136)`) e relativo mapping in `_stream_event_to_sse` (`[lessons/routes.py:2115](../../src/aix/webui/lessons/routes.py#L2115)`). Colore dedicato tra i token `--aix-`* in `[aix-brand.css](../../src/aix/webui/static/css/aix-brand.css)`.

> **Separazione di responsabilità:** il Critic LLM giudica la *qualità pedagogica* — soggettiva, e va bene che lo faccia un modello. Il validator giudica la *fattibilità* — oggettiva, e non deve dipendere dal buon cuore di un LLM. È il mescolarli che produce 91% su una lezione irrealizzabile.

**Verifica in locale.**

```powershell
pytest tests/unit/test_constraint_validator.py -v
```

Otto casi positivi e otto negativi, su lezioni di test fissate su disco. Nessuna dipendenza esterna: si esegue in meno di un secondo.

End-to-end con il profilo di riferimento (BYOD=NO, 25 studenti, 120 min):
*Cosa deve succedere:* la card del validator compare tra Writer e Critic, e mostra **0 violazioni bloccanti** prima che il Critic assegni un punteggio.
*Cosa deve smettere di succedere:* un punteggio alto non deve più poter convivere con attività che richiedono dispositivi assenti.

Per verificare il retry, forzare temporaneamente una violazione (es. `AIX_VALIDATOR_FORCE_RULE=byod_off`): la lezione deve essere rigenerata e la seconda passata deve risultare pulita.

**Rischi e flag.** `AIX_VALIDATOR_ENABLED` (default `true`) e `AIX_VALIDATOR_MAX_RETRIES` (default 2), seguendo la convenzione `AIX_`* già in `[.env.example](../../.env.example)`. Rischio principale: **falsi positivi** che bloccano lezioni valide — per questo le regole 4 e 7 partono come `warning` e non innescano retry. Le regole vanno tarate su un set di lezioni reali prima di promuoverle a `blocking`.

**Piazzamento metriche.** Questa separazione Critic (qualità) vs validator (fattibilità) è il prerequisito della strategia di valutazione: il Critic resta segnale di pipeline, non ground truth; le violazioni del validator sono metriche deterministiche di Layer 1/5 (feasibility pass rate, blocking violations per rule). I casi golden devono includere profili BYOD=NO / BES / durata stretta proprio per tarare queste regole. Vedi [§13](#13-integrazione-metriche-ai-e-criteri-di-successo) e `AI_Evaluation_Metrics_Strategy.md` §§5.1, 5.4–5.5.

---



### 1.4 Retry mirato

*4-6h*

**Cosa si cambia.** Su violazione `blocking`, iniettare **solo** `remediation` + `evidence` nel prompt del Writer, riusando il canale `refinement_instruction` già esistente: `[nodes.py:525](../../src/aix/agent/graph/nodes.py#L525)` → `[writer_agent.py:355](../../src/aix/agent/agents/writer_agent.py#L355)`. Nessun meccanismo nuovo.

Fino alla Fase 3 il retry rigenera l'intero documento; dopo, solo il blocco che ha violato.

**Verifica in locale.** Il test del punto 1.3 con violazione forzata copre anche questo. Controllare nei log che il prompt di retry contenga la sola istruzione di rimedio, non l'intera lista di vincoli ripetuta.

---



## 6. Fase 2 — Risorse: qualità, persistenza, libreria

*46-66h · chiude entrambi i BLOCKING della lista di prodotto*

### 2.1 Allowlist + rerank semantico per la ricerca web AUTO

*Risolve D4 — punto 1 BLOCKING · 16-24h*

**Sintomo.** La sezione "Ricerca web" del pannello risorse propone contenuti commerciali, portali generalisti e materiale fuori ordine di scuola. Le risorse recuperate sono spesso in inglese e riferite a *higher education* anche per una classe di secondaria di primo grado.

**Causa.** [D4](#d4--la-ricerca-web-auto-filtra-solo-lessicalmente).

**Cosa si cambia.**

Quattro filtri in cascata, dal più economico al più costoso, in sostituzione di `_is_relevant` (`[retriever_agent.py:598](../../src/aix/agent/agents/retriever_agent.py#L598)`):

**① Allowlist di domini didattici** — nuovo `data/registry/web_sources.yaml`, editabile senza deploy:

- istituzionali: `miur.gov.it`, `istruzione.it`, `indire.it`, `invalsi.it`, `erasmus-plus.ec.europa.eu`
- accademici: `*.edu`, `*.ac.uk`, domini universitari italiani
- OER ed enciclopedici: `oercommons.org`, `khanacademy.org`, `wikipedia.org`, `openstax.org`, `doabooks.org`
- riviste ed editori scolastici con contenuto aperto

Due modalità: `strict` (solo allowlist) e `soft` (allowlist prioritaria, il resto ammesso se supera la soglia semantica). **Default** `strict` finché non validata su lezioni reali.

**② Blocklist euristica estesa** — oltre ai 20 domini di `[external_apis.py:1115](../../src/aix/agent/media/external_apis.py#L1115)`, pattern su URL e snippet per e-commerce e lead generation: `/prodotto/`, `/shop/`, `/corso-a-pagamento`, e nel testo `prezzo`, `acquista`, `iscriviti ora`, `€`, `sconto`.

**③ Rerank semantico con soglia** — riuso diretto di `rank_live_media` da `[media_ranker.py](../../src/aix/agent/media/media_ranker.py)`, estendendo `_BUCKET_TEXT_FIELDS` con `"web": ("title", "snippet")`. Query di ranking costruita come già fa `_live_media_ranking_query` (`[lessons/routes.py:333](../../src/aix/webui/lessons/routes.py#L333)`): argomento + query pulita + materia + grado.

> **Sotto** `AIX_WEB_MIN_RELEVANCE` **(default 0.35) il risultato si scarta. Se nessun risultato supera la soglia, la sezione "Ricerca web" non compare affatto.**
>
> Questo è un **cambio di comportamento visibile**: oggi il pannello mostra sempre qualcosa. Mostrare meno — e talvolta niente — è la scelta corretta per una piattaforma didattica, ma va comunicata, perché un docente potrebbe leggere l'assenza della sezione come un malfunzionamento. Prevedere una riga esplicativa: *"Nessuna risorsa web ha superato la soglia di pertinenza per questo argomento."*

**④ Filtro di lingua e ordine di scuola** — aggiungere alla query un qualificatore di grado derivato da `GRADE_LABELS`, e scartare i risultati non italiani quando il profilo è italiano. Risolve le 8 risorse tutte in inglese su *higher education*.

**Verifica in locale.**

Serve rete (DDGS) e LLM (embedding per il rerank).

```powershell
pytest tests/integration/test_web_source_filter.py -v -m integration
```

Set di 15 query reali su materie diverse con esiti attesi; con gli snapshot DDGS della Fase 0 diventa eseguibile offline e deterministico.

End-to-end:

```powershell
uvicorn aix.api.main:app --reload --port 8000
```

Generare 5 lezioni su materie diverse (storia, matematica, scienze, italiano, lingua straniera) e ispezionare la sezione "Ricerca web" di ciascuna.

*Cosa deve succedere:* ogni link è una fonte didattica, istituzionale o accademica, in italiano, coerente con l'ordine di scuola del profilo.
*Cosa deve smettere di succedere:* nessun e-commerce, nessun portale generalista, nessun corso a pagamento, nessun risultato in inglese su *higher education* per una classe di secondaria di primo grado.

Confronto A/B rapido, per misurare l'effetto:

```powershell
$env:AIX_WEB_SOURCE_MODE = "soft"    # comportamento permissivo
# genera, annota i risultati
$env:AIX_WEB_SOURCE_MODE = "strict"  # comportamento nuovo
# rigenera la stessa lezione e confronta
```

**Rischi e flag.**

- `AIX_WEB_SEARCH_ENABLED` (default `true`) — **kill-switch**: se la qualità resta insufficiente sul set di validazione, si disattiva l'intera sezione senza toccare il codice.
- `AIX_WEB_SOURCE_MODE` = `strict` | `soft` (default `strict`).
- `AIX_WEB_MIN_RELEVANCE` (default `0.35`).

Rischio principale: allowlist troppo stretta → sezione quasi sempre vuota. La modalità `soft` è la valvola di sfogo, e la soglia va tarata sul set di 15 query prima del rilascio. Costo aggiuntivo: una chiamata di embedding per lezione (trascurabile, e già pagata sul percorso live-media).

**Piazzamento metriche.** Gli A/B `soft` vs `strict` e la soglia `AIX_WEB_MIN_RELEVANCE` vanno misurati con le metriche di retrieval/media della strategia AI: share of allowlisted domains, relevance≥threshold, off-scope commercial rate, language/school-level mismatch. Il set di 15 query di taratura diventa slice del golden set (retrieval-focused). Vedi [§13](#13-integrazione-metriche-ai-e-criteri-di-successo) e `AI_Evaluation_Metrics_Strategy.md` §5.2.

---



### 2.2 Persistenza dei media

*Risolve D5 — punto 2 BLOCKING · 10-14h*

**Sintomo.** Riaprendo una lezione dalla sezione "Le mie lezioni", il pannello risorse è vuoto. Le risorse recuperate durante la generazione sono perse, anche in blocco.

**Causa.** [D5](#d5--i-media-non-vengono-persistiti).

**Cosa si cambia.**

Due nuove colonne su `Lesson` (`[lessons/models.py](../../src/aix/webui/lessons/models.py)`):

```python
curated_media_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
media_live_json:    Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
```

Scritte insieme a `lesson_plan_md` alla conclusione del run, e rilette al reload al posto di `"media": None` (`[lessons/routes.py:977](../../src/aix/webui/lessons/routes.py#L977)` e `:1337`). Il template `media_panel.html` non richiede modifiche: riceve già `media` come parametro.

Nello stesso passaggio, persistere il `meta_json` completo sull'ultima assistant turn, così che la striscia di metriche non collassi nel sintetico di `[lessons/routes.py:926](../../src/aix/webui/lessons/routes.py#L926)`.

**Migrazione.** Il repo **non usa Alembic**. Le colonne si aggiungono al pattern esistente `_apply_dev_schema_hotpatches()` in `[webui/db.py:192-227](../../src/aix/webui/db.py#L192-L227)`, che già gestisce `lesson_plan_md`, `error_message`, `teacher_query`, `uploaded_files_json`:

```python
("lesson", "curated_media_json", "ALTER TABLE lesson ADD COLUMN curated_media_json TEXT"),
("lesson", "media_live_json",    "ALTER TABLE lesson ADD COLUMN media_live_json TEXT"),
```

Su Postgres l'`ALTER TABLE` gira già sotto il `pg_advisory_xact_lock` che protegge l'inizializzazione multi-worker.

**Verifica in locale.**

```powershell
uvicorn aix.api.main:app --reload --port 8000
```

1. Generare una lezione, annotare quante risorse compaiono in ciascuna sezione del pannello.
2. Ricaricare `/webui/lesson/{id}` con F5.

*Cosa deve succedere:* il pannello si ripopola identico, con gli stessi conteggi e gli stessi badge di provenienza (`✓ Verificato` / `auto`).
*Cosa deve smettere di succedere:* il pannello non deve più tornare allo stato "Le risorse multimediali appariranno qui…".

Controllo diretto sul DB:

```powershell
python -c "import sqlite3,json; c=sqlite3.connect('data/webui/webui.db'); r=c.execute('SELECT curated_media_json FROM lesson ORDER BY updated_at DESC LIMIT 1').fetchone(); print(json.dumps(json.loads(r[0]), indent=2)[:800])"
```

Verificare anche la retrocompatibilità: le lezioni **preesistenti** avranno `NULL` e devono continuare a renderizzare lo stato vuoto senza errori.

**Rischi e flag.** Nessun flag: è una correzione. Rischio contenuto — crescita della dimensione delle righe (qualche KB per lezione, irrilevante). Attenzione a serializzare solo dati, non oggetti Pydantic.

**Piazzamento metriche.** Persistere `meta_json` completo (non il sintetico `approved/revision_count`) è il prerequisito MVP della telemetria online in `AI_Evaluation_Metrics_Strategy.md` §8.2: approval, revision count, Critic scores, node timings, media counts, e in seguito validator outcomes. Senza questo, dashboard e scorecard pilot non hanno dati al reload. Vedi [§13](#13-integrazione-metriche-ai-e-criteri-di-successo).

---



### 2.3 Libreria "Le mie risorse"

*Punto 3 IMPROVEMENT · 20-28h*

**Sintomo.** Non c'è modo di selezionare e conservare risorse dopo averle viste. Ogni lezione riparte da zero e il lavoro di selezione del docente si perde.

**Causa.** Feature mai realizzata. Presuppone [2.2](#22-persistenza-dei-media): senza persistenza dei media non c'è nulla da mettere in libreria.

**Cosa si cambia.**

Nuova tabella in `[lessons/models.py](../../src/aix/webui/lessons/models.py)`:

```python
class SavedResource(Base):
    __tablename__ = "saved_resource"
    id, owner_id, lesson_id (nullable)
    kind: str          # video | paper | oer | web | tool
    category: str      # scelta dal docente, libera
    title, url, source: str
    provenance: str    # "curated" | "auto"  ← EU AI Act
    payload_json: dict
    note: Optional[str]
    created_at
```

Nuovo `src/aix/webui/resources/routes.py` con `GET/POST/DELETE /webui/resources`, nuovi template `pages/resource_library.html` e `partials/resource_card.html`, voce nella `[navbar.html](../../src/aix/webui/templates/partials/navbar.html)`.

> **Tutto modellato sul pattern già collaudato di** `SavedProfile` (`[lessons/routes.py:2313-2382](../../src/aix/webui/lessons/routes.py#L2313-L2382)`): stessa struttura di tabella, stesse tre rotte, stesso partial htmx, stessa gestione owner. **Zero pattern nuovi da inventare.**

In `media_panel.html` ogni item guadagna un pulsante segnalibro con `hx-post="/webui/resources"` e swap OOB del solo item (l'icona passa a "salvato"). La categoria si assegna dalla libreria, con una `datalist` alimentata dalle categorie già usate dall'utente — non un'enum rigida, che invecchierebbe subito.

**Il badge di provenienza viaggia con la risorsa salvata.** È un requisito di conformità, non un dettaglio estetico: la distinzione `✓ Verificato` (database curato, rivisto da esperti) vs `auto` (recuperato da sistemi di IA, da verificare) è già impostata in `[media_panel.html:283-296](../../src/aix/webui/templates/partials/media_panel.html#L283-L296)` e discende da `[Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md](Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md)`. Una risorsa che perde la provenienza nel passaggio in libreria è una regressione di conformità.

**Verifica in locale.**

```powershell
uvicorn aix.api.main:app --reload --port 8000
```

1. Generare una lezione, salvare 3 risorse di tipo diverso (un video curato, un articolo auto, un link web).
2. Aprire `/webui/resources`.

*Cosa deve succedere:* le 3 risorse compaiono con il badge di provenienza corretto; assegnando una categoria e ricaricando, il filtro per categoria funziona; la risorsa resta accessibile anche dopo aver **cancellato la lezione** di origine (`lesson_id` è nullable con `ON DELETE SET NULL`).
*Cosa deve smettere di succedere:* nessuna risorsa deve perdere il badge `auto` diventando indistinguibile da una verificata.

```powershell
pytest tests/unit/test_saved_resources.py -v
```

**Rischi e flag.** Nessun flag. Rischio principale: divergenza tra il badge mostrato in lezione e quello in libreria — da coprire con un test esplicito sulla propagazione di `provenance`.

---



## 7. Fase 3 — Documento strutturato a blocchi

*60-84h · cambia il contratto del Writer; è la fondazione di rigenerazione mirata, tracciabilità ed export ricchi*

> **Piazzamento metriche.** I blocchi (`objectives`, `activity`, …) abilitano eval e raffina *per sezione* nella strategia AI (grounding/generation §5.3). Finché la lezione è un unico Markdown, le metriche restano documento-livello e il Critic non può isolare il pezzo rotto.



### 3.1 Modello `LessonDocument`

*Risolve D6*

**Sintomo.** La risposta arriva come blocco unico. Non è possibile intervenire su una sezione senza rigenerare tutto.

**Causa.** [D6](#d6--la-lezione-è-una-stringa-markdown-senza-struttura).

**Cosa si cambia.**

Nuovo `src/aix/api/schemas/lesson_document.py`:

```python
class Block(BaseModel):
    block_id: str          # stabile: "obj-1", "act-2", "assess-1"
    kind: Literal["objectives","materials","intro","activity",
                  "assessment","differentiation","sources"]
    title: str
    body_md: str

class ActivityBlock(Block):
    minutes: int
    tool_ids: list[str]      # validati contro il Tool Registry (1.2)
    kg_node_ids: list[str]   # solo nodi effettivamente recuperati
    resource_ids: list[str]  # solo risorse del pool / live
    grouping: Literal["individual","pairs","small_group","whole_class"]

class LessonDocument(BaseModel):
    title: str
    duration_min: int
    blocks: list[Block]
```

Validatori Pydantic che rifiutano `tool_ids` fuori registry e `kg_node_ids` non presenti tra quelli recuperati → errore di validazione → **retry mirato con il messaggio d'errore in contesto**. Rimuovere la dataclass morta `LessonPlanStructure` (`[state.py:294](../../src/aix/agent/graph/state.py#L294)`) e popolare finalmente `lesson_plan_structured` (`[state.py:139](../../src/aix/agent/graph/state.py#L139)`).

**Nota su OpenRouter — structured output.**
`[config.py:73 build_completion_kwargs](../../src/aix/core/config.py#L73)` emette oggi solo:

```python
# config.py:122
kwargs["response_format"] = {"type": "json_object"}
```

JSON mode semplice, mai `json_schema`. Da aggiungere:

- `response_format: {"type": "json_schema", ...}` con lo schema di `LessonDocument`;
- `provider: {"require_parameters": true}` per **escludere i provider che ignorano lo schema** — il supporto non è uniforme su OpenRouter;
- una lista `models: [...]` di fallback;
- `usage: {"include": true}` per ottenere il **costo reale per lezione**, dato che servirà per le stime di effort sui bandi.

Catena di fallback se un modello non supporta lo schema: tool-calling → parser `_extract_json` già induriti (`[planner_agent.py:24](../../src/aix/agent/agents/planner_agent.py#L24)`, coperti da `tests/unit/test_json_parse_hardening.py`).

**Verifica in locale.**

```powershell
pytest tests/unit/test_lesson_document.py -v          # validatori, offline
pytest tests/integration/test_writer_structured.py -v # contratto reale
```

Verificare nei log che `usage` riporti i token e il costo. Controllare che un `tool_id` inventato provochi un retry e non un salvataggio.

**Rischi e flag.** `AIX_WRITER_STRUCTURED=true|false` per poter tornare al Markdown libero. Rischio reale: alcuni modelli degradano in qualità con vincoli di schema stretti — da misurare con il protocollo già definito in `[Protocollo_Valutazione_Comparativa_Modelli_LLM.md](Protocollo_Valutazione_Comparativa_Modelli_LLM.md)`.

---



### 3.2 Persistenza dei blocchi

**Cosa si cambia.** Tabella `lesson_block` (`lesson_id, block_id, kind, ordinal, payload_json, updated_at`), stesso pattern CQRS di `LessonMessage` — documentato in `[lessons/models.py:43-51](../../src/aix/webui/lessons/models.py#L43-L51)`.

`lesson_plan_md` **resta**, come proiezione denormalizzata rigenerata dai blocchi a ogni scrittura. Così export, stampa, `card-fragment` e tutto ciò che oggi legge la stringa continuano a funzionare **senza alcuna modifica**. È la scelta che rende questa fase incrementale invece che una riscrittura.

**Verifica in locale.** Dopo una generazione, controllare che `lesson_block` contenga N righe e che `lesson_plan_md` ricomposto sia equivalente a quello che il Writer avrebbe prodotto. Le lezioni preesistenti (senza blocchi) devono continuare a renderizzare dal solo `lesson_plan_md`.

---



### 3.3 Rigenerazione per blocco

**Sintomo.** "Raffina" rigenera 6.770 caratteri per cambiare un'attività, distruggendo le parti che il docente aveva già approvato.

**Causa.** `[lessons/routes.py:1197-1210](../../src/aix/webui/lessons/routes.py#L1197-L1210)` mappa il codice di raffinamento e lo deposita in `profile["__refinement__"]`, da cui il service lo passa al Writer come istruzione sull'intero documento.

**Cosa si cambia.** Nuovo `POST /webui/lesson/{id}/block/{block_id}/refine`: esegue **solo il Writer sul singolo blocco**, con il resto del documento come contesto in sola lettura, poi rivalida con il ConstraintValidator (1.3).

> **Eseguire prima [11.3](#113-time-travel-per-la-rigenerazione).** Con il time-travel di LangGraph la rigenerazione riprende dal checkpoint successivo al retrieve — il contesto recuperato è già nello stato, quindi non serve ritoccare Neo4j né ricostruirlo a mano. Senza, questo intervento deve reimplementare la ricostruzione del contesto e costa di più.

Sostituisce il rimbalzo su `/run`, **mantenendo i 5 chip di raffinamento già presenti** in `[chat_lesson_card.html](../../src/aix/webui/templates/partials/chat_lesson_card.html)` (`simplify`, `deepen`, `more_activities`, `adapt_class`, `custom`). Ogni blocco guadagna un pulsante di rigenerazione locale e resta modificabile a mano via textarea inline con swap htmx.

**Verifica in locale.** Generare una lezione, salvare l'output, rigenerare **un solo blocco attività**, salvare di nuovo e confrontare:

```powershell
Compare-Object (Get-Content before.md) (Get-Content after.md)
```

*Cosa deve succedere:* le differenze devono ricadere **esclusivamente** nel blocco richiesto.
*Cosa deve smettere di succedere:* nessun altro blocco deve cambiare, nemmeno nella punteggiatura. Il costo in token deve calare di circa un ordine di grandezza (leggibile da `usage`).

---



### 3.4 Chip di tracciabilità e onestà dei segnali

**Sintomo.** Le metriche non dicono nulla a un docente, e due scale di confidenza diverse convivono nella stessa schermata.

**Causa.** Tre problemi distinti:

- `[chat_retriever_card.html:111-128](../../src/aix/webui/templates/partials/chat_retriever_card.html#L111-L128)` mostra una griglia di quattro numeri — *nodi KG*, *relazioni*, *strategie*, *media* — che sono semplici `len()` di liste. "68 strategie" mentre in lezione se ne vedono sei sembra un'esagerazione.
- Due scale di confidenza compresenti: `75% confidence` del Planner (`[chat_planner_card.html:57-59](../../src/aix/webui/templates/partials/chat_planner_card.html#L57-L59)`) e `VERY_HIGH confidence` del Retriever (`[chat_retriever_card.html:107](../../src/aix/webui/templates/partials/chat_retriever_card.html#L107)`, scala definita in `[retriever_agent.py:335-343](../../src/aix/agent/agents/retriever_agent.py#L335-L343)`).
- L'avvertenza "il contenuto disciplinare non è verificato dal KG" è in fondo alla pagina, dove non la legge nessuno.

**Cosa si cambia.**

**① Chip di tracciabilità.** Ogni `ActivityBlock` rende un chip cliccabile **attività → checkpoint UDL → fonte**, alimentato da `kg_node_ids` e `resource_ids`.

> È il differenziale reale del GraphRAG: nessun assistente generalista può dire *da dove viene* una strategia. Chiude anche il punto 6 della lista (riferimenti embedded nella risposta) per la via corretta — dai dati, non dalla penna del modello.

**② Metriche sostituite.** Al posto della griglia di quattro numeri: **"N strategie effettivamente usate in questa lezione"**, cliccabile verso i blocchi che le impiegano. I conteggi grezzi restano disponibili nel dettaglio espandibile, per chi li vuole.

**③ Confidenza unificata.** Una sola scala per Planner e Retriever — la scala ordinale a 5 livelli è la più onesta, perché una percentuale suggerisce una precisione che non c'è — e mostrare la soglia sotto la quale il sistema si ferma invece di procedere.

**④ Avviso inline.** Badge accanto ai blocchi disciplinari, distinto dal verde dei contenuti KG-verified. **Due colori, un significato: verde = tracciato nel grafo, ambra = da verificare.** I token per success e warning esistono già in `[aix-brand.css](../../src/aix/webui/static/css/aix-brand.css)` (`--aix-success`, `--aix-warning`), stessi già usati per `✓ Verificato` / `auto` nel pannello media: **la coerenza cromatica tra pannello risorse e corpo della lezione è parte del messaggio.**

**Verifica in locale.** Con il replay della Fase 0, aprire una lezione: cliccare un chip deve portare al checkpoint UDL corrispondente e alla fonte. Verificare che in tutta la schermata compaia **una sola** convenzione di confidenza, e che i badge verde/ambra siano coerenti tra lezione e pannello risorse.

---



### 3.5 Export `.docx` e guida di conduzione

**Sintomo.** Manca l'export che i docenti usano davvero.

**Causa.** Esistono solo `md` e `txt` (`[lessons/routes.py:2007-2070](../../src/aix/webui/lessons/routes.py#L2007-L2070)`); il pulsante etichettato "PDF" apre in realtà una pagina di stampa e delega a `Ctrl+P` del browser. Nessuna dipendenza `python-docx` nel progetto.

**Cosa si cambia.** Con i blocchi strutturati diventano possibili due artefatti:

- `.docx` via `python-docx`, con stili reali (heading, elenchi, tabella dei tempi) — ~200 righe di renderer;
- **guida di conduzione stampabile**: la scaletta minutata da tenere in mano durante le due ore. È un artefatto **diverso** dal piano di lezione e ricavabile **solo** dai blocchi con `minutes` — impossibile da produrre dal Markdown libero attuale.

**Verifica in locale.** Scaricare entrambi e aprirli in Word/LibreOffice: heading corretti, nessun markdown grezzo residuo, tempi coerenti con quelli in lezione. Verificare che la marcatura EU AI Act (`AI_GENERATED_HEADER` da `[ai_marking.py](../../src/aix/core/ai_marking.py)`) sia presente anche nel `.docx`.

---



## 8. Fase 4 — UX, trace, multimodalità

*68-110h*

### 4.1 Profilo versionato + UX di salvataggio

*Punto 5 IMPORTANT · 14-20h*

**Sintomo.** La pagina di creazione del profilo non dà l'impressione che venga salvato tutto: sembra che si conservino solo durata, materia e titolo — che sono invece proprio i campi che cambiano da una lezione all'altra. Il profilo **viene effettivamente salvato** e si ritrova nella home, ma la UI non lo comunica.

**Causa.** Il problema è di **modello dati**, non di posizione del bottone. `SavedProfile` (`[lessons/models.py:160](../../src/aix/webui/lessons/models.py#L160)`) conserva un blob `profile_json` indistinto: dati stabili di classe e aula mescolati a dati effimeri di lezione. L'expander "Salva come profilo" sta a metà della pagina di creazione, prima delle sezioni classe e ambiente, e quella posizione comunica — correttamente rispetto al modello dati, ma sbagliata rispetto all'intenzione — che si stia salvando ciò che sta sopra.

**Cosa si cambia.**

Separare nettamente i due concetti:

```python
class LessonRequest(BaseModel):        # effimero, cambia ogni volta
    subject: str; topic: str
    duration_min: int; objectives: list[str]

class EducationalProfile(BaseModel):   # persistente, versionato
    id: UUID; name: str; version: int
    group: EducationalGroup            # studenti, BES, caratteristiche
    classroom: ClassroomEnvironment    # LIM, WiFi, BYOD, mobilità
```

Aggiungere `version: int` a `SavedProfile`. La lezione salva `profile_id + profile_version`: **la provenance resta immutabile**, ma si offre un'azione esplicita *"aggiorna al profilo corrente"* quando la versione è cambiata.

In UI:

- selettore in testa alla pagina: `Profilo: 3A liceo scientifico ⌄ v3`;
- **"Salva come nuovo profilo" spostato in fondo**, dopo le sezioni classe e ambiente;
- **badge persistente** `salvato · 2 minuti fa`.

> Il salvataggio già funziona: il problema è che non è *visibile*, e in UX una cosa invisibile non esiste.

Nello stesso passaggio, due incoerenze minori ma corrosive per la fiducia:

- *"Nessun obiettivo definito → aggiungi"*: il sistema chiede l'intento pedagogico e poi genera comunque. **O è obbligatorio, o non lo si chiede.**
- Stato "bozza" in alto e "Lezione finalizzata" in basso nella stessa schermata.

**Verifica in locale.** Creare un profilo, generare una lezione, modificare il profilo (v2), riaprire la vecchia lezione: deve mostrare **v1** con l'offerta di aggiornamento, non silenziosamente v2. Creare una seconda lezione riusando il profilo senza reinserire nulla di classe e aula.

**Piazzamento metriche / pilot.** Nello stesso passaggio UX va inserito il feedback docente minimo della scorecard pilot (`Utile` / `Non utile` + motivo opzionale), documentato in `AI_Evaluation_Metrics_Strategy.md` §§5.7, 16–17. È indipendente dal versioning del profilo ma vive nella stessa superficie "fiducia del docente". Il gate di chi può lasciare quel feedback in staging/prod è [§12 Approccio B](#12-gate-pilot--autenticazione-approccio-b).

---



### 4.2 Trace collassabile

*6-10h*

**Sintomo.** Due schermate di pipeline prima di arrivare alla lezione. Chi usa lo strumento ogni giorno vuole il prodotto, non il processo.

**Cosa si cambia.** Durante la generazione il trace è **perfetto**: riempie l'attesa e costruisce fiducia — non va toccato. **A generazione conclusa deve collassare in una striscia sola** (`Come è stata costruita ⌄`), con la lezione in cima.

Si implementa nell'handler `sse-close` già presente in `[chat_pane.html:86-128](../../src/aix/webui/templates/partials/chat_pane.html#L86-L128)`, che al termine dello stream già ricarica `card-fragment` e `chat-input-fragment` via `htmx.ajax`: basta collassare `#chat-cards` in un `<details>` e portare la lesson card in testa. **Nessun meccanismo nuovo.**

**Verifica in locale.** Con il replay della Fase 0: durante lo stream le card sono espanse e leggibili; a stream concluso il trace è una riga sola e la lezione è la prima cosa visibile. Riaprendo la lezione, il trace deve partire già collassato.

---



### 4.3 VisualIR + renderer deterministici

*Punto 7 IMPROVEMENT — la mossa strategica · 48-80h*

**Sintomo.** Le lezioni sono solo testo. Mappe concettuali, linee del tempo e attività digitali vanno costruite a mano dal docente.

**Cosa si cambia.**

L'errore da evitare è **far generare all'LLM il JSON di Excalidraw**: è verboso, pieno di coordinate, `seed`, `versionNonce`, e si rompe. La strada corretta è **una rappresentazione intermedia semantica, molti renderer deterministici**:

```
Writer LLM → VisualIR (Pydantic) ─┬→ to_excalidraw.py → .excalidraw
                                  ├→ to_mermaid.py    → timeline / mindmap
                                  ├→ to_h5p.py        → .h5p
                                  └→ to_geojson.py    → mappa uMap / Leaflet
```

`VisualIR` contiene nodi, archi, gruppi, etichette e un `layout_hint`. **Il layout si calcola in Python** con `networkx` + `grandalf` (o graphviz), poi si emettono le coordinate.

> Il modello fa la parte semantica, in cui è bravo; Python fa la geometria, in cui l'LLM è pessimo. Testabile con golden file in pytest: **zero token per verificare una regressione.**

Struttura come package separato `src/aix/renderers/` con test golden-file, così i renderer sono verificabili offline. Punto di innesto: `mermaid_generator.py` e `diagram_factory.py` esistono già in `[src/aix/agent/media/](../../src/aix/agent/media/)`, oggi esposti solo via MCP (`[media_tools.py:418](../../src/aix/mcp/tools/media_tools.py#L418)`).

**Ordine dei target, per valore crescente:**

1. **Mermaid** — già in repo, costo quasi nullo. Timeline e mappe concettuali.
2. **Excalidraw** — formato aperto e documentato, editabile subito dal docente, connettore già disponibile.
3. **H5P** — **il target che vale di più per una scuola.** È lo standard OSS per contenuti interattivi: si auto-ospita, entra in Moodle, è GDPR-friendly. Generare un `.h5p` significa zippare `h5p.json` + `content/content.json` con uno schema fisso; Question Set, Dialog Cards, Interactive Video e Course Presentation sono perfettamente alla portata.
  > Con H5P si sostituiscono **Kahoot, Genially ed Edpuzzle** — tre servizi chiusi che chiedono account agli studenti — con artefatti che la scuola possiede. Chiude anche, per la via giusta, il vincolo BYOD della Fase 1: un contenuto H5P proiettato sulla LIM non richiede dispositivi personali.
4. **TimelineJS** (Knight Lab), alimentato da JSON, se serve una timeline più ricca di quella di Mermaid.

**Verifica in locale.**

```powershell
pytest tests/renderers/ -v    # golden file, nessuna chiamata LLM
```

Poi manualmente: aprire il `.excalidraw` prodotto su excalidraw.com e verificare che sia editabile; caricare il `.h5p` su un'istanza Moodle o sul player H5P standalone e verificare che l'attività funzioni.

**Rischi e flag.** Rischio principale: sottostima della complessità di H5P (schemi diversi per ogni tipo di contenuto). Mitigazione: partire da **un solo** tipo (Question Set) e validare l'intera catena prima di aggiungerne altri.

> Il messaggio commerciale diventa netto: **AixLearning non ti dà una lista di link, ti dà i materiali già montati e modificabili.**

---



## 9. Effort e sequenza


| Fase  | Contenuto                                                           | Effort       | Giorni          |
| ----- | ------------------------------------------------------------------- | ------------ | --------------- |
| **P** | Gate pilot Approccio B (`@fem.digital` + OTP) — *doc a parte*       | 6–10 gg      | parallelo a 0–1 |
| **M** | Backbone metriche (telemetria + golden v1 + runner) — *doc a parte* | 2–4 sett     | parallelo a 0–2 |
| 0     | Harness record/replay                                               | 8-12h        | 1-2             |
| 1     | Profilo completo + Tool Registry + ConstraintValidator              | 52-70h       | 7-9             |
| 2     | Filtro web AUTO + persistenza media + libreria risorse              | 46-66h       | 6-9             |
| 3     | Documento a blocchi + rigenerazione mirata + tracciabilità + export | 60-84h       | 8-11            |
| 4     | Profilo versionato + trace collassabile + VisualIR/H5P              | 68-110h      | 9-14            |
| 11    | Uso di LangGraph: progresso reale, interrupt, time-travel, retry    | 28-42h       | 4-6             |
|       | **Totale prodotto (0–4 + §11)**                                     | **262-384h** | **35-51**       |


La [sezione 11](#11-uso-di-langgraph) è numerata a parte perché è **trasversale**: 11.1 si accompagna alla Fase 0, 11.3 va eseguito *prima* di 3.3 e ne riduce l'effort. Le righe **P** e **M** non sono effort di questo piano: dettaglio in [§12](#12-gate-pilot--autenticazione-approccio-b) e [§13](#13-integrazione-metriche-ai-e-criteri-di-successo).

### Dettaglio Fasi 1 e 2


| #   | Intervento                                           | Effort |
| --- | ---------------------------------------------------- | ------ |
| 1.1 | Render completo del profilo + etichette canoniche    | 6-10h  |
| 1.2 | Tool Registry                                        | 14-20h |
| 1.3 | ConstraintValidator deterministico                   | 24-32h |
| 1.4 | Retry mirato                                         | 4-6h   |
| 2.1 | Allowlist + rerank semantico per la ricerca web AUTO | 16-24h |
| 2.2 | Persistenza dei media                                | 10-14h |
| 2.3 | Libreria "Le mie risorse"                            | 20-28h |




### Dipendenze

```
P (Approccio B) ──────────────────────► onboarding docenti su staging/prod
M (telemetria + golden) ──► misura regressioni su 1.x / 2.x / 3.x

0 ──────────────────────────────────► (abilita test rapidi su 3 e 4)
                                    └──► cattura fixture per golden set (M)

1.1 ──► 1.3 ──► 1.4
   └──► 1.2 ──┘         1.2 ──► 3.1 (validazione tool_ids)
                        1.2 ──► 4.3 (alternative OSS)
                        1.3 ──► metriche fattibilità (M)

2.1 (indipendente) ──► KPI retrieval/media (M)
2.2 ──► 2.3
2.2 ──► meta_json completo (prerequisito telemetria online M)

3.1 ──► 3.2 ──► 3.3 ──► 3.5
   └──► 3.4              1.3 ──► 3.3 (rivalidazione per blocco)
                         3.x ──► eval per-sezione (M)

4.1, 4.2 (indipendenti)   3.1 ──► 4.3 (VisualIR nel contratto del Writer)
4.1 ──► feedback Utile/Non utile (scorecard pilot M)

§10 checkpointer ──► 11.2
                 └─► 11.3 ──► 3.3   (riduce l'effort di 3.3)
11.1 (indipendente) ──► arricchisce i trace registrati in Fase 0
11.4 (indipendente)
```



### Sequenza raccomandata

Tre binari in parallelo, poi il percorso prodotto:

1. **Gate P — Approccio B** in parallelo a tutto: senza di esso non si invitano docenti FEM su un ambiente condiviso. Dettaglio [§12](#12-gate-pilot--autenticazione-approccio-b).
2. **Fase 0 + telemetria minima (M Phase 1)** in parallelo a 1.1: harness per velocità di sviluppo; `meta_json` ricco per misurare.
3. **Punto di partenza prodotto: 1.1.** Sei-dieci ore, nessuna dipendenza da Neo4j né da LLM per i test, e da solo sistema buona parte delle lezioni irrealizzabili.

Poi **2.1** in parallelo se ci sono due persone: è indipendente da tutta la Fase 1 e chiude il BLOCKING che ha originato questo documento.

**Solo dopo 1.1 (+ idealmente 1.3 e 2.1) e Gate P verdi:** onboarding docenti e raccolta feedback umano della scorecard.

**Le Fasi 1 e 2 insieme (98-136h, 13-18 giorni) sono ciò che, se un valutatore di bando apre lo strumento, fa la differenza tra "generatore di testo carino" e "sistema che ragiona sui vincoli reali di un'aula".** Le Fasi 3 e 4 costruiscono il differenziale, ma vanno posate su fondamenta sane. Gate auth e metriche non sostituiscono quel lavoro: lo rendono pilotabile e dimostrabile.

### Riuso verso il dominio GBL

`[GBL_Domain_Effort_Estimate.md](GBL_Domain_Effort_Estimate.md)` stima 134-206h per la v1 del dominio Game-Based Learning. Due componenti di questo piano sono **riusabili al 100%** da quel lavoro:

- il **Tool Registry (1.2)** — il Game Catalog è lo stesso pattern, con `game_catalog` su Postgres e pagina admin;
- i **renderer (4.3)** — il Game Creator Assistant produce Activity Template più JSON machine-readable, esattamente l'architettura VisualIR → molti target.

Vale la pena tenerne conto nella sequenza: fare 1.2 prima dell'avvio GBL riduce l'effort di quest'ultimo.

---



## 10. Igiene dati

Debito da chiudere comunque, indipendente dalle fasi.

**Refuso nel Knowledge Graph.** Esiste un nodo etichettato `sustaints effort and persistence` in Neo4j. Le label finiscono **nei prompt e nella UI**, quindi il refuso è visibile al docente. Correggere alla fonte, riallineare `data/kg/udl/kg_udl_neo4j.json` e reingerire:

```powershell
python -m scripts.ingest.data_ingestion_neo4j --file data/kg/udl/kg_udl_neo4j.json --domain udl
python -m scripts.audit.audit_domain_graph --domain udl
```

**Chiave YouTube Data API in chiaro** nel `.env` locale. Il file è gitignored e non risulta committato, ma se è mai stata condivisa va ruotata.

**Debito di retrieval già tracciato** in `[docs/reports/NEXT_SESSION.md](../reports/NEXT_SESSION.md)`: `relevance_score` sempre `0.5`, `educational_context_type` sempre `"assessment"`, nodi Challenge che filtrano nei risultati. Incidono sulla qualità del retrieval e quindi a cascata su tutto il resto: vanno affrontati prima della Fase 3, che sul retrieval costruisce la tracciabilità.

`QUICKSTART.md` **è obsoleto.** Rimanda a Streamlit, che è l'app legacy. La [sezione 2](#2-ambiente-di-test-locale) di questo documento la rimpiazza di fatto per il percorso WebUI: aggiungere un rimando esplicito in testa al runbook.

`langgraph-checkpoint-postgres` **non risulta installato** nell'ambiente di sviluppo locale, benché `requirements.txt` lo elenchi (`>=2.0,<3.0`). `get_checkpointer()` degrada silenziosamente restituendo `None`, il grafo compila senza checkpointer e **la memoria multi-turn resta disattivata**. È un reperto rilevante perché `[run_local.py](../../scripts/ops/run_local.py)` esiste proprio per far funzionare quel checkpointer su Windows — e in quell'ambiente non può funzionare comunque, per un motivo diverso e più banale. Inoltre è un **prerequisito bloccante** per gli interventi [11.2](#112-interrupt-per-il-flusso-a-due-fasi) e [11.3](#113-time-travel-per-la-rigenerazione), che senza checkpointer non sono realizzabili.

```powershell
pip install "langgraph-checkpoint-postgres>=2.0,<3.0"
python -c "import langgraph.checkpoint.postgres.aio as m; print('OK', m.AsyncPostgresSaver.__name__)"
```

Verificare poi nei log di avvio la riga `[checkpointer] AsyncPostgresSaver ready … multi-turn enabled`. Se compare invece la variante "degraded", la memoria multi-turn è spenta.

---



## 11. Uso di LangGraph

*28-42h · risolve D7*

Questa sezione risponde a una domanda specifica: **stiamo usando a pieno le funzionalità di LangGraph utili al nostro scopo?** La risposta è no — ne usiamo circa un quinto, e tre funzionalità native sono riscritte a mano. Non tutto ciò che la libreria offre serve a questo prodotto: la domanda giusta è **quali funzionalità ci risparmiano codice che stiamo già mantenendo**, e sono quattro.

### 11.1 Progresso reale al posto dei segnaposto

*10-14h · risolve D7①  e D7②*

**Sintomo.** Durante la generazione il docente vede tre card con uno spinner e un'etichetta fissa. Nella fase più lunga — il retriever — non succede visivamente nulla per 10-20 secondi. Il trace *sembra* mostrare cosa accade, ma per buona parte del tempo mostra un'attesa.

**Causa.** [D7①](#d7--langgraph-è-usato-al-minimo-e-tre-sue-funzionalità-sono-riscritte-a-mano) e [D7②](#d7--langgraph-è-usato-al-minimo-e-tre-sue-funzionalità-sono-riscritte-a-mano). `stream_mode="updates"` (`[service.py:1419](../../src/aix/webui/agent/service.py#L1419)`) emette solo a nodo concluso: **è un limite strutturale della modalità, non una svista.**

**Cosa si cambia.**

`astream` accetta `stream_mode` come **lista**; con una lista i chunk arrivano come tuple `(mode, chunk)`. Quindi la modalità `custom` si **affianca** a `updates` senza sostituirla:

```python
async for mode, chunk in graph.astream(
    initial_state,
    config=run_config,
    stream_mode=["updates", "custom"],
):
    if mode == "updates":
        ...   # esattamente la logica di oggi, invariata
    elif mode == "custom":
        yield StreamEvent(kind="progress", payload=chunk)
```

Dentro i nodi si emette progresso reale con `get_stream_writer()` (da `langgraph.config`):

```python
# nodes.py — dentro retrieve_node
writer = get_stream_writer()
writer({"phase": "cypher", "msg": "Interrogo il knowledge graph…"})
...
writer({"phase": "nodes", "msg": f"{len(nodes)} nodi recuperati", "count": len(nodes)})
writer({"phase": "media", "msg": "Cerco risorse per 3 concetti chiave"})
writer({"phase": "web",  "msg": f"Ricerca web: {kept} link tenuti, {dropped} scartati"})
```

Punti di emissione consigliati, scelti perché corrispondono a passaggi che **già producono un log**: invio della query Text2Cypher, conteggio nodi recuperati, avvio del media lookup, esito del filtro web (i conteggi esistono già in `[retriever_agent.py:643-648](../../src/aix/agent/agents/retriever_agent.py#L643-L648)`), avvio della scrittura, avvio della valutazione.

**Nello stesso intervento si elimina** `[write_stream.py](../../src/aix/agent/graph/write_stream.py)`**:** i token del writer diventano eventi `custom` sullo stesso canale, e il secondo `EventSource` su `/writer-stream` più il MutationObserver in `_base.html` spariscono. Un solo canale SSE, e **il typewriter torna a funzionare in multi-worker**.

> ⚠️ **Correzione a un piano registrato nel codice.** `[service.py:48](../../src/aix/webui/agent/service.py#L48)` prevede di *"switch the writer agent to* `stream_mode="messages"`*"*. **Non funzionerebbe:** la modalità `messages` intercetta i callback LangChain, ma il writer chiama il client OpenAI grezzo (`client.chat.completions.create(stream=True)`, `[writer_agent.py:115-117](../../src/aix/agent/agents/writer_agent.py#L115-L117)`); `ChatOpenAI` è usato solo in `llm_chain.py` e `text2cypher.py`. Adottare `messages` richiederebbe di riscrivere il writer su LangChain. `custom` **funziona con qualunque SDK** ed è il rimpiazzo drop-in.

**Verifica in locale.**

```powershell
uvicorn aix.api.main:app --reload --port 8000
```

Generare una lezione e osservare la card del retriever.

*Cosa deve succedere:* la card si aggiorna almeno 4-5 volte durante l'attesa, con testo che riflette operazioni reali e conteggi veri, verificabili incrociando i log del server.
*Cosa deve smettere di succedere:* nessuna fase deve restare ferma su uno spinner per più di 3-4 secondi; nessuna etichetta deve essere generica ("Elaborazione in corso…").

Verifica multi-worker, che è il motivo principale dell'intervento:

```powershell
uvicorn aix.api.main:app --port 8000 --workers 4
```

Generare più lezioni in parallelo: il typewriter del writer deve funzionare **in tutte**. Con `write_stream.py` fallisce in modo intermittente, a seconda del worker su cui atterra la richiesta `/writer-stream`.

**Rischi e flag.** `AIX_STREAM_PROGRESS` (default `true`). Rischio basso: la modalità `custom` si affianca a `updates`, quindi con il flag spento il comportamento è identico a oggi. Attenzione a non emettere eventi troppo fitti — un throttle lato client come quello già presente per i token del writer (150 ms) è sufficiente.

**Sinergia con la Fase 0.** L'harness record/replay registra proprio questi eventi: farli prima significa registrare trace già ricchi, invece di doverli ricatturare dopo.

---



### 11.2 `interrupt()` per il flusso a due fasi

*8-12h · risolve D7③*

**Sintomo.** Alla conferma dell'intento pedagogico la pipeline riparte da capo: **il planner viene eseguito due volte**, e il docente paga due volte l'attesa e i token della prima fase.

**Causa.** [D7③](#d7--langgraph-è-usato-al-minimo-e-tre-sue-funzionalità-sono-riscritte-a-mano). Il flag `__planner_only__` viaggia dentro `educational_profile_json`; `lesson_stream` ha un ramo separato che istanzia il `PlannerAgent` **fuori dal grafo**, emette le card, poi riporta `status` a `draft`. La conferma del chip fa un `POST /run` che riavvia l'intera pipeline.

**Cosa si cambia.**

Un `interrupt()` dopo il nodo `plan`, e una ripresa con `Command`:

```python
# nodes.py — in coda a plan_node, quando serve la conferma dell'intento
if state.get("await_intent_confirmation"):
    scelta = interrupt({"kind": "intent_choice", "options": intent_options})
    return {"pedagogical_intent": scelta}
```

```python
# service.py — alla conferma del chip, sullo STESSO thread_id
async for mode, chunk in graph.astream(
    Command(resume=intent_code),
    config=run_config,
    stream_mode=["updates", "custom"],
):
    ...
```

Il grafo si ferma dopo il planner, lo stato resta nel checkpoint, e alla ripresa **riparte dal punto esatto** senza rieseguire il planner. Spariscono: il flag smuggled in `educational_profile_json`, il ramo di codice duplicato in `lesson_stream`, e il reset di `status` a `draft`.

> ⚠️ **Prerequisito bloccante:** `interrupt()` richiede un checkpointer attivo. Vedi il reperto in [sezione 10](#10-igiene-dati) — oggi in locale il checkpointer degrada silenziosamente a `None`. Senza quel fix, questo intervento non è realizzabile.

**Verifica in locale.**

```powershell
uvicorn aix.api.main:app --reload --port 8000
```

Creare una lezione, arrivare ai chip di intento, sceglierne uno.

*Cosa deve succedere:* nei log deve comparire **una sola** riga `[PlannerAgent]` per l'intero flusso; la ripresa deve essere immediata e la card del planner già presente non deve essere ricostruita.
*Cosa deve smettere di succedere:* nessuna seconda esecuzione del planner; `status` non deve più oscillare `running → draft → running`.

Misura oggettiva del risparmio, leggibile da `usage` una volta attivo il punto [3.1](#31-modello-lessondocument): i token del planner devono dimezzarsi sul flusso completo.

**Rischi e flag.** Rischio medio: è un cambio di flusso di controllo, non additivo. Va tenuto dietro `AIX_INTENT_INTERRUPT` (default `false` in prima battuta) con il percorso attuale come fallback, finché non è validato. Attenzione alle lezioni già in corso al momento del deploy: un `interrupt` pendente su un thread creato dal vecchio flusso non esiste, quindi la ripresa deve degradare al comportamento attuale.

---



### 11.3 Time-travel per la rigenerazione

*6-10h · abilita e semplifica [3.3](#33-rigenerazione-per-blocco)*

**Sintomo.** Rigenerare una parte della lezione significa rieseguire tutto: planner, retriever (Neo4j + media + web), writer, critic.

**Causa.** Il checkpointer è attivo ma `aupdate_state` e `get_state_history` non vengono mai chiamati. La colonna `checkpoint_id` su `LessonMessage` esiste già, riservata a questo scopo e mai popolata (`[lessons/models.py:266-270](../../src/aix/webui/lessons/models.py#L266-L270)`).

**Cosa si cambia.**

1. **Popolare** `checkpoint_id` sull'assistant turn, leggendolo dal risultato di `astream` — è il pezzo mancante che il commento nel modello già anticipa.
2. Per la **rigenerazione per blocco** ([3.3](#33-rigenerazione-per-blocco)): invece di rieseguire la pipeline, riprendere dal checkpoint **successivo al retrieve** e rieseguire solo il writer sul blocco. Il contesto recuperato è già nello stato: non serve ritoccare Neo4j.
3. Per la **rigenerazione da un turno passato**: `aget_state_history` per elencare i checkpoint del thread e `aupdate_state` per ramificare, che è il caso d'uso canonico del time-travel.

> Questo intervento **riduce** l'effort di [3.3](#33-rigenerazione-per-blocco) invece di aggiungersi: senza time-travel, la rigenerazione per blocco deve ricostruire il contesto a mano.

**Verifica in locale.**

Serve Postgres e il checkpointer funzionante (vedi [sezione 10](#10-igiene-dati)).

```powershell
python scripts/ops/run_local.py
```

Generare una lezione, poi rigenerare un blocco.

*Cosa deve succedere:* nei log della rigenerazione **non** devono comparire chiamate a Neo4j né al planner; solo il writer. La latenza deve scendere in modo netto.
*Cosa deve smettere di succedere:* nessun nuovo recupero di media o link web per una semplice riscrittura.

Ispezione dei checkpoint:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/agent_threads.db'); print(c.execute('SELECT COUNT(*) FROM checkpoints').fetchone())"
```

**Rischi e flag.** Rischio medio-basso, ma la crescita del DB dei checkpoint va monitorata: senza una policy di retention i thread si accumulano. Prevedere uno script di pulizia dei thread più vecchi di N giorni, sul modello degli altri script in `scripts/ops/`.

---



### 11.4 Retry e cache a livello di nodo

*4-6h*

**Sintomo.** Un errore transitorio — un timeout di rete verso OpenRouter, un blip di Neo4j — fa fallire l'**intera** lezione, che finisce in `status="error"` con la card "Riprova". Il docente rifà tutto da capo.

**Causa.** I nodi sono registrati senza policy di ritentativo: `add_node("retrieve", retrieve_node)`. Ogni gestione di errore è a carico del codice applicativo. `[concurrency.py](../../src/aix/core/concurrency.py)` gestisce concorrenza e rate limiting, **non** il retry di un nodo fallito.

**Cosa si cambia.**

`add_node` accetta `retry_policy` e `cache_policy` (firme verificate su LangGraph 1.1.10):

```python
from langgraph.types import RetryPolicy, CachePolicy

workflow.add_node(
    "retrieve", retrieve_node,
    retry_policy=RetryPolicy(max_attempts=3, backoff_factor=2.0,
                             retry_on=(TimeoutError, ConnectionError)),
)
workflow.add_node(
    "write", write_node,
    retry_policy=RetryPolicy(max_attempts=2, retry_on=(TimeoutError,)),
)
```

`retry_on` va tenuto **stretto**: solo errori di rete e timeout. Un errore di validazione o di parsing JSON non va ritentato alla cieca — per quello esistono i parser già induriti e il retry mirato di [1.4](#14-retry-mirato).

**Cache in sviluppo.** `cache_policy=CachePolicy(ttl=...)` sul nodo `retrieve`, con `compile(cache=...)`: durante l'iterazione sulla UI la stessa query non ricolpisce Neo4j né le API media. È complementare all'harness della [Fase 0](#4-fase-0--harness-recordreplay) — il replay salta l'intero grafo, la cache lo esegue ma senza I/O ripetuto. **Da tenere spenta in produzione**, dove il contenuto live deve restare live.

**Durabilità.** `astream` accetta `durability` (`"sync"` | `"async"` | `"exit"`), che controlla quando il checkpoint viene scritto. Il default va bene; vale la pena conoscerlo se la scrittura dei checkpoint dovesse emergere come costo in produzione.

**Verifica in locale.**

```powershell
pytest tests/unit/test_node_retry_policy.py -v
```

Simulare un `ConnectionError` transitorio nel retriever con un mock che fallisce due volte e poi riesce: la lezione deve completare senza che l'utente veda un errore. Verificare all'opposto che un errore di validazione **non** venga ritentato.

**Rischi e flag.** Rischio basso, ma un `retry_on` troppo largo trasformerebbe un bug deterministico in tre esecuzioni lente prima di fallire lo stesso — da cui la raccomandazione di limitarlo agli errori di rete. La cache va dietro `AIX_NODE_CACHE_ENABLED` (default `false`).

---



### Cosa NON vale la pena adottare

Per completezza, le funzionalità valutate e **scartate**, con il motivo:


| Funzionalità                          | Perché no                                                                                                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Send` (fan-out map-reduce)           | Il parallelismo del retriever è già gestito con `asyncio.gather` dentro il nodo e funziona. `Send` sposterebbe la stessa logica nel grafo senza guadagno. |
| `BaseStore` (memoria a lungo termine) | Il pattern CQRS su `LessonMessage` copre già il caso d'uso, con il vantaggio di essere interrogabile in SQL.                                              |
| Subgraph                              | La pipeline è lineare con un solo loop. Non c'è composizione da fattorizzare.                                                                             |
| `stream_mode="messages"`              | Richiederebbe di riscrivere gli agenti su LangChain. Vedi la correzione in [11.1](#111-progresso-reale-al-posto-dei-segnaposto).                          |




### Effort e dipendenze della sezione 11


| #    | Intervento                              | Effort     | Dipende da                                                            |
| ---- | --------------------------------------- | ---------- | --------------------------------------------------------------------- |
| 11.1 | Progresso reale al posto dei segnaposto | 10-14h     | —                                                                     |
| 11.2 | `interrupt()` per il flusso a due fasi  | 8-12h      | checkpointer funzionante ([§10](#10-igiene-dati))                     |
| 11.3 | Time-travel per la rigenerazione        | 6-10h      | checkpointer funzionante; abilita [3.3](#33-rigenerazione-per-blocco) |
| 11.4 | Retry e cache a livello di nodo         | 4-6h       | —                                                                     |
|      | **Totale**                              | **28-42h** |                                                                       |


**Sequenza raccomandata:** prima il fix del checkpointer (§10, minuti), poi **11.1** — che è indipendente, dà il beneficio più visibile al docente, chiude un bug latente di multi-worker e produce trace più ricchi per l'harness della Fase 0. **11.3** va fatto *prima* di [3.3](#33-rigenerazione-per-blocco), non dopo, perché ne riduce l'effort.

---



## 12. Gate pilot — autenticazione Approccio B

*Non è un fix della pipeline; è il prerequisito di accesso al pilot FEM.*  
*Fonte canonica:* `[Pilot_Registration_Approaches.md](Pilot_Registration_Approaches.md)` (Approccio B + Appendice B).

### Perché sta in questo documento

Questo piano migliora ciò che il docente *vede e riceve* dopo il login. Senza un gate di registrazione, lo staging/prod resta a registrazione aperta: chiunque può creare account e generare lezioni. Per il pilot FEM il gate scelto è l'**Approccio B**:

1. l'utente inserisce l'email;
2. il sistema accetta solo indirizzi sul dominio autorizzato `fem.digital`;
3. invia un OTP via email (Google Workspace SMTP relay);
4. la verifica OTP completa registrazione/login.

Non è un allowlist di singoli indirizzi in v1: è un **domain gate** + OTP.

### Dove si piazza rispetto alle fasi


| Momento                                   | Cosa fare                                             | Dipende da questo piano?                                           |
| ----------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------ |
| **Subito / in parallelo a Fase 0–1**      | Implementare Approccio B su staging                   | No — binario auth indipendente                                     |
| **Prima dell'onboarding docenti**         | Gate P verde + smoke test email `noreply@fem.digital` | Sì: senza P non si invita nessuno                                  |
| **Dopo 1.1 (idealmente anche 1.3 / 2.1)** | Invito docenti FEM e raccolta feedback                | Sì: evita di misurare un sistema ancora "carino ma irrealizzabile" |
| **Fasi 3–4**                              | Nessuna dipendenza auth aggiuntiva                    | —                                                                  |




### Cosa NON fare qui

- Non riscrivere il piano di implementazione auth in questo file (route, modello OTP, anti-enumerazione, env): resta nell'Appendice B di `Pilot_Registration_Approaches.md`.
- Non mischiare Approccio A (invite code) come piano primario del pilot FEM: A resta alternativa documentata, B è la scelta operativa per `@fem.digital`.



### Criterio di pronto-pilot (auth)

- solo `@fem.digital` ottiene OTP;
- OTP corretto → sessione; errato/scaduto/riuso → rifiuto;
- nessun OTP/credenziale in log o risposte API;
- effort indicativo: 6 gg minimo, 8–10 gg raccomandato (config Workspace esclusa).

---



## 13. Integrazione metriche AI e criteri di successo

*Come si misura se le fasi di questo piano funzionano.*  
*Fonte canonica:* `[AI_Evaluation_Metrics_Strategy.md](AI_Evaluation_Metrics_Strategy.md)`.

### Perché serve

Oggi l'unico "successo" di pipeline è spesso il Critic LLM (e a volte un `meta` sintetico al reload). Questo piano introduce controlli deterministici (validator, allowlist, persistenza). Senza metriche indipendenti, non si sa se 1.1/1.3/2.1 hanno davvero ridotto lezioni irrealizzabili o link fuori scope — e non si può dimostrare regressione a un valutatore di bando.

### Mappa fase → metriche / artefatti


| Intervento in questo piano                        | Cosa diventa misurabile                                          | Dove nella strategia metriche                           |
| ------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------- |
| **Fase 0** harness                                | Cassette/trace riusabili; costo basso di cattura casi            | Golden dataset §§6–7; runner offline                    |
| **1.1** profilo completo + label BES              | Meno attività BYOD impossibili; meno reinterpretazioni BES       | Feasibility + pedagogical/UDL §§5.4–5.5                 |
| **1.3** ConstraintValidator                       | `blocking_violations`, pass rate fattibilità, retry count        | Layer operativo/agent §§5.1, 5.4; Critic ≠ ground truth |
| **1.4** retry mirato                              | Revisioni validator vs Critic separate nei meta                  | Agent-behavior metrics                                  |
| **2.1** allowlist + rerank                        | Domain allowlist hit-rate, relevance score, commercial off-scope | Retrieval/media §5.2                                    |
| **2.2** persistenza media + `meta_json`           | KPI online al reload; scorecard pilot alimentata                 | Online eval / persistence MVP §8.2                      |
| **3.x** LessonDocument a blocchi                  | Eval e raffina per sezione; grounding per blocco                 | Generation metrics §5.3; section-level evals            |
| **3.4** chip provenienza                          | Trasparenza fonti / claim support                                | Grounding + compliance §§5.3, 5.6                       |
| **4.1** UX profilo + **feedback Utile/Non utile** | Human metrics e acceptance pilot                                 | §§5.7, 16–17                                            |
| **11.1** progresso reale                          | Trace affidabili per harness e Langfuse                          | Instrumentation §§9, 12                                 |




### Sequenza metriche allineata a questo piano

Allineamento con la roadmap di `AI_Evaluation_Metrics_Strategy.md` §19:

1. **M0–M1 (parallelo a Fase 0 / 2.2):** ownership metriche + telemetria deterministica (`meta_json` ricco, timings, media counts, poi validator outcomes).
2. **M2 (dopo primi fix 1.1 / catture Fase 0):** golden set v1 (~20 casi Neuro/UDL), con slice BYOD/BES/web-noise.
3. **M3:** runner offline framework-neutral + report A/B prima/dopo 1.3 e 2.1.
4. **M4–M5:** Ragas/DeepEval + Langfuse — *dopo* che i fix di fattibilità e retrieval esistono; non prima, altrimenti si ottimizza il giudice sul sintomo sbagliato.
5. **M6:** feedback umano in WebUI — richiede Gate P ([§12](#12-gate-pilot--autenticazione-approccio-b)) e docenti reali.



### Criteri di successo minimi per dichiarare "Fasi 1–2 pronte al pilot"

Soglie indicative (dettaglio e ownership nel doc metriche; da congelare prima del pilot):

- **Fattibilità:** 0 violazioni `blocking` del ConstraintValidator sui casi golden BYOD=NO / durata / gruppi; Critic alto non basta più.
- **Retrieval web:** riduzione netta di risultati commerciali/fuori ordine di scuola sul set di taratura 2.1 (allowlist + relevance).
- **Persistenza:** media e meta presenti dopo F5/reload.
- **Operativo:** nessun silent Critic-parse-error conteggiato come approval; error budget rispettato.
- **Umano (appena Gate P attivo):** rate `Utile` e motivi `Non utile` tracciati nella scorecard settimanale.



### Cosa resta fuori da questo file

Definizioni esaustive di metriche, schema golden record, Ragas/DeepEval/Langfuse, governance LLM-as-judge, privacy eval e CI gates: **non duplicarli qui**. Questo §13 fissa solo *dove* quel lavoro si aggancia alle fasi prodotto.

---



## Riferimenti

- `[docs/product/Pilot_Registration_Approaches.md](Pilot_Registration_Approaches.md)` — gate pilot; **Approccio B** (`@fem.digital` + OTP) per onboarding FEM
- `[docs/product/AI_Evaluation_Metrics_Strategy.md](AI_Evaluation_Metrics_Strategy.md)` — metriche, golden set, scorecard pilot, Ragas/DeepEval/Langfuse
- `[docs/product/Dynamic_Media_Retrieval_Plan.md](Dynamic_Media_Retrieval_Plan.md)` — spec del layer media AUTO
- `[docs/product/Teacher_UX_Pedagogical_Framework_Plan.md](Teacher_UX_Pedagogical_Framework_Plan.md)` — framework UX didattico (F1-F6)
- `[docs/product/Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md](Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md)` — provenienza e trasparenza
- `[docs/product/GBL_Domain_Effort_Estimate.md](GBL_Domain_Effort_Estimate.md)` — dominio GBL, componenti condivise
- `[docs/release/Technical_Documentation.md](../release/Technical_Documentation.md)` — documentazione tecnica canonica (v1.3)
- `[docs/architecture/Frontend_Platform_Evaluation.md](../architecture/Frontend_Platform_Evaluation.md)` — ADR-0001, scelta htmx/WebAwesome
- `[docs/reports/NEXT_SESSION.md](../reports/NEXT_SESSION.md)` — debito di retrieval aperto

