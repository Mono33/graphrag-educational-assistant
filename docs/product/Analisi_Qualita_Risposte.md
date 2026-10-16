# Analisi della qualità delle risposte — il caso «ciclo dell'acqua» (16/10/2026)

Branch: `feature/response-quality-improvement` (da `chore/repo-reorg` @ `615f7af`) ·
Scritto il 16/10/2026

Il caso: la lezione di staging `e6289f76-7d8e-4e3c-a4c0-7bfb3aa125b3` (seconda
primaria, ciclo dell'acqua, 22 alunni, 2 ispanofoni, 2 con ADHD, dominio UDL), le
due risposte veloci che l'hanno seguita, la risposta di un concorrente (askludia) e
il commento della collega esperta UDL («manca l'ambiente»).

Le tre domande di Angelo:

- **A.** Perché nei turni successivi l'assistente dice «Non ho accesso diretto al
  grafo»?
- **B.** Usiamo davvero ciò che il grafo restituisce? Che cosa arriva al Writer,
  quanto, in che forma, che cosa si perde?
- **C.** Come rendere la risposta dinamica e aderente ai dati del grafo, e come
  alzare la qualità (rispetto al concorrente e al punto della collega)?

---

## In breve

- **A.** Il turno veloce (`src/aix/webui/agent/light.py`) legge solo la lezione, il
  profilo, gli ultimi 6 turni e i documenti allegati. Del retrieval **non sopravviveva
  nulla** oltre alla corsa: né nel messaggio salvato (solo i conteggi) né in un posto
  che il turno veloce legga. Il commit `b6cda33` **non c'entra**: la copia che ha tolto
  non la leggeva nessuno, e il turno veloce non legge il checkpoint di LangGraph.
- **B.** Del grafo arrivava al Writer una fetta piccola e sbagliata. Di **85 nodi,
  56 raccomandazioni e 213 relazioni** il Writer ha visto 20 nodi, 10 raccomandazioni
  **senza testo** e 15 relazioni. Erano le prime della lista, e la lista era
  nell'ordine in cui erano partite le ricerche: in testa c'erano **8 nodi del profilo
  classe** («Group till 15», «Cohesive»…) senza descrizione. Le **12 difficoltà ADHD e
  degli alunni non italofoni**, le strategie che il grafo collega loro e le due che
  **sconsiglia** per l'ADHD non sono mai arrivate. La sezione Differenziazione è
  conoscenza del modello, non del grafo. In più, il Writer ha ricevuto come «Wikipedia
  Summary» le voci **Sam Neill** e **Poltergeist di Enfield**.
- **C.** Il grafo UDL contiene già ciò che manca a entrambe le risposte: **nodi
  sull'ambiente** (Colour «Neutral tones: Neutral colours reduce overstimulation,
  aiding focus», Lighting, Acoustics, Furniture) e i **checkpoint UDL** («address
  distractions», «Vary and honor the movement»). Nessuna ricerca li raggiungeva. Con
  le modifiche di questo branch, il Writer riceve una **mappa bisogno → che cosa il
  grafo suggerisce / sconsiglia**, il profilo della classe e l'ambiente. Il turno
  veloce riceve lo stesso materiale. Rigenerata con Haiku 5.5, la lezione nomina
  8 strategie del grafo legate ai bisogni della classe e ha un punto «Ambiente di
  apprendimento» basato su quei nodi. Nel turno veloce l'assistente elenca che cosa
  la lezione usa e che cosa no.

Fatto in questo branch: 10 commit con test, suite verde (1 180 test). Proposto e
non fatto: la sezione «barriere → linee guida UDL» (richiede archi nel grafo che
oggi non ci sono), il critic che controlla l'aderenza al grafo, la pulizia di
text2cypher e dei dati del grafo (vedi §5).

---

## 1. Da dove viene l'evidenza

- **Langfuse** (sessione = id lezione): 3 trace. `lesson.turn` del turno 1 (40,5 s,
  $0,036), `lesson.light` dei turni 2 e 3 (9,7 s e 8,6 s). Ho letto gli input e gli
  output di plan, retrieve, delle 12 chiamate text2cypher, del Writer e del Critic.
- **Neo4j** di produzione (`graph.aiforlearning.digital`), sola lettura: etichette,
  archi, nodi `Context`, ambiente e checkpoint UDL.
- **Codice** a `615f7af`.
- **Rigenerazione** con le modifiche: 2 lezioni e 4 risposte veloci, con Haiku 5.5
  via OpenRouter, a partire dallo stato salvato nel trace (piano, profilo, nodi
  recuperati). Il planner e il retriever non sono stati rieseguiti. In totale 6
  chiamate, **$0,014**.

---

## 2. A — Perché i turni successivi «non hanno accesso al grafo»

**Che cosa legge un turno veloce.** `run_light_turn` (`light.py`) costruisce un
messaggio con: la data, il profilo, la lezione (`lesson.lesson_plan_md`), i passaggi
dei documenti allegati (se c'è il RAG privato), gli ultimi 6 turni e la richiesta.
Il prompt (`src/aix/agent/prompts/light_prompt.py`) diceva di rispondere «dalla
lezione, dai passaggi dei documenti allegati e dalla conversazione». Il Knowledge
Graph non era mai nominato. Il docstring lo dice in modo esplicito: «Not a LangGraph
node: nothing here needs the graph's state».

**I numeri del trace.** Turno 2: un'unica generazione, input di 9 280 caratteri
(≈4 000 token: 232 nuovi + 3 775 scritti in cache), nessuna chiamata a strumenti.
Lo strumento `cerca_nel_grafo` di `light_tools.py` esiste, ma
`AIX_LIGHT_TOOLS_ENABLED` non è impostato, quindi è spento. Anche se fosse acceso,
farebbe una ricerca nuova e restituirebbe i primi 10 nodi, non ciò che era stato
trovato per la lezione.

**Che cosa restava salvato.** Il messaggio del Writer (`LessonMessage.meta_json`,
costruito da `_extract_meta` in `service.py`) conteneva solo i conteggi: `nodes_count`,
`recommendations_count`, `search_queries_count`. I nodi recuperati restano nel
checkpoint di LangGraph (`retrieved_nodes`), ma il turno veloce non lo legge.

**Il commit `b6cda33`** ha messo a `None` `graphrag_results`, una copia per query
dei nodi e delle relazioni che nessuno leggeva. `retrieved_nodes` e
`retrieved_relationships` sono rimasti. Non è la causa.

**Turno 3, «Il documento non contiene marcatori ``».** Il prompt citava letteralmente
il marcatore, `` `[[material:…]]` ``. Senza documenti, il modello lo ha citato alla
docente per spiegare perché non citava nulla. `drop_unknown_markers`
(`src/aix/agent/validation/citation_support.py`) ha tolto il marcatore e ha lasciato
i due backtick.

**Corretto in questo branch** (commit `75b14ab`, `00a2cbe`, `79e8a97`):

- il messaggio del Writer conserva in `meta_json["kg_digest"]` il materiale del grafo
  che il Writer ha ricevuto (`kg_evidence.digest`, al massimo 9 000 caratteri), con il
  numero di nodi e di relazioni;
- il turno veloce lo legge dall'ultimo turno completo e lo mette subito dopo la
  lezione, nella parte del messaggio che resta uguale da un turno all'altro. Così il
  provider lo tiene in cache insieme alla lezione;
- il prompt spiega come rispondere alle domande sul grafo: che cosa la lezione usa e
  che cosa no, e i nomi dei nodi spiegati in italiano. Una lezione scritta prima di
  questa modifica non ha l'elenco, e il prompt lo dice invece di «non ho accesso»;
- il marcatore non è più scritto per esteso nel prompt, e un marcatore scartato si
  porta via i suoi backtick;
- il tetto di token del turno veloce passa da 3 000 a 4 500. Nella rigenerazione, la
  risposta al turno 2 aveva usato tutti i 3 000 token (circa 1 700 di ragionamento) e
  si era interrotta a metà elenco.

**Risultato rigenerato** (stesse domande della docente). Al turno 2 l'assistente
elenca per bisogno (ADHD, non italofoni, profilo classe, ambiente, checkpoint UDL)
che cosa la lezione usa (✅), che cosa usa in parte e che cosa no, e chiude
proponendo due elementi da aggiungere. Al turno 3 risponde «No, le strategie usate
non sono solo sei» ed elenca usate e non usate.

---

## 3. B — Che cosa arriva davvero al Writer

### 3.1 Le ricerche

Il piano ha 8 query in inglese. `retrieve_node` (`src/aix/agent/graph/nodes.py`)
aggiunge i termini del profilo: «CICLO DELL'ACQUA», «scienze», e i codici dei bisogni
**così come sono**, «DLDS» e «ADHD». Le query diventano 12. Ognuna passa da
text2cypher (Gemini 3.1 Flash Lite, con un prompt di schema di circa 22 000 token).
Alcuni esempi presi dal trace:

| Domanda | Cypher generato |
|---|---|
| «UDL strategies for early primary heterogeneous classes» | `MATCH (c:Context {name: "Group till 15"})-[:SUGGESTS]->(m:LearningMethodology)` |
| «cooperative learning with small groups in primary school» | `MATCH (c:Context {name: "Group till  15"})…` |
| «DLDS» | `… WHERE toLower(n.name) CONTAINS "dlds"` (nessun risultato) |
| «scienze», «CICLO DELL'ACQUA» | ricerca di testo nei nodi (nessun risultato: il grafo è di pedagogia) |
| «teaching strategies for ADHD students» | `MATCH (a:Adhd)-[r:SUGGESTS]->(s)` ✔ |

«Group till 15» compare tra i nodi d'esempio del prompt di schema, e il modello lo
ricopia come se fosse la risposta. Per questo i nodi del profilo classe entrano in
due ricerche su tre.

### 3.2 L'unione e l'ordine

`RetrieverAgent.retrieve` (`src/aix/agent/agents/retriever_agent.py`) unisce i
risultati nell'ordine delle query, senza ordinarli. Ogni lettore poi prende la testa
della lista:

- la card prende i primi 5 nodi («Concetti più rilevanti», `_build_retriever_payload`
  in `service.py`);
- il Writer prende i primi 20 nodi e le prime 10 raccomandazioni;
- la ricerca dei media prende i primi 15 nodi;
- il Critic prende 10 + 20.

L'ordine reale degli 85 nodi del trace:

| posizioni | che cosa |
|---|---|
| 0–7 | 8 nodi `Context` (Group till 15/20/30, Cohesive, Split in groups, With disruptive elements, Motivated, Gender gap), **descrizione vuota** |
| 8–25 | metodologie e strategie |
| 26–33, 51–54, 82–84 | **le difficoltà ADHD (7) e degli alunni non italofoni (5)**, più 3 dello spettro autistico |
| 34–50 | le strategie collegate a quelle difficoltà (Bilingual resources, Chunking instructions, Glossaries…) |
| 55–84 | strumenti, valutazione, «Long Frontal Lesson», «Passive Learning» |

La card «56 strategie» conta le raccomandazioni, compresi i nodi del profilo classe.

### 3.3 Che cosa è stato passato (generazione `c826cb12`: 20 637 caratteri, 8 631 token di input, 2 810 di output)

| | recuperati | al Writer | come |
|---|---|---|---|
| nodi | 85 | 20 | `_format_nodes`: `nodes[:20]`, descrizione troncata a 150 caratteri. 8 su 20 sono nodi `Context` senza testo; **0 nodi di bisogno** |
| raccomandazioni | 56 | 10 | `_format_recommendations`: `recommendations[:10]`. 4 su 10 sono nodi `Context`. **Nessuna ha testo**: il formatter legge `description`, ma il context builder scrive `implementation_guidance`. Il Writer ha letto righe come «### Cooperative Learning (Confidence: VERY_HIGH)» e basta |
| relazioni | 213 (con duplicati: 123 SUGGESTS, 44 VECTOR_SIMILAR, 8 NO_SUGGESTS) | 15 | `_format_relationships`: le prime 15. Partono **tutte da nodi `Context`**, e 5 sono `VECTOR_SIMILAR` («Group till 15 —VECTOR_SIMILAR→ Group till 20») |

**Che cosa si è perso**, ed era nel risultato:

- ADHD · *Difficulty sustaining focus* → il grafo suggerisce Multisensory Activities,
  Focus Tasks e Gammification, e **sconsiglia Long Frontal Lesson e Passive Learning**;
- ADHD · *Reduced working memory capacity* → Scaffolding, Chunking;
  *Limited metacognitive monitoring* → Goal–Plan–Do–Review; e altre 4;
- non italofoni · *Limited proficiency…* → Bilingual resources;
  *Struggles to follow fast-paced instruction* → Chunking instructions;
  *Limited academic vocabulary* → Glossaries; e altre 2;
- per una classe di 22 il profilo pertinente è «Group till 30». Il grafo dice:
  «Group till 30 —**NO_SUGGESTS**→ Station Rotation». La lezione ha proposto Station
  Rotation come attività 2: l'aveva presa da «Group till 15».

La lezione nomina 6 strategie del grafo (Station Rotation, Cooperative Learning, Game
Based Learning, Scaffolding, Multisensory Activities, Differentiated Instruction):
sono quelle che stavano in testa alla lista. Le misure per l'ADHD e per gli
ispanofoni (timer, cognati spagnoli…) sono plausibili, ma vengono dal modello, non
dal grafo. Al turno 3 il modello stesso lo ammette.

### 3.4 Gli altri difetti della stessa lezione

- **Wikipedia**: i concetti del piano sono in inglese («water cycle»). La ricerca
  avveniva solo su it.wikipedia e prendeva il primo risultato del full-text qualunque
  fosse. Risultato: «Sam Neill» per *water cycle*, «Poltergeist di Enfield» per
  *evaporation*, «Condensato di Bose-Einstein» per *condensation*. Da qui la frase
  della lezione «Wikipedia non fornisce una voce specifica sul ciclo dell'acqua».
- **Dominio**: il prompt ibrido (argomento in parte fuori dal grafo, il caso più
  comune) diceva «neuroscience-based» quattro volte, e la nota obbligatoria in fondo
  alla lezione «Knowledge Graph di neuroscienze dell'apprendimento», qualunque fosse
  il dominio (`writer_prompt.py`).
- **Gruppi**: il vincolo «cooperative groups must not exceed 5 members»
  (`profile_renderer.py`) lasciava la divisione al modello, che l'ha svolta nel testo:
  «4 gruppi: 3 da 5 e 1 da 7 non ammesso…».
- **Critic** (Gemini 3.5 Flash Lite, non Sonnet): ha approvato con 4/5 su tutto e 0
  revisioni. Vede gli stessi 10 + 20 elementi di testa e non controlla in alcun modo
  l'aderenza al grafo.
- **Durata**: 10 + 35 + 10 = 55 minuti su 60, e il validatore l'ha fatta passare.
  Anche le due lezioni rigenerate arrivano a 55. Il difetto c'era già prima e non
  dipende da queste modifiche.

---

## 4. C — Il confronto, e che cosa contiene il grafo

### 4.1 Le tre risposte

| | nostra (16/10) | concorrente | nostra rigenerata (branch) |
|---|---|---|---|
| Struttura | lezione classica: intro, 3 attività, valutazione | ciclo dell'indagine, termini introdotti dopo che i bambini descrivono il fenomeno | lezione classica, con dimostrazione e gioco di movimento |
| Bisogni → strategie | generiche, dal modello | barriere mappate a «Guideline 2, Consideration 2.1»… (numerazione **non verificabile**: il concorrente non mostra fonti) | per ogni bisogno, la strategia del grafo con il suo nome e la difficoltà che affronta (Focus Tasks, Chunking, Scaffolding, Goal–Plan–Do–Review, Multisensory Activities, Gammification, Peer to Peer, Cooperative Learning; glossario bilingue ed esempi contestualizzati per gli ispanofoni, con i nodi Bilingual resources, Glossaries e Contextualized examples nelle Fonti) |
| Ciò che il grafo sconsiglia | ignorato (Station Rotation proposto) | — | rispettato (niente Station Rotation, niente lezione frontale lunga) |
| Ambiente (il punto della collega) | assente | assente | punto «Ambiente di apprendimento»: pochi stimoli alle pareti, colori neutri e non saturi, luce moderata, banchi vicino alla cattedra e lontano dalla finestra, rumore contenuto. Nelle Fonti: i nodi Colour, Lighting, Acoustics e i checkpoint «address distractions» e «Vary and honor the movement» |
| Assunzioni da verificare, opzioni di valutazione, domanda finale | no | sì | no (vedi §5, proposte 3 e 4) |
| Fonti disciplinari | Wikipedia sbagliata | — | Wikipedia «Water cycle», «Evaporation» |
| Nota sul dominio | «neuroscienze» | — | «Knowledge Graph UDL» |
| Turni successivi | «non ho accesso al grafo» | — | esplorazione del grafo per bisogno, con usato / non usato |

Il vantaggio possibile sul concorrente non sta nel copiare la sua forma, ma nel
**fondare** le stesse cose sui dati: la sua mappatura alle linee guida è dichiarata,
la nostra può essere verificata nodo per nodo.

### 4.2 Che cosa c'è nel grafo UDL (letto da Neo4j)

- **Linee guida e checkpoint UDL 3.0**: 2 `Principle`, 10 `Guideline` («recruit
  interest», «perception», «language and symbols», «interaction»…), 55 `Checkpoint`
  («address distractions», «Clarify vocabulary», «Vary and honor the movement»…), legati
  da `Guideline —MENTIONS→ Checkpoint`. Non ci sono i numeri (7.3, 2.1): gli id sono
  del tipo `concept_optimize_choice_and_autonomy`.
- **Ambiente**: `Colour` (5), `Lighting` (4), `Acoustics` (3), `Furniture` (5),
  `Smells` (4), `Textures` (4), `SensoryEnvironment` (7), con archi espliciti
  («Vibrant tones —MAY_INCREASE→ sensory overstimulation», «Noisy sounds —INCREASES→
  Distraction»).
- **Ciò che manca**: **nessun arco** collega le difficoltà ADHD (o degli alunni
  stranieri) ai checkpoint o ai nodi d'ambiente. Il legame «bisogno → checkpoint» oggi
  non è nel grafo.
- **Profilo classe**: 8 nodi `Context`. La fascia «Group till N» è stata interpretata
  come «classe fino a N alunni»: **va confermato con chi ha costruito il grafo**.

### 4.3 Che cosa è stato fatto

| commit | che cosa | effetto |
|---|---|---|
| `8602b0c` | ordinamento unico dopo l'unione (`src/aix/agent/kg_evidence.py`): prima le strategie, poi i nodi UDL e d'ambiente, poi i bisogni, gli strumenti, il resto, e in fondo il profilo classe; a parità, il punteggio del retrieval | card, Writer, media e Critic vedono prima le strategie |
| `7bb806b` | il Writer legge prima la **mappa dei bisogni**: ogni bisogno della classe con che cosa il grafo suggerisce e sconsiglia, i nodi del profilo che corrispondono a *questa* classe (fascia di numerosità e caratteristiche), l'ambiente e i checkpoint. Le raccomandazioni portano il loro testo; via i nodi `Context`, il boilerplate «Apply X with appropriate adaptations» e i `VECTOR_SIMILAR`. Il prompt di sistema chiede una strategia del grafo per ogni bisogno, con il nome del grafo e in prosa | la Differenziazione si costruisce sul grafo |
| `59abe13` | `fetch_learning_environment`: lettura diretta, per etichetta, dei nodi d'ambiente e dei checkpoint pertinenti, quando il dominio è UDL e la classe ha ADHD, autismo, DOP o elementi di disturbo. Sul grafo reale: 17 nodi + 4 checkpoint. Si spegne con `AIX_KG_ENVIRONMENT_ENABLED=false` | il punto «Ambiente di apprendimento» è basato sui nodi del grafo |
| `64a898b` | la divisione in gruppi arriva già calcolata («2 groups of 5 and 3 groups of 4») e va scritta come un dato di fatto | niente aritmetica nel testo |
| `805254b` | la nota in fondo e il prompt ibrido nominano il grafo del dominio | «Knowledge Graph UDL» |
| `75b14ab` | il turno veloce riceve il materiale del grafo salvato con la lezione | risponde alla domanda A |
| `00a2cbe` | marcatore: niente più «``» davanti alla docente | |
| `9dddf75` | Wikipedia: un risultato che non ha parole in comune con la query viene scartato, e si cerca su en.wikipedia quando it.wikipedia non ha la voce (verificato dal vivo). I bisogni del profilo si cercano per etichetta, non per codice | fonti disciplinari corrette |
| `62b3e0b` | anche le raccomandazioni seguono il livello del loro nodo (prima erano in testa 5 tipi di valutazione); ciò che il grafo sconsiglia non appare tra i nodi da usare; nelle Fonti vanno solo i nodi applicati | correzioni emerse dalla rigenerazione |
| `79e8a97` | turno veloce: risposte compatte e tetto di 4 500 token | risposta completa |

**Costi misurati.** Il prompt del Writer passa da 20 637 a 28 476 caratteri, cioè da
circa 8,6k a 11,5k token di input. Per lezione, $0,0024 → $0,0029. Il turno veloce
aggiunge 2,7–3,3k token alla parte in cache (input del turno 2: da ≈4,0k a 6,7–7,3k):
la prima volta si paga per intero, i turni successivi entro pochi minuti la leggono a
un decimo del prezzo.

**Rischi.**

- Le modifiche ai prompt sono state lette su 2 rigenerazioni di **una sola** lezione.
  Prima del deploy vanno provate sul set delle prove (`scripts/eval/`).
- L'ordinamento cambia anche i nodi che la ricerca dei media usa per scegliere i video:
  da verificare su staging.
- L'interpretazione di «Group till N» va confermata (§4.2).
- `TurnOut.meta` dell'API sessioni ora include anche `kg_digest` (qualche KB): va
  deciso se esporlo ai client o filtrarlo.

---

## 5. Proposte non implementate, in ordine di priorità

| # | proposta | effetto atteso | costo | rischio |
|---|---|---|---|---|
| 1 | **Barriere → linee guida UDL fondate sul grafo**: aggiungere archi `Adhd/ForeignStudents/… —ADDRESSED_BY→ Checkpoint` (curati con l'esperta UDL), poi una sezione o un blocco «Barriere e linee guida» in cui ogni bisogno porta la barriera, il checkpoint (con la sua linea guida) e l'azione nella lezione. È il punto di forza del concorrente, ma con fonti verificabili | alto | medio: lavoro sul grafo, più prompt e una sezione nuova | medio: una sezione nuova tocca parser, outline ed export |
| 2 | **Critic sull'aderenza**: passargli la stessa mappa dei bisogni e un criterio «ogni bisogno della classe ha una strategia del grafo; niente di ciò che il grafo sconsiglia; durata = profilo». Oggi è Gemini 3.5 Flash Lite e approva tutto (4/5) | alto: chiude il ciclo | basso | basso |
| 3 | **Verifica deterministica della durata**: somma dei minuti delle sezioni = durata del profilo. Le tre lezioni viste arrivano a 55 su 60 | medio | basso | basso |
| 4 | **Forma della lezione guidata dall'intento**: struttura per indagine quando l'argomento è scientifico, «assunzioni da verificare», 2 opzioni di valutazione e una domanda finale alla docente | medio-alto | medio | medio: nuove sezioni |
| 5 | **text2cypher**: togliere i nodi d'esempio dal prompt di schema (l'ancora «Group till 15»); non mandare l'argomento disciplinare (scienze, CICLO DELL'ACQUA) al grafo di pedagogia quando lo scope è parziale; per i bisogni del profilo, query deterministiche (`MATCH (n:Adhd)-[r]-(m)`) invece del Cypher scritto dal modello | alto sulla precisione del retrieval | medio | medio: va misurato con `query_metrics` |
| 6 | **Qualità dei dati del grafo**: descrizioni attribuite al nodo sbagliato (Debate ha quella del GBL; i nodi `Context` hanno quella della metodologia; «Difficulty sustaining focus suggests Universal Design for Learning.»), refusi (Poject, Gammification, UniversaryTool, sustaints), nomi troncati («L2).»), doppi spazi, quasi-duplicati (Chunking / Chunking instructions / Chunking tasks into smaller steps) | medio | medio: lavoro editoriale | basso |
| 7 | **Fatto il 16/10 (§7)** — **Storia dei turni veloci**: `_ANSWER_CHARS = 2500` taglia la risposta precedente. Nella rigenerazione, al turno 3 il modello ha concluso che «la risposta precedente era incompleta». Alzare il limite per le risposte, oppure riassumerle | medio | basso: qualche token in più per turno | basso |
| 8 | **Strumenti del turno veloce** (`AIX_LIGHT_TOOLS_ENABLED`): con il materiale salvato a disposizione, `cerca_nel_grafo` può servire per approfondimenti mirati («e per la discalculia?»). Da misurare prima di accendere | medio | basso | basso |
| 9 | **Media**: i video curati parlano delle metodologie (Station Rotation, che cos'è l'UDL), non del ciclo dell'acqua. Valutare la ricerca live sull'argomento disciplinare | medio | medio | basso |
| 10 | **Badge di copertura UDL**: `_extract_meta` conta etichette con «REPRESENT/ACTION/ENGAGE», che nel grafo non esistono, quindi il badge non compare mai. Va contato su Checkpoint e Guideline risalendo al loro principio | basso | basso | basso |
| 11 | **Card del retriever**: «56 strategie» comprende i nodi del profilo classe. Contare solo strategie e strumenti | basso | basso | basso |

---

## 6. Che cosa non è stato verificato

- Una corsa completa della catena su questo branch: planner e retriever non sono
  stati rieseguiti. La rigenerazione parte dallo stato del trace, con i nodi
  d'ambiente letti dal grafo reale.
- Il rendering WebUI del nuovo punto «Ambiente», la cache del provider su turni reali
  ravvicinati e l'effetto dell'ordinamento sui media.
- Il salvataggio di `kg_digest` dentro `run_agent_stream` non ha un test end-to-end:
  sono testati il digest e la sua lettura nel turno veloce.
- Staging e produzione hanno ancora il codice di prima.

### Spesa per i modelli

6 generazioni, tutte Haiku 5.5: 2 lezioni e 4 risposte veloci, **$0,0144** in totale.
Nessuna chiamata a Sonnet. Trace e grafo letti senza costo per i modelli.

---

## 7. Prova dal vero e il dominio neuro (16/10, dopo il report)

### 7.1 La catena completa sullo stack locale

Lo stack locale (`aix-cachetest`) è stato portato su questo branch e il caso è stato
rigenerato con la catena completa, planner e retriever compresi, attraverso l'API
delle sessioni: stessa richiesta, profilo con ADHD e alunni non italofoni, 22
alunni, 60 minuti, più le due domande di follow-up della docente.

| | staging (prima) | UDL, questo branch | neuro, prima del §7.2 | neuro, dopo il §7.2 |
|---|---|---|---|---|
| concetti in testa | Group till 15/20/30, Cohesive… | Multisensory Activities, Self-Regulated Learning, Peer to Peer… | Students' Independent Metacognitive Development, Adaptive Strategies, Sustained Engagement, Predictability, Reduced Anxiety (effetti) | Visual Aids, Structured Routines, Step-by-step Instruction, Concrete Manipulatives |
| Station Rotation (sconsigliata dal grafo) | attività centrale | assente | assente | assente |
| ambiente | assente | dai nodi Colour, Lighting, Acoustics | generico, non dal grafo | dai nodi External Distractions, Salient Stimuli, Supportive Classroom Environment |
| turno 2 «esploriamo il grafo» | «non ho accesso al grafo» | elenco per bisogno, usato / non usato | elenco con effetti presentati come strategie | difficoltà ADHD con il loro perché, neuromiti da evitare, nessun nodo per gli alunni non italofoni (detto apertamente) |
| turno 3 | «documento senza marcatori ``» | «No, non sono solo sei» | «la mia risposta precedente si è interrotta» (falso) | corretto |
| durata | 60 | **55** | **55** | **53** |

### 7.2 Il grafo neuro è fatto diversamente

Letto in sola lettura il 16/10:

- **Cosa dice cos'è un nodo.** Nel grafo neuro ogni concetto ha una sua etichetta, spesso
  con un solo nodo; il tipo sta nella proprietà `category`: «ADHD», «Dyslexia»,
  «Dyscalculia», «Autism Spectrum», «Tourette's Disorder» (le difficoltà); «Teaching
  Practices», «Support Systems», «LearningStrategies», «Feedback» (cosa fare);
  «Educational Results», «Pedagogical Strategies», «Developmental Outcomes»,
  «Evidence-based» (effetti: «Reduced Anxiety», «Learning Gains»); «External
  Influences» (l'ambiente); «Neuromyths».
- **Neuromiti archiviati come pratiche.** Diversi miti sono archiviati come «Teaching
  Practices»: «Pseudoscientific exercises» (Brain Gym), «Modality preference myth»
  (stili di apprendimento), «Hydration myth», «Music-IQ myth». Una classificazione per
  sola categoria li avrebbe messi in testa alle strategie.
- **Nessun percorso da difficoltà a strategia.** Le difficoltà ADHD non portano a
  strategie, nemmeno in due passi. Portano a ciò che comportano: «Difficulty
  Managing Frustration or Stress» scatena «Anxiety»; «Inefficient Rehearsal or
  Organization» compromette «Encoding and Transfer of Knowledge»; «High Creativity»
  favorisce il problem solving. È il *perché* di una scelta, non la scelta.
- **Nessun nodo per gli alunni non italofoni.**
- **Descrizioni con errori.** Molte descrizioni sono copiate da un arco a più nodi
  («X leads to Y, reflecting the interconnected nature…»), e «Reduced Stigma» e
  «Improved Engagement» sono archiviati come pratiche.

Implementato (`25ec05a`):

- `kg_evidence.kind` riconosce prima i miti, poi i nodi neuro per categoria;
- i bisogni portano «→ nel KG: scatena Anxiety; …»;
- gli effetti finiscono in una sezione «motivazioni, non attività»;
- i miti non sono mai proposti come nodi da usare. Arrivano al Writer solo quelli in cui
  una lezione per questa classe potrebbe cadere (stili di apprendimento sempre; con
  l'ADHD zucchero, acqua, multitasking, Brain Gym), al massimo 6;
- `RetrieverAgent.fetch_neuro_class_evidence` legge direttamente i bisogni della
  classe con i loro effetti, i neuromiti e, quando serve, l'ambiente, come
  `fetch_learning_environment` fa per UDL. Lo spegne lo stesso flag
  `AIX_KG_ENVIRONMENT_ENABLED`;
- il prompt del Writer ha tre regole per queste sezioni. Nella lezione rigenerata:
  «il suono è pensato per sostenere la memoria, non come stile di apprendimento».

Corretto anche (`bdc7f80`) il punto 7 del §5. Nella cronologia l'ultima risposta veloce
resta intera; le precedenti, se tagliate, dicono che il taglio è nostro. Il «…»
faceva dire al modello che la sua risposta si era interrotta.

### 7.3 Ancora aperto

- **Durata.** Tutte e quattro le lezioni rigenerate arrivano a 53–55 minuti su 60, e il
  Critic dà 5/5. Le proposte 2 e 3 del §5 diventano le prime da fare.
- **Strategie attribuite al grafo sbagliato.** «Focus Tasks» è citato sotto il Knowledge
  Graph in una lezione neuro, ma è un nodo del grafo UDL. La risposta veloce lo ha
  ammesso al turno 3.
- **Alunni non italofoni.** Il grafo neuro non ha nulla su di loro, e la lezione lo
  compensa con scelte proprie (dichiarate come tali al turno 2).
- **Video in UDL.** Una lezione UDL ha proposto come video un link a una ricerca su
  YouTube.

Costo delle prove dal vero: 4 lezioni e 8 risposte veloci con Haiku 5.5, meno di $0,05.

### 7.4 Durata e nomi del grafo (16/10, sera)

- **Durata.** Il controllo esisteva (`rule_duration_mismatch`), ma è spento per
  impostazione predefinita (`AIX_VALIDATOR_ENABLED=false`) e tollera il 10% per
  scelta: 53–55 su 60 passavano. Il prompt chiedeva già una durata «esatta». Ora il
  profilo passa al Writer una ripartizione già sommata («Introduzione 10 minuti +
  Attività 40 minuti + Valutazione 10 minuti = 60 minuti»), come per i gruppi.
  Rigenerate UDL e neuro: **60 su 60** tutte e due, con le attività che sommano 40.
  Il parser dei minuti ora accetta anche «10′», la notazione che il Writer aveva
  copiato da una prima versione della ripartizione.
- **«Focus Tasks» nelle lezioni neuro.** Veniva dall'esempio scritto nelle regole del
  prompt: un nome di nodo UDL. L'esempio ora non nomina nessun nodo, e la regola dice
  che sotto il Knowledge Graph vanno solo nomi presenti nell'evidenza. Nella lezione
  neuro rigenerata non compare più.
- Resta da decidere se accendere il validatore su staging (§5, proposte 2–3): con la
  ripartizione il caso tipico non lo fa più scattare, quindi il costo di un secondo
  passaggio del Writer sarebbe raro.
