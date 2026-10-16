# Analisi dell'integrazione con AIxLearning

**Data:** 16 ottobre 2026 · **Autore:** Angelo Casali (con Claude)
**Codice analizzato:** `FEM-modena/aixlearning` su `main` (`67b6cd7c`), più i branch
`production`, `staging`, `feature/graph-rag`, `feature/party-files`; la nostra API su
`chore/repo-reorg` (`f3daecb`); il documento di Louis *Technical Implementation
Roadmap — Private RAG API and AIxLearning Integration* (3 ottobre 2026).

**Domanda:** l'API che stiamo costruendo per i chatbot è compatibile con la
piattaforma AIxLearning, e cosa va considerato prima che i dev la integrino?

**Risposta breve:** la direzione è giusta. Sessioni, turni, documenti, export e
spiegazioni combaciano con il modo in cui AIxLearning è costruita. Le cose da
sistemare stanno nel contratto tra i due sistemi: otto incompatibilità concrete
(§3), quattro decisioni (§4) e alcune idee del documento di Louis da adottare
(§5).

---

## 1. Come funziona oggi AIxLearning

- **I «chatbot» sono gli strumenti di `lesson_planner`**: 24 tipi di `LessonPlan`
  (UDL, NEURO, GOAL_SETTER, ESCAPE_ROOM_DESIGNER…), registrati in
  `lesson_planner/tools/categories/*.py`. L'app `chatbot` è un'altra cosa: è il
  Prompt Lab per gli studenti.
- **Ogni strumento è una conversazione**:
  - `LessonPlan` → `Conversation` → `Message`;
  - il primo messaggio genera il piano;
  - i successivi passano da un classificatore d'intento (testo, quiz o immagine).
- **La generazione gira in un worker Dramatiq** (`streaming_service/streaming.py`)
  con `max_retries=3`. I token arrivano al browser con **Mercure**, a gruppi di
  10 delta (`SEND_EVERY_N_CHUNKS`); alla fine la piattaforma pubblica l'HTML
  completo (`marko`, GFM).
- **A ogni turno rimandano al modello tutta la storia**, più il testo completo di
  ogni file caricato (`clients/text_client.py`).
- **GraphRAG oggi lo usa solo lo strumento NEURO**:
  - attraverso `POST /api/v1/context` (`graphrag_service/graph_rag_integration.py`);
  - il contesto diventa il system prompt;
  - se il grafo non ha dati, si torna al prompt di Langfuse;
  - UDL è su `feature/graph-rag`, non unito.
- **Il prompt viene composto lato loro** (`methods/prompt_assemblers.py`): lingua,
  aula (`Classroom`), classe (`Party`: grado, numero di studenti, bisogni speciali,
  caratteristiche) e testo dell'utente, tutto in un'unica stringa.
- **Anonimizzano tutto** (`anonymizer/`, Presidio + spaCy `it_core_news_md` /
  `en_core_web_sm`, riconoscitori propri, elenchi di personaggi pubblici):
  - il testo dell'utente prima di ogni chiamata al modello;
  - i file (TXT, PDF, DOCX, massimo 5 MB) al caricamento.
  Dei file **conservano solo il testo anonimizzato**, non l'originale.
- **Crediti**: li scalano prima di generare e li rimborsano dopo il terzo
  tentativo fallito.
- **Export**: Word e PDF propri, con watermark.
- **Accesso**: in produzione è arrivato il login OpenID Connect; gli utenti Django
  hanno id numerici.

### Dove girano

| Istanza AIxLearning | VM | Configurazione GraphRAG |
| --- | --- | --- |
| produzione | `aixl-fe-01` | `GRAPH_API_ENDPOINT=https://graph.aiforlearning.digital/fast-api`, utente `aiforl-graph` |
| corporate | `aixl-fe-01` | da verificare |
| corporate-staging | `aiforl-fe-02` | da verificare |
| staging | `we-fe-02` | **nessuna**: lo staging di AIxLearning non chiama GraphRAG |

Noi siamo su `we-graph-01`. La chiamata passa **da internet**, attraverso Traefik:
non c'è una rete Docker comune.

---

## 2. Cosa combacia già

- **Conversazione.** `Conversation.thread_id` esiste e può contenere il nostro
  `session_id`. I loro stati (waiting, streaming, ready, error) corrispondono ai
  nostri.
- **Formato.** Il nostro markdown (tabelle, elenchi) si vede bene con il loro
  renderer GFM.
- **Documenti.** I «file della classe» (`feature/party-files`, `party.ContextFile`)
  sono lo stesso modello del nostro «carica una volta, usa per id».
- **Profilo della classe.** `educational_profile` era stato scritto 1:1 sui loro
  `Party` e `Classroom` (ma vedi §3 A–C: nel frattempo i loro modelli sono cambiati).
- **Accesso.** Basic auth di un service account, con l'utente finale indicato a
  parte: è lo schema che usano già.
- **Domini.** L'agente conosce `udl` e `neuro`, proprio i due strumenti che
  AIxLearning ha già collegato (o sta collegando) a GraphRAG.

---

## 3. Incompatibilità concrete

| # | Problema | Conseguenza | Chi sistema |
| --- | --- | --- | --- |
| A | **Codici BES cambiati.** I loro (`party.StudentDisabilities`) sono DSA, ADHD, DOP, DF, **SC**, PD, SA, **CL**; noi accettiamo ancora DCGL, DCGM, DCGS e DLDS. | Una classe con SC o CL riceve **422** | noi: accettare SC e CL, convertire i vecchi codici |
| B | **`students_number`.** Da loro il default è 0; noi vogliamo un valore tra 1 e 40. | 422 su ogni classe lasciata a 0 | noi |
| C | **Campi dell'aula.** Da loro sono a tre stati (Sì / No / Non so), compreso `forniture_mobility`; da noi sono booleani, oppure Sì / No / Parzialmente. Ci manca il tipo di ambiente (presenza, remoto, ibrido). | «Non so» viene rifiutato | noi: accettare i loro valori; il campo mancante da aggiungere |
| D | **Più istanze, un solo service account.** Produzione, corporate, corporate-staging e staging hanno id utente numerici indipendenti. Con un solo account, l'utente 42 di corporate e l'utente 42 di produzione diventano **la stessa persona** da noi. | Documenti e sessioni condivisi tra persone diverse | entrambi: un'identità per istanza (tenant) |
| E | **Tentativi ripetuti senza idempotenza.** Dramatiq ripete fino a 3 volte; noi non abbiamo `Idempotency-Key`. | Turni doppi e costo fino a triplo. Dopo un timeout il nuovo tentativo riceve `409 already_running`, e la lezione completata non arriva mai | noi: chiave di idempotenza; documentare il recupero con `GET /sessions/{id}` |
| F | **Credenziali in produzione.** Traefik controlla `aiforl-graph` e passa l'intestazione all'app, ma l'app in produzione non ha `GRAPH_API_USERNAME` / `GRAPH_API_PWD`. | Dalla configurazione, oggi `/agent/*` con Basic auth risponde 401 in produzione | noi, in cloud-infra (al momento del rilascio) |
| G | **Streaming.** La loro interfaccia è costruita sui token; noi mandiamo le fasi e poi il risultato completo. Il Critic può anche riscrivere la lezione. | Senza token, circa 50 secondi di schermo vuoto | noi: token, più un markdown finale che sostituisce il testo |
| H | **Storia della conversazione.** A ogni turno rimandano tutta la storia e tutto il testo dei file. | Con noi la storia la teniamo noi: si manda solo il messaggio nuovo | loro |

---

## 4. Decisioni

### 4.1 Anonimizzazione: resta, ma da che parte?

Oggi la piattaforma anonimizza prima del modello, e dei file tiene solo il testo
anonimizzato. La nostra API dei documenti invece vuole il file vero, per leggere
pagine, impaginazione e scansioni (OCR). Mandare l'originale così com'è
aggirerebbe la garanzia della piattaforma: i dati personali arriverebbero da noi,
agli embedding di OpenRouter e a Langfuse.

**L'anonimizzazione resta** (Angelo, 16/10/2026: è un lavoro che ha seguito lui).
Resta da scegliere dove farla. Ci sono due strade.

#### Strada 1: la piattaforma la tiene (come oggi)

La piattaforma anonimizza e ci manda testo già pulito: i messaggi così come sono,
i documenti come **Markdown con un titolo per pagina** (`## Pagina 3`).

- **Da noi non serve nulla.** Il Markdown lo accettiamo già, e il nostro parser fa
  di ogni titolo una sezione: la citazione diventa «Pagina 3». Il loro estrattore
  PDF ha già le pagine (`anonymizer/document.py`, `_extract_pdf_text`): basta
  scriverle con un titolo invece di unirle con una riga vuota.
- **La garanzia più forte**: i dati personali non lasciano mai la piattaforma.
- **Un solo proprietario del codice** (il loro team), nessun modello spaCy in più
  sulla nostra VM.
- **Si perde**: l'OCR delle scansioni (il loro estrattore legge solo il testo
  digitale, quindi una scansione arriva vuota), le immagini, i PPTX, le tabelle
  come unità. I limiti restano i loro: 5 MB, TXT/PDF/DOCX.

#### Strada 2: la spostiamo nella nostra API

La piattaforma manda il file originale e il testo dell'utente; noi anonimizziamo
**prima** di qualunque uscita verso terzi.

- **Si guadagna**: scansioni, PPTX, immagini e tabelle, letti dal nostro parser e
  poi anonimizzati; un solo punto per documenti, messaggi e profilo.
- **Vale anche per la nostra WebUI**: i docenti che caricano direttamente da noi
  oggi non sono anonimizzati, e con questa strada lo sarebbero.
- **Costa**: memoria e manutenzione (vedi sotto), e i dati personali arrivano sul
  nostro server, anche se non escono verso terzi.

#### Raccomandazione

**Pilota con la strada 1**: costa zero da noi e tiene la garanzia attuale. La
strada 2 diventa necessaria quando servono le scansioni o i PPTX, oppure quando
vogliamo anonimizzare anche la nostra WebUI. La strada 1 non impedisce la 2: il
contratto dell'API non cambia.

#### Se si sceglie la strada 2

In pratica:

- **documenti:** dopo la lettura (parser e OCR) e prima di embedding, mappa del
  documento e Langfuse. Pagine e citazioni restano corrette, perché si anonimizza
  il testo estratto pagina per pagina;
- **messaggi:** prima di Planner, Writer e risposte veloci;
- **risposta:** decidere se rimettere i nomi veri (abbiamo la mappa) o lasciare i
  segnaposto. Oggi la piattaforma salva la mappa ma non la usa per ricostruire i
  nomi.

Da verificare prima di iniziare:

- **Memoria.** Presidio con spaCy `it_core_news_md` + `en_core_web_sm` aggiunge
  diverse centinaia di MB per processo. La VM ha 4 GB senza swap, Neo4j ne usa
  circa 2,3, e girano due container dell'API (produzione e staging). Va misurato:
  in alternativa un solo servizio di anonimizzazione condiviso, oppure lo swap.
- **OCR via OpenRouter.** Una scansione letta con il modello di visione esce
  *prima* di poter essere anonimizzata. Con Tesseract locale il problema non c'è.
- **Riuso del codice.** Il modulo `anonymizer/` è loro (GMemoli, giugno 2026) ed è
  maturo: 1.711 righe di servizio, riconoscitori propri, test. Va portato, non
  riscritto, mantenendo un'unica fonte di verità tra i due repo.
- **Personaggi pubblici.** Gli elenchi `public_figures_*.json` evitano di
  anonimizzare «Garibaldi» in una lezione di storia: per noi è essenziale, perché
  il grafo e i documenti ne sono pieni.

### 4.2 Quali chatbot

L'agente copre i domini `udl` e `neuro`, cioè 2 strumenti su 24. Gli altri restano
sul sistema della piattaforma (eventualmente con `/context`). Una «serie di
chatbot» più ampia richiede domini nuovi nel grafo.

### 4.3 Crediti

Una risposta veloce costa molto meno di una lezione, ma la piattaforma scala i
crediti prima di sapere che cosa tornerà (`lesson`, `answer` o `handoff`). Va
deciso il momento dell'addebito: per esempio un addebito minimo, completato a
risultato noto, oppure un costo dichiarato dall'API in anticipo per modalità.

### 4.4 Export e watermark

La piattaforma ha già il suo export con watermark: probabilmente lo terrà. Il
nostro (`/sessions/{id}/export`) resta disponibile, e scrive la dichiarazione IA
nel documento.

---

## 5. Il documento di Louis

### 5.1 Già superato su `chore/repo-reorg`

Il documento descrive advanced al 3 ottobre. Da allora sono stati fatti:

- l'API dei documenti (upload, versioni, stato, collegamento, download
  dell'originale);
- le sessioni con risposte veloci, rimando a una lezione nuova e rifiniture;
- gli export (md, txt, pdf, docx, scaletta) e le spiegazioni (planner,
  retriever, critic, struttura);
- il 409 sui documenti non pronti, e codici di errore stabili `{code, message}`;
- l'identità per utente finale, legata alla proprietà dei documenti.

Anche il timore su `AIX_WRITER_MAX_TOKENS=2000` è superato: nel `.env` condiviso è
8000.

### 5.2 Dove ci discostiamo apposta

Louis propone di **non** usare righe `Lesson` come sessioni dell'API, ma dei
*material set* separati. Abbiamo fatto il contrario, e resta la scelta giusta:

- è il motivo per cui l'API ha risposte veloci, storia ed export senza
  riscriverli;
- il rischio principale che Louis indica non si presenta: due turni sulla stessa
  sessione si escludono (`already_running`), e la chiave del recupero dei
  documenti è l'id della lezione, unico per sessione.

Ci perdiamo il fissare una versione specifica di un documento per sessione: oggi
vale l'ultima versione pronta.

### 5.3 Da adottare

1. **`Idempotency-Key`** su upload e turni (§3 E).
2. **Tenant per istanza** della piattaforma (§3 D).
3. **Identità firmata prima della produzione** (RS256 con `iss`, `aud`, `sub`,
   `tenant`, `exp`).
   - Louis accetta Basic auth con id utente solo se il servizio è interno. Qui non
     lo è: VM diverse, chiamate via internet.
   - Per il pilota bastano una **lista di IP ammessi** in Traefik sui percorsi
     `/api` e credenziali per istanza.
4. **Chiudere la superficie pubblica.**
   - Sullo staging `/docs` e `/openapi.json` rispondono 200 a chiunque.
   - `/api/v1/context` non ha autenticazione nell'app (in produzione la mette
     Traefik, sullo staging nessuno).
5. **Id di correlazione** tra il trace Langfuse della piattaforma e il nostro.
6. **Documenti obbligatori.** Un documento che *deve* essere seguito e non è
   disponibile fa fallire il turno, invece di essere ignorato. Il 409 dice quali
   documenti mancano, con `Retry-After`.
7. **Anteprima della pagina citata**, per citazioni verificabili (più avanti).
8. **Il suo §15 come lista delle decisioni** da chiudere con i dev.

### 5.4 Dove non combacia con la realtà

- Immagina GraphRAG sulla stessa rete Docker di AIxLearning: sono su VM diverse.
- La sua stima di 116–192 ore per il piano dell'API dei documenti è in buona parte
  già spesa: restano i punti di §5.3.

---

## 6. Ordine di lavoro

| # | Lavoro | Stato |
| --- | --- | --- |
| 1 | Profilo della classe compatibile con la piattaforma (§3 A, B, C) | da fare |
| 2 | Streaming dei token nell'SSE delle sessioni (§3 G) | da fare |
| 3 | Idempotenza e tenant (§3 D, E) | da fare |
| 4 | Staging pronto per i dev: `chore/repo-reorg` su staging, credenziali solo su `api-staging`, lista di IP, `/docs` protetto | in attesa (Angelo) |
| 5 | Anonimizzazione (§4.1): scegliere tra strada 1 (piattaforma, Markdown per pagina) e strada 2 (nostra API, dopo la verifica della memoria) | da decidere |
| 6 | Riunione con i dev: §4, §5.3, §15 di Louis | dopo il 4 |
