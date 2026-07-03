# Risposte e Domande per Product Team — Agentic GraphRAG

**Data:** 2026-06-07  
**Contesto:** preparazione meeting Product Team su Agentic GraphRAG / WebUI  
**Fonte di partenza:** transcript "Team AIforLearning - Agentic GraphRag - Product - 2026_06_04"  
**Obiettivo:** rispondere punto per punto ai dubbi emersi e preparare domande di prodotto da chiarire prima di consolidare il prototipo e avviare user testing.

> Nota operativa: nel transcript il prossimo incontro è indicato come "martedì 17:00-18:00"; nel messaggio di lavoro è stato citato "lunedì". Verificare data/orario ufficiale.

---

## Executive Summary

La nostra posizione è che **Agentic GraphRAG non va presentato come un semplice nuovo chatbot**, ma come una **nuova architettura specializzata per la progettazione didattica assistita da IA**. La differenza principale rispetto agli strumenti AI precedenti è che non c'è un unico prompt che produce direttamente un output, ma una pipeline esplicita:

1. **Planner** — interpreta la richiesta, individua obiettivo didattico, concetti e query.
2. **Retriever** — recupera conoscenza dal Knowledge Graph e fonti/media.
3. **Writer** — genera la lezione adattandola al profilo educativo.
4. **Critic** — valuta qualità, coerenza pedagogica e può chiedere revisione.

Questa architettura è già implementata con LangGraph e FastAPI/WebUI, e la WebUI attuale è una **reference implementation**: mostra molti dettagli perché serve a capire, testare e discutere cosa deve rimanere visibile nel prodotto finale.

Le decisioni centrali per il Product Team non sono solo grafiche. Sono decisioni di prodotto su **quanto rendere visibile l'agenticità**:

- Mostrare solo la lezione finale = UX più semplice ma meno trasparenza.
- Mostrare tutti gli agenti come oggi = massima trasparenza ma rischio sovraccarico.
- Mostrare una sintesi progressiva = migliore compromesso consigliato per il prodotto.

La raccomandazione è adottare una UX a **progressive disclosure**:

- Vista docente standard: passaggi chiari ma non tecnici ("Progettazione", "Ricerca", "Scrittura", "Revisione").
- Vista dettagli / power user: query, concetti, copertura KG, fonti, critic verdict.
- Vista debug interna (solo admin): JSON, nomi tool, payload tecnici, tracing.

---

## 1. Considerazioni Tecniche e Architetturali

### 1.1 Integrazione dello strumento

**Punto emerso**: chiarire se sarà un tool separato o se tutti gli strumenti esistenti adotteranno questa architettura. Capire destino degli strumenti attuali e dei prompt esistenti.

**La nostra risposta:**  
Agentic GraphRAG deve essere pensato come una **nuova capacità architetturale** e non necessariamente come un tool separato da lasciare isolato. Per la fase pilota, è corretto mantenerlo come **modulo/prototipo autonomo con WebUI dedicata**, perché questo ci permette di osservare e validare l'esperienza agentica. In produzione, la direzione più solida è integrarlo dentro AixLearning tramite API e componenti frontend nativi, preservando il valore degli strumenti attuali, aggiungendo raggionamenti agentici adeguati ove serve.

In altre parole: *questa WebUI vi mostra il *dietro le quinte* dell'architettura agentica, così che AixLearning possa ispirarsene e integrare elementi agentici necessari per gli utenti finali.

Vale anche la pena inquadrare il contesto: le architetture agentiche (e quelle che verranno) stanno ridisegnando le interfacce dei prodotti AI — non è a caso da anni, perfino le AI generaliste come ChatGPT/Claude etc hanno introdotto una "modalità agente", modalità deep research e cosi via. Hanno dovuto fare scelte di design dettate dall'avento di queste nuove architetture. La differenza è che le generaliste aggiungono l'agenticità mantenendo un visivo semplice; noi abbiamo una AI specialistica/verticale in ambito education ( **verticale e specializzata**) di quel ragionamento per la didattica, esponendone gli elementi tecnici proprio perché possiate scegliere cosa rendere visibile e cosa integrare.


**Raccomandazione:**  
Non migrare automaticamente tutti gli strumenti esistenti alla nuova architettura. Conviene classificare gli strumenti in tre gruppi:

1. **Strumenti da mantenere invariati** perché semplici e già efficaci (es.: un correttore ortografico, un riempi-template, una ricerca glossario — metterci sopra una pipeline agentica sarebbe spreco).
2. **Strumenti da integrare come tool/capability** dentro la pipeline agentica (es.: un prompt "genera quiz" o "riassumi testo" che diventa uno strumento richiamabile da un agente).
3. **Strumenti da ripensare come esperienze agentiche** quando richiedono pianificazione, recupero conoscenza, generazione e controllo qualità (es.: "progetta una lezione completa adattata alla classe" - è esattamente ciò che fa Agentic GraphRAG).

I prompt esistenti non vanno buttati: possono diventare **asset riusabili** dentro singoli agenti o tool specializzati.

**Due strade possibili per l'integrazione:**

- **a) Integrazione nell'app nativa AixLearning:** il frontend nativo adotta gli elementi agentici giudicati necessari e concordati tra team product e  team AI, mantenendo l'esperienza esistente. È la direzione già in valutazione, discussa con i DEV e di cui abbiamo già valutato positivamente la fattibilità tecnica.

  **Rassicurazione tecnica:** nell'app nativa AixLearning adottare questi pattern agentici — inclusa la *progressive disclosure* — **non richiede riscrivere il frontend**, ma solo **aggiunte incrementali ai template Django**. Il frontend nativo usa già le stesse tecnologie del nostro prototipo (htmx, WebAwesome, Mercure/SSE): le stesse primitive con cui realizziamo viste espandibili, card degli agenti e streaming in tempo reale.

- **b) Nuovo prodotto end-to-end a partire da questa WebUI:** con il supporto del design, si costruisce a quattro mani un'esperienza utenti completa che evolve/sostituisce la precedente, partendo dalla nostra webUI già esistente. Non ancora discussa: in questo senso il confronto con voi è necessario

Noi stiamo aperti a procedere con entrambe le strade.

---

### 1.2 Concorrenza e richieste simultanee

**Punto emerso:** capacità di gestire richieste concorrenti.

**La nostra risposta:**  
Prima una distinzione importante: **la concorrenza dell'app nativa AixLearning resta invariata.** Questa WebUI agentica non modifica il modo in cui l'app nativa gestisce oggi le richieste concorrenti.

Parlando invece della **WebUI agentica**: l'architettura è già compatibile con la concorrenza a livello applicativo , FastAPI usa endpoint async, LangGraph viene eseguito in modo asincrono, lo streaming SSE aggiorna l'interfaccia mentre la pipeline lavora, e il sistema usa sessioni DB per richiesta. Tuttavia, per una produzione con molti utenti serve aggiungere controlli operativi espliciti.

**Dove siamo oggi:**  
Oggi il sistema è adatto a un **pilot controllato** e a un numero limitato di utenti concorrenti. Sono già presenti:

- ✅ streaming SSE per le run;
- ✅ registro in-memory per evitare doppio stream sulla stessa lezione nello stesso processo;
- ✅ checkpointer SQLite/Postgres per memoria multi-turn;
- ✅ fetch esterni e media off-critical-path;
- ✅ rate limiting su alcune API esterne.

**Punti da considerare per la produzione (già in agenda dal team AI):**  
Non è ancora implementato un sistema completo di:

- 📋 rate limit per utente;
- 📋 coda globale delle generazioni;
- 📋 limite globale di chiamate LLM concorrenti;
- 📋 coordinamento cross-worker del registro run;
- 📋 dashboard operativa per carico/costi.

**Raccomandazione - due fasi distinte:**

1. **Pilot controllato (oggi):** per pochi utenti selezionati (team AI, product, domain expert o un set di docenti scelti). L'infrastruttura attuale (SSE, checkpointer, registro in-memory) è sufficiente.
2. **Se la WebUI diventa un altro prodotto finale:** il team AI allarga semplicemente la scala — rate limit per utente, coda globale, limite globale di chiamate LLM concorrenti, coordinamento cross-worker, visibilità admin vs user, Postgres checkpointer, metriche e policy di retry.


---

### 1.3 Provider, OpenRouter e gestione dei dati

**Punto emerso:** oggi si usa OpenRouter; dubbi su provider terzi e pratiche di gestione dati.

**La nostra risposta:**  
Tecnicamente il sistema usa un client OpenAI-compatible configurato verso OpenRouter, quindi il provider è **sostituibile via configurazione**. Questa è una buona scelta architetturale perché evita lock-in e permette di scegliere modelli diversi per qualità, costo, latenza o compliance.

**Dove siamo oggi:**  
La configurazione supporta:

- `OPENROUTER_API_KEY`;
- `OPENROUTER_BASE_URL`;
- `LLM_MODEL`;
- modelli differenziabili per critic / Text2Cypher;
- embedding model configurabile.

**Criteri consigliati:**  
La scelta provider non deve essere solo tecnica. Va valutata con criteri di:

- data retention;
- uso dei dati per training;
- DPA / termini privacy;
- localizzazione dei dati;
- giurisdizione applicabile (es. CLOUD Act per provider USA);
- SLA e affidabilità;
- auditability;
- costo per token;
- qualità sui task educativi in italiano.

**Raccomandazione:**  
Mantenere l'astrazione OpenAI-compatible, ma definire una **provider policy** con due livelli:

1. Provider ammessi per pilot interno.
2. Provider ammessi per produzione con utenti reali.

**Provider candidati** (posizionamento indicativo, da confermare con DPA/legal):

| Provider | Posizionamento tipico | Più adatto a |
|---|---|---|
| OpenRouter | router multi-provider: massima flessibilità, ma il trattamento dati dipende dal modello sottostante e dalle impostazioni di routing | pilot; in produzione solo se Enterprise (EU in-region, ZDR - Zero Data Retention, DPA verificati) |
| Azure OpenAI | offerta enterprise con region UE e impegni contrattuali (no training sui dati, da confermare per piano/regione) | produzione |
| Mistral | provider europeo (Francia), interessante per la residenza dati UE | produzione |
| Anthropic (Claude) | qualità elevata su scrittura/ragionamento; da verificare DPA e residenza | pilot → produzione |
| Google Vertex | offerta enterprise con region UE | produzione |

Dato che il sistema è teacher-facing e non student-facing (a mia conoscenza), il rischio è più basso, ma resta fondamentale evitare invio di dati personali di studenti. Il profilo educativo dovrebbe restare aggregato: numero studenti, BES/DSA per tipologia, livello, contesto aula, non nomi o dati identificativi. Questo è coerente con il principio GDPR di **minimizzazione dei dati**.

**Nota compliance (EU AI Act):**  
Un sistema che genera contenuti per gli utenti rientra negli obblighi di trasparenza dell'**Art. 50 EU AI Act**: i contenuti generati da IA vanno marcati come tali, con applicazione prevista dal **2 agosto 2026** (salvo aggiornamenti normativi, da verificare con Legal/DPO). Conviene quindi introdurre già nel prototipo un'etichetta persistente "Contenuto generato da IA".

Va inoltre distinto chi ha quali responsabilità: il **fornitore del modello** (es. OpenAI, Anthropic), il **fornitore del sistema** (Agentic GraphRAG) e il **deployer** che lo mette a disposizione dei docenti (FEM/AixLearning). L'obbligo di trasparenza — e di tracciare quale modello ha generato il contenuto — ricade su chi offre il sistema agli utenti **indipendentemente dal provider scelto**: cambiare provider non elimina questo dovere. Il tema si collega sia alla scelta provider (1.3) sia a cosa mostrare all'utente (2.1).

*Nota: questa sezione offre un quadro operativo, non una consulenza legale; la validazione finale dei termini di trattamento dati spetta a legal/DPO.*

**Decisione da prendere (Legal):**  
Quali provider sono accettabili per il pilot e quali per la produzione? Raccomandazione/suggestione: policy a due livelli — flessibilità (es. OpenRouter) per il pilot, provider con DPA e residenza UE verificati per la produzione, previo allineamento con FEM/legal/DPO.

---

### 1.4 Impatto sui costi

**Punto emerso:** valutare impatto della nuova architettura sui costi esterni.

**La nostra risposta:**  
Un'architettura agentica costa potenzialmente più di una singola chiamata LLM, perché usa più passaggi. Tuttavia produce un output più controllato: pianificazione, retrieval, generazione e revisione. Quindi il confronto corretto non è solo "costo per chiamata", ma **costo per output utile e affidabile** — più accuratezza, meno allucinazioni, contenuto pedagogico più *grounded* e arricchito con media/fonti. È onesto dirlo: introdurre l'agentico aumenta inevitabilmente i costi , come per ogni azienda che adotta queste architetture.

**Leve già presenti o previste:**

- scelta modello via env;
- modello più economico per critic / Text2Cypher;
- limiti token per writer e critic;
- numero massimo di revisioni;
- Corrective RAG disattivabile;
- media live off-critical-path e cache;
- modello indipendente per Critic e Text2Cypher (Planner e Writer condividono il modello principale).

**Oggi non facciamo ancora:** budget/metering a runtime, profili di costo selezionabili nel prodotto (oggi si impostano via configurazione) e cache delle generazioni LLM.

**Raccomandazione:** (questo blocco risponde alla domanda implicita: possiamo controllare quanto costa ? )  
Per la fase prodotto, definire tre profili di costo:

1. **Pilot qualità alta** — modello forte, pochi utenti, focus su feedback.
2. **Produzione bilanciata** — modello forte solo dove serve, modelli economici per task tecnici.
3. **Modalità economica** — meno revisioni, token cap più stretti, live media limitata/cache-first.

**Decisione product/business:**  
Qual è il costo massimo accettabile per una lezione generata? E quale trade-off accettiamo tra qualità, latenza e costo?  - l'indagine rispondendo a tale domanda può essere lo standard base per indirizzare i costi extra.
Come trasferiamo il maggior valore sul prezzo? (es. revisione abbonamenti, tier premium per le funzioni agentiche). Da allineare con il *sales* di FEM — è una scelta di business, non puramente architetturale/tecnica.

---

## 2. Esperienza Utente e Interfaccia (UX/UI) — Trasparenza Agentica

### 2.1 Visibilità degli agenti

**Punto emerso:** oggi tutti gli agenti vengono mostrati per comprensione interna/debug; decidere quali mostrare agli utenti finali.

**La nostra risposta:**  
Non consigliamo di nascondere completamente l'architettura agentica. Uno dei valori del sistema è che il docente può vedere che l'AI non "sputa" una risposta, ma passa da progettazione, ricerca, scrittura e revisione. Questo aumenta trasparenza, fiducia e AI literacy.

**Tuttavia:** non tutto ciò che vediamo noi deve essere mostrato all'utente finale.

**Raccomandazione di UX a tre livelli:**

1. **Vista docente standard:** mostra i passaggi con etichette pedagogiche:
   - Progettazione
   - Ricerca nel Knowledge Graph
   - Scrittura della lezione
   - Revisione pedagogica
2. **Dettagli espandibili:** query, concetti chiave, fonti, punteggi, critic verdict.
3. **Debug interno:** JSON, nomi tool, payload, confidenze tecniche, tracing.

**Chi vede cosa (modello cumulativo):**

| Ruolo | Vista docente standard | Dettagli | Debug interna |
|---|:--:|:--:|:--:|
| Docente | ✅ | — | — |
| Power user (docente esperto) | ✅ | ✅ | — |
| Admin | ✅ | ✅ | ✅ |

Il power user è un docente avanzato che arriva fino a "Dettagli"; il debug interno resta riservato all'admin. Quindi *super power user* e *admin* sono ruoli distinti.

**Best practice:**  
In AI in education la trasparenza deve aiutare la decisione pedagogica, non esporre complessità tecnica fine a sé stessa. Le linee guida **UNESCO (2023)** sull'IA generativa in educazione e l'**Art. 14 EU AI Act** (sorveglianza umana) ribadiscono il principio di controllo umano significativo: il docente deve poter capire e correggere l'output, non subirlo.

**Decisione da prendere:**  
modello a 3 ruoli (docente / power user / admin) come scala di trasparenza standard? raccomandazione/suggestione: è la nostra proposta; se preferite una scala diversa - o esistono profili utente che non conosciamo - resta comunque un input di partenza su cui ragionare.

---

### 2.2 Flusso delle informazioni

**Punto emerso:** capire se gli utenti finali vedranno la progressione sequenziale degli agenti.

**La nostra risposta:**  
Sì, ma in forma sintetica e progressiva. La sequenza è importante perché spiega *come* nasce la lezione:

1. identificazione obiettivo/concetti;
2. recupero evidenze dal KG;
3. generazione;
4. revisione qualità.

Questo è coerente con i pattern moderni di "agent mode" e "deep research": l'utente non vede solo il risultato, ma anche i passaggi principali che rendono il risultato più affidabile.

**Raccomandazione:**  
Mostrare la sequenza, ma evitare jargon:

- usare nomi pedagogici, non nomi interni;
- collassare dettagli tecnici;
- evidenziare cosa serve al docente: obiettivi, copertura, fonti, qualità;
- non mostrare raw query salvo in sezione "Dettagli".

**Decisione da prendere:**  
La UI finale deve essere più "process-oriented" o più "output-oriented"? La nostra raccomandazione/suggestione è un compromesso: processo visibile ma non invasivo.

---

### 2.3 Sovraccarico visivo

**Punto emerso:** ridurre overload nella presentazione delle lezioni e dei contenuti.

**La nostra risposta:**  
Il rischio è reale. La WebUI attuale mostra molto perché è utile in fase di prototipo, debugging e confronto. Nel prodotto finale va applicata una gerarchia più forte. La teoria del carico cognitivo (Sweller) e il pattern di *progressive disclosure* (Nielsen Norman Group) anche utilizzata da OpenAI, indicano la direzione: mostrare prima ciò che serve alla decisione, rendere il resto espandibile.

**Pattern già presenti utili:**

- card agentiche progressive;
- sezioni `<details>` collassabili;
- pannello risorse laterale;
- bozza writer collassata;
- critic details espandibili;
- legenda fonti/verificato/auto.

**Raccomandazione (confermare con design):**  
La pagina dovrebbe avere tre livelli:

1. **Lezione finale come centro dell'esperienza.**
2. **Segnali di fiducia visibili ma compatti**: copertura KG, fonti, critic verdict, badge verificato/auto.
3. **Dettagli agentici espandibili** per chi vuole capire di più.

La calibrazione fine di cosa mostrare è un lavoro di design: è proprio qui che interviene l'expertise di un designer UX/UI, che decide cosa aggiungere o togliere per bilanciare chiarezza e completezza, in base agli input condivisi.

**Decisione da prendere:**  
Quali informazioni devono essere sempre visibili e quali solo on-demand? in base a docente standard/ power user/ admin  - raccomandazione/suggestione: sempre = lezione + segnali di fiducia compatti (copertura KG, fonti, critic verdict, badge verificato/auto); on-demand = query, concetti raw e dettagli per-agente.

---

### 2.4 Scheda Risorse

**Punto emerso:** la scheda risorse ha ricevuto feedback positivi.

**La nostra risposta:**  
Concordiamo: il pannello "Risorse multimediali" è uno degli elementi più forti del prodotto perché collega la lezione a risorse concrete e rende visibile la base informativa del sistema.

Oggi il pannello unisce due tipi di risorse: quelle **curate**, selezionate dalla knowledge base curata e rivista da esperti (badge `✓ Verificato`), e quelle **dinamiche**, recuperate automaticamente in tempo reale in base alla richiesta del docente (badge `auto`).

**Raccomandazione:**  
Mantenere se possibile il pannello risorse come componente chiave del prodotto. Non trattarlo come un extra, ma come parte della fiducia: fonti, media, approfondimenti e materiali collegati.

---

## 3. Onboarding e Profili Utente

### 3.1 Profilo educativo

**Punto emerso:** il profilo educativo è considerato utile.

**La nostra risposta:**  
Il profilo educativo è uno dei principali vantaggi competitivi del sistema. Permette all'AI di generare non una lezione generica, ma una lezione adattata a classe, livello, tempo, BES/DSA e obiettivo didattico. Questo è coerente con i principi dello **Universal Design for Learning (UDL, CAST)**: progettare per la variabilità degli studenti fin dall'inizio, non adattare a posteriori.

**Cosa raccoglie oggi:**

- ✅ materia;
- ✅ argomento specifico;
- ✅ dominio KG;
- ✅ tempo disponibile;
- ✅ livello scolastico;
- 🔧 numero studenti;
- ✅ BES/DSA e bisogni educativi;
- 🔧 caratteristiche classe;
- 🔧 attributi studenti;
- 🔧 ambiente aula;
- ✅ intento pedagogico scelto in chat.

---

### 3.2 Domande di onboarding e materia d'insegnamento

**Punto emerso:** chiarire scopo delle domande iniziali, in particolare le materie d'insegnamento.

**La nostra risposta:**  
Le domande iniziali non servono solo a "profilare" l'utente. Servono a migliorare tre aspetti:

1. **Rilevanza pedagogica:** livello, BES/DSA, tempo e obiettivo cambiano la struttura della lezione.
2. **Retrieval dal Knowledge Graph:** materia e argomento guidano ricerca, concetti e fonti.
3. **Controllo dell'output:** il sistema può verificare se sta rispondendo al contesto giusto.

L'enfasi su obiettivo e intento didattico è coerente con il **Backward Design** (Wiggins & McTighe): si parte dal risultato di apprendimento desiderato e si progetta a ritroso.

**Decisione da prendere**  
La materia deve essere obbligatoria nel profilo utente generale, nella singola lezione, o entrambe? Raccomandazione/suggestione: obbligatoria solo quando serve valore reale, non come barriera iniziale.

---

### 3.3 Utenti non insegnanti

**Punto emerso:** definire gestione di utenti che non sono insegnanti.

**La nostra risposta:**  
Gli utenti finali sono i docenti: il sistema è teacher-facing by design (documentazione regolatoria, modelli dati e UX assumono un docente o un professionista educativo). Accanto al docente c'è un solo ruolo interno, l'**admin**; per il dettaglio di cosa vede ciascun ruolo (docente / power user / admin) si veda §2.1. Non è invece progettato oggi come esperienza per studenti, genitori o utenti generici (almeno, a conoscenza del Team AI ad oggi)

**Perché è importante:**  
Cambiare pubblico cambia:

- tono dell'output;
- responsabilità;
- sicurezza;
- privacy;
- livello di spiegazione;
- rischio regolatorio;
- metriche di successo.

**Raccomandazione:**  
Non aprire la stessa esperienza a utenti non insegnanti senza definire use case separati. Più che un limite, è un'opportunità: ognuno di questi pubblici è una potenziale nuova applicazione dell'AI per l'ecosistema AixLearning, ma richiede use case, tono, privacy e responsabilità dedicati - non si riusa l'esperienza docente così com'è. Possibili futuri profili:

- progettista didattico;
- tutor/educatore;
- studente (solo con esperienza dedicata e limiti diversi);
- genitore (solo se il prodotto lo richiede).

**Decisione da prendere:**  
Quali ruoli utente sono nel perimetro della prima release? Raccomandazione/suggestione: partire da docenti/power user, poi estendere.

---

## 4. Output e Contenuti Generabili

### 4.1 Formato di output: il testo resta ottimale?

**Punto emerso:** chiedersi se il testo rimane formato ottimale alla luce della nuova architettura.

**La nostra risposta:**  
Sì: il testo/Markdown resta il formato **canonico e "universale"** delle AI conversazionali. È di fatto la lingua franca degli LLM - modificabile, esportabile, interoperabile e accessibile ; quindi non un compromesso temporaneo, ma la base naturale dell'output ora e in prospettiva.

**Nessun limite "tecnico" al formato:**  
Le nuove generazioni di LLM sono multimodali, perciò in teoria non c'è limite ai formati che un agente può generare. Il punto non è "testo *o* altro": il testo resta la spina dorsale e gli altri formati (video, quiz, mappe, slide) lo **accompagnano dove serve**, secondo gli obiettivi didattici e la richiesta del docente.

**In pratica:**  
Il backend produce la lezione in testo come sorgente, e il frontend la trasforma in esperienze diverse:

- piano lezione;
- scheda attività;
- scaletta;
- quiz;
- slide;
- mappa concettuale;
- risorse multimediali.

**Raccomandazione:**  
Mantenere testo/Markdown come "source of truth" e aggiungere output derivati in modo modulare ; coerente con lo standard di interoperabilità del testo e con un uso dei media "quando servono", non decorativo.


---

### 4.2 Contenuti non testuali: immagini, slide, quiz

**Punto emerso:** serve strategia per contenuti non testuali.

**La nostra risposta:**  
L'architettura agentica è adatta a contenuti non testuali, ma conviene introdurli come **tool/agent specializzati** e non mescolarli subito nella generazione principale.

**Dove siamo oggi:**  
- 🔧 scaffolding e moduli per media/diagrammi (Mermaid, image generator, Canva placeholder) presenti, ma **non ancora collegati** al flusso di generazione della lezione;
- ✅ pannello risorse già parte dell'esperienza;
- 📋 azioni post-generazione (quiz, slide, mappa, immagine) non ancora implementate.

**Strategia consigliata:**

1. Stabilizzare la lezione testuale + risorse.
2. Aggiungere "azioni post-generazione" (già in agenda del Team AI, a livello di WebUI):
   - genera quiz dalla lezione;
   - genera slide;
   - genera mappa concettuale;
   - genera immagine/diagramma;
   - genera scheda esercizi.
3. Far passare gli output non testuali da revisione/controllo qualità.

**Best practice:**  
In educazione, contenuti non testuali devono essere accessibili, modificabili e coerenti con obiettivi didattici. I principi della *multimedia learning* (Mayer) ricordano che testo e immagini funzionano se integrati e coerenti: non basta generare immagini belle, devono ridurre il carico cognitivo e servire al task pedagogico.

il team AI fornirà un documento strategico per la gestione dei contenuti non testuali

---

### 4.3 Concetti chiave

**Punto emerso:** evidenziare i singoli concetti chiave è considerato importante.

**La nostra risposta:**  
Sì, è un punto molto rilevante. Oggi i concetti chiave sono già individuati e mostrati nelle card Planner/Retriever, ma non abbiamo ancora una vera esperienza di highlighting dentro la lezione finale.

**Perché è importante:**

- aiuta il docente a capire il focus concettuale;
- rende visibile il legame con il Knowledge Graph;
- supporta ripasso e metacognizione;
- può alimentare mappe, glossary, quiz e approfondimenti.

**Raccomandazione:**  
Introdurre progressivamente:

1. chips concetti chiave sopra la lezione;
2. glossario concetti;
3. highlight nella lezione;
4. collegamenti a KG / risorse;
5. suggerimenti "cosa esplorare dopo".

**Decisione che spetta al Product Team:**  
Vogliamo che i concetti chiave siano un elemento di navigazione, di spiegazione, o entrambi?

---

## 5. Sintesi per il Product Team

| Tema | Punto fermo | Implicazione product |
|---|---|---|
| Architettura | Non è solo un nuovo frontend: la UX deriva da una pipeline AI multi-agente. | La UI finale deve decidere quali segnali del processo rendere visibili al docente. |
| Utente target | Il sistema è pensato come esperienza teacher-facing. | Il docente deve restare in controllo e poter correggere/adattare l'output. |
| Trasparenza | La trasparenza è un valore di prodotto, non solo tecnico. | Mostrare alcuni passaggi aiuta fiducia, AI literacy e controllo umano. |
| Complessità visiva | Non tutto ciò che esiste nell'architettura va mostrato sempre. | Serve progressive disclosure: vista semplice, dettagli espandibili, debug interno. |
| Profilo educativo | Il profilo educativo è una leva forte di personalizzazione. | Va mantenuto, ma semplificato per non sembrare un form burocratico. |
| Risorse | La scheda risorse collega lezione, fonti, media e fiducia. | È un componente strategico, non un extra laterale. |
| Output | Il testo/Markdown resta il formato canonico iniziale. | Può diventare la base per quiz, slide, mappe, immagini e attività. |
| Costi | L'agenticità aumenta i passaggi, ma può aumentare qualità e grounding. | Servono budget, modelli differenziati, metriche e una policy di pricing/abbonamento. |
| Pilot | Il pilot deve restare controllato. | Prima user testing con power user, poi hardening produzione. |
| Decisioni aperte | Molte scelte sono product, non solo tecniche. | Visibilità agenti, ruoli utente, output prioritari e provider ammessi vanno decisi insieme. |

---

## 6. Decisioni Product da Chiudere Dopo il Meeting

1. Perimetro utenti della prima release.
2. Livello di visibilità degli agenti.
3. Campi minimi del profilo educativo.
4. Output prioritario oltre alla lezione testuale.
5. Ruolo del pannello risorse nella UX finale.
6. Policy provider/privacy per pilot e produzione.
7. Budget/latency target per generazione.
8. Metriche dello user testing.
9. Roadmap: cosa entra nel prototipo aggiornato di Simone e cosa resta futuro.

---

## 7. Sintesi Finale

Agentic GraphRAG introduce una logica diversa rispetto ai precedenti prodotti AI: non è un unico prompt, ma una catena controllata di agenti specializzati. Questo cambia anche il frontend: la UX non deve solo raccogliere input e mostrare output, ma decidere **quanta parte del processo rendere leggibile al docente**.

La nostra raccomandazione è non trasformare la complessità architetturale in complessità visiva. Il prodotto dovrebbe mostrare i segnali che generano fiducia:

- obiettivo didattico;
- concetti chiave;
- fonti e risorse;
- copertura del Knowledge Graph;
- revisione pedagogica;
- distinzione verificato / auto.

Il resto deve rimanere espandibile o interno. In questo modo l'architettura agentica diventa valore di prodotto, non rumore.

---

## 8. Riferimenti a best practice e ricerca

Le risposte di questo documento si appoggiano a framework riconosciuti di IA in educazione, design dell'interazione e regolazione:

- **UNESCO (2023)** — *Guidance for generative AI in education and research*: controllo umano, agency del docente, uso responsabile. → 2.1, 2.2, 3.3
- **EU AI Act** — Art. 14 (sorveglianza umana significativa), Art. 50 (obbligo di marcare i contenuti generati da IA, applicazione 2 agosto 2026). → 1.3, 2.1
- **GDPR** — minimizzazione dei dati e profilo educativo aggregato (no dati identificativi di studenti). → 1.3
- **CAST — Universal Design for Learning (UDL)**: progettare per la variabilità degli studenti (BES/DSA, livello, contesto). → 3.1, 3.2
- **Wiggins & McTighe — Backward Design**: partire dall'obiettivo di apprendimento e progettare a ritroso. → 3.2
- **Mayer — Multimedia Learning principles**: testo e contenuti visivi efficaci solo se integrati e coerenti con l'obiettivo. → 4.2
- **Sweller — Cognitive Load Theory** e **Nielsen Norman Group — Progressive Disclosure**: ridurre il sovraccarico mostrando prima l'essenziale, rendendo il resto espandibile. → 2.1, 2.2, 2.3

> I riferimenti orientano le decisioni di prodotto, non sono vincoli rigidi: l'obiettivo è coniugare evidenza pedagogica, conformità normativa e usabilità.
