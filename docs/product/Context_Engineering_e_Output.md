# Context engineering e nuovo output — analisi e misure (13 ottobre 2026)

Obiettivo: **dimezzare il costo di generazione** senza perdere qualità, e rendere
l'output più utile per la docente. Tutti i numeri sono misurati su chiamate vere
(tracce Langfuse, replay delle stesse richieste, lezioni rigenerate sullo stack
locale); le decisioni sulla qualità sono state prese con letture alla cieca di
Angelo, perché il Critic automatico non distingue le lezioni buone da quelle che
in classe non si userebbero.

## 1. Dove siamo

| | costo per turno |
| --- | --- |
| 30/9–6/10, Sonnet 5 senza correzioni | lezione completa $0,17–0,31 |
| 7/10 mattina, cache di Cypher e giudice | lezione completa $0,08–0,17 |
| 13/10, Sonnet 5.5 · sforzo `low` + cache | lezione completa su argomento del grafo **$0,05–0,07**; lezione leggera $0,03–0,05 con gli strumenti accesi |

Le due cose che hanno pesato di più non erano di context engineering:

- **Sforzo di ragionamento.** `AIX_THINKING_EFFORT=low` arrivava solo a Claude 4:
  Sonnet 5 ragionava senza limiti (~20% della spesa) e troncava le lezioni. Con
  `low` le stesse chiamate costano il 30–40% in meno e ci mettono metà del tempo.
- **Modello.** Sonnet 5.5 · `low` è la versione preferita alla lettura alla cieca,
  allo stesso prezzo di Sonnet 5. Sonnet 5 · `low`, Mistral Large 4 e Haiku 4.5
  sono stati scartati (lezioni non presentabili); Gemini 3.5 Flash è presentabile
  ma costa solo il 15–25% in meno.

## 2. I cinque principi

### Prompt caching — applicato
- Schema Cypher per dominio (~22–27k token, identico a ogni domanda): **-88%** dal
  vivo. Su Gemini la cache implicita non scatta: serve il segno esplicito.
- Istruzioni del giudice delle citazioni: **-46%** per chiamata, qualità invariata
  (le differenze tra le run sono pari al rumore del modello).
- System di Writer, revisione e planner; testa della lezione leggera.
- **Limite misurato:** OpenRouter serve Sonnet 5.5 da più istanze (Google,
  Bedrock), ciascuna con la sua cache. Con richieste diverse la lettura riesce
  **4 volte su 8** per una docente sola (-34% sul system del planner); sale con più
  docenti attive. Il segno conviene sopra ~22% di letture.
- Cache dei risultati: risposte del grafo (24 h, chiave anche su
  `AIX_KG_VERSION`) e verdetti del giudice (7 giorni).

### Compressione con stato — in parte
- Fatto: lo stato salvato non contiene più `graphrag_results`, una copia per
  ricerca di nodi e relazioni che nessuno leggeva.
- Misurato, non ancora fatto: la **storia della conversazione** arriva 4–5 volte
  nello stesso turno (planner ×2, Writer, Critic) e il riassunto dei turni vecchi
  è ricalcolato da Sonnet a ogni turno. Nel replay, togliere le ripetizioni e
  compattare il grafo vale **-10%** sul Writer (vedi §4); attende la lettura alla
  cieca.

### Divulgazione progressiva — misurata
- I documenti allegati erano già a livelli (mappa, poi passaggi).
- **Lezione leggera progressiva** (solo le sezioni pertinenti): toglie solo il 5%
  dell'input, perché la lezione è il 21% di un turno leggero. Non attivata.
- **Prima la scaletta** (prototipo): la scaletta da sola costa il 36% in meno e
  arriva in 9 s invece di 25; scaletta + una sezione costa il 64% in più della
  lezione intera, perché il secondo passo rimanda tutto l'input. → l'output
  incrementale conviene **solo** se ogni passo riceve il contesto della sua parte.

### Scelta degli strumenti — applicata alla lezione leggera (spenta, in prova)
- Nei turni leggeri **i passaggi dei documenti sono il 70% dell'input** e venivano
  cercati prima di ogni risposta. Con `AIX_LIGHT_TOOLS_ENABLED` il modello ha
  `cerca_nei_documenti` e `cerca_nel_grafo` e li usa solo se servono.
- Su 5 richieste tipiche della stessa lezione: **-25%** in totale; le domande che
  non riguardano i documenti **-36%…-77%** e primo token in 0,9–3,6 s invece di
  5–10 s; la domanda sui documenti +42% (un giro di ricerca e più citazioni).
- Da fare: provarla nella WebUI; esporre gli stessi strumenti dal server MCP per i
  client esterni.

### Context graph — applicato (spento, in prova)
- Le relazioni del grafo (es. «ADHD —SUGGESTS→ pause attive») erano recuperate ma
  mai date al Writer. Con `AIX_WRITER_GRAPH_RELATIONS` arrivano come ≤15 righe.
- Su 4 lezioni rigenerate: Writer **+15%** di costo, Critic uguale in 3 casi su 4.
  Vale solo se migliora le lezioni: lettura alla cieca in attesa.

## 3. Altre tecniche

| tecnica | esito |
| --- | --- |
| Sforzo di ragionamento per modello (Claude 5.x, Gemini Flash/Pro) | fatto: -30/40% Sonnet, -18% Gemini Flash, niente troncamenti |
| Giudice: una domanda ripetuta chiesta una volta | fatto: -9% delle chiamate del giudice |
| Giudice su modello economico | scartato: su 49 coppie vere Gemini flash-lite lascia passare 7 citazioni sbagliate su 8 |
| Planner su Gemini 3.5 Flash | scartato: intento 27/30, ricerche nel grafo sovrapposte al 41% contro il 68% di Sonnet con se stesso, più lento |
| Schema Cypher più corto | scartato: ogni variante cambia i risultati (38–57% contro un rumore dell'85–91%) |
| Revisione solo quando serve | poco valore: le revisioni vere sono 3 turni su 24 |
| API batch (-50%) per lavori non interattivi (mappe, OCR) | da valutare |
| Parti fisse generate da codice (Fonti, Materiali) | da valutare |
| Conferma dell'intento prima di generare | esiste come card dell'intento; da estendere (sotto) |

## 4. Il nuovo meccanismo di output — proposta

Oggi ogni turno completo scrive una lezione intera (~7.000 caratteri, sempre le
stesse sezioni) che la docente deve leggere tutta. Le misure dicono che si può
spendere meno **e** dare di più solo se ogni passo riceve il contesto della sua
parte. La proposta, in tre passi:

1. **Capire il bisogno.** La card dell'intento esiste già (6 opzioni). Va estesa
   a "che cosa ti serve adesso": la lezione intera, solo la scaletta, un'attività,
   una verifica, gli adattamenti per BES, una scheda per gli studenti. Ogni tipo
   ha il suo formato e la sua lunghezza.
2. **Scaletta prima.** Titolo, obiettivi, sezioni con minuti e una riga ciascuna,
   più 3–4 proposte "cosa sviluppo adesso?". Misurato: -36% di costo, 9 s, un
   terzo della lettura.
3. **Sezioni su richiesta, con contesto mirato.** Ogni sezione viene scritta con la
   scaletta + i soli materiali che la riguardano, cercati con gli strumenti (§2):
   senza questo il secondo passo costa più della lezione intera. La modalità a
   sezioni (branch `feature/section-edits-optional`, opzione sperimentale) dà già
   la struttura per salvarle, modificarle e annullarle.

Effetto atteso: la docente legge solo ciò che le serve; il costo scende quanto più
spesso si ferma alla scaletta o a una o due sezioni. Da misurare con docenti vere
(quante sezioni chiedono in media) prima di renderla il comportamento predefinito.

## 5. Decisioni aperte

1. Letture alla cieca in attesa: Writer asciutto, prima la scaletta, relazioni del
   grafo nel Writer.
2. Strumenti nella lezione leggera: provarli nella WebUI e accenderli.
3. Modalità a sezioni come opzione (branch separato).
4. Storia una volta sola e riassunto salvato (dopo la lettura del Writer asciutto).
5. Esporre gli strumenti dal server MCP.
