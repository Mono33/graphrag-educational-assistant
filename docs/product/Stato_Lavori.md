# Stato dei lavori — Advanced Input & Export

Branch: `feature/advanced-input-and-export` · Aggiornato: 2026-09-16 (`bfd172a`, **non ancora pushati**: `e3de068`, `6f0664c`, `8fffca8`, `bfd172a`; `b6f7192` è su origin)

Questo file ha un solo scopo: in qualunque momento si deve poter leggere **cosa
è fatto**, **cosa manca** e **cosa aspetta qualcun altro**, senza ricostruirlo
dal `git log` o dalla memoria di una conversazione.

**Come si tiene.** A ogni passo che chiude qualcosa — di norma un commit — si
sposta la voce da «Da fare» al «Registro», con la data e l'hash. Il registro si
scrive in coda, non si riordina. Se un lavoro si scopre più grande di come era
stato scritto, si spezza qui prima di iniziarlo, non dopo.

I riferimenti alle PR sono quelli della guida di Louis
(`Advanced_Input_and_Export_Architecture_v2-1_Angelo.md`); i riferimenti a T-*
sono quelli del piano di lavoro.

---

## In una riga

**Fase 0 completa, e la Fase 1 è partita: PR 4 fatta.** Il grafo è risanato e
testato, le migrazioni ci sono, due difetti che erano in produzione sono chiusi,
il retrieval condizionale funziona davvero e si vede nell'interfaccia, e lo
strato che regge input e output — storage durevole, materiali versionati,
accesso per proprietario, coda di job — esiste ed è testato.

Quello strato non è più inerte: dal 2026-09-11 l'upload lo alimenta **in
ombra**. Con `AIX_MATERIALS_ENABLED` acceso ogni file caricato entra anche nella
pipeline nuova, senza che la docente veda niente di diverso. Il passo dopo è
usarlo davvero.

---

## Da fare

### Subito

Dal 2026-09-14 c'è un piano completo fino al tempo di test — blocchi **B0–B8**,
in `C:\Users\Angelo\.claude\plans\swirling-imagining-tiger.md`. In sintesi:

1. **B0 — due risposte arrivate il 2026-09-14.** FEM **ha** un dominio Google
   Workspace for Education: Slides e Classroom si potranno testare con consenso
   *Internal*, quindi senza audit di Google e senza i refresh token che scadono
   ogni 7 giorni. E il **corpus lo generiamo noi**, come insieme di esempio —
   è il prossimo lavoro concreto, perché senza documenti veri il benchmark non
   misura niente. Restano il template Slides e le decisioni UX U1–U4. La policy
   per l'embedding è chiusa il 2026-09-15: OpenRouter con Zero Data Retention.
2. **B1 · Q1 — chiuso** (`628baaa` + `fb95e51`): il profilo intero arriva ai
   prompt, e il validator deterministico separa «è fattibile» da «è bella».
3. **B3 · PR 6 — fatto** (`090ffab`, `e3de068`, `6f0664c`): RAG privato, dietro
   `AIX_PRIVATE_RAG_ENABLED` (spenta).
4. **B2 · PR 5 — in corso.** Parsing e OCR, un parser per formato dietro la sua
   flag. Diviso in due: prima il parsing puro (IR, controllo dei file in
   ingresso, parser, OCR) — **fatto, `387034a`** — poi il collegamento
   all'upload e la persistenza — **fatto, `730fed2`**. Resta: corpus derivato
   (DOCX, PPTX, foto) con script e manifest; Tesseract su Windows per chi
   sviluppa in locale.

### 2026-09-15 — Parsing e OCR, prima metà di PR 5 (`387034a`)

Nuovo pacchetto `src/aix/webui/materials/parsing/`: `intake` (se un file si può
leggere), `parsers` (uno per formato, ognuno con la sua flag `AIX_PARSE_*`),
`ocr` (Tesseract, solo sulle pagine senza text layer), `ir` (`DocumentIRV1`, con
locator e confidenza su ogni blocco).

**Il controllo in ingresso guarda i byte, non il nome.** Rifiutati: doppie
estensioni attive (`.pdf.exe`), HTML e SVG, contenuto che non corrisponde
all'estensione, Office con macro anche se rinominato `.docx`, vecchi `.doc`,
PDF con password vera, ZIP che si espandono in modo anomalo, oltre 200 pagine.
Un PDF corrotto è un rifiuto con un messaggio in italiano, non un 500.

**Il corpus vero ha corretto due scelte che i test sintetici non vedevano:**

1. Il DM 254/2012 veniva rifiutato come «protetto da password». È cifrato solo
   contro la modifica, e si apre con password vuota: capita spesso nei PDF
   ministeriali. Ora si accetta; resta rifiutato chi chiede davvero una password.
2. Il limite di 60 pagine scartava due documenti ministeriali digitali (68 e 97
   pagine), che si leggono in 2 secondi. Il costo sulla VM è l'OCR, non la
   lettura: il limite in ingresso sale a 200 pagine, e l'OCR ha un suo tetto di
   30 pagine per documento, oltre il quale le pagine restano non lette con un
   avviso.

**Misurato in un container Linux con `--memory=4g --cpus=2`**, lo stesso profilo
della VM, con Tesseract 5.3 e il pacchetto italiano:

| Documento | Pagine | Tempo | Esito |
| --- | --- | --- | --- |
| Indicazioni nazionali, allegato | 97 | 2,5 s | tutto text layer |
| DM 254/2012 | 68 | 1,7 s | tutto text layer |
| Presentazione linee guida PEI | 38 | 0,2 s | tutto text layer |
| Modello PEI | 2 | 0,0 s | tutto text layer |
| Scansione Vergani, pp. 21–28 | 8 | 7,0 s | 8 pagine OCR, confidenza media 0,93 |

Picco di memoria: 54 MB Python, 55 MB Tesseract. L'OCR su un libro italiano
dell'Ottocento sbaglia pochissimo («svilupperd» per «svilupperà»). Timore della
memoria ridimensionato: su pagine di questa taglia l'OCR non è il rischio.

**Verificato:** 35 test nuovi; in Linux passano tutti, OCR compreso; su Windows
33 più 2 saltati perché Tesseract non c'è. Suite completa: 607 passati, i 2
fallimenti noti di `test_concurrency_limits` su Windows.

### 2026-09-15 — L'upload della chat usa il parser vero (`730fed2`)

Il docente può allegare dalla chat **PDF, Word, PowerPoint, testo e immagini**,
anche scansioni. Il file passa dal controllo in ingresso e dal parser, in un
thread perché l'OCR non blocchi le altre chat.

- **Errori leggibili.** File rifiutato o danneggiato: 422 con un messaggio in
  italiano. Il PDF corrotto che dava 500 nel percorso legacy è chiuso.
- **La chip dice cosa è successo:** «OCR 8 pag.», «3 pag. non lette», «bassa
  qualità», «nessun testo leggibile», con bordo di avviso. Le voci del manifest
  hanno una nuova chiave `parse`; quelle vecchie si vedono come prima.
- **Il Writer sa quando il testo viene dall'OCR**: il titolo dell'allegato lo
  dichiara (§7.8), così nomi e numeri non vengono citati come certi.
- **Worker su `ir@1`.** Con `AIX_MATERIALS_ENABLED` acceso salva il risultato
  strutturato nella nuova tabella `document_ir` (migrazione `5d1c7a2e9b40`, SQL
  PostgreSQL verificato offline) e il riassunto in `parse_quality`. I job
  `pypdf@1` già in coda restano distinti.
- **Dockerfile:** `tesseract-ocr` con `ita` ed `eng`, niente OCRmyPDF né
  Ghostscript.
- `benchmark_parsers.py` ora conta pagine con text layer, OCR e non lette, e
  mostra il codice di rifiuto.

**Verificato:** 611 test passati (i 2 noti su Windows); server locale avviato,
migrazione applicata al DB di sviluppo, `/api/v1/health` e `/webui/` a 200.
**Non verificato a mano:** l'upload dal browser — tocca ad Angelo. Su Windows
senza Tesseract le immagini e le scansioni risultano «non lette».

### 2026-09-15 — Primo test reale dal browser, e due correzioni (`9795d9e`)

Angelo ha caricato `guerra-fredda-def.pdf` (3 pagine, dispensa di storia) e
generato una lezione con `AIX_MATERIALS_ENABLED` spento. **L'upload funziona:**
3 pagine con text layer, nessun avviso, lezione approvata dal Critic (4,6) e
costruita sul documento. Il doppio «run accepted» nel log non è un bug: è il
flusso a due fasi (Planner e scelta dell'intento, poi il run completo).

**Due difetti trovati nei log, corretti:**

1. **Il Planner non vedeva gli allegati** — per progetto li riceveva solo il
   Writer. Ha dichiarato «contenuto del documento allegato (non disponibile per
   l'analisi)» e ha cercato quella frase su Wikipedia (trovando «Documenti
   Google»), YouTube e OpenAlex. Ora riceve un estratto breve di ogni allegato
   (1500 caratteri per file, 4500 in tutto), sia nel run completo sia nella fase
   della scheda d'intento, con l'istruzione di non trattarlo come mancante.
2. **L'estratto per file era tagliato a 8000 caratteri.** La dispensa ne ha
   9985: si è perso l'ultimo quinto — distensione, Reagan, Gorbaciov, 1989 — e
   la lezione non li nomina. Il limite per file sale a 24.000; resta 48.000 in
   totale.

**Da osservare al prossimo test:** la lezione ha saltato anche parti presenti
nell'estratto (Corea, Bandung, muro di Berlino). Può essere una scelta del
Writer per stare nei 60 minuti; da rivalutare dopo le due correzioni.

**Verificato:** 617 test passati (i 2 noti su Windows). Server locale riavviato.

### 2026-09-16 — Secondo test reale: la correzione funziona, due difetti di flusso (`1da930e`)

Stessa dispensa, lezione nuova, profilo con materia «Storia», argomento «guerra
fredda», intento «Correggere un'idea errata», aula con LIM, WiFi e BYOD.
**La correzione di ieri si vede nei log:** il Planner chiede fonti esterne per
«Guerra Fredda, Conferenza di Yalta 1945, cortina di ferro, Piano Marshall, NATO
e Patto di Varsavia» — ieri cercava «documento allegato». La lezione prodotta usa
il documento (Corea, Vietnam, Cuba, propaganda, Yalta) e cita Sciacca come fonte.

**Due difetti trovati da Angelo, corretti:**

1. **La scheda dell'intento chiedeva una cosa già decisa.** `pedagogical_intent`
   era già `misconception` nel profilo, ma la fase 1 guarda solo se la lezione è
   nuova. Risultato: generazione interrotta a metà e una chiamata al Planner
   sprecata — 78 secondi fra il primo `run accepted` (09:40:22) e il secondo
   (09:41:40). Ora, se l'intento è nel profilo, si va dritti al run completo.
2. **La scheda del Planner non nominava la guerra fredda.** Mostrava solo i
   `key_concepts`, che per costruzione sono strategie didattiche cercate nel KG
   (il KG è di pedagogia, non di storia). L'argomento stava in `subject_concepts`
   e non era mostrato da nessuna parte. Ora la scheda ha due righe distinte:
   «Argomento della lezione · dal documento e da fonti esterne» e «Strategie
   didattiche cercate nel Knowledge Graph».

**Non è un difetto:** la lezione usa Mentimeter, Kahoot e dispositivi personali —
l'aula dichiara BYOD, WiFi e LIM, e il validatore dice `feasible (0 warning)`.

**Verificato:** 623 test passati (i 2 noti su Windows), 6 nuovi sul flusso
dell'intento e sulla scheda.

### 2026-09-16 — Turno a vuoto: il difetto peggiore trovato finora (`803262d`)

Terzo test: secondo documento allegato (`Cold-War.pdf`, 7 pagine) e richiesta di
ampliare la lezione. **Due turni di fila, quattro minuti, zero caratteri scritti.**

**Causa:** `AIX_WRITER_MAX_TOKENS=2000` nel `.env` di sviluppo (il default nel
codice è 8000) con un modello di ragionamento — `anthropic/claude-sonnet-5`. Con
il prompt più grande (due allegati, 33.985 caratteri, più il thread) il budget se
n'è andato tutto nel ragionamento: 231–260 «thinking chunks» e nessun testo. La
rete di sicurezza esisteva già ma **riprovava con lo stesso tetto**, cioè chiedeva
al modello di rifare esattamente ciò che non era riuscito a fare.

**Il difetto peggiore non è l'errore, è il turno che sembra riuscito.** Al turno 2
il run è finito `complete`, e nel thread è stato salvato come risposta un
messaggio lungo esattamente quanto la lezione precedente: `final_lesson_plan`
sopravvive nel checkpoint del thread, e nessuno controllava se *questo* turno
avesse scritto qualcosa. Il docente ha visto la stessa lezione di prima, senza
alcun avviso. Il turno 3 è invece fallito con un messaggio generico (incidente
C16017) che non diceva cosa fare.

**Tre difese, una per test:**

1. il riprova dopo una risposta vuota ha un pavimento di 4000 token, non il tetto
   che ha appena fallito;
2. una bozza vuota diventa un errore esplicito che **nomina la variabile da
   cambiare** (`AIX_WRITER_MAX_TOKENS`, consigliato 8000);
3. da uno stato che riporta un errore non si legge più nessuna lezione: la
   lezione precedente non può più diventare la risposta a una nuova richiesta.

**Da fare, non nel codice:** in `.env` di sviluppo alzare
`AIX_WRITER_MAX_TOKENS` da 2000 a 8000 (il valore che il codice usa di default).
Da rivedere anche il limite complessivo del contesto allegati (48k caratteri) ora
che un singolo file può portarne 24k: due allegati lunghi più il thread fanno un
prompt molto grande.

**Verificato:** 627 test passati (i 2 noti su Windows), 4 nuovi.

### 2026-09-16 — B3 · PR 6, prima metà: chunking ed embedding (`090ffab`)

Il muro dei token non si supera alzando i limiti: si supera non mettendo più il
documento intero nel prompt. Questa è la prima metà del RAG privato.

- **Chunker struttura-aware** (§8.3): pezzi da 250–450 token che **tengono il
  locator** (pagina/slide e block id, quello che una citazione deve poter
  aprire), il titolo sopra il testo che introduce, le tabelle intere, le slide
  mai mescolate, sovrapposizione del 12% fra pezzi della stessa sezione.
  `chunk@1` è scritto su ogni riga: cambiare le regole è una nuova versione e un
  ri-chunk, non un indice misto.
- **Embedding** `baai/bge-m3` via OpenRouter, **1024 dimensioni verificate
  contro il provider** prima di scriverle in una migrazione (una chiamata con la
  parola «prova»). §8.4 chiede un modello locale: su questa VM non ci sta, e la
  deroga — ZDR attiva, decisa il 2026-09-15 — è scritta nel modulo, non ereditata.
  La dimensione è controllata su **ogni** risposta: un modello che tornasse a
  1536 fallirebbe altrimenti nel database, lontano dalla causa.
- **`material_chunk`** (migrazione `9a4f2b7c61de`): su PostgreSQL `vector(1024)`
  più FTS italiano e indice GIN, su SQLite JSON — stesso modello ORM, grazie a un
  tipo che cambia per dialetto (30 righe, nessuna dipendenza nuova). SQL
  PostgreSQL verificato offline: estensione, tabella, conversione, indici.
- **Il worker indicizza** dopo aver salvato l'IR, così un errore del provider
  ripete l'embedding e non il parsing. Ri-eseguire il job **sostituisce** i chunk:
  accodarli raddoppierebbe il peso di ogni pezzo nel ranking.
- Tutto dietro `AIX_PRIVATE_RAG_ENABLED`, spento: a flag spenta niente chunk,
  niente embedding, nessun testo che esce dalla macchina.

**Seconda metà, da fare:** `PrivateMaterialRetriever` (§8.5, autorizzazione
*dentro* la query), `MaterialContextItem` `data_only` (§8.2), nodo parallelo al
GraphRAG, citazioni `[[material:…#page=7]]` (§8.6), budget di contesto separati
(§8.7) e il binding lezione-materiale. Solo allora l'allegato smette di essere
incollato nel prompt.

**Verificato:** 643 test passati (i 2 noti su Windows), 16 nuovi.

### 2026-09-16 — Review di Louis su `730fed2`: i bloccanti sono veri (`36d07ac`)

Louis ha rivisto il branch e conclude: **la Fase 0 non è chiusa**. Verificati i
suoi 12 punti uno per uno: **9 reali**, 1 già risolto (il test di topologia qui
passa, 32 test), 2 sono giudizi su cosa appartenga alla Fase 0.

**I tre bloccanti, corretti in `36d07ac`:**

1. **Un database di produzione non sarebbe mai migrato.** L'adozione confrontava
   lo schema vivo con i **modelli attuali**: siccome i modelli hanno `material`,
   `document_ir` e `material_chunk`, ogni database reale risultava «incompleto»,
   veniva rifiutato e restava fuori dal controllo delle migrazioni, con
   `alembic_version` vuota. Ora il confronto è con la **baseline**, e il database
   viene marcato lì e poi migrato in avanti.
   **Il test lo nascondeva**: costruiva il database «legacy» con
   `Base.metadata.create_all`, cioè con dentro le tabelle nuove — l'unica forma
   che in produzione non può esistere. Ora lo costruisce eseguendo la migrazione
   baseline. Due test nuovi: uno verifica che un database pre-branch arrivi a
   head **con** le tabelle nuove, l'altro che le righe della docente sopravvivano.
2. **A flag spente il prodotto non era quello di prima.** La graffetta accettava
   DOCX, PPTX e immagini e faceva OCR comunque. Ora il parser sta dietro
   `AIX_UPLOAD_PARSER_ENABLED` (spenta): senza flag si torna a PDF/TXT/MD letti
   con pypdf. Unica differenza voluta: un PDF corrotto resta un 422 con un
   messaggio, non un 500. L'`accept` del file picker e il tooltip ora **chiedono
   alla flag**, invece di tenere una lista propria.
3. **Il validatore era acceso di default** e bloccava una lezione che *nomina*
   DSA senza profilo. Ora è spento di default, e la regola `bes_label_drift` non
   si applica se non ci sono bisogni dichiarati: senza dichiarazione non c'è
   niente da cui derivare.

**Altri punti chiusi:** volume `app_materials` in compose e cartella
nell'immagine (i blob sopravvivono a un redeploy); PDF e scaletta dietro flag
(accese, come Louis consente) verificate anche nella rotta e non solo nel
bottone; **tutte** le flag nuove documentate in `.env.example`, che non ne aveva
nessuna.

**Restano aperti, nell'ordine in cui li farei:**

| # | Cosa | Perché |
| --- | --- | --- |
| 1 | `BindingRepository.bind` non verifica che la lezione sia dello stesso proprietario | latente finché il binding non si usa — cioè fino alla seconda metà di PR 6 |
| 2 | Documento illeggibile → `ready_with_warnings` invece di `failed` | uno stato che dice «pronto» per qualcosa che non si è potuto leggere |
| 3 | Il lease del job non si rinnova durante il parse; `complete()` può girare senza lease | un OCR lungo può far scadere il lease e far lavorare due worker |
| 4 | `GET /auth/logout` cambia stato | va POST |
| 5 | `POST /api/v1/agent/*` accetta il cookie della WebUI ed è fuori da CSRF | o CSRF anche lì, o non accettare il cookie |
| 6 | Nessun test su PostgreSQL (migrazioni e claim) | la Fase 0 li chiedeva su entrambi i dialetti |

**Verificato:** 648 test passati (i 2 noti su Windows), 5 nuovi.

### 2026-09-16 — Chiusi anche i sei punti minori della review (`b6f7192`)

1. **Il lease del job si rinnova mentre l'handler lavora** (a un terzo del
   lease). Prima una scansione di 40 pagine — minuti di OCR contro due minuti di
   lease — poteva essere riclamata da un secondo worker a metà strada, e
   entrambi scrivevano le stesse righe. Se `complete()` trova il lease perduto,
   ora lo dice nel log: è la traccia visibile di un lavoro fatto due volte.
2. **Il binding controlla anche la lezione**, non solo il materiale: altrimenti
   un docente poteva allegare un proprio documento alla lezione di un altro, e
   nessuna lettura filtrata per proprietario se ne sarebbe accorta.
3. **Logout è POST** sotto CSRF. Il `GET` ritirato è **dichiarato** nei due test
   di superficie additiva, con la ragione, invece di essere cancellato dal file
   di baseline: la baseline resta un registro di fatto, la rimozione è una
   decisione che si vede in un diff.
4. **`/api/v1` è protetto quando la richiesta viaggia sul cookie della WebUI.**
   L'esenzione diceva «autentica per richiesta»: non è più vero da quando le
   rotte agent accettano il cookie, che il browser allega a qualunque cosa una
   pagina terza chieda. Chi presenta `Authorization` (Bearer o Basic — servizi e
   MCP) non prende in prestito niente e resta fuori.
5. **Documento illeggibile → `failed`**, non `ready_with_warnings`: «pronto»
   inviterebbe a scriverci sopra una lezione.
6. **Primi test su PostgreSQL** (`tests/integration/test_postgres_dialect.py`,
   saltati senza `AIX_TEST_POSTGRES_URL`). Eseguiti contro
   `pgvector/pgvector:pg16`: **4 passati** — tutte le migrazioni si applicano, la
   colonna è davvero `vector(1024)` con `tsvector` italiano e indice GIN, un
   vettore fa andata e ritorno attraverso l'ORM, e `SKIP LOCKED` dà un job a un
   solo worker. Sono ~9 minuti: restano fuori dalla suite veloce.

**In `.env` di sviluppo** è stato aggiunto `AIX_UPLOAD_PARSER_ENABLED=true`, così
il test manuale continua con DOCX, PPTX e immagini mentre il default resta spento.

**Verificato:** 654 test passati (i 2 noti su Windows), 9 nuovi; 4 su PostgreSQL.

### 2026-09-16 — B3 · PR 6, seconda metà: il RAG privato è completo (`e3de068`, `6f0664c`)

Il documento non viene più incollato nel prompt: viene cercato.

- **`PrivateMaterialRetriever`** (§8.5). L'autorizzazione è **dentro** la query —
  proprietario e versioni legate a *questa* lezione — non un filtro applicato
  dopo: un filtro dopo è quello che una modifica futura dimentica, e su pgvector
  l'indice ANN filtra comunque dopo la traversata. Su PostgreSQL ordina con
  `<=>`, su SQLite calcola il coseno in Python sugli stessi chunk autorizzati.
- **Nodo parallelo al GraphRAG** (§8.1): `plan` apre i due rami, `write` li
  richiude. Entrambi scrivono `current_step` nello stesso superstep — il caso per
  cui i reducer esistevano — e il test costruisce il grafo vero, non lo ispeziona.
  La WebUI registra la ricerca per sessione (`material_source`), perché
  `aix.agent` non può importare `aix.webui`: un run da API o CLI non registra
  niente e mantiene la topologia di prima.
- **Il binding lezione-materiale ora si scrive**, perché adesso qualcuno lo
  onora: è esattamente l'insieme di versioni contro cui la ricerca è autorizzata.
- **Il Writer riceve i passaggi**, con un budget suo (§8.7, ~12k caratteri), il
  marcatore da ricopiare e l'avviso quando il testo viene da OCR. Se la ricerca
  non trova nulla — indice ancora in costruzione, flag spenta — l'allegato
  arriva al Writer come prima (§5.4, hybrid). Anche la revisione li riceve:
  perderli riscriverebbe la lezione lontano dal documento.
- **Citazioni leggibili** (§8.6): `[[material:…#page=7]]` diventa
  «(guerra-fredda.pdf, p. 7)». Un marcatore che nomina un documento che la
  lezione non può usare **sparisce** invece di diventare un riferimento rotto, e
  un indirizzo interno non arriva mai alla docente — né in chat né in export.
- Un fallimento della ricerca costa alla lezione il suo allegato, mai la lezione.

**Non ancora fatto, da §8:** il download autorizzato dell'originale e l'anteprima
di pagina (la citazione oggi nomina pagina e documento, non apre il file); il
Critic non distingue ancora il grounding sul materiale privato da quello sul KG.

**Verificato:** 688 test passati (i 2 noti su Windows), 34 nuovi fra chunking,
retrieval, grafo e citazioni.

### 2026-09-16 — Le variabili d'ambiente, misurate (`8fffca8`, `bfd172a`)

`.env.example` aveva **95 variabili, 65 delle quali commentate**, e ne mancavano
**24 che il codice legge davvero** — fra cui `WEBUI_CSRF_ENABLED`,
`AIX_DB_AUTO_MIGRATE`, tutto il blocco S3 e tre aggiunte da noi la settimana
scorsa. Un file che sembra autorevole e non lo è costa più di un file assente.

Misurato leggendo il sorgente, non a memoria: **98 variabili lette**, di cui solo
**5 morte** (`API_CONTAINER_NAME`, `API_CONTAINER_PORT`, `NODE2VEC_MODEL_DIR`,
e `LMSTUDIO_*` che usano solo due script).

- `.env.example`: **da 574 righe a 87**. Le sei obbligatorie, le tre da decidere
  per ambiente, le flag e le chiavi opzionali.
- **Nuovo `docs/ENVIRONMENT.md`**: tutte e 98, con il default vero e cosa succede
  se non la imposti. Verificato meccanicamente: copertura completa.
- `deploy/.env.prod.example` non aveva **nessuna** delle flag nuove: aggiunte con
  gli avvisi che meritano (volume dello storage, testo che esce dalla VM per
  l'embedding, REINDEX di pgvector).

**Il segreto di sessione.** `WEBUI_AUTH_SECRET` firma cookie di sessione, link di
reset, link di verifica e i token CSRF. Se non è impostato il codice usa una
costante **pubblicata in questo repository** e lo dice in un warning — una riga
che un deploy scorre via. In produzione **è impostato** (confermato da Angelo:
nessun buco aperto). Ora un ambiente che si dichiara `production`, `prod`,
`staging` o `pilot` **si rifiuta di partire** sul ripiego; su un portatile o con
`ENVIRONMENT` non impostata resta il warning, perché un controllo che scatta
sulla macchina di chi sviluppa insegna solo a mettere un valore qualsiasi.

**Verificato:** 701 test passati (i 2 noti su Windows), 13 nuovi.
4. **B4 · PR 7 — fatto** (`f8150e1`): PDF vero e scaletta con i minuti.
5. **Corpus — avviato il 2026-09-15.** Cinque PDF pubblici del Ministero e una
   scansione reale da Internet Archive sono in `data/eval/corpus/originali/`
   (esclusi da git). I quattro documenti ministeriali sono digitali al 100%. La
   scansione ha **355 pagine** (non 50: il primo benchmark si fermava lì) e testo
   solo sulla copertina, eppure risultava `empty: no`. La misura «vuoto sì/no»
   nasconde un documento scansionato quasi per intero: ora `parse_quality` conta
   le pagine con text layer, quelle lette via OCR e quelle non lette.
   In `derivati/` c'è un estratto di 8 pagine della scansione (pp. 21–28), perché
   l'originale supera il limite di pagine. Mancano ancora DOCX, PPTX e foto di
   fotocopie derivati, con uno script e un manifest.

**Raccogliere i numeri, intanto.** PR 4 misura su ogni upload quanto `pypdf`
riesce a estrarre; `scripts/eval/benchmark_parsers.py` fa la stessa misura su
una cartella, senza WebUI né database. La percentuale di `empty: true` è quanti
PDF delle docenti sono scansionati, ed è il dato che dimensiona il lavoro di OCR.

### Fase 0 — chiusa

Niente. Le voci del gate della guida sono soddisfatte: Alembic operativo, upload
legacy durevoli, CSRF e sanitizer attivi, zero accessi cross-user nei test, flag
centralizzate, recovery del worker dimostrato.

Resta fuori `document_ir`, che è nel DDL dell'Appendice A.2 ma non nell'obiettivo
della PR 3: è l'output del parser e arriva con lui, in Fase 1. Aggiungerlo adesso
avrebbe creato una tabella vuota senza nessuno che ci scrive.

### Igiene — non blocca il gate, ma è noto

| | Cosa | Perché prima o poi |
| --- | --- | --- |
| T13 | Retention dei checkpoint | ogni run effimero scrive `ephemeral-{uuid}` che nessuno rilegge e nessun percorso cancella. Verificato: niente sotto `scripts/ops/` |
| T14 | Streaming su `stream_mode="messages"`, ritiro di `agent/graph/write_stream.py` | il bus è un `dict` globale, quindi per-processo. Non è rotto oggi perché la produzione gira a un worker: **è bloccante nel momento in cui si passa a due** |
| — | `marked.min.js` è caricato senza versione | `<script src=".../npm/marked/marked.min.js">` prende sempre l'ultima. DOMPurify accanto è pinnato; questo no |
| — | `/auth/logout` è una GET | un `<img src="/auth/logout">` disconnette la docente. Fastidio, non danno; per sistemarlo serve cambiare il link in navbar in un POST |
| — | Il Critic non sa quando il retrieval è stato riusato | `retrieval_reused` è in stato, il prompt non lo legge: valuta il grounding come se le fonti fossero appena arrivate |
| — | `force_retrieval` non ha un controllo in interfaccia | il campo di stato c'è, il grafo lo onora: manca la decisione UX (§6.4, decisione U2) |
| — | Il planner classifica «traduci in inglese» come `lesson_creation` | il routing non ne dipende più, ma la classificazione resta sbagliata a monte |

---

## Numerazione delle fasi — quale vale

Tre documenti in `docs/product/` usano la parola «Fase» con tre significati
diversi. Nessuno dei tre è sbagliato: descrivono assi di lavoro diversi. Ma
«partiamo da Fase 0+1+2» significa tre cose a seconda di chi lo dice, e questo
sì che è un problema.

**Canonica è la numerazione della guida di Louis**
(`Advanced_Input_and_Export_Architecture_v2-1_Angelo.md`, §5). Non perché sia
migliore, ma perché è quella a cui sono agganciati i numeri di PR, i gate, il
documento delle decisioni e ormai i commenti nel codice (`Fase 0 · PR 3`).
Rinumerare costerebbe churn senza guadagnare chiarezza.

| Fase | Nella guida (canonica) | Stato |
| --- | --- | --- |
| 0 | Fondazioni: migrazioni, storage, sicurezza, coda | chiusa |
| 1 | Material pipeline in **shadow mode** | prossima |
| 2 | Parsing, preview e private RAG | |
| 3 | Export locale | |
| 4 | Google account link e Slides | |
| 5 | Classroom DRAFT | |
| 6 | Revisione avanzata dei sorgenti | |

### L'altro asse: la qualità dell'output dell'agente

`WebUI_Improvement_Strategy.md` ha anch'esso «Fase 0–3», ma parla d'altro: i
sette difetti D1–D7 dell'output dell'agente. Non è input/output, è *cosa scrive*
il Writer. Per non collidere, d'ora in poi quelle vanno chiamate **Q0–Q3**
(«qualità»), mai «Fase».

| | In `WebUI_Improvement_Strategy.md` | Rapporto con la guida |
| --- | --- | --- |
| **Q0** | Harness record/replay | indipendente — abilita tutto il resto senza spendere token |
| **Q1** | Fondamenta della fattibilità (D1 profilo mutilato, D2 etichette BES grezze, D3 validator, D4 ricerca web) | indipendente: nessuna sovrapposizione con la guida |
| **Q2** | Risorse: qualità, persistenza, libreria (D5) | **si sovrappone** alla Fase 1–2. La persistenza dei media la risolve il material pipeline; la *qualità* della selezione no |
| **Q3** | Documento strutturato a blocchi (D6) | **si sovrappone** alla Fase 3: `LessonDocumentV1 -> ExportIRV1` è la stessa cosa vista dall'altro lato |

**D7 — «LangGraph usato al minimo» è in buona parte già fatto**, il 2026-09-08:
compilazione unica, `input_schema`, reducer, `RetryPolicy`, retrieval
condizionale. Va spuntato lì dentro invece di restare aperto.

### Cosa NON perdere

Q1 non è coperto da nessuna fase della guida, ed è la parte che tocca più da
vicino la qualità percepita dalla docente: un profilo educativo reso per intero
al Writer, e le etichette BES in italiano invece che come codici enum grezzi.
La guida non lo cita perché parla di input e output, non di prompt. Va tenuto
vivo qui.

### Che cosa significa «Fase 0+1+2»

Sotto la numerazione canonica: fondazioni (fatte), material pipeline in shadow
mode, parsing e private RAG. **Non** include l'export locale, che è la Fase 3.
Se l'intenzione della riunione era «arriviamo all'export», allora l'ambito è
0+1+2+3 — ed è una domanda da porre a Louis in una riga, non da assumere.

---

## In attesa di altri

| Cosa | Da chi | Blocca |
| --- | --- | --- |
| `alembic stamp head` sul DB di produzione, prima del primo `upgrade` | team DEV | il primo deploy con migrazioni |
| Variabili d'ambiente delle nuove flag | team DEV | l'attivazione, non il deploy |
| Ricostruzione dell'immagine per il volume degli allegati | team DEV | il volume è già nel compose (`21dfd53`); aggiungerlo contro un'immagine precedente ricrea il caso root-owned |
| ~~Ratifica di `bge-m3` via OpenRouter invece che locale~~ | — | **Chiuso il 2026-09-15.** Confermato via OpenRouter, con la policy **Zero Data Retention** attivata sull'account, e vale anche per i modelli di embedding. È la risposta alla preoccupazione della decisione 05 sui materiali con dati personali: il testo attraversa i provider ma non viene conservato. La dimensione resta 1024 |
| Conferma che «Fase 0+1+2» include o esclude l'export | Louis | l'ambito, non il lavoro: vedi «Numerazione delle fasi» sopra. Una riga di risposta |

---

## Registro

Dal più vecchio al più recente. Solo ciò che è entrato nel branch.

### 2026-09-08 — Baseline verificabile

La CI era rossa e `pytest` non era installato: finché non è stato chiuso questo
blocco, nessuna affermazione di verifica valeva niente.

| Commit | Cosa |
| --- | --- |
| `ad52985` | chiusi i sei errori di lint aperti |
| `f19449b` | ruff pinnato **esatto**: il formatter cambia le sue scelte di a-capo tra minori, quindi con `>=` la CI installava ogni mattina un formatter diverso e il check diventava rosso senza che nessuno toccasse il codice |
| `f8fc611` | reformat di massa, isolato in un commit suo |
| `b7ae51c` | quel commit escluso da `git blame` |
| `2a371f7` | set di dipendenze di nuovo installabile |
| `352df91` | primi test sulla topologia del grafo compilato: prima non ne esisteva nessuno, e le due funzioni dove vive tutta la logica di diramazione non erano esercitate da niente |

### 2026-09-08 — Risanamento del grafo

Quattro correzioni retroattive: nessuna cambia il comportamento osservabile,
tutte rendono possibile ciò che viene dopo.

| Commit | Cosa |
| --- | --- |
| `0fed375` | grafo compilato una volta per processo. Il service costruiva un orchestratore nuovo a ogni run, quindi la cache era sempre vuota e `compile()` girava per ogni lezione |
| `e64480e` | `input_schema` e reducer ordine-indipendenti. `create_initial_state()` azzerava esplicitamente ogni campo di output: al secondo turno il retrieval del primo veniva cancellato prima che il grafo partisse, e **lo stato del checkpointer era funzionalmente inutilizzato** |
| `f69a7aa` | timeout e retry espliciti sulle chiamate LLM. Il budget era quello di default dell'SDK — 600 secondi — quindi una chiamata appesa poteva tenere una lezione per mezz'ora mentre l'heartbeat la dichiarava sana |
| `c5338dc` | retrieval condizionale: dal secondo turno il grafo può riusare il corpus invece di ricercare — ma non scattava mai, corretto il giorno dopo in `8e540e5` |

### 2026-09-08 — Due difetti che erano in produzione

| Commit | Cosa |
| --- | --- |
| `3ca2e0b` | il materiale caricato dalla docente non arrivava alla revisione: `revise()` non accettava il parametro. Dal punto di vista di chi usa il sistema, l'assistente si dimenticava il file appena il Critic chiedeva una modifica |
| `21dfd53` | gli allegati si perdevano alla sostituzione del container: `docker-compose.prod.yml` non montava un volume per `data/webui/uploads` |

### 2026-09-08 — Migrazioni (PR 0)

| Commit | Cosa |
| --- | --- |
| `8648795` | schema WebUI sotto Alembic. Prima era `create_all` più `ALTER TABLE` a mano dietro un `if not sqlite: return`, quindi **in produzione su Postgres nessuna patch veniva applicata**. I DB esistenti si marcano, non si ricreano |
| `ca0c124` | lock rigenerato perché la produzione riceva Alembic |
| `43514d8` | changelog |

### 2026-09-08 — CI

| Commit | Cosa |
| --- | --- |
| `51dbcb5` | i test di integrazione non girano più su un runner che non ha Neo4j né le chiavi LLM |
| `0d17c57` | un modulo di test non ne importa più un altro. Funzionava sotto `python -m pytest`, che mette la working directory su `sys.path`, e falliva sotto `pytest` nudo — che è quello che gira in CI |

### 2026-09-09 — Il retrieval condizionale funziona, e la sicurezza WebUI parte

| Commit | Cosa |
| --- | --- |
| `8e540e5` | il retrieval condizionale non scattava mai. Il planner veniva chiamato con la sola domanda e nessun thread, quindi «traduci la lezione in inglese» gli sembrava una richiesta nuova: restituiva i concetti di una lezione che nessuno aveva chiesto, e il router — che confrontava quei concetti col corpus — non ha mai scelto il riuso su un thread reale. Ora il planner riceve la conversazione e il corpus e risponde direttamente; il confronto sui concetti resta come fallback. Inoltre la chat lasciava appeso il turno riusato: il ciclo SSE scarta i nodi che non conosce, quindi `reuse` non emetteva nulla e la card del retriever filava all'infinito |
| `e997574` | CSRF su `/webui` e `/auth`. Prima nessuno verificava da dove arrivasse una richiesta che cambia stato: con la sessione in un cookie, qualunque pagina aperta nel browser della docente poteva far cancellare una lezione o un allegato a suo nome. Token double-submit firmato sulla sessione, un middleware ASGI invece di una dipendenza per rotta, `/api/v1` non toccato |

**Verificato prima del commit:** ruff pulito, 301 test passati, e uno smoke contro
l'applicazione vera (non contro un'app di test) su tutti e quattro i percorsi —
nuova lezione, login, register, DELETE htmx: 403 senza token, passano con.
Restano rosse solo le due `test_concurrency_limits`, che falliscono su
`aix/core/concurrency.py` — file non toccato — per la risoluzione del timer su
Windows (`waited 0.0s`).

### 2026-09-09 — Storage e modelli dei materiali (PR 2)

| Commit | Cosa |
| --- | --- |
| `03748aa` | un upload era un nome di file dentro una lista JSON sulla riga della lezione, scritto con una `write()` normale su un percorso costruito da quel nome. Ora: `StorageBackend` con `LocalDurableStorage` per il pilot e uno scaffold `S3CompatibleStorage`, chiavi derivate da proprietario e digest — **mai** dal nome del file o dall'email — scrittura via file temporaneo, `fsync` e rename, e verifica del checksum su `put`, `get` e `delete`. Quattro tabelle secondo l'Appendice A.2: sostituire un file **aggiunge una versione** invece di sovrascriverla, così una lezione continua a risolvere il file da cui è stata scritta. Ogni metodo dei repository richiede `owner_id` come argomento keyword obbligatorio, anche le letture per chiave primaria |

**Un difetto intercettato prima del commit.** L'autogenerate di Alembic aveva
compilato il default booleano con il dialetto SQLite su cui girava, producendo
`enabled BOOLEAN DEFAULT 1`. PostgreSQL lo rifiuta — *column is of type boolean
but default expression is of type integer* — quindi la migrazione sarebbe
passata in locale e fallita sul database di produzione. Trovato rendendo l'SQL
offline per entrambi i dialetti prima di committare; ora `DEFAULT true` su
Postgres e `DEFAULT 1` su SQLite.

**Verificato:** upgrade, downgrade e di nuovo upgrade su SQLite; `alembic check`
non rileva differenze fra modelli e migrazione; 56 test nuovi, 357 in totale.

**Non collegato a nessuna rotta.** L'adapter shadow sull'endpoint di upload è la
PR 4, e `AIX_MATERIALS_ENABLED` è spento di default: il percorso legacy resta
l'unico che gira.

### 2026-09-09 — La sicurezza WebUI è completa (PR 1)

| Commit | Cosa |
| --- | --- |
| `d999e52` | **Sanitizzazione.** Il corpo della lezione è reso da Markdown e marcato `\| safe` in quattro template; python-markdown lascia passare l'HTML grezzo per scelta, e quel Markdown è scritto da un LLM, da un corpus, da ciò che digita la docente e dal testo estratto dai file che carica. Ora è filtrato contro una allowlist nell'unica funzione che produce quell'HTML. Lo stream del writer aveva lo stesso buco lato client, dove l'output di `marked` finisce in `innerHTML`: DOMPurify pinnato, e se uno dei due script non carica il riquadro degrada a testo semplice. **Errori.** Quattro percorsi mettevano `str(exc)` in un campo che la chat mostra — la query SQL fallita, l'host del database, i percorsi del container, l'URL chiamato. Ora il fallimento viene loggato con un codice incidente e alla docente arriva una frase che dice cosa fare, con quel codice. **IDOR.** Due utenti esercitano ogni rotta owner-scoped in entrambi i versi |

**Il comportamento era già corretto, mancava il test.** Ho sondato tutte e dodici
le rotte owner-scoped come utente sbagliato prima di scrivere gli assert: tutte
404, nessuna perdita di contenuto. Il valore del test è che se un `WHERE` sparisce
qualcuno se ne accorge — il sintomo non è un errore, è una docente che legge la
lezione di un'altra. Ogni caso verifica anche il verso opposto, perché
un'applicazione che desse 404 a tutti soddisfarebbe la prima metà.

**Due difetti trovati mentre lavoravo:**

- il mio primo test IDOR usava due client con `dependency_overrides`, che vive
  sull'app e non sul client: il secondo fixture sovrascriveva il primo e «Bruno»
  agiva da Anna. Tre test rossi che sembravano bug dell'app; ho verificato la
  rotta profili a mano prima di concludere che era il test;
- `logger.exception()` legge `sys.exc_info()`, quindi fuori da un blocco `except`
  registra il codice incidente **senza** l'eccezione — perdendo esattamente ciò
  che il codice serve a ritrovare. Ora l'eccezione è passata esplicitamente.

**Sul lock:** aggiungere `nh3` ricompilando su Windows aveva tolto `uvloop`,
`jeepney` e `secretstorage` e aggiunto `pywin32` — l'immagine di produzione è
Linux. Ricompilato con `--python-platform x86_64-manylinux_2_36` come il lock
originale: il diff è esattamente una riga.

**Verificato:** 410 test passati, ruff pulito, e uno smoke sull'applicazione vera
(CSRF ancora attivo, DOMPurify caricato, sanitizzazione prima di `innerHTML`).

### 2026-09-09 — La coda di job durevole (PR 3)

| Commit | Cosa |
| --- | --- |
| `a3ec87b` | `ingestion_job` più `JobDispatcher` e worker interno. Un job è una riga committata prima che qualunque lavoro cominci, quindi un riavvio a metà parsing perde il tentativo e non il lavoro. Il claim legge id candidati e poi fa un `UPDATE` condizionale: il lock di riga fa sì che esattamente un worker ottenga `rowcount 1` e gli altri passino oltre. `SKIP LOCKED` è usato **solo** su PostgreSQL e **solo** per risparmiare un round trip a chi perde — la correttezza non ci dipende, ed è questo che tiene in piedi SQLite. Al posto dell'heartbeat c'è un lease rinnovato a ogni step, perché un job è una sequenza di passi con una scrittura in mezzo, non una coroutine unica |

**Perché un lease e non un heartbeat.** `agent_run` rinnova un heartbeat perché
una run è una coroutine continua. Un job no: un worker che muore blocca il suo
job per al massimo un lease, poi chiunque può prenderlo — con il tentativo già
addebitato, così un job che uccide sistematicamente il suo worker si arrende
invece di mangiare la coda.

**Un bug di portabilità trovato dai test.** SQLite restituisce le colonne
`DateTime(timezone=True)` senza tzinfo, e SQLAlchemy valuta la `WHERE` di un
`UPDATE` anche in Python sulle righe già in identity map: lì un valore aware e
uno naive si incontrano e sollevano. Ora tutto il modulo usa UTC naive come
`agent_run`, ed è proprio la coerenza fra i due a evitare che ricapiti.

**Verificato:** 25 test nuovi (le quattro condizioni di accettazione della guida
hanno una sezione ciascuna), 435 in totale, ruff pulito, migrazione su e giù e su
di nuovo con `alembic check` senza differenze.

**Inerte per costruzione:** il worker non parte se `AIX_MATERIALS_ENABLED` è
spento, e il handler predefinito **rifiuta** invece di marcare pronto un
documento che nessuno ha processato — completarlo lascerebbe la docente ad
aspettare contenuto che non arriverà mai.

---

## Decisioni prese il 2026-09-09 (da ratificare con Louis)

Angelo decide per andare avanti; Louis rivede a lavoro svolto.

### Parser: `pypdf@1` oggi, e Tesseract per le scansioni — **non** Docling

> **Corretto il 2026-09-14.** Il 9 settembre avevo consigliato Docling come
> secondo profilo. Su questa VM è sbagliato, e la correzione è verificata, non
> un ripensamento.

`parser_profile_version` è una stringa e fa parte della chiave unica di
`ingestion_job`: **un profilo nuovo è un job nuovo**, cioè un re-parse del
corpus senza cancellare niente a mano. Lo schema è già costruito per cambiare
parser dopo, quindi la scelta non blocca nulla.

| Opzione | Valutazione |
| --- | --- |
| **`pypdf`** (oggi, `pypdf@1`) | Nessuna dipendenza nuova. Perde tabelle e ordine di lettura sui layout a due colonne, e su un PDF scansionato restituisce **stringa vuota** |
| **Tesseract** via OCRmyPDF — *consigliato per le scansioni* | ~100 MB medi per pagina, p90 200–300 MB, con `--jobs 1` e downsample. È l'OCR locale che la guida stessa propone in §3.3, e mantiene vero il «parser e OCR locali» di §12.4 |
| **Docling** | **Escluso da questa VM.** Picchi di 3–4 GB su un singolo PDF e `bad_alloc` anche su macchine da 16 GB (issue docling #2877, #2149). Con 4 GB condivisi fra Postgres, app e Caddy, un solo PDF scansionato abbatte il container e con lui la chat. Rientra in gioco solo a 8 GB |
| **PyMuPDF** | **Scartato.** AGPL-3.0 oppure licenza commerciale Artifex, e `pyproject.toml` dichiara questo progetto `Proprietary` |
| **Marker** | Qualità alta ma pensato per GPU; la VM non ne ha |

Il momento giusto per misurare resta quando ci sono documenti veri:
`scripts/eval/benchmark_parsers.py` esiste apposta.

### Embedding: `bge-m3` via OpenRouter, 1024 dimensioni

OpenRouter **ha** un endpoint `/embeddings` e serve `baai/bge-m3` — 1024
dimensioni, multilingue, quindi adatto all'italiano. Le altre due BGE su
OpenRouter (`bge-base-en-v1.5` 768, `bge-large-en-v1.5` 1024) sono **solo
inglese**: sarebbero la scelta sbagliata per lezioni in italiano a prescindere
dalla dimensione.

Il repo chiama già OpenRouter per gli embedding dei media
(`embedding_model = "openai/text-embedding-3-small"` con
`base_url = https://openrouter.ai/api/v1`), quindi il client esiste già.

**Quello che questa scelta cambia davvero.** La decisione 03 del 2026-09-08
diceva «embedder **locale** per il corpus privato», e la ragione non era tecnica:
il corpus è fatto dei materiali della docente, e la decisione 05 prevede
un'etichetta per quelli che contengono dati personali di studenti. Passare da
OpenRouter significa che quel testo esce dal perimetro e attraversa i provider a
cui OpenRouter instrada (per `bge-m3`: Parasail e DeepInfra). Non è un blocco —
è una scelta di rischio, e va presa sapendola invece che ereditandola.

Due cose la rendono reversibile a costo quasi nullo:

- **la dimensione è 1024 in entrambi i casi.** `bge-m3` locale e `bge-m3` via
  OpenRouter sono lo stesso modello, quindi `vector(1024)` è giusto comunque e
  la migrazione non aspetta questa decisione;
- lo schema salva già `embedding_model` per chunk, quindi un cambio di hosting è
  tracciabile. *(Caveat: provider diversi possono normalizzare o quantizzare in
  modo leggermente diverso, quindi vettori prodotti altrove non sono garantiti
  identici — un cambio di hosting va trattato come re-embedding, non come
  configurazione.)*

Mitigazione che costa una riga: OpenRouter espone controlli `data_collection`
per provider nella richiesta. Vanno impostati espliciti prima del primo
materiale reale, non dopo.

### 2026-09-11 — L'upload alimenta la pipeline in ombra (PR 4)

| Commit | Cosa |
| --- | --- |
| `50473c3` | `POST /webui/lesson/{id}/upload` continua a produrre lo stesso manifest, lo stesso estratto e lo stesso frammento htmx. Con la flag accesa memorizza **anche** i byte in modo durevole, registra materiale e prima versione, e li accoda. Tre chiavi opzionali finiscono nella voce di manifest perché una run in ombra sia ispezionabile dopo; il frammento non cambia — shadow mode che si vede non è shadow mode. Il worker parte nel lifespan con un `ExtractionHandler` che gira l'estrazione già in uso e registra **quanto bene è andata** invece di cosa ha prodotto |

**Il vincolo che ha governato la PR:** «failure del worker non rompe la chat».
Il lavoro shadow gira su una sessione propria, dopo il commit legacy, dietro
funzioni che non possono sollevare. I test rompono lo storage e la coda
*apposta* e verificano che la docente riceva comunque il suo file.

**Due difetti trovati scrivendo i test:**

- `session.refresh(lesson)` ricarica la colonna JSON dal database, quindi la
  voce di manifest che la route aveva in mano diventava un dizionario orfano e
  l'annotazione shadow scriveva nel vuoto. Ora la voce viene ritrovata per id
  nella lista ricaricata;
- `extract_text` **solleva** su un PDF corrotto. Nell'handler ora è catturato e
  registrato come `unreadable: true`: quei byte non diventeranno leggibili al
  terzo tentativo, e ritentare cinque volte spenderebbe uno slot su una
  conclusione già nota.

**Verificato:** 18 test nuovi, 453 in totale, ruff pulito. E contro
l'applicazione vera: il DB di dev è migrato da solo a `48f042517314`, il worker
scrive `[jobs] worker ... started` all'avvio e `stopped` in chiusura con
teardown immediato, e con la flag spenta non parte.

---

### 2026-09-14 — Il modello vede finalmente tutta l'aula (Q1 · 1.1)

| Commit | Cosa |
| --- | --- |
| `628baaa` | Il Writer costruiva la sezione del profilo da **cinque campi**. Numero di studenti, dispositivi, LIM, WiFi, postazioni PC, mobilità degli arredi, caratteristiche della classe: raccolti dal form, salvati, mostrati nella sidebar, e **mai** nel prompt. Ecco perché una lezione chiedeva Kahoot dagli smartphone a una classe che dichiara di non averne: il modello non ignorava il vincolo, non l'aveva mai visto. Ora `profile_renderer` rende ogni campo con la sua etichetta e riscrive l'aula come **vincoli imperativi**. Gli stessi vincoli vanno anche al Planner, così il retrieval non si spende su attività che l'aula non può fare |

**I codici BES non arrivano più grezzi.** Venivano uniti come `ADHD, DCGM, DLDS`
e il modello li riespandeva a memoria: `DCGM` diventava disprassia invece di
disabilità cognitiva media, `DLDS` un disturbo del linguaggio invece di uno
studente che impara l'italiano. Le etichette italiane esistevano già in
`educational_profile.py`; le usavano solo i template.

**Una regola che vale la pena ricordare:** *ignoto non è no*. `has_lim=None`
significa che la docente non ha risposto e non produce nessuna riga; solo un
`False` esplicito diventa «non disponibile». Altrimenti un profilo compilato a
metà vieterebbe metà dell'aula.

**Anche:** `scripts/eval/benchmark_parsers.py`, che fa la misura del worker su
una cartella di documenti — la metà «in batch» del piano, e lo strumento che
trasforma il corpus in numeri appena arriva.

**Verificato:** 41 test nuovi, 490 in totale, ruff pulito. Restano le due
`test_concurrency_limits` note su Windows.

---

### 2026-09-14 — «È fattibile» smette di essere un'opinione (Q1 · 1.3)

| Commit | Cosa |
| --- | --- |
| `fb95e51` | Il Critic giudicava qualità pedagogica e fattibilità con la stessa rubrica 1–5: ecco perché una lezione che chiedeva Kahoot dagli smartphone a venticinque studenti senza dispositivi, in un'aula con i banchi fissi, tornava al 91%. Niente nella pipeline sapeva **contare**. Ora un nodo `validate` sta fra Writer e Critic: Python puro, nessuna chiamata al modello, sette regole che rispondono a domande numerabili |

**La separazione è il punto.** Il Critic continua a giudicare se una lezione è
*buona* — è soggettivo, e va bene che lo faccia un modello. Il validator decide
se è *possibile* — è oggettivo, e non deve dipendere dall'umore di un LLM.

**Le regole, e perché alcune non bloccano.** `duration_mismatch`, `byod_off`,
`no_wifi`, `furniture_static` e `bes_label_drift` bloccano: si appoggiano a un
fatto dichiarato nel profilo. `group_size` e `no_projection` sono solo
avvertimenti, perché si appoggiano a euristiche sul testo — un falso positivo
che riscrive in silenzio una lezione valida è peggio di una nota visibile.

`bes_label_drift` è la **sentinella di regressione** del profile renderer: se
qualcuno rimette i codici grezzi nel prompt, fallisce prima che se ne accorga
una docente.

**Il retry riusa il canale che c'era.** Una violazione bloccante rimanda la
bozza al Writer con la sola remediation, attraverso `refinement_instruction`.
Il budget è quello del validator, non del Critic: spendere una revisione su una
lezione infattibile toglierebbe alla docente un tentativo per averne una
*migliore*, che è un fallimento diverso.

**Due errori miei, trovati dai test:**

- il router confrontava il budget con `>=` su un contatore già incrementato,
  quindi concedeva **una riscrittura in meno** di quante ne dichiarasse;
- avevo scritto un test che pretendeva una sola violazione su una bozza che ne
  produce legittimamente due — Kahoot in quell'aula manca sia del dispositivo
  sia della connessione. Il codice aveva ragione, il test no.

**Verificato:** 34 test nuovi, 524 in totale, ruff pulito. Restano le due
`test_concurrency_limits` note su Windows.

---

### 2026-09-15 — Dipendenze di parsing ed export; il PDF diventa un file vero (PR 7)

| Commit | Cosa |
| --- | --- |
| `ec917a6` | `python-docx`, `python-pptx`, `pypdfium2`, `pytesseract`, `filetype`, e `markdown-it-py` dichiarato. Tutte con licenza permissiva — il progetto è `Proprietary`. Lock compilato per Linux come l'originale: si muove esattamente di quei cinque più `xlsxwriter`, transitiva di `python-pptx` |
| `f8150e1` | Il pulsante «PDF» apriva `/print` e lasciava tutto al dialogo di stampa del browser. Ora la lezione diventa `LessonDocumentV1`, poi `ExportIRV1` (§10.1–10.4), e il PDF è generato sul server con ReportLab. Più `format=outline`: la scaletta in testo con i minuti di ogni sezione e il totale confrontato con la durata |

**Il font.** Noto Sans è incluso nel repo (licenza OFL), perché l'Helvetica di
ReportLab copre poco oltre il set occidentale. Ma Noto Sans non ha frecce, «≤»,
spunte né emoji, che il Writer usa: un glifo mancante stampa un quadratino. Ogni
carattere è confrontato con la mappa del font: alcuni sono sostituiti
(→ diventa «->»), gli altri tolti e registrati.

**Provato su contenuto vero**, non solo su fixture: sei lezioni reali del DB di
dev, 2–4 pagine, 11–29 ms ciascuna, tabelle comprese. I caratteri tolti sono solo
emoji. Una lezione non dichiara la durata: l'adapter lo segnala, non la inventa.

**Un test sbagliato, trovato subito.** Pensavo che fosse `_safe_href` a bloccare
i link `javascript:`. In realtà `markdown-it` in modalità CommonMark non li crea
proprio. Le difese sono due e il test ora verifica la garanzia — nulla diventa
cliccabile — non il livello che la fornisce.

`/print` resta nella card, rinominato «Stampa». **Verificato:** 50 test nuovi,
574 in totale.

---

## Un difetto preesistente trovato il 2026-09-11, non toccato

`extract_text` (`src/aix/webui/lessons/uploads.py:57`) solleva su un PDF
corrotto o troncato, e `lesson_upload` cattura solo `ValueError`: un PDF rotto
caricato dalla docente diventa un **500**, non un messaggio.

Non l'ho corretto perché è il percorso legacy, che questa PR ha il compito di
lasciare intatto. È una riga — trasformare l'eccezione in un `ValueError` con
un messaggio in italiano, come gli altri controlli di `save_upload`.
