# Risposta mediana: specifica per lezione e follow-up

**Origine:** lettura alla cieca degli 8 casi di `scripts/eval/quality_ab.py` (16/10/2026),
fatta da Angelo con Claude. Salvata il 19/10/2026 come riferimento per il lavoro sulla
qualità delle risposte (`docs/product/Analisi_Qualita_Risposte.md`).

**Le due versioni**

- **Sistema G**: il branch `feature/response-quality-improvement`. Il follow-up parte
  dalla traccia reale del grafo («Il grafo ha trovato 74 concetti…», collegamento
  bisogno → strategia).
- **Sistema R**: l'agente dello staging. Il follow-up ricostruisce le fonti dal testo
  della lezione («Non ho accesso diretto al grafo…»).

La mappatura dedotta dalla lettura (G = Y nei casi 1–5 e 7, G = X nei casi 6 e 8)
coincide caso per caso con `chiave.json`.

---

## 1. Cosa tenere di ciascuno

| Aspetto | Prendi da G | Prendi da R | Scarta |
|---|---|---|---|
| Tempi e gruppi | I tempi tornano e i gruppi corrispondono al numero di alunni | – | R: 5 lezioni su 7 sbagliano i tempi; c'è un «gruppo da 1» |
| Uso del grafo | Bisogno → strategia, strategie sconsigliate (Station Rotation con classi grandi), neuromiti | – | R usa la Station Rotation contro il grafo (casi 1 e 3) |
| Ambito del follow-up | Si limita ai profili presenti in classe (caso 7: «discalculia non nel profilo») | – | G caso 4 inventa bisogni assenti (ADHD, autismo, non italofoni) |
| Lunghezza del follow-up | – | Breve e leggibile | G elenca decine di nodi irrilevanti («dim, poor, vibrant, warm…») |
| Onestà sulle fonti | – | Segnala i riferimenti che non c'entrano (van Knippenberg nel caso 7) e le fonti da verificare | G cita riferimenti estranei senza segnalarli (memoria di riconoscimento nei roditori, ChatGPT) |
| Precisione disciplinare | – | La cautela sull'evidenza multisensoriale nel caso 7, la nota DSM-5 sull'«Asperger» | G caso 7: «gatto» diviso in «ga-to» e il confronto p/b basato sulla «pancia a destra», entrambi sbagliati |
| Chiusura | Lacuna concreta e proposta (caso 8 R: misconcezione «l'acqua sparisce») | «Se vuoi posso…» | – |
| Lingua | Nomi dei nodi tradotti in italiano (G caso 5) | – | Nomi solo in inglese, refusi ereditati dal grafo («Gammification»), «Stato Llano» |
| Obiettivi | – | Obiettivi per livelli di Bloom (ricordare, comprendere, applicare, esprimere), scelti per la lezione e non come formula fissa | – |

**In sintesi:** la pipeline di G (recupero reale e rispetto dei vincoli), più uno strato
editoriale con le qualità di R: filtrare, essere brevi, essere onesti. E, aggiunta di
Angelo: ciò che viene dal grafo va **rielaborato in consigli pratici**, applicabili in
classe, non elencato.

---

## 2. Regole per la lezione

**Vincoli strutturali** (verificati in modo deterministico prima della consegna)

1. La somma delle fasi è uguale alla durata, valutazione inclusa. Nessuna fase da «0
   minuti».
2. Gruppi: nessun gruppo da 1, gruppi da 3 a 5, somma uguale al numero di alunni. Il
   numero di ruoli o di stazioni è coerente con quello dei gruppi o dei membri.
3. Una strategia che il grafo marca come sconsigliata per il profilo della classe non
   si usa.

**Contenuto**

4. Il contenuto disciplinare viene da fonti in italiano, adatte al livello (Wikipedia
   it prima di en). Gli obiettivi non vanno oltre la richiesta: in seconda primaria
   niente solidificazione e fusione se si chiede di «elencare le fasi».
5. Obiettivi: 2–4, osservabili, con verbi della tassonomia di Bloom scelti in base
   all'intento e al grado (un'introduzione resta su ricordare e comprendere). Non un
   elenco fisso.
6. Nel testo per la docente nessun gergo della pipeline: niente «nodo», «confidenza
   MEDIA», «VERY_HIGH», «il KG collega». Si scrive «La ricerca indica…».
7. I riferimenti scientifici che non sostengono nessuna scelta della lezione si tolgono
   prima di scrivere, non si segnalano dopo.
8. Gli esempi linguistici e scientifici (sillabazione, coppie di lettere, «l'acqua
   diventa aria») sono controllati dal giudice in un campo `accuratezza`.

**Differenziazione**

9. Una riga per ogni bisogno dichiarato: bisogno → strategia (in italiano, il nome del
   nodo tra parentesi) → che cosa fa la docente, quando, con quale materiale → in quale
   attività compare. Una strategia che non diventa un'azione non si nomina.
10. Ambiente di apprendimento: 2–4 righe al massimo, solo se pertinente al profilo.

---

## 3. Regole per il follow-up «Cosa hai trovato nel grafo?»

- **Fonte:** la traccia del recupero (nodi e relazioni passati al Writer), non il testo
  della lezione. Se la traccia non c'è, dirlo in una riga e fermarsi lì.
- **Ambito:** solo i bisogni presenti nel profilo della classe. I nodi degli altri
  profili si riassumono in una riga («Il grafo ha restituito anche nodi sulla
  discalculia, non pertinenti a questa classe»).
- **Lunghezza:** circa 15–20 righe:
  - tutte le strategie usate, una riga ciascuna, con il punto della lezione in cui
    compaiono;
  - al massimo 5 strategie non usate ma pertinenti, ciascuna con il motivo;
  - le strategie sconsigliate dal grafo e i neuromiti evitati, in una riga;
  - il resto in un conteggio («altri 31 nodi su ambiente e materiali, non usati»).
- **Etichette:** prima in italiano, poi il nome del nodo tra parentesi.
- **Fonti:** quali riferimenti sostengono quale scelta, e quali sono solo di sfondo.
- **Chiusura:** una lacuna concreta, con una proposta d'azione.
- **Coerenza:** stesso registro della lezione (tu o lei, mai tutti e due). Ogni conteggio
  dichiarato deve corrispondere a ciò che si elenca o si riassume.

### Template

```
Il grafo ha restituito {N} concetti e {M} relazioni; ecco quelli pertinenti alla tua classe.

**{Bisogno 1, in italiano}**
- {strategia it} ({nodo}): usata in {punto della lezione}.
- {strategia it} ({nodo}): non usata, perché {motivo}.

**{Bisogno 2}**
- …

**Sconsigliate dal grafo ed evitate:** {strategia}, {neuromito}.
**Fonti:** {rif. A} sostiene {scelta}; {rif. B} è solo di sfondo.
**Non pertinenti:** {K} nodi su {temi}.

Lacuna: {cosa manca}. {Proposta concreta}?
```

### Esempio: caso 2 (frazioni equivalenti, discalculia e plusdotazione)

> Il grafo ha restituito strategie per i due profili della classe e per gruppi fino a 30
> alunni. Ecco quelle pertinenti.
>
> **Alunno con discalculia**
> - Retta dei numeri (*Number lines*): usata nell'Attività 2.
> - Rappresentazioni visive della grandezza (*visual magnitude representations*):
>   strisce colorate nell'Attività 1.
> - Promemoria esterno (*external memory aids*): cartellino «Passi» sul banco.
> - Ripasso distanziato (*spaced repetition*): ripresa nei giorni successivi.
>
> **Alunno plusdotato**
> - Curricolo compattato (*compacted curriculum*): salta le attività base dopo il primo
>   controllo.
> - Arricchimento e domande socratiche: scheda «sfida» e «perché la regola funziona
>   sempre?».
> - Apprendimento guidato dagli interessi (*interest-driven learning*): non usato,
>   perché la lezione non conosce gli interessi dell'alunno.
>
> **Sconsigliate dal grafo ed evitate:** esercizi ripetitivi senza variazione per il
> plusdotato, Station Rotation per una classe di 24.
> **Fonti:** Mazzocco et al. (2011) sostiene il lavoro sulla grandezza; Schreffler et al.
> (2019) è solo di sfondo sull'UDL.
> **Non pertinenti:** 11 nodi su ambiente e altre metodologie (STEAM, Debate…).
>
> Lacuna: per il plusdotato manca un aggancio ai suoi interessi. Vuoi che aggiunga una
> variante della sfida legata a un tema che gli piace?

---

## 4. Controlli da aggiungere a `scripts/eval/quality_ab.py`

| Controllo | Livello | Casi che lo farebbero scattare |
|---|---|---|
| Etichette di confidenza nel testo per la docente («confidenza MEDIA», «(VERY_HIGH)») | 1 | 3-X, 7-X |
| «gruppo da 1» e fasi da «0 minuti» | 1 | 3-X, 6-Y |
| Ripetizioni del tipo «bravo/bravo» | 1 | 6-Y |
| Follow-up che cita bisogni assenti dal profilo | 1 (confronto con il profilo) | 4-Y |
| Conteggio dichiarato nel follow-up che non corrisponde agli elementi elencati | 1 | 6-X |
| Follow-up oltre 25 righe | 1 | 1-Y, 3-Y, 4-Y |
| Strategia sconsigliata dal grafo ma usata nella lezione | 1 (serve la traccia) | 1-X, 3-X |
| Riferimento bibliografico estraneo al tema | 2 | 4-X, 4-Y, 7-Y, 8-X |

**Non è un difetto: il commento `<!-- ai-generated … trace_id -->`.** La lettura lo
vedeva perché la pagina alla cieca mostra il markdown grezzo. È un commento HTML voluto
(dichiarazione IA, Art. 50, leggibile dalle macchine): nella WebUI e in qualunque client
che renda il markdown non si vede. Per questo è escluso dai controlli.

---

## 5. Piano di lavoro (19/10/2026)

1. **Deterministici, subito** (fatto il 19/10: `2ee3d1c`):
   - solo i bisogni dichiarati nella mappa dell'evidenza (caso 4: con nessun bisogno
     dichiarato venivano elencati tutti);
   - refusi del grafo e gergo;
   - riferimenti fuori tema filtrati prima del Writer;
   - i controlli del §4 in `quality_ab.py`.
2. **Il cuore** (fatto il 19/10: `fc95208`; misure in `Analisi_Qualita_Risposte.md` §7.6):
   - strategie del grafo → azioni pratiche (§2.9);
   - obiettivi alla Bloom (§2.5);
   - il follow-up del §3, con l'elenco usate / non usate calcolato nel codice, non dal
     modello.
3. **Il giudice:** il Critic con una lista di controllo (esattezza disciplinare, aderenza
   al grafo, vincoli). Il costo per lezione va misurato.

Dopo ogni fase: confronto sugli stessi 8 casi e lettura alla cieca.
