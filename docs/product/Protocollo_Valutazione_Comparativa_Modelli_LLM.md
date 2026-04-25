# PROTOCOLLO DI VALUTAZIONE COMPARATIVA PER AIxLEARNING

**Obiettivo:** Validare in modo oggettivo (1) il valore aggiunto dell'architettura GraphRAG rispetto al sistema precedente, e (2) il confronto tra modelli LLM candidati per la pipeline completa AIxLearning (LLM Assistant + GraphRAG).

> **Nota:** Questo protocollo valuta la qualità dell'output finale dell'intero sistema (LLM Assistant che utilizza il contesto fornito dal GraphRAG), non solo la componente GraphRAG isolata.

---

## FASE 0 — Validazione Architetturale: LLM Alone vs LLM + GraphRAG (3–4 giorni)

### Fondamento scientifico

Questo confronto segue la metodologia di **ablation study** con **factorial design**, prassi standard nella ricerca AI per isolare il contributo di un componente specifico e verificarne la generalizzabilità (arXiv:2411.19463). Il framework ARES (arXiv:2311.09476, NAACL 2024) definisce tre dimensioni di valutazione per sistemi RAG: Context Relevance, Answer Faithfulness, Answer Relevance. Studi recenti (arXiv:2502.11371, 2025) confermano che GraphRAG supera gli LLM puri specificamente su query multi-hop e relazionali, mentre la valutazione umana di esperti di dominio rimane il gold standard per applicazioni specialistiche (RSC Advances, 2026).

**Design fattoriale a 2 variabili:**
- **Fattore 1 (Architettura):** LLM Alone vs LLM + GraphRAG
- **Fattore 2 (Modello LLM):** GPT-4o (baseline storica), Claude 4.6, Gemini 3

**Vincolo critico:** All'interno di ogni coppia, il modello LLM è identico. L'unica variabile è la presenza/assenza del contesto GraphRAG. Questo isola il contributo del Knowledge Graph e ne verifica l'effetto indipendentemente dal modello scelto.

### 0.1 Condizioni a confronto

| Coppia | Modello | Condizione A — LLM Alone | Condizione B — LLM + GraphRAG |
|--------|---------|--------------------------|-------------------------------|
| **1 (baseline storica)** | GPT-4o | Prompt hardcoded pre-GraphRAG, nessun KG | System prompt + response template + `kg_context_formatted` |
| **2 (candidato)** | Claude 4.6 | Prompt generico, nessun KG | System prompt + response template + `kg_context_formatted` |
| **3 (candidato)** | Gemini 2.5 or 3 | Prompt generico, nessun KG | System prompt + response template + `kg_context_formatted` |

**Approccio a livelli per contenere il carico di valutazione:**
- **Modello primario** (il candidato di produzione): test completo su 20 query
- **Modelli di conferma** (gli altri due): test ridotto su 10 query (le più discriminanti: multi-hop, strategie, conoscenza negativa)
- **Totale coppie da valutare:** ~40 (gestibile in 3–4 giorni)

### 0.2 Set di query (20 query: 10 Neuro + 10 UDL)

| Categoria | Neuro | UDL | Cosa testa |
|-----------|-------|-----|------------|
| Concetto singolo | 2 | 2 | Recall di conoscenza di dominio |
| Strategia per variabilità | 2 | 2 | Mapping learner → strategia (SUGGESTS) |
| Conoscenza negativa/restrittiva | 1 | 2 | Approcci da evitare o limitazioni (UDL: NO_SUGGESTS; Neuro: IMPAIRS, REDUCES, LIMITS) |
| Query multi-hop | 2 | 2 | Traversamento del grafo (es. variabilità → strategia → strumento) |
| Fuori dominio (controllo) | 1 | 1 | Fallback: entrambi i sistemi devono gestirlo |
| Query relazionale | 2 | 1 | Relazioni tra concetti (MITIGATED_BY, LEADS, PREDICTS, ASSOCIATES_TO) |

### 0.3 Valutazione da parte dei domain experts

Per ogni coppia di risposte (anonimizzate, ordine randomizzato), i domain experts valutano:

| Criterio | Cosa misura | Perché è rilevante per GraphRAG |
|----------|------------|-------------------------------|
| **Accuratezza fattuale** (1–5) | La risposta contiene informazioni corrette e specifiche del dominio? | Il KG fornisce dati curati dagli esperti |
| **Specificità** (1–5) | La risposta cita concetti, strumenti o strategie specifiche? | Il KG contiene nomi reali (es. Coggle, Kami, Focus Tasks) |
| **Conoscenza negativa/restrittiva** (0/1) | La risposta indica approcci da evitare o limitazioni? | Solo GraphRAG ha relazioni negative (NO_SUGGESTS, IMPAIRS, REDUCES, LIMITS) |
| **Tracciabilità** (0/1) | Le fonti delle affermazioni sono identificabili? | GraphRAG fornisce cypher_query e raw_nodes |
| **Preferenza globale** | A migliore / B migliore / Parità | Giudizio complessivo |

### 0.4 Criteri di successo

- **Win rate GraphRAG > 60%** sulle query di dominio (escludendo le fuori dominio) → architettura validata; > 70% = validazione forte
- **Conoscenza negativa/restrittiva presente in ≥ 60%** delle query dove il KG contiene relazioni negative → il KG aggiunge conoscenza che il LLM da solo non produce
- **Specificità media GraphRAG ≥ 3.0/5** vs LLM Alone → il contesto KG migliora concretamente le risposte; ≥ 3.5 = miglioramento forte
- **Effetto model-agnostic:** se il win rate GraphRAG è > 60% per almeno 2 modelli su 3, il valore dell'architettura è confermato indipendentemente dal provider LLM. (NB: l'ampliamento del dato nella knowledge graph è anche un fattore da tenere in considerazione - le nostre KG al momento sono medie basse)

### 0.5 Nota implementativa — Frontend di valutazione (Lovable)

Il digital twin di AIxLearning attualmente in sviluppo su Lovable (ref. ClickUp task "Aumentare spiegabilità dell'output") può servire come interfaccia di valutazione per la Fase 0. Le funzionalità di explainability in fase di sviluppo (visualizzazione delle fonti KG, primary/supporting methodologies, tracciabilità delle risposte) dimostrano direttamente il vantaggio di GraphRAG che la Fase 0 mira a misurare.

**Requisiti per la validità scientifica della valutazione:**
- (Se possibile) Il frontend deve supportare due modalità: **(A) LLM Alone** (query inviata direttamente all'LLM con prompt generico, senza contesto KG) e **(B) LLM + GraphRAG** (query inviata tramite API GraphRAG con system prompt + response template + `kg_context_formatted`). Altrimenti fare focus su come utilizzare quelle info per l'explainability.
- Le risposte devono essere presentate side-by-side, anonimizzate (il valutatore non deve sapere quale è A e quale è B)
- Per la valutazione controllata, i modelli LLM nel frontend devono corrispondere a quelli definiti nella Sezione 0.1 (GPT-4o, Claude 4.6, Gemini 2.5 or 3)

---

## FASE 1 — Preparazione del Benchmark per Confronto Modelli (1 giorno)

### 1.1 Costruzione del set di query

I domain experts selezionano 25 query rappresentative, distribuite come segue:

| Categoria | N. query | Esempio |
|-----------|----------|---------|
| Domande dirette su concetti | 5 | "Cos'è la metacognizione?" |
| Domande su relazioni tra concetti | 5 | "Qual è la differenza tra motivazione intrinseca ed estrinseca?" |
| Domande su strategie didattiche | 5 | "Come posso insegnare le frazioni a studenti con discalculia?" |
| Domande complesse multi-hop | 5 | "Quali strategie di scaffolding supportano l'autoregolazione in studenti ADHD?" |
| Domande edge-case / fuori dominio | 5 | "Come si calcola un integrale?" (test fallback) |

### 1.2 Raccolta risposte

Per ciascuna query, eseguire la pipeline completa (GraphRAG + LLM Assistant) con entrambi i modelli (stessi parametri, stesso KG), salvando:
- La risposta finale generata dall'LLM Assistant
- Il `cypher_query` prodotto dal GraphRAG
- I `raw_nodes` recuperati dal Knowledge Graph
- Il `confidence_level`
- Il `processing_time_ms`

---

## FASE 2 — Valutazione A/B Cieca (2–3 giorni)

### Fondamento scientifico
Ispirato alla metodologia Chatbot Arena (LMSYS, 800K+ voti, 90+ modelli — arXiv:2403.04132). Utilizza confronto pairwise anonimizzato con modello statistico Bradley-Terry. La ricerca conferma che i voti crowdsourced sono allineati con le valutazioni degli esperti.

### 2.1 Setup
- Le 25 coppie di risposte vengono presentate ai domain experts in un foglio Google/Excel
- L'ordine (Risposta A / Risposta B) è randomizzato per ogni query
- I nomi dei modelli sono nascosti (anonimizzati)

### 2.2 Per ogni coppia, l'esperto indica:

| Campo | Opzioni |
|-------|---------|
| Preferenza globale | A migliore / B migliore / Parità |
| Motivazione (facoltativa) | Testo libero, 1 frase |

### 2.3 Analisi
- Win rate per modello (% di vittorie su 25 query)
- Win rate per categoria di query
- Se win rate è tra 40–60%, la differenza non è significativa (con 25 query)
- Se win rate è > 65%, c'è una differenza significativa

---

## FASE 3 — Valutazione con Rubrica Likert (in parallelo con Fase 2)

### Fondamento scientifico
Basato su framework PEARL (MDPI Information, 2024) e studi su rubric-based assessment in contesti educativi (arXiv:2510.06253), che mostrano forte concordanza tra valutazione con rubrica e giudizio degli esperti.

### 3.1 Ogni risposta viene valutata singolarmente su 5 criteri (scala 1–5):

| Criterio | 1 (Insufficiente) | 3 (Sufficiente) | 5 (Eccellente) |
|----------|-------------------|------------------|-----------------|
| **Pertinenza** | Non risponde alla domanda | Risponde parzialmente | Risponde in modo preciso e completo |
| **Correttezza scientifica** | Contiene errori fattuali | Sostanzialmente corretto, alcune imprecisioni | Scientificamente rigoroso |
| **Struttura didattica** | Nessuna struttura pedagogica | Struttura presente ma incompleta | Segue modello I Do/We Do/You Do con coerenza |
| **Completezza** | Mancano strategie chiave | Copre i punti principali | Copre tutti gli aspetti rilevanti con esempi |
| **Adeguatezza al contesto italiano** | Non adattato | Parzialmente adattato | Perfettamente calato nel contesto scolastico italiano |

### 3.2 Analisi
- Media per criterio per ciascun modello
- Delta medio per criterio (dove un modello eccelle rispetto all'altro)
- Inter-rater agreement (Cohen's kappa) se ci sono 2+ valutatori

---

## FASE 4 — Metriche Automatiche RAGAS (automatizzato)

### Fondamento scientifico
RAGAS (Retrieval-Augmented Generation Assessment) è un framework open-source per la valutazione di pipeline RAG senza risposte di riferimento (docs.ragas.io). Internamente usa un LLM come "giudice" per decomporre e valutare le risposte.

### 4.1 Per ogni query, calcolare automaticamente:

| Metrica | Cosa misura | Soglia accettabile |
|---------|-------------|-------------------|
| Faithfulness | Le affermazioni nella risposta sono supportate dal contesto KG? | ≥ 0.85 |
| Answer Relevancy | La risposta affronta effettivamente la domanda? | ≥ 0.80 |
| Context Precision | I nodi recuperati dal KG sono pertinenti? | ≥ 0.70 |

### 4.2 Limitazioni note
- Queste metriche usano internamente un LLM come "giudice" (non sono puramente algoritmiche), creando una dipendenza circolare (LLM che giudica LLM)
- La metrica di Faithfulness non verifica la correttezza fattuale assoluta, ma solo la coerenza con il contesto recuperato
- Utili come segnale di monitoraggio continuo e per rilevare regressioni, **non sostituiscono la valutazione umana** (Fasi 2 e 3)

### 4.3 Uso in produzione
Queste metriche possono essere calcolate automaticamente per ogni query, creando un dashboard di monitoraggio continuo della qualità.

---

## FASE 5 — Report Decisionale

### 5.1 Tabella riassuntiva

| Dimensione | GPT-4o | Modello B | Delta |
|------------|--------|-----------|-------|
| Win rate A/B (%) | — | — | — |
| Pertinenza media (1–5) | — | — | — |
| Correttezza scientifica media | — | — | — |
| Struttura didattica media | — | — | — |
| Completezza media | — | — | — |
| Adeguatezza IT media | — | — | — |
| RAGAS Faithfulness | — | — | — |
| RAGAS Answer Relevancy | — | — | — |
| Tempo medio (ms) | — | — | — |
| Costo stimato per query | — | — | — |

### 5.2 Criteri di decisione
- Se il Modello B vince su A/B test con > 65% **E** punteggi Likert medi più alti su ≥ 3/5 criteri → **Migrare**
- Se il Modello B vince su A/B test con 50–65% → **Differenza non significativa**, valutare costo/beneficio
- Se il Modello B perde → **Mantenere GPT-4o**

---

## Timeline stimata

| Fase | Durata |
|------|--------|
| Fase 0: Validazione architetturale multi-modello (LLM Alone vs LLM + GraphRAG) | 3–4 giorni |
| Fase 1: Preparazione benchmark confronto modelli | 1 giorno |
| Fase 2: A/B test cieco | 2–3 giorni |
| Fase 3: Rubrica Likert | In parallelo con Fase 2 |
| Fase 4: RAGAS | Automatico (1 ora) |
| Fase 5: Report decisionale | 1 giorno |
| **Totale** | **~9–11 giorni** |

> **Nota:** La Fase 0 può essere eseguita indipendentemente dalle Fasi 1–5. Il risultato della Fase 0 è un prerequisito per comunicare agli stakeholder il valore dell'architettura GraphRAG. Bonus: i risultati della Fase 0 forniscono anche dati preliminari sul confronto tra modelli (B1 vs B2 vs B3), che possono informare le Fasi 1–5 o ridurne la portata se i risultati sono già conclusivi.

---

## Riferimenti

- **Ablation methodology for RAG** — arXiv:2411.19463 (Understanding Design Decisions of RAG Systems)
- **ARES** — arXiv:2311.09476 (NAACL 2024, Automated RAG Evaluation: Context Relevance, Answer Faithfulness, Answer Relevance)
- **RAG vs GraphRAG systematic evaluation** — arXiv:2502.11371 (2025, unified evaluation protocol)
- **RAGAS** — docs.ragas.io (open source, Apache 2.0)
- **LMSYS Chatbot Arena** — arXiv:2403.04132 (NeurIPS 2024)
- **PEARL Framework** — MDPI Information, 2024
- **LLM-as-Judge biases** — ICLR 2025, ACL/IJCNLP 2025
- **Educational rubric assessment** — arXiv:2510.06253
- **Human evaluation as gold standard for domain-specific RAG** — RSC Advances, 2026
