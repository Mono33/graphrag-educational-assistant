# AIxLearning — Design Handoff per UI di Produzione

> **Documento di consegna dal Team AI al team Design/Product di FEM**
> per portare il prodotto AIxLearning Agentic GraphRAG da prototipo
> tecnico (Mirror Stack) a interfaccia utente production-ready destinata
> agli insegnanti.
>
> **Autore:** Team AI (Louis Mono, Angelo Casali)
> **Data:** maggio 2026
> **Versione:** 1.0
> **Stato:** DRAFT — in attesa di feedback Design/Product/Direzione

---

## 0. Indice

1. [Executive summary](#1-executive-summary)
2. [Contesto del prodotto](#2-contesto-del-prodotto)
3. [Cosa esiste oggi (Mirror Stack MVP)](#3-cosa-esiste-oggi-mirror-stack-mvp)
4. [Perché serve un designer (gap analysis)](#4-perché-serve-un-designer-gap-analysis)
5. [Personas](#5-personas)
6. [Mappa schermate e stati attuali](#6-mappa-schermate-e-stati-attuali)
7. [Vincoli tecnici (non negoziabili)](#7-vincoli-tecnici-non-negoziabili)
8. [Cosa fornisce il Team AI al Designer](#8-cosa-fornisce-il-team-ai-al-designer)
9. [Cosa deve produrre il Designer](#9-cosa-deve-produrre-il-designer)
10. [Workflow proposto e timeline](#10-workflow-proposto-e-timeline)
11. [Definition of Done per la produzione](#11-definition-of-done-per-la-produzione)
12. [Decisioni aperte (per Product/Direzione)](#12-decisioni-aperte-per-productdirezione)
13. [Appendici](#13-appendici)

---

## 1. Executive summary

Il Team AI ha completato un MVP funzionante (chiamato internamente
**Mirror Stack**) della web app **AIxLearning Agentic GraphRAG**. L'MVP
è stato costruito con uno stack server-side (FastAPI + Jinja2 + htmx +
WebAwesome 3.x + Tailwind) **con l'obiettivo esplicito di validare
end-to-end la pipeline agentica** — non di essere il prodotto finale.

L'MVP oggi:
- copre tutti i flussi essenziali (login → creazione lezione → chat con
  agente → editing profilo → esportazione lezione);
- è esposto via FastAPI e raggiungibile a `http://127.0.0.1:8765/webui/`
  in locale;
- dispone di una API pubblica (`/api/v1/*`) e di un server MCP
  (`/mcp/`) — entrambi documentati e riusabili da altre interfacce.

**Cosa manca per essere customer-facing in produzione:**
una UI/UX progettata da un designer, con design system coerente,
copywriting curato per insegnanti italiani, accessibilità WCAG 2.1 AA,
responsive mobile, e onboarding.

Questo documento descrive **come dividere il lavoro** tra Team AI
(che continua a evolvere il backend, l'agente e l'API) e Design/Product/Dev
(che prende in mano l'esperienza utente fino al deploy in produzione).

---

## 2. Contesto del prodotto

### 2.1 Cosa fa AIxLearning

Una piattaforma web in cui un **insegnante italiano** (scuola
primaria, secondaria di I e II grado) può:

1. Inserire il **profilo educativo** della propria classe (numero
   studenti, livello scolastico, BES presenti, caratteristiche, aula
   disponibile).
2. Scrivere una richiesta in linguaggio naturale (es. *"Crea una lezione
   di 45 minuti sulla fotosintesi clorofilliana adattata a una classe
   con 2 studenti DSA"*) e/o caricare materiali di contesto (PDF, TXT).
3. Ricevere un **piano di lezione personalizzato** generato da una
   pipeline agentica (Planner → Retriever → Writer → Critic) che
   recupera contenuti dal Knowledge Graph del dominio scelto (Neuro o
   UDL) e cita risorse multimediali curate.
4. Esportare la lezione in Markdown, TXT o PDF per usarla in classe.

### 2.2 Domini supportati (ad ora - estesi in futuro)

- **Neuro** — Neurodidattica (neuroscienze cognitive applicate
  all'apprendimento, BES, autoregolazione, mindset, ecc.).
- **UDL** — Universal Design for Learning (3 principi UDL, 9 linee
  guida, 31 checkpoint).

### 2.3 Stack tecnologico (MVP attuale)

| Layer | Tecnologia | Note |
|---|---|---|
| Frontend | Jinja2 + htmx 2 + WebAwesome 3.x + Tailwind CSS + Alpine.js | Server-side rendering, no SPA |
| Backend | FastAPI + uvicorn | Python 3.11+ |
| Database (app) | SQLite via SQLAlchemy async (Postegrl per prod) | utenti, lezioni, file caricati |
| Knowledge Graph | Neo4j 5.x | due grafi: `neuro` + `udl` |
| LLM | OpenRouter (Claude / GPT) | per Planner/Writer/Critic |
| Auth | FastAPI-Users (cookie + JWT Bearer) | email/password |
| Observability | Langfuse + GlitchTip | prompts e errori |
| Streaming | Server-Sent Events (SSE) | progressivo update chat |

> **L'API REST e il server MCP sono già pronti per essere consumati da
> qualunque altro frontend** (es. una React SPA, Lovable, app mobile
> nativa) — quindi se il design proposto richiedesse cambio di stack
> frontend, il backend non andrebbe rifatto.

---

## 3. Cosa esiste oggi (Mirror Stack MVP)

### 3.1 URL di riferimento

- **MVP locale:** `http://127.0.0.1:8765/webui/`
- **Repo GitHub:** `FEM-modena/graphrag-aixlearning` (branch `chore/repo-reorg`)
- **API docs interattive:** `http://127.0.0.1:8765/docs`
- **MCP Streamable HTTP:** `http://127.0.0.1:8765/mcp/`

Per accedere al deploy condiviso: chiedere al Team AI le istruzioni
(Cloudflare Tunnel / ngrok / VPS — vedi `docs/integrations/MCP_Setup.md`
sezione "Production deployment notes").

### 3.2 Mappa rotte WebUI

| Metodo | Path | Scopo |
|---|---|---|
| `GET` | `/webui/` | Home (redirect a /lessons o /login) |
| `GET` | `/auth/login` `/auth/register` | Login / signup |
| `POST` | `/auth/login` `/auth/register` `/auth/logout` | |
| `GET` | `/webui/lessons` | Lista delle lezioni dell'insegnante |
| `DELETE` | `/webui/lesson/{id}` | Elimina una lezione |
| `GET` | `/webui/lesson/new` | Form: nuovo profilo educativo + lezione |
| `POST` | `/webui/lesson` | Crea la lezione |
| `GET` | `/webui/lesson/{id}` | Workspace chat (3 colonne) |
| `POST` | `/webui/lesson/{id}/run` | Avvia generazione agente |
| `GET` | `/webui/lesson/{id}/stream` | SSE in arrivo dall'agente |
| `POST` | `/webui/lesson/{id}/upload` | Carica file di contesto |
| `DELETE` | `/webui/lesson/{id}/upload/{fid}` | Rimuovi file caricato |
| `GET` | `/webui/lesson/{id}/profile` | Sidebar profilo (read-only) |
| `GET` | `/webui/lesson/{id}/profile/edit` | Sidebar profilo (form) |
| `POST` | `/webui/lesson/{id}/profile` | Salva modifiche profilo |
| `GET` | `/webui/lesson/{id}/export?format=md\|txt` | Download |
| `GET` | `/webui/lesson/{id}/print` | Pagina print-friendly (per PDF) |

### 3.3 Schermate principali (vedi screenshot in `assets/`)

1. **Home / Login / Register**
2. **Le mie lezioni** (lista a card)
3. **Nuova lezione** (form lungo: titolo, dominio, materia, argomento,
   durata, classe, BES, caratteristiche, attributi studenti, aula)
4. **Chat workspace** — layout a 3 colonne:
   - Sinistra: **Profilo educativo** (sidebar collassabile, in modalità
     view o edit inline)
   - Centro: **Chat** (input testuale + paperclip per allegati,
     bolle conversazione, card progressive Planner → Retriever → Writer
     → Critic, card finale lezione con pulsanti download)
   - Destra: **Risorse multimediali** (sidebar con risorse curate
     dal Retriever Agent: video YouTube, OER, paper accademici,
     diagrammi Mermaid)
5. **Pagina print-friendly** (per export PDF via stampa browser)

### 3.4 Stati per ogni schermata

Per ogni schermata sopra esistono almeno 4 stati che il designer dovrà
ripensare e illustrare in Figma:

| Stato | Esempio |
|---|---|
| **Empty** | Nessuna lezione ancora creata → mostra CTA "Crea la prima lezione" |
| **Loading** | Agente in esecuzione (60–90s) → progress, spinner, card progressive |
| **Error** | Agente fallito, parsing JSON fallito, KG offline, OpenRouter timeout |
| **Success** | Lezione completa con tutte le card popolate |
| **Partial** | Pianificazione fatta, retriever in corso, writer in attesa |

### 3.5 Modello dati — `EducationalProfile`

Lo schema condiviso tra UI e API (Pydantic). Il designer deve conoscerlo
perché tutti i campi devono apparire in form e sidebar:

```python
EducationalProfile {
    subject_area: str          # es. "Scienze"
    specific_topic: str        # es. "Fotosintesi clorofilliana"
    time_available_minutes: int  # 15..480
    group: EducationalGroup {
        title: str             # es. "3A Liceo Scientifico"
        students_number: int   # 1..40
        grade: GradeLevel      # 6 enum (Infanzia → Università)
        disabilities: list[DisabilityType]    # 10 enum BES
        class_features: list[ClassFeature]    # 5 enum
        student_attributes: list[StudentAttribute]  # 6 enum
    }
    classroom: ClassroomEnvironment {
        title: str              # es. "Aula 101"
        forniture_mobility: enum    # NO/Parzialmente/Sì
        own_device: enum            # NO/Solo BES/Sì
        has_lim, has_wifi, has_suite, pc_station: bool
    }
}
```

**Enum localizzati** in italiano (le label sono già definite in
`src/aix/webui/lessons/labels.py`).

---

## 4. Perché serve un designer (gap analysis)

### 4.1 Cosa l'MVP fa bene

- ✅ Tutti i flussi funzionano end-to-end
- ✅ Streaming progressivo (l'utente vede l'agente "pensare")
- ✅ Profilo educativo strutturato e validato
- ✅ Esportazione MD/TXT/PDF
- ✅ Richiamo asincrono e idempotente
- ✅ Auth, sessioni, multi-tenant per insegnante

### 4.2 Cosa manca per essere customer-facing

| Area | Gap |
|---|---|
| **Visual design** | Nessun design system, palette ad-hoc, tipografia di default |
| **Brand** | Nessuna brand guideline AIxLearning / FEM coerente, no logo dedicato |
| **UX writing** | Copy provvisori, tono non testato con insegnanti reali |
| **Onboarding** | Non esiste una procedura di benvenuto / tour guidato |
| **Responsive mobile** | Layout pensato desktop-first, sidebar fissa; mobile non testato sotto 768px |
| **Accessibilità** | No annotazioni WCAG, contrasto non verificato, focus states web component default, no test screen reader |
| **Empty states** | Stati vuoti spartani, mancano illustrazioni / CTA progettate |
| **Error states** | Errori dell'agente ("Planner JSON parse failed") esposti grezzi all'utente |
| **Microinterazioni** | Loader generici, niente skeleton, niente toast curati |
| **Iconografia** | Mix di icone WebAwesome / emoji — non coerente |
| **Pagina print** | Funziona ma non è progettata graficamente |
| **Dashboard insegnante** | Non esiste analytics o cronologia avanzata (filtri, ricerca, tag) |

### 4.3 Conclusione

Il Mirror Stack è la **prova che l'agente funziona**. Per il deploy in
produzione serve trasformarlo in un prodotto che un insegnante prende in
mano senza istruzioni, comprende, si fida, e usa quotidianamente.
Quel salto richiede il lavoro di un designer di prodotto.

---

## 5. Personas

> Le personas sono basate sul target dichiarato di FEM AIxLearning
> (insegnanti italiani K-12) ma vanno **validate con il team Product** e
> raffinate dal designer con interviste a insegnanti reali.

### 5.1 Maria — la docente di scienze esperta

- **Età:** 45 — **Ruolo:** docente liceo scientifico (3A)
- **Tech literacy:** media (usa Drive, qualche LIM, niente AI)
- **Pain points:** poco tempo per preparare lezioni inclusive, frustrata
  da generatori AI generici che non considerano BES specifici
- **Successo:** in 5 minuti ottiene un piano di lezione di 45 minuti
  che cita fonti, suggerisce un video YouTube, e include differenziazione
  per i suoi 2 studenti DSA

### 5.2 Lucia — la docente di sostegno inclusione-first

- **Età:** 32 — **Ruolo:** docente di sostegno scuola media
- **Tech literacy:** alta
- **Pain points:** vuole strumenti UDL strutturati, non improvvisazione
- **Successo:** apre il dominio UDL, costruisce strategie di engagement
  per uno studente ADHD + uno DCGL nello stesso gruppo

### 5.3 Giovanni — il coordinatore didattico

- **Età:** 50 — **Ruolo:** funzione strumentale, supervisione di un
  Consiglio di classe
- **Tech literacy:** medio-bassa
- **Pain points:** vuole materiali condivisibili tra docenti, esportabili
  in PDF, riconducibili a riferimenti scientifici
- **Successo:** scarica un PDF con citazioni e lo distribuisce al
  collegio docenti

> Nota: il target potrebbe estendersi a **tutor private**, **formatori
> aziendali** (FORMAZIONE_SUL_LAVORO è uno dei livelli supportati),
> **docenti universitari**. Da decidere con Product.

---

## 6. Mappa schermate e stati attuali

Il designer riceverà uno **screenshot pack** completo (vedi §8). Qui un
riepilogo testuale per orientarsi:

```
HOME (/webui/)
 └── se non loggato → /auth/login
 └── se loggato     → /webui/lessons

AUTH
 ├── /auth/login          [empty / error: credenziali errate / success]
 └── /auth/register       [empty / error: validation / success]

LESSONS LIST (/webui/lessons)
 ├── empty: 0 lezioni     → CTA "Crea la prima lezione"
 ├── loaded: N card       → griglia, ognuna con stato draft/in-progress/complete
 └── action: delete       → conferma + ricarica lista

NEW LESSON (/webui/lesson/new)
 ├── form lungo: titolo, dominio, materia, argomento, durata,
 │   classe (nome, n°, livello, BES, caratteristiche, attributi),
 │   aula (nome, mobilità, BYOD, dotazioni)
 ├── validation errors    → callout rosso in alto
 └── success              → redirect a /webui/lesson/{id}

CHAT WORKSPACE (/webui/lesson/{id})
 ├── 3-COLUMN LAYOUT (desktop)
 │   ├── LEFT  Profile sidebar (view / edit inline)
 │   ├── CENTER Chat (input + bolle + card agentiche progressive)
 │   └── RIGHT Media panel (popolato dal Retriever)
 ├── stati chat:
 │   ├── empty                   → "Inizia la conversazione"
 │   ├── user-message-sent       → bolla utente + spinner agente
 │   ├── planner-running         → card Planner con badge "in corso"
 │   ├── planner-done            → card Planner mostra intent + queries
 │   ├── retriever-running       → card Retriever
 │   ├── retriever-done          → card Retriever + media panel popolato
 │   ├── writer-pending          → placeholder
 │   ├── writer-done             → card Writer streaming markdown
 │   ├── critic-running          → card Critic
 │   ├── critic-done             → card Critic con score
 │   ├── lesson-card-final       → card finale con download MD/TXT/PDF
 │   └── error                   → card rossa con messaggio
 └── action: aggiorna profilo    → swap sidebar in edit mode
 └── action: upload allegato     → paperclip → file appare sotto input

PRINT VIEW (/webui/lesson/{id}/print)
 └── single-page version, ottimizzata per Ctrl+P → PDF
```

---

## 7. Vincoli tecnici (non negoziabili)

Il designer **deve progettare entro questi vincoli**, altrimenti il
costo di implementazione esplode:

### 7.1 Stack frontend fisso

- **Server-Side Rendering (Jinja2 + htmx)** — niente React/Vue/Svelte.
  Le pagine sono generate dal server e htmx fa swap parziali.
- **WebAwesome 3.x** è la component library di base
  (https://webawesome.com/docs/components). Il designer **deve
  scegliere le sue primitive UI da quella libreria** (wa-button,
  wa-card, wa-input, wa-select, wa-tag, wa-callout, wa-icon, wa-tooltip,
  wa-tabs, wa-dialog, wa-checkbox, wa-radio, wa-progress, wa-spinner,
  wa-avatar, wa-tree, ecc.) o segnalare componenti mancanti.
- **Tailwind CSS** per layout, spaziatura, responsive, custom styling.
- **Alpine.js** per micro-interazioni dichiarative (mostra/nascondi,
  toggle).
- **Animazioni:** ridotte, solo CSS (transition, animation), no
  JavaScript-heavy animation libraries.

### 7.2 Lingua

- **Italiano-first** per tutto il copy utente.
- Inglese ammesso solo nei termini tecnici settoriali (es. "feedback",
  "scaffolding") già usati negli ambienti scolastici.
- Tono: **professionale, empatico, concreto**. Mai paternalistico.
- Da definire con il designer: tono "tu" o "lei" per gli insegnanti?
  (raccomandazione AI team: "tu", come Google for Education).

### 7.3 Accessibilità

- **Target WCAG 2.1 livello AA** (legge italiana — Stanca, AGID).
- Contrasto testo/sfondo ≥ 4.5:1 (3:1 per testo grande).
- Focus state visibile su tutti i controlli interattivi.
- Tutti i form etichettati (`label`/`aria-label`).
- Componenti web (WebAwesome) sono già accessibili di base ma vanno
  verificati con screen reader (NVDA / VoiceOver).
- Modalità scura: opzionale ma consigliata (Tailwind ha `dark:` variants).
- Dimensioni touch target ≥ 44x44 px su mobile.

### 7.4 Mobile responsive

- **Desktop (≥1280px):** layout 3-colonne completo.
- **Tablet (768–1279px):** sidebar collassabili, chat al centro.
- **Mobile (360–767px):** stack verticale, sidebar in drawer / modal.
- **Print:** pagina dedicata `/webui/lesson/{id}/print`.

### 7.5 Performance

- L'agente impiega **60–90 secondi** in caso medio. Il design DEVE
  rendere percepibile il progresso (no spinner generico per 90s).
- Tempo di first paint ≤ 1s.
- Niente immagini > 200 KB se non necessario.
- Niente font esterni pesanti — usare system stack o un singolo
  webfont leggero (es. Inter via font-display: swap).

### 7.6 Browser support

- Chrome / Edge / Firefox / Safari — ultime 2 versioni stable
  (consistente con WebAwesome 3.x).
- Niente IE11.
- Mobile: Safari iOS ≥ 16, Chrome Android ≥ recente.

### 7.7 Sicurezza & privacy

- Dati insegnanti / studenti: schema `EducationalProfile` non contiene
  PII di studenti (solo aggregati: numero, BES tipologia, ecc.).
- File caricati: usati solo come contesto, mai inviati al KG.
- GDPR: il design del flusso registrazione deve includere consenso
  trattamento dati.

---

## 8. Cosa fornisce il Team AI al Designer

> **Inputs** che il Team AI consegna al designer prima dell'inizio del
> lavoro. Tutto già materialmente disponibile o producibile in 1-2
> giorni.

| # | Deliverable | Formato | Stato |
|---|---|---|---|
| 1 | **Screenshot pack completo** — ogni schermata in ogni stato (empty/loading/error/success), desktop + mobile | PNG @ 2x in `docs/design/screenshots/` | DA PRODURRE |
| 2 | **Live URL del MVP** — accesso allo staging condiviso | Cloudflare Tunnel / ngrok | DA CONFIGURARE |
| 3 | **Sitemap & route inventory** — tutte le rotte WebUI | Markdown (questa sezione §3.2) | DONE |
| 4 | **Schema dati** — `EducationalProfile`, `Lesson`, `Classroom`, `Group` esportati come JSON Schema | JSON Schema generato da Pydantic | DA AUTOMATIZZARE |
| 5 | **Catalogo stati** — empty/loading/error/success per ogni schermata (vedi §6) | Markdown + screenshot | DA PRODURRE |
| 6 | **API spec** — `/openapi.json` + inventario tool MCP | Swagger UI live + `docs/integrations/MCP_Setup.md` | DONE |
| 7 | **Sample copy attuale** — tutti gli stringhi italiani usati nei template Jinja | i18n catalog (es. `.po` o flat JSON) | DA ESTRARRE |
| 8 | **Esempio output reale** — 3-5 piani di lezione completi prodotti dall'agente, per dominio Neuro e UDL | Markdown export | DA SELEZIONARE |
| 9 | **Performance budget reale** — distribuzione tempi di esecuzione agente (P50, P90, P99) | CSV / dashboard Langfuse | DONE (raccolta in corso) |
| 10 | **Audit di accessibilità baseline** — risultati Lighthouse + axe-core sull'MVP attuale | report HTML | DA ESEGUIRE |
| 11 | **Glossario tecnico-pedagogico** — Neuro, UDL, BES, ZDP, scaffolding, ecc. | Markdown bilingue | DA SCRIVERE |
| 12 | **Lista personas iniziali** — bozza in §5, da raffinare con interviste | Markdown | DRAFT |
| 13 | **Brand assets esistenti** — logo FEM, palette, font (se disponibili) | da chiedere a FEM Comunicazione | DEPENDS |

> **Tempo stimato Team AI** per produrre 1+5+7+8+10+11: **2-3 giorni
> uomo**. Schedulabile la settimana prima del kickoff design.

---

## 9. Cosa deve produrre il Designer

### 9.1 Deliverable richiesti (in ordine di priorità)

| # | Deliverable | Formato | Per chi serve |
|---|---|---|---|
| 1 | **UX audit** dell'MVP attuale: heuristic evaluation (Nielsen 10) + lista priorità | Doc / Slide | AI team + Product |
| 2 | **Information architecture** rivista: sitemap, navigazione, tassonomia | Diagramma + Doc | AI team + Product |
| 3 | **Wireframe low-fi** di tutti i flussi (desktop + mobile) | Figma frame | AI team |
| 4 | **Design system** in Figma: token (colori, spacing, type scale, shadow, radius), iconografia, componenti mappati su WebAwesome 3.x | Figma library | AI team |
| 5 | **Mockup hi-fi** desktop e mobile per ogni schermata × ogni stato | Figma frame | AI team |
| 6 | **Prototipo cliccabile** dei flussi principali (login → nuova lezione → chat → export) | Figma prototype | User testing + AI team |
| 7 | **Annotazioni accessibilità** WCAG 2.1 AA (contrasto, focus, ARIA, screen reader notes) | Inline in Figma | AI team |
| 8 | **Spec di interazione** (microinterazioni, transizioni, skeleton, toast) | Doc / Figma | AI team |
| 9 | **Asset export** (logo, illustrazioni empty state, icone custom se servono) | SVG + PNG | AI team |
| 10 | **Copy review** — revisione di tutti gli strings italiani (tono, chiarezza) | Doc condiviso | AI team |
| 11 | **Onboarding flow** — primo accesso, tour guidato, tooltip di benvenuto | Figma + spec | AI team |
| 12 | **Print template** — design pagina lezione stampata | Figma + spec print CSS | AI team |
| 13 | **Brand guidelines** AIxLearning (se non esistono): logo lockup, palette, tipografia, voce | PDF / Figma | tutti |

### 9.2 Responsabilità trasversali del designer

- **Garantire coerenza visiva** tra tutte le schermate.
- **Documentare ogni decisione** di design con razionale (perché questa
  scelta? perché questo colore? perché questo flusso?).
- **Validare con utenti reali**: almeno **3 sessioni di user testing**
  con insegnanti del target prima del go-live.
- **Iterare** sul feedback Team AI durante implementazione (non
  consegnare e sparire).
- **Rispettare i vincoli tecnici** in §7 (verificare con AI team prima di
  proporre componenti non disponibili in WebAwesome).

---

## 10. Workflow proposto e timeline

> Stima conservativa per un lavoro **part-time** del designer
> (4 ore/giorno). Comprime a metà se full-time.

```
SETTIMANA 0  ── Kickoff (4h)
                Team AI + Designer + Product. Allineamento su scope,
                personas, vincoli, timeline. Consegna del pacchetto §8.

SETTIMANA 1-2 ── DISCOVERY (Designer)
                ├── UX audit
                ├── Personas review (con interviste a 3-5 insegnanti)
                ├── Information architecture
                ├── Sitemap + flow chart
                └── Output: doc audit + IA proposal
                CHECKPOINT: Team AI + Product approvano IA

SETTIMANA 3-4 ── WIREFRAME + DESIGN SYSTEM (Designer)
                ├── Wireframe low-fi tutti i flussi
                ├── Design system / Figma library
                └── Output: Figma file con wireframe + tokens
                CHECKPOINT: Team AI verifica fattibilità tecnica
                            (componenti WebAwesome, layout htmx)

SETTIMANA 5-7 ── MOCKUP HI-FI (Designer)
                ├── Mockup desktop di tutte le schermate × stati
                ├── Mockup mobile
                ├── Annotazioni accessibilità
                ├── Spec interazione
                └── Output: Figma completo + spec doc
                CHECKPOINT: User testing prototipo (3 insegnanti)

SETTIMANA 8   ── REFINEMENT (Designer)
                ├── Iterazione su feedback user testing
                ├── Copy review finale
                └── Asset export

SETTIMANA 9-12 ── IMPLEMENTAZIONE (Team AI)
                ├── Port design system in Tailwind config
                ├── Update tutti i template Jinja
                ├── QA cross-browser + mobile
                ├── Audit accessibilità finale (axe-core, Lighthouse)
                └── Output: PR su staging

SETTIMANA 13  ── UAT + GO-LIVE
                ├── Test con 5-10 insegnanti reali
                ├── Bug fixes
                └── Deploy in produzione
```

**Totale:** ~13 settimane (3 mesi) calendar time, **part-time** Designer
+ Team AI in parallelo su altri task. Comprimibile a 6-8 settimane se
sia Designer che AI team lavorano full-time sul progetto.

---

## 11. Definition of Done per la produzione

Una schermata è considerata "production-ready" quando soddisfa **tutti**
i seguenti criteri:

### 11.1 Design

- [ ] Mockup Figma approvati da Product + Team AI
- [ ] Tutti gli stati (empty/loading/error/success/partial) progettati
- [ ] Versione mobile ≤ 360px progettata
- [ ] Annotazioni accessibilità presenti
- [ ] Copy italiano revisionato

### 11.2 Implementazione

- [ ] Template Jinja allineato al mockup pixel-perfect (±2px)
- [ ] Componenti WebAwesome usati in modo idiomatico
- [ ] Layout responsive testato a 360 / 768 / 1024 / 1280 / 1920 px
- [ ] Niente regressioni nei test esistenti (`pytest`)
- [ ] htmx swap funzionano senza FOUC

### 11.3 Qualità

- [ ] **Lighthouse** ≥ 90 su Performance, Accessibility, Best Practices, SEO
- [ ] **axe-core**: 0 issue critici / serious
- [ ] **Test manuale screen reader** (NVDA su Windows o VoiceOver su Mac)
- [ ] Cross-browser test: Chrome, Edge, Firefox, Safari (ultime 2 versioni)
- [ ] Test su iPhone Safari + Android Chrome (browser stack o device reali)

### 11.4 Contenuti

- [ ] Tutti gli error message italiani (no leak di errori tecnici inglesi)
- [ ] Tooltips e help text presenti per campi non ovvi
- [ ] Empty state con CTA chiara
- [ ] Loading state con messaggio contestuale (es. "L'agente sta
      cercando contenuti rilevanti…" invece di spinner generico)

### 11.5 User testing

- [ ] Almeno 3 sessioni di moderated testing con insegnanti del target
- [ ] Issue prioritari ≥ P2 risolti prima del go-live
- [ ] System Usability Scale (SUS) ≥ 70

---

## 12. Decisioni aperte (per Product/Direzione)

Domande che **non possiamo decidere internamente al Team AI** e che il
designer/product devono chiarire prima di iniziare:

1. **Brand identity AIxLearning**: esiste già un brand book FEM da
   seguire? O AIxLearning ha la sua identità visiva? Chi è il
   custode (Comunicazione FEM)?
2. **Logo**: chi lo disegna? Esiste già?
3. **Dominio in produzione**: `aixlearning.fem.digital`? `aix.fem.it`?
4. **Modello di accesso**:
   - libero con email/password?
   - SSO con account FEM esistente?
   - codice istituto / scuola?
5. **Multi-tenancy**: una scuola può avere più insegnanti che condividono
   il workspace? Oppure ogni insegnante è isolato (situazione attuale)?
6. **Modello di business**: gratuito? Freemium? Licenza scuola? Influenza
   pricing page, paywall, billing — quindi UI.
7. **Dati studenti**: l'`EducationalProfile` aggrega ma nominalmente è
   anonimo. Vogliamo permettere all'insegnante di **salvare profili
   classe** riusabili (es. "3A liceo") per non re-inserirli ogni volta?
8. **Storico lezioni**: cronologia infinita? Ricerca? Tag? Cartelle?
   Condivisione tra colleghi?
9. **Lingua**: italiano-only al lancio? Multilingua dopo (inglese,
   francese)?
10. **Mobile**: PWA / responsive web only? O app nativa (futura)?
11. **Integrazione FEM**: dovremo agganciarci a Moodle FEM, ai sistemi
    informativi delle scuole, a Google Classroom?
12. **Marketing**: serve una landing page pubblica oltre al webapp?
    (Probabile sì)

---

## 13. Appendici

### A. Documenti di riferimento

| Doc | Path | Scopo |
|---|---|---|
| `ClickUp_Agentic_GraphRAG_Update.md` | `docs/product/` | Roadmap tecnica completa CORE 0-5 |
| `HANDOFF_Angelo_FEM_Mirror_Stack.md` | `docs/product/` | Onboarding tecnico per developer FEM |
| `MCP_Setup.md` | `docs/integrations/` | Setup server MCP per integrazioni esterne |
| `langfuse_prompts_reference.md` | `docs/prompts/` | Mappa prompt agente |
| `last_fixes.md` | `docs/reports/` | Cambiamenti UI recenti |
| `educational_profile.py` | `src/aix/api/schemas/` | Schema Pydantic completo |
| `labels.py` | `src/aix/webui/lessons/` | Etichette italiane enum |

### B. Glossario

- **AIxLearning** — nome prodotto / progetto
- **Mirror Stack** — codename interno dell'MVP frontend (FastAPI + htmx)
- **Agentic GraphRAG** — pipeline agentica con accesso a Knowledge Graph
- **Neuro** — dominio Neurodidattica
- **UDL** — Universal Design for Learning
- **BES** — Bisogni Educativi Speciali (DSA, ADHD, ecc.)
- **ZDP** — Zona di Sviluppo Prossimale (Vygotsky)
- **Planner / Retriever / Writer / Critic** — i 4 agenti della pipeline
- **MCP** — Model Context Protocol (Anthropic) — protocollo per esporre
  tool dell'agente a client esterni (Claude Desktop, Cursor IDE, ecc.)
- **WebAwesome** — component library web components di Shoelace evolution
- **htmx** — libreria per swap HTML server-driven (no JSON, no SPA)
- **SSE** — Server-Sent Events, streaming dell'agente verso il browser
- **WCAG 2.1 AA** — standard accessibilità W3C, livello AA = obbligo IT
- **AGID Linee Guida Accessibilità** — implementazione italiana WCAG

### C. Contatti

| Ruolo | Persona | Reach |
|---|---|---|
| Tech lead AI | Louis Mono | louis.mono@fem.digital |
| AI engineer | Angelo Casali | angelo.casali@fem.digital |
| Product | _da assegnare_ | _ — _ |
| Designer | _da assegnare_ | _ — _ |
| Direzione | _ — _ | _ — _ |

---

## Stato del documento

| Versione | Data | Autore | Note |
|---|---|---|---|
| 1.0 | maggio 2026 | Team AI | Prima versione DRAFT — in attesa feedback Design/Product |

**Prossimi step:**
1. Condividere con Direzione / Product per approvazione scope
2. Identificare il designer (interno FEM o consulente)
3. Schedulare kickoff (settimana 0)
4. Team AI prepara il pacchetto di §8 nei 5 giorni precedenti il kickoff
