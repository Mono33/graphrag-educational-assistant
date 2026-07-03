# Prerequisiti per il Pilot Interno dell'Agente Agentic GraphRAG

**Da:** Team AI (LM)
**A:** Direzione FEM, Team DEV / IT
**Data:** 15 maggio 2026
**Oggetto:** Decisioni e prerequisiti necessari per il lancio del pilot interno
**Documento di riferimento:** `docs/product/Internal_Production_Deployment_Plan.md`

---

## Contesto

Il sistema Agentic GraphRAG è ora pronto per essere esposto internamente a un piccolo gruppo di domain expert FEM (~5-10 persone) attraverso un URL dedicato HTTPS. Prima di procedere con l'implementazione tecnica (~9-11 giorni-uomo, dettagliata nel piano `Internal_Production_Deployment_Plan.md`), abbiamo bisogno di alcune decisioni e autorizzazioni da parte di FEM.

**Il pilot è un'iniziativa separata e parallela all'integrazione con AixLearning** (documentata in `Dev_Handoff_AgenticGraphRAG_Integration.md`): non si tratta del plug-in nel frontend Django esistente, ma di un'istanza standalone della nostra applicazione (FastAPI + WebUI), ospitata su un server dedicato e accessibile solo a utenti FEM autorizzati. Servirà a validare il sistema con i domain expert prima del cutover sulla produzione AixLearning.

---

## Decisioni e prerequisiti richiesti

### 1. Server / VM di produzione

Per ospitare l'applicazione 24/7 ci serve una VM Linux (Ubuntu 24.04 LTS, minimo 4 GB RAM, 40 GB disco SSD, Docker-capable).

**Opzioni:**

| Opzione | Costo mensile | Note |
|---|---|---|
| **Hetzner Cloud — CX22** *(Germania, 4 GB RAM)* | **~€5/mese** *(~€60/anno)* | Provider serio, dati in UE, billing semplice |
| **Hetzner Cloud — CCX13** *(8 GB RAM)* | ~€13/mese | Più margine in caso di più utenti concorrenti |
| **Scaleway DEV1-M** *(Francia)* | ~€7/mese | Alternativa francese |
| **Server FEM interno** *(se IT ha capacità)* | **gratis** | Ownership completamente on-premise |

**Quello che ci serve:**

- Decidere quale opzione preferite. Il nostro consiglio: **Hetzner CX22 a ~€5/mese** (sufficiente per il pilot e pronto in 5 minuti), oppure un server FEM interno se disponibile.
- **L'account/VM deve essere intestato a FEM, non a una persona fisica**, per ragioni di billing, ownership e responsabilità GDPR (FEM è il data controller).
- Indicateci chi può autorizzare la spesa o assegnarci una macchina interna.

Per non bloccare lo sviluppo, possiamo procedere temporaneamente con un noleggio personale (Hetzner CX22 sul conto privato di LM), ma **prima del lancio del pilot l'account deve essere trasferito a FEM** (Hetzner supporta i trasferimenti gratuiti).

### 2. Dominio / URL pubblico

L'applicazione sarà raggiungibile via HTTPS al seguente URL pubblico:

**`https://agente.aiforlearning.digital`**

**Importante:** **non andiamo a modificare o sovrascrivere nulla di esistente.** Il DNS funziona così:

- `aiforlearning.digital` è il dominio padre (*apex*)
- `graph.aiforlearning.digital` è un sottodominio già esistente che punta al server Neo4j — **rimane invariato**
- Aggiungeremo un **nuovo sottodominio** `agente.aiforlearning.digital` che punterà al nostro server

I sottodomini sono completamente indipendenti tra loro: aggiungerne uno nuovo non tocca quelli esistenti.

**Quello che ci serve:**

**a) Conferma sulla proprietà del dominio `aiforlearning.digital`.**
Nel recente commit di Angelo (`990afde`, file `external_apis.py`, header OpenAlex polite-pool) compare l'email `angi36casali@gmail.com`. Vorremmo confermare:

- Il dominio è registrato a nome di FEM o a nome personale di Angelo?
- Se è personale, valutiamo se registrare un nuovo dominio corporate FEM (es. `agente-fem.it`, ~€20/anno) per sostenibilità a lungo termine — in tal caso il sottodominio scelto (`agente`) potrebbe diventare semplicemente `https://agente-fem.it` (apex) o un sottodominio del nuovo apex corporate.

**b) Accesso al pannello DNS.** Chi gestisce `aiforlearning.digital` (Cloudflare, Aruba, GoDaddy o altro)? La modifica richiesta è banale — un record DNS di tipo A:

```
Tipo:   A
Nome:   agente
Valore: <IP del nostro server>
TTL:    3600
```

Sono 30 secondi di lavoro per chi ha accesso al pannello.

### 3. FEM come data controller GDPR

L'applicazione tratta dati personali (account utente FEM, log delle conversazioni con l'agente, profili educativi della classe). Ai sensi del GDPR ci serve una conferma formale che:

- **FEM agisce come data controller**
- Il Team AI agisce come data processor
- Il trattamento è coperto dalla policy GDPR esistente di FEM, oppure occorre redigere un addendum specifico per questo pilot

Possiamo preparare una bozza di addendum se utile.

### 4. Lista degli utenti pilot

Per il lancio ci servono **5-10 domain expert FEM** disponibili a partecipare alla fase iniziale. Per ciascun utente:

- Nome e cognome
- Indirizzo email professionale
- Dominio di competenza (UDL? Neuroscienze? Entrambi?)
- Disponibilità a fornire feedback strutturato

Possiamo identificare il gruppo insieme alla Direzione, oppure ricevere una lista già pronta.

### 5. Canale di comunicazione per il lancio

Come vorrebbe la Direzione FEM comunicare il pilot ai partecipanti?

- Email dalla Direzione
- Slack / Teams / canale interno
- Riunione di kick-off
- Combinazione

Suggeriamo anche un **canale di feedback continuo** (form ClickUp, indirizzo email dedicato, canale Slack ad hoc) per raccogliere segnalazioni durante la durata del pilot.

### 6. Referente FEM per il pilot

Per le decisioni operative durante il pilot (priorità sul feedback, calendario, comunicazione interna) ci serve **un unico referente FEM**. Chi sarà?

---

## Tempistiche stimate

Una volta ricevute le risposte ai punti 1-6, l'implementazione segue il piano dettagliato in `Internal_Production_Deployment_Plan.md`, articolato su 6 onde:

| Settimana | Tema | Note |
|---|---|---|
| **Settimane 1-2** | Infrastruttura, dominio HTTPS, autenticazione utenti | Onde 1-3 del piano |
| **Settimana 3** | Osservabilità, conformità EU AI Act | Onde 4-5; **deadline normativa: 2 agosto 2026 per Articolo 50** |
| **Settimana 4** | Smoke test, onboarding, lancio interno | Onda 6 |

**Stima totale:** ~9-11 giorni-uomo (Team AI), ~1 giorno (Team DEV / IT FEM).
**Lancio realistico:** fine settimana 3 / inizio settimana 4 dal kickoff.

---

## Riepilogo richieste con priorità

| Priorità | Punto | Cosa serve da FEM |
|---|---|---|
| 🔴 **Urgente** | 1. VM | Autorizzazione spesa Hetzner (~€5/mese) **o** assegnazione VM interna |
| 🔴 **Urgente** | 2a. Proprietà dominio | Conferma se `aiforlearning.digital` è FEM o personale di Angelo |
| 🔴 **Urgente** | 2b. DNS | Accesso al pannello DNS per aggiungere il record A `agente → <IP server>` |
| 🟠 **Alta** | 3. GDPR data controller | Conferma formale della Direzione |
| 🟠 **Alta** | 4. Utenti pilot | Lista di 5-10 domain expert |
| 🟡 **Media** | 5. Canale comunicazione | Decisione su come annunciare il pilot |
| 🟡 **Media** | 6. Referente FEM | Nomina del referente unico |

**I punti #1 e #2 sono bloccanti** — tutta l'infrastruttura tecnica dipende da queste due decisioni. I punti #3-6 possono essere risolti in parallelo durante le prime settimane di sviluppo.

---

## Tempistica di risposta richiesta

Per non bloccare l'avvio del pilot, ci servirebbero risposte sui **punti #1 e #2 entro fine settimana prossima (22 maggio 2026)**. I punti #3-6 possono seguire entro la settimana successiva.

Rimaniamo a disposizione per chiarimenti o per una breve riunione (~30 min) se preferite discutere a voce.

---

## Documenti di riferimento

- **Piano dettagliato del pilot interno:** `docs/product/Internal_Production_Deployment_Plan.md`
- **Conformità EU AI Act + UNI 11621-8:** `docs/product/Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md`
- **Integrazione AixLearning (track parallelo):** `docs/product/Dev_Handoff_AgenticGraphRAG_Integration.md`
- **Stato avanzamento progetto:** `docs/product/ClickUp_Agentic_GraphRAG_Update.md`

---

Cordiali saluti,
**Team AI — LM**
