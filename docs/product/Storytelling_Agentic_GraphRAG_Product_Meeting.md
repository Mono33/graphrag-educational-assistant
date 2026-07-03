# Storytelling — Presentare Agentic GraphRAG al Product Team

**Contesto:** meeting Product Team. Louis racconta; **Angelo condivide il video** della WebUI.
**Durata target:** 5–7 minuti (c'è anche una versione da 60 secondi in fondo).
**Obiettivo:** far capire *perché* l'agenticità conta e *come* si inserisce in un prodotto AI esistente — senza tecnicismi — e aprire la vera discussione di prodotto: **quanto del processo mostrare**.
**Companion:** il documento `Risposte_Domande_Product_Team_Agentic_GraphRAG.md` contiene le risposte punto per punto e le decisioni aperte. Questo speech serve a *introdurle*, non a sostituirle.

---

## Messaggi chiave (cosa devono ricordarsi a fine intervento)

1. Il protagonista non è la tecnologia: è il **docente** che deve preparare una lezione affidabile per una classe reale, con tempi stretti e bisogni diversi.
2. Dietro la "modalità agente" delle AI che già usano c'è un'**architettura complessa**: i generalisti la nascondono, noi possiamo **scegliere quanto mostrarla**.
3. Il nostro **non è un chatbot generalista**: è un'**AI specializzata per la didattica**, ancorata a conoscenza verificata e al profilo della classe.
4. **L'architettura AI guida le scelte di prodotto/frontend**: la nuova domanda di design non è "quale schermata", ma *"quanta parte del ragionamento rendiamo visibile, a chi e con quale linguaggio?"*.

---

## Lo storytelling (parlato)

### 0. Premessa — partire dal bisogno reale
> "Immaginiamo un docente che deve preparare una lezione per una classe reale: tempi stretti, livelli diversi, BES/DSA, bisogno di fonti affidabili.
>
> Il punto non è avere una risposta qualsiasi dall'AI. Il punto è avere un sistema che costruisce una proposta didattica, la ancora a conoscenza verificata e gli permette di capire **perché** quella proposta ha senso."

*(Obiettivo: far capire subito che il valore non è "un altro chatbot", ma fiducia, controllo e qualità didattica.)*

### 1. Hook — partire da ciò che conoscono già
> [Domanda aperta alla sala, poi 5 secondi di pausa]
>
> "Prima di mostrarvi cosa abbiamo costruito, una domanda semplice: **secondo voi a cosa serve davvero la *modalità agente* di ChatGPT?** O il *deep research*? O la *modalità canvas / disegno*?"

*(Obiettivo: partire da un'esperienza che hanno già fatto tutti. Raccogli una o due risposte, non correggerle.)*

### 2. La rivelazione — dietro al bottone c'è un'architettura
> "Quei bottoni non sono un effetto grafico. Quando attivate 'agent mode' o 'deep research', l'AI **smette di fare un singolo prompt → risposta**. Inizia a **pianificare**, **cercare**, **scrivere** e **verificare**: è una piccola squadra di agenti che lavora in sequenza.
>
> Le AI generaliste hanno fatto una scelta precisa: **nascondere tutta questa complessità dietro un'unica etichetta semplice**."

### 3. Il ponte al video — "quello che c'è dietro"
> [Angelo condivide / scorre il video della WebUI]
>
> "Quello che vi sta mostrando Angelo è esattamente **quello che c'è dietro**. Se ChatGPT non avesse incapsulato tutto in un bottone 'agent', somiglierebbe molto a questo: **Planner → Retriever → Writer → Critic**.
>
> Noi non l'abbiamo nascosto: l'abbiamo **reso visibile di proposito**. Questa WebUI è una *reference implementation* — ci serve per decidere **insieme** cosa tenere visibile e cosa semplificare nel prodotto finale."

### 4. La differenza — non generalista, ma specializzata
> "C'è però una differenza enorme. ChatGPT e Claude sono **generalisti**: sanno un po' di tutto e a volte inventano.
>
> Il nostro è un'**AI specializzata per la didattica**: ogni lezione è ancorata a un **Knowledge Graph curato da esperti** e adattata al **profilo della classe**. Non improvvisa: recupera, ragiona, verifica.
>
> Ed essendo specializzata, abbiamo un lusso che i generalisti non hanno: possiamo **scegliere cosa mostrare e cosa no**, in base a ciò che serve davvero al docente."

### 5. La tesi — l'architettura guida il prodotto
> "Ed è qui il punto per voi, come Product. Con l'agenticità **cambia la domanda di design**.
>
> Prima era: *'quale bottone, quale schermata'*. Adesso è: ***'quanta parte del ragionamento dell'AI rendiamo visibile, a chi, e con quale linguaggio?'***
>
> Oggi **sono le architetture AI a guidare le scelte di prodotto e di frontend**, non il contrario. Chi disegna il prodotto deve capire l'architettura, perché è l'architettura che apre — o chiude — le possibilità.
>
> La vera opportunità di prodotto non è solo generare contenuti: è trasformare il ragionamento dell'AI in **fiducia, controllo e trasparenza** per il docente."

### 6. Come si integra in un prodotto esistente — *progressive disclosure*
> "La buona notizia è che non dobbiamo scegliere fra 'tutto visibile' e 'tutto nascosto'. Si fa **progressive disclosure**, a livelli:
>
> - il **docente** vede passaggi chiari e non tecnici — *Progettazione, Ricerca, Scrittura, Revisione*;
> - chi vuole **approfondire** apre i dettagli — fonti, concetti chiave, copertura del Knowledge Graph, verdetto del Critic;
> - il **debug tecnico** resta interno.
>
> È lo stesso principio con cui ChatGPT vi scrive 'sto cercando…' senza mostrarvi il codice. E nel nostro caso non è un rifacimento: nell'app nativa AixLearning si aggiunge **in modo incrementale**, perché le tecnologie di base (htmx, WebAwesome, Mercure/SSE) ci sono già."

### 7. Chiusura — il ponte alle decisioni
> "Quindi oggi **non vi chiediamo di decidere la grafica**. Vi chiediamo di decidere **quanto del processo mostrare, per chi e con quale linguaggio**.
>
> È esattamente da lì che parte il documento che vi abbiamo preparato: **risposte punto per punto** ai vostri dubbi, e **le domande di prodotto** su cui ci serve la vostra decisione."

---

## Versione 60 secondi (se manca tempo)

> "Un docente non ha bisogno di una risposta qualsiasi: deve preparare una lezione affidabile per una classe reale, con livelli diversi e fonti da poter spiegare.
>
> Avete presente la *modalità agente* o il *deep research* di ChatGPT? Dietro quel bottone c'è un'architettura che **pianifica, cerca, scrive e verifica** — loro la nascondono. Quello che vi mostra Angelo è la stessa cosa, ma **resa visibile**.
>
> La differenza è che il nostro **non è un generalista**: è un'**AI specializzata per la didattica**, ancorata a conoscenza verificata e al profilo della classe — quindi riduce il rischio di risposte inventate e aumenta la tracciabilità.
>
> Il punto per voi è uno solo: **quanta parte di questo processo vogliamo mostrare al docente?** Da quella scelta dipende il prodotto."

---

## Filo conduttore (in una riga)

**Dal bisogno reale del docente → al bottone "agent" che tutti conoscono → a "ecco cosa c'è dietro" (il video) → "il nostro è specializzato, non generalista" → "l'architettura guida il prodotto" → "quanto mostriamo? decidiamolo insieme" → le risposte e le domande del documento.**

---

## Possibili domande dalla sala (risposte-lampo)

- **"Ma allora è come ChatGPT?"** → No: specializzato sulla didattica, **ancorato al Knowledge Graph** e al **profilo della classe** → riduce il rischio di risposte inventate e aumenta la tracciabilità delle fonti.
- **"Perché mostrare gli agenti? Non confonde il docente?"** → Si mostra **solo l'essenziale** (4 passaggi leggibili); il resto è **a richiesta**. È una scelta di prodotto, non un obbligo tecnico.
- **"Quanto costa e quanto è lento?"** → L'agenticità aggiunge passaggi, ma alza qualità e grounding; leve e numeri nel documento (sezioni *Costi* e *Concorrenza*).
- **"Si può mettere nell'app attuale?"** → Sì, in modo **incrementale**: stesse tecnologie di frontend già in uso, niente rewrite.
