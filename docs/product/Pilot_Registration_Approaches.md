# Opzioni di registrazione per il pilot interno

**Data:** 23 luglio 2026  
**Ultimo aggiornamento:** 1 agosto 2026  
**Stato:** documento decisionale; nessuna delle due opzioni è ancora implementata  
**Obiettivo:** limitare la creazione di account prima dell'onboarding degli
utenti pilot.

## 1. Stato attuale

La pagina `Registrati` raccoglie email, nome visualizzato e password. Dopo la
creazione dell'utente, il backend genera immediatamente il cookie di
autenticazione e reindirizza alla WebUI.

Al momento:

- la registrazione non richiede un invito;
- non viene verificato il possesso dell'indirizzo email;
- `User.is_verified` esiste, ma non limita login o accesso alla WebUI;
- non è configurato un servizio SMTP per l'invio di email.

Il problema da risolvere non è il login automatico in sé, ma la possibilità per
chiunque raggiunga la pagina di creare un account.

## 2. Approccio A — Simple invitation-code flow

### Esperienza utente

1. FEM genera un codice pilot robusto.
2. FEM invia manualmente il codice agli utenti selezionati, insieme alle
   istruzioni di onboarding.
3. L'utente apre la pagina `Registrati`.
4. Nel modulo esistente inserisce email, nome, password e codice di invito.
5. Il backend verifica il codice prima di creare l'account.
6. Se il codice è corretto, l'account viene creato, l'utente viene autenticato
   automaticamente e reindirizzato alla WebUI.
7. Se il codice è errato o assente, la registrazione viene rifiutata senza
   creare alcun record.

Non viene aperta una seconda pagina e l'applicazione non invia email. L'unica
modifica visibile è il campo `Codice di invito` nel modulo esistente.

### Implementazione

- Aggiungere `WEBUI_REGISTRATION_MODE=open|invite`.
- Aggiungere `WEBUI_INVITE_CODES` come secret server-side, con uno o più codici
  separati da virgola.
- Mantenere `open` come default locale per compatibilità; configurare `invite`
  in staging e produzione.
- In modalità `invite`, rifiutare tutte le registrazioni se nessun codice è
  configurato.
- Confrontare i codici in tempo costante.
- Non registrare nei log, persistere o ripresentare nel form il codice inviato.
- Conservare il login automatico soltanto dopo la validazione dell'invito.

### Impatto previsto

- Modifica minima del frontend: un campo aggiuntivo, nessuna nuova pagina.
- Nessuna migrazione del database.
- Nessun servizio email.
- Nessun cambiamento per utenti già registrati o per il normale login.

### Limiti

- Un codice condiviso può essere inoltrato ad altre persone.
- Il codice autorizza la registrazione, ma non dimostra che l'utente possieda
  l'indirizzo email dichiarato.
- Non offre un audit individuale degli inviti, a meno di introdurre in futuro
  codici monouso su database.

### Controlli operativi

- Generare un codice ad alta entropia:

  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(24))"
  ```

- Memorizzarlo solo nel secret store o nel file `.env` del server.
- Condividerlo tramite un canale privato.
- Ruotarlo immediatamente se esposto.
- Chiudere o ruotare l'invito al termine della finestra di onboarding.

### Criteri di accettazione

- Codice assente o errato: nessun utente creato.
- Codice corretto: account creato, cookie impostato, redirect alla WebUI.
- Il codice non compare in log, risposta HTML o database.
- Il login degli utenti esistenti resta invariato.
- La modalità locale `open` continua a funzionare.

## 3. Approccio B — Codice inviato dopo l'inserimento dell'email

### Chiarimento

Il codice inviato automaticamente a un indirizzo è un **codice di verifica
email**, non un invito. Se viene inviato a qualsiasi indirizzo inserito, la
registrazione rimane pubblica: il sistema verifica soltanto che l'utente
controlli quella casella.

Per limitare davvero il pilot, questo approccio deve combinare:

1. un dominio email FEM autorizzato (`fem.digital`) oppure, se in futuro sarà
   necessario limitare il pilot a un sottoinsieme, una lista di indirizzi
   pre-approvati;
2. un codice monouso inviato soltanto a un indirizzo autorizzato.

### Esperienza utente raccomandata

1. FEM configura il dominio autorizzato `fem.digital`.
2. L'utente compila il modulo `Registrati` con email, nome e password.
3. Il backend normalizza l'email e verifica che il dominio sia esattamente
   `fem.digital`.
4. Se autorizzata, crea l'account come `is_verified=False`, genera un codice
   monouso e lo invia via email.
5. L'utente viene reindirizzato a una nuova pagina `Verifica email`.
6. L'utente inserisce il codice ricevuto.
7. Se valido, il backend imposta `is_verified=True`, invalida il codice, crea
   il cookie di autenticazione e reindirizza alla WebUI.
8. Finché la verifica non è completata, login e accesso restano bloccati.

Le risposte verso indirizzi non autorizzati devono essere generiche, per non
rivelare quali email appartengono al pilot.

### Infrastruttura necessaria

- SMTP FEM o provider transazionale, ad esempio SES, SendGrid o Mailgun.
- Indirizzo mittente dedicato, ad esempio
  `noreply@aiforlearning.digital`.
- Credenziali conservate come secret cloud.
- Configurazione SPF e DKIM per ridurre il rischio spam.
- Persistenza delle richieste di verifica e del loro stato.
- Template email e monitoraggio degli errori di consegna.

### Controlli di sicurezza

- Codice casuale, monouso e valido per circa 10 minuti.
- Memorizzazione del digest del codice, mai del valore in chiaro.
- Massimo numero di tentativi per verifica.
- Intervallo minimo tra due invii.
- Rate limiting per indirizzo e IP.
- Un nuovo invio invalida il codice precedente.
- Login rifiutato quando `is_verified=False`.
- Pulizia periodica degli account o delle richieste non completate.

### Impatto previsto

- Una nuova pagina di verifica e relativi stati di errore/scadenza.
- Flusso per reinviare il codice.
- Modifiche al login per bloccare utenti non verificati.
- Nuovo modello o tabella per inviti e challenge di verifica.
- Configurazione e gestione operativa del provider email.
- Maggiore copertura di test per scadenza, tentativi, reinvio e concorrenza.

### Criteri di accettazione

- Solo le email appartenenti esattamente al dominio autorizzato ricevono un
  codice.
- Un codice errato, scaduto o già utilizzato viene rifiutato.
- L'account non può accedere prima della verifica.
- Un codice valido verifica una sola volta l'account e consente l'accesso.
- Reinvio, scadenza e limiti di tentativi funzionano senza rivelare la
  presenza dell'email nella lista pilot.
- Nessun codice o segreto SMTP compare nei log.

## 4. Confronto decisionale

### Approccio A

- È adatto a un piccolo gruppo interno e a una scadenza pre-pilot ravvicinata.
- Richiede una modifica minima alla pagina esistente.
- Non dipende da infrastruttura esterna.
- Offre un controllo sufficiente se il codice è robusto, privato e temporaneo.

### Approccio B

- Verifica il possesso dell'email e può autorizzare utenti individualmente.
- Richiede una lista di invitati per evitare che la registrazione resti
  pubblica.
- Introduce un secondo passaggio visibile e dipendenze SMTP.
- È più appropriato per una fase successiva o per un accesso pubblico più
  strutturato.

## 5. Raccomandazione pre-pilot

Per il pilot interno iniziale, adottare l'**Approccio A**:

- codice di invito condiviso e ad alta entropia;
- campo aggiuntivo nel modulo `Registrati`;
- login automatico soltanto dopo validazione;
- rotazione o chiusura della registrazione dopo l'onboarding.

Valutare l'**Approccio B** dopo il pilot, quando FEM avrà definito provider
email, mittente, gestione degli indirizzi autorizzati e responsabilità
operative.

---

## Appendice A — Piano di implementazione dell'Approccio A

### A.1 Decisioni e configurazione

1. Introdurre `WEBUI_REGISTRATION_MODE` con valori:
   - `open`: comportamento corrente, utile in locale;
   - `invite`: registrazione consentita solo con codice valido.
2. Introdurre `WEBUI_INVITE_CODES` come lista di codici separati da virgola.
3. Configurare `invite` in staging e produzione.
4. In modalità `invite`, se non è presente alcun codice valido, bloccare tutte
   le nuove registrazioni invece di tornare automaticamente a `open`.
5. Conservare i codici reali soltanto nel secret store o nel file `.env` del
   server. Nei template versionati usare esclusivamente placeholder.

### A.2 Validazione backend

1. Aggiungere un piccolo servizio/helper nella componente auth che:
   - legge e normalizza i codici configurati;
   - elimina valori vuoti;
   - confronta il valore ricevuto con `secrets.compare_digest`;
   - non include mai il codice nei log.
2. Aggiungere `invite_code` ai parametri form di `register_post()`.
3. Validare il codice prima di costruire o persistere `UserCreate`.
4. In caso di codice assente o errato:
   - restituire lo stesso messaggio generico;
   - non creare l'utente;
   - non ripresentare il codice nel form;
   - conservare soltanto email e nome già inseriti.
5. In caso di codice valido, mantenere il flusso corrente:
   creazione utente, emissione del cookie JWT e redirect alla WebUI.

### A.3 Modifica minima del modulo

1. Aggiungere al modulo `Registrati` un campo password denominato
   `Codice di invito`.
2. Visualizzarlo soltanto quando `WEBUI_REGISTRATION_MODE=invite`.
3. Aggiungere il testo:
   `Registrazione riservata ai partecipanti al pilot FEM`.
4. Non creare nuove pagine e non modificare il normale modulo di login.

### A.4 Configurazione e documentazione

Aggiornare:

- `.env.example` con modalità `open` e un esempio senza segreti;
- `deploy/.env.prod.example` con modalità `invite` e placeholder;
- runbook di deploy con generazione, distribuzione, rotazione e chiusura del
  codice;
- documentazione tecnica del flusso auth.

Esempio di generazione:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

### A.5 Test

1. Modalità locale `open`: registrazione invariata.
2. Modalità `invite` con codice corretto: utente creato, cookie emesso,
   redirect `303`.
3. Codice assente o errato: nessun utente creato.
4. Configurazione `invite` senza codici: comportamento fail-closed.
5. Codice non presente in HTML, log o database.
6. Login e account esistenti non modificati.
7. Due codici attivi durante una rotazione funzionano; il codice rimosso viene
   rifiutato dopo il riavvio.

### A.6 Rollout e rollback

1. Distribuire il codice con modalità ancora `open`.
2. Verificare in staging entrambi i percorsi.
3. Inserire il codice reale nel secret environment del server.
4. Attivare `WEBUI_REGISTRATION_MODE=invite`.
5. Condividere il codice fuori dall'applicazione tramite onboarding privato.
6. Al termine delle registrazioni, ruotare il codice o disabilitare il form.
7. In caso di problemi, riportare temporaneamente la modalità a `open`;
   nessuna migrazione database deve essere annullata.

### A.7 File previsti

- `src/aix/webui/auth/routes.py`;
- eventuale nuovo helper `src/aix/webui/auth/registration.py`;
- `src/aix/webui/templates/pages/auth_register.html`;
- `.env.example`;
- `deploy/.env.prod.example`;
- test unitari e di integrazione auth;
- documentazione di deploy e tecnica.

**Effort indicativo:** 1–2 giornate di sviluppo, includendo test e smoke test
in staging.

---

## Appendice B — Piano di implementazione dell'Approccio B

### B.1 Decisione per il pilot FEM

Il dominio confermato è:

```text
fem.digital
```

Per il pilot sono autorizzati tutti gli indirizzi che appartengono esattamente
a questo dominio, ad esempio `louis.mono@fem.digital`.

Non è quindi necessaria, nella prima versione, una tabella con la lista dei
singoli partecipanti. Il backend deve estrarre il dominio dall'indirizzo
normalizzato e confrontarlo per uguaglianza con `fem.digital`. Non deve usare
un controllo testuale permissivo come `endswith`, che potrebbe accettare domini
non autorizzati.

Se in futuro il pilot dovrà essere limitato a un sottoinsieme di persone, si
potrà aggiungere una tabella `pilot_allowlist` senza modificare il meccanismo
OTP.

### B.2 Prerequisiti email

Il dominio `fem.digital` usa già Google Workspace. L'opzione preferita è quindi
il relay SMTP di Google Workspace, evitando un nuovo server email o un provider
transazionale separato.

Il responsabile Google Workspace deve:

1. autorizzare il relay per l'IP pubblico statico della VM FEM;
2. richiedere TLS;
3. limitare il mittente al dominio FEM;
4. autorizzare o creare `noreply@fem.digital`;
5. verificare SPF, DKIM e DMARC;
6. eseguire un invio di prova da staging.

Configurazione applicativa prevista:

```text
WEBUI_REGISTRATION_MODE=domain_otp
WEBUI_ALLOWED_EMAIL_DOMAINS=fem.digital
WEBUI_EMAIL_PROVIDER=smtp
WEBUI_SMTP_HOST=smtp-relay.gmail.com
WEBUI_SMTP_PORT=587
WEBUI_SMTP_USE_TLS=true
WEBUI_EMAIL_FROM=noreply@fem.digital
WEBUI_EMAIL_FROM_NAME=AixLearning
```

Se il relay è autorizzato tramite IP, username e password SMTP potrebbero non
essere necessari. La configurazione definitiva deve essere confermata
dall'amministratore Workspace.

### B.3 Modello dati per il challenge OTP

Aggiungere una tabella `email_verification_challenge`, compatibile con SQLite
locale e Postgres di produzione, con almeno:

- `id`;
- `user_id`, univoco e collegato all'utente;
- `code_digest`;
- `created_at`;
- `expires_at`;
- `attempt_count`;
- `last_sent_at`;
- `consumed_at`.

Mantenere un solo challenge corrente per utente. Il reinvio aggiorna la stessa
riga con un nuovo digest, azzera i tentativi e rinnova la scadenza. In questo
modo il codice precedente diventa immediatamente invalido.

Il codice deve:

- essere generato con `secrets`;
- contenere sei cifre;
- scadere dopo circa dieci minuti;
- consentire al massimo cinque tentativi;
- essere monouso;
- essere memorizzato come HMAC/digest usando un secret
  `WEBUI_OTP_PEPPER`, mai in chiaro.

Le operazioni di verifica e consumo devono essere atomiche, così due worker non
possono utilizzare con successo lo stesso codice.

### B.4 Servizio email

1. Creare un'astrazione `EmailSender` indipendente dal provider.
2. Implementare il backend SMTP Google Workspace.
3. Prevedere un backend fittizio/in-memory per i test.
4. Inviare email testuali e HTML contenenti:
   - codice OTP;
   - durata di validità;
   - avviso di non condivisione;
   - riferimento al supporto FEM.
5. Applicare un timeout all'invio.
6. Registrare gli errori operativi senza includere OTP, password SMTP o
   contenuto sensibile.

Per il numero ridotto di utenti pilot è sufficiente attendere l'invio SMTP
nella richiesta. Non sono necessari Redis, Celery o una coda dedicata.

### B.5 Nuovo flusso di registrazione

In modalità `domain_otp`, `register_post()` deve:

1. normalizzare l'email;
2. estrarre e verificare il dominio esatto;
3. restituire sempre una risposta generica che non confermi se l'indirizzo è
   autorizzato o già registrato;
4. non creare utenti e non inviare email per domini non autorizzati;
5. per un dominio autorizzato:
   - creare l'utente con `is_verified=False`;
   - creare il challenge OTP nella stessa transazione;
   - eseguire il commit;
   - inviare il codice;
   - non emettere ancora il normale cookie di autenticazione;
6. impostare un cookie HttpOnly temporaneo e firmato che identifichi il flusso
   di verifica, senza inserire email o user ID nell'URL;
7. reindirizzare a `/auth/verify`.

Se l'invio SMTP fallisce dopo la creazione dell'utente, l'errore viene
monitorato e l'utente può usare il reinvio. Il dettaglio tecnico non deve essere
mostrato nel browser.

### B.6 Pagina e route di verifica

Aggiungere:

- `GET /auth/verify`: mostra il form OTP;
- `POST /auth/verify`: controlla e consuma il codice;
- `POST /auth/verify/resend`: genera e invia un nuovo codice rispettando il
  cooldown.

La pagina `Verifica email` deve mostrare:

- indirizzo mascherato, ad esempio `l***@fem.digital`;
- campo codice;
- tempo di validità;
- pulsante di verifica;
- pulsante di reinvio;
- messaggi generici per codice errato, scaduto o già utilizzato.

Quando il codice è valido, un'unica transazione deve:

1. marcare il challenge come consumato;
2. impostare `User.is_verified=True`;
3. eseguire il commit;
4. emettere il normale cookie JWT;
5. cancellare il cookie temporaneo;
6. reindirizzare alla WebUI.

### B.7 Blocco degli utenti non verificati

Creare dipendenze auth basate su:

```python
fastapi_users.current_user(active=True, verified=True)
```

Applicarle a:

- dashboard e route lezioni della WebUI;
- richieste API effettuate con cookie utente;
- richieste API effettuate con Bearer token utente.

Il normale login deve rifiutare o reindirizzare gli utenti con
`is_verified=False`. Le credenziali tecniche/service account devono restare
separate e non devono dipendere dalla verifica email.

Questa verifica è indispensabile: controllare soltanto la pagina WebUI
permetterebbe di aggirare il flusso tramite un'altra route autenticata.

### B.8 Rate limiting e pulizia

Applicare limiti persistenti e compatibili con più worker:

- tentativi massimi sul challenge;
- cooldown tra due invii;
- limite di reinvii per utente;
- limite di richieste di registrazione/verifica per IP, gestito al proxy o con
  stato condiviso su database.

Prevedere una pulizia periodica per challenge scaduti e account non verificati
più vecchi della finestra definita. Per il pilot può essere uno script
operativo schedulato, senza introdurre un nuovo servizio.

### B.9 Compatibilità con gli utenti esistenti

Gli account attuali hanno normalmente `is_verified=False`, anche se oggi
possono accedere.

Prima di attivare `domain_otp`:

1. identificare gli account esistenti legittimi;
2. eseguire un backfill una tantum impostando `is_verified=True`;
3. verificare login WebUI e accesso API;
4. attivare soltanto dopo il nuovo blocco degli utenti non verificati.

Saltare questo passaggio causerebbe il blocco degli utenti già registrati.

### B.10 Test

1. Email `@fem.digital`: challenge creato ed email inviata.
2. Dominio diverso: nessun utente, challenge o email.
3. Risposte browser equivalenti per dominio consentito e non consentito.
4. Utente non verificato: login e accesso WebUI/API bloccati.
5. OTP corretto: verifica, consumo singolo, cookie e redirect.
6. OTP errato, scaduto, già usato o oltre i tentativi: rifiuto.
7. Reinvio: codice precedente invalidato e cooldown rispettato.
8. Due verifiche concorrenti: una sola può riuscire.
9. Errore SMTP: nessuna informazione interna mostrata e reinvio possibile.
10. OTP e credenziali assenti da log e risposte.
11. Account esistenti verificati: nessuna regressione.
12. Modalità locale `open`: comportamento corrente preservato.

### B.11 Rollout

1. Confermare il relay con l'amministratore Google Workspace.
2. Distribuire codice, tabelle e configurazione mantenendo modalità `open`.
3. Eseguire il backfill degli utenti esistenti.
4. Configurare e verificare `noreply@fem.digital`.
5. Testare l'invio reale da staging.
6. Validare il flusso con:
   - un indirizzo FEM autorizzato;
   - un dominio esterno;
   - OTP errato e scaduto;
   - reinvio.
7. Impostare `WEBUI_REGISTRATION_MODE=domain_otp`.
8. Monitorare errori SMTP, tempi di consegna, tentativi e account pendenti.

Rollback:

1. riportare `WEBUI_REGISTRATION_MODE=open`;
2. mantenere le nuove tabelle, che non interferiscono con il flusso corrente;
3. gestire manualmente eventuali utenti rimasti non verificati.

### B.12 File previsti

- `src/aix/webui/auth/routes.py`;
- `src/aix/webui/auth/models.py` o un nuovo modulo auth dedicato;
- `src/aix/webui/auth/dependencies.py`;
- `src/aix/webui/auth/manager.py`;
- nuovo servizio OTP;
- nuovo servizio email;
- `src/aix/webui/templates/pages/auth_verify.html`;
- `src/aix/webui/db.py`;
- `.env.example`;
- `deploy/.env.prod.example`;
- route API che usano autenticazione utente;
- test unitari e di integrazione auth;
- runbook di deploy e documentazione tecnica.

**Effort indicativo:** 6 giornate per la versione minima; 8–10 giornate per
la versione raccomandata con anti-enumerazione, concorrenza, reinvio, test
completi e hardening di staging. La configurazione Workspace/DNS dipende dai
responsabili FEM e non è inclusa nell'effort di sviluppo.

