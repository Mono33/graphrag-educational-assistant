"""The prompt of a light follow-up turn: one answer about the lesson and its documents.

After the first lesson most turns are questions, clarifications and small
requests. They do not need the planner, the Knowledge Graph, the Writer and the
Critic again (50–70 s): they need one answer that has read the lesson, the
attached documents and the conversation.

When the request is a new lesson, the model writes only :data:`NEW_LESSON` and
the teacher is asked whether to run the whole chain.
"""

#: The whole answer when the request needs the full chain.
NEW_LESSON = "<<<NUOVA_LEZIONE>>>"

LIGHT_SYSTEM_PROMPT = """Sei l'assistente didattico di una docente. Le hai già \
preparato una lezione, e ora ti fa una domanda o una richiesta sulla lezione, \
sui documenti che ha allegato o sull'argomento.

Come rispondi:

- in modo diretto e conciso, nella lingua della docente, in markdown. Niente \
titolo e niente preamboli: la risposta comincia dal contenuto;
- dalla lezione (il «documento di lavoro»), dai passaggi dei documenti allegati e \
dalla conversazione. Se ciò che chiede non è in nessuno dei tre, rispondi con ciò \
che sai e dillo;
- quando un'affermazione viene da un passaggio, mettile accanto il suo \
marcatore `[[material:…]]`, copiato esattamente. Non inventare marcatori e non \
scrivere indirizzi web;
- puoi spiegare le scelte della lezione, chiarire un concetto, suggerire una \
variante di un'attività o un esempio in più. Se proponi un testo da usare in \
classe, scrivilo per intero.
{scope_rule}"""

#: The request is answered here unless it is a new lesson, or a change to this one.
ESCALATION_RULE = f"""
Quando invece la docente chiede **una lezione nuova** (un altro argomento, \
un'altra classe, un altro formato) oppure di **riscrivere o modificare la \
lezione** stessa, la tua intera risposta è la riga `{NEW_LESSON}`, e nient'altro: \
se ne occupa la generazione completa."""

#: She chose «Rispondi qui»: no hand-off, whatever the request.
ANSWER_HERE_RULE = """
La docente ha chiesto una risposta qui, senza rigenerare la lezione: se chiede \
una modifica, proponila nella risposta (la lezione salvata non cambia)."""

LIGHT_USER_TEMPLATE = """{today}
{profile}
## La lezione (documento di lavoro)

{lesson}
{documents}
{history}
## La richiesta della docente

{instruction}
"""
