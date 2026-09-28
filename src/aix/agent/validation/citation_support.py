"""Remove a citation whose passage does not say what the sentence claims (§8.6).

The markers themselves are sound: measured on 2026-09-21, the Writer only ever
copies back a marker it was handed — it invents none. What it does do is attach
an authentic marker to a sentence the passage does not support. Two real lessons
cited a page of Common Core standards for claims about NATO and about the
Soviet collapse, and a page of exercise questions for a speech printed
elsewhere. Every one of those was checkable, and every one would have wasted the
teacher's time and spent the trust the citations exist to earn.

**Why the sentence survives and only the marker goes.** The claim is usually
right — it comes from the Knowledge Graph, from the rest of the material, or
from what the model knows. It is the *attribution* that is wrong. Deleting the
sentence would remove content the teacher asked for; deleting the marker leaves
a lesson that says one thing less about where it got something, which is
honest. A citation a teacher cannot verify is worse than no citation.

**Why the model does the judging.** Counting words in common was measured first
and is unusable: it flags an Italian sentence citing an English passage, and
since the lesson is written in Italian whatever the document's language, that is
most citations. On the eight golden cases: words in common 5/8 with three false
alarms, the model 8/8 with none
(``scripts/eval/check_citation_support.py``, ``data/eval/citations/cases.yaml``).

**Fail open, always.** A timeout, a refusal, an unparseable answer, a missing
key — every one of them leaves the draft exactly as it was. A checker that
strips citations because the network hiccuped is worse than no checker: the
teacher cannot tell the difference between "the model chose not to cite" and
"something broke", and neither can we.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: How much of the evidence behind one marker the judge is shown.
#:
#: Generous on purpose. At 700 — chosen to keep the prompt small — a marker
#: covering two chunks of 2 883 characters showed the judge 24% of them, and
#: three sound citations were removed because the sentence that supported them
#: was in the part nobody sent: Bandung, the NATO/Warsaw Pact pair and the
#: Truman Doctrine, all of them on the page they pointed at (2026-09-21).
#:
#: A citation is removed on this evidence, so the evidence has to be the whole
#: of it. 4 000 covers two full chunks — the chunker's ceiling is 450 tokens,
#: about 1 600 characters — and the check is one call for the whole lesson, so
#: eight markers at this size is a few thousand tokens, once.
_PASSAGE_CHARS = 4000

#: How much of the claim the judge is shown, when a line runs long.
_CLAIM_CHARS = 600

#: Our own marker syntax. Used only to take markers *out* of a claim — the
#: authorization of a citation is decided elsewhere, on the version id.
_MARKER = "[[material:"
_ANY_MARKER = re.compile(r"\[\[material:[^\]]*\]\]")


def drop_unknown_markers(draft: str, items: Optional[list[dict[str, Any]]]) -> str:
    """Only the markers the Writer was handed survive; any other is removed.

    Measured on 2026-09-21: the Writer only ever copies back markers it was
    given. But a marker is plain text, and a teacher's document is plain text
    too: a file carrying ``[[material:<someone else's id>#page=1]]`` in its body
    hands the Writer a marker nobody retrieved. In the WebUI the renderer drops
    it, because it knows which documents the lesson is bound to. The public API
    renders without that list, so the marker became "(materiale allegato,
    p. 1)" -- a citation to nothing -- and the support check, which judges only
    the markers it knows, let it through (2026-09-25).

    Not a judgement and not a model call, so it runs whether or not the
    citation check is switched on.
    """
    if not draft or _MARKER not in draft:
        return draft

    known = {str(item.get("citation") or "") for item in items or []}
    removed: list[str] = []

    def _keep(match: re.Match) -> str:
        if match.group(0) in known:
            return match.group(0)
        removed.append(match.group(0))
        return ""

    cleaned = _ANY_MARKER.sub(_keep, draft)
    if not removed:
        return draft
    logger.warning(
        "[citation-check] %d marker(s) the Writer was not given were removed: %s",
        len(removed),
        ", ".join(sorted(set(removed)))[:300],
    )
    return re.sub(r" +([.,;:])", r"\1", cleaned)


def enabled() -> bool:
    """On by default — read the literal, not the shape of the expression.

    It only ever costs anything when a draft actually carries private citations:
    no markers, no call.
    """
    return (os.getenv("AIX_CITATION_CHECK_ENABLED") or "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _claim_around(text: str, start: int, end: int) -> str:
    """The claim the marker is attached to: its line.

    **Not its sentence.** The first version cut at the nearest ``.?!;:`` and
    handed the judge fragments — a real lesson produced «paura di nuove
    invasioni) .», because the sentence contained «vs.» and the cut landed
    inside it. Asked to rule on half a claim the judge says «unsupported», and
    on the first real draft it removed two thirds of the citations, most of them
    sound (2026-09-21).

    **Nor its markdown block.** A bulleted list has no blank lines between its
    items, so "the block between two blank lines" swallowed the whole list and
    handed the judge things like ``page=3&block=e8d9…]] - Guerra del Vietnam
    [[material:5c07…`` — several claims and their neighbours' markers, run
    together. Four citations were removed on that (2026-09-21).

    So: the marker's own line, plus the line above it when that line is a title
    of its own — non-empty and carrying no marker — because a numbered item
    routinely splits in two::

        9. **Il ruolo della NATO nel contenimento**
           Approfondimento sul ruolo delle alleanze militari… [[material:…]].

    Every marker is then stripped from the result: the judge is reading a claim,
    and an address is noise inside one.
    """
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    claim = text[line_start:line_end]

    # The title line, but only when this line is indented under it. Markdown
    # says a continuation is indented, and that is the difference between the
    # two-line numbered item above and two bullets in a row — «- NATO e Patto di
    # Varsavia» followed by «- Guerra di Corea» are two claims, not one.
    if line_start > 1 and claim[:1].isspace():
        previous_end = line_start - 1
        previous = text[text.rfind("\n", 0, previous_end) + 1 : previous_end]
        if previous.strip() and _MARKER not in previous:
            claim = f"{previous} {claim}"

    claim = _ANY_MARKER.sub("", claim)
    return " ".join(claim.split())[:_CLAIM_CHARS]


def _occurrences(draft: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every place a known marker appears, with the claim around it.

    Driven by the markers we handed the Writer rather than by a pattern match on
    the draft: those are strings this system produced, so there is nothing to
    parse and nothing to get wrong.

    **Several passages can share one marker**, because a citation anchors on the
    page and the first block and a PDF page is usually one block. So the marker
    is what is grouped on, and the judge sees every passage behind it: the claim
    holds if any of them carries it. Grouping also keeps one position from being
    cut twice — which, when a page had four chunks, deleted the text after the
    marker along with it.
    """
    passages: dict[str, list[str]] = {}
    for item in items or []:
        marker = str(item.get("citation") or "")
        text = str(item.get("text") or "").strip()
        if marker and text:
            passages.setdefault(marker, []).append(text)

    found: list[dict[str, Any]] = []
    for marker, texts in passages.items():
        joined = "\n\n".join(texts)[:_PASSAGE_CHARS]
        start = draft.find(marker)
        while start != -1:
            end = start + len(marker)
            found.append(
                {
                    "marker": marker,
                    "start": start,
                    "end": end,
                    "claim": _claim_around(draft, start, end),
                    "passage": joined,
                }
            )
            start = draft.find(marker, end)
    return found


#: **Il prompt non nomina nessun documento, e non deve.**
#:
#: La prima versione che passava tutti i casi d'oro elencava tre generi — gli
#: standard curricolari, le consegne di un esercizio, l'attività inventata — e
#: per spiegarli citava le Indicazioni nazionali, i Common Core, «Reagan»,
#: «la LIM». Era una tassonomia ricavata dai tre PDF di prova: un documento di
#: un genere non previsto — un manuale, un articolo, un'antologia, un testo di
#: legge — non ricadeva in nessuna voce, e lo strumento deve reggere documenti
#: che non abbiamo mai visto (Angelo, 2026-09-22).
#:
#: Le tre voci dicevano però tutte la stessa cosa, scoperta un caso alla volta
#: invece che enunciata una volta sola: **il genere del passaggio non decide
#: niente, decide che cosa fa la frase.** Il prompt ora dice quello, e mostra la
#: forma del giudizio con esempi **inventati e senza materia** — due coppie in
#: cui lo stesso passaggio sostiene una frase e non ne sostiene un'altra.
#:
#: Misurato, riscrivendolo tre volte:
#:
#: - solo il principio, in prosa: **9/11**, due falsi allarmi;
#: - principio + esempi astratti: **10/11**, 5 su 5 di quelle da togliere, un
#:   falso allarme — tenuta, questa;
#: - in più «nel dubbio tieni la citazione»: **9/11**. L'asimmetria dei costi è
#:   vera come decisione di prodotto e come istruzione al giudice non funziona:
#:   non ha recuperato il caso al limite e ha fatto passare uno standard che
#:   nomina «Reagan». Revocata sulla misura.
#:
#: La versione che nominava i documenti faceva 11/11. Il caso in più è una
#: riformulazione al limite, e quel punto non vale un prompt che va aggiornato
#: a ogni genere nuovo: si chiude allargando i casi d'oro ad altri tipi di
#: documento, non aggiungendo una riga per quello che abbiamo sottomano.
_PROMPT = """Ti do una frase di una lezione e il passaggio del documento che la
frase cita. Dimmi se il passaggio sostiene la frase.

Rispondi con una sola parola, "supported" oppure "unsupported".

Giudica una cosa sola: il passaggio **afferma** quello che la frase afferma?
Vale anche se lo afferma in parte, con parole diverse, o in un'altra lingua.

Due criteri, in quest'ordine:

1. **Che cosa fa la frase, e il passaggio fa la stessa cosa?** Un testo può
   raccontare qualcosa, oppure prescrivere o proporre qualcosa: un obiettivo,
   una consegna, un'indicazione di lavoro. Un passaggio che racconta sostiene
   una frase che racconta lo stesso fatto. Un passaggio che prescrive o propone
   sostiene una frase che prescrive o propone la stessa cosa. Sono due modi
   diversi di essere la fonte giusta, e valgono allo stesso modo. Se invece i
   due parlano dello stesso argomento facendo cose diverse — la frase racconta
   un fatto e il passaggio si limita a chiedere di studiarlo, o viceversa —
   allora è "unsupported".

2. **Nominare un argomento non è affermarne qualcosa.** Un passaggio che cita un
   titolo, rimanda altrove, elenca voci o nomina un tema di sfuggita non
   sostiene una frase che di quel tema racconta il contenuto.

La frase è scritta con le parole di chi prepara la lezione, non con quelle del
documento: una riformulazione conta quanto una citazione. Chiedi la stessa
sostanza, non le stesse parole.

Il genere del passaggio non decide da solo: lo stesso testo può essere la fonte
giusta per una frase e quella sbagliata per un'altra. Non chiederti che tipo di
pagina è: chiediti se ci trovi quello che la frase dice.

Due esempi inventati, per la forma del giudizio e non per il contenuto.

Passaggio: «Obiettivo: lo studente sa confrontare due fonti che raccontano lo
stesso avvenimento.»
→ Frase: «Confrontare due racconti diversi di uno stesso avvenimento.»
  "supported": la frase enuncia quell'obiettivo, con parole sue.
→ Frase: «L'insegnante proietta le due fonti e le legge ad alta voce alla
  classe.»
  "unsupported": è una scelta di chi prepara la lezione; il passaggio dice che
  cosa lo studente deve saper fare, non come si conduce l'ora.

Passaggio: «Rifletti: quale dei due fattori pesò di più? Motiva la risposta.»
→ Frase: «Discussione a piccoli gruppi su quale dei due fattori pesò di più.»
  "supported": la frase propone quell'attività, e il passaggio è quell'attività.
→ Frase: «Il primo fattore pesò più del secondo.»
  "unsupported": il passaggio pone la domanda, non dà la risposta.

Una docente aprirà quella pagina aspettandosi di trovarci quello che la frase
dice. Giudica così.

FRASE: {claim}

PASSAGGIO: {passage}"""


async def judge(pairs: list[dict[str, Any]], client: Any) -> list[str]:
    """One verdict per (claim, passage), in order. One call each, all at once.

    **Not one call for the whole lesson.** That was the first design, and it
    was cheaper and wrong: asked to rule on six citations of the same document
    at once — four of them activities the model had invented — the judge marked
    the two sound ones unsupported along with them. Measured on the golden
    cases: 11/11 one at a time, 10/11 in a single batch, and the one it lost
    was the borderline case that matters
    (``scripts/eval/check_citation_support.py``, 2026-09-22).

    They run concurrently, so a lesson with ten citations costs one round trip,
    not ten. And a pair that fails is kept on its own — the others are still
    judged, which the batch could not do.

    Public because the golden cases call *this*. A measurement of something you
    do not ship is worth nothing, and this file has already proved it twice.
    """
    if not pairs:
        return []
    return list(await asyncio.gather(*(_judge_one(pair, client) for pair in pairs)))


async def _judge_one(pair: dict[str, Any], client: Any) -> str:
    """One citation. Anything that goes wrong means «keep it» — see the module docstring."""
    from aix.core.config import config as app_config

    try:
        response = await client.chat.completions.create(
            model=os.getenv("AIX_CITATION_CHECK_MODEL") or app_config.openai.model,
            messages=[
                {
                    "role": "user",
                    "content": _PROMPT.format(claim=pair["claim"], passage=pair["passage"]),
                }
            ],
            max_tokens=2000,
            temperature=0,
        )
        answer = (response.choices[0].message.content or "").strip().lower()
    except Exception as exc:  # noqa: BLE001 — fail open, per citation
        logger.warning("[citation-check] one citation could not be judged (%s) — keeping it", exc)
        return "supported"

    if "unsupported" in answer:
        return "unsupported"
    if "supported" in answer:
        return "supported"
    logger.warning("[citation-check] unreadable verdict %r — keeping the citation", answer[:60])
    return "supported"


async def drop_unsupported_citations(
    draft: str, items: Optional[list[dict[str, Any]]], *, client: Any = None
) -> str:
    """The draft with unverifiable citations removed, and nothing else changed.

    Returns ``draft`` untouched when the check is off, when there is nothing to
    check, or when anything goes wrong.
    """
    if not enabled() or not draft or not items:
        return draft

    occurrences = _occurrences(draft, items)
    if not occurrences:
        return draft

    if client is None:
        from aix.core.config import config as app_config

        client = app_config.openai.get_async_client()

    verdicts = await judge(occurrences, client)

    # Right to left, so the offsets of the ones not yet removed stay valid.
    removed = 0
    for occurrence, verdict in sorted(
        zip(occurrences, verdicts, strict=True), key=lambda pair: pair[0]["start"], reverse=True
    ):
        if verdict == "unsupported":
            draft = draft[: occurrence["start"]] + draft[occurrence["end"] :]
            removed += 1
            logger.info(
                "[citation-check] removed %s — the passage does not support: %.80s",
                occurrence["marker"],
                occurrence["claim"],
            )

    if removed:
        logger.info(
            "[citation-check] %d of %d citations removed as unsupported",
            removed,
            len(occurrences),
        )
    # A citation removed mid-sentence leaves the space before the full stop.
    return re.sub(r" +([.,;:])", r"\1", draft)
