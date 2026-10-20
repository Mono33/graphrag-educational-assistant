"""The judge: a checklist of what a teacher would have to correct, each with its quote.

Why this exists
---------------
The Critic scores five broad criteria from 1 to 5 on the first 3,000 characters
of a lesson (``AIX_CRITIC_LESSON_MAX_CHARS``) and approves at an average of 3.5.
Lessons are about 7,000 characters long: it never reads the later activities,
the assessment or the differentiation. On the evaluation of 2026-10-19 it
approved 8 lessons out of 8 -- among them a fractions quiz whose answer was
wrong, «gatto» split as «ga-to», a «belly» rule that does not tell p from b
(``docs/product/Specifica_Risposta_Mediana.md``, second blind read).

The judge is the other kind of reviewer. It reads the whole lesson and answers
a fixed list of yes/no questions (:data:`CHECKS`); every problem it reports
carries the exact quote from the lesson, why it is wrong and how to fix it. A
problem whose quote is not in the lesson is dropped: a reviewer that cannot
point at the sentence has not found it.

What the code can count -- minutes, groups, the graph's nodes, what the graph
advises against -- is checked in code (``validation.rules``, ``kg_evidence``).
The judge is only for what needs knowledge: the subject, the pupils' age, the
pedagogy.

Not wired into the chain yet: ``scripts/eval/check_judge.py`` measures it first,
on lessons with errors of known kinds put in on purpose.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
_DEFAULT_PROVIDER = "google-vertex/global"

#: The questions, by id. The ids are what the judge reports and what the
#: evaluation scores; the text is what the model reads.
CHECKS: dict[str, str] = {
    "esempi_lingua": (
        "Ogni esempio di lingua è corretto: divisione in sillabe, ortografia, "
        "grammatica, distinzione tra lettere o suoni, e i criteri dati agli alunni per "
        "riconoscerli?"
    ),
    "fatti": (
        "Ogni affermazione disciplinare (scienze, storia, geografia, matematica, "
        "neuroscienze) è corretta, senza miti presentati come fatti?"
    ),
    "esercizi": (
        "Ogni esercizio, quiz o domanda chiusa ha una e una sola risposta corretta, e "
        "la soluzione indicata (se c'è) è davvero quella giusta?"
    ),
    "esempi_chiari": (
        "Gli esempi concreti e numerici sono chiari e non ambigui per alunni di "
        "questa età (lo stesso intero, le stesse unità, nessun confronto fuorviante)?"
    ),
    "perimetro": (
        "Obiettivi e contenuti restano nel perimetro della richiesta e sono adatti "
        "al livello, senza concetti molto oltre l'età?"
    ),
    "sicurezza": (
        "Materiali e attività sono sicuri per l'età: niente acqua molto calda, "
        "fiamme, fonti di calore, oggetti taglienti o sostanze in mano agli alunni "
        "piccoli?"
    ),
    "risorse": (
        "Ogni risorsa (video, testo, sito) è adatta a chi la usa: quelle sul metodo "
        "didattico sono per la docente e non vengono mostrate agli alunni?"
    ),
    "classe": (
        "Le attività sono sostenibili per questa classe: durata del lavoro "
        "individuale o scritto, tempi di attenzione, bisogni dichiarati?"
    ),
}

#: A problem the teacher must fix before using the lesson, or one worth fixing.
SEVERITIES = ("bloccante", "minore")
#: Questions whose problems are errors of content or of safety: always blocking.
ALWAYS_BLOCKING = frozenset({"esempi_lingua", "fatti", "esercizi", "sicurezza"})
#: How it is fixed: the quote replaced by the fix, or the section around it revised.
KINDS = ("sostituzione", "revisione")

_SYSTEM = """Sei un revisore esperto di didattica e delle discipline scolastiche. \
Ricevi una lezione preparata per una docente, con la sua richiesta e il profilo \
della classe. Il tuo compito è trovare ciò che la docente dovrebbe correggere \
prima di usarla in classe: errori, non preferenze di stile.

Rispondi a ciascuna domanda della lista. Per ogni problema che trovi:
- «check»: l'id della domanda;
- «severity»: «bloccante» se è un errore di contenuto, una soluzione sbagliata, \
un rischio per gli alunni o una risorsa per la docente data agli alunni; \
«minore» se è un esempio poco chiaro o un'attività poco adatta alla classe;
- «quote»: la frase della lezione copiata ESATTAMENTE, carattere per carattere \
(al massimo 200 caratteri). Senza citazione esatta il problema non viene \
considerato;
- «why»: perché è sbagliata, in una frase;
- «fix»: il testo corretto da mettere al posto della citazione, pronto da usare: \
copre la stessa parte della citazione (se citi mezza frase, riscrivi quella mezza \
frase) e non è mai un'istruzione per chi corregge («Eliminare…», «Sostituire…», \
due alternative con «oppure»): se la correzione non si scrive come testo, è una \
«revisione»;
- «kind»: «sostituzione» se basta mettere «fix» al posto della citazione (un \
esempio, una data, una soluzione, una frase); «revisione» se va cambiata la \
sezione intorno (gruppi e ruoli che non tornano, un'attività da ripensare): in \
questo caso «fix» descrive in breve la correzione.

Le «Verifiche fatte dal sistema», quando ci sono, sono calcoli esatti (sillabe, \
frazioni, operazioni, minuti, alunni nei gruppi): non rifarli e non contraddirli. \
Ogni verifica [ERRATO] è un problema da riportare, con la sua citazione e, se c'è, \
la sua correzione; ciò che una verifica dà [corretto] non è un errore.

Non segnalare ciò che è corretto, né scelte didattiche diverse dalle tue ma \
legittime. Se non trovi problemi, la lista è vuota. Rispondi solo con JSON:
{"issues": [{"check": "...", "severity": "...", "kind": "...", "quote": "...", \
"why": "...", "fix": "..."}]}"""

_USER = """## Richiesta della docente
{request}

## Profilo della classe
{profile}

## Domande
{checks}

{facts}## Lezione
{lesson}"""


@dataclass
class Issue:
    check: str
    severity: str
    quote: str
    why: str
    fix: str
    kind: str = "sostituzione"
    #: Found by the code (``validation.facts``), not by the model.
    computed: bool = False


@dataclass
class JudgeReport:
    """What the judge found. ``issues`` are the ones whose quote is in the lesson."""

    issues: list[Issue] = field(default_factory=list)
    #: Reported without a quote that is in the lesson: not counted.
    unquoted: list[Issue] = field(default_factory=list)
    model: str = ""
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Optional[float] = None
    error: str = ""

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "bloccante"]


def judge_enabled() -> bool:
    """``AIX_JUDGE_ENABLED``: the judge reviews and repairs every lesson. Off by default."""
    return (os.getenv("AIX_JUDGE_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")


def facts_enabled() -> bool:
    """``AIX_JUDGE_FACTS``: computed facts given to the judge. On by default."""
    return (os.getenv("AIX_JUDGE_FACTS") or "true").strip().lower() in ("1", "true", "yes", "on")


def judge_model() -> str:
    return os.getenv("AIX_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def _provider_routing(model: str) -> Optional[dict]:
    """Pinned to the provider the citation judge uses, when on OpenRouter (see there)."""
    raw = os.getenv("AIX_JUDGE_PROVIDER", os.getenv("AIX_CITATION_CHECK_PROVIDER", ""))
    raw = (raw or _DEFAULT_PROVIDER).strip()
    if raw.lower() in ("none", "") or not model.startswith("anthropic/"):
        return None
    return {"order": [p.strip() for p in raw.split(",") if p.strip()], "allow_fallbacks": True}


def _flat(text: str) -> str:
    """For finding a quote: case, spacing and Markdown emphasis do not count."""
    text = re.sub(r"[*_`]", "", text or "")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.lower().split())


def quote_in(quote: str, lesson: str) -> bool:
    """Whether ``quote`` is in ``lesson``, emphasis and spacing aside; an ellipsis
    the model put in for a cut is allowed if each piece is there."""
    flat_quote, flat = _flat(quote), _flat(lesson)
    parts = re.split(r"\s*(?:\.\.\.|…)\s*", flat_quote)
    if len(parts) == 1:
        # A short quote is fine when it is the whole of it: «ga-to» is the error.
        return len(flat_quote) >= 3 and flat_quote in flat
    # Cut by the model: each piece long enough not to match anywhere by chance.
    pieces = [p for p in parts if len(p) >= 8]
    return bool(pieces) and all(piece in flat for piece in pieces)


def _profile_text(profile: Any) -> str:
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump(mode="json", exclude_none=True)
    if not profile:
        return "(non specificato)"
    return json.dumps(profile, ensure_ascii=False)


def parse_issues(raw: str) -> list[Issue]:
    """The issues in the model's answer; a malformed one is skipped, not fatal."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the answer")
    data = json.loads(match.group(0))
    issues = []
    for item in data.get("issues") or []:
        if not isinstance(item, dict):
            continue
        check = str(item.get("check") or "").strip()
        severity = str(item.get("severity") or "").strip().lower()
        # Unknown or missing: a revision, the cautious way (no blind replacement).
        kind = str(item.get("kind") or "").strip().lower()
        kind = kind if kind in KINDS else "revisione"
        issues.append(
            Issue(
                check=check if check in CHECKS else "altro",
                # A wrong fact, example, answer key or a danger is never minor,
                # whatever the model says: «l'acqua si trasforma in aria» came
                # back «minore» on 2026-10-19 and stayed in the lesson.
                severity="bloccante"
                if check in ALWAYS_BLOCKING
                else (severity if severity in SEVERITIES else "minore"),
                quote=str(item.get("quote") or "").strip(),
                why=str(item.get("why") or "").strip(),
                fix=str(item.get("fix") or "").strip(),
                kind=kind,
            )
        )
    return issues


def _missed(facts: list[Any], issues: list[Issue]) -> list[Issue]:
    """The wrong facts the judge did not report, as problems: the model cannot forget them.

    Minutes are left out: a whole-lesson count the validator already acts on, with
    no sentence to correct in place.
    """
    reported = " ".join(_flat(i.quote) for i in issues)
    missed = []
    for fact in facts:
        if not fact.wrong or "minuti" in fact.statement or _flat(fact.quote) in reported:
            continue
        missed.append(
            Issue(
                check=fact.check,
                severity="bloccante",
                quote=fact.quote,
                why=fact.statement,
                fix=fact.fix or fact.statement,
                kind="sostituzione" if fact.fix else "revisione",
                computed=True,
            )
        )
    return missed


async def judge_lesson(
    lesson: str,
    *,
    request: str = "",
    profile: Any = None,
    model: Optional[str] = None,
    client: Any = None,
) -> JudgeReport:
    """Run the checklist on ``lesson``. Never raises: a failure is in ``error``."""
    from aix.agent.validation.facts import computed_facts, render_facts
    from aix.core.config import config as app_config
    from aix.core.config import reasoning_extra_body

    model = model or judge_model()
    report = JudgeReport(model=model)
    checks = "\n".join(f"- {key}: {text}" for key, text in CHECKS.items())
    facts = computed_facts(lesson, profile) if facts_enabled() else []
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": _USER.format(
                request=request or "(non indicata)",
                profile=_profile_text(profile),
                checks=checks,
                facts=(
                    f"## Verifiche fatte dal sistema (esatte)\n{render_facts(facts)}\n\n"
                    if facts
                    else ""
                ),
                lesson=lesson,
            ),
        },
    ]
    extra = dict(reasoning_extra_body(model) or {})
    if routing := _provider_routing(model):
        extra["provider"] = routing
    extra["usage"] = {"include": True}
    started = time.monotonic()
    try:
        client = client or app_config.openai.get_async_client()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0,
            extra_body=extra,
        )
        raw = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        report.input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        report.output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = getattr(usage, "cost", None)
        report.cost = float(cost) if isinstance(cost, (int, float)) else None
        right = [_flat(f.quote) for f in facts if not f.wrong]
        for issue in parse_issues(raw):
            if any(_flat(issue.quote) and _flat(issue.quote) in quote for quote in right):
                # A computed fact says this is right («lu-na»): the model is wrong.
                continue
            (report.issues if quote_in(issue.quote, lesson) else report.unquoted).append(issue)
        report.issues += _missed(facts, report.issues)
    except Exception as exc:  # noqa: BLE001 — the judge never breaks a lesson
        report.error = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("[judge] %s", report.error)
    report.seconds = round(time.monotonic() - started, 1)
    return report


_VERIFY_SYSTEM = """Sei un revisore esperto di didattica e delle discipline \
scolastiche. Una lezione è stata corretta in alcuni punti. Per ogni modifica ricevi \
il testo prima, il testo dopo e il problema che doveva risolvere.

Per ciascuna rispondi:
- «ok» se il testo nuovo è corretto nei contenuti e non introduce errori;
- «errata» SOLO se il testo nuovo contiene un errore di contenuto (un fatto, un \
esempio, una soluzione sbagliati), un rischio per gli alunni, o non ha senso: \
una frase rotta o che si ripete, o un'istruzione per chi corregge al posto del \
testo («Eliminare la sezione…», due alternative con «oppure»).

Ricevi la riga intera com'era e com'è ora, non solo il pezzo cambiato.

Il testo nuovo si confronta con il testo prima, non con una lezione ideale: se è \
meglio di prima o uguale, è «ok». Un problema risolto solo in parte, un problema \
che c'era già prima, o una scelta didattica diversa dalla tua sono «ok».

Prima scrivi il perché in una o due frasi, poi il verdetto che ne segue. Rispondi \
solo con JSON: {"verdicts": [{"id": 1, "why": "...", "verdict": "ok"}]}"""


@dataclass
class Verification:
    """Per change, whether it may stay: ``ok[i]`` for ``changes[i]``."""

    ok: list[bool] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    cost: Optional[float] = None
    seconds: float = 0.0
    error: str = ""


def in_lines(before: str, after: str, lesson: str) -> tuple[str, str]:
    """``before`` and ``after`` widened to the whole lines ``after`` sits in."""
    at = lesson.find(after) if lesson and after else -1
    if at < 0:
        return before, after
    start = lesson.rfind("\n", 0, at) + 1
    end = lesson.find("\n", at + len(after))
    line = lesson[start : len(lesson) if end == -1 else end]
    return line.replace(after, before, 1), line


async def verify_changes(
    changes: list[Any],
    *,
    lesson: str = "",
    request: str = "",
    model: Optional[str] = None,
    client: Any = None,
) -> Verification:
    """Ask, for each change the repair made, whether the new text is right.

    Not the whole checklist again: on 2026-10-19 a second full review undid good
    repairs -- it quoted the corrected answer key for a pedagogical remark, and a
    rewritten line while saying its examples were correct. Asked only about the
    changes, the question is the one that matters: is the new text wrong? On a
    failure every change stays (``ok`` all true) and ``error`` says why.

    With ``lesson`` (the repaired one), each change is shown as the whole lines
    it sits in: a fix that broke its sentence is visible only there.
    """
    from aix.core.config import config as app_config
    from aix.core.config import reasoning_extra_body

    result = Verification(ok=[True] * len(changes), reasons=[""] * len(changes))
    if not changes:
        return result
    model = model or judge_model()
    blocks = []
    for n, change in enumerate(changes, 1):
        problems = "; ".join(i.why for i in change.issues if i.why) or "-"
        before, after = in_lines(change.before, change.after, lesson)
        blocks.append(
            f"### Modifica {n}\nProblema: {problems}\n\nPrima:\n{before}\n\nDopo:\n{after}"
        )
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {
            "role": "user",
            "content": f"## Richiesta della docente\n{request or '(non indicata)'}\n\n"
            + "\n\n".join(blocks),
        },
    ]
    extra = dict(reasoning_extra_body(model) or {})
    if routing := _provider_routing(model):
        extra["provider"] = routing
    extra["usage"] = {"include": True}
    started = time.monotonic()
    try:
        client = client or app_config.openai.get_async_client()
        response = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=1500, temperature=0, extra_body=extra
        )
        raw = response.choices[0].message.content or ""
        cost = getattr(getattr(response, "usage", None), "cost", None)
        result.cost = float(cost) if isinstance(cost, (int, float)) else None
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        for item in json.loads(match.group(0) if match else "{}").get("verdicts") or []:
            index = int(item.get("id", 0)) - 1
            if 0 <= index < len(changes):
                result.ok[index] = str(item.get("verdict", "ok")).strip().lower() != "errata"
                result.reasons[index] = str(item.get("why") or "")
    except Exception as exc:  # noqa: BLE001 — the changes stay; the judge never breaks a lesson
        result.error = f"{type(exc).__name__}: {exc}"[:300]
        logger.warning("[judge] verification failed: %s", result.error)
    result.seconds = round(time.monotonic() - started, 1)
    return result
