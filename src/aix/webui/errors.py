"""What a failure is allowed to say to the teacher (Fase 0 · PR 1).

Four places in the streaming path did the same thing::

    msg = str(exc) or exc.__class__.__name__
    lesson.error_message = msg[:480]

and that string is rendered in the chat. ``str(exc)`` is written for whoever
reads the logs, not for whoever is teaching tomorrow morning: SQLAlchemy puts
the failing statement in it, asyncpg puts host and database, an OS error puts
the absolute path of the container filesystem, an HTTP client puts the URL it
called. None of that helps a teacher, and some of it describes the inside of
the deployment to anyone looking over their shoulder.

It is also, on its own terms, a bad message. "connection refused" and
"ClientResponseError" tell the person nothing about whether to wait, retry, or
call someone.

So a failure produces two different things now. The teacher gets one of a small
set of sentences that say what to *do*, plus a short incident code. The log gets
the exception, the traceback and the same code. When someone reports "errore
codice 7F3A21", that code is in the logs, and nothing had to be leaked into the
page for the two to be connected.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Deliberately generic, because it is shown for anything unrecognised. It says
#: what happened, that it was recorded, and that retrying is reasonable.
GENERIC = (
    "Si è verificato un errore imprevisto durante la generazione. "
    "Il problema è stato registrato: puoi riprovare."
)

#: Recognised families, matched by exception class name rather than by import.
#: Importing openai, asyncpg and neo4j here to test with ``isinstance`` would
#: make an error path depend on four optional packages being installed — and an
#: error path that raises while reporting an error is the worst kind.
_BY_NAME: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("RateLimitError", "TooManyRequests"),
        "Il servizio di generazione è momentaneamente sovraccarico. Riprova fra qualche minuto.",
    ),
    (
        ("APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout"),
        "La generazione ha impiegato troppo tempo e è stata interrotta. "
        "Riprova, oppure riduci la richiesta.",
    ),
    (
        ("APIConnectionError", "ConnectionError", "ClientConnectorError", "ServiceUnavailable"),
        "Non è stato possibile contattare il servizio di generazione. "
        "Riprova fra qualche minuto; se persiste, segnalalo al referente tecnico.",
    ),
    (
        ("AuthenticationError", "PermissionDeniedError", "AuthError"),
        "Il servizio di generazione ha rifiutato le credenziali della "
        "piattaforma. Serve un intervento del referente tecnico.",
    ),
    (
        ("AtCapacity",),
        "Ci sono troppe generazioni in corso in questo momento. Riprova fra qualche minuto.",
    ),
)


@dataclass(frozen=True)
class UserFacingError:
    """What the teacher sees, and the code that ties it to a log line."""

    message: str
    incident: str

    def __str__(self) -> str:
        return f"{self.message} (codice {self.incident})"


def _classify(exc: BaseException) -> str:
    """The sentence for this failure, by exception class name.

    Walks the class hierarchy so a subclass of a recognised error is still
    recognised — ``openai.RateLimitError`` arriving wrapped, for instance.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    for candidates, message in _BY_NAME:
        if names.intersection(candidates):
            return message
    return GENERIC


def report(exc: BaseException, *, context: str = "") -> UserFacingError:
    """Log the real failure, return what may be shown.

    The incident code is short on purpose: a teacher may have to read it out
    loud or type it into a message. Six hex characters are plenty to find one
    line in a day of logs, and it is not an identifier of anything — it is
    minted here and never stored anywhere but the log and the message.
    """
    incident = secrets.token_hex(3).upper()
    # ``exc_info=exc`` and not ``logger.exception``: the latter reads
    # ``sys.exc_info()``, so it only records anything when it is called from
    # inside an ``except`` block. Every caller today is — but a caller that was
    # not would log an incident code with no exception attached to it, which
    # loses exactly the thing the code exists to find.
    logger.error("[webui] failure incident=%s %s", incident, context or "", exc_info=exc)
    return UserFacingError(message=_classify(exc), incident=incident)
