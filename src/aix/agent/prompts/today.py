"""Today's date, for the agents that judge and write about the world.

A lesson on current affairs, 2026-09-25: the Critic asked the Writer to clarify
whether the 2026 migration crisis in Ceuta was "a hypothetical scenario" and to
find "a more robust source for a future event". The event was real and its
source had been fetched that day; the model simply did not know what day it
was, and anything after its training reads to it as the future. Neither the
Writer nor the Critic was ever told the date.

The note says two things, and the second matters as much as the first: events
the provided sources place before today are real, and events no source reports
must not be added. Knowing the date is a reason to trust the sources, not a
licence to write news from memory.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

_MONTHS = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)


def today_note(today: Optional[date] = None) -> str:
    """A short section to append to a prompt; ``today`` is for tests."""
    today = today or date.today()
    spoken = f"{today.day} {_MONTHS[today.month - 1]} {today.year}"
    return (
        "\n\n## Data di oggi\n"
        f"Oggi è il {spoken}. Fatti che le fonti fornite collocano fino a oggi sono reali, "
        "anche se non li conosci: non trattarli come ipotetici o futuri. Non aggiungere però "
        "fatti recenti che le fonti non riportano.\n"
    )
