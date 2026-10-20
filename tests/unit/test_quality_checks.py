"""What ``scripts/eval/quality_ab.py`` must catch: the third blind read (2026-10-19)
found defects in the texts that every check reported as clean."""

from __future__ import annotations

import pytest
from scripts.eval.quality_ab import measure, total_mismatch, usage_contradictions

pytestmark = pytest.mark.unit

LESSON = """# Le frazioni

## Attività 1 (20 minuti)
- Domande socratiche (*Socratic questioning*): «perché?» e «come lo sai?».
- Un compagno tutor (Adaptive Strategies).

*Fonti: Quantum jargon misuse (neuromito evitato); Visual Organizers.*

---
⚠️ **Nota sulle fonti**: Il contenuto disciplinare (argomento specifico) proviene da fonti esterne
(Wikipedia, pubblicazioni accademiche) e non è stato verificato dal Knowledge Graph FEM.
"""


def test_a_follow_up_whose_parts_exceed_what_the_graph_returned_is_caught():
    answer = (
        "Il grafo ha restituito 53 elementi per questa lezione.\n\n"
        "**Usati nella lezione (2):**\n- **A**: x\n- **B**: y\n\n"
        "**Non usati, tra i più pertinenti (5 su 51):**\n"
        + "".join(f"- **N{i}**: z\n" for i in range(5))
        + "\n**Il resto (51 elementi):** linee guida UDL."
    )

    assert total_mismatch(answer)
    assert not total_mismatch(answer.replace("51 elementi):**", "46 elementi):**"))


def test_without_a_stated_rest_nothing_claims_to_add_up():
    answer = "Il grafo ha restituito 78 elementi.\n\n**Strategie usate**\n- **A**: x\n"

    assert not total_mismatch(answer)


def test_the_follow_up_s_lists_are_read_against_the_lesson_body():
    answer = (
        "**Usati nella lezione:**\n"
        "- **Uso improprio del gergo quantistico** (*Quantum jargon misuse*): nelle sfide.\n\n"
        "**Suggerite ma non usate (2)**\n"
        "- *Socratic questioning*: le domande sono solo nell'introduzione.\n"
        "- *Adaptive Strategies* (ADHD): nessun cambio di strategia.\n"
        "- *Plusdotazione*: un bisogno, non una strategia.\n"
    )

    wrong = usage_contradictions(answer, LESSON)

    assert wrong == [
        "usato, assente: Uso improprio del gergo quantistico",
        "non usato, presente: Socratic questioning",
        "non usato, presente: Adaptive Strategies",
    ]


def test_a_note_claiming_attachments_that_were_never_sent_is_caught():
    claimed = LESSON.replace(
        "(argomento specifico) proviene da fonti esterne",
        "proviene dai materiali allegati dalla docente",
    )

    assert measure("udl", {}, claimed, "")["false_provenance"]
    assert not measure("udl", {}, LESSON, "")["false_provenance"]


def test_kg_in_the_sources_is_jargon_not_a_prompt_leak():
    row = measure("udl", {}, LESSON.replace("Visual Organizers.", "Visual Organizers (KG)."), "")

    assert row["leaks"] == [] and row["jargon"] >= 1


def test_a_list_grouped_by_need_is_read_name_by_name():
    """«**ADHD:** A, B» and «**Per tutta la classe:** C»: the label is not a strategy."""
    answer = (
        "Il grafo ha restituito 9 elementi.\n\n**Strategie usate nella lezione**\n"
        "- **ADHD:** Socratic questioning [in «Attività 1»], Adaptive Strategies.\n"
        "- **Per tutta la classe:** Feedback loops [in «Valutazione»].\n\n"
        "**Il resto:** 6 elementi, non usati o non pertinenti."
    )

    assert usage_contradictions(answer, LESSON) == ["usato, assente: Feedback loops"]
    assert not total_mismatch(answer)
