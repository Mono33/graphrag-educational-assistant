"""What the Knowledge Graph returned, in the order and the shape a lesson can use.

The retriever merges the results of each search in the order the searches ran,
and every reader took the head of that list: the Writer the first 20 nodes, the
first 10 recommendations and the first 15 relationships; the retriever card the
first 5 names. On the staging lesson of 2026-10-16 (2nd grade, water cycle, two
pupils with ADHD and two Spanish speakers) the first search had been about the
class, so the head was eight class-profile nodes -- «Group till 15», «Group till
20», «Group till 30», «Cohesive», «Split in groups» -- with no description; the
card showed them as the «most relevant concepts», and the Writer's 15
relationships were all about class size. Of 85 nodes, the ten ADHD and
non-native-speaker difficulties and the strategies the graph links to them
(«Chunking instructions», «Bilingual resources», «Focus Tasks», and the two it
advises *against* for ADHD, «Long Frontal Lesson» and «Passive Learning») never
reached the Writer. The lesson's differentiation was the model's own knowledge.

This module is the one place that decides:

* :func:`rank_nodes` / :func:`rank_recommendations` -- strategies and tools
  first, the learner-need nodes next, class-profile nodes last;
* :func:`render_evidence` -- the evidence as the Writer and the light turns read
  it: each need of *this* class with what the graph suggests and advises
  against, the class-profile nodes that match *this* class, the learning
  environment and the UDL checkpoints when they were retrieved, then the
  strategies with their descriptions;
* :func:`digest` -- the same text, bounded, kept with the lesson so a follow-up
  turn can talk about what the graph returned (until 2026-10-16 it could not:
  «Non ho accesso diretto al grafo»).

**Two graphs, two ways of saying what a node is.** The UDL graph says it with
labels (``Adhd``, ``Colour``, ``Context``…). The neuroscience graph has one
label per concept, often for a single node, and says what a node is with its
``category`` property: «ADHD» (the difficulties), «Teaching Practices» and
«Support Systems» (what to do), «Educational Results» and «Pedagogical
Strategies» (effects: «Reduced Anxiety», «Learning Gains» -- the neuro lesson of
2026-10-16 offered them as strategies), «External Influences» (the room),
«Neuromyths». And several myths are filed under «Teaching Practices»
(«Pseudoscientific exercises» is Brain Gym, «Modality preference myth» the
learning styles): read as categories alone, the reordering would have put them
at the head of the strategies. So a myth is recognised first, and only ever
shown as something not to do.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

#: KG learner-variability labels, with the name the teacher knows them by.
NEED_GROUPS: dict[str, str] = {
    "Adhd": "ADHD",
    "ForeignStudents": "Competenze linguistiche",
    "AutismSpectrum": "Spettro autistico",
    "Dyslexia": "Dislessia",
    "Dyscalculia": "Discalculia",
    "Gifted": "Plusdotazione",
    "SensoryDisabilities": "Disabilità sensoriali",
}

#: Profile need codes (``educational_profile.DISABILITY_LABELS``) -> KG labels.
PROFILE_NEED_LABELS: dict[str, set[str]] = {
    "ADHD": {"Adhd"},
    "DLDS": {"ForeignStudents"},
    "CL": {"ForeignStudents"},
    "SA": {"AutismSpectrum"},
    "DSA": {"Dyslexia", "Dyscalculia"},
    "PD": {"Gifted"},
}

#: Profile class features -> the name of the KG ``Context`` node that describes them.
PROFILE_CONTEXT_NAMES: dict[str, str] = {
    "COESA": "cohesive",
    "DIVISA_IN_GRUPPI": "split in groups",
    "ELEMENTI_DI_DISTURBO": "with disruptive elements",
    "MOTIVATA": "motivated",
    "GENDER_GAP": "gender gap",
}

#: The physical classroom, as the UDL graph models it (lighting, colours, noise…).
ENVIRONMENT_LABELS: frozenset[str] = frozenset(
    {
        "Colour",
        "Lighting",
        "Acoustics",
        "Furniture",
        "Smells",
        "Textures",
        "SensoryEnvironment",
        "SpatialAdaptability",
    }
)
UDL_LABELS: frozenset[str] = frozenset({"Principle", "Guideline", "Checkpoint"})
CONTEXT_LABEL = "Context"

#: The neuroscience graph's ``category`` values (lowercased), by what they hold.
#: Verified on the graph on 2026-10-16; none of them is a UDL category.
NEURO_NEED_CATEGORIES: dict[str, str] = {
    "adhd": "ADHD",
    "dyslexia": "Dislessia",
    "dyscalculia": "Discalculia",
    "autism spectrum": "Spettro autistico",
    "tourette": "Sindrome di Tourette",  # stored as «Tourette’s Disorder»
}
NEURO_STRATEGY_CATEGORIES: frozenset[str] = frozenset(
    {"teaching practices", "teachingpractices", "support systems", "learningstrategies", "feedback"}
)
NEURO_OUTCOME_CATEGORIES: frozenset[str] = frozenset(
    {"pedagogical strategies", "educational results", "developmental outcomes", "evidence-based"}
)
NEURO_ENVIRONMENT_CATEGORIES: frozenset[str] = frozenset({"external influences"})
NEURO_ENVIRONMENT_NAMES: frozenset[str] = frozenset(
    {"educational environment", "supportive classroom environment", "mistake-friendly classroom"}
)
#: «Not evidence-based» holds the concepts a myth claims to improve («Retention»,
#: «Focus»), carrying the myth's description: neither a strategy nor a myth to name.
NEURO_NOISE_CATEGORIES: frozenset[str] = frozenset({"not evidence-based"})
_MYTH = re.compile(r"\bmyths?\b|pseudoscien|debunked", re.IGNORECASE)

#: Edges that say «use this» / «do not use this». VECTOR_SIMILAR is an embedding
#: neighbourhood, not a pedagogical claim: «Group till 15 —VECTOR_SIMILAR→ With
#: disruptive elements» tells a teacher nothing.
_POSITIVE = frozenset({"SUGGESTS", "MITIGATED_BY", "SUPPORTS", "SUPPORTS_BY", "REDUCES"})
_NEGATIVE = frozenset({"NO_SUGGESTS"})
_STRATEGY = re.compile(
    r"Strateg|Methodolog|Approach|Technique|Framework|Practice|Support|Scaffold|Instruction",
    re.IGNORECASE,
)
_TOOL = re.compile(r"Tool$", re.IGNORECASE)
_CLASS_SIZE = re.compile(r"till\s*(\d+)", re.IGNORECASE)

#: Bounds: the Writer reads the evidence on every lesson and the light turn on
#: every follow-up, so it is kept to a few thousand characters (~1.5k tokens).
MAX_CHARS = 9000
_MAX_STRATEGIES = 24
_DESC_CHARS = 160


def _labels(node: dict[str, Any]) -> list[str]:
    raw = node.get("labels") or []
    return [str(label) for label in (raw if isinstance(raw, (list, tuple)) else [raw])]


def _name(node: dict[str, Any]) -> str:
    return " ".join(str(node.get("name") or node.get("title") or "").split())


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _category(node: dict[str, Any]) -> str:
    return _norm(node.get("category") or "")


def neuro_need(node: dict[str, Any]) -> Optional[str]:
    """The need a neuroscience node describes («ADHD»), or ``None``."""
    category = _category(node)
    for prefix, group in NEURO_NEED_CATEGORIES.items():
        if category.startswith(prefix):
            return group
    # The node «Adhd» itself sits under «Neurodevelopmental Processes».
    return "ADHD" if _norm(_name(node)) == "adhd" else None


def is_myth(node: dict[str, Any]) -> bool:
    """A neuromyth, whatever category the graph filed it under."""
    if _category(node) == "neuromyths":
        return True
    return bool(_MYTH.search(_name(node)))


def kind(node: dict[str, Any]) -> str:
    """``strategy``, ``tool``, ``need``, ``environment``, ``udl``, ``context``,
    ``outcome``, ``myth``, ``noise`` or ``other``. See the module docstring."""
    labels = _labels(node)
    if CONTEXT_LABEL in labels:
        return "context"
    if is_myth(node):
        return "myth"
    if any(label in NEED_GROUPS for label in labels) or neuro_need(node):
        return "need"
    category = _category(node)
    if (
        any(label in ENVIRONMENT_LABELS for label in labels)
        or category in NEURO_ENVIRONMENT_CATEGORIES
        or _norm(_name(node)) in NEURO_ENVIRONMENT_NAMES
    ):
        return "environment"
    if any(label in UDL_LABELS for label in labels):
        return "udl"
    if any(_TOOL.search(label) for label in labels):
        return "tool"
    if category in NEURO_NOISE_CATEGORIES:
        return "noise"
    if category in NEURO_OUTCOME_CATEGORIES:
        return "outcome"
    if category in NEURO_STRATEGY_CATEGORIES or any(_STRATEGY.search(label) for label in labels):
        return "strategy"
    return "other"


def need_group(node: dict[str, Any]) -> Optional[str]:
    """The teacher's name for the need a node describes, in either graph."""
    for label in _labels(node):
        if label in NEED_GROUPS:
            return NEED_GROUPS[label]
    return neuro_need(node)


_TIER = {
    "strategy": 0,
    "udl": 1,
    "environment": 1,
    "need": 2,
    "tool": 3,
    "outcome": 4,
    "other": 4,
    "context": 5,
    "myth": 6,
    "noise": 7,
}


def _score(item: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def rank_nodes(nodes: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Strategies first, class-profile nodes last; the retrieval's score within a tier.

    Stable, so among equals the order of the searches is kept.
    """
    items = [n for n in nodes or [] if isinstance(n, dict)]
    return sorted(items, key=lambda n: (_TIER[kind(n)], -_score(n, "rank_score", "score")))


def rank_recommendations(
    recommendations: Optional[list[dict[str, Any]]],
    nodes: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Strategies first, in the nodes' tiers; the most relevant first within a tier.

    «Apply Group till 15 with appropriate adaptations» is not a strategy, and on
    2026-10-16 four of the Writer's ten «Teaching Methodologies» were that. By
    relevance alone the head was five assessment designs (score 1.0), so the
    tier of the recommendation's node comes first.
    """
    tiers = {_name(n): _TIER[kind(n)] for n in nodes or [] if isinstance(n, dict)}
    items = [r for r in recommendations or [] if isinstance(r, dict)]

    def _key(rec: dict[str, Any]) -> tuple[int, float]:
        if "contesto" in _norm(rec.get("category") or ""):
            tier = _TIER["context"]
        else:
            tier = tiers.get(_name(rec), _TIER["other"])
        return (tier, -_score(rec, "relevance_score", "score"))

    return sorted(items, key=_key)


def is_context_recommendation(rec: dict[str, Any], nodes: Optional[list[dict[str, Any]]]) -> bool:
    context = {_name(n) for n in nodes or [] if isinstance(n, dict) and kind(n) == "context"}
    return _name(rec) in context or "contesto" in _norm(rec.get("category") or "")


# ---------------------------------------------------------------------------
# The profile: which needs and which class descriptors are *this* class's
# ---------------------------------------------------------------------------


def _profile_group(profile: Any) -> dict[str, Any]:
    if hasattr(profile, "model_dump"):
        profile = profile.model_dump(mode="json")
    if not isinstance(profile, dict):
        return {}
    group = profile.get("group") or {}
    return group if isinstance(group, dict) else {}


def _codes(value: Any) -> list[str]:
    if not value:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [str(getattr(v, "value", v) or "").strip().upper() for v in values if v]


def class_need_labels(profile: Any) -> set[str]:
    """The KG need labels of the class's declared needs (empty when none is declared)."""
    group = _profile_group(profile)
    raw = group.get("disabilities")
    if not raw and isinstance(profile, dict):
        raw = profile.get("disabilities")
    labels: set[str] = set()
    for code in _codes(raw):
        labels |= PROFILE_NEED_LABELS.get(code, set())
    return labels


#: The class-profile nodes as a teacher would say them. «Group till 30» reached
#: the teacher verbatim, in the sources and in the follow-up answers, in 3 of 8
#: lessons of the 2026-10-16 evaluation (``scripts/eval/quality_ab.py``).
_CONTEXT_LABELS: dict[str, str] = {
    "cohesive": "Classe coesa",
    "split in groups": "Classe divisa in gruppi",
    "with disruptive elements": "Classe con elementi di disturbo",
    "motivated": "Classe motivata",
    "gender gap": "Classe con divario di genere",
}


def context_label(node: dict[str, Any]) -> str:
    """A class-profile node's name in Italian («Group till 30» -> «Classe fino a 30 alunni»)."""
    name = _name(node)
    size = _CLASS_SIZE.search(name)
    if size:
        return f"Classe fino a {size.group(1)} alunni"
    return _CONTEXT_LABELS.get(_norm(name), name)


#: Profile need codes -> the teacher's names of the needs (``need_group``).
PROFILE_NEED_GROUPS: dict[str, set[str]] = {
    "ADHD": {"ADHD"},
    "DLDS": {"Competenze linguistiche"},
    "CL": {"Competenze linguistiche"},
    "SA": {"Spettro autistico"},
    "DSA": {"Dislessia", "Discalculia"},
    "PD": {"Plusdotazione"},
}


#: How teachers name the needs in a request, when the profile does not carry them
#: («Ci sono 2 bambini con adhd e 2 non italofoni»): the WebUI form is optional.
_NEEDS_IN_TEXT: dict[str, re.Pattern[str]] = {
    "ADHD": re.compile(r"\badhd\b|iperattiv|deficit (?:di|dell')\s?attenzione", re.I),
    "Competenze linguistiche": re.compile(
        r"non italofon|stranier|\bl2\b|italiano come (?:seconda|lingua seconda)"
        r"|madre ?lingua (?!italian)|lingua madre (?!italian)",
        re.I,
    ),
    "Dislessia": re.compile(r"dislessi|\bdsa\b", re.I),
    "Discalculia": re.compile(r"discalculi|\bdsa\b", re.I),
    "Spettro autistico": re.compile(r"autis|asperger|spettro autistico", re.I),
    "Plusdotazione": re.compile(r"plusdotat|alto potenziale|gifted", re.I),
}


def class_need_groups(profile: Any, request: str = "") -> set[str]:
    """The class's needs, by the names ``need_group`` gives them: those the profile
    declares and those the request names. Empty when neither names one -- and then
    no need is shown: on 2026-10-16 a class with none declared got every need the
    graph returned (ADHD, autism, non-native speakers), and the follow-up answer
    presented them as the class's (``scripts/eval/quality_ab.py``, case 4)."""
    group = _profile_group(profile)
    raw = group.get("disabilities")
    if not raw and isinstance(profile, dict):
        raw = profile.get("disabilities")
    groups: set[str] = set()
    for code in _codes(raw):
        groups |= PROFILE_NEED_GROUPS.get(code, set())
    for need, pattern in _NEEDS_IN_TEXT.items():
        if request and pattern.search(request):
            groups.add(need)
    # «DSA» covers dyslexia and dyscalculia; «un alunno con DSA (discalculia)» is
    # one of them. The follow-up of 2026-10-19 gave the class a dyslexia it had
    # not declared, and a «lacuna» about it.
    specific = {
        need
        for need, word in (("Dislessia", "dislessi"), ("Discalculia", "discalculi"))
        if word in request.lower()
    }
    if specific:
        groups -= {"Dislessia", "Discalculia"} - specific
    return groups


#: What the graph advises against, as a lesson would describe it in Italian:
#: the gloss tells the Writer what the format is, the pattern finds it in a lesson
#: that renamed it. On 2026-10-19 a lesson for a class the graph says should not
#: rotate between stations had «Le fasi a stazioni»: groups rotating over four
#: «cartellini-stazione», and nothing caught it under the Italian name.
ADVISED_GLOSSES: dict[str, tuple[str, re.Pattern[str]]] = {
    "station rotation": (
        "gruppi che ruotano tra stazioni o postazioni",
        # Something has to rotate: «Caccia alle fasi a stazioni» with every group
        # working at its own table is not a rotation (a false alarm of 2026-10-19).
        re.compile(
            r"(?:ruot|rotazion|girano|si spostano|cambiano)\w*[^.\n]{0,40}(?:stazion|postazion)"
            r"|(?:stazion|postazion)\w*[^.\n]{0,40}(?:a rotazione|ruot|girano)",
            re.IGNORECASE,
        ),
    ),
    "long frontal lesson": (
        "spiegazione frontale lunga",
        re.compile(r"lezione frontale (?:lunga|di \d{2})|spiegazione frontale di \d{2}", re.I),
    ),
    "passive learning": (
        "ascolto passivo prolungato",
        re.compile(r"ascolto passivo|apprendimento passivo", re.IGNORECASE),
    ),
}


def _with_gloss(name: str) -> str:
    gloss = ADVISED_GLOSSES.get(_norm(name))
    return f"{name} ({gloss[0]})" if gloss else name


#: Where the sources begin, in the shapes the Writer gives them: «*Fonti:*», «##
#: Fonti:», a single italic line «*Fonti: Cooperative learning (…); …*», or a
#: section of resources for the teacher. The last two were read as the lesson's
#: body, and every node they list counted as used (third blind read, 2026-10-19).
_SOURCES = re.compile(
    r"^[ \t]*[*_]{0,2}Fonti[ \t]*:|^#+[ \t]*(?:Fonti|Riferimenti|Risorse (?:di approfondimento|per|video))"
    r"|nota sulle fonti",
    re.IGNORECASE | re.MULTILINE,
)


def _advised_in_lesson(name: str, lesson: str) -> bool:
    """Whether the lesson uses ``name``, by its graph name or as a lesson describes it.

    Only the body counts: «Station Rotation (non applicata, … esclusa)» in the
    sources is the opposite of using it.
    """
    lesson = lesson_body(lesson)
    if _in_lesson(name, _norm(lesson)):
        return True
    gloss = ADVISED_GLOSSES.get(_norm(name))
    return bool(gloss and gloss[1].search(lesson))


def _matching_context(context_nodes: list[dict[str, Any]], profile: Any) -> list[dict[str, Any]]:
    """The class-profile nodes that describe this class: its size band and its features."""
    group = _profile_group(profile)
    wanted = {
        PROFILE_CONTEXT_NAMES[c]
        for c in _codes(group.get("class_features"))
        if c in PROFILE_CONTEXT_NAMES
    }
    students = group.get("students_number")
    size_node: Optional[dict[str, Any]] = None
    if isinstance(students, int) and students > 0:
        bands = []
        for node in context_nodes:
            match = _CLASS_SIZE.search(_name(node))
            if match and int(match.group(1)) >= students:
                bands.append((int(match.group(1)), node))
        if bands:
            size_node = min(bands, key=lambda band: band[0])[1]
    return [node for node in context_nodes if node is size_node or _norm(_name(node)) in wanted]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _edges(relationships: Optional[list[dict[str, Any]]]) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str, str]] = set()
    edges: list[tuple[str, str, str]] = []
    for rel in relationships or []:
        if not isinstance(rel, dict):
            continue
        triple = (
            " ".join(str(rel.get("source") or "").split()),
            str(rel.get("type") or rel.get("relationship") or "").strip().upper(),
            " ".join(str(rel.get("target") or "").split()),
        )
        if all(triple) and triple not in seen:
            seen.add(triple)
            edges.append(triple)
    return edges


def _links(
    name: str, edges: list[tuple[str, str, str]], by_name: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """What a need or context node suggests and advises against, either edge direction."""
    suggests: list[str] = []
    avoid: list[str] = []
    for source, rel, target in edges:
        if source == name:
            other = target
        elif target == name and rel in _POSITIVE:
            other = source
        else:
            continue
        other_node = by_name.get(other)
        if other_node is not None and kind(other_node) in ("need", "context"):
            continue
        bucket = avoid if (rel in _NEGATIVE and source == name) else suggests
        if rel in _POSITIVE or rel in _NEGATIVE:
            if other not in bucket:
                bucket.append(other)
    return suggests, avoid


_MAX_MYTHS = 6

#: Words that make a myth relevant: to any differentiation, and to a need.
_MYTH_ALWAYS = ("style", "modality", "vark")
_MYTH_BY_NEED = {
    "ADHD": (
        "sugar",
        "hyperactiv",
        "water",
        "hydration",
        "attention",
        "multitask",
        "movement",
        "brain gym",
        "pseudoscien",
    ),
    "Dislessia": ("dyslexia", "overlay", "coloured", "reading", "water"),
    "Spettro autistico": ("sensory", "music", "mozart"),
}


def _myth_relevance(node: dict[str, Any], needs: set[str]) -> int:
    text = _norm(f"{_name(node)} {node.get('description') or ''}")
    score = sum(2 for word in _MYTH_ALWAYS if word in text)
    for need in needs:
        score += sum(1 for word in _MYTH_BY_NEED.get(need, ()) if word in text)
    return score


#: The neuroscience graph's relations, as the Writer reads them.
_EFFECT_VERBS = {
    "TRIGGERS": "scatena",
    "LEADS_TO": "porta a",
    "REDUCES": "riduce",
    "IMPAIRS": "compromette",
    "INTERFERES_WITH": "interferisce con",
    "UNDERMINES": "indebolisce",
    "DISRUPTS": "disturba",
    "FACILITATES": "favorisce",
    "SUPPORTS": "sostiene",
    "ENHANCES": "potenzia",
    "IMPROVES": "migliora",
}


def _effects(name: str, edges: list[tuple[str, str, str]], limit: int = 5) -> list[str]:
    """What a neuroscience need node brings about, in a few words each."""
    found: list[str] = []
    for source, rel, target in edges:
        if source != name or rel not in _EFFECT_VERBS:
            continue
        phrase = f"{_EFFECT_VERBS[rel]} {target}"
        if phrase not in found:
            found.append(phrase)
        if len(found) >= limit:
            break
    return found


def advised_against(
    nodes: Optional[list[dict[str, Any]]],
    relationships: Optional[list[dict[str, Any]]],
    profile: Any = None,
    request: str = "",
) -> set[str]:
    """What the graph advises against for this class's needs and class profile.

    «Long Frontal Lesson» for ADHD, «Station Rotation» for a class of 22, and in
    the neuroscience graph its myths («Pseudoscientific exercises» is Brain Gym):
    names no reader should offer as strategies to pick.
    """
    ranked = rank_nodes(nodes)
    by_name: dict[str, dict[str, Any]] = {}
    for node in ranked:
        by_name.setdefault(_name(node), node)
    edges = _edges(relationships)
    wanted = class_need_groups(profile, request)
    sources = [node for node in ranked if kind(node) == "need" and need_group(node) in wanted]
    sources += _matching_context([n for n in ranked if kind(n) == "context"], profile)
    names: set[str] = set()
    for node in sources:
        names.update(_links(_name(node), edges, by_name)[1])
    # Never offered as something to use: a myth, and the concepts a myth claims
    # to improve, which carry the myth's description (see the module docstring).
    names.update(_name(node) for node in ranked if kind(node) in ("myth", "noise"))
    return names


def _short(text: Any, limit: int = _DESC_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


#: What the context builder writes when it has nothing to say about a node.
_BOILERPLATE = re.compile(r"^Apply .+ with appropriate adaptations$", re.IGNORECASE)


def guidance_by_name(recommendations: Optional[list[dict[str, Any]]]) -> dict[str, str]:
    """A recommendation's guidance by name, for nodes that came without a description."""
    guidance: dict[str, str] = {}
    for rec in recommendations or []:
        if not isinstance(rec, dict):
            continue
        text = " ".join(
            str(rec.get("implementation_guidance") or rec.get("description") or "").split()
        )
        if text and not _BOILERPLATE.match(text):
            guidance.setdefault(_name(rec), text)
    return guidance


#: Typos in the graph's names, corrected where a teacher would read them. The
#: names are kept as they are everywhere else: queries and edges match on them.
#: The graph itself should be fixed (``docs/product/Analisi_Qualita_Risposte.md``
#: §5, proposal 6); until then «Gammification» reached the lessons.
GRAPH_TYPOS: dict[str, str] = {
    "Gammification": "Gamification",
    "Poject": "Project",
    "sustaints": "sustains",
}
_TYPO = re.compile("|".join(re.escape(typo) for typo in GRAPH_TYPOS))


def fix_graph_typos(text: str) -> str:
    """``text`` with the graph's known typos corrected."""
    return _TYPO.sub(lambda match: GRAPH_TYPOS[match.group(0)], text) if text else text


def render_evidence(*args: Any, **kwargs: Any) -> str:
    """The retrieved graph as the lesson can use it: see :func:`_render_evidence`.

    The graph's typos are corrected in what it returns (:func:`fix_graph_typos`).
    """
    return fix_graph_typos(_render_evidence(*args, **kwargs))


def _render_evidence(
    nodes: Optional[list[dict[str, Any]]],
    relationships: Optional[list[dict[str, Any]]] = None,
    profile: Any = None,
    recommendations: Optional[list[dict[str, Any]]] = None,
    *,
    max_chars: int = MAX_CHARS,
    include_strategies: bool = True,
    request: str = "",
) -> str:
    """The retrieved graph as the lesson can use it, in Italian, bounded.

    ``include_strategies=False`` leaves out the flat list of strategies, for a
    reader that lists the nodes itself (the Writer does, for every domain).
    Empty when nothing was retrieved.
    """
    ranked = rank_nodes(nodes)
    if not ranked:
        return ""
    by_name: dict[str, dict[str, Any]] = {}
    for node in ranked:
        by_name.setdefault(_name(node), node)
    edges = _edges(relationships)
    guidance = guidance_by_name(recommendations)
    advised_against: set[str] = set()
    sections: list[str] = []

    # 1. The class's needs -> what the graph suggests and advises against. The
    # neuroscience graph links a need to what it brings about rather than to
    # strategies («Difficulty Managing Frustration or Stress» TRIGGERS «Anxiety»):
    # that is the «why» of a choice, and is rendered as such.
    wanted = class_need_groups(profile, request)
    need_lines: list[str] = []
    for node in ranked:
        if kind(node) != "need":
            continue
        group = need_group(node) or ""
        if group not in wanted:
            continue
        suggests, avoid = _links(_name(node), edges, by_name)
        effects = [] if (suggests or avoid) else _effects(_name(node), edges)
        if not suggests and not avoid and not effects:
            continue
        advised_against.update(avoid)
        line = f"- **{group} · {_name(node)}**"
        if suggests:
            line += f" → il KG suggerisce: {', '.join(suggests)}"
        if avoid:
            line += f"; sconsiglia: {', '.join(_with_gloss(n) for n in avoid)}"
        if effects:
            line += f" → nel KG: {'; '.join(effects)}"
        need_lines.append(line)
    if need_lines:
        sections.append(
            "### Bisogni della classe → che cosa indica il Knowledge Graph\n"
            + "\n".join(need_lines)
        )

    # 2. The class-profile nodes that describe this class.
    context_nodes = [n for n in ranked if kind(n) == "context"]
    context_lines: list[str] = []
    for node in _matching_context(context_nodes, profile):
        suggests, avoid = _links(_name(node), edges, by_name)
        if not suggests and not avoid:
            continue
        advised_against.update(avoid)
        line = f"- **{context_label(node)}** (profilo della classe)"
        if suggests:
            line += f" → suggerisce: {', '.join(suggests)}"
        if avoid:
            line += f"; sconsiglia: {', '.join(_with_gloss(n) for n in avoid)}"
        context_lines.append(line)
    if context_lines:
        sections.append("### Profilo della classe nel Knowledge Graph\n" + "\n".join(context_lines))

    # 2b. The neuromyths retrieved: what must not be proposed, and why. The few
    # that a lesson for this class could slip into come first (learning styles
    # in any differentiation; sugar, water, multitasking, Brain Gym with ADHD).
    myth_lines = []
    # Only those: «Music-IQ myth» in a lesson on the water cycle is space taken
    # from the evidence, not a warning anyone needs.
    myths = sorted(
        (node for node in ranked if kind(node) == "myth" and _myth_relevance(node, wanted) > 0),
        key=lambda node: -_myth_relevance(node, wanted),
    )
    for node in myths[:_MAX_MYTHS]:
        line = f"- **{_name(node)}**"
        if node.get("description"):
            line += f": {_short(node['description'], 150)}"
        myth_lines.append(line)
    if myth_lines:
        sections.append("### Da evitare: neuromiti (nodi del KG)\n" + "\n".join(myth_lines))

    # 3. The learning environment, when it was retrieved.
    env_lines = []
    for node in ranked:
        if kind(node) != "environment":
            continue
        label = next(
            (label for label in _labels(node) if label in ENVIRONMENT_LABELS),
            "Ambiente",
        )
        line = f"- {label} · **{_name(node)}**"
        if node.get("description"):
            line += f": {_short(node['description'], 130)}"
        env_lines.append(line)
    if env_lines:
        sections.append("### Ambiente di apprendimento (nodi del KG)\n" + "\n".join(env_lines))

    # 4. UDL guidelines and checkpoints, when they were retrieved.
    udl_lines = []
    for node in ranked:
        if kind(node) != "udl":
            continue
        label = next(label for label in _labels(node) if label in UDL_LABELS)
        guideline = node.get("guideline")
        line = f"- {label} **{_name(node)}**" + (
            f" (linea guida: {guideline})" if guideline else ""
        )
        if node.get("description"):
            line += f": {_short(node['description'], 130)}"
        udl_lines.append(line)
    if udl_lines:
        sections.append("### Linee guida e checkpoint UDL (nodi del KG)\n" + "\n".join(udl_lines))

    # 5. The strategies and tools, with what the graph says about them. One the
    # graph advises against for this class is listed above, under its need, and
    # not here: «Long Frontal Lesson» is not a strategy to pick for ADHD.
    strategy_lines = []
    for node in ranked if include_strategies else []:
        if kind(node) not in ("strategy", "tool") or _name(node) in advised_against:
            continue
        if len(strategy_lines) >= _MAX_STRATEGIES:
            break
        tag = "strumento" if kind(node) == "tool" else "strategia"
        line = f"- **{_name(node)}** ({tag})"
        about = node.get("description") or guidance.get(_name(node))
        if about:
            line += f": {_short(about)}"
        strategy_lines.append(line)
    if strategy_lines:
        sections.append("### Strategie e strumenti recuperati\n" + "\n".join(strategy_lines))

    # 6. What the strategies are expected to bring about, per the graph. Not
    # activities: «Reduced Anxiety» is the reason for a routine, not a step.
    outcomes = [_name(n) for n in ranked if kind(n) == "outcome"][:10]
    if outcomes:
        sections.append(
            "### Effetti attesi secondo il KG (motivazioni, non attività)\n- " + ", ".join(outcomes)
        )

    # Cut line by line, not section by section: a section too long for what is
    # left keeps its first lines (the best ranked) instead of vanishing. A
    # heading with no line under it is dropped.
    text = ""
    for section in sections:
        lines = section.split("\n")
        start = len(text)
        for number, line in enumerate(lines):
            piece = ("\n\n" if text and number == 0 else "\n" if text else "") + line
            if len(text) + len(piece) > max_chars:
                return text[:start] if number <= 1 else text
            text += piece
    return text


#: How many suggested-but-unused elements a follow-up lists (the spec's «at most 5»).
_MAX_UNUSED = 5


def _in_lesson(name: str, lesson_lower: str) -> bool:
    """``name`` as whole words. A substring match counted «Stem» as used because of
    «system» in the lesson's AI-disclosure comment (second blind read, 2026-10-19)."""
    words = _norm(fix_graph_typos(name))
    return (
        bool(words)
        and re.search(rf"(?<![\w-]){re.escape(words)}(?![\w-])", lesson_lower) is not None
    )


_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)|https?://\S+")


def lesson_body(lesson: str) -> str:
    """The lesson as a teacher reads it, for telling what it uses: no HTML comment,
    no link (a video titled «What is Service Learning?» is not Service Learning
    being applied), and nothing from the sources on."""
    text = _COMMENT.sub(" ", lesson or "")
    text = _SOURCES.split(text, maxsplit=1)[0]
    return _LINK.sub(" ", text)


_SECTION = re.compile(r"^(?:#{1,6}\s+(.+)|\*\*([^*\n]+)\*\*\s*:?\s*)$")
_SECTION_MINUTES = re.compile(r"\s*[(–—-]\s*\d+\s*min[^)]*\)?\s*$", re.IGNORECASE)


def _where(name: str, body: str) -> str:
    """The section of the lesson where ``name`` first appears: «Attività 2 – Dettato»."""
    words = _norm(fix_graph_typos(name))
    pattern = re.compile(rf"(?<![\w-]){re.escape(words)}(?![\w-])")
    section = ""
    for line in body.splitlines():
        heading = _SECTION.match(line.strip())
        if heading:
            section = _SECTION_MINUTES.sub("", (heading.group(1) or heading.group(2))).strip(" *#:")
        if words and pattern.search(_norm(line)):
            return section
    return ""


def _needs_not_in_graph(profile: Any, ranked: list[dict[str, Any]]) -> list[str]:
    """The needs the teacher declared that no node of the graph's answer is about.

    «Disabilità cognitiva media» has no group in the graph: the follow-up of
    2026-10-20 said it «non compare tra i bisogni del profilo», and it did.
    Named with the platform's words.
    """
    from aix.api.schemas.educational_profile import DISABILITY_LABELS

    group = _profile_group(profile)
    raw = group.get("disabilities")
    if not raw and isinstance(profile, dict):
        raw = profile.get("disabilities")
    present = {need_group(n) for n in ranked if kind(n) == "need"}
    missing = []
    for code in _codes(raw):
        groups = PROFILE_NEED_GROUPS.get(code, set())
        if not groups & present:
            missing.append(DISABILITY_LABELS.get(code, code))
    return missing


def usage(
    nodes: Optional[list[dict[str, Any]]],
    relationships: Optional[list[dict[str, Any]]],
    profile: Any,
    lesson: str,
    request: str = "",
) -> dict[str, Any]:
    """Which of the graph's elements the lesson uses, computed rather than asked.

    A follow-up asked «what did the graph find?» used to work it out from the
    evidence and the lesson, and its counts did not match what it listed
    (``docs/product/Specifica_Risposta_Mediana.md`` §3). Matched on the graph's
    name in the lesson's text: the Writer names a strategy in Italian with the
    graph's name in brackets, so a strategy applied without its name is counted
    as unused -- the safe side.
    """
    ranked = rank_nodes(nodes)
    lesson_lower = _norm(lesson_body(lesson))
    by_name: dict[str, dict[str, Any]] = {}
    for node in ranked:
        by_name.setdefault(_name(node), node)
    edges = _edges(relationships)
    wanted = class_need_groups(profile, request)
    avoid = advised_against(nodes, relationships, profile, request)

    usable = [n for n in ranked if kind(n) in ("strategy", "tool", "environment", "udl")]
    # Used: any element of the graph the lesson names, not only the strategies --
    # Autonomy, Competence and Relatedness are outcomes in the neuro graph, and a
    # lesson built on them said «Usati: 0» (blind read, 2026-10-20).
    named = [n for n in ranked if kind(n) not in ("need", "context", "myth", "noise")]
    used = list(dict.fromkeys(_name(n) for n in named if _in_lesson(_name(n), lesson_lower)))

    unused: list[str] = []
    for node in ranked:
        if kind(node) != "need" or need_group(node) not in wanted:
            continue
        for name in _links(_name(node), edges, by_name)[0]:
            label = f"{name} ({need_group(node)})"
            if name in used or name in avoid:
                continue
            # A suggestion the lesson names is used, whatever the graph calls it:
            # Adaptive Strategies is not a «strategy» node, and was listed as
            # unused while the lesson labelled it (third blind read, 2026-10-19).
            if _in_lesson(name, lesson_lower):
                used.append(name)
            elif label not in unused:
                unused.append(label)
    myths = [_name(n) for n in ranked if kind(n) == "myth"]
    advised = sorted(name for name in avoid if name not in myths)
    # Up to five unused in all: the ones for the class's needs first, then the
    # graph's best-ranked. The follow-up used to pick these itself, from the
    # evidence, and then copied a «resto» that did not subtract them (53
    # returned: 2 used + 5 listed + «il resto 51»), or listed as unused what the
    # lesson names (Adaptive Strategies, 2026-10-19).
    unused = unused[:_MAX_UNUSED]
    other_unused: list[str] = []
    for name in (_name(n) for n in usable):
        if len(unused) + len(other_unused) >= _MAX_UNUSED:
            break
        named = {u.split(" (")[0] for u in unused}
        if name not in used and name not in avoid and name not in named | set(other_unused):
            other_unused.append(name)
    body = lesson_body(lesson)
    found = {
        "used": used,
        "where": {name: _where(name, body) for name in used},
        "unused_for_needs": unused,
        "unused_other": other_unused,
        "advised_against_used": sorted(
            name for name in advised if _advised_in_lesson(name, lesson)
        ),
        "advised_against_avoided": [n for n in advised if not _advised_in_lesson(n, lesson)],
        "myths_avoided": [n for n in myths if not _in_lesson(n, lesson_lower)][:_MAX_MYTHS],
        "needs": sorted(wanted),
        "needs_not_in_graph": _needs_not_in_graph(profile, ranked),
    }
    # The rest is everything the follow-up does not list, so that the numbers it
    # copies add up to what the graph returned («il resto: 55» of 56 did not).
    listed = {
        name.split(" (")[0]
        for key, names in found.items()
        if key != "needs" and isinstance(names, list)
        for name in names
    }
    found["total"] = len({_name(n) for n in ranked})
    found["other"] = max(0, found["total"] - len(listed))
    return found


def render_usage(found: dict[str, Any]) -> str:
    """:func:`usage` as the follow-up reads it."""
    if not found:
        return ""
    lines = ["### Uso nella lezione (calcolato dal sistema, non da stimare)"]
    if found.get("total"):
        lines.append(f"- Elementi restituiti dal grafo: {found['total']}")
    needs = ", ".join(found.get("needs") or []) or "nessuno dichiarato né nominato"
    lines.append(f"- Bisogni della classe: {needs}")
    if found.get("needs_not_in_graph"):
        lines.append(
            "- Dichiarati nel profilo, ma senza strategie nel grafo (sono comunque bisogni "
            "della classe; la lezione li tratta con scelte proprie): "
            + ", ".join(found["needs_not_in_graph"])
        )
    where = found.get("where") or {}
    used = [f"{name} [in «{where[name]}»]" if where.get(name) else name for name in found["used"]]
    lines.append(f"- Usati nella lezione ({len(used)}): {', '.join(used) or '—'}")
    unused = list(found.get("unused_for_needs") or []) + list(found.get("unused_other") or [])
    if unused:
        lines.append(
            f"- Non usati, tra i più pertinenti ({len(unused)}; tra parentesi il bisogno "
            "per cui il grafo li suggerisce): " + ", ".join(unused)
        )
    if found.get("advised_against_avoided"):
        lines.append(
            "- Sconsigliati dal grafo per questa classe ed evitati: "
            + ", ".join(found["advised_against_avoided"])
        )
    if found.get("advised_against_used"):
        lines.append(
            "- ATTENZIONE, sconsigliati dal grafo ma presenti nella lezione: "
            + ", ".join(found["advised_against_used"])
        )
    if found.get("myths_avoided"):
        lines.append("- Neuromiti evitati: " + ", ".join(found["myths_avoided"]))
    lines.append(
        f"- Il resto, non usato o non pertinente (esclusi tutti quelli elencati sopra): "
        f"{found.get('other', 0)}"
    )
    return fix_graph_typos("\n".join(lines))


def digest(
    state: dict[str, Any], profile: Any = None, lesson: str = ""
) -> Optional[dict[str, Any]]:
    """What the graph returned for the turn, small enough to keep with the lesson.

    ``None`` when nothing was retrieved. Kept in the Writer message's
    ``meta_json`` and read by the light turns (see ``aix.webui.agent.light``).
    """
    nodes = state.get("retrieved_nodes") or []
    if not nodes:
        return None
    text = render_evidence(
        nodes,
        state.get("retrieved_relationships") or [],
        profile if profile is not None else state.get("educational_profile"),
        state.get("recommendations") or [],
        request=str(state.get("raw_user_turn") or state.get("teacher_query") or ""),
    )
    if not text:
        return None
    lesson = lesson or str(state.get("final_lesson_plan") or state.get("lesson_plan_draft") or "")
    found = (
        usage(
            nodes,
            state.get("retrieved_relationships") or [],
            profile if profile is not None else state.get("educational_profile"),
            lesson,
            request=str(state.get("raw_user_turn") or state.get("teacher_query") or ""),
        )
        if lesson
        else {}
    )
    return {
        "version": 2,
        "usage": found,
        "usage_text": render_usage(found),
        "nodes": len(nodes),
        "relationships": len(state.get("retrieved_relationships") or []),
        "recommendations": len(state.get("recommendations") or []),
        "text": text,
    }


def environment_lookup_enabled() -> bool:
    """``AIX_KG_ENVIRONMENT_ENABLED``: read the graph's classroom nodes when needed. On by default."""
    raw = os.getenv("AIX_KG_ENVIRONMENT_ENABLED")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def environment_needed(profile: Any, request: str = "") -> bool:
    """Whether the class's needs call for advice on the learning environment.

    ADHD and the autism spectrum (attention, sensory load), and a class the
    teacher describes as having disruptive elements. The UDL expert's point on
    2026-10-16: for a pupil with ADHD the room must not be overloaded with
    stimuli -- neutral, orderly -- and neither our lesson nor the competitor's
    said a word about it.
    """
    group = _profile_group(profile)
    raw = group.get("disabilities")
    if not raw and isinstance(profile, dict):
        raw = profile.get("disabilities")
    needs = set(_codes(raw))
    features = set(_codes(group.get("class_features")))
    named = class_need_groups(None, request) & {"ADHD", "Spettro autistico"}
    return bool(needs & {"ADHD", "SA", "DOP"} or named or "ELEMENTI_DI_DISTURBO" in features)
