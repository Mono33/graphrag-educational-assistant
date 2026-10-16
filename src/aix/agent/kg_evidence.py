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
"""

from __future__ import annotations

import re
from typing import Any, Optional

#: KG learner-variability labels, with the name the teacher knows them by.
NEED_GROUPS: dict[str, str] = {
    "Adhd": "ADHD",
    "ForeignStudents": "Alunni non italofoni",
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
MAX_CHARS = 7000
_MAX_STRATEGIES = 24
_DESC_CHARS = 160


def _labels(node: dict[str, Any]) -> list[str]:
    raw = node.get("labels") or []
    return [str(label) for label in (raw if isinstance(raw, (list, tuple)) else [raw])]


def _name(node: dict[str, Any]) -> str:
    return " ".join(str(node.get("name") or node.get("title") or "").split())


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def kind(node: dict[str, Any]) -> str:
    """``strategy``, ``tool``, ``need``, ``environment``, ``udl``, ``context`` or ``other``."""
    labels = _labels(node)
    if CONTEXT_LABEL in labels:
        return "context"
    if any(label in NEED_GROUPS for label in labels):
        return "need"
    if any(label in ENVIRONMENT_LABELS for label in labels):
        return "environment"
    if any(label in UDL_LABELS for label in labels):
        return "udl"
    if any(_TOOL.search(label) for label in labels):
        return "tool"
    if any(_STRATEGY.search(label) for label in labels):
        return "strategy"
    return "other"


_TIER = {
    "strategy": 0,
    "udl": 1,
    "environment": 1,
    "need": 2,
    "tool": 3,
    "other": 4,
    "context": 5,
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
    """Most relevant first; a recommendation that is a class-profile node goes last.

    «Apply Group till 15 with appropriate adaptations» is not a strategy, and on
    2026-10-16 four of the Writer's ten «Teaching Methodologies» were that.
    """
    context = {_name(n) for n in nodes or [] if isinstance(n, dict) and kind(n) == "context"}
    items = [r for r in recommendations or [] if isinstance(r, dict)]

    def _key(rec: dict[str, Any]) -> tuple[int, float]:
        is_context = _name(rec) in context or "contesto" in _norm(rec.get("category") or "")
        return (1 if is_context else 0, -_score(rec, "relevance_score", "score"))

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


def _short(text: Any, limit: int = _DESC_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


#: What the context builder writes when it has nothing to say about a node.
_BOILERPLATE = re.compile(r"^Apply .+ with appropriate adaptations$", re.IGNORECASE)


def _guidance(recommendations: Optional[list[dict[str, Any]]]) -> dict[str, str]:
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


def render_evidence(
    nodes: Optional[list[dict[str, Any]]],
    relationships: Optional[list[dict[str, Any]]] = None,
    profile: Any = None,
    recommendations: Optional[list[dict[str, Any]]] = None,
    *,
    max_chars: int = MAX_CHARS,
) -> str:
    """The retrieved graph as the lesson can use it, in Italian, bounded.

    Empty when nothing was retrieved.
    """
    ranked = rank_nodes(nodes)
    if not ranked:
        return ""
    by_name: dict[str, dict[str, Any]] = {}
    for node in ranked:
        by_name.setdefault(_name(node), node)
    edges = _edges(relationships)
    guidance = _guidance(recommendations)
    advised_against: set[str] = set()
    sections: list[str] = []

    # 1. The class's needs -> what the graph suggests and advises against.
    wanted = class_need_labels(profile)
    need_lines: list[str] = []
    for node in ranked:
        if kind(node) != "need":
            continue
        group = next(label for label in _labels(node) if label in NEED_GROUPS)
        if wanted and group not in wanted:
            continue
        suggests, avoid = _links(_name(node), edges, by_name)
        if not suggests and not avoid:
            continue
        advised_against.update(avoid)
        line = f"- **{NEED_GROUPS[group]} · {_name(node)}**"
        if suggests:
            line += f" → il KG suggerisce: {', '.join(suggests)}"
        if avoid:
            line += f"; sconsiglia: {', '.join(avoid)}"
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
        line = f"- **{_name(node)}** (profilo della classe)"
        if suggests:
            line += f" → suggerisce: {', '.join(suggests)}"
        if avoid:
            line += f"; sconsiglia: {', '.join(avoid)}"
        context_lines.append(line)
    if context_lines:
        sections.append("### Profilo della classe nel Knowledge Graph\n" + "\n".join(context_lines))

    # 3. The learning environment, when it was retrieved.
    env_lines = []
    for node in ranked:
        if kind(node) != "environment":
            continue
        label = next(label for label in _labels(node) if label in ENVIRONMENT_LABELS)
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
    for node in ranked:
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

    text = ""
    for section in sections:
        if len(text) + len(section) + 2 > max_chars:
            break
        text += ("\n\n" if text else "") + section
    return text


def digest(state: dict[str, Any], profile: Any = None) -> Optional[dict[str, Any]]:
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
    )
    if not text:
        return None
    return {
        "version": 1,
        "nodes": len(nodes),
        "relationships": len(state.get("retrieved_relationships") or []),
        "recommendations": len(state.get("recommendations") or []),
        "text": text,
    }


def environment_needed(profile: Any) -> bool:
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
    return bool(needs & {"ADHD", "SA", "DOP"} or "ELEMENTI_DI_DISTURBO" in features)
