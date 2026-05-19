# Teacher UX — Pedagogical Framework + 6 Features Plan

## Context

The platform already collects a comprehensive educational profile before generation —
more context than any competitor. The remaining gap: the AI knows WHO is in the room
but not WHY the teacher is there, and the teacher has no visibility into the AI's
reasoning or the pedagogical principles behind it.

**Strategic backbone: UbD + UDL 3.0 + ARCHED**

The platform's 5 natural phases each map to a specific framework:

| Phase | Framework | What happens | Technical anchor |
|---|---|---|---|
| 1 — Discover | Design Thinking | Class profile: grade, subject, BES, time. KG surfaces UDL barriers | Existing form + profile sidebar |
| 2 — Define | Backward Design | "What should students be able to do?" before generating anything | **NEW: pedagogical_intent field + planner objectives step** |
| 3 — Generate | ARCHED | Multi-stage visible pipeline: each agent card is a conversation stage | **ENHANCE: label SSE cards with phase names; planner exposes objectives for approval** |
| 4 — Refine | SAM (iterative) | Multi-turn loop already exists; frame explicitly as "prototype → feedback → improve" | **ENHANCE: rename/frame Rigenera as SAM refinement cycle** |
| 5 — Reflect | UbD + UDL 3.0 | After lesson: did students achieve the objective? UDL coverage badge | **NEW: KG "What's Next?" + UDL coverage badge + reflection prompt** |

---

## Feature 1 — Pedagogical Intent Field *(Backward Design — Phase 2)*

### Purpose
Upgrades the planner from "find content about X" to "find content that achieves Y via X".
The teacher's explicit intent anchors every downstream agent node.

### UI design: guided chips + optional freetext

The field uses **pre-built option cards** (single-select, like the existing toggle-cards)
plus an optional "Aggiungi dettagli" freetext expander. Guided selection is the default;
freetext is available but not required.

```
Obiettivo pedagogico (opzionale)
────────────────────────────────
[○ Prima introduzione]  [○ Approfondimento]  [○ Correggere un'idea errata]
[○ Preparazione verifica]  [○ Applicazione pratica]  [○ Collegamento interdisciplinare]

  ▼ Aggiungi dettagli specifici (opzionale)
  [                                                              ] textarea
```

**Predefined options (stored as code + label + prompt instruction):**

| Code | Label | Injected into prompt |
|---|---|---|
| `intro` | Prima introduzione | "Introduce the concept for the first time — assume no prior knowledge" |
| `deepening` | Approfondimento | "Deepen understanding of a concept the class has already encountered" |
| `misconception` | Correggere un'idea errata | "Challenge and correct a common misconception about this topic" |
| `assessment` | Preparazione alla verifica | "Consolidate knowledge and prepare students for an upcoming assessment" |
| `practice` | Applicazione pratica | "Focus on practical exercises, activities, and concrete application" |
| `interdisciplinary` | Collegamento interdisciplinare | "Build bridges between this concept and other subject areas" |

**Stored value:** `pedagogical_intent = "{code}: {optional_freetext}"` (e.g. `"misconception: practice alone doesn't increase WM capacity"`). The code alone is valid if no freetext is added.

### Data flow
```
lesson_new.html (chip selector + optional textarea)
  → POST: pedagogical_intent_code + pedagogical_intent_detail
  → form_to_profile_dict() → merged into pedagogical_intent string
  → EducationalProfile.pedagogical_intent (Pydantic field)
  → Lesson.educational_profile_json["pedagogical_intent"]
  → service.py → create_initial_state(pedagogical_intent=...)
  → AgentState.pedagogical_intent
  → planner_agent: informs objective generation
  → writer_agent: injected as "Obiettivo pedagogico: {label} — {detail}"
  → lesson card: shows intent pill with the label (not the raw code)
```

### Files to change

| File | Change |
|---|---|
| `src/aix/api/schemas/educational_profile.py` | Add `pedagogical_intent: Optional[str] = None` to `EducationalProfile` |
| `src/aix/webui/lessons/routes.py` | Add `PEDAGOGICAL_INTENT_LABELS` dict (code → label + prompt instruction); pass to template context in `lesson_new_get()` |
| `src/aix/webui/templates/pages/lesson_new.html` | Add chip-selector section in Section 1 after `time_available_minutes`. Use radio-card pattern (same as toggle-cards). Alpine.js `x-show` to reveal freetext when a chip is selected |
| `src/aix/webui/lessons/schemas.py` | In `form_to_profile_dict()`: merge `pedagogical_intent_code` + `pedagogical_intent_detail` into `pedagogical_intent` string |
| `src/aix/agent/graph/state.py` | Add `pedagogical_intent: Optional[str]` to `AgentState`; add to `create_initial_state()` |
| `src/aix/webui/agent/service.py` | Extract from `profile_dict`; pass to `create_initial_state()` |
| `src/aix/agent/graph/nodes.py` | Pass `pedagogical_intent` from state to `writer.write()` and `planner.plan()` |
| `src/aix/agent/agents/writer_agent.py` | Inject into `edu_profile_section`; resolve code to full prompt instruction |
| `src/aix/webui/templates/partials/chat_lesson_card.html` | Show intent pill with human-readable label in metadata strip |

---

## Feature 2 — Intent Display in Profile Sidebar *(Design Thinking — Phase 1 → 2 bridge)*

The existing `profile_sidebar.html` already shows class + classroom context.
Add "Intento pedagogico" at the top as the missing phase-2 bridge.

### Files to change

| File | Change |
|---|---|
| `src/aix/webui/templates/partials/profile_sidebar.html` | Add "Intento pedagogico" section before the "Lezione" block. If set: teal callout with intent text. If not set: muted "Nessun intento definito — [aggiungi →]" linking to the inline edit form |
| `src/aix/webui/templates/partials/profile_sidebar_edit.html` | Add `pedagogical_intent` chip-selector + freetext to the inline edit form (teachers can update intent without leaving the workspace) |
| `src/aix/webui/lessons/routes.py` | Confirm `POST /webui/lesson/{id}/profile` parses and saves `pedagogical_intent` back into `educational_profile_json` |

---

## Feature 3 — ARCHED Phase Labels on SSE Cards *(ARCHED — Phase 3)*

### Purpose
The SSE pipeline (planner → retriever → writer → critic) already has distinct cards
but they currently show only internal labels. Upgrading them with ARCHED-aligned phase
labels makes the AI's reasoning transparent and turns the progress stream into a
pedagogical conversation the teacher can follow.

### Proposed labels
| Existing card | New label | ARCHED phase |
|---|---|---|
| Planner card | "Progettazione: obiettivi + concetti chiave" | Situation Analysis |
| Retriever card | "Ricerca nella Knowledge Graph — N concetti trovati" | Knowledge Retrieval |
| Writer card | "Scrittura della lezione" | Content Generation |
| Critic card | "Revisione pedagogica" | Quality Assurance |

### Files to change

| File | Change |
|---|---|
| SSE card partials (`chat_*card*.html`) | Update card header labels + add one-line phase description. Confirm exact filenames by globbing `templates/partials/chat_*card*.html` at implementation time |

---

## Feature 3b — Guided Refinement Loop *(SAM — Phase 4)*

### Purpose
Replace the unstructured "Rigenera" black-box button with a **guided SAM refinement
panel** that captures the teacher's feedback before regenerating. The teacher's
selection becomes additional context passed to the writer, making each iteration
explicitly directed rather than random.

### UI flow
When the teacher clicks "Raffina" (rename of current "Rigenera"):
1. A lightweight inline panel expands **below the lesson footer** (no modal, no page
   change — same pattern as Alpine.js x-show on the form):

```
Come vorresti migliorare questa lezione?
────────────────────────────────────────
[○ Semplifica]        Riduci complessità e lunghezza
[○ Approfondisci]     Aumenta dettaglio e rigore scientifico
[○ Più attività]      Aggiungi esercizi pratici e interattivi
[○ Adatta alla classe] Ricalibra su profilo classe (BES, livello, tempo)
[○ Personalizza →]    [freetext: specifica cosa cambiare...]

[Annulla]  [Raffina lezione →]
```

2. Teacher selects one option (or writes freetext) and confirms.
3. The selection is posted alongside the regeneration trigger.
4. Writer receives: original lesson + `refinement_instruction` (from code → prompt
   instruction mapping) + optional freetext. It improves the lesson rather than
   regenerating from scratch.
5. The refinement choice is stored in `LessonMessage.meta_json["refinement"]` for
   conversation history.

**Refinement option → prompt instruction mapping:**

| Code | Label | Injected instruction |
|---|---|---|
| `simplify` | Semplifica | "Reduce length by ~30%, simplify vocabulary, keep only essential concepts" |
| `deepen` | Approfondisci | "Add scientific depth, more precise terminology, nuanced examples" |
| `more_activities` | Più attività | "Replace passive sections with at least 2 interactive activities or exercises" |
| `adapt_class` | Adatta alla classe | "Re-calibrate to the class profile: BES accommodations, time constraint, available tech" |
| `custom` | Personalizza | Use freetext verbatim as instruction |

### Files to change

| File | Change |
|---|---|
| `src/aix/webui/templates/partials/chat_lesson_card.html` | Replace "Rigenera" button with "Raffina" + Alpine.js `x-show` refinement panel. Panel contains radio-card options + optional freetext + confirm button |
| `src/aix/webui/lessons/routes.py` | `POST /webui/lesson/{id}/run` — accept optional `refinement_code` + `refinement_detail` form fields alongside existing run trigger |
| `src/aix/webui/agent/service.py` | In `run_agent_stream()`: if `refinement_code` present, build `refinement_instruction` string from mapping + freetext; pass as additional context to `create_initial_state()` |
| `src/aix/agent/graph/state.py` | Add `refinement_instruction: Optional[str]` to `AgentState` |
| `src/aix/agent/agents/writer_agent.py` | When `refinement_instruction` present, prepend to writer user prompt: "Improvement requested: {instruction}" + include the previous lesson as reference |

---

## Feature 4 — Backward Design Objectives in Planner *(Backward Design — Phase 2 deep)*

### Purpose
If the teacher provided `pedagogical_intent`, the planner should explicitly generate
**learning objectives** (what students will know/do/understand by the end) before
searching the KG. These objectives are shown in the planner card for teacher review —
making the design-backward step visible, not hidden in the prompt.

### Change to planner agent
In `src/aix/agent/agents/planner_agent.py`:
- When `pedagogical_intent` is present in state, prepend to planner system/user prompt:
  `"Teacher's learning objective: {intent}. Generate lesson objectives that fulfill this intent, then map them to KG concepts."`
- Planner output should include an `objectives` list (e.g., `["Students can explain...", "Students can apply..."]`)
- Store in `AgentState.plan.objectives` (or add `objectives: Optional[List[str]]` to plan schema)

In the planner card template:
- Render objectives as a visible bullet list: *"Traguardi di apprendimento: …"*

### Files to change

| File | Change |
|---|---|
| `src/aix/agent/agents/planner_agent.py` | Inject `pedagogical_intent` into planner prompt; instruct planner to output explicit learning objectives |
| `src/aix/agent/graph/state.py` | Confirm planner output schema includes `objectives: Optional[List[str]]` or add it |
| Planner card template | Render `plan.objectives` as bullet list when present |

---

## Feature 5 — UDL 3.0 Coverage Badge *(UDL 3.0 — Phase 5)*

### Purpose
UDL is one of the two KG domains. The retrieved nodes already contain UDL-tagged
concepts. Compute which UDL principles (Representation / Action+Expression /
Engagement) the lesson covers and show a coverage badge in the lesson card.

### Logic
At lesson completion, in `_extract_meta()` (service.py):
1. Count retrieved nodes tagged with UDL labels (available in node `labels` list)
2. Bucket them into the 3 UDL principles
3. Store: `meta_json["udl_coverage"] = {"representation": N, "action": N, "engagement": N, "total": N}`

In `chat_lesson_card.html`:
- Add a UDL badge to the metadata strip: `"UDL: 7 principi"` (or breakdown by principle)
- Only shown when `lesson.domain` includes UDL content (domain == "udl" or == "all")

### Files to change

| File | Change |
|---|---|
| `src/aix/webui/agent/service.py` | In `_extract_meta()`: compute `udl_coverage` from retrieved nodes; add to returned meta dict |
| `src/aix/webui/templates/partials/chat_lesson_card.html` | Add UDL badge in metadata strip when `meta.udl_coverage` present |

---

## Feature 6 — KG "What's Next?" Panel *(UbD Transfer — Phase 5)*

### Purpose
After every completed lesson, show 3–5 adjacent KG concepts with bridging sentences.
Turns single-lesson use into curriculum navigation. Pre-fills the new lesson form
when clicked (carries forward domain and intent context).

### Data flow
```
lesson_show() route (when status == "complete")
  → read lesson.educational_profile_json["specific_topic"]
  → call get_concept_neighbors(concept, domain, limit=5)  ← new public wrapper
  → pass adjacent_concepts list to template context

chat_lesson_card.html (after lesson body, before footer)
  → {% include "partials/what_next_panel.html" %}

what_next_panel.html
  → 3–5 cards: concept name + rel_type + description
  → each links to /webui/lesson/new?topic={name}&domain={domain}
     carrying pedagogical context forward
```

### Files to change

| File | Change |
|---|---|
| `src/aix/retrieval/graph_retriever.py` | Add `get_concept_neighbors(concept_name, domain, limit=5) -> list[dict]` wrapping `_get_educational_neighbors()` with session management |
| `src/aix/webui/lessons/routes.py` | In `lesson_show()`: when `status == "complete"`, derive concept from profile, call `get_concept_neighbors()`, pass to context |
| `src/aix/webui/lessons/routes.py` | In `lesson_new_get()`: read `?topic=` and `?domain=` query params; pass as defaults into `fv` dict for form pre-fill |
| `src/aix/webui/templates/partials/chat_lesson_card.html` | After lesson body (line ~111), before footer: include `what_next_panel.html` |
| `src/aix/webui/templates/partials/what_next_panel.html` | **NEW** — "Cosa esplorare dopo" panel with concept cards |

---

## Implementation Order

| Step | Feature | Effort | Why |
|---|---|---|---|
| 1 | F1: Intent chip selector + freetext | Low-Med | Foundation — every downstream feature uses pedagogical_intent |
| 2 | F2: Sidebar intent display | Very low | Zero backend after F1; immediate teacher visibility |
| 3 | F3: ARCHED card labels | Very low | Template-only; pipeline transparency immediately |
| 4 | F3b: Guided refinement panel | Low-Med | High UX impact; replaces blind regenerate with SAM loop |
| 5 | F6: KG "What's Next?" panel | Low | Uses existing graph; no agent changes |
| 6 | F4: Planner objectives | Medium | Needs planner prompt tuning; deepens Backward Design |
| 7 | F5: UDL badge | Medium | Needs meta extraction logic; natural finish to the framework layer |

---

## Verification

### Feature 1 — Intent Chip Selector
1. Create lesson with chip selected (e.g. "misconception") + freetext → `educational_profile_json["pedagogical_intent"]` = `"misconception: practice alone doesn't increase WM capacity"`
2. Create lesson with chip only (no freetext) → stored as `"intro"` (code only)
3. Create lesson with no chip → `pedagogical_intent` absent, generation unchanged (backward compatible)
4. Writer prompt in logs shows resolved label: "Obiettivo pedagogico: Correggere un'idea errata — practice alone doesn't increase WM capacity"
5. Lesson card shows intent pill with human-readable label

### Feature 2 — Sidebar
1. Lesson with intent → teal callout visible in sidebar
2. Lesson without intent → muted "aggiungi" prompt visible
3. Inline edit updates intent without page reload

### Feature 3 — ARCHED Labels
1. Trigger a lesson generation → each streaming card shows its phase label
2. Planner card shows concept count; retriever card shows KG node count

### Feature 3b — Guided Refinement
1. Complete a lesson → "Raffina" button visible in footer (not "Rigenera")
2. Click "Raffina" → refinement panel expands inline (no page change)
3. Select "Semplifica" → confirm → lesson regenerates with simplification instruction
4. Select "Personalizza" → freetext appears → confirm → custom instruction used
5. Check `LessonMessage.meta_json` → `refinement` key present with chosen option
6. No option selected (Annulla) → panel collapses, nothing triggered

### Feature 4 — Planner Objectives
1. Lesson with intent → planner card shows "Traguardi di apprendimento" bullet list
2. Lesson without intent → planner card unchanged (backward compatible)

### Feature 5 — UDL Badge
1. Complete a lesson in "udl" or "all" domain → badge shows in metadata strip
2. Lesson in "neuro" only domain → no UDL badge (graceful)

### Feature 6 — "What's Next?" Panel
1. Complete any lesson → panel shows 3–5 adjacent concepts
2. Click concept card → `/webui/lesson/new` opens with topic pre-filled
3. No KG neighbors found → panel hidden, no error
4. Draft/running lesson → panel absent
