# Next Session — Remaining Issues

**Date written:** 10 April 2026  
**Context:** After fixing `total_relationships: 0` in the API (CASE 4 relationship extraction + `re.DOTALL` bug in `_parse_return_aliases`).

---

## Issue 1 — Challenge nodes leaking into methodology recommendations

**Symptom:** Nodes like `"Difficulty sustaining focus"`, `"Increased cognitive load during reading tasks"`, `"Impaired inhibitory control"`, `"Rapid learning pace"` appear in `primary_methodologies` and `supporting_methodologies` in the API response.

**Root cause:** These are student characteristic / symptom nodes stored in the KG alongside strategy nodes. The CASE 4 Cypher returns both source (challenge) and target (strategy) nodes in each row. The `MethodologyRanker` in `context_builder.py` receives all retrieved nodes and does not filter by label — so challenge nodes labelled `Gifted`, `Adhd`, `Dyslexia`, etc. pass through as if they were teaching strategies.

**Where to fix:** `context_builder.py` — `_is_methodology()` method (already has a label filter). Extend it to explicitly reject nodes whose labels are challenge/characteristic types (e.g. `Gifted`, `Adhd`, `Dyslexia`, `Autism`, `Dyscalculia`, `ForeignStudent`) unless they also carry a methodology label. Cross-reference with the UDL domain config's `get_valid_methodology_labels()`.

---

## Issue 2 — "Long frontal reading lessons" passes the junk filter

**Symptom:** `"Long frontal reading lessons"` appears in `supporting_methodologies`. It is a negative-approach node (something teachers should NOT do) that leaks through.

**Root cause:** The junk filter in `_is_methodology()` (`context_builder.py`) checks for the exact strings `"Long Frontal Lesson"` and `"Long frontal reading lessons"`, but the actual node name in the KG is a slightly different variant.

**Where to fix:** `context_builder.py` — `_NEGATIVE_APPROACH_NAMES` frozenset. Either add the exact variant string from the KG, or switch to a substring/case-insensitive check (e.g. `"frontal"` + `"lesson"` in name).

Additionally, nodes connected via `NO_SUGGESTS` relationships should never appear in recommendations regardless of name. Consider filtering by checking `node.get('rel_type') == 'NO_SUGGESTS'` in `_is_methodology()`.

---

## Issue 3 — `educational_context_type` always resolves to `"assessment"`

**Symptom:** `"educational_context_type": "assessment"` for a query that is clearly a teaching scenario (2-hour philosophy lesson with Plato's Cave).

**Root cause:** The `_build_student_profile()` method in `context_builder.py` infers `educational_context` from keywords in the query. The keyword matching for `"assessment"` is triggering on unrelated content (possibly `"obiettivi"` / `"objectives"` or similar).

**Where to fix:** `context_builder.py` — `_build_student_profile()` or the `_detect_educational_context()` helper. Review the keyword list for `"assessment"` and tighten it. The UDL use case is almost always a teaching/lesson-planning context, not assessment.

---

## Issue 4 — All `relevance_score: 0.5` (no differentiation)

**Symptom:** Every node in the API response has `relevance_score: 0.5`. There is no ranking signal to distinguish more relevant strategies from less relevant ones.

**Root cause:** CASE 4 nodes are built with empty labels or generic labels, so `_calculate_relevance_score()` in `context_builder.py` cannot apply domain boosts or label-based scoring. All nodes fall through to the default score of 0.5.

**Where to fix:** Two-pronged:
1. Ensure CASE 4 nodes carry correct Neo4j labels (the `col_to_label` mapping in `graph_retriever.py` CASE 4 must produce non-empty labels — verify `alias_labels` is populated for the UDL Cypher).
2. In `context_builder.py` `_calculate_relevance_score()`, verify that label-based boosts and query-keyword matching logic are correctly wired for UDL domain nodes.

---

## Issue 5 — Generic `implementation_guidance` and `classroom_applications`

**Symptom:** Many nodes show `"Apply X with appropriate adaptations"` and `"Consultare il contesto specifico della classe..."` — copy-paste placeholder text.

**Root cause:** These nodes have no `description` or `category` field populated in the KG (they are bare name-only nodes), so `_create_recommendation()` in `context_builder.py` falls back to generic template strings.

**Where to fix:** Either:
- Enrich the KG data for these nodes (add `description`, `classroom_applications`, etc. to the Neo4j node properties), OR
- In `context_builder.py` `_create_recommendation()`, use the node's KG relationships (via `triples`) to infer implementation guidance dynamically — e.g. if node X `SUGGESTS` node Y, include Y's name as a concrete application of X.

---

## Summary table

| # | Issue | File | Method |
|---|---|---|---|
| 1 | Challenge nodes in methodology list | `context_builder.py` | `_is_methodology()` |
| 2 | Negative-approach nodes pass filter | `context_builder.py` | `_NEGATIVE_APPROACH_NAMES`, `_is_methodology()` |
| 3 | Wrong `educational_context_type` | `context_builder.py` | `_build_student_profile()` / `_detect_educational_context()` |
| 4 | All scores = 0.5, no ranking | `context_builder.py`, `graph_retriever.py` | `_calculate_relevance_score()`, CASE 4 label assignment |
| 5 | Generic implementation guidance | `context_builder.py` + KG data | `_create_recommendation()` |
