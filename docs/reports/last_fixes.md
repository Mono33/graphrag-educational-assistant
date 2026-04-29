# Last Fixes — WebUI Lesson Workspace (2026-04-28)

Five improvements applied to `src/aix/webui/` on branch `chore/repo-reorg`.

---

## Fix 1 — "Still generating" spinner glitch

**Problem:** The spinner in `chat_pane.html` had no `id` and remained visible after the agent finished, even after the final lesson card appeared.

**Root cause:** The spinner `<div>` is a sibling of `#chat-cards`. htmx-SSE appends the final card and closes the EventSource but has no directive to remove the spinner.

**Solution:** OOB swap — same pattern used by the media panel.
- `src/aix/webui/templates/partials/chat_pane.html` — added `id="chat-spinner"` to the spinner div.
- `src/aix/webui/lessons/routes.py` — `_stream_event_to_sse()` now appends `<div id="chat-spinner" hx-swap-oob="outerHTML"></div>` to both `done` and `error` SSE payloads, replacing the spinner out-of-band when the run ends.

---

## Fix 2 — Download lesson (MD, TXT, PDF)

**New routes:**
- `GET /webui/lesson/{id}/export?format=md` — downloads the raw Markdown.
- `GET /webui/lesson/{id}/export?format=txt` — strips markdown symbols, downloads plain text.
- `GET /webui/lesson/{id}/print` — opens a print-friendly HTML page; user saves as PDF via browser Print dialog (no server-side binary dependency).

**Files changed:**
- `src/aix/webui/lessons/routes.py` — added `lesson_export` and `lesson_print` route handlers.
- `src/aix/webui/templates/pages/lesson_print.html` — new standalone print template with `@media print` CSS and `window.print()` auto-trigger.
- `src/aix/webui/templates/partials/chat_lesson_card.html` — download buttons (MD / TXT / PDF) added to the lesson card footer.

---

## Fix 3 — Sidebar scrolling

**Problem:** Both sidebars use `sticky top-4` but had no height cap, causing overflow on short viewports (user had to scroll the whole page to reach the bottom of the educational profile or resources panel).

**Solution:** Added `overflow-y-auto max-h-[calc(100vh-5rem)]` to the `<aside>` element in:
- `src/aix/webui/templates/partials/profile_sidebar.html`
- `src/aix/webui/templates/partials/media_panel.html`
- `src/aix/webui/templates/partials/profile_sidebar_edit.html`

Each sidebar now scrolls independently within the viewport.

---

## Fix 4 — Lessons history ("Le mie lezioni")

**Problem:** Lessons were persisted on completion but there was no way to revisit them — no list page and no navbar entry.

**Solution:**
- `src/aix/webui/lessons/routes.py` — new `GET /webui/lessons` route; queries all lessons for the current user ordered by `created_at DESC`.
- `src/aix/webui/templates/pages/lesson_list.html` — new card-grid page: status colour strip, domain tag, subject/topic, class name, creation date. Empty-state CTA if no lessons.
- `src/aix/webui/templates/partials/navbar.html` — "Le mie lezioni" button added before "Nuova lezione" (both in the top bar and in the user dropdown).
- `src/aix/webui/templates/partials/profile_sidebar_edit.html` — `lesson_title` text input added as the first field so teachers can name their lessons.
- `src/aix/webui/lessons/routes.py` `lesson_profile_save()` — now persists `lesson.title` from the form (no migration needed; column already existed).

---

## Fix 5 — Change domain in profile editor

**Problem:** `lesson.domain` was set at lesson creation and was not editable from the inline profile editor.

**Solution:**
- `src/aix/webui/templates/partials/profile_sidebar_edit.html` — added `<wa-select name="domain">` with options `neuro / udl / all` inside the Lezione fieldset, pre-filled with `lesson.domain`.
- `src/aix/webui/lessons/routes.py` `lesson_profile_save()` — extracts `domain` from form data, validates it against `{"neuro", "udl", "all"}`, and updates `lesson.domain`. Unknown values are silently ignored, keeping the existing domain. No DB migration needed.

---

## Files changed summary

| File | Fix(es) |
|------|---------|
| `src/aix/webui/templates/partials/chat_pane.html` | 1 |
| `src/aix/webui/lessons/routes.py` | 1, 2, 4, 5 |
| `src/aix/webui/templates/partials/chat_lesson_card.html` | 2 |
| `src/aix/webui/templates/pages/lesson_print.html` *(new)* | 2 |
| `src/aix/webui/templates/partials/profile_sidebar.html` | 3 |
| `src/aix/webui/templates/partials/media_panel.html` | 3 |
| `src/aix/webui/templates/partials/profile_sidebar_edit.html` | 3, 4, 5 |
| `src/aix/webui/templates/partials/navbar.html` | 4 |
| `src/aix/webui/templates/pages/lesson_list.html` *(new)* | 4 |

---

# Follow-up Fixes — SSE Truncation, Profile Retrieval, Delete, Langfuse (2026-04-29)

Additional improvements on branch `chore/repo-reorg`.

---

## Fix 6 — Spinner OOB approach replaced (SSE truncation + spinner glitch)

**Problem:** The OOB spinner removal (Fix 1) used `hx-swap-oob` inside SSE payloads — unsupported by htmx-ext-sse 2.2.4. Additionally, `_render_partial()` collapses lesson card HTML to a single line (~22-26 KB), which htmx-ext-sse silently truncates.

**Solution:**
- SSE `done` event now sends a <1 KB loading placeholder (`id="lesson-card-loading"`) instead of the full card.
- A `<script>` in `chat_pane.html` listens for `htmx:sseClose`, removes the spinner, then fetches the full card via `GET /webui/lesson/{id}/card-fragment` (regular HTTP, no SSE size limit).
- New route `GET /webui/lesson/{id}/card-fragment` renders `chat_lesson_card.html` as a standalone fragment.

**Files changed:**
- `src/aix/webui/templates/partials/chat_pane.html` — JS `htmx:sseClose` handler + placeholder logic
- `src/aix/webui/lessons/routes.py` — removed `_spinner_oob`; SSE `done` → placeholder; new `lesson_card_fragment` route

---

## Fix 7 — Profile-enriched GraphRAG retrieval

**Problem:** Retriever returned 0 KG nodes for generic queries like "Crea una lezione efficace" because the planner only received `teacher_query`, ignoring the educational profile.

**Solution:** `retrieve_node()` in `nodes.py` augments `plan.search_queries` with `specific_topic`, `subject_area`, and `disabilities` from `state["educational_profile"]` before calling `retriever.retrieve(plan)`. Log confirmation: `[Node: Retrieve] Profile enrichment added N terms`.

**Files changed:**
- `src/aix/agent/graph/nodes.py` — profile enrichment in `retrieve_node()`

---

## Fix 8 — Delete lesson from "Le mie lezioni"

**Problem:** No way to delete a lesson from the lesson history page.

**Solution:**
- New `DELETE /webui/lesson/{id}` route that deletes the lesson (auth + ownership check) and returns `204 + HX-Redirect: /webui/lessons`.
- Each lesson card on `lesson_list.html` wrapped in `<div class="relative group/card">` with a hover-reveal trash button outside the `<a>` link (no Alpine needed; `hx-target="closest .relative"` removes the whole card wrapper on success).

**Files changed:**
- `src/aix/webui/lessons/routes.py` — `lesson_delete` route
- `src/aix/webui/templates/pages/lesson_list.html` — hover-reveal delete button

---

## TODO — Langfuse prompt integration (pending `.env` setup)

**Goal:** Move `system_prompt`, `writer_prompt`, `response_template` from Python domain files to Langfuse. Changes in Langfuse UI take effect within **60 seconds** — no server restart needed.

**All code is already written.** Only the setup steps below remain.

### Checklist

- [x] **1. Install dependency** — done
- [x] **2. Add Langfuse credentials to `.env`** — done (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL`)
- [x] **3. Upload the 6 prompts to Langfuse** — done (all 6 visible in Langfuse UI, version 1, 2026-04-29 17:29:31)
- [x] **4. Verify in Langfuse UI** — done (all 6 prompts confirmed: `neuro.*` + `udl.*`)
- [x] **5. Restart the API server** — done
- [x] **6. Smoke-test agent mode** — done (2026-04-29 18:00:26); logs confirmed `[WriterAgent] Applied domain extension for 'neuro'`, no `Langfuse prompt unavailable` errors; full pipeline: 33 nodes → 11,567-char lesson → critic approved
- [x] **7. Smoke-test prompt editing** — done (2026-04-29); removed `RISPONDI SEMPRE IN ITALIANO` from both neuro prompts in Langfuse UI; before/after JSON comparison confirmed change propagated within 60 s

### What was already done (code in repo)

| File | Change |
|---|---|
| `requirements.txt` | Added `langfuse>=2.0.0` |
| `src/aix/domains/langfuse_prompts.py` *(new)* | Langfuse client singleton + `fetch_prompt(name, ttl=60s)` |
| `src/aix/domains/base_config.py` | Added abstract `get_writer_prompt()` |
| `src/aix/domains/neuro_domain.py` | All 3 prompt methods now fetch from Langfuse |
| `src/aix/domains/udl_domain.py` | Same pattern |
| `src/aix/agent/configs/domain_prompts.py` | Agent writer calls `cfg.get_writer_prompt()` (separate from legacy `system_prompt`) |
| `scripts/ops/seed_langfuse_prompts.py` *(new)* | One-time upload script — contains full prompt text |
| `docs/prompts/langfuse_prompts_reference.md` *(new)* | Full prompt texts, names, and where each is called |

### Prompt ownership per mode

| Langfuse name | Who calls it | When |
|---|---|---|
| `neuro.system_prompt` | `llm_chain.py` | Legacy GraphRAG mode |
| `neuro.writer_prompt` | `domain_prompts.py` | Agent mode (appended to writer) |
| `neuro.response_template` | `llm_chain.py` | Legacy GraphRAG mode |
| `udl.system_prompt` | `llm_chain.py` | Legacy GraphRAG mode |
| `udl.writer_prompt` | `domain_prompts.py` | Agent mode |
| `udl.response_template` | `llm_chain.py` | Legacy GraphRAG mode |

`writer_prompt` and `system_prompt` start with identical text — they diverge independently once edited in Langfuse.

---

## How to add a new domain with Langfuse prompts

Follow this pattern whenever a new domain (e.g. `stem`, `math`, `sel`) is created.

### 1. Naming convention

Use `{domain_key}.{prompt_type}` — lowercase, dot-separated, no spaces:

| Langfuse name | Purpose |
|---|---|
| `{domain}.system_prompt` | Full standalone system prompt — legacy GraphRAG mode (`llm_chain.py`) |
| `{domain}.writer_prompt` | Domain expertise block — agent mode, appended to base writer prompt |
| `{domain}.response_template` | Response formatting instructions — legacy GraphRAG mode |

Example for a future `stem` domain:
- `stem.system_prompt`
- `stem.writer_prompt`
- `stem.response_template`

### 2. Create the prompts in Langfuse

**Option A — via the seed script** (recommended, keeps text version-controlled):

1. Add the three prompt strings as constants in `scripts/ops/seed_langfuse_prompts.py`:
   ```python
   STEM_SYSTEM_PROMPT = """..."""
   STEM_RESPONSE_TEMPLATE = """..."""
   ```
2. Add three entries to the `prompts` list in `main()`:
   ```python
   ("stem.system_prompt",     STEM_SYSTEM_PROMPT),
   ("stem.writer_prompt",     STEM_SYSTEM_PROMPT),   # same text initially
   ("stem.response_template", STEM_RESPONSE_TEMPLATE),
   ```
3. Run: `python scripts/ops/seed_langfuse_prompts.py`

**Option B — directly in Langfuse UI** (faster, but text not in git):

Langfuse UI → Prompts → `+ New prompt`:
- Name: `stem.system_prompt` | Type: `Text` | paste content → Save
- Repeat for `stem.writer_prompt` and `stem.response_template`
- Assign label `production` to each version

### 3. Create the domain Python class

In `src/aix/domains/stem_domain.py`, extend `BaseDomainConfig` and implement the three prompt methods using the shared `fetch_prompt` helper:

```python
from aix.domains.base_config import BaseDomainConfig
from typing import Dict

class STEMDomainConfig(BaseDomainConfig):

    def _get_name(self) -> str:
        return "stem"                       # must match Langfuse prefix

    def _get_display_name(self) -> str:
        return "STEM"

    def _get_icon(self) -> str:
        return "🔬"

    def get_system_prompt(self) -> str:
        from aix.domains.langfuse_prompts import fetch_prompt
        return fetch_prompt("stem.system_prompt")

    def get_writer_prompt(self) -> str:
        from aix.domains.langfuse_prompts import fetch_prompt
        return fetch_prompt("stem.writer_prompt")

    def get_response_template(self) -> str:
        from aix.domains.langfuse_prompts import fetch_prompt
        return fetch_prompt("stem.response_template")

    # ... implement remaining abstract methods (get_node2vec_weights, etc.)
```

The key rule: `_get_name()` must return exactly the prefix used in Langfuse (e.g. `"stem"` → `stem.system_prompt`).

### 4. Register the domain

In `src/aix/domains/__init__.py`, import and register the new class so `get_domain_config("stem")` resolves it. Check the existing pattern for `neuro` and `udl`.

### 5. Add static fallback (optional but recommended)

In `src/aix/agent/configs/domain_prompts.py`, add a static extension to `DOMAIN_EXTENSIONS` as a fallback if Langfuse is unreachable during agent mode:

```python
STEM_WRITER_EXTENSION = """
## 🔬 STEM Principles (Domain Extension)
...
"""

DOMAIN_EXTENSIONS["stem"] = {
    "writer": STEM_WRITER_EXTENSION,
    "critic": "",   # fill in later
}
```

This is the safety net — agent mode falls back to it silently if Langfuse is down.

### 6. Verify

- Langfuse UI → Prompts: confirm `stem.system_prompt`, `stem.writer_prompt`, `stem.response_template` exist
- Restart server
- Select `stem` domain in the webui → generate a lesson → no `Langfuse prompt unavailable` in logs
