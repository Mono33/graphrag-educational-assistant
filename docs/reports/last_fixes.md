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
