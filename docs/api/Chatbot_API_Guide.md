# AixLearning agent API — guide for chatbot integrations

**For:** teams building an application on top of the agent (chatbots, assistants)
**From:** AI team · **Date:** 2026-10-14 · **Base path:** `/api/v1`

Everything the WebUI does is available here: the documents a teacher attaches
(private RAG), the conversation, the fast answers, the lesson with its full
explanation, the downloads. The API uses the same code as the WebUI, so the
same request gives the same result.

Interactive reference: `GET /docs` (Swagger) on the server.

---

## 1. The flow in four calls

```text
1. POST /api/v1/materials              upload a document          → material_id (processing)
   GET  /api/v1/materials/{id}         poll until status = ready
2. POST /api/v1/sessions               start a conversation       → session_id
3. POST /api/v1/sessions/{id}/turns    send a message             → lesson | answer | handoff
4. GET  /api/v1/sessions/{id}/export?format=docx                  → the lesson as a file
```

A runnable version of this flow: `scripts/eval/api_smoke.py`.

## 2. Authentication and end users

- **Service account** (an application): HTTP Basic with the credentials the AI team
  gives you (`GRAPH_API_USERNAME` / `GRAPH_API_PWD` on the server).
- **End user**: send `X-End-User-Id: <your opaque id>` on every call. Each id is a
  separate owner: one end user never sees another's documents or sessions (a
  `404`, which does not say whose they are). Without the header, everything
  belongs to your application's own default user.
- The id is stored only as a hash. Do not send names or e-mails; any stable
  opaque string (up to 200 printable characters) works.
- A WebUI teacher calling with their own login (JWT cookie) acts as themselves;
  their API sessions also appear in their WebUI.

## 3. Documents (private RAG)

| Call | What it does |
| --- | --- |
| `POST /materials` (multipart, field `file`) | Stores the document, starts reading it. `202` with `status: "processing"`. |
| `GET /materials/{id}` | `processing` → `ready` (searchable) or `failed` (`error_code`). |
| `GET /materials` | The end user's documents, with the sessions using each. |
| `DELETE /materials/{id}` | Removes it from search and from the library. |

- Formats and limits are the WebUI's: PDF, DOCX, PPTX, TXT, Markdown, images;
  up to 100 MB and 1000 pages; macro-enabled, encrypted and disguised files are
  refused (`422` with a `code`: `magic_mismatch`, `ole_container`, `too_large`…).
  Exact list: `GET /catalog` → `materials`.
- Scans are read with OCR. `quality` tells you what was read: pages, OCR pages,
  unread pages, warnings.
- A document is uploaded **once** and used by any number of sessions.
- A turn waits for its documents: a turn sent while one is still `processing` is
  refused with `409 materials_preparing`. Poll the document, then send the turn.
- The text of documents is sent to the embedding provider, as in the WebUI.
- Documents are kept until you delete them.

## 4. Sessions

`POST /sessions`:

```json
{
  "domain": "udl",
  "title": "La Guerra fredda",
  "educational_profile": {
    "pedagogical_intent": "intro",
    "group": {"grade": "SECONDARIA_II_GRADO"}
  },
  "material_ids": ["…"]
}
```

- `domain`: `neuro` or `udl` (`GET /catalog` → `domains`).
- `educational_profile`: the WebUI's profile form (class, BES, classroom, devices);
  codes and their labels in `GET /catalog` → `profile`. Set
  `pedagogical_intent` here: the API does not ask for it with a card, as the
  WebUI does.
- `material_ids`: documents the session's turns may search. More can be bound
  later with `POST /sessions/{id}/materials` or in a turn.

`GET /sessions/{id}` returns the profile, `status` (`draft`, `running`,
`complete`, `error`), the current lesson (`lesson_markdown`, `lesson_structure`,
`cited_documents`), every turn, and the bound documents. `GET /sessions` lists
them; `DELETE /sessions/{id}` deletes one.

The language is detected from the messages, as in the WebUI.

## 5. Turns

`POST /sessions/{id}/turns`:

```json
{"message": "Crea una lezione di 45 minuti sulla Guerra fredda", "mode": "auto"}
```

| Field | Meaning |
| --- | --- |
| `message` | What the user writes (up to 2000 characters, no minimum: «sì» is a message). |
| `mode` | `auto` (default): as the WebUI. `answer`: answer here, never hand off. `lesson`: rewrite the lesson with the whole chain. |
| `refinement` | `{"code": "simplify", "detail": "…"}`, a WebUI refinement chip; rewrites the lesson. Codes in `GET /catalog`. |
| `search_graph` | On a follow-up that rewrites the lesson: `false` reuses what the Knowledge Graph found, `true` searches again. Omit to let the agent decide. |
| `material_ids` | Documents to bind before this turn. |

The result's `kind`:

| `kind` | When | Time | What to show |
| --- | --- | --- | --- |
| `lesson` | the first turn, `mode: lesson`, a refinement | ~30–90 s | `markdown`, plus `structure`, `planner`, `retriever`, `critic`, `lesson_meta` |
| `answer` | a question about the lesson or the documents | ~3–6 s | `markdown`; the lesson did not change |
| `handoff` | in `auto`, the request needs a new lesson | ~1–2 s | ask the user; to proceed, resend `handoff_query` with `mode: "lesson"` (or `"answer"` to have it answered here) |

This is the WebUI's behaviour: a question gets a fast answer; «make me a lesson
on volcanoes» does not silently rewrite the lesson, it asks first.

### Streaming

`POST /sessions/{id}/turns/stream` takes the same body and answers with
Server-Sent Events. Each `data:` frame is JSON with a `kind`:
`planner`, `retriever`, `writer_pending`, `writer`, `critic`, `done` for a lesson,
`light_pending`, `light_done`, `light_escalate` for an answer or a hand-off. The
last frame is `{"kind": "result", "result": {…the same body /turns returns…}}`.
A refusal or failure is a frame `{"kind": "error", "status": …, "code": …, "message": …}`.

## 6. What a lesson comes with

- `markdown`: the lesson. Citations of the user's documents are written as
  «(dispensa.pdf, p. 3)» and listed at the end under «Fonti dai tuoi
  materiali». The first line is a machine-readable AI-generated comment.
- `structure`: `title`, `duration_minutes`, `sections` (title, level, minutes),
  `timed_outline`, `warnings` (e.g. minutes that do not add up).
- `cited_documents`: names of the documents cited.
- `planner`: intent and its label, scope (`in_scope` / `partial_scope` /
  `out_of_scope`) and confidence, key and subject concepts, the search queries.
- `retriever`: what the Knowledge Graph returned (counts, top concepts, curated
  media in `media`), `coverage_tier`, `retrieval_outcome`, `kg_unreachable`
  (the graph did not answer: not the same as an out-of-scope question), and
  `reused` with its reason when a follow-up reused an earlier search.
- `critic`: approved or not, score (0-5 and percent), the assessment and the
  revision instructions.
- `lesson_meta`: duration, revisions, scores, counts, UDL coverage.
- `GET /sessions/{id}/media`: papers, Wikipedia and videos on the subject, found
  live (the WebUI's right panel; empty when nothing is found).
- `ai_disclosure` in the body and `X-AI-Generated: true` in the headers (AI Act
  Art. 50): show the notice next to generated content.

## 7. Downloads

`GET /sessions/{id}/export?format=md|txt|pdf|docx|outline` — the WebUI's buttons:
Markdown, plain text, PDF, Word (editable, headings and tables kept) and the
timed outline. Citations are resolved the same way.

## 8. Catalog

`GET /catalog`: domains, pedagogical intents, refinement codes, turn modes,
profile codes with their Italian labels, document formats and limits. Build your
choices from it rather than hard-coding them.

## 9. Errors

Every error body is `{"detail": {"code": "…", "message": "…"}}`. The `message` is
Italian, for a person; the `code` is stable, for your code.

| HTTP | `code` | Meaning |
| --- | --- | --- |
| 401 | — | Missing or wrong credentials. |
| 404 | `unknown_session`, `unknown_material`, `no_lesson_yet` | Not this end user's, or no lesson to download yet. |
| 409 | `already_running` | A turn of this session is still running. |
| 409 | `materials_preparing` | A bound document is still being read. Retry when `ready`. |
| 409 | `no_lesson_yet` | `mode: answer` before the first lesson. |
| 413 / 422 | `too_large`, `magic_mismatch`, … | The document was refused. |
| 422 | `empty_message`, `invalid_end_user_id` | Fix the request. |
| 429 | `rate_limited` | Too many turns (default 10 a minute per end user). Wait `Retry-After` seconds. |
| 502 | `generation_failed` | The generation failed; retry. |
| 503 | `at_capacity` | Every generation slot is busy (default 6 at once). Retry shortly. |
| 503 | `materials_disabled` | Documents are switched off on this server. |

## 10. One-shot generation (unchanged)

`POST /api/v1/agent/run` and `/agent/stream` still generate one lesson from one
request, without a session or documents. They now also return the `critic`
card and the extra planner and retriever fields. For anything conversational,
use sessions.
