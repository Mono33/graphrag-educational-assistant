# AixLearning agent API — guide for chatbot integrations

**For:** teams building an application on top of the agent (chatbots, assistants)
**From:** AI team · **Date:** 2026-10-14 · **Base path:** `/api/v1`

Everything the WebUI does is available here: the documents a teacher attaches
(private RAG), the conversation, the fast answers, the lesson with its full
explanation, the downloads. The API uses the same code as the WebUI, so the
same request gives the same result.

Reference: **`docs/api/openapi.json`**, the contract exported for this version
(only the `/api` routes; load it in Swagger UI, Postman or a client generator).
`/docs`, `/redoc` and `/openapi.json` are switched off on staging and production
(`AIX_API_DOCS_ENABLED`), where the service has no proxy authentication in front;
they stay on in a local stack.

**Staging:** `https://agente.aiforlearning.digital/api/v1`. Every route needs the
credentials below, except `GET /health` and `GET /context/domains`.

---

## 1. The flow in four calls

```text
1. POST /api/v1/materials              upload a document          → id (processing)
   GET  /api/v1/materials/{id}         poll until status = ready
2. POST /api/v1/sessions               start a conversation       → session_id
3. POST /api/v1/sessions/{id}/turns    send a message             → lesson | answer | handoff
4. GET  /api/v1/sessions/{id}/export?format=docx                  → the lesson as a file
```

A runnable version of this flow: `scripts/eval/api_smoke.py`.

## 2. Authentication and end users

- **Service account** (an application): HTTP Basic with the credentials the AI team
  gives you. **Each instance of your platform gets its own** (production,
  corporate, staging…): the account is part of every end user's identity, so the
  same `X-End-User-Id` from two instances is two different people. Use the
  instance's credentials, never another's (server side: `GRAPH_API_USERNAME` /
  `GRAPH_API_PWD` and `AIX_API_CLIENTS`).
- **End user**: send `X-End-User-Id: <your opaque id>` on every call. Each id is a
  separate owner: one end user never sees another's documents or sessions (a
  `404`, which does not say whose they are). Without the header, everything
  belongs to your application's own default user.
- The id is stored only as a hash. Do not send names or e-mails; any stable
  opaque string (up to 200 printable characters) works.
- A WebUI teacher calling with their own login (JWT cookie) acts as themselves;
  their API sessions also appear in their WebUI.

### Retries: `Idempotency-Key`

Send `Idempotency-Key: <a unique id for this attempt's job>` on
`POST /sessions`, `POST /sessions/{id}/turns` (and `/turns/stream`) and
`POST /materials`, and **reuse the same key when you retry the same job**. For
example, a Dramatiq message id plus the turn number.

| Situation | Answer |
| --- | --- |
| First request with the key | Runs normally. |
| Same key, first request **still running** | `409 idempotency_in_progress` with `Retry-After`. Wait and retry with the same key, or read `GET /sessions/{id}`. |
| Same key, first request **succeeded** | The first answer again, with `Idempotent-Replayed: true`: no second lesson, no second cost. A session or a document comes back as it is **now** (a document may be `ready` by then). |
| Same key, first request **failed** | Runs again: a failure is not kept. |
| Same key, **different body** | `422 idempotency_key_reused`: a bug in the client. |

- Keys are per end user and per operation, and are kept for 24 hours.
- `/turns` and `/turns/stream` share their keys, so a retry may switch from one
  to the other and still get the first turn. The stream then sends just the
  `result` frame.
- Without a key every request is new, as before: a retry of a finished turn
  writes a second lesson.

## 3. Documents (private RAG)

The document library is the WebUI's own (`/webui/resources`): what a chatbot
uploads for an end user, that user would see there. Uploads, versions and
attaching need `AIX_MATERIAL_LIBRARY_ENABLED` on the server (otherwise `404
library_disabled`).

| Call | What it does |
| --- | --- |
| `POST /materials` (multipart: `file`; optional `display_name`, `purpose`, `lesson_id`) | Stores the document and starts reading it. `201` with `status: "processing"`. |
| `GET /materials/{id}` | `status`: `processing` → `ready` (searchable, `passages`) or `failed` (`error_code`); `latest_version.parse_quality` says what was read. |
| `GET /materials?limit=&offset=` | The end user's documents (`items`, `total`), each with `attached_lesson_ids` (the sessions using it). |
| `POST /materials/{id}/versions` | A new version of the same document. |
| `POST /materials/{id}/attach` `{"lesson_id": …}` | Use it in a session (same as `material_ids` on a session or a turn). |
| `DELETE /materials/{id}/lessons/{session_id}` | Stop using it in one session; it stays in the library. |
| `GET /materials/{id}/versions/{version_id}/source` | Download the original file. |
| `DELETE /materials/{id}` | Remove it from search and from the library. |
| `GET` `POST` `DELETE /materials/saved/resources[/{id}]` | Bookmarks of external resources (videos, articles), as the WebUI's bookmark. |

- Formats and limits are the WebUI's: PDF, DOCX, PPTX, TXT, Markdown, images;
  up to 100 MB and 1000 pages; macro-enabled, encrypted and disguised files are
  refused (`422` with a `code`: `magic_mismatch`, `ole_container`…; `413
  too_large`). Exact list: `GET /catalog` → `materials`.
- Scans are read with OCR: `parse_quality` reports pages, OCR pages, unread pages.
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
- The profile takes the AIxLearning platform's `Party` and `Classroom` values as
  they are stored:
  - the eight BES codes (`DSA`, `ADHD`, `DOP`, `DF`, `SC`, `PD`, `SA`, `CL`); the
    older `DCGL`, `DCGM`, `DCGS` and `DLDS` are still accepted;
  - classroom questions as `true`/`false` or `YES` / `NO` / `UNKNOWN`;
  - `environment_type` (`LOCAL`, `REMOTE`, `HYBRID`), `participation_type` and
    `prevalent_modality_type`.

  `UNKNOWN`, `"none"`, an omitted field and `students_number: 0` all mean *not
  said*: the lesson then makes no assumption about it, instead of ruling
  something out. A code the API does not know is refused with `422`.
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
| `lesson` | the first turn, `mode: lesson`, a refinement | ~30–90 s | `markdown`, plus `structure`, `planner`, `retriever`, `critic` or `judge` (see §6), `lesson_meta` |
| `answer` | a question about the lesson or the documents | ~3–6 s | `markdown`; the lesson did not change |
| `handoff` | in `auto`, the request needs a new lesson | ~1–2 s | ask the user; to proceed, resend `handoff_query` with `mode: "lesson"` (or `"answer"` to have it answered here) |

This is the WebUI's behaviour: a question gets a fast answer; «make me a lesson
on volcanoes» does not silently rewrite the lesson, it asks first.

A `handoff` turn is **not kept in the session**: `GET /sessions/{id}` does not
list it, because a message with no answer would sit in the history as a question
nobody took. The client keeps `handoff_query` and, if the user confirms, resends
it with `mode: "lesson"` (that turn is kept). Whether a request is a question or
a new lesson is the model's reading: «applica le correzioni alla lezione» was
answered in the WebUI («which corrections?») and handed off in an API replay of
the same conversation. Both readings are reasonable; the client must handle both.

Two turns of the same session at once: the second is refused with
`409 already_running`; retry it when the first ends (or read the session).

The documents bound to a session stay bound for every later turn. When the user
moves to another subject in the same conversation («now a lesson on volcanoes»
after one on the water cycle), the lesson keeps the new subject: the documents
are used only where they speak of it. If the platform knows the documents no
longer concern the conversation, it can open a new session or detach them
(`DELETE /materials/{id}/lessons/{session_id}`).

### Streaming

`POST /sessions/{id}/turns/stream` takes the same body and answers with
Server-Sent Events. Each `data:` frame is JSON with a `kind`:
`planner`, `retriever`, `writer_pending`, `writer`, then the review, `done` for a
lesson; `light_pending`, `light_done`, `light_escalate` for an answer or a
hand-off. The review depends on the server's configuration:

| Server | Review events | What they mean |
| --- | --- | --- |
| judge off (default) | `critic_pending`, `critic` | the Critic scores the lesson and may ask for a rewrite (a new `draft`) |
| judge on (`AIX_JUDGE_ENABLED=true`) | `judge_pending`, `judge` | the judge reads the whole lesson and corrects in place only what is wrong, typically 5–15 s after the text is written; `judge.data` is the `JudgeInfo` of the result |

Handle both: a client must not assume either. The last frame is
`{"kind": "result", "result": {…the same body /turns returns…}}`. A refusal or
failure is a frame `{"kind": "error", "status": …, "code": …, "message": …}`.

The text arrives while it is written, for lessons and answers alike:

```text
{"kind": "text", "text": "# La cellula\n\nLa cellula è", "draft": 1}
{"kind": "text", "text": " l'unità della vita…", "draft": 1}
{"kind": "draft_end", "draft": 1}
{"kind": "judge_pending", "data": {…}}
{"kind": "judge", "data": {"found": 3, "replaced": 2, "fixed": […], …}}
{"kind": "result", "result": {"markdown": "# La cellula\n\n…", "judge": {…}, …}}
```

- Append `text` to what you show.
- When `draft` grows, the lesson is being written again (a Critic revision, the
  attached documents left unused, groups that do not add up to the class):
  **clear the text and start over**.
- Between `draft_end` and `result` there can be a pause of several seconds (the
  review): show that the lesson is being checked, using `critic_pending` or
  `judge_pending`.
- The text is a **preview**. The final `result.markdown` has passed the checks:
  tidied, corrected by the judge when it is on, unsupported citations removed,
  citations resolved to document names. **Replace** the preview with it: with
  the judge on, it differs from the preview wherever a correction was made.
- The internal `[[material:…]]` citation markers never appear in the preview.
- On a `light_escalate` (hand-off), discard the preview.
- A piece can be a few characters or a few words. Show it as it comes; if you
  relay it (for example through Mercure), batching a few pieces is fine.

## 6. What a lesson comes with

- `markdown`: the lesson. Citations of the user's documents are written as
  «(dispensa.pdf, p. 3)» and listed at the end under «Fonti dai tuoi
  materiali» («Sources from your materials» under an answer written in
  English). Under the title, the class and the duration from the profile
  («**Livello:** Primaria · 22 alunni», «**Durata:** 60 minuti»); at the end, the
  sources note (whether the content comes from the user's documents or from
  external sources). The first line is a machine-readable AI-generated comment.
- `structure`: `title`, `duration_minutes`, `sections` (title, level, minutes),
  `timed_outline`, `warnings` (e.g. minutes that do not add up).
- `cited_documents`: names of the documents cited.
- `planner`: intent and its label, scope (`in_scope` / `partial_scope` /
  `out_of_scope`) and confidence, key and subject concepts, the search queries.
- `retriever`: what the Knowledge Graph returned (counts, top concepts, curated
  media in `media`), `coverage_tier`, `retrieval_outcome`, `kg_unreachable`
  (the graph did not answer: not the same as an out-of-scope question), and
  `reused` with its reason when a follow-up reused an earlier search.
- `critic` (judge off): approved or not, score (0-5 and percent), the assessment
  and the revision instructions. `null` when the judge is on.
- `judge` (judge on, `JudgeInfo`): the targeted review, the WebUI's «Revisione
  mirata» card. `found` problems, of which `blocking`; `fixed`, the corrections
  now in the lesson, each with `check`, `why`, `before` → `after` and
  `teacher_material` (the sentence came from the user's documents, which the judge
  does not read: show it as «to check against your document»); `points`, what
  could not be corrected, to check by hand; `reverted`, corrections judged wrong
  on verification and put back; `error` when the review failed (the lesson is then
  the uncorrected one). `null` when the judge is off. The WebUI shows nothing of
  this under the lesson: the corrections are already in `markdown`, and the card is
  where the teacher sees what changed.
- `lesson_meta`: duration, revisions, scores, counts, UDL coverage; with the judge
  on also `review_notes` (= `judge.points`) and the judge's raw summary.
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

Application errors from the materials and sessions routes use
`{"detail": {"code": "…", "message": "…"}}`. The `message` is Italian, for a
person; the `code` is stable, for your code. Authentication failures, standard
FastAPI request-validation errors and some errors from the older one-shot
`/agent/*` routes have their established shapes, so clients must also handle
those cases until the envelope is normalized.

| HTTP | `code` | Meaning |
| --- | --- | --- |
| 401 | — | Missing or wrong credentials. |
| 404 | `unknown_session`, `unknown_material`, `no_lesson_yet` | Not this end user's, or no lesson to download yet. |
| 404 | `library_disabled` | Document uploads are switched off on this server. |
| 409 | `already_running` | A turn of this session is still running. |
| 409 | `materials_preparing` | A bound document is still being read. Retry when `ready`. |
| 409 | `no_lesson_yet` | `mode: answer` before the first lesson. |
| 413 / 422 | `too_large`, `magic_mismatch`, … | The document was refused. |
| 422 | `empty_message`, `invalid_end_user_id` | Fix the request. |
| 429 | `rate_limited` | Too many turns (default 10 a minute per end user). Wait `Retry-After` seconds. |
| 502 | `generation_failed` | The generation failed; retry. |
| 503 | `at_capacity` | Every generation slot is busy (default 6 at once). Retry shortly. |
| 503 | `materials_disabled` | Document search is switched off on this server. |

## 10. One-shot generation (unchanged)

`POST /api/v1/agent/run` and `/agent/stream` still generate one lesson from one
request, without a session or documents. They now also return the `critic`
card (or the `judge` card, when the judge is on) and the extra planner and
retriever fields. For anything conversational, use sessions.

---

## Appendix A — Dev integration checklist and current boundaries

This appendix does not change the endpoint contract described above. It
collects the integration rules the AI and Dev teams should freeze together
before production. The implementation review is
[`review_api_contract.md`](review_api_contract.md).

### Which API to use

- Use `/materials`, `/sessions` and `/sessions/{id}/turns[/stream]` for the
  conversational lesson tools and private RAG.
- Keep `/context` for existing tools that only need Knowledge Graph context.
- Keep `/agent/run` and `/agent/stream` only for one-shot generation without
  persistent private documents.

The older [`API_INTEGRATION_GUIDE.md`](API_INTEGRATION_GUIDE.md) documents the
GraphRAG-only `/context` integration; it is not the guide for the new
conversational flow.

### What the platform must persist

Persist the GraphRAG `session_id` and returned material IDs. After a session is
created, send only the new user turn: GraphRAG stores the conversation and
material bindings, so the platform must not resend the full history and full
document text on every request.

Use instance-specific Basic credentials and a stable opaque
`X-End-User-Id`. Production, corporate and staging must not share credentials.

### Dramatiq retries

Generate one `Idempotency-Key` per logical upload, session creation or turn,
and reuse that exact value for every Dramatiq retry of the same operation.
Do not generate a new key inside each retry attempt.

When a streamed request is retried through the blocking endpoint, the same key
still identifies the original turn. A replayed result has
`Idempotent-Replayed: true`.

### Mercure adapter

The adapter must:

1. append `text` frames for the active draft;
2. clear the preview when the `draft` number increases;
3. discard the preview on `light_escalate`;
4. replace the preview with the final `result.markdown`;
5. treat the final `result` frame as authoritative.

### Current contract boundaries

These points are intentionally visible to Dev because they remain joint work:

- **Anonymization:** decide whether AIxLearning uploads anonymized,
  page-structured Markdown or GraphRAG receives and anonymizes originals.
- **Production identity:** Basic auth plus an IP allowlist is the pilot
  boundary; signed identity remains recommended before production.
- **Required documents:** `purpose="must_follow"` is accepted, but a failed
  required document is not yet a hard failure.
- **Preparing documents:** `409 materials_preparing` currently carries names in
  the human message; it does not yet return a structured material list or
  `Retry-After`.
- **Provenance:** citations are resolved in Markdown and document names are
  returned, but there is not yet a structured source/version/page payload or a
  page-preview endpoint.
- **Errors:** handle both the application `{code, message}` envelope and
  standard FastAPI validation/authentication errors.
- **Credits and exports:** agree how `lesson`, `answer` and `handoff` are
  charged, and whether AIxLearning keeps its own watermark/export pipeline.
- **Tracing:** agree a correlation identifier across AIxLearning, GraphRAG and
  Langfuse.

### Minimum staging acceptance

Before Dev starts end-to-end testing:

- run all Alembic migrations;
- enable and audit durable material storage;
- configure one service credential per platform instance;
- restrict the staging API by network policy;
- protect `/docs`, `/redoc`, `/openapi.json` and `/context`;
- export a version-pinned OpenAPI document for the Dev implementation;
- run `scripts/eval/api_smoke.py` against the deployed stack;
- verify that two instances using the same end-user value remain isolated.
