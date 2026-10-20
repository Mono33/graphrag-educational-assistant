# AixLearning agent API — quickstart

**For:** the AIxLearning platform developers · **Start here.** One page: how it
works and the calls to make. The details are in
[`Chatbot_API_Guide.md`](Chatbot_API_Guide.md).

## What it is

A teacher talks to the agent in a conversation. The agent writes a lesson
adapted to the class, answers questions about it, and can use documents the
teacher uploads. **The agent keeps the conversation and the documents**: your
platform stores only two ids (`session_id`, `material_id`) and sends one new
message at a time.

## Access

| | Staging |
| --- | --- |
| Base URL | `https://agente.aiforlearning.digital/api/v1` |
| Auth | HTTP Basic: username and password, given to you separately |
| End user | header `X-End-User-Id: <your teacher's id>` on **every** call |

- `X-End-User-Id`: any stable opaque id from your database (no names, no
  e-mails). Each id sees only its own sessions and documents.
- One set of credentials per platform instance (staging, production…). Keep them
  server side, never in the browser.

## The flow

```text
(optional) 1. POST /materials                upload a document    → material id
           2. GET  /materials/{id}           repeat until "ready"
           3. POST /sessions                 start a conversation → session_id
           4. POST /sessions/{id}/turns      send each message    → lesson | answer | handoff
(optional) 5. GET  /sessions/{id}/export?format=docx              → the lesson as a file
```

Examples with `curl` (set `U=username:password` and `B=https://agente.aiforlearning.digital/api/v1`):

```bash
# 1. upload a document (optional)
curl -u "$U" -H "X-End-User-Id: teacher-42" -F "file=@dispensa.pdf" "$B/materials"
#   → {"id": "…", "status": "processing", …}

# 2. wait until it is read (a few seconds to a few minutes for a scan)
curl -u "$U" -H "X-End-User-Id: teacher-42" "$B/materials/<material_id>"
#   → "status": "ready"   (or "failed" with an "error_code")

# 3. start a conversation
curl -u "$U" -H "X-End-User-Id: teacher-42" -H "Content-Type: application/json" \
  -d '{"domain": "udl",
       "educational_profile": {"pedagogical_intent": "intro",
                               "group": {"grade": "SECONDARIA_II_GRADO"}},
       "material_ids": ["<material_id>"]}' \
  "$B/sessions"
#   → {"session_id": "…", …}

# 4. send a message (repeat for every message of the teacher)
curl -u "$U" -H "X-End-User-Id: teacher-42" -H "Content-Type: application/json" \
  -H "Idempotency-Key: turn-<session_id>-1" \
  -d '{"message": "Crea una lezione di 45 minuti sulla Guerra fredda"}' \
  "$B/sessions/<session_id>/turns"
```

## What comes back from a message

Show `markdown`; what else to do depends on `kind`:

| `kind` | When | Time | What you do |
| --- | --- | --- | --- |
| `lesson` | the first message, or a request to rewrite the lesson | 30–90 s | show `markdown` (the lesson) |
| `answer` | a question about the lesson or the documents | 3–6 s | show `markdown` (the lesson did not change) |
| `handoff` | the teacher asked for a **new** lesson mid-conversation | 1–2 s | ask «Do you want a new lesson?»; if yes, resend `handoff_query` with `"mode": "lesson"` |

Always show `ai_disclosure` (AI Act Art. 50) next to generated content.

## Three rules that save you trouble

1. **Use streaming in the UI.** `POST /sessions/{id}/turns/stream` takes the same
   body and sends the text while it is written (Server-Sent Events). Append each
   `text` frame; if `draft` increases, clear and start over; when the `result`
   frame arrives, **replace** everything with `result.markdown` (it is the checked
   version). Use `/turns` (no stream) for background jobs.
2. **Send an `Idempotency-Key`** on every `POST`, and the **same** key when you
   retry the same job (e.g. a Dramatiq retry). A retry then returns the first
   result instead of paying for a second lesson.
3. **One message at a time per session.** A second message while the first is
   still running gets `409 already_running`: wait and retry.

## Errors to handle

The body is `{"detail": {"code": "…", "message": "…"}}`: branch on `code`, show
`message` (Italian) to the person.

| HTTP | `code` | What to do |
| --- | --- | --- |
| 401 | — | wrong credentials |
| 404 | `unknown_session`, `unknown_material` | not this teacher's, or deleted |
| 409 | `already_running` | a message of this session is still running: retry later |
| 409 | `materials_preparing` | a document is still being read: wait for `ready` |
| 429 | `rate_limited` | too many messages: wait `Retry-After` seconds |
| 502 / 503 | `generation_failed`, `at_capacity` | retry shortly |

## The files you receive

| File | Use it for |
| --- | --- |
| `QUICKSTART.md` (this page) | understanding the flow and starting |
| `Chatbot_API_Guide.md` | every detail: profile codes, stream events, what a lesson carries, downloads |
| `openapi.json` | **not to read**: import it in Postman or Swagger, or generate a typed client from it |

Codes accepted in the profile (classes, BES, intents) come from
`GET /catalog`: build your menus from it rather than hard-coding them.
