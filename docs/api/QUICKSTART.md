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

## How a chat works, end to end

The browser never calls the agent: your **backend** does, with the credentials.

```text
teacher's browser ──(1) message──▶ your backend / worker ──(2) POST /turns/stream──▶ agent
       ▲                                   │  ◀──(3) text frames, then result──────────┘
       └──────(4) relay (Mercure…)─────────┘
```

1. The teacher writes in your chat. Your frontend sends it to your backend.
2. Your backend (e.g. a Dramatiq job) looks up the chat's `session_id` in your
   database — or creates the session on the chat's first message and stores it —
   and calls `/turns/stream`.
3. The agent sends the text while it is written, then one `result` frame.
4. Your backend relays the pieces to the browser (Mercure, WebSocket…) and,
   at the end, stores `result.markdown` as the chat's assistant message.

What your database needs: **chat id → `session_id`**, and **uploaded file →
`material_id`**. Nothing else: the agent keeps the history.

A worker in Python (`httpx`); `publish` is whatever sends to the browser:

```python
import json, httpx

BASE = "https://agente.aiforlearning.digital/api/v1"
AUTH = ("<username>", "<password>")          # from your server's settings

def send_message(session_id, teacher_id, message, job_id, publish, mode="auto"):
    headers = {"X-End-User-Id": teacher_id, "Idempotency-Key": job_id}
    draft = 0
    with httpx.stream("POST", f"{BASE}/sessions/{session_id}/turns/stream",
                      auth=AUTH, headers=headers,
                      json={"message": message, "mode": mode},
                      timeout=httpx.Timeout(10, read=180)) as r:
        if r.status_code != 200:              # 401, 404, 409 idempotency…
            r.read()
            raise RuntimeError(r.status_code, r.text)
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            frame = json.loads(line[5:])
            kind = frame["kind"]
            if kind == "text":
                if frame["draft"] != draft:   # the lesson is written again
                    draft = frame["draft"]
                    publish({"type": "reset"})
                publish({"type": "text", "text": frame["text"]})
            elif kind in ("planner", "retriever", "judge", "critic"):
                publish({"type": "card", "card": kind, "data": frame["data"]})
            elif kind in ("judge_pending", "critic_pending"):
                publish({"type": "status", "text": "Sto controllando la lezione…"})
            elif kind == "light_escalate":    # a hand-off: drop the preview
                publish({"type": "reset"})
            elif kind == "error":             # already_running, rate_limited…
                raise RuntimeError(frame["status"], frame.get("code"), frame.get("message"))
            elif kind == "result":
                result = frame["result"]
                if result["kind"] == "handoff":
                    publish({"type": "confirm_new_lesson", "query": result["handoff_query"]})
                else:
                    publish({"type": "final", "markdown": result["markdown"]})
                return result
```

If the teacher confirms a hand-off, call `send_message(…, result["handoff_query"],
<new job id>, publish, mode="lesson")`.

## Showing what the WebUI shows: the cards and the media

Everything the WebUI shows next to a lesson comes from the API, with the same
field names (a test keeps the two in step):

| WebUI | During the stream (frame `kind`, its `data`) | In the final `result` |
| --- | --- | --- |
| Planner card: intent, scope, concepts, searches | `planner`, after ~3–5 s | `result.planner` |
| Knowledge Graph card: what was found, coverage | `retriever`, after ~10 s | `result.retriever` |
| Curated resources from the graph (videos, articles, textbooks) | inside `retriever` → `media` | `result.retriever.media` |
| «Revisione mirata» card: corrections made, points to check | `judge_pending` (show «checking…»), then `judge` | `result.judge` |
| Right panel: papers, Wikipedia, videos found live | — | `GET /sessions/{id}/media`, after the lesson |

- In the stream, show each card as its frame arrives, as the WebUI does; the
  `result` frame carries them all again, so a client that does not stream gets
  the same cards from `/turns`.
- `judge`: `fixed` lists each correction (`why`, `before` → `after`; with
  `teacher_material: true`, say «to check against your document»), `points` what
  could not be corrected. The corrections are already in `markdown`.
- `GET /sessions/{id}/media`: call it once after a lesson; it searches the first
  time (a few seconds) and replays afterwards. Empty lists when nothing is found.
- With the judge off on a server, the review card is `critic` instead of `judge`
  (`critic_pending`, `critic`): handle both.
- An `answer` turn has no cards: the lesson and its cards stay as they were.

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

Not needed for the chat: `API_INTEGRATION_GUIDE.md` and
`Explainability_API_Guide_for_Frontend.md` describe `POST /context`, the
Knowledge Graph context for tools that build their own prompts;
`review_api_contract.md` is the AI team's review of this contract.

Codes accepted in the profile (classes, BES, intents) come from
`GET /catalog`: build your menus from it rather than hard-coding them.
