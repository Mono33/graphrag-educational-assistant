# Agentic GraphRAG — Technical Integration Guide

**Date:** May 13, 2026
**Audience:** AixLearning DEV team (hands-on-keyboard reference)
**Companion to:** [Dev_Handoff_AgenticGraphRAG_Integration.md](Dev_Handoff_AgenticGraphRAG_Integration.md)

---

## 1. API Endpoints

| Endpoint | Verb | Auth | Purpose |
|---|---|---|---|
| `/docs` | GET | none | Swagger UI — browse + **Try it out** with example payloads |
| `/redoc` | GET | none | ReDoc — read-only HTML spec |
| `/openapi.json` | GET | none | OpenAPI 3.1 spec for code generation |
| `/api/v1/health` | GET | none | Liveness probe |
| **`/api/v1/agent/run`** | **POST** | **Basic Auth** | **Sync JSON — full lesson plan in one response** |
| **`/api/v1/agent/stream`** | **POST** | **Basic Auth** | **SSE — multi-event stream** |
| `/api/v1/context` | POST | Basic Auth | Legacy one-shot KG retrieval (unchanged) |

**Stability guarantee:** existing paths are locked by an automated regression test. We will never rename or remove an endpoint. New response fields are always additive.

---

## 2. Request Schema

`POST /api/v1/agent/run` and `POST /api/v1/agent/stream` accept the same JSON body:

```json
{
  "query": "crea una lezione sulla motivazione intrinseca",
  "domain": "neuro",
  "language": "it",
  "session_id": "conversation-uuid-123",
  "max_revisions": 2,
  "educational_profile": {
    "group": {
      "title": "3°A scuola primaria",
      "students_number": 24,
      "grade": "primaria_3",
      "disabilities": ["DSA", "ADHD"],
      "class_features": ["heterogeneous"],
      "student_attributes": ["curious", "energetic"]
    },
    "classroom": {
      "title": "Aula 12",
      "forniture_mobility": "flexible",
      "has_lim": true,
      "has_wifi": true,
      "has_suite": false,
      "pc_station": "shared",
      "own_device": "byod"
    },
    "time_available_minutes": 60,
    "subject_area": "Scienze",
    "specific_topic": "fotosintesi clorofilliana"
  },
  "teacher_provided_context": null
}
```

The `educational_profile` fields map 1:1 to your existing Django models:
- `group` → `party.models.Party` (grade, students_number, students_disabilities, party_feature)
- `classroom` → `classroom.models.Classroom` (forniture_mobility, has_lim, has_wifi, etc.)

---

## 3. Sync Response Schema

`POST /api/v1/agent/run` returns:

```json
{
  "status": "success",
  "lesson_plan_md": "# Lezione: la motivazione intrinseca\n\n## Obiettivi…",
  "planner": { "intent": "lesson_plan", "key_concepts": [...], "external_apis_needed": true },
  "retriever": { "nodes_count": 7, "coverage_tier": "healthy", "media": [...] },
  "writer": { "tokens_in": 4231, "tokens_out": 1879 },
  "critic": { "approved": true, "revisions": 0, "scores": {...} },
  "media_pool": [ /* curated articles / videos / OER */ ],
  "trace_id": "langfuse-trace-uuid"
}
```

---

## 4. SSE Event Taxonomy

`POST /api/v1/agent/stream` returns `text/event-stream`. Each event: `event: <kind>\ndata: <json>\n\n`

| `kind` | When | Key payload fields |
|---|---|---|
| `planner` | After Planner completes | `intent`, `key_concepts`, `external_apis_needed` |
| `retriever` | After Retriever completes | `nodes_count`, `coverage_tier` (healthy/limited/out_of_scope), `domain_label_short`, `media` |
| `writer_pending` | Writer starts composing | `phase: "drafting"` |
| `writer_token_delta` | Per-token streaming | `delta: "..."` |
| `writer` | Writer completes | `lesson_plan_md` (full), `tokens_in`, `tokens_out` |
| `critic` | Critic completes | `approved`, `revisions`, `scores`, `critique` |
| `done` | Pipeline complete | `trace_id`, `total_duration_ms` |
| `error` | Any agent fails | `error`, `phase` (planner/retriever/writer/critic) |

For V1 integration, you only need to consume `writer_token_delta` (map to your `LLMTextDeltaEvent`), `done` (map to `LLMResponseCompletedEvent` + `LLMTextDoneEvent`), and `error`. The other events can be ignored initially and consumed later for frontend enhancements.

---

## 5. Implementation — New Service Wrapper

Create `lesson_planner/service/agentic_graph_rag.py` — mirrors the existing `GraphRagService` shape:

```python
import os
import json
import httpx
from httpx_tenacity import TenaciousTransport
from typing import Generator, Any

from aixlearning.settings import GRAPH_API_ENDPOINT
from lesson_planner.clients.streaming_events import (
    LLMStreamStartedEvent,
    LLMTextDeltaEvent,
    LLMStreamEvent,
    LLMTextDoneEvent,
    LLMResponseCompletedEvent,
)
from lesson_planner.methods import logger

transport = TenaciousTransport.create()

DOMAIN_BY_PLAN_TYPE = {"UDL": "udl", "NEURO": "neuro"}


class AgenticGraphRagService(httpx.Client):
    transport: TenaciousTransport = transport

    def __init__(self, *args, **kwargs):
        base_url = kwargs.pop("base_url", None) or os.environ.get("GRAPH_API_ENDPOINT")
        username = kwargs.pop("username", None) or os.environ.get("GRAPH_API_USERNAME")
        password = kwargs.pop("password", None) or os.environ.get("GRAPH_API_PWD")
        auth = httpx.BasicAuth(username, password)
        headers = {
            **kwargs.pop("headers", {}),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        super().__init__(*args, **{**kwargs, "base_url": base_url, "headers": headers, "auth": auth})

    def stream_agent(
        self,
        *,
        plan_type: str,
        query: str,
        party,
        classroom,
        language: str = "it",
        session_id: str | None = None,
    ) -> Generator[LLMStreamEvent, Any, None]:
        payload = self._build_payload(plan_type, query, party, classroom, language, session_id)
        logger.info(f"Calling Agentic GraphRAG /stream with payload keys: {list(payload.keys())}")
        accumulated = ""
        with self.stream("POST", GRAPH_API_ENDPOINT + "/api/v1/agent/stream",
                         json=payload, timeout=None) as r:
            r.raise_for_status()
            yield LLMStreamStartedEvent(response_id=r.headers.get("X-Trace-Id", session_id))
            buffer = ""
            for chunk in r.iter_text(chunk_size=1024):
                buffer += chunk
                while "\n\n" in buffer:
                    raw_event, buffer = buffer.split("\n\n", 1)
                    kind, data = self._parse_sse(raw_event)
                    if kind == "writer_token_delta":
                        accumulated += data.get("delta", "")
                        yield LLMTextDeltaEvent(delta=data["delta"])
                    elif kind == "writer":
                        accumulated = data.get("lesson_plan_md", accumulated)
                    elif kind == "done":
                        yield LLMResponseCompletedEvent(
                            output=accumulated,
                            input_tokens=data.get("tokens_in", 0),
                            output_tokens=data.get("tokens_out", 0),
                            total_tokens=data.get("total_tokens", 0),
                        )
                        yield LLMTextDoneEvent(item_id=data.get("trace_id"))
                    elif kind == "error":
                        raise RuntimeError(f"Agent error in phase '{data.get('phase')}': {data.get('error')}")

    @staticmethod
    def _parse_sse(raw_event: str) -> tuple[str, dict]:
        kind, data_line = "message", "{}"
        for line in raw_event.splitlines():
            if line.startswith("event: "):
                kind = line[7:].strip()
            elif line.startswith("data: "):
                data_line = line[6:]
        return kind, json.loads(data_line) if data_line else {}

    def _build_payload(self, plan_type, query, party, classroom, language, session_id) -> dict:
        return {
            "query": query,
            "domain": DOMAIN_BY_PLAN_TYPE[plan_type],
            "language": language,
            "session_id": session_id,
            "max_revisions": 2,
            "educational_profile": _map_profile(party, classroom),
        }


def _map_profile(party, classroom) -> dict:
    """Adapter: Django Party + Classroom → Pydantic EducationalProfile shape."""
    return {
        "group": {
            "title": getattr(party, "title", None),
            "students_number": party.students_number,
            "grade": party.grade,
            "disabilities": [d.feature for d in party.students_disabilities.all()],
            "class_features": [f.feature for f in party.party_feature.all()],
        } if party else None,
        "classroom": {
            "title": getattr(classroom, "title", None),
            "forniture_mobility": classroom.forniture_mobility,
            "has_lim": classroom.has_lim,
            "has_wifi": classroom.has_wifi,
            "has_suite": classroom.has_suite,
            "pc_station": classroom.pc_station,
            "own_device": classroom.own_device,
        } if classroom else None,
        "time_available_minutes": getattr(party, "time_available_minutes", None),
        "subject_area": getattr(party, "subject_area", None),
        "specific_topic": getattr(party, "specific_topic", None),
    }
```

---

## 6. Implementation — Routing Update

Edit `lesson_planner/streaming_service/streaming.py` to dispatch UDL + NEURO to the new service:

```python
from lesson_planner.service.agentic_graph_rag import AgenticGraphRagService

AGENTIC_PLAN_TYPES = {"UDL", "NEURO"}

@dramatiq.actor(max_retries=STREAMING_RETRIES, max_backoff=30)
def stream_text_response(conversation_id: int, user_prompt: str, tokens_spent: int, ...):
    conversation = Conversation.objects.get(id=conversation_id)
    dct, pct = spend_credits(conversation.lesson_plan.created_by, tokens_spent, audit_action)
    tool = get_all_slug_tools()[conversation.lesson_plan.plan_type]
    observability_data = setup_observability(conversation, is_rag=True, user_message_id=user_message_id)
    try:
        if conversation.lesson_plan.plan_type in AGENTIC_PLAN_TYPES:
            with AgenticGraphRagService() as client:
                stream = client.stream_agent(
                    plan_type=conversation.lesson_plan.plan_type,
                    query=user_prompt,
                    party=conversation.lesson_plan.party,
                    classroom=conversation.lesson_plan.classroom,
                    language=lang,
                    session_id=str(conversation.id),
                )
                stream_handler = StreamEventHandler(conversation, tool, observability_data, intent)
                for event in stream:
                    stream_handler.handle(event)
                return
        else:
            # Existing flow — unchanged
            is_rag, system_prompt_message = tool.get_system_prompt(user_prompt=user_prompt, lang=lang)
            with TextClient(conversation) as client:
                stream = client.generate(
                    system_prompt=system_prompt_message, message=user_prompt,
                    observability=observability_data,
                )
                stream_handler = StreamEventHandler(conversation, tool, observability_data, intent)
                for event in stream:
                    stream_handler.handle(event)
                return
    except Exception as exc:
        # Existing error handling — unchanged (refund credits, set ERROR status, etc.)
        ...
```

---

## 7. Implementation — Docker Compose

Add the GraphRAG API service to `docker-compose.prod.yaml`:

```yaml
services:
  graphrag-api:
    image: ghcr.io/fem-modena/graphrag-aixlearning:latest
    networks: [fem-internal]
    environment:
      - WEBUI_CORS_ALLOW_ORIGINS=https://aixlearning.fem-modena.it
      - DATABASE_URL=${GRAPHRAG_DB_URL}
      - OPENAI_MODEL=anthropic/claude-sonnet-4.5
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - AIX_CORRECTIVE_RAG_ENABLED=false

  backend:
    environment:
      - GRAPH_API_ENDPOINT=http://graphrag-api:8765
      - GRAPH_API_USERNAME=${GRAPH_API_USERNAME}
      - GRAPH_API_PWD=${GRAPH_API_PWD}
```

The GraphRAG API service is NOT publicly exposed — only AixLearning's backend and worker reach it from inside the `fem-internal` Docker network.

---

## 8. Environment Variables (our side)

| Env var | Default | Purpose |
|---|---|---|
| `AIX_CORRECTIVE_RAG_ENABLED` | `false` | Multi-pass retrieval grading. Keep `false` until latency fixes ship |
| `AIX_CORRECTIVE_RAG_MAX_ATTEMPTS` | `2` | Max retry attempts when Corrective RAG is ON |
| `AIX_COVERAGE_HEALTHY_THRESHOLD` | `5` | Min KG nodes for `coverage_tier=healthy` (range 1-50) |
| `WEBUI_CORS_ALLOW_ORIGINS` | `*` | Comma-separated origin list — lock to your hostname in prod |
| `OPENAI_MODEL` | `anthropic/claude-sonnet-4.5` | LLM used inside the agents (via OpenRouter) |

---

## 9. Operational SLAs

| Metric | Target |
|---|---|
| `/api/v1/agent/run` p95 latency | ≤ 180 s (KG-covered), ≤ 240 s (out-of-scope) |
| `/api/v1/agent/stream` first-byte latency | ≤ 5 s (Planner event) |
| Availability | 99.5% monthly (post-deploy) |
| Schema changes | Strictly additive; breaking changes only with `/api/v2/` |

---

## 10. FAQ

### Do we have to migrate to JWT, or can we keep Basic Auth?

Basic Auth is fine for V1. We accept the same `GRAPH_API_USERNAME` + `GRAPH_API_PWD` credential on `/api/v1/agent/*` as on the existing `/api/v1/context`. JWT migration is planned for a later phase and will coexist with Basic Auth — no breaking change.

### What happens to the other 20 plan_types?

Nothing. All 20 keep using `TextClient` → OpenRouter `openai/gpt-4o` exactly as today. The `AGENTIC_PLAN_TYPES = {"UDL", "NEURO"}` set is the only routing decision.

### Will teachers notice any difference?

For V1 (no frontend changes): same chat UI, same markdown rendering, same Mercure streaming. The only difference is content quality — UDL and NEURO lesson plans will be KG-grounded with a 4-agent quality check. Latency is comparable (60-180 s vs. ~90 s for single-pass).

### What about Langfuse traces?

Already wired. Your `_handle_stream_ended` writes a `GenerationTrace` row using `observability_data.trace.trace_id` — the new wrapper does the same. The agent emits a `trace_id` in its `done` event, which flows into your existing `GenerationTrace.trace_id`. Production observability is free on day one.

### What if the agent fails?

The Dramatiq actor's existing `try/except` with 3 retries + `max_backoff=30` applies unchanged. After the retry budget is exhausted, `Conversation.status` flips to `ERROR` and credits are refunded — same code path as today's `TextClient` failures. We provide a structured `error` SSE event with `phase` so your retry logic can be smarter if needed.

---

**Document owner:** LM (AI Team)
