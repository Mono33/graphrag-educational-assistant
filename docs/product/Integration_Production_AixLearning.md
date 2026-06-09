# Production Integration — Agentic GraphRAG ↔ AixLearning Native

**Status:** 📘 Reference / Design *(not yet ticketed)*
**Last Updated:** May 13, 2026 PM
**Owners:** LM *(GraphRAG side)*, FEM platform team *(AixLearning side — TBD coordinator)*
**Scope:** how the agentic GraphRAG FastAPI service (this repo) plugs into the native AixLearning Django suite (`github.com/FEM-modena/aixlearning`, private).

---

## 1. Executive Summary

This document captures the production-integration story for plugging the agentic GraphRAG into the native AixLearning Django app — not into our current `chore/repo-reorg` WebUI, which stays as the dev / admin / smoke-test surface. The integration is API-to-API: AixLearning calls our FastAPI service at `/api/v1/agent/run` and `/api/v1/agent/stream`; we never touch their database or auth backend directly. Three integration shapes (iframe, Jinja-template port, service-only) are formally enumerated in `docs/architecture/Frontend_Platform_Evaluation.md` §6.5 and are all preserved by the architecture we shipped in CORE 1–2. Most of the heavy lifting (JWT shape, `EducationalProfile` schema alignment, public-contract OpenAPI spec, SSE plumbing) is already done. What remains is operational: Hetzner deploy, JWT signing migration from HS256 → RS256 with AixLearning as a trusted issuer, two CORS lines, and a FEM-side decision on whether to iframe, port templates, or build native Django views against our JSON.

---

## 2. Context: Three Repositories, Three Roles

Before going further, it's important to disambiguate the three Git remotes we work with, because the names overlap:

| Remote | URL | Role |
|---|---|---|
| **AixLearning Native** *(integration target)* | `github.com/FEM-modena/aixlearning` *(private)* | The production Django suite — "AI for Learning". Owned by FEM (GMemoli + willywongi). This is what teachers actually log into. **Not our codebase.** |
| **graphrag-aixlearning** *(our shared backend repo)* | `github.com/FEM-modena/graphrag-aixlearning` — remote `fem` | The agentic GraphRAG backend + WebUI we develop in. Our team contributes here; Angelo contributes the media pool / KG-side here too. |
| **Personal fork** | `github.com/Mono33/graphrag-educational-assistant` — remote `origin` | LM's personal mirror of the above. Same commits, no team dependency. |

> **In this document, "we" = `graphrag-aixlearning` repo. "AixLearning" = `FEM-modena/aixlearning` Django app.**

---

## 3. AixLearning Native — Stack Reconstruction *(updated 2026-05-13)*

From direct inspection of the `production` branch of the private repo:

| Property | Value |
|---|---|
| **Languages** | Python 45.6 %, HTML 34.8 %, JavaScript 12.6 %, CSS 6.0 %, Dockerfile 0.2 % |
| **Branches** | 17 (`main`, `production` 51 commits ahead of `main`, plus 15 feature branches) |
| **Commits** | ~1,980 on `production` |
| **Contributors** | GMemoli, willywongi (Francesco Pongiluppi) |
| **Top-level directories** | `.github/workflows/`, `.junie/`, `aixlearning/` *(Django app package)*, `bootstrap/`, `frontend/`, `sample_configs/`, `scheduler/` |
| **Top-level files** | `docker-compose.dev.yaml`, `docker-compose.prod.yaml`, `docker-compose.test.yaml`, `docker-compose.yaml`, `bun.lock`, `index.ts`, `package.json`, `create_palette.py`, `icon_extractor.py` |
| **Description** | *"AI for Learning è una suite di strumenti potenziati da AI al servizio dell'educazione"* |

```text
┌─────────────────────────────────────────────────────────────────┐
│  AixLearning native (github.com/FEM-modena/aixlearning)         │
├─────────────────────────────────────────────────────────────────┤
│  Backend       Django (Python)                                  │
│  Templates     Django templates → server-rendered HTML          │
│  Hypermedia    htmx 2 (form posts + partial swaps)              │
│  Components    WebAwesome (web components, ex-Shoelace)         │
│  Real-time     Mercure hub (SSE + JWT) — "init mercure infra"   │
│  Build         Bun + index.ts (TS/CSS bundling, no React)       │
│  Background    scheduler/ (Celery or APScheduler — TBD)         │
│  Deployment    Docker Compose × {dev, test, prod}               │
│  Domain models party.models.Party, classroom.models.Classroom   │
└─────────────────────────────────────────────────────────────────┘
```

**Operational implications for us:**

1. **Same component library** *(WebAwesome)*. Any HTML we render is visually identical regardless of which template engine produces it.
2. **Same hypermedia philosophy** *(htmx)*. Forms post, partials swap. SSE channels are first-class.
3. **Mercure hub already running in production.** If we want cross-app notifications (*"your lesson is ready"* appears in AixLearning's nav), we publish to the same hub with the shared JWT secret — zero new infra.
4. **Bun + `index.ts` + `package.json`** is their **asset pipeline only**, not a JS framework. The HTML still server-renders from Django templates; Bun just bundles TS/CSS. So integration doesn't require us to ship a JS bundle.

---

## 4. Our Agent's Integration Surface — What FEM Calls

Our FastAPI service (default port 8765, will be behind Caddy/Traefik in production) exposes everything FEM needs to drive the agent from Django:

| Surface | Path | Method | Returns | Auth |
|---|---|---|---|---|
| **Sync agent** | `/api/v1/agent/run` | `POST` | `AgentRunResponse` — `lesson_plan_md` + `meta` + `planner` + `retriever` (60–120 s wall-clock) | cookie OR `Authorization: Bearer <jwt>` |
| **Streaming agent** | `/api/v1/agent/stream` | `POST` | SSE — discriminated-union JSON events: `planner` / `retriever` / `writer_pending` / `writer` / `critic` / `done` / `error` | cookie OR Bearer |
| **GraphRAG context** *(pre-existing)* | `/api/v1/context` | `POST` | Raw KG retrieval (no agent loop). Useful for AixLearning features that want retrieval without a full lesson. | cookie OR Bearer |
| **Health** | `/api/v1/health` | `GET` | `{status: "ok", ...}` for liveness / readiness probes | public |
| **OpenAPI spec** | `/openapi.json` | `GET` | Machine-readable contract — feed to `openapi-generator` or `datamodel-code-generator` to produce a typed Python/Django client | public |
| **Swagger UI** | `/docs` | `GET` | Browsable / "Try it out" UI for the FEM team to explore the contract | public *(in dev — protect in prod)* |
| **ReDoc** | `/redoc` | `GET` | Alternative read-only docs view | public *(in dev)* |

### 4.1 Request shape — `POST /api/v1/agent/run` (and `/stream`)

```json
{
  "query": "Crea una lezione di 45 minuti sulla fotosintesi clorofilliana adattata a una classe con 2 studenti DSA",
  "domain": "neuro",
  "language": "it",
  "session_id": "optional-client-correlation-id",
  "educational_profile": {
    "group": {
      "title": "3A Liceo Scientifico",
      "students_number": 25,
      "grade": "SECONDARIA_II_GRADO",
      "disabilities": ["ADHD", "DSA"],
      "class_features": ["MOTIVATA"],
      "student_attributes": ["PUNTI_DI_ECCELLENZA", "PUNTI_DI_CADUTA"]
    },
    "classroom": {
      "title": "Aula 101",
      "forniture_mobility": "PARTIALLY",
      "has_lim": true,
      "has_wifi": true,
      "has_suite": true,
      "pc_station": false,
      "own_device": "BES"
    },
    "time_available_minutes": 45,
    "subject_area": "Scienze",
    "specific_topic": "Fotosintesi"
  },
  "teacher_provided_context": "Optional plain-text extract from teacher-uploaded files (PDF / TXT / Markdown), up to 48k chars",
  "max_revisions": 2
}
```

**Crucial property:** every field name in `educational_profile` maps **1:1 to AixLearning's domain models** (`party.models.Party.disabilities`, `classroom.models.Classroom.has_lim`, etc.). This is by design — the Pydantic schema was ported from `FEM-modena/graphrag-aixlearning@Angelo`'s `api/schemas/educational_profile.py` precisely so AixLearning can `.model_dump()` a Django object and POST the result with **zero translation layer**.

### 4.2 SSE event taxonomy — what AixLearning's Django view consumes

```text
event: planner          → data: {intent, intent_label, scope, key_concepts, search_queries, ...}
event: retriever        → data: {nodes_count, recommendations_count, media_counts, media, coverage_tier, domain_label_short, ...}
event: writer_pending   → data: {note: "Writer node is generating..."}
event: writer           → data: {revision_idx, lesson_md_partial}        ← will be token-stream after #LAT-1
event: critic           → data: {approved, scores, critique, revision_instructions}
event: done             → data: {duration_seconds, approved, revision_count}, lesson_plan_md: "<final markdown>"
event: error            → data: {}, error: "<message>"
```

Discriminator: `kind` field (also the SSE `event:` name). Clients `switch` on `kind` and update their UI incrementally. The Pydantic models for each variant are exported as a union at `aix.api.schemas.AgentStreamEvent` — feeding `/openapi.json` to a code generator gives the FEM team a typed Django enum to switch on.

### 4.3 Auth model

- **Today (dev / our WebUI):** FastAPI-Users issues HS256 JWTs in an `HttpOnly` cookie *and* accepts the same JWT via `Authorization: Bearer …`.
- **For AixLearning embed (production):** we switch the signing algorithm to **RS256** so AixLearning can issue tokens with **their** private key and we verify them with their **public** key (no shared secret). The JWT claim shape we already accept — `iss`, `sub`, `email`, `domain` — was deliberately chosen to be SSO-ready:
  - `iss="aixlearning"` (their tokens) — added to our `JWT_TRUSTED_ISSUERS` list
  - `iss="graphrag-aixlearning"` (our tokens) — still accepted from our WebUI for backwards compat
- **Cookie domain:** if AixLearning lives at `aixlearning.fem-modena.it` and we live at `graphrag.fem-modena.it`, we set the JWT cookie on `.fem-modena.it` (parent domain) so both apps read it. CORS is then a non-issue for same-origin reads; we still set `allow_origins=["https://aixlearning.fem-modena.it"]` for the cross-subdomain XHR/SSE calls.

---

## 5. The Three Integration Shapes

From `docs/architecture/Frontend_Platform_Evaluation.md` §6.5 — all three are preserved by the architecture; FEM picks whichever fits operational reality at integration time.

### 5.1 Comparison matrix

| Dimension | **Shape 1 — Service + iframe** | **Shape 2 — Service + Jinja → Django port** ⭐ | **Shape 3 — Service-only, AixLearning writes own UI** |
|---|---|---|---|
| **What changes on AixLearning side** | One Django view rendering `<iframe src=".../webui/lesson/{id}">` + parent-domain JWT cookie | New Django app `aixlearning/graphrag/`. Port `src/aix/webui/templates/**/*.html` (Jinja2 → Django syntax). ~3-5 days. | Write fresh Django templates from scratch using FEM's UX guidelines; consume only `/api/v1/agent/*` JSON+SSE. |
| **What changes on GraphRAG side** | 2 config lines: trust AixLearning JWT issuer; allow AixLearning origin in CORS | 2 config lines (same as Shape 1). Optionally retire `src/aix/webui/` once Django version live. | 2 config lines (same as Shape 1). `src/aix/webui/` stays as internal/admin/dev UI. |
| **Effort (calendar)** | Smallest (~1 day end-to-end) | Medium (~3-5 days for the port + QA) | Largest (FEM owns design + impl — 1-2 weeks depending on scope) |
| **UX quality** | Acceptable — but iframe is visible (separate scrollbar, no native nav inside it) | **Indistinguishable** from native AixLearning | Whatever the AixLearning UX team produces |
| **Ownership boundary** | Two apps loosely coupled by JWT | One Django app + one FastAPI service — clean SOA | Two teams with a JSON contract |
| **Velocity after integration** | Frontend changes ship from our repo; teacher sees them immediately | Frontend changes need a Jinja → Django port re-pass (or stay in our repo and we re-port) | Frontend changes are owned by FEM; we ship API changes |
| **When to pick** | FEM is busy and wants "appears inside" fast | Polished native UX + FEM has bandwidth | FEM has strong UX opinions and wants full surface ownership |

### 5.2 Why Shape 2 is the strategic target

The whole reason we built our WebUI on Path C (FastAPI + Jinja2 + htmx + WebAwesome + Tailwind) was to make Shape 2 cheap. Jinja2 → Django template syntax is **~90 % overlap**: `{% if %} / {% for %} / {% block %}` carry over verbatim; `{{ x.attr }}` works; component invocations (`<wa-button>`, `<wa-input>`, `<wa-dialog>`) are HTML, not templating, so they need zero translation. The 10 % delta is:

| Jinja2 | Django template | Translation effort |
|---|---|---|
| `{{ url_for('view', id=x) }}` | `{% url 'view' id=x %}` | mechanical find-replace |
| `{{ x \| default('') }}` | `{{ x \| default_if_none:"" }}` | mechanical (a few filter renames) |
| Macro `{% macro foo(x) %}` | `{% include "foo.html" with x=x %}` | mechanical or `inclusion_tag` |
| Inline Python expressions | Django template tags or context processors | sometimes requires a tiny custom tag |
| `{% extends %}` | identical | no change |
| WebAwesome tags `<wa-*>` | identical | no change |
| htmx attrs `hx-*` | identical | no change |
| Tailwind classes | identical | no change |

For a sense of scope: the 4 P5-re-skinned pages we shipped (Dashboard, Library, Create Lesson, Workspace) are ~1,500 lines of Jinja total. A 3-5 day port is a realistic estimate.

### 5.3 Shape 3 is the most decoupled

If the FEM platform team wants **full UI ownership** (their own design system, their own routing, their own asset pipeline already powered by Bun), Shape 3 is the cleanest. They consume `/api/v1/agent/run` and `/api/v1/agent/stream`, render whatever Django templates fit AixLearning's existing UX, and our WebUI just stays around as our internal admin/dev surface (where we test agent changes before they hit the FEM frontend).

**Shape 3 is the most aligned with the framing of *"AixLearning calls our FastAPI for the agentic part"*** — which is how this question was posed.

### 5.4 Mixing shapes

Shapes 1 → 2 → 3 are **not mutually exclusive over time**. A realistic phased path:

1. **Week 0:** Shape 1 (iframe) — teachers see the agent inside AixLearning today, MVP-quality.
2. **Weeks 1-2:** Shape 2 (template port) — the iframe disappears, the agent feels native.
3. **Months later:** if FEM wants to redesign the teacher-facing surface, they switch to Shape 3 — they own the Django templates, we own the API. No breaking change at the contract boundary.

This sequencing is supported by the OpenAPI strictly-additive regression test (`test_openapi_inventory_strictly_additive`) which prevents any future path rename from breaking FEM's consumer.

---

## 6. Phase A — Quick Bridge (Shape 1, ~1-2 calendar days)

Goal: make the agent visible inside AixLearning **this week**, with minimum FEM-team coordination.

### 6.1 What we ship

| Task | File(s) | Effort | Owner |
|---|---|---|---|
| Deploy `graphaixlearning` to Hetzner via Docker Compose (Postgres swap, Caddy TLS) — this is CORE 2 #6.6 P6 | `docker-compose.prod.yaml`, `Caddyfile`, `requirements.txt` | ~1 day | LM |
| Switch FastAPI-Users JWT signing from HS256 → RS256 | `src/aix/webui/auth/__init__.py`, `.env`, `.env.example` | ~2 h | LM |
| Add AixLearning as a trusted issuer | same as above | ~30 min | LM |
| CORS: allow `https://aixlearning.fem-modena.it` (the AixLearning origin) | `src/aix/api/main.py` | ~10 min | LM |
| Cookie domain: set to `.fem-modena.it` (parent) so both apps read the JWT | same | ~10 min | LM |
| Health check / readiness probe documentation for FEM ops | `docs/runbooks/Production_Readiness_Probes.md` *(new)* | ~30 min | LM |

### 6.2 What FEM ships

| Task | File(s) on AixLearning side | Effort | Owner |
|---|---|---|---|
| Add a Django view at `aixlearning.fem-modena.it/lessons/new/ai` that issues an RS256 JWT for the logged-in user and renders an iframe pointing at `https://graphrag.fem-modena.it/webui/lesson/new?token=<jwt>` | new view + template | ~2-4 h | FEM |
| Confirm parent-domain cookie is readable from both subdomains | infra config | ~1 h | FEM |

### 6.3 Verification checklist

- [ ] Teacher logs into AixLearning at `https://aixlearning.fem-modena.it` → cookie set on `.fem-modena.it`.
- [ ] Teacher clicks "Crea lezione con AI" → Django view renders iframe to our `/webui/lesson/new`.
- [ ] Our FastAPI accepts the JWT (RS256, `iss=aixlearning`) and renders the lesson form.
- [ ] Teacher fills the educational profile → submits → sees the chat workspace stream the agent.
- [ ] `done` event fires → final lesson appears.
- [ ] If teacher closes the iframe and re-opens it, they see their lesson history (our `Lesson` table is keyed by JWT `sub`).

### 6.4 Known iframe limitations (acceptable for Phase A)

- Different scrollbar inside the iframe.
- No native AixLearning nav inside the iframe.
- "Open in new tab" lands on our `/webui/`, not on AixLearning.
- Browser back-button navigates the iframe, not the parent — may confuse teachers.

These are the exact reasons Phase B (Shape 2 or Shape 3) exists.

---

## 7. Phase B — Native Integration (Shape 2 ⭐ recommended, ~5 days)

Goal: replace the iframe with native Django views so the agent feels **indistinguishable** from the rest of AixLearning.

### 7.1 What FEM ships

| Task | File(s) on AixLearning side | Effort |
|---|---|---|
| Create new Django app `aixlearning/graphrag/` (`views.py`, `urls.py`, `models.py`, `forms.py`) | new dir | ~1 day |
| Port Jinja templates (~1,500 lines, 4 re-skinned pages + 5 partials) | `aixlearning/graphrag/templates/` | ~2 days |
| Replace `aix.webui.agent.service.run_agent_stream` calls with `httpx.AsyncClient.stream("POST", "http://graphrag-api:8765/api/v1/agent/stream", ...)` over the internal Docker network | `aixlearning/graphrag/views.py` | ~0.5 day |
| Route lesson-ready notifications through AixLearning's existing Mercure hub | `aixlearning/graphrag/views.py` + Mercure publish call | ~0.5 day |
| QA: ensure WebAwesome components render identically (low risk — same library) | both | ~0.5 day |
| Documentation: integration runbook in AixLearning's docs | `docs/agents.md` *(AixLearning side)* | ~0.5 day |

### 7.2 What we ship

| Task | File(s) | Effort |
|---|---|---|
| Land #LAT-1 (Writer Token Streaming) **before** the Phase B cutover so teachers don't see 60-100 s blank-card waits in production | `src/aix/webui/agent/service.py`, `src/aix/agent/graph/...` | ~3-5 h *(per CORE 3 C #LAT-1 estimate)* |
| Land #LAT-7 (Single-Pass Retriever Efficiency) for the same reason | retriever modules | ~3-4 h |
| Optional: retire `src/aix/webui/templates/` or freeze it as the dev/admin UI | doc note | ~0 |

### 7.3 Routing example — what the Django view looks like

```python
# aixlearning/graphrag/views.py (FEM side, illustrative)
from django.http import StreamingHttpResponse
from django.shortcuts import render
import httpx

GRAPHRAG_BASE = "http://graphrag-api:8765"   # internal Docker network

async def lesson_new(request):
    profile = build_educational_profile(request.user)  # ← reuses party.models / classroom.models
    return render(request, "graphrag/lesson_new.html", {"profile": profile})

async def lesson_stream(request, lesson_id):
    """Proxy our /api/v1/agent/stream to the htmx-listening template."""
    payload = build_payload_from_form(request)
    jwt = issue_internal_jwt(request.user)

    async def _proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{GRAPHRAG_BASE}/api/v1/agent/stream",
                json=payload,
                headers={"Authorization": f"Bearer {jwt}"},
            ) as r:
                async for line in r.aiter_lines():
                    yield line + "\n"

    return StreamingHttpResponse(_proxy(), content_type="text/event-stream")
```

The Django template then uses `htmx`'s SSE extension exactly as our Jinja template does:

```html
<div hx-ext="sse" sse-connect="{% url 'graphrag:lesson_stream' lesson.id %}">
  <div sse-swap="planner"   hx-target="#planner-card"></div>
  <div sse-swap="retriever" hx-target="#retriever-card"></div>
  <div sse-swap="writer"    hx-target="#writer-card"></div>
  ...
</div>
```

Identical paradigm to ours; only the template-engine syntax differs.

---

## 8. Phase B-alternative — Shape 3 (Service-only, FEM owns UI)

If FEM prefers full UI ownership rather than a template port:

### 8.1 What FEM ships

| Task | Effort |
|---|---|
| Generate Python client from `/openapi.json` with `datamodel-code-generator` or `openapi-python-client` | ~1 h |
| Design fresh Django templates fitting AixLearning's UX | ~3-5 days |
| Implement the same `httpx.stream(...)` proxy as Shape 2 §7.3 | ~0.5 day |
| Wire htmx SSE with whatever event vocabulary FEM prefers (they can rename events client-side) | ~0.5 day |

### 8.2 What we ship

| Task | Effort |
|---|---|
| Same JWT/CORS/networking as Phase A | already done after Phase A |
| Commit to maintaining `/api/v1/agent/*` as the **only** contract — `/webui/*` is no longer a public surface | doc-only |

### 8.3 Trade-off vs Shape 2

- **+ Cleanest separation:** FEM owns design end-to-end. We never block on their UX team.
- **+ AixLearning consistency:** the agent surface matches the rest of AixLearning automatically because it *is* the rest of AixLearning.
- **− Higher FEM effort:** ~5-10 days vs ~3-5 days for the template port.
- **− Loses our four P5-reskinned pages as a starting point:** Dashboard, Library, Create Lesson, Workspace — all the careful warm-academic UX work — gets re-done from scratch.

Recommendation: **Shape 2 if FEM has bandwidth in the next 2 weeks; Shape 3 if they don't and prefer to schedule the UX work on their own roadmap.**

---

## 9. Network & Security Configuration

### 9.1 JWT migration HS256 → RS256

Today:

```python
# src/aix/webui/auth/__init__.py (current, HS256)
SECRET = os.getenv("WEBUI_JWT_SECRET", "dev-secret-change-me")
JWT_STRATEGY = JWTStrategy(secret=SECRET, lifetime_seconds=3600, algorithm="HS256")
```

Production:

```python
# new — RS256 + multi-issuer
PRIVATE_KEY = os.getenv("WEBUI_JWT_PRIVATE_KEY_PEM")        # ours (signs our cookie tokens)
PUBLIC_KEY_OURS = os.getenv("WEBUI_JWT_PUBLIC_KEY_PEM")     # verifies our tokens
PUBLIC_KEY_AIXLEARNING = os.getenv("AIXLEARNING_JWT_PUBLIC_KEY_PEM")   # verifies their tokens
TRUSTED_ISSUERS = {
    "graphrag-aixlearning": PUBLIC_KEY_OURS,
    "aixlearning":          PUBLIC_KEY_AIXLEARNING,
}

class MultiIssuerJWTStrategy(JWTStrategy):
    async def read_token(self, token: str | None, user_manager) -> models.UP | None:
        if not token:
            return None
        unverified = jwt.get_unverified_claims(token)
        issuer = unverified.get("iss")
        public_key = TRUSTED_ISSUERS.get(issuer)
        if public_key is None:
            return None
        try:
            claims = jwt.decode(token, public_key, algorithms=["RS256"], issuer=issuer)
        except jwt.InvalidTokenError:
            return None
        return await user_manager.get_by_email(claims["email"])
```

Required `.env` keys (`.env.example` will document them):

```bash
WEBUI_JWT_PRIVATE_KEY_PEM=<base64 RSA private key>
WEBUI_JWT_PUBLIC_KEY_PEM=<base64 RSA public key>
AIXLEARNING_JWT_PUBLIC_KEY_PEM=<base64 RSA public key handed off by FEM>
```

### 9.2 CORS

```python
# src/aix/api/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aixlearning.fem-modena.it",       # production AixLearning
        "https://graphrag.fem-modena.it",          # our WebUI
        "http://127.0.0.1:8765",                   # dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```

### 9.3 Parent-domain cookie (Shape 1 only)

```python
# src/aix/webui/auth/__init__.py
COOKIE_DOMAIN = ".fem-modena.it"           # leading dot = both subdomains read it
COOKIE_SECURE = True                       # HTTPS only
COOKIE_SAMESITE = "lax"                    # allow top-level navigation; XHR same-site
```

Shapes 2 and 3 don't need a parent-domain cookie because the entire request stays within AixLearning's origin — the Django view server-side does an `httpx` call to our internal Docker hostname.

### 9.4 Internal Docker network (Shape 2 / Shape 3)

```yaml
# docker-compose.prod.yaml — illustrative slice showing the two-service deployment
services:
  graphrag-api:
    image: ghcr.io/fem-modena/graphrag-aixlearning:latest
    networks:
      - fem-internal
    environment:
      - WEBUI_JWT_PRIVATE_KEY_PEM=${WEBUI_JWT_PRIVATE_KEY_PEM}
      - AIXLEARNING_JWT_PUBLIC_KEY_PEM=${AIXLEARNING_JWT_PUBLIC_KEY_PEM}
      - DATABASE_URL=postgresql://...
    # no ports exposed publicly — only Caddy reaches it from outside

  aixlearning:
    image: ghcr.io/fem-modena/aixlearning:latest
    networks:
      - fem-internal
    environment:
      - GRAPHRAG_API_BASE=http://graphrag-api:8765
    # Django's view does httpx.AsyncClient.stream("POST", f"{GRAPHRAG_API_BASE}/api/v1/agent/stream", ...)

  caddy:
    image: caddy:2
    ports:
      - "80:80"
      - "443:443"
    networks:
      - fem-internal
      - public
    # Caddyfile routes aixlearning.fem-modena.it → aixlearning:8000
    #                graphrag.fem-modena.it    → graphrag-api:8765/webui   (Shape 1 only)
```

Notable: in Shape 2 and Shape 3, the GraphRAG API is **not publicly exposed** — only AixLearning can talk to it from inside the Docker network. This is dramatically safer than Shape 1 (where the WebUI must be public for the iframe to load).

---

## 10. Open Dependencies Before Production Go-Live

Status as of 2026-05-13 PM, after `cb3e31b` (#9.UX-5):

| Dependency | CORE ticket | Status today | Blocker for which shape? |
|---|---|---|---|
| **Hetzner deploy + Postgres migration** | CORE 2 #6.6 P6 | 🔴 TODO *(~1 day)* | **🔴 Hard blocker for all 3 shapes** |
| **JWT RS256 + multi-issuer** | new — derived from CORE 2 #7 | 🔴 TODO *(~2 h)* | **🔴 Hard blocker for all 3 shapes** |
| **CORS + cookie-domain alignment** | new | 🔴 TODO *(~30 min)* | **🔴 Hard blocker for Shape 1**; not needed for Shapes 2/3 |
| **Input/Output Guardrails** | CORE 2 #8 | 🔴 TODO *(3-5 h)* | 🟠 Strongly recommended before any teacher traffic |
| **P5.5 Final Lesson View re-skin** | CORE 2 #6.6 P5.5 | 🔴 TODO | 🟠 Want for Shape 1 / Shape 2 *(without it the final lesson page has placeholders)* |
| **P5.6 Settings page re-skin** | CORE 2 #6.6 P5.6 | 🔴 TODO | 🟡 Nice-to-have |
| **#LAT-1 Writer Token Streaming** | CORE 3 C #LAT-1 | 🔴 TODO *(3-5 h)* | 🟠 **Strongly recommended** — without it teachers see 60-100 s of blank-card waiting on every lesson |
| **#LAT-7 Single-Pass Retriever Efficiency** | CORE 3 C #LAT-7 | 🔴 TODO *(3-4 h)* | 🟠 Reduces single-pass retrieve from ~102 s → ~55-65 s |
| **#11b Langfuse tracing dashboard** | CORE 2 #11 phase b | 🟡 foundation done by AG | 🟡 Not a blocker, but you'll want it live so production traffic is observable from day 1 |
| **#9.UX-4 B1 + B2** *(CR-ON outcome correctness)* | CORE 2 #9.UX-4 | 🟡 deferred | 🟢 Not a blocker if `AIX_CORRECTIVE_RAG_ENABLED=false` (production default) |

**Critical path to "AixLearning can call our API in production":**
1. Hetzner deploy *(1 day)*
2. JWT RS256 + CORS *(2.5 h)*
3. Pick a shape *(coordination call with FEM)*

Everything else is "should ship before public teacher traffic" but isn't strictly required for the wiring to work.

---

## 11. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FEM has no bandwidth for the template port (Shape 2) | Medium | Medium — pushes us to Shape 1 (acceptable) or Shape 3 (more FEM effort) | Document Shape 1 as a permanent option; teachers see the agent regardless |
| AixLearning's RS256 public key isn't readily exportable | Low | Low — fallback to a shared secret for HS256 across both apps | Coordinate with FEM ops; both apps support either algorithm |
| Production latency complaints from teachers (60-100 s blank Writer card) | **High** *(without #LAT-1)* | Medium — usability issue, not correctness | Land #LAT-1 *before* opening Phase B to real traffic |
| Mercure hub on AixLearning side blocks GraphRAG publishes | Low | Low — degrade gracefully (notifications optional, not required for lesson generation) | Wrap publish call in a `try / except`; log failures, don't block the request |
| OpenAPI contract drift between our spec and FEM's code-gen client | Medium | Low — caught at integration test time | `test_openapi_inventory_strictly_additive` already prevents path renames; new fields are always additive |
| Educational profile schema diverges between Django models and our Pydantic | **Low** *(historically aligned)* | High — every request would fail validation | Add a periodic Django ↔ Pydantic schema parity test on AixLearning side *(future #22)* |
| `AIX_CORRECTIVE_RAG_ENABLED=true` gets re-enabled before #LAT-7 lands → 200 s+ false-positive retries | Medium | High — production outage equivalent | Keep `AIX_CORRECTIVE_RAG_ENABLED=false` until #LAT-7 + R1 grader-input fix ship together |

---

## 12. Recommended Sequencing

Calendar-week view assuming start-of-week kick-off and FEM team is responsive at the integration touchpoints:

| Week | Our side | FEM side |
|---|---|---|
| **Week 1** | CORE 2 #6.6 P6 — Hetzner deploy + Postgres migration | Decide on Shape 1 vs Shape 2 vs Shape 3 (this doc is the input) |
| **Week 1** | JWT RS256 migration + CORS + cookie domain *(2.5 h)* | Generate RS256 keypair on AixLearning side, hand off public key |
| **Week 2** | If Shape 1: ready to go live | If Shape 1: add Django iframe view (~half a day) → **MVP ships end of Week 2** |
| **Week 2-3** | Land #LAT-1 Writer Token Streaming + #LAT-7 retriever efficiency *(6-9 h total)* | If Shape 2: start Django app + template port |
| **Week 3-4** | Land CORE 2 #8 Guardrails + #11b Langfuse dashboard | If Shape 2: finish template port + cutover from iframe → native Django views |
| **Week 4+** | Continue P5.5 + P5.6 page re-skins; refresh as Shape 2 ports the Jinja → Django mappings | Decommission iframe; agent now feels native |

**Total to "Shape 1 MVP in production":** ~2 calendar weeks.
**Total to "Shape 2 native in production":** ~4 calendar weeks.
**Total to "Shape 3 native with FEM-owned UI":** depends on FEM scheduling — could be 6-10 weeks.

---

## 13. Appendix — Quick-Reference Contract

| Item | Where |
|---|---|
| Swagger UI *(browse + Try it out)* | `https://graphrag.fem-modena.it/docs` |
| ReDoc *(read-only)* | `https://graphrag.fem-modena.it/redoc` |
| OpenAPI spec *(for code-gen)* | `https://graphrag.fem-modena.it/openapi.json` |
| Health check | `GET /api/v1/health` |
| Sync agent | `POST /api/v1/agent/run` |
| Streaming agent | `POST /api/v1/agent/stream` |
| Raw GraphRAG context *(no agent)* | `POST /api/v1/context` |
| Pydantic request model | `aix.api.schemas.AgentRunRequest` |
| Pydantic sync response | `aix.api.schemas.AgentRunResponse` |
| Pydantic SSE event union | `aix.api.schemas.AgentStreamEvent` |
| Educational profile schema *(canonical, 1:1 with `party.models` + `classroom.models`)* | `aix.api.schemas.educational_profile.EducationalProfile` |

### 13.1 Authoritative documents

- **Architecture (this repo):** `docs/architecture/Frontend_Platform_Evaluation.md` *(ADR-0001, integration shape matrix)*
- **Roadmap (this repo):** `docs/product/ClickUp_Agentic_GraphRAG_Update.md` *(CORE 2 #7, CORE 6 P7)*
- **Auth contract (this repo):** `src/aix/webui/auth/__init__.py` *(FastAPI-Users, JWT shape)*
- **Public API (this repo):** `src/aix/api/routes/agent.py` *(routes)*, `src/aix/api/schemas/agent.py` *(schemas)*

### 13.2 Coordination touchpoints with FEM

| What we need from FEM | When | Format |
|---|---|---|
| RS256 public key for `iss="aixlearning"` JWTs | Before Phase A go-live | PEM string, env-injected |
| Confirmation of cookie domain `.fem-modena.it` (or alternate subdomain plan) | Before Phase A go-live | email / Slack |
| Mercure hub publish endpoint + shared JWT secret *(optional, for cross-app notifications)* | Before Phase B native cutover | env-injected URL + secret |
| Shape decision *(1 / 2 / 3)* | Week 1 | meeting + ADR addendum |
| Designated FEM coordinator *(GMemoli or willywongi)* | Week 1 | email / Slack |
| Pre-production AixLearning hostname *(staging)* | Week 2 | DNS entry |

---

**Document owner:** LM
**Reviewers:** GMemoli, willywongi *(FEM AixLearning native team)*; Diego, Simone, Filippo *(FEM strategic, see Frontend_Platform_Evaluation.md §9)*

