# Frontend Platform Evaluation & Architecture Decision

> **Status:** Draft for review (CORE 2 Subtask #6.5 — Research Spike)
> **Author:** LM
> **Date:** 2026-04-26
> **Decision blockers:** CORE 2 #7 (FastAPI Agent Endpoint), #8 (Guardrails), #12 (SSE Streaming), CORE 6 (Deployment).
> Until this ADR lands the API contract is unsafe to fix, because auth scheme, CORS, streaming protocol, payload shape, session model, and multi-tenancy all depend on the chosen frontend.

---

## 1. Executive Summary

After investigating the AixLearning native repository and benchmarking the modern alternatives, the recommendation is **NOT** the working hypothesis (Vercel + Next.js + Vercel AI SDK) and **NOT** an immediate native embed. Instead:

> **Primary recommendation (Path C — "Mirror Stack"):** Build the Agentic GraphRAG frontend on **FastAPI + htmx 2 + WebAwesome + Tailwind CSS + `sse-starlette`**, deployed via Docker Compose on the same infrastructure profile as the main AixLearning platform.
>
> **Fallback / phase 2:** Migrate the templates into the AixLearning Django app as a native section (Path B) once the product is stable and the FEM platform team has bandwidth.
>
> **Working hypothesis (Path A — Vercel + Next.js):** Defer. Keep it on the table only if a separate B2C consumer brand emerges that *cannot* live inside FEM-modena infrastructure.

Why: AixLearning is *already* a server-rendered hypermedia app (Python + htmx + WebAwesome + Mercure SSE + Bun build). A "Mirror Stack" frontend matches its philosophy 1:1, which means: same component library, same skill set, same Docker deployment story, **and an embed path that is essentially "copy the templates into the Django app"** instead of a rewrite.

---

## 2. Context & Constraints

| Constraint | Detail |
|------------|--------|
| **Backend already chosen** | FastAPI (Python) — `src/aix/api/main.py`, GraphRAG + Agentic LangGraph endpoints. Not negotiable. |
| **Domain expertise** | Italian K-12 special-needs education (BES/DSA, ADHD, UDL). UI must be Italian-first. |
| **Streaming required** | Writer agent emits tokens; Critic emits revision events; tools emit progress. SSE > WebSocket is the right primitive. |
| **Auth model** | Teacher accounts; per-request `EducationalProfile` (CORE 1 #2.5) maps 1:1 to FEM `party.models` and `classroom.models`. |
| **Future embed** | The product is intended to live inside the main FEM platform eventually; the technical decisions today should not punish that path. |
| **File upload** | Teachers will upload lesson PDFs / class rosters. Multipart support required. |
| **Team size** | Small. AG and LM are the primary developers; Simone owns frontend UX direction. |
| **Cost budget** | Hetzner / Coolify / VPS profile. Not Vercel-Pro pricing. |
| **Time pressure** | CORE 2 needs to ship in ~weeks, not quarters. |

---

## 3. AixLearning Native — Stack Reconstruction

The repo at `github.com/FEM-modena/aixlearning` is private, but the structure visible in the screenshot, combined with branch / commit messages and public skill files, lets us reconstruct the stack with high confidence.

### 3.1 Evidence

| Signal | What it tells us |
|--------|------------------|
| **Languages bar:** Python 45.5 %, HTML 34.9 %, JS 12.6 %, CSS 6.8 % | Server-rendered Python web app, *not* a SPA. The HTML share is templates. |
| `pyproject.toml` at the root + `aixlearning/` package folder | Modern Python project, almost certainly **Django** (the package layout, the template-heavy structure, and the FEM platform descriptions all match Django). |
| `bootstrap/`, `frontend/`, `index.ts`, `package.json`, `bun.lock` | **Bun**-based asset pipeline (TypeScript / CSS bundling), *not* a Webpack / Next.js app. |
| `.aiassistant/rules/skills/webawesome/` | Team uses **WebAwesome** (the Shoelace successor) as the component library. |
| Commit *"fix bump up htmx to fix checkboxes aria-checked attribute"* | **htmx 2** is the hypermedia layer — that's how the UI gets its interactivity without a JS framework. |
| Commits *"init mercure infra"*, *"add mercure to docker"*, *"add mercure to docker-compose"* | Real-time updates are pushed via the **Mercure** hub (SSE + JWT), not WebSockets. |
| `docker-compose.{dev,prod,test,yaml}` | Docker Compose multi-environment deployment; almost certainly self-hosted (Hetzner / FEM infra), not Vercel-style PaaS. |
| `scheduler/`, `create_palette.py`, `icon_extractor.py` | Background workers + design-system tooling — points to a mature platform, not an MVP. |
| Branches *"feature/credit-changes"*, *"staging"*, "126 commits ahead of main" | Active multi-team development on a long-lived `staging` branch. |

### 3.2 Stack summary (high-confidence reconstruction)

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
│  Background    scheduler/ (likely Celery or APScheduler)        │
│  Deployment    Docker Compose × {dev, test, prod}               │
│  Modeling      party.models.Party, classroom.models.Classroom   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 What this means for the GraphRAG frontend

1. The team's **mental model** is hypermedia, not SPA. A Next.js app would be a foreign body that nobody on the FEM side can comfortably maintain.
2. WebAwesome is the **shared design language**. If GraphRAG uses the same component set, the UI looks like AixLearning by default — no separate design system to fund.
3. Mercure is **already running in production** — GraphRAG can publish to the same hub for free if streaming needs to be visible across both apps later.
4. Embedding GraphRAG into AixLearning eventually means **moving Jinja templates into Django templates** (a syntactic rewrite, not an architectural one) instead of iframing a Next.js bundle.

---

## 4. The Three Candidates

### Path A — Vercel + Next.js + Vercel AI SDK *(working hypothesis)*

**Stack:** Next.js 15 / App Router, Vercel AI SDK v6, `useChat`, `@ai-sdk/langchain` adapter, Auth.js, Tailwind, deployed via `vercel deploy`. FastAPI backend stays put; the Next.js app calls it (or uses Vercel functions as a thin proxy).

**Strengths**
- Best-in-class streaming UX out of the box (`useChat`, `addToolApprovalResponse`, `streamObject`, native SSE protocol).
- Largest ecosystem of LangChain / LangGraph JavaScript bindings (matters if you ever rewrite agents in TS).
- `vercel deploy` gives zero-config preview environments for Lovable-style iteration.
- Edge runtime keeps p99 latency low for a B2C audience.

**Weaknesses (in this context)**
- **Mismatch with AixLearning:** Next.js is the opposite paradigm (client-rendered, SPA, Node toolchain). Future embed becomes a rewrite, not a migration.
- **Two stacks to maintain forever:** Python on the API + TypeScript on the frontend = duplicated validation, duplicated types, duplicated build pipelines.
- **Cost ceiling:** Vercel Pro is ~$20/user/mo plus function/bandwidth. Hetzner costs ~€5/mo total. For a small Italian K-12 product the math doesn't work.
- **Hidden complexity:** Auth.js + CORS to FastAPI + token refresh + SSE proxying through Vercel functions = lots of glue code.
- **No-one on FEM side knows it:** Skill mismatch with the AixLearning platform team.

**Verdict:** Powerful, but optimised for a problem we don't have (massive consumer scale, a JS-native team, a brand-new design system).

---

### Path B — AixLearning Native Embed *(eventual goal, not "now")*

**Stack:** Implement the GraphRAG UI as a Django app inside the existing AixLearning monorepo. Reuse `party.models`, `classroom.models`, the htmx layer, the WebAwesome components, and the Mercure hub. Hit the FastAPI service over the internal Docker network.

**Strengths**
- **Single sign-on for free:** Teacher already logged into AixLearning → already logged in here.
- **`EducationalProfile` is literally the same Python objects** — no schema translation, no DTOs.
- **Operational unity:** one deploy, one monitoring stack, one billing model, one support channel.
- **UI consistency by construction:** the design language is whatever AixLearning is using today.
- **Lowest long-term TCO:** marginal hosting cost is zero.

**Weaknesses (right now)**
- **Couples your release cadence to the AixLearning platform team.** Their `staging` is 126 commits ahead of `main` — they ship on their own schedule. You can't ship every day until your changes are reviewed and merged by a team that has its own roadmap.
- **Requires deep familiarity with their app:** file layout, settings, custom auth, design tokens, Mercure topics, scheduler conventions. None of that is documented for an outside contributor yet.
- **Risk of being blocked:** any blocker on the AixLearning side (unrelated migration, conflicting branch, infra change) blocks GraphRAG too.
- **Technical proof not yet established:** the agent loop, the streaming UI, the tool-approval flow — none of these have been demonstrated end-to-end inside the Django app.

**Verdict:** This is the *destination*. It's the wrong *starting point* — we don't yet know what we want to embed.

---

### Path C — "Mirror Stack": FastAPI + htmx + WebAwesome + Tailwind ⭐ *recommended*

**Stack:**

```text
┌─────────────────────────────────────────────────────────────────┐
│  graphaixlearning frontend (recommended)                         │
├─────────────────────────────────────────────────────────────────┤
│  Backend       FastAPI (already exists — src/aix/api/main.py)   │
│  Templates     Jinja2 → server-rendered HTML fragments           │
│  Hypermedia    htmx 2 (hx-post / hx-swap / hx-ext="sse")        │
│  Components    WebAwesome (same library AixLearning uses)        │
│  Styling       Tailwind CSS (utilities) + WebAwesome themes     │
│  Light JS      Alpine.js for tiny client-state islands          │
│  Streaming     sse-starlette + FastAPI StreamingResponse        │
│  Auth          FastAPI-Users (JWT) — JWT shape compatible       │
│                with AixLearning so SSO can be wired later       │
│  Real-time     Optional: publish to AixLearning's Mercure hub   │
│  Build         Tailwind CLI; no Node bundler required           │
│  Deployment    Docker Compose on Hetzner / Coolify              │
└─────────────────────────────────────────────────────────────────┘
```

**Strengths**
- **Same paradigm as AixLearning** → embedding later means *moving Jinja templates into Django templates* (a syntax port), not a rewrite. Component selectors stay identical because both apps use the same WebAwesome elements (`<wa-button>`, `<wa-input>`, `<wa-dialog>`, etc.).
- **One language across the whole stack** — Python backend, Python templates, Python tests. The whole team can read every file.
- **Streaming is trivial:** FastAPI `StreamingResponse` + `hx-ext="sse"` is the canonical pattern in 2026. Tool-approval prompts and Critic revision events become HTML fragments swapped into the page.
- **Zero JS build pipeline** by default. Tailwind has a 100 KB CLI; that's it. We can add Bun later if we ever want to mirror AixLearning's bundling.
- **Hosting cost = €5–10/mo** on a single Hetzner CX22 + a managed Postgres.
- **Same component library as AixLearning** → visual consistency for free, no design-system divergence.
- **No vendor lock-in:** every piece is OSS. Pick up and move whenever you want.
- **Future-proof:** if a SPA frontend is ever needed, the FastAPI JSON endpoints are unchanged — Next.js can be added *on top* (B2C brand) without replacing anything.

**Weaknesses**
- **htmx ergonomics:** less "magical" than React. Some animations / drag-and-drop interactions are easier in React. Mitigation: 99 % of the agent UI is forms + streaming text + structured cards, all of which htmx handles natively.
- **No first-class chat-state library:** there's no `useChat` equivalent. We have to manage the conversation state on the server (which we want anyway, to persist sessions in Postgres / Redis).
- **Less off-the-shelf demo polish:** no slick "Vercel template" you can clone in 30 seconds. We'll write the templates ourselves — but that's a 1-2 week investment, not a 2-month one.

**Verdict:** Same operational profile as AixLearning, same skill set, same component library, same deployment story, ~10× cheaper than Vercel, and it positions us perfectly for the eventual embed. This is the **lowest-regret choice** by every metric that matters here.

---

## 5. Comparison Matrix

Scoring: 1 (poor) to 5 (excellent), in this product context.

| # | Criterion | A: Vercel + Next.js | B: AixLearning embed (now) | C: Mirror Stack ⭐ |
|---|-----------|:---:|:---:|:---:|
| 1 | **Streaming UX (SSE) maturity** | 5 | 4 | 4 |
| 2 | **Auth integration with FEM platform** | 2 | 5 | 4 |
| 3 | **Time-to-first-deploy** | 3 | 1 | 5 |
| 4 | **Customisation ceiling** | 5 | 4 | 4 |
| 5 | **Vendor lock-in (lower is better)** | 2 | 5 | 5 |
| 6 | **Total Cost of Ownership (12 mo)** | 2 | 5 | 5 |
| 7 | **`EducationalProfile` integration** | 3 | 5 | 4 |
| 8 | **AixLearning visual / paradigm consistency** | 1 | 5 | 5 |
| 9 | **File upload / multipart support** | 5 | 5 | 5 |
| 10 | **Skill match with current team** | 2 | 3 | 5 |
| 11 | **Future-embed cost (delta from "now" choice)** | 1 | 0 (already there) | 4 |
| 12 | **Independence from AixLearning release cadence** | 5 | 1 | 5 |
| | **Total (out of 60)** | **36** | **43** | **55** |

Interpretation:
- **Path A** is dragged down by skill mismatch, paradigm mismatch, and TCO.
- **Path B** is the strongest *long-term* answer but loses badly on time-to-first-deploy and release independence — both of which matter *now* during product discovery.
- **Path C** scores highest because it is the only option that is *both* independent of the AixLearning release cadence today *and* a low-cost migration into AixLearning later.

---

## 6. Recommended Architecture (Path C)

### 6.1 Repository layout (additions only)

```text
graphaixlearning/
├── src/aix/
│   ├── api/                       (existing FastAPI)
│   ├── webui/                     ← NEW
│   │   ├── __init__.py
│   │   ├── routes.py              (HTML routes, return Jinja fragments)
│   │   ├── deps.py                (auth, current_user, EducationalProfile)
│   │   ├── streaming.py           (sse-starlette helpers)
│   │   └── templates/
│   │       ├── _base.html         (loads WebAwesome, Tailwind, htmx, Alpine)
│   │       ├── partials/
│   │       │   ├── chat_message.html
│   │       │   ├── tool_event.html
│   │       │   ├── critic_revision.html
│   │       │   └── lesson_card.html
│   │       ├── pages/
│   │       │   ├── home.html
│   │       │   ├── new_lesson.html
│   │       │   └── lesson_detail.html
│   │       └── forms/
│   │           ├── educational_profile.html
│   │           └── upload_pdf.html
│   └── ...
├── webui_static/                  ← NEW (Tailwind output, JS islands)
│   ├── tailwind.css
│   └── alpine_islands/
└── pyproject.toml                 (+jinja2, +sse-starlette, +fastapi-users)
```

### 6.2 Streaming sequence (Writer agent token stream)

```mermaid
sequenceDiagram
    participant T as Teacher (browser, htmx + WebAwesome)
    participant F as FastAPI /webui
    participant L as LangGraph (Writer)
    T->>F: POST /webui/lesson  (multipart: query + EducationalProfile)
    F-->>T: 200 OK  (HTML fragment with empty <div id="lesson-stream"
                        sse-connect="/webui/lesson/{id}/stream"
                        sse-swap="message">)
    F->>L: stream tokens via LangGraph
    loop for each token
        L-->>F: token / tool_start / tool_end / critic_revision
        F-->>T: SSE: <wa-tag variant="success">..</wa-tag><span>token</span>
    end
    L-->>F: completion event
    F-->>T: SSE: <button hx-get="/webui/lesson/{id}">Open lesson</button>
```

The same SSE channel can carry **tool approval prompts** as full WebAwesome dialogs (just send the dialog markup as an event, htmx swaps it into a portal `<div>`). The mental model stays "the server tells the browser what HTML to show next."

### 6.3 Auth contract (compatible with future SSO)

- `POST /webui/auth/login` → JWT cookie (HTTP-only, SameSite=Lax). JWT claims: `sub`, `email`, `domain`, `iss="graphaixlearning"`.
- Once we move to embed, AixLearning issues the same JWT shape with `iss="aixlearning"` and we accept both → SSO done.

### 6.4 Real-time integration with AixLearning (optional, later)

If teachers want notifications in the main AixLearning app *("your lesson is ready"*):
- FastAPI publishes `POST` to AixLearning's existing **Mercure hub** with the same JWT secret.
- Zero new infrastructure — we share the hub that AixLearning has already been running in production since the *"init mercure infra"* commit.

### 6.5 Three integration shapes for the eventual AixLearning embed

> **Decision deferred:** the embed itself is **explicitly out of scope for CORE 1–5** and lives at the end of CORE 6 (Deployment & Production), once the agent is feature-complete and stable. The point of documenting the shapes here is to prove the recommended architecture *enables* all three — we are not committing to one yet.

When embed time arrives, the FEM platform team and the GraphRAG team can pick whichever of these fits operational reality:

| | **Shape 1 — Service + iframe** | **Shape 2 — Service + template port** ⭐ | **Shape 3 — Service-only, AixLearning writes its own UI** |
|---|---|---|---|
| **What changes on AixLearning side** | Add an `<iframe>` page; share JWT in a parent-domain cookie (SSO). | Add a new Django app `aixlearning/graphrag/`. Port `src/aix/webui/templates/` → Django templates (Jinja2 → Django syntax, ~3-5 days). | Write fresh Django templates from scratch using FEM's UX guidelines; consume our `/api/v1/agent/*` JSON+SSE only. |
| **What changes on GraphRAG side** | Two config lines: trust AixLearning JWT issuer; allow AixLearning origin in CORS. | Two config lines (same as Shape 1). Optionally retire `src/aix/webui/` once the Django version is live, or keep it as the dev/admin UI. | Two config lines (same as Shape 1). `src/aix/webui/` continues to live as the internal/admin/dev UI. |
| **Effort** | Smallest (~1 day). | Medium (~3-5 days for the template port + QA). | Largest (the AixLearning team owns design + implementation). |
| **UX quality** | Acceptable but the iframe is visible (different scrollbar, no native nav). | **Indistinguishable** from native AixLearning. | Whatever AixLearning's UX team produces. |
| **Ownership boundary** | Two separate apps loosely coupled by JWT. | One Django app, one FastAPI service — clean SOA. | Two teams with a contract (the JSON API). |
| **When to pick it** | FEM platform team is busy and just wants the app to "appear" inside AixLearning fast. | We want polished, native UX and FEM has bandwidth to absorb a small Django app. | FEM has strong UX opinions and prefers to own the surface entirely. |

**Why this matrix matters today:** Path C is the only one of the three frontend candidates from §4 that keeps **all three shapes** open. Path A (Vercel + Next.js) only supports Shape 1. Path B (immediate native embed) is itself just Shape 2 — but Shape 2 reached prematurely, before we know what we're embedding.

---

## 7. Phased Roadmap

| Phase | Scope | Effort |
|-------|-------|--------|
| **P0 — Skeleton** | `src/aix/webui/` package, `_base.html`, Tailwind CLI, WebAwesome via CDN, htmx + Alpine, dummy `/webui/` route returning a styled "hello" page. | 0.5 day |
| **P1 — Auth + lesson form** | FastAPI-Users, `/webui/login`, `/webui/lesson` form, `EducationalProfile` form (mirrors `src/aix/api/schemas/educational_profile.py`). | 2 days |
| **P2 — Streaming chat** | SSE endpoint, htmx `hx-ext="sse"`, partial templates for token / tool / critic events. Wire to LangGraph Agent. | 2 days |
| **P3 — Tool approval + file upload** | WebAwesome `<wa-dialog>` swapped via SSE for high-risk tool calls; multipart upload form for lesson PDFs. | 2 days |
| **P4 — Lesson library + history** | Lesson list, search, export-as-PDF. | 2 days |
| **P5 — Polish + Italian copy** | Italian UI strings, accessibility audit, mobile breakpoints. | 2 days |
| **P6 — Hetzner deploy** | Docker Compose, Caddy/Traefik TLS, Postgres, optional Mercure. | 1 day |
| **P7 — Embed handoff (end of CORE 6, *not now*)** | Choose one of the three integration shapes (§6.5), document the template port to Django, hand off to AixLearning platform team. | 1-5 days when ready (depends on chosen shape) |

Total to a usable internal release: **~10 working days**. Vercel/Next.js path estimated at ~25–30 days for the same surface area, with twice the maintenance debt afterwards.

---

## 8. ADR-0001 — Frontend Platform

- **Status:** Proposed
- **Context:** The Agentic GraphRAG product needs a teacher-facing UI before it can be considered shippable. Three candidate stacks were evaluated against twelve criteria (see §5). The product must stream agent output, support `EducationalProfile`, and remain on a Hetzner/VPS budget.
- **Decision:** Build the frontend on **FastAPI + htmx + WebAwesome + Tailwind + Jinja2 + sse-starlette** (Path C). Reject Path A (Vercel + Next.js) for now due to skill, paradigm, and cost mismatch. Defer Path B (native Django embed) until the product is stable and the AixLearning platform team has bandwidth.
- **Consequences (positive):** Lowest TCO and time-to-first-deploy; perfect skill match; same component library and paradigm as the eventual embed target; no vendor lock-in; FastAPI JSON contract is unchanged so Path A can still be added later (B2C brand) without rewriting.
- **Consequences (negative):** No `useChat`-style library; we own conversation state on the server (acceptable — we want it persisted anyway). Some advanced animations easier in React are harder here (acceptable — the agent UI is mostly forms + streaming text + cards).
- **Reversal cost:** Low. The FastAPI JSON endpoints stay; only the `src/aix/webui/` templates change. Migration to Path B = port Jinja templates to Django templates (similar grammar). Migration to Path A = build a parallel Next.js app on top of the same `/api/v1/...` JSON contract — no breaking change.
- **Embed optionality:** Path C explicitly preserves all three integration shapes documented in §6.5 (iframe, template port, JSON-only). The embed itself is deferred to **the end of CORE 6** and is not on the CORE 1-5 critical path.

---

## 9. Open Questions for Simone, Diego, Filippo

1. **WebAwesome Pro vs Free:** AixLearning may already have a Pro license. If yes, GraphRAG can reuse Pro components (Data Grid, Date Picker, Toast). If no, Free covers everything we need for the agent UI.
2. **Mercure topic naming:** if/when we publish notifications, do we use a `graphrag.*` topic prefix or hang under `aixlearning.*`?
3. **JWT issuer convention:** what's the JWT `iss` and key-rotation story on the AixLearning side today? We want our JWT to drop in cleanly when SSO happens.
4. **Italian copy ownership:** does Simone draft UI strings, or do we propose a first cut?
5. **Embed timeline:** any guidance on when the AixLearning platform team can absorb this app? That decides whether P7 is "this quarter" or "Q3".

---

## 10. Appendix — Key Sources Reviewed

- *AI SDK Core: Tool Calling*, ai-sdk.dev (Vercel) — `needsApproval`, `useChat`, `addToolApprovalResponse`.
- *Adapters: LangChain*, ai-sdk.dev — `@ai-sdk/langchain` `toUIMessageStream`.
- *FastAPI + HTMX: The No-Build Full-Stack* (Blake Crosley) — production patterns.
- *Streaming in 2026: SSE vs WebSockets vs RSC* (JetBI) — protocol selection.
- *Mercure protocol* (Dunglas / mercure.rocks) — hub deployment, JWT.
- *Web Awesome* (webawesome.com) — license, `<wa-*>` web components, framework-agnostic usage.
- *Lovable Review 2026* — confirms Lovable cannot host a FastAPI agent backend.
- *AixLearning repo screenshot* (private, FEM-modena/aixlearning, staging branch, 2026-04-26) — language mix, folder layout, commit messages confirming htmx + WebAwesome + Mercure + Bun + Docker Compose.
