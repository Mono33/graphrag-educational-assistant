# Internal Production Deployment Plan — Agentic GraphRAG for FEM

**Date:** May 15, 2026
**From:** AI Team (LM)
**To:** FEM Direction, AI Team, Operations
**Scope:** Standalone deployment of the Agentic GraphRAG agent + WebUI as an internal FEM tool, accessible to ~5-10 domain experts via a dedicated HTTPS URL.

> **Important:** This document is **NOT** about the AixLearning DEV integration. That parallel track is covered in `Dev_Handoff_AgenticGraphRAG_Integration.md` and `Dev_Technical_Integration_Guide.md`. This document is about deploying our standalone app at a dedicated FEM-internal URL where domain experts can log in and test the full Agentic GraphRAG pipeline directly.

---

## 1. Purpose

We want FEM domain experts to access the Agentic GraphRAG agent + WebUI through a dedicated, secured internal URL: **`https://agente.aiforlearning.digital`**. This pilot serves three purposes:

1. **Validate the system with real domain experts** before any AixLearning production cutover.
2. **Collect real usage data** through Langfuse traces to drive subsequent quality and latency work.
3. **Establish a FEM-owned production environment** that survives team changes and is fully compliant with EU AI Act obligations.

The deployment is independent of AixLearning. The two paths converge later — when the AixLearning DEV team plugs our agent into their Django frontend (see `Dev_Handoff_AgenticGraphRAG_Integration.md`) — but the internal pilot launches first and runs on its own infrastructure.

---

## 2. Goals and assumptions

### 2.1 Goals

- **Stable**: 99% monthly uptime, no data loss between deployments.
- **Secure**: HTTPS-only, user authentication, no public agent API.
- **Observable**: every lesson generation produces a Langfuse trace; errors page the AI team.
- **Compliant**: EU AI Act Article 50 (AI-generated content marking) implemented before the August 2, 2026 enforcement deadline.
- **Reversible**: a documented rollback plan that anyone can execute in < 10 minutes.

### 2.2 Assumptions and prerequisites

| Assumption | Status | Action needed |
|---|---|---|
| **VM / VPS available** *(Linux, 4 GB RAM min, 40 GB disk, Docker-capable)* | ✅ PROVIDED | FEM will reuse the existing GraphRAG instance: Debian, 2 vCPU, 4 GB RAM, 50 GB disk, Docker/networking/CD already configured |
| **Domain or subdomain editable** *(for HTTPS endpoint)* | ✅ DONE | `agente.aiforlearning.digital` now resolves to `91.99.147.27` |
| Neo4j Aura instance already running | ✅ DONE | Reused as-is (`bolt+s://graph.aiforlearning.digital:7687`) |
| Langfuse cloud account provisioned | ✅ DONE | Reused as-is — keys already in our local `.env` |
| OpenRouter API key | ✅ DONE | Reused as-is |
| FEM agrees to act as data controller (GDPR) | TODO | Direction-level confirmation |
| Pilot user list (~5-10 FEM domain experts) identified | TODO | Direction-level identification |

### 2.3 Total effort and timeline

| Phase | Effort | Calendar | Hard deadline |
|---|---|---|---|
| Wave 1 — Data layer + container infra | ~2-3 days | Week 1 | — |
| Wave 2 — Public hostname + TLS | ~0.5-1 day | Week 1-2 | — |
| Wave 3 — User auth + guardrails | ~2 days | Week 2 | — |
| Wave 4 — Observability + reliability | ~1.5-2 days | Week 2-3 | — |
| Wave 5 — EU AI Act compliance | ~1.5 days | Week 3 | **August 2, 2026** |
| Wave 6 — Internal pilot launch | ~1.5 days | Week 3-4 | — |
| **Total** | **~9-11 working days** | **~3-4 weeks** | |

**Earliest realistic internal launch: end of Week 3 from kickoff.**

### 2.4 Current infrastructure status *(updated 2026-06-02)*

Francesco confirmed that FEM does **not** need to provision a new VM from scratch. The current GraphRAG instance is already dimensioned for the internal pilot: **2 vCPU, 4 GB RAM, 50 GB disk**, running on FEM's standard **Debian** template. Networking and Docker are already configured, and FEM already has a continuous-deployment pipeline in place.

The DNS blocker is also resolved: `agente.aiforlearning.digital` now has an A record pointing to `91.99.147.27` and has been verified with `nslookup`.

**Implication:** the blocker has moved from infrastructure provisioning to **deployment/configuration through the existing GitHub CD pipeline**. The next work is to merge/push the production-ready GraphRAG version, ensure the production environment variables are set correctly, and decide how `agente.aiforlearning.digital` should coexist with the existing `graph.aiforlearning.digital` endpoint during transition.

---

## 3. FAQ — Infrastructure prerequisites (read this before kickoff)

### 3.1 What is a VM/VPS and why do we need one?

A "Virtual Private Server" is a Linux computer in a data center that runs 24/7. We SSH into it, install Docker, and run our stack. Once it's running, our laptops become irrelevant — the server is always reachable from the internet at the public URL.

### 3.2 What does a VM cost?

| Provider | Plan | Cost | Pros | Cons |
|---|---|---|---|---|
| **Hetzner Cloud — CX22** *(Germany)* | 2 vCPU, 4 GB RAM, 40 GB SSD | **~€5/month** *(€60/year)* | Cheapest serious provider, FEM data stays in EU, super easy UI | Need a credit card |
| **Hetzner Cloud — CCX13** *(Germany)* | 2 vCPU, 8 GB RAM, 80 GB SSD | ~€13/month | More headroom for ~10 concurrent users | More than V1 strictly needs |
| **Scaleway DEV1-M** *(France)* | 3 vCPU, 4 GB RAM, 40 GB SSD | ~€7/month | French data center | UI a bit less friendly |
| **AWS / Azure / GCP** | t3.medium or equiv. | ~€15-30/month | Enterprise-grade | Overkill, complex billing |
| **FEM internal server** *(if available)* | depends | **FREE** | No external billing, full on-premise control | Depends on FEM IT having spare capacity |

**Recommendation: Hetzner CX22 (€5/month) or a FEM internal VM (€0) — both are sufficient for the pilot.**

### 3.3 Can I create the VM myself?

Yes, technically — but **don't**, for legal and operational reasons:

| Concern | Why FEM should own it |
|---|---|
| Billing | Should be FEM's expense, not personal credit card |
| Ownership when team changes | FEM keeps access to data when individuals leave |
| GDPR data controller | The legal data controller (FEM) must own the infrastructure |
| Tax / accounting | Corporate expense vs personal |

**Recommended workflow:**
1. AI team asks FEM Direction to authorise a ~€5-15/month Hetzner VPS *(or assign an internal VM)*.
2. FEM creates the Hetzner account with FEM billing details, or assigns the internal VM.
3. AI team receives SSH access and installs the stack.

**If FEM is slow:** AI team can provision a Hetzner VM on a personal card temporarily (1-week stopgap) but **must transfer the account to FEM before any pilot launch**. Hetzner supports free account transfers.

### 3.4 What is a domain and what's a subdomain?

A domain is the address typed into the browser. It has layers:

```
graph.aiforlearning.digital
└┬─┘ └────────┬─────────┘ └┬┘
 │            │             │
 │            │             └── top-level domain (TLD)
 │            └── apex domain (the root — owned by someone)
 └── subdomain (a label inside the apex)
```

The **apex domain** `aiforlearning.digital` is the "apartment building." The **subdomain** `graph` is one specific apartment inside it (currently pointing to Neo4j). **You can have unlimited subdomains, all independent of each other.**

Adding `agente.aiforlearning.digital` does NOT touch or break `graph.aiforlearning.digital`. They're separate apartments in the same building.

### 3.5 What subdomain we chose for the pilot

**Chosen subdomain: `agente.aiforlearning.digital`** *(Italian-friendly, immediately understandable to FEM domain experts and Direction).*

Other options considered (kept here for the record):

| Subdomain | URL | Notes |
|---|---|---|
| `agente` *(chosen)* | `https://agente.aiforlearning.digital` | Italian-friendly, clear to non-technical FEM users |
| `graphrag` | `https://graphrag.aiforlearning.digital` | Technical, matches repo name |
| `lezioni` | `https://lezioni.aiforlearning.digital` | Teacher-friendly |
| `pilota` | `https://pilota.aiforlearning.digital` | Communicates "this is a pilot" |
| `assistente` | `https://assistente.aiforlearning.digital` | Generic, friendly |
| `labs` | `https://labs.aiforlearning.digital` | Experimental connotation |

### 3.6 How do we actually add the subdomain?

Whoever owns `aiforlearning.digital` (FEM or Angelo — see §3.7) logs into the DNS provider — Cloudflare, Aruba, GoDaddy, etc. — and adds:

```
Type: A
Name: agente
Value: <our VM's IP address>
TTL: 3600
```

That's a 30-second change. Within ~5 minutes `agente.aiforlearning.digital` resolves to our VM. Nothing else is affected.

### 3.7 Open question to confirm with Angelo

In Angelo's recent commit `990afde` (`external_apis.py`, OpenAlex polite-pool header), he uses the email `angi36casali@gmail.com`. That's a personal Gmail address, which suggests the apex domain `aiforlearning.digital` *may* be registered to him personally rather than to FEM.

**Question to ask Angelo:**

> *"Il dominio `aiforlearning.digital` è registrato a tuo nome personale o a nome di FEM? Per il pilot interno vorremmo aggiungere un subdomain (`agente.aiforlearning.digital`). Se il dominio è personale, valutiamo se registrare un nuovo dominio FEM-corporate."*

### 3.8 What if the domain is Angelo-personal? Can we register a new one?

Yes, easily. Costs are ~€10-25/year:

| Provider | Cost | TLD | Notes |
|---|---|---|---|
| **Cloudflare Registrar** | ~€8-10/year | `.com` | Cheapest, best DNS UI, no upsell |
| **Aruba** | ~€15-25/year | `.it` | Italian provider, useful for `.it` |
| **Namecheap / GoDaddy** | ~€10-15/year | `.com` | Standard |

Examples of fresh corporate-FEM domains:
- `agente-fem.com` (~€10/yr)
- `aix-fem.it` (~€20/yr)
- `agente-graphrag.it` (~€20/yr)
- `assistente-fem.it` (~€20/yr)

### 3.9 Final URL — what FEM domain experts will see

The pilot URL will be:

**`https://agente.aiforlearning.digital`**

*(Fallback if the apex `aiforlearning.digital` turns out to be personally-owned by Angelo: a new corporate-FEM domain such as `https://agente-fem.it` — see §3.8.)*

FEM domain experts type this URL into their browser, log in with credentials we provision, and use the full Agentic GraphRAG WebUI we already have today.

---

## 4. Wave-by-wave plan

The deployment is sequenced into 6 waves, each depending on the previous. Within a wave, items may run in parallel.

### Wave 1 — Data layer + container infrastructure *(Week 1, ~2-3 days)*

Set up the production data layer and ship the Docker stack on the chosen VM.

| # | Activity | Origin ticket | Effort | Owner | Notes |
|---|---|---|---|---|---|
| 1 | **PostgresSaver migration** *(replace `AsyncSqliteSaver` checkpointer)* | CORE 4 `#15.a` | 1-2 h | LM | ✅ DONE locally (2026-05-16): `LANGGRAPH_DATABASE_URL` selects `AsyncPostgresSaver`; startup calls `setup()`; smoke confirmed table bootstrap + `aput`/`aget_tuple` round-trip against Postgres 16 |
| 2 | **Production Docker Compose** *(api + webui + postgres + caddy)* | CORE 2 `#6.6 P6` | 4-6 h | LM | ✅ DONE locally (2026-05-16): `deploy/docker-compose.prod.yml` defines app + Postgres 16 + Caddy 2; app/Postgres are internal-only; `docker compose config --quiet` passed with `.env.prod.example` |
| 3 | **Persistent volumes + backup strategy** | new | 2 h | LM | ✅ Scripts ready (2026-05-16): `deploy/scripts/{backup_postgres,restore_postgres,backup_caddy}.sh` + cron template in `deploy/README.md` §5.2. Atomic dumps with 7-day retention; restore takes a safety snapshot first. Off-host copy + side-VM restore drill remain ops tasks once the VM is provisioned. |
| 4 | **`requirements.txt` lockfile + multi-stage Dockerfile** *(production image)* | CORE 2 `#6.6 P6` | 2-3 h | LM | ✅ DONE locally (2026-05-16): `requirements.lock.txt` generated for Python 3.12/Linux with hashes; Dockerfile uses slim Bookworm, non-root `aix` user, container port `8765`, and baked-in `/api/v1/health` `HEALTHCHECK`. Linux VM build/healthcheck remains the final infra verification. |
| 5 | **VM provisioning + Docker installation** | new | 2-3 h | FEM + LM | ✅ PROVIDED by FEM: existing GraphRAG instance, Debian, 2 vCPU, 4 GB RAM, 50 GB disk; Docker/networking/CD already configured |

**Wave 1 deliverable:** the app stands up cleanly on a fresh VM via `docker compose up -d`, with persistent Postgres volume + working Neo4j-driver connectivity. Local-style URL still — no DNS yet.

**2026-05-16 status note:** items 1-4 are now locally prepared/validated. The LangGraph checkpointer side of item 1 is validated on a throwaway Postgres container (`pg-aix-smoke`, Postgres 16). The WebUI database was already URL-driven via `WEBUI_DATABASE_URL`; production simply points it at Postgres. Historical SQLite checkpoint migration is intentionally skipped for the internal pilot, so production starts with empty conversation memory and empty WebUI data. Item 2 adds the production deploy folder (`deploy/docker-compose.prod.yml`, `deploy/Caddyfile`, `deploy/.env.prod.example`, `deploy/README.md`) and validates Compose syntax/env interpolation without requiring real production secrets. Item 3 provides backup/restore scripts and cron guidance. Item 4 locks the production dependency set and hardens the Docker runtime; the remaining proof is a full build + healthcheck on the Linux VM.

**2026-06-02 status note:** the VM-side prerequisite is now considered solved. FEM will reuse the existing GraphRAG production instance instead of provisioning a fresh VPS. The deploy path is therefore not "SSH to a new machine and bootstrap Docker" but "push/merge to GitHub and let FEM's continuous-deployment pipeline deploy the new version." Linux VM build/healthcheck remains the final validation gate, but not an infrastructure blocker.

---

### Wave 2 — Public hostname + TLS + CORS lockdown *(Week 1-2, ~0.5-1 day)*

Make the WebUI reachable over HTTPS at a real URL.

| # | Activity | Origin ticket | Effort | Owner | Notes |
|---|---|---|---|---|---|
| 6 | **DNS A record** *(`agente.aiforlearning.digital` → VM IP)* | new | 30 min | FEM (or whoever owns the apex) | ✅ DONE: `agente.aiforlearning.digital` resolves to `91.99.147.27` |
| 7 | **Caddy reverse proxy + auto-Let's-Encrypt TLS** | CORE 2 `#6.6 P6` | 1-2 h | LM | Caddyfile auto-resolves cert; HTTPS forced; HSTS header set |
| 8 | **Production `.env` template + CORS lockdown** | new | 1 h | LM | `WEBUI_CORS_ALLOW_ORIGINS=https://agente.aiforlearning.digital`, all secret values rotated from local `.env` |
| 9 | **Secrets management** *(no secrets in image; `.env` mounted from host, owned `root` `0600`)* | new | 1 h | LM | Or migrate to Docker secrets if FEM standardises on Swarm/k8s later |

**Wave 2 deliverable:** `https://agente.aiforlearning.digital` resolves, serves the WebUI over HTTPS with a valid TLS certificate. CORS rejects all non-FEM origins.

**2026-06-02 status note — `agente` vs `graph`:** DNS is done, but DNS alone does not make the WebUI live. It only means browsers know that `agente.aiforlearning.digital` should reach server `91.99.147.27`. The deployed reverse proxy/application must still be configured to accept that hostname and serve the new app.

Current code supports a single FastAPI process that serves both surfaces:

- `/webui/*` — standalone WebUI for Mode A
- `/api/v1/context` — legacy GraphRAG API already used by the old AixLearning integration
- `/api/v1/agent/run` and `/api/v1/agent/stream` — new Agentic GraphRAG API
- `/docs` and `/openapi.json` — Swagger/OpenAPI documentation

The current production `Caddyfile` is parameterized by `AIX_DOMAIN`. If `AIX_DOMAIN=agente.aiforlearning.digital`, then `https://agente.aiforlearning.digital` will expose the WebUI and the API/docs for the same FastAPI app.

Transition point to clarify with FEM DEV: whether `graph.aiforlearning.digital` should remain active for the existing API/docs while `agente.aiforlearning.digital` becomes the pilot WebUI hostname. The safest transition is to keep both hostnames working temporarily: `graph` for legacy API compatibility, `agente` for the standalone WebUI pilot. Native AixLearning Mode B should eventually call the GraphRAG service over the internal Docker network rather than depending on either public hostname.


So: DNS blocker is solved. VM blocker is also basically solved because they will reuse the existing GraphRAG instance. The next blocker is now deployment/configuration through their GitHub CD pipeline, not infrastructure provisioning.

---

### Wave 3 — User authentication + access control for FEM experts *(Week 2, ~2 days)*

Make the WebUI safe to expose by gating it behind FEM-issued user accounts.

| # | Activity | Origin ticket | Effort | Owner | Notes |
|---|---|---|---|---|---|
| 10 | **User-account provisioning workflow** *(admin CLI: `aix-cli users create <email> <name>`)* | CORE 2 `#6.6 P5` extension | 3-4 h | LM | Argon2 password hashing already in stack; one-time emailed credentials; first-login forced password change |
| 11 | **Login page + session cookie hardening** *(SameSite=Strict, Secure, HttpOnly)* | CORE 2 `#6.6 P5` | 2 h | LM | Verify and tighten existing session middleware |
| 12 | **Input/output guardrails** *(prompt injection + harmful output filter)* | CORE 2 `#8` | 3-5 h | LM | Pydantic input validation on `EducationalProfile`; regex pre-filter on user query; harm filter on `lesson_plan_md` |
| 13 | **Per-user rate limiting** *(e.g. 30 lessons/day, 10/hour)* | new | 2 h | LM | Slowapi middleware; configurable per user role |
| 14 | **Admin role + "view all lessons" page** *(for AI team oversight)* | new | 2-3 h | LM | Read-only audit view; not strictly needed for V1 if pilot is small |

**Wave 3 deliverable:** FEM experts log in with their own accounts; admins can provision, suspend, or reset users from CLI. The system rejects unauthorised input.

---

### Wave 4 — Observability + reliability *(Week 2-3, ~1.5-2 days)*

Make sure we can see what's happening in production and recover gracefully from common failures.

| # | Activity | Origin ticket | Effort | Owner | Notes |
|---|---|---|---|---|---|
| 15 | **Langfuse traces verified end-to-end + dashboard cards** *(P50/P95 latency per phase, KG coverage, retrieval outcomes)* | CORE 2 `#11b` | 3-4 h | LM | Foundation already landed by AG; needs prompt seeding + dashboard build |
| 16 | **GlitchTip / Sentry error monitoring** *(DSN already in `.env`)* | partial | 1 h | LM | Verify error events flow; create alerts for `event=agent_parse_error` and unhandled exceptions |
| 17 | **SSE backpressure + reconnect hardening** | CORE 2 `#12` | 2-3 h | LM | Prevent flaky Wi-Fi from breaking lesson generation; add `retry: 3000ms` to all SSE responses |
| 18 | **Health-check endpoint with deep checks** *(Postgres + Neo4j-driver + LLM ping)* | new | 1-2 h | LM | `GET /health` returns 200 only if all 3 are reachable; Caddy uses it for safer upgrades |
| 19 | **Log rotation + retention** | new | 1 h | LM | Docker `json-file` driver with 10 MB × 5 files cap; structured JSON logs from `logging` |
| 20 | **Connectivity probe at startup** *(already implemented)* | DONE *(2026-05-10)* | 0 h | — | `src/aix/core/connectivity_probe.py` — verify it logs cleanly in prod |

**Wave 4 deliverable:** every lesson produces a Langfuse trace; the AI team is paged on errors; flaky networks don't break in-flight lessons.

---

### Wave 5 — EU AI Act compliance *(Week 3, ~1.5 days) — HARD DEADLINE August 2, 2026*

Implement the regulatory items identified in `Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md`.

| # | Activity | Source | Effort | Owner | Notes |
|---|---|---|---|---|---|
| 21 | **Machine-readable AI-generated content marking** *(Markdown comment + HTTP header + C2PA metadata in exports)* | Reg. doc action #1 | 1 day | LM | `<!-- ai-generated: true, system: agentic-graphrag, trace_id: ... -->` at top of every `lesson_plan_md`; `X-AI-Generated: true` HTTP header; PDF/DOCX exports include first-page notice |
| 22 | **Visible "Generato dall'IA" disclosure** *(persistent footer + per-message badge)* | Reg. doc action #2 | 2 h | LM | Django/template change; existing aix-brand.css styling reused |
| 23 | **"How the AI works" teacher guide** *(static page accessible from header)* | Reg. doc action #6 | 2 h | LM | Plain-Italian explanation of the 4-agent pipeline; non-technical, for AI literacy obligations |
| 24 | **Environmental sustainability telemetry** *(`total_tokens`, `estimated_kwh`, `co2eq_grams` per run)* | Reg. doc action #3 (UNI 11621-8 Theme 2) | 1 day | LM | Add to `AgentRunMeta` and Langfuse traces; conversion factor ~0.001-0.003 kWh per 1k tokens |

**Wave 5 deliverable:** every lesson plan is clearly marked as AI-generated in both UI and exported artifacts; the system is Article 50 compliant by the regulatory deadline.

---

### Wave 6 — Internal launch to FEM domain experts *(Week 3-4, ~1.5 days)*

Smoke-test, document, and open the door.

| # | Activity | Origin | Effort | Owner | Notes |
|---|---|---|---|---|---|
| 25 | **End-to-end production smoke** *(real FEM expert account, real KG, real LLM, full UDL + Neuro queries)* | new | 0.5 day | LM + AG | Sign-off gate before pilot access; verify latency budget meets targets |
| 26 | **Rollback plan documented** *(blue-green or simple `docker compose down && checkout previous tag && up`)* | new | 2 h | LM | One-page Markdown in `docs/ops/rollback.md`; tested once on staging |
| 27 | **Onboarding doc for FEM domain experts** *(login instructions + how to write a good prompt + how to set the educational profile)* | new | 3 h | LM | Markdown + 5-10 screenshots; lives in `docs/users/internal_onboarding.md` |
| 28 | **Pilot user list + credentials provisioned** | new | 1 h | LM | Bulk-create 5-10 accounts; send welcome email |
| 29 | **Feedback channel** *(ClickUp form, Loom-recorded bug report flow, or shared inbox)* | new | 1 h | LM | Quick way for pilots to report issues without filing GitHub tickets |
| 30 | **Internal launch announcement** *(Slack / Teams / email from Direction)* | FEM-driven | 1 h | FEM Direction + LM | Sets expectations: "this is a pilot, give us feedback, expect 30-60 s waits" |

**Wave 6 deliverable:** 5-10 FEM domain experts have working accounts at `https://agente.aiforlearning.digital`, can run lessons, and have a documented way to give feedback.

---

## 5. Critical path — the 5 hard blockers

If you stripped everything else, these are the minimum items that block launch:

1. **Production server provisioned + Docker stack running** *(Wave 1)*
2. **HTTPS at the public URL** *(Wave 2)*
3. **User authentication + at least one admin account** *(Wave 3)*
4. **Article 50 AI disclosure** *(Wave 5 — legal deadline August 2, 2026)*
5. **End-to-end smoke test on production** *(Wave 6)*

Everything else is "production quality" rather than "production possible."

**Current blocker after FEM infra response:** items 1 and 2 are no longer blocked by VM/DNS availability. The next blocker is deployment/configuration through FEM's existing GitHub CD pipeline: production environment variables, hostname handling (`agente` and possibly `graph`), TLS issuance, and first production smoke test.

---

## 6. Explicitly deferred

These items are tracked in `ClickUp_Agentic_GraphRAG_Update.md` but **NOT** gating the internal pilot launch:

| Deferred | Why | When to revisit |
|---|---|---|
| CORE 3 A — Query Decomposition (#13), Citation Grounding (#14), Semantic Caching (#17), Model A/B (#18) | Quality wins, not safety/launch blockers | After pilot — based on Langfuse data |
| CORE 3 C remaining — `#LAT-2` / `#LAT-3` / `#LAT-6` / `#LAT-7` partials | Latency wins, perceived UX is already good after Angelo's `990afde` | Sprint 2 post-launch |
| CORE 4 #15.b Conversation Memory Hardening *(full scope — telemetry, dedup)* | Multi-turn already works; hardening is operational polish | After 2 weeks of pilot data |
| CORE 4 #15.c Conversation Memory UX V2 *(time-travel, branch-from)* | Design-only, not validated with users yet | After pilot survey |
| CORE 4 #16 Human Loop / RLHF | Premature without pilot data | 3-6 months post-launch |
| CORE 5 additional MCP servers *(beyond #20, which is DONE)* | One MCP server is enough for V1 | When third-party LLM apps want to integrate |
| `#9.UX-4` Corrective-RAG B1/B2 fixes | Only matters if `AIX_CORRECTIVE_RAG_ENABLED=true`; default is OFF | When CR re-enabled |
| WebUI page re-skins beyond P5.4 (P5.5 / P5.6) | Aesthetic, not functional | After pilot feedback |
| Multi-region deployment / CDN edge | Single VM is fine for ~10 users | When user count > 100 |

---

## 7. Concurrency in production — scale-up roadmap

This section answers a recurring product-team question ("can it handle many teachers at once?") with a concrete plan. It is intentionally **out of scope for the ~5-10 user pilot** (Waves 1-6 above are enough to launch), but it is the path from **pilot-grade** to **production-grade** concurrency. It mirrors the two-phase recommendation in the companion answers doc (`Risposte_Domande_Product_Team_Agentic_GraphRAG.md` §1.2) and the scaling note in `docs/release/Technical_Documentation.md` §17.2.

### 7.1 Where we are today (grounded in the code)

**Already in place ✅** — the async foundation is real:

- FastAPI async endpoints + LangGraph async execution + SSE streaming, so one process interleaves many requests on the event loop.
- Per-request DB sessions; Postgres checkpointer for multi-turn memory (Wave 1 #1).
- A **per-lesson, in-memory** duplicate-run guard (`_ACTIVE_RUNS` in `src/aix/webui/lessons/routes.py`) that stops a second stream attaching to the *same* lesson within the *same* process.
- Token-bucket rate limiting on **external media APIs only** (`RateLimiter` in `src/aix/agent/media/external_apis.py`).

**Missing for many simultaneous users ❌** — there is no explicit *load control*:

- No global cap/queue on concurrent agent runs → N users = N full pipelines = N simultaneous LLM calls, with no backpressure.
- No process-wide limit on concurrent **LLM** calls (the only limiter is for external media APIs), so a burst can hit provider rate limits / spike cost.
- `_ACTIVE_RUNS` is **per-process** and **non-persistent** → it does not coordinate across multiple workers/replicas.
- The compiled graph + agents are **module-level singletons** (`aix.agent.graph.nodes`) — fine for one box, must be revisited for parallel throughput.
- No per-user rate limiting yet (planned as Wave 3 #13), no operational load/cost dashboard.

### 7.2 Phase A — single-worker hardening *(cheapest; enough for tens of users on one larger VM)*

Ordered so the highest-value, lowest-risk items come first. None require a horizontal-scale refactor.

| # | Activity | Effort | Owner | Notes |
|---|---|---|---|---|
| 31 | **Global generation queue + bounded concurrency** *(`asyncio.Semaphore` caps N in-flight pipelines; excess requests queue)* | 4-6 h | LM | The single biggest lever. Surface an "in coda…" state in the WebUI when queued; cap is env-configurable (e.g. `AIX_MAX_CONCURRENT_RUNS`) |
| 32 | **Process-wide concurrent-LLM-call cap** *(token bucket / semaphore across planner+writer+critic)* | 3-4 h | LM | Protects against OpenRouter 429s and cost spikes independent of #31; tune to provider limits |
| 33 | **Per-user rate limiting** *(e.g. 30 lessons/day, 10/hour)* | 2 h | LM | Same item as Wave 3 #13 — listed here for completeness; slowapi middleware, per-role config |
| 34 | **Load-shedding / graceful "sistema occupato"** *(reject at capacity instead of failing mid-run)* | 2-3 h | LM | When queue (#31) is full, return a friendly "riprova tra poco" rather than a hard error or a slow run |
| 35 | **Concurrency load test** *(k6 / Locust: simulate N concurrent teachers; measure P95, 429 rate, RAM)* | 4-6 h | LM | Produces the real "safe concurrent users on this VM" number; feeds #36 |
| 36 | **Vertical capacity sizing** *(measure per-run RAM/CPU; decide CX22 4 GB → CCX13 8 GB if needed)* | 2 h | LM + FEM | Cheapest scale-up is a bigger single VM (see §3.2); revisit only if #35 shows headroom is short |

**Phase A pilot note:** strictly, the pilot needs none of these, but **#31 + #34** are cheap insurance and worth doing opportunistically even for the internal pilot.

### 7.3 Phase B — multi-worker / horizontal scale *(needed for hundreds of users; requires shared state)*

| # | Activity | Effort | Owner | Notes |
|---|---|---|---|---|
| 37 | **DB-backed run registry** *(replace in-memory `_ACTIVE_RUNS` with a Postgres row + heartbeat)* | 1-1.5 d | LM | Prereq for >1 worker: lets workers see each other's in-flight runs and recover stale rows after a crash |
| 38 | **Revisit module-level singletons** *(graph + agents → worker pool or per-request graph; verify async/thread safety)* | 1-2 d | LM | Removes the shared-mutable-state risk under truly parallel runs; consolidate streaming into one `aix.agent.streaming` module (per Tech Doc §17.2) |
| 39 | **Multi-worker / multi-replica deploy** *(uvicorn `--workers` or replicas behind Caddy)* | 0.5-1 d | LM + FEM | Safe only after #37 + #38; Postgres checkpointer + WebUI DB are already shared, so state is ready |
| 40 | **Operational concurrency dashboard** *(active runs, queue depth, LLM call rate, cost/run, 429 rate)* | 0.5 d | LM | Extends the Langfuse work in Wave 4 #15; the cockpit for tuning #31/#32 caps |

### 7.4 Recommendation

1. **Pilot (now):** ship as-is; optionally land **#31** and **#34** as cheap safety nets.
2. **If the WebUI grows into a real multi-user product:** do **Phase A** first (single bigger VM gets you surprisingly far), then **Phase B** only when load tests (#35) or user count justify horizontal scale. Do **not** enable multi-worker (#39) before the DB-backed registry (#37) and singleton review (#38), or the per-process guard will silently break.

---

## 8. Open questions for FEM Direction

These must now be resolved before the production pilot can be opened to users:

1. **Hostname transition:** should `graph.aiforlearning.digital` remain active for existing API/docs while `agente.aiforlearning.digital` serves the standalone WebUI pilot?
2. **Reverse-proxy/CD configuration:** should the FEM deployment accept both hostnames, or should it switch the primary public hostname from `graph` to `agente`?
3. **Production environment variables:** confirm final values for `AIX_DOMAIN`, `WEBUI_CORS_ALLOW_ORIGINS`, `WEBUI_AUTH_SECRET`, Postgres credentials, Langfuse keys, OpenRouter key, Neo4j password, and TLS email in the CD-managed production environment.
4. **FEM owner of the deployment:** confirm the operational owner for production deploys, secrets, DNS changes, rollback, and incident response.
5. **Pilot user list:** which ~5-10 FEM domain experts? *(needed for Wave 6 account provisioning)*
6. **Launch communication channel:** how will FEM Direction announce the pilot to participants? *(Slack, email, internal meeting?)*

---

## 9. Cross-references

- **AixLearning DEV integration** *(separate track)*: `Dev_Handoff_AgenticGraphRAG_Integration.md`, `Dev_Technical_Integration_Guide.md`
- **Regulatory compliance**: `Regulatory_Alignment_EU_AI_Act_UNI_11621_8.md`
- **Production analysis (early study)**: `Integration_Production_AixLearning.md`
- **Project status / all open tickets**: `ClickUp_Agentic_GraphRAG_Update.md`
- **API surface (OpenAPI)**: `src/aix/api/routes/agent.py`, Swagger UI at `/docs`
- **Connectivity probe**: `src/aix/core/connectivity_probe.py`
- **Existing Production-Readiness Appendix A** *(focused on AixLearning DEV integration)*: `ClickUp_Agentic_GraphRAG_Update.md` → APPENDIX A

---

**Document owner:** LM (AI Team)
**Intended reviewers:** FEM Direction, AI Team (LM + AG), Operations
**Version:** 1.1
**Next review:** After FEM confirms hostname/CD configuration for `agente.aiforlearning.digital` and `graph.aiforlearning.digital`
