# Review of the AIxLearning API contract

**Review date:** 18 October 2026  
**Reviewed branch:** `chore/repo-reorg`  
**Reviewed commit:** `615f7af`  
**Compared with:** *Technical Implementation Roadmap — Private RAG API and AIxLearning Integration* (3 October 2026)

## Executive assessment

The work on `chore/repo-reorg` is a real API implementation, not only a design
or documentation proposal. It exposes the private-material pipeline,
conversations, lesson turns, streaming, exports and explainability through
authenticated JSON and SSE endpoints.

The contract is:

- **ready for Dev client development**;
- **ready for a protected staging pilot after deployment configuration**;
- **not yet ready to freeze as the final production contract**.

The distinction is important: the API is integrated into the GraphRAG FastAPI
application, but the corresponding client integration in the official
AIxLearning platform is still joint AI/Dev work.

## Scope and evidence

This review covers:

- `docs/api/Chatbot_API_Guide.md`;
- `docs/api/PRIVATE_MATERIALS_API.md`;
- `docs/product/Analisi_Integrazione_AIxLearning.md`;
- the routes and schemas under `src/aix/api/`;
- the shared turn orchestration in `src/aix/webui/agent/turns.py`;
- owner-scoped material storage and retrieval;
- the API idempotency migration and implementation;
- the focused API, ownership, profile, streaming and parity tests;
- `scripts/eval/api_smoke.py`;
- the original private-RAG/API roadmap.

The branch contains 66 focused test functions across the API-specific test
modules. On 18 October the focused suite expanded to 80 parametrized test cases
and passed locally (`80 passed`); the updated API smoke script also passed Ruff
and Python compilation checks. Production readiness still requires running
that smoke flow against the deployed stack and completing joint AIxLearning
end-to-end verification.

## What was implemented well

### 1. A complete conversational surface

The API now provides the flow an external platform needs:

1. upload and poll a private document;
2. create a persistent session;
3. send a lesson, answer or refinement turn;
4. follow the result through SSE or wait for JSON;
5. reload the session;
6. retrieve media and export the lesson.

This closes the central gap identified in the original roadmap: private RAG is
no longer limited to the standalone WebUI.

### 2. Reuse of the WebUI application logic

An API session is backed by the existing `Lesson` model and turns are accepted
and driven through the shared turn service. This was a deliberate alternative
to the roadmap's separate `material_set` model.

The choice is technically credible because it provides:

- the same ownership rules as the WebUI;
- the same private-material bindings and retrieval;
- persistent history;
- fast answers that do not rewrite the lesson;
- handoff before replacing the lesson;
- refinement support;
- the same exports and explainability payloads.

It also reduces the risk that WebUI and API behavior drift into two separate
implementations.

### 3. Owner and tenant isolation

The API maps a service account plus `X-End-User-Id` to an internal owner.
Owner-scoped queries are then reused consistently for lessons, materials,
versions, bindings and downloads.

The later tenant fix is especially important: each AIxLearning instance can
have separate credentials through `AIX_API_CLIENTS`. The same numeric platform
user ID in production and corporate therefore maps to different owners.

Wrong-owner identifiers return `404`, avoiding an ownership oracle.

### 4. Retry-safe asynchronous integration

`Idempotency-Key` is implemented for:

- material creation;
- session creation;
- blocking turns;
- streamed turns.

Successful requests are replayed without a second generation or second model
cost. In-progress requests return `409 idempotency_in_progress`; failed
requests release the key; and reuse with a different body returns
`422 idempotency_key_reused`.

This directly addresses AIxLearning's Dramatiq retry behavior.

### 5. Streaming suitable for Mercure

The session SSE endpoint now emits text while the Writer is generating. Draft
numbers make Critic-driven rewrites explicit:

- append text for the current draft;
- clear the preview when the draft number increases;
- replace the preview with the final `result.markdown`.

This is compatible with a Dev adapter that batches deltas and republishes them
through Mercure.

### 6. Compatibility with the current platform profile

The contract now accepts:

- the current BES codes, including `SC` and `CL`;
- legacy saved codes;
- `students_number: 0` as unknown;
- tri-state classroom answers;
- local, remote and hybrid environments;
- the platform's participation and modality values.

This closes the concrete schema mismatches found by comparing the two
repositories.

### 7. Additive backward compatibility

The existing surfaces remain available:

- `/api/v1/context` for GraphRAG-only retrieval;
- `/api/v1/agent/run` and `/api/v1/agent/stream` for one-shot generation;
- `teacher_provided_context` for older clients.

The new materials, sessions and catalog routes are additive. The idempotency
migration adds a table without changing existing data.

## Comparison with the original roadmap

| Roadmap area | Current status | Review |
| --- | --- | --- |
| Owner-scoped materials API | Implemented | Upload, list, detail, versions, attach, detach, source download and retirement exist. |
| Persistent material sets | Implemented differently | Sessions reuse `Lesson` and `LessonMaterialBinding` instead of introducing a separate set model. |
| Private retrieval in the public API | Implemented through sessions | The one-shot `/agent/*` routes intentionally remain document-less. |
| Conversation API | Implemented | History, fast answers, handoff and refinements are available. |
| Readiness | Partial | Processing blocks a turn, but failed `must_follow` material is not yet enforced as a hard failure. |
| Citations | Partial | Markdown citations are resolved and document names are returned; structured source/locator objects are missing. |
| Original-file access | Implemented | Owner-scoped source download exists. |
| Page preview | Not implemented | Required for directly verifiable page citations. |
| External identity | Pilot implementation | Per-instance Basic credentials plus an opaque end-user header; signed production identity remains open. |
| Idempotency | Implemented | Persistent, owner-scoped and shared between blocking and streamed turns. |
| Token streaming | Implemented | Includes Critic revision drafts and final replacement semantics. |
| Platform client integration | Not implemented here | AIxLearning must still build its API client, persistence and Mercure adapter. |

## Incomplete or risky areas

### Production blockers

#### Signed identity and perimeter protection

The current Basic-authentication design is suitable for a pilot only when TLS,
an IP allowlist and per-instance credentials are enforced.

Before production, the teams should either implement the proposed signed
identity (`iss`, `aud`, `sub`, `tenant`, `exp`) or formally approve and document
an equivalent security boundary.

The deployment must also protect:

- `/docs`;
- `/redoc`;
- `/openapi.json`;
- `/api/v1/context`.

The application currently mounts the documentation openly, and `/context`
does not have the same application-level authentication dependency as the new
session/material routes.

#### Anonymization ownership

The platform currently anonymizes messages and extracted document text before
model calls. Sending original documents directly to GraphRAG would change that
privacy guarantee because data could reach embeddings, vision OCR and tracing.

The pilot should freeze one of two contracts:

1. AIxLearning keeps anonymization and uploads page-structured Markdown; or
2. GraphRAG receives originals and anonymizes locally before every external
   provider or trace.

This decision must precede the final Dev upload implementation.

### Contract hardening before production freeze

#### Required-material semantics

The API accepts `purpose="must_follow"`, but the run guard currently blocks only
materials that are still processing. A failed required material can therefore
be omitted from retrieval while generation continues.

Recommended change:

- fail the turn when a required bound version is failed, deleted or otherwise
  unavailable;
- return a stable machine-readable list of affected material/version IDs.

#### `materials_preparing` response

The current `409 materials_preparing` response includes document names only in
the human-readable Italian message. It does not expose a structured list and
does not currently set `Retry-After`.

Dev should not be required to parse prose. Add structured material information
and a retry hint before freezing the contract.

#### Citation and provenance payload

The final Markdown contains readable citations and `cited_documents` contains
document names. It does not yet expose a structured `sources[]` containing:

- material ID;
- selected version ID;
- display name;
- page/section locator;
- availability or tombstone status.

The underlying binding pins a selected version, and the explicit attach route
accepts a `version_id`, but the session response does not expose the binding's
selected version. This limits reproducibility for an external platform.

#### Error normalization

Custom material/session errors use:

```json
{"detail": {"code": "...", "message": "..."}}
```

Normal Pydantic validation errors and some one-shot Agent API errors retain
their FastAPI or plain-string shapes. The Dev client therefore needs either:

- a documented union of error shapes; or
- a normalization handler that guarantees one versioned envelope.

#### Production Writer token setting

The code default and local generated environment use
`AIX_WRITER_MAX_TOKENS=8000`, but `deploy/.env.prod.example` still sets `2000`.
That lower value previously produced empty or truncated output with a reasoning
model.

The production value must be deliberately validated and the template,
environment documentation and deployed secret must agree.

### Joint AI/Dev decisions

The following are product/integration decisions rather than defects in the
implemented API:

- when to charge and refund credits for `lesson`, `answer` and `handoff`;
- whether AIxLearning or GraphRAG owns PDF/DOCX export and watermarking;
- correlation IDs across AIxLearning, GraphRAG and Langfuse;
- source-preview ownership;
- retention and deletion policy;
- accepted file types and limits for the first pilot;
- whether the final lesson is stored only by AIxLearning or by both systems;
- rollout beyond the current UDL and neuroscience domains.

## Backward-compatibility assessment

The implementation is largely backward compatible:

- old endpoint paths remain;
- existing one-shot requests remain valid;
- private-material fields are optional on the new session flow;
- old profile codes remain accepted;
- new database changes are additive;
- feature flags can keep the material library or retrieval disabled.

The main compatibility risk is operational rather than syntactic: deploying
new routes without the migrations, durable blob volume, worker, credentials
and feature flags would expose a contract that cannot complete its advertised
flow.

## Contract to explain to the Dev team

### Authentication

For the pilot, every server-to-server call uses:

- HTTP Basic credentials dedicated to the AIxLearning instance;
- `X-End-User-Id` with a stable opaque identifier;
- `Idempotency-Key` on material creation, session creation and turns.

Credentials must never be shared across production, corporate and staging.

### State owned by each system

AIxLearning should store:

- the GraphRAG `session_id`;
- GraphRAG material IDs;
- user-facing conversation and lesson records required by the platform;
- credit and business state.

GraphRAG stores:

- document bytes and derived indexes;
- owner-scoped material bindings;
- agent conversation state;
- the current lesson and agent traces.

After session creation, AIxLearning sends only the new turn. It must not resend
the complete history and complete file text on every call.

### Retry behavior

For the same logical job, Dramatiq must reuse the same idempotency key.

- `409 idempotency_in_progress`: wait and retry with the same key, or read the
  session;
- `Idempotent-Replayed: true`: accept the returned original result;
- `422 idempotency_key_reused`: treat as a client bug;
- `429` or `503`: honor `Retry-After` when present and retry safely.

### Streaming behavior

The Mercure adapter must:

- append `text` frames for the active draft;
- clear the preview when `draft` increases;
- discard the preview on `light_escalate`;
- replace all preview text with the final `result.markdown`;
- preserve phase events if the UI uses planner/retriever/critic information.

## Recommended acceptance sequence

1. Correct the documentation and freeze pilot terminology.
2. Configure and protect API staging.
3. Decide the anonymization path.
4. Export a version-pinned OpenAPI document for Dev.
5. Dev implements authentication, upload/poll, session persistence,
   idempotency and SSE-to-Mercure mapping.
6. Add required-document and structured-error hardening.
7. Run the API smoke script against staging.
8. Run cross-instance and cross-user isolation tests.
9. Run joint end-to-end tests for UDL and neuroscience.
10. Decide credits, exports, tracing and production identity.

## Final verdict

Angelo's implementation has closed most of the engineering gap described in
the original roadmap. It should be treated as an **API contract release
candidate**:

- implemented and credible;
- strongly aligned with the current AIxLearning architecture;
- backward compatible at the endpoint level;
- appropriate for Dev onboarding and protected staging;
- still dependent on security, privacy, provenance and joint product decisions
  before production freeze.

## Addendum — 20 October 2026: the judge in place of the Critic

Merged into `chore/repo-reorg` after this review, from
`feature/response-quality-improvement`. Additive: with `AIX_JUDGE_ENABLED`
unset or `false` nothing above changes.

With `AIX_JUDGE_ENABLED=true` the Critic leaves the chain. After the Writer, a
judge reads the whole lesson, corrects in the text only what is wrong
(replacements and rewrites of the activity around a problem, each verified), and
receives computed facts it does not have to work out (syllable splits, fraction
equivalences and quiz keys, arithmetic, minutes, pupils in the groups). A lesson
that ignores the attached documents still goes back to the Writer once, by a
check in code.

What changes for a client:

- **Turn result:** `critic` is `null`; a typed `judge` (`JudgeInfo`) carries
  `found`, `blocking`, `fixed[]` (each correction: `check`, `why`, `before`,
  `after`, `kind`, `teacher_material`), `points[]` (what could not be corrected),
  `reverted`, `seconds`, `error`. The same on `/agent/run`.
- **Streaming:** `judge_pending` and `judge` instead of `critic_pending` and
  `critic`; a pause of typically 5–15 s between the end of the text and the
  result. The final `result.markdown` is the corrected lesson and differs from
  the preview wherever a correction was made: the replace-the-preview rule above
  is now essential, not cosmetic.
- **Mercure adapter:** preserve `judge_pending`/`judge` as phase events, as for
  `critic`; a client should handle both configurations.
- **`lesson_meta.review_notes`:** now only what the judge could not correct
  (= `judge.points`).

The parity guard (`tests/unit/test_api_webui_parity.py`) covers the judge's card,
and `scripts/eval/api_smoke.py` passed end to end against a stack with the judge
on (upload, session, lesson, answer, hand-off, exports, media, catalog).
