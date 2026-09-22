# Review: `feature/advanced-input-and-export` — Phase 0 readiness and backward compatibility

**Date:** 15 September 2026  
**Branch reviewed:** `fem/feature/advanced-input-and-export` @ `730fed2`  
**Baseline:** `chore/repo-reorg` / `b2b7742` (the WebUI + Agentic GraphRAG we already had)  
**Author:** Louis (review of Angelo’s implementation)

This is the written report of the review. A compact interactive version can be opened beside chat: `canvases/angelo-phase0-implementation-review.canvas.tsx`.

---

## 1. Verdict

**Phase 0 is not closed. Do not proceed as if the next phase can start on a green foundation.**

Angelo has built a large, serious piece of work: Alembic, CSRF, sanitization, owner-scoped material storage, a durable job queue, shadow uploads, DocumentIR parsing with OCR, and a real PDF/outline export. Several of those pieces are **good**. They are not yet a complete, backward-compatible Fase 0.

Three facts decide the verdict:

1. A **real pre-feature database cannot be adopted**. The migrator compares the live schema against *current* models, sees the new material tables as “missing”, logs an error, stamps nothing, and leaves `alembic_version` empty. Reproduced on a genuine `b2b7742` SQLite schema: `revision=None`, no `material`, no `document_ir`.
2. **Default runtime is not the previous WebUI.** With `AIX_MATERIALS_ENABLED` off, the paperclip still accepts DOCX/PPTX/images, runs the new parser (and OCR on the request thread), the classroom validator is **on by default**, and the UI always offers PDF + Scaletta. The architecture required: *all new flags false → previous behaviour*.
3. The branch has **already entered Fase 2 (parsing) and Fase 3 (local PDF/outline)**, plus an always-on GraphRAG validator that was never a Fase 0 item. Private RAG, previews and Google export are still missing — correctly — but the skipped Fase 0 gates are still blockers.

**Recommendation:** keep the branch, review it commit-by-commit, fix the blockers below, then finish shadow mode (Fase 1) with flags off in production. Do not merge as-is.

---

## 2. Scope of the review

| Item | Value |
|---|---|
| Files changed | **163** (65 added, 98 modified) |
| Diff size | **+18,795 / −2,622** |
| Commits ahead of baseline | **30**, all Angelo-Casali |
| Isolated checkout | `graphaixlearning-advanced-review` worktree (current `chore/repo-reorg` tree was not switched) |
| Lint | `ruff check` and `ruff format --check` passed |
| Branch unit tests | **583 passed, 4 failed, 2 skipped** (see §7) |
| Pre-existing unit tests vs new code | **83/83 passed** when materials / validator / corrective RAG / conditional retrieval were forced off |
| API + MCP tests | **26/26 passed** |

Compared documents: `Advanced_Input_and_Export_Architecture.md` §5.2–5.4, §13–14, Appendix B–C; strategy doc C.1 (Punti 1 e 2).

---

## 3. Is it backward compatible with the WebUI / Agentic GraphRAG we had?

**Partially. Compatible if optional flags are forced off *and* you ignore the new parser/validator/export defaults. Not compatible under the code as it would actually start today.**

### 3.1 What still matches the previous product

- Uploads are **not** written into the shared Knowledge Graph. They still become `teacher_provided_context` for the Writer only.
- Chat, SSE lesson generation, Markdown export, TXT export and `/print` still exist. MD/TXT bytes are tested as unchanged (`test_markdown_and_text_are_unchanged`).
- Knowledge Graph retrieval, media ranking and MCP/API agent contracts still work (26 API/MCP tests green).
- Conditional retrieval (`AIX_CONDITIONAL_RETRIEVAL_ENABLED`) and Corrective RAG default **off**, so `plan → retrieve → write` is still the retrieval spine — *except* the new `validate` node, which is always in the graph.
- CSRF is additive for HTML/htmx; templates send the token.
- A Critic-triggered revision now **keeps** the teacher file (`teacher_provided_context` is passed to `WriterAgent.revise()`). That is a fix of a previous defect, not a regression.

### 3.2 What is not the previous behaviour

| Surface | Before (`b2b7742`) | Now (`730fed2`) | Compatible? |
|---|---|---|---|
| Paperclip formats | PDF, TXT, MD only; pypdf / UTF-8 decode | Also DOCX, PPTX, images; magic-byte checks; OCR | **No** — not behind `AIX_MATERIALS_ENABLED` |
| Corrupt PDF | Could 500 | User-safe 422 | Better, but still a behaviour change |
| Agent graph | `plan → retrieve → write → critique` | Always `write → validate → critique`; validator **default on** | **No** |
| Lesson mentioning “DSA” with no profile | Allowed | Blocked by `bes_label_drift` | **No** (reproduced) |
| PDF button | Browser print dialog | Server-rendered PDF download | Additive, always on |
| Scaletta | Absent | Always offered | Additive, always on |
| Follow-up KG search | Every turn | Reuse possible; flag default off | Yes if flag stays off |
| Classroom profile in prompts | Five fields | Full constraints + labels | Improvement; prompt change |

The 83 old unit tests passing is useful but **narrow**: that suite never covered upload formats, export PDF, or the validator. Compatibility of those surfaces was checked by reading routes and by targeted new tests.

---

## 4. Phase 0 gates

Architecture §5.2 / §14.2.

| Gate | Status | Evidence |
|---|---|---|
| Alembic on empty DB | **Pass** | Fresh DB upgrades to head in unit tests |
| Existing DBs reach head | **Fail** | Adoption uses `Base.metadata` *including* new tables. Real legacy DB: left untouched, `revision=None` |
| Legacy upload survives container replace | **Pass** | `app_uploads` → `data/webui/uploads`; image creates the dir before chown |
| New material blobs survive replace | **Fail** (shadow) | Default root `data/webui/materials`; compose has no volume |
| CSRF + sanitizer | **Partial** | `/webui` + `/auth` covered. Cookie-auth `POST /api/v1/agent/*` is outside CSRF. `GET /auth/logout` still changes state |
| Zero IDOR on current lesson routes | **Pass** | Owner filter + `test_idor.py` |
| Binding ownership complete | **Fail** (latent) | `bind()` does not check that the lesson belongs to the same owner |
| Feature flags centralised | **Fail** | Single `AIX_MATERIALS_ENABLED`. Architecture names `AIX_MATERIALS_PIPELINE_ENABLED`, `AIX_MATERIALS_SHADOW_MODE`, parse/OCR flags. **None of these appear in `.env.example`** |
| Worker recovery | **Partial** | Claim/lease/retry tested on SQLite only. Parse can outlive 120s lease; complete() can then run without a live lease. No PostgreSQL test |
| Flag off = previous behaviour | **Fail** | Parser, new formats, validator, PDF/outline all run with materials flag off |
| C.1 decisions closed | **Unverified** | Changelog cites `docs/product/Decisioni_Gate_Fase_0.md`; **file is not on the branch** |

---

## 5. What is OK (keep)

These pieces are aligned with Fase 0 intent and should stay:

- Alembic baseline + additive revisions (the *idea* is right; adoption of old DBs is not).
- Durable local storage: SHA-256 keys, fsync, checksum on read, S3 class explicitly unimplemented.
- Owner-scoped repositories (`owner_id` required).
- In-DB job queue with lease, backoff, `SKIP LOCKED` on Postgres as an optimisation.
- Shadow adapter: legacy upload commits first; shadow failure does not fail the teacher request; no lesson binding in shadow (honest).
- HTML sanitization (`nh3`) and safer error pages.
- CSRF double-submit HMAC for WebUI forms and htmx.
- Graph compiled once per process; input schema so follow-ups can keep the retrieval corpus.
- LLM timeout + RetryPolicy on plan/write/critique.
- Intake allowlist, magic bytes, macro/OLE refusal, zip-bomb bounds (good security, but they currently apply to the *legacy* paperclip too).
- MD/TXT export left on the old path; PDF/outline are extra formats on the same owner-checked route.
- Tesseract ITA+ENG in the Docker image; OCR absence recorded as `ocr_unavailable` rather than a crash.
- Worker parse moved to `asyncio.to_thread` after earlier review (upload parse also uses `to_thread`).

---

## 6. What is not OK for Phase 0 / next phase

**Must fix before calling Fase 0 done**

1. **DB adoption.** Stamp existing databases at baseline `c8d88f37e59c`, then `upgrade head`. Never `stamp head` when additive tables are missing. Tests today seed “legacy” from *current* `Base.metadata`, so they hide the production case.
2. **Production volume** for `data/webui/materials` (and `mkdir` in the image, same pattern as uploads).
3. **Gating.** With flags off: paperclip = PDF/TXT/MD + old extract; no OCR on the request; no DOCX/PPTX/images; validator off or no-op; PDF/outline either flagged or clearly accepted as an additive Punto 2A delivery.
4. **Validator default.** `AIX_VALIDATOR_ENABLED` defaults to `true`. Without a profile, `bes_label_drift` blocks ordinary Italian text containing DSA/ADHD/etc. Default it **false** until rules are proven, or skip BES rules when no disabilities were declared.

**Should fix before merge / before enabling shadow in production**

5. Put the real env vars in `.env.example`. Split pipeline vs shadow.
6. CSRF for cookie-authenticated `/api/v1` *or* stop accepting the WebUI cookie on those POSTs.
7. Change logout to POST (or accept CSRF on GET logout — POST is the correct fix).
8. Check `lesson.owner_id` in `BindingRepository.bind`.
9. Renew the job lease during parse; do not `complete()` if the lease is gone.
10. Unreadable documents → `failed`, not `ready_with_warnings`.
11. Add at least one PostgreSQL migration + claim test (Phase 0 explicitly required both dialects).
12. Topology test currently fails: extra edge `validate → __end__`. Either the graph or the test is wrong; CI on GitHub will see this if the job is run.
13. Bound upload size *before* reading the body (Caddy `request_body max_size` + FastAPI limit). OCR of a 30-page scan can exceed the 120s job lease.
14. Surface validator failures in the WebUI; today they can fail the graph while the teacher still sees a successful run.

**Not Phase 0 — already started, must not be mistaken for “Fase 0 complete”**

- DocumentIR + OCR + chat formats (architecture Fase 2 / strategy Fase 1).
- PDF + outline (architecture Fase 3 / strategy Punto 2A).
- Classroom validator + full profile renderer (WebUI improvement Q1, not in Fase 0 gates).

**Still missing for Punto 1 / next architecture phase**

Private RAG, chunks/embeddings, citations with page/slide, authorised original download, page preview, lesson–material binding in the agent, Google OAuth/Slides/Classroom.

### 6.1 Parsing and export follow-up (Punto 1 / 2A)

These are not Phase 0 gates, but they are already on the branch and they affect whether the teacher-facing paperclip and PDF can be trusted.

**P1 — fix before enabling materials / OCR in any environment**

| Issue | Where | Why it matters |
|---|---|---|
| Unbounded upload parse | `lessons/uploads.py`, `parsing/ocr.py`; Caddy has no `request_body max_size` | The handler reads the whole file into memory, then OCR can run 30 pages × 45s on the request thread. No per-user rate limit. Easy DoS of the WebUI process. |
| Job lease vs OCR | `jobs.py` 120s lease; OCR worst case ~1350s | A second worker can claim the same job after 2 minutes. Handler `complete()` and worker `complete()` can race. |
| Parser profile vs IR | `extract_document` records `pypdf@1`; IR is `ir@1` | Provenance on the stored document does not match the artefact the agent later consumes. |
| Validator failures invisible | `nodes.py` `validate_node`; classroom templates | Graph can fail or rewrite while the WebUI still looks like a normal success. |
| Short PDF pages discarded | `parsing/pdf.py` skips pages with &lt; 25 characters | Sparse slides, title pages, and image-heavy PDFs silently lose content. |

**P2 — track, do not block Phase 0 close-out**

- BES vs “no device” rules can contradict each other (`bes_label_drift` vs device constraints).
- Multipage TIFF: only the first page is OCR’d.
- OCR language packs (`ita+eng`) are not verified at startup; CI has no Tesseract so OCR paths are untested there.
- Unicode download filenames and silent drop of unsupported PDF glyphs.
- Prompt injection from uploaded documents into `teacher_provided_context` (expected residual until a dedicated sanitiser exists).
- Export fonts are not declared in package data; PDF generation can fail in a clean install.

---

## 7. Tests actually run

| Suite | Result | Note |
|---|---|---|
| `tests/unit` on 730fed2 | 583 passed, 4 failed, 2 skipped | 3 failures were `No module named pptx` in this laptop venv (not the branch). After installing `python-pptx`, those 3 passed. **1 remaining failure:** `test_default_topology_wires_the_linear_path_and_both_loops` extra edge `('validate', '__end__')`. OCR tests skipped without Tesseract on Windows. |
| Pre-existing `tests/unit` from b2b7742 against 730fed2 | 83 passed | Flags forced off |
| `tests/api` + `tests/mcp_server` | 26 passed | Agent API / MCP surface still up |
| PostgreSQL | Not run | No Postgres in this review environment; branch tests are SQLite-only |

---

## 8. File inventory

163 paths vs `b2b7742`. **A** = added, **M** = modified.

### 8.1 Phase 0 foundations

| Status | Path | What it is |
|---|---|---|
| A | `alembic.ini` | Alembic config |
| A | `alembic/README` | Stock readme |
| A | `alembic/env.py` | App URL + batch mode |
| A | `alembic/script.py.mako` | Revision template |
| A | `alembic/versions/2026_09_08_1251-c8d88f37e59c_baseline_webui_schema.py` | Baseline |
| A | `alembic/versions/2026_09_09_1511-71b24d63b410_add_material_versions_and_durable_blobs.py` | material / version / blob / binding |
| A | `alembic/versions/2026_09_09_1558-48f042517314_add_the_durable_ingestion_job_queue.py` | ingestion_job |
| M | `src/aix/webui/db.py` | `init_db` migrate/adopt (**adoption bug**) |
| A | `src/aix/webui/auth/csrf.py` | CSRF middleware |
| M | `src/aix/webui/auth/routes.py` | Token wiring; logout still GET |
| M | `src/aix/webui/routes.py` | CSRF include |
| A | `src/aix/webui/sanitize.py` | nh3 sanitizer |
| A | `src/aix/webui/errors.py` | Safe errors / incident ids |
| A | `src/aix/webui/materials/__init__.py` | Package |
| A | `src/aix/webui/materials/models.py` | ORM + later DocumentIR |
| A | `src/aix/webui/materials/repository.py` | Owner-scoped access |
| A | `src/aix/webui/materials/storage.py` | LocalDurableStorage + S3 scaffold |
| A | `src/aix/webui/materials/jobs.py` | JobDispatcher |
| A | `src/aix/webui/materials/worker.py` | In-process worker |
| A | `src/aix/webui/materials/shadow.py` | Best-effort shadow upload |
| A | `src/aix/webui/materials/handlers.py` | Was pypdf measure; now DocumentIR |
| M | `src/aix/webui/lessons/routes.py` | Shadow + later parse/export |
| M | `src/aix/webui/templates/_base.html` | CSRF meta / htmx |
| M | `src/aix/webui/templates/pages/auth_login.html` | CSRF field |
| M | `src/aix/webui/templates/pages/auth_register.html` | CSRF field |
| M | `src/aix/webui/templates/pages/lesson_new.html` | CSRF field |
| M | `deploy/docker-compose.prod.yml` | `app_uploads` volume only |
| M | `Dockerfile` | uploads dir + Tesseract |
| M | `src/aix/api/main.py` | Migrations + worker lifespan |
| M | `CHANGELOG.md` | Fase 0 notes; cites missing decisions file |

### 8.2 Backward-compat / GraphRAG fixes (mostly good)

| Status | Path | What it is |
|---|---|---|
| M | `src/aix/agent/graph/state.py` | `AgentInput`, reducers, persist retrieval across turns |
| M | `src/aix/agent/graph/lesson_planner_graph.py` | Compile-once; conditional retrieval; **always-on validate** |
| M | `src/aix/agent/graph/nodes.py` | Revision keeps attachment; reuse; validator node |
| M | `src/aix/agent/graph/checkpointer.py` | Mechanical / small |
| M | `src/aix/agent/orchestrator.py` | Uses process graph |
| M | `src/aix/webui/agent/service.py` | Sanitize HTML; retrieval reuse card; profile |
| M | `src/aix/agent/agents/writer_agent.py` | `revise()` gets teacher context; full profile |
| M | `src/aix/agent/agents/planner_agent.py` | Classroom constraints; retrieval reuse prompt |
| A | `src/aix/agent/prompts/profile_renderer.py` | Full classroom as constraints |
| M | `src/aix/agent/prompts/planner_prompt.py` | Follow-up retrieval question |
| M | `src/aix/core/config.py` | LLM timeout / retry |
| M | `src/aix/webui/templates/partials/chat_retriever_card.html` | Reuse messaging |

### 8.3 Phase 2 parsing (started early)

| Status | Path | What it is |
|---|---|---|
| A | `alembic/versions/2026_09_15_1800-5d1c7a2e9b40_add_document_ir.py` | `document_ir` |
| A | `src/aix/webui/materials/parsing/__init__.py` | Package |
| A | `src/aix/webui/materials/parsing/intake.py` | Allowlist, magic, zip/macro/page limits |
| A | `src/aix/webui/materials/parsing/ir.py` | DocumentIRV1 |
| A | `src/aix/webui/materials/parsing/ocr.py` | Tesseract + pypdfium2 |
| A | `src/aix/webui/materials/parsing/parsers.py` | PDF/DOCX/PPTX/text/image |
| M | `src/aix/webui/lessons/uploads.py` | **Legacy path now calls the new parser (ungated)** |
| M | `src/aix/webui/templates/partials/chat_input.html` | Accepts Word, PPT, images |
| M | `src/aix/webui/templates/partials/chat_attachments.html` | Parse quality on chip |
| A | `scripts/eval/__init__.py` | Eval package |
| A | `scripts/eval/benchmark_parsers.py` | Parser benchmark |

### 8.4 Phase 3 local export (started early)

| Status | Path | What it is |
|---|---|---|
| A | `src/aix/export/__init__.py` | Package |
| A | `src/aix/export/lesson_document.py` | LessonDocumentV1 |
| A | `src/aix/export/export_ir.py` | ExportIRV1 |
| A | `src/aix/export/pdf.py` | ReportLab PDF |
| A | `src/aix/export/outline.py` | Timed outline |
| A | `src/aix/export/fonts/NotoSans-*.ttf` + `OFL.txt` | Bundled fonts |
| M | `src/aix/webui/templates/partials/chat_lesson_card.html` | PDF + Scaletta buttons |
| M | `src/aix/webui/static/css/aix-brand.css` | Small style |

### 8.5 Out of Phase 0 (GraphRAG product change)

| Status | Path | What it is |
|---|---|---|
| A | `src/aix/agent/validation/__init__.py` | Package |
| A | `src/aix/agent/validation/models.py` | Violations |
| A | `src/aix/agent/validation/rules.py` | Duration, BYOD, wifi, furniture, BES codes |
| A | `src/aix/agent/validation/validator.py` | **Default ON** |

### 8.6 Tests added or substantially extended

| Status | Path |
|---|---|
| A | `tests/unit/agent_doubles.py` |
| A | `tests/unit/test_benchmark_parsers.py` |
| A | `tests/unit/test_conditional_retrieval.py` |
| A | `tests/unit/test_constraint_validator.py` |
| A | `tests/unit/test_csrf.py` |
| A | `tests/unit/test_db_migrations.py` |
| A | `tests/unit/test_document_parsing.py` |
| A | `tests/unit/test_export_document.py` |
| A | `tests/unit/test_export_pdf.py` |
| A | `tests/unit/test_graph_topology.py` |
| A | `tests/unit/test_idor.py` |
| A | `tests/unit/test_ingestion_jobs.py` |
| A | `tests/unit/test_llm_retry.py` |
| A | `tests/unit/test_material_repository.py` |
| A | `tests/unit/test_material_storage.py` |
| A | `tests/unit/test_profile_reaches_the_prompts.py` |
| A | `tests/unit/test_profile_renderer.py` |
| A | `tests/unit/test_reuse_card.py` |
| A | `tests/unit/test_revision_keeps_material.py` |
| A | `tests/unit/test_sanitize_and_errors.py` |
| A | `tests/unit/test_shadow_upload.py` |
| A | `tests/unit/test_state_partition.py` |
| A | `tests/unit/test_validator_in_the_graph.py` |
| M | `tests/unit/test_docker_assets_manifest.py` |
| M | `tests/unit/test_init_db_locking.py` |
| M | `tests/unit/test_json_parse_hardening.py` |
| M | `tests/unit/test_plan_node_profile_wins_over_history.py` |
| M | `tests/unit/test_retriever_payload_outcome.py` |
| M | `tests/api/test_agent_routes.py` |
| M | `tests/mcp_server/*` (5 files + conftest) |
| M | `tests/integration/*` (7 files; CI now ignores this folder) |

### 8.7 Mechanical / tooling (mostly ruff reformat, not product behaviour)

| Status | Path |
|---|---|
| M | `.env.example` (LLM timeout / conditional retrieval only — **no materials flags**) |
| A | `.git-blame-ignore-revs` |
| M | `.github/workflows/ci.yml` |
| M | `.gitignore` |
| M | `pyproject.toml` |
| M | `requirements.txt` |
| M | `requirements.lock.txt` |
| M | `apps/cli/run_agent.py` |
| M | `apps/streamlit/main.py` |
| M | `src/aix/api/routes/agent.py` |
| M | `src/aix/api/schemas/models.py` |
| M | `src/aix/core/concurrency.py` |
| M | `src/aix/core/run_registry.py` |
| M | `src/aix/agent/media/live_media.py` |
| M | `src/aix/agent/media/media_cache.py` |
| M | `src/aix/agent/media/media_config.py` |
| M | `src/aix/agent/media/media_ranker.py` |
| M | `scripts/_phase3c_rewrite_imports.py` |
| M | `scripts/audit/*` (3) |
| M | `scripts/data_prep/*` (5) |
| M | `scripts/diagnostic/*` (6) |
| M | `scripts/ingest/*` (3) |
| M | `scripts/md_to_docx.py` |
| M | `scripts/media_pool/**` (9) |
| M | `scripts/ml/*` (3) |
| M | `scripts/ops/*` (4) |

---

## 9. Commit map (runtime-relevant)

Oldest → newest on the branch after `b2b7742`:

| SHA | Subject | Phase |
|---|---|---|
| ad52985 | ruff lint | Mechanical |
| f19449b / f8fc611 / b7ae51c | pin ruff + reformat | Mechanical |
| 2a371f7 | installable deps | Tooling |
| 352df91 / 0fed375 / e64480e | graph tests, compile-once, input schema | GraphRAG fix — good |
| f69a7aa | LLM timeout/retry | GraphRAG fix — good |
| c5338dc / 8e540e5 | conditional retrieval | Flagged off — good |
| 3ca2e0b | revision keeps material | Compat fix — good |
| 21dfd53 | persist uploads volume | Fase 0 — good |
| 8648795 / ca0c124 | Alembic | Fase 0 — adoption logic wrong |
| 43514d8 | changelog | Docs |
| 51dbcb5 / 0d17c57 | CI skip integration; test isolation | Tooling |
| e997574 | CSRF | Fase 0 — partial |
| 03748aa | storage + models | Fase 0 — good |
| d999e52 | sanitize + errors | Fase 0 — good |
| a3ec87b | job queue | Fase 0 — good |
| 50473c3 / 0850bfe | shadow + test isolation | Fase 1 start |
| 628baaa | full classroom in prompts | Extra, useful |
| fb95e51 | validator **default on** | Extra — breaks old GraphRAG defaults |
| ec917a6 | parse/OCR/export deps | Fase 2–3 |
| f8150e1 | PDF + outline | Fase 3 / Punto 2A |
| 387034a / 730fed2 | DocumentIR + chat formats | Fase 2 / Punto 1 — **ungated on legacy upload** |

---

## 10. Suggested message to Angelo

Fase 0 machinery is in good shape conceptually, but we cannot treat the gate as passed:

- existing production DBs will not migrate;
- shadow files are not durable in compose;
- the paperclip and the validator already change the live product with the materials flag off.

Please fix those four items (DB stamp-at-baseline, materials volume, gate the parser, validator default false) before we review Fase 1 as “done” or start private RAG. PDF/outline can stay as an additive Punto 2A slice once MD/TXT remain the default path.

---

## 11. Sources

- Isolated worktree `730fed2` vs `b2b7742`
- `docs/product/Advanced_Input_and_Export_Architecture.md`
- `docs/product/AI_Strategy_Advanced_Inputs_and_Exports.md` (C.1)
- Reproduced migration on a `b2b7742` schema
- Reproduced `bes_label_drift` with `profile=None`
- Unit / API / MCP runs listed in §7
