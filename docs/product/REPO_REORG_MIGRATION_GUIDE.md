# Repo Reorganization — Migration Guide

**Audience:** Angelo (and any future collaborator joining after `phase-3c-complete`)
**Date:** 25 April 2026
**Tag to check out for the new layout:** `phase-3c-complete`
**Source of truth (technical detail):** `CHANGELOG.md` sections "0a / 0b / 0c — Repository reorganization"

---

## TL;DR — what changed in 60 seconds

We turned `graphaixlearning` into a **modern installable Python project**. All
importable code now lives under a single top-level package `aix`, located at
`src/aix/`. The repo root no longer contains stray Python modules or
package directories.

```
BEFORE                                  AFTER
graphaixlearning/                       graphaixlearning/
├── config.py                           └── src/
├── graph_retriever.py                      └── aix/                  ← new package
├── context_builder.py                          ├── core/
├── text2cypher.py                              │   └── config.py
├── multilingual_text2cypher.py                 ├── retrieval/
├── query_metrics.py                            │   ├── graph_retriever.py
├── llm_chain.py                                │   ├── context_builder.py
├── agent/                                      │   ├── text2cypher.py
├── api/                                        │   ├── multilingual_text2cypher.py
├── domains/                                    │   └── query_metrics.py
└── ...                                         ├── generation/
                                                │   └── llm_chain.py
                                                ├── agent/
                                                ├── api/
                                                └── domains/
```

**What you have to do once after pulling:**

```bash
git pull
pip install -e ".[dev]"     # registers the new src/aix/ package as editable
```

That is it. FastAPI, Streamlit, the CLI, pytest, Neo4j, the KG data, the
`.env` file, every API endpoint contract — all unchanged in behaviour.

---

## Why we did this (one paragraph)

A flat root layout works fine for prototypes but breaks down as soon as you
need: an installable package, predictable test discovery, a Docker image
that doesn't rely on `cwd`, CI that doesn't rely on `sys.path` hacks,
multiple entry points (FastAPI + Streamlit + CLI), and a single canonical
home for application code. The standard Python answer is the **`src/`
layout** — it is what `pip`, `setuptools`, `pytest`, `mypy`, `ruff`, and
every modern packaging tool expect. We now follow it. The `aix` package
name is short, project-specific, and lets us import everything as
`aix.<subpackage>.<module>`.

---

## Quick reference — old path → new path

### Source modules (the 7 ex-root files)

| Old (pre-3C, root level)         | New (`src/aix/...`)                            |
|----------------------------------|------------------------------------------------|
| `config.py`                      | `src/aix/core/config.py`                       |
| `graph_retriever.py`             | `src/aix/retrieval/graph_retriever.py`         |
| `context_builder.py`             | `src/aix/retrieval/context_builder.py`         |
| `text2cypher.py`                 | `src/aix/retrieval/text2cypher.py`             |
| `multilingual_text2cypher.py`    | `src/aix/retrieval/multilingual_text2cypher.py`|
| `query_metrics.py`               | `src/aix/retrieval/query_metrics.py`           |
| `llm_chain.py`                   | `src/aix/generation/llm_chain.py`              |

### Source packages (the 3 ex-root directories)

| Old (pre-3C) | New (`src/aix/...`) |
|---|---|
| `agent/`   | `src/aix/agent/`   |
| `api/`     | `src/aix/api/`     |
| `domains/` | `src/aix/domains/` |

### Imports — old → new

| Old import                                 | New import                                      |
|--------------------------------------------|-------------------------------------------------|
| `from config import config`                | `from aix.core.config import config`            |
| `from graph_retriever import …`            | `from aix.retrieval.graph_retriever import …`   |
| `from context_builder import …`            | `from aix.retrieval.context_builder import …`   |
| `from text2cypher import …`                | `from aix.retrieval.text2cypher import …`       |
| `from multilingual_text2cypher import …`   | `from aix.retrieval.multilingual_text2cypher import …` |
| `from query_metrics import …`              | `from aix.retrieval.query_metrics import …`     |
| `from llm_chain import …`                  | `from aix.generation.llm_chain import …`        |
| `from agent import AgentOrchestrator`      | `from aix.agent import AgentOrchestrator`       |
| `from agent.graph.state import …`          | `from aix.agent.graph.state import …`           |
| `from api.main import app`                 | `from aix.api.main import app`                  |
| `from api.schemas.models import …`         | `from aix.api.schemas.models import …`          |
| `from domains.udl_domain import …`         | `from aix.domains.udl_domain import …`          |

### Entry points (no longer "python <file>.py at repo root")

| Old command                          | New command                                        | `make` shortcut |
|--------------------------------------|----------------------------------------------------|-----------------|
| `uvicorn api.main:app …`             | `uvicorn aix.api.main:app --reload --port 8000`    | `make api`      |
| `streamlit run streamlit_app.py`     | `streamlit run apps/streamlit/main.py`             | `make streamlit`|
| `python test_agent.py`               | `python apps/cli/run_agent.py`                     | `make agent`    |
| `pytest`                             | `pytest tests/ -v`                                 | `make test`     |
| `python train_node2vec.py`           | `python scripts/ml/train_node2vec.py neuro`        | —               |
| `python data_ingestion_neo4j.py …`   | `python scripts/ingest/data_ingestion_neo4j.py …`  | —               |

### Data & artifacts (Phase 3B layout — predates 3C, included here for completeness)

| Old path (pre-3B)                          | New path                                  |
|--------------------------------------------|-------------------------------------------|
| `models/{*_node2vec_*}`                    | `artifacts/node2vec/`                     |
| `models/embeddings_cache/`                 | `artifacts/embeddings_cache/`             |
| `JSON_reference.json`                      | `data/reference/JSON_reference.json`      |
| `kg_neuro_neo4j.json` (root)               | `data/kg/neuro/kg_neuro_neo4j.json`       |
| `kg_udl_neo4j.json` (UDLdata/)             | `data/kg/udl/kg_udl_neo4j.json`           |
| `kg_neuro_media_mapping.json` (root)       | `data/media/kg_neuro_media_mapping.json`  |
| `kg_neuro_resources.json` (root)           | `data/media/kg_neuro_resources.json`      |

---

## How to fix a branch that still uses old imports

If you are on a feature branch that was forked **before** `phase-3c-complete`
and you start getting `ModuleNotFoundError: No module named 'config'` (or
`'agent'`, `'api'`, `'domains'`, etc.), do this:

```bash
# 1. Make sure you have the post-reorg version of the script
git fetch origin
git checkout phase-3c-complete -- scripts/_phase3c_rewrite_imports.py

# 2. Dry-run first to see what would change (no files written)
python scripts/_phase3c_rewrite_imports.py --dry-run --verbose

# 3. If the diff looks right, run it for real
python scripts/_phase3c_rewrite_imports.py --verbose

# 4. Re-install the project in the venv
pip install -e ".[dev]"

# 5. Verify
python -c "import aix; from aix.core.config import config; print('OK')"
pytest tests/ --collect-only -q
```

The script is **deterministic and idempotent**: running it twice produces no
extra changes the second time. It rewrites every `from <old>` /
`import <old>` statement under `src/`, `apps/`, `scripts/`, and `tests/` to
the matching `aix.*` path, using the same `ROOT_MAP` that built the canonical
commit `1a109cf`.

---

## Common pitfalls and how to fix them

### `ModuleNotFoundError: No module named 'aix'`

Cause: you pulled the new layout but did not re-install the package.

```bash
pip install -e ".[dev]"
```

If it still fails, clear stale Python bytecode caches inside the venv:

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ |
    Remove-Item -Recurse -Force
```

### `ModuleNotFoundError: No module named 'config'` (or `agent`, `api`, `domains`, …)

Cause: your branch still has old root-level imports.

Fix: run the rewrite script (see "How to fix a branch" above).

### Streamlit / FastAPI fails to start with import errors

Cause: you ran a command from the wrong directory, or the venv is not
active. Both apps require the editable install to be visible on `sys.path`.

```bash
# Activate venv (Windows PowerShell)
..\venv\Scripts\Activate.ps1
pip install -e ".[dev]"
make api          # or: make streamlit, make agent
```

### My IDE marks `aix.*` imports as unresolved

Cause: the IDE picked up the wrong Python interpreter.

Fix: point the IDE at `..\venv\Scripts\python.exe` (the same one that ran
`pip install -e .[dev]`). VS Code: `Python: Select Interpreter` from the
command palette.

---

## What stayed the same (you do not need to relearn anything here)

- **API contract**: all endpoints (`/api/v1/context`, future `/api/v1/agent/lesson`)
  keep their request/response shapes. The `EducationalProfile` work in
  CORE 1 #2.5 lands as documented.
- **Neo4j queries, KG schema, domain configs (UDL + Neuro)**: 100% unchanged
  in behaviour. Only their file path moved.
- **Environment variables**: `.env.example` is the source of truth and was
  already aligned with the new artifact paths during Phase 3B
  (`NODE2VEC_MODEL_DIR=./artifacts/node2vec`).
- **Git history**: every moved file is a `git mv` at 88-100% similarity, so
  `git log --follow src/aix/agent/orchestrator.py` walks back to the
  original `agent/orchestrator.py` blame without breakage.
- **Docker, Makefile, CI**: updated to point to `aix.api.main:app` and
  `mypy src/aix/`. No behaviour change for users — only the internal
  invocation path changed.
- **The Agentic GraphRAG implementation plan** in
  `docs/product/ClickUp_Agentic_GraphRAG_Update.md`: subtask scopes,
  priorities, dependencies, and effort estimates are all unchanged. Only
  the file paths inside that doc were refreshed for the new layout.

---

## Where to look when something is unclear

| Question                                         | Read this                                                            |
|--------------------------------------------------|----------------------------------------------------------------------|
| What exactly changed in each phase?              | `CHANGELOG.md` — sections "0a", "0b", "0c"                           |
| What does the new project tree look like?        | `README.md` — section "Project Structure"                            |
| How do I run the API / Streamlit / CLI / tests?  | `README.md` — section "Project Structure" (entry-points cheat sheet) |
| What's the agentic GraphRAG roadmap?             | `docs/product/ClickUp_Agentic_GraphRAG_Update.md`                    |
| Why did we adopt the `src/` layout?              | This guide, "Why we did this"                                        |
| Anything weird with my local install?            | This guide, "Common pitfalls"                                        |
| The rewrite script — what does it match exactly? | `scripts/_phase3c_rewrite_imports.py` — `ROOT_MAP` constant          |

---

## Final sanity test (5 minutes)

After `git pull` + `pip install -e ".[dev]"`, run these and you should be good:

```bash
# 1. Package import works
python -c "import aix; from aix.core.config import config; from aix.agent import AgentOrchestrator; print('aix', aix.__version__)"
# Expected: aix 0.2.0

# 2. Tests are discovered
pytest tests/ --collect-only -q
# Expected: 17 tests collected

# 3. FastAPI boots and Swagger UI loads
make api
# Then open http://localhost:8000/docs in your browser

# 4. Streamlit boots
make streamlit
# Then open http://localhost:8501
```

If all four pass, you are fully on the new layout. Welcome to `phase-3c-complete`. ✅
