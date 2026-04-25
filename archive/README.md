# archive/

Reserved location for **deprecated but historically interesting** modules
that have been replaced by newer implementations but should not be deleted
yet (e.g., for reference, A/B comparison, or rollback safety nets).

## Conventions

- Files placed here MUST have a `_old` or `_legacy` suffix
  (e.g. `graph_retriever_old.py`, `text2cypher_legacy.py`).
- Add a short header comment in each archived file explaining:
  1. Date archived
  2. What replaced it
  3. Why it was kept (and not deleted)
- Code under `archive/` is **not imported** by any production module.
- It is excluded from `ruff` and `mypy` checks (see `pyproject.toml`).
- It is **not** excluded from `pytest` collection — but no tests live here
  either.

## Status

Empty placeholder created in Phase 3B (repo reorganization, Apr 2026).
The legacy `_old` modules referenced in early planning notes were not
present in the working tree at reorg time, so this folder starts empty
and will be populated only when a deprecation event occurs.
