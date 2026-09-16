"""One embedding column that works on both databases.

Production is PostgreSQL with pgvector, where the column is a real
``vector(1024)`` and the index is what makes retrieval fast. Development here is
SQLite, which has no vector type at all. The alternative to this file is two
schemas — and a test suite that therefore never exercises the column production
uses.

So: ``vector(N)`` on PostgreSQL, a JSON array everywhere else. Thirty lines
rather than a dependency, and the same ORM model on both. The similarity search
itself still differs by dialect — pgvector's ``<=>`` operator on PostgreSQL, a
Python cosine over the owner's own chunks on SQLite — and that lives in the
retriever, where it can be read next to the query it changes.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator, UserDefinedType


class _PgVector(UserDefinedType):
    """The pgvector column type, in the little of it that is needed here."""

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **_kw: Any) -> str:
        return f"vector({self.dimension})"

    def bind_processor(self, _dialect: Any):
        def process(value: Optional[list[float]]) -> Optional[str]:
            if value is None:
                return None
            return "[" + ",".join(repr(float(v)) for v in value) + "]"

        return process

    def result_processor(self, _dialect: Any, _coltype: Any):
        def process(value: Any) -> Optional[list[float]]:
            if value is None or isinstance(value, list):
                return value
            return [float(v) for v in str(value).strip("[]").split(",") if v.strip()]

        return process


class EmbeddingVector(TypeDecorator):
    """``vector(N)`` on PostgreSQL, JSON array on SQLite."""

    impl = JSON
    cache_ok = True

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def load_dialect_impl(self, dialect: Any):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(_PgVector(self.dimension))
        return dialect.type_descriptor(JSON())
