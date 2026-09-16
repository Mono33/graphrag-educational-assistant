"""Embeddings for the teacher's own material (§8.4).

§8.4 asks for a local multilingual model by default, and this pilot cannot run
one: the VM is 2 vCPU and 4 GB shared with Postgres, and bge-m3 alone needs
~2.2 GB of RAM to serve. The deviation was decided explicitly on 2026-09-15 —
``baai/bge-m3`` through OpenRouter, on an account with **Zero Data Retention**
enabled, which covers embedding models too — and it is recorded here rather than
inherited silently, because §3.1.3 is private-by-default and this is the one
place where the teacher's text leaves the machine.

What that decision requires of this module:

* the model, its dimension and its provider are **recorded on every chunk**
  (``material_chunk.embedding_model``), so a later change is a visible
  re-indexing rather than a silent mix of two vector spaces;
* the dimension is **verified on every response**, never assumed. A model that
  quietly returns 1536 where the column holds 1024 would fail at the database
  with a message about arrays, far from the cause;
* **batches**, because one HTTP request per chunk turns a 40-page document into
  a few hundred round trips on the pilot's single worker.

Measured 2026-09-16 against OpenRouter: ``baai/bge-m3`` → 1024 dimensions.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "baai/bge-m3"
DEFAULT_DIMENSION = 1024

#: One request per batch. Small enough that a failure costs little work, large
#: enough that a long document is a handful of calls.
BATCH_SIZE = 32


class EmbeddingError(RuntimeError):
    """The embeddings could not be produced, or came back the wrong shape."""


def private_rag_enabled() -> bool:
    """Off by default: with the flag down nothing is chunked, embedded or sent."""
    return (os.getenv("AIX_PRIVATE_RAG_ENABLED") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def embedding_model() -> str:
    return (os.getenv("AIX_PRIVATE_EMBEDDING_MODEL") or DEFAULT_MODEL).strip()


def embedding_dimension() -> int:
    raw = os.getenv("AIX_PRIVATE_EMBEDDING_DIMENSION")
    try:
        return int(raw) if raw else DEFAULT_DIMENSION
    except ValueError:
        return DEFAULT_DIMENSION


def _client():
    from aix.config.settings import app_config

    return app_config.openai.get_async_client()


async def embed_texts(texts: list[str], *, client=None) -> list[list[float]]:
    """Embed the texts, in order, in batches. Raises ``EmbeddingError``.

    Raising rather than returning ``None`` per text: a chunk without its vector
    is invisible to retrieval, so a document half-embedded would answer half the
    questions it should and look healthy doing it. The job fails, retries with
    its backoff, and the material stays ``processing`` until it is whole.
    """
    if not texts:
        return []

    model = embedding_model()
    expected = embedding_dimension()
    client = client or _client()

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        try:
            response = await client.embeddings.create(model=model, input=batch)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — provider errors are all the same here
            raise EmbeddingError(f"embedding request failed for {model}: {exc}") from exc

        if len(response.data) != len(batch):
            raise EmbeddingError(
                f"{model} returned {len(response.data)} vectors for {len(batch)} texts"
            )

        # The provider is free to reorder; ``index`` is what says where each
        # vector belongs. Sorting by it costs nothing and removes the question.
        for item in sorted(response.data, key=lambda d: getattr(d, "index", 0)):
            vector = list(item.embedding)
            if len(vector) != expected:
                raise EmbeddingError(
                    f"{model} returned {len(vector)} dimensions, "
                    f"the index holds {expected} (AIX_PRIVATE_EMBEDDING_DIMENSION)"
                )
            vectors.append(vector)

    logger.info("[private-rag] embedded %d chunks with %s", len(vectors), model)
    return vectors


async def embed_query(text: str, *, client=None) -> Optional[list[float]]:
    """One vector for a search. ``None`` when the text is empty."""
    if not text.strip():
        return None
    return (await embed_texts([text], client=client))[0]
