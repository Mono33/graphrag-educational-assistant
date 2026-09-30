"""Langfuse tracing: every model call with its cost, time, tokens, input and output.

One trace per conversation turn and per ingested document, tagged with the
teacher (``user_id``: her account id, never her email) and the lesson
(``session_id``), so Langfuse can sum cost and time per teacher and per lesson.

* Model calls: the OpenAI-compatible clients of ``core.config`` become Langfuse's
  drop-in wrappers, so every chat and embedding call is a generation with its
  tokens and cost -- OpenRouter returns the cost of each call, and the wrapper
  records it as is.
* Graph nodes: a LangChain callback gives each node its own span and timing.

Off unless ``AIX_TRACING_ENABLED`` is on and the Langfuse keys are set.
``AIX_TRACING_CONTENT=none`` keeps costs, tokens and times but sends no text.
Nothing here may fail a run: every call is guarded.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import ExitStack
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)

_OMITTED = "[contenuto non inviato: AIX_TRACING_CONTENT=none]"


def tracing_enabled() -> bool:
    flag = (os.getenv("AIX_TRACING_ENABLED") or "false").strip().lower()
    keys = os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    return flag in ("1", "true", "yes", "on") and bool(keys)


def _content_enabled() -> bool:
    return (os.getenv("AIX_TRACING_CONTENT") or "full").strip().lower() != "none"


def _drop_content(*, data: Any, **_: Any) -> Any:
    return _OMITTED if data is not None else None


def _environment() -> str:
    """Langfuse accepts ``[a-z0-9-_]``, not starting with «langfuse»."""
    raw = (os.getenv("ENVIRONMENT") or "development").strip().lower()
    cleaned = re.sub(r"[^a-z0-9_-]", "-", raw)[:40].lstrip("-") or "development"
    return "env-" + cleaned if cleaned.startswith("langfuse") else cleaned


@lru_cache(maxsize=1)
def get_langfuse() -> Any:
    """The one Langfuse client of this process, prompt fetching included.

    Built before any wrapper or callback asks for it, so its settings -- the
    environment, the release and the content mask -- are the ones that apply.
    """
    from langfuse import Langfuse

    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        base_url=os.getenv("LANGFUSE_BASE_URL")
        or os.getenv("LANGFUSE_HOST")
        or "https://cloud.langfuse.com",
        tracing_enabled=tracing_enabled(),
        environment=_environment(),
        release=os.getenv("GIT_SHA") or None,
        mask=None if _content_enabled() else _drop_content,
    )


def openai_classes() -> tuple[type, type]:
    """``(OpenAI, AsyncOpenAI)``: Langfuse's traced drop-ins when tracing is on."""
    if tracing_enabled():
        try:
            get_langfuse()
            from langfuse.openai import AsyncOpenAI, OpenAI

            return OpenAI, AsyncOpenAI
        except Exception:  # noqa: BLE001 — untraced calls beat no calls
            logger.warning(
                "[tracing] Langfuse unavailable, calling the model untraced", exc_info=True
            )
    from openai import AsyncOpenAI, OpenAI

    return OpenAI, AsyncOpenAI


def graph_callbacks() -> list[Any]:
    """LangChain callbacks that give each graph node a span; empty when off."""
    if not tracing_enabled():
        return []
    try:
        get_langfuse()
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:  # noqa: BLE001
        logger.warning("[tracing] no graph callback", exc_info=True)
        return []


class Run:
    """One trace: opened with :func:`start_run`, closed with :meth:`end`.

    Not a ``with`` block because a turn is an async generator that yields
    between opening and closing; the context it sets is the one the graph's
    tasks inherit, so every model call inside lands in this trace.
    """

    def __init__(self, stack: Optional[ExitStack] = None, span: Any = None) -> None:
        self._stack = stack
        self._span = span

    @property
    def trace_id(self) -> Optional[str]:
        return getattr(self._span, "trace_id", None)

    def end(self, *, output: Any = None, metadata: Optional[dict] = None, error: str = "") -> None:
        if self._stack is None:
            return
        try:
            if self._span is not None:
                self._span.update(
                    output=output,
                    metadata=metadata,
                    level="ERROR" if error else None,
                    status_message=error or None,
                )
            self._stack.close()
        except Exception:  # noqa: BLE001
            logger.warning("[tracing] could not close the trace", exc_info=True)
        finally:
            self._stack = None


def start_run(
    name: str,
    *,
    user_id: Any = None,
    session_id: Any = None,
    tags: tuple[str, ...] = (),
    metadata: Optional[dict[str, Any]] = None,
    input: Any = None,
) -> Run:
    """Open a trace for one turn or one ingestion; a no-op when tracing is off."""
    if not tracing_enabled():
        return Run()
    try:
        from langfuse import propagate_attributes

        stack = ExitStack()
        stack.enter_context(
            propagate_attributes(
                user_id=str(user_id) if user_id else None,
                session_id=str(session_id) if session_id else None,
                tags=[t for t in tags if t],
                metadata={k: str(v)[:200] for k, v in (metadata or {}).items() if v is not None},
                trace_name=name,
            )
        )
        span = stack.enter_context(
            get_langfuse().start_as_current_observation(name=name, as_type="agent", input=input)
        )
        return Run(stack, span)
    except Exception:  # noqa: BLE001
        logger.warning("[tracing] could not open a trace for %s", name, exc_info=True)
        return Run()
