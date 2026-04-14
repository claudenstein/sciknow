"""Phase 50.C — lightweight span tracer.

Context-manager API (`with span("embed", query=q): …`) that records
per-operation timing + metadata to the PostgreSQL `spans` table.
Nested spans track parent/child via a contextvar, so the DAG shape
emerges automatically without callers having to pass trace/span
handles through signatures.

Modelled on Langfuse's span interface — but deliberately local:
Langfuse-as-a-service conflicts with sciknow's no-Docker /
single-user principles, and the 10% of Langfuse that matters
(start/end/parent/metadata) fits in ~250 LOC.

Callers should fail quietly when the DB isn't reachable — a tracer
that aborts a streaming generator because PostgreSQL hiccuped would
be worse than no tracer. All persistence paths below are wrapped in
try/except and log to the module logger on failure.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

logger = logging.getLogger("sciknow.observability.tracer")

# Contextvars let nested `span()` calls discover their parent without
# an explicit handle argument; they're also async-task-safe (each
# asyncio Task gets its own view).
_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sciknow_trace_id", default=None,
)
_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sciknow_span_id", default=None,
)


def current_trace() -> str | None:
    """Return the active trace id (a UUID string), or None outside a trace."""
    return _trace_id.get()


def current_span() -> str | None:
    """Return the active span id (a UUID string), or None if no span is open."""
    return _span_id.get()


class Span:
    """One span in a trace. Created by the `span()` context manager;
    rarely instantiated directly. Callers can mutate `metadata` on a
    running span (e.g. set `metadata['tokens'] = 142` once known) and
    the final persist picks up the updated dict."""

    __slots__ = (
        "id", "trace_id", "parent_id", "name",
        "started_at", "started_monotonic",
        "status", "metadata", "error",
    )

    def __init__(self, *, name: str, trace_id: str, parent_id: str | None) -> None:
        self.id = str(uuid.uuid4())
        self.trace_id = trace_id
        self.parent_id = parent_id
        self.name = name
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.status = "ok"
        self.metadata: dict[str, Any] = {}
        self.error: str | None = None

    def update(self, **kv: Any) -> None:
        """Merge extra metadata into the span (before close). Values
        must be JSON-serialisable; non-serialisable values are str()-ed."""
        for k, v in kv.items():
            try:
                json.dumps(v)
                self.metadata[k] = v
            except TypeError:
                self.metadata[k] = repr(v)[:500]


def start_trace(name: str | None = None, **metadata: Any) -> str:
    """Begin a new trace. Returns the trace id; subsequent `span()`
    calls on the same thread/task inherit it. If a trace is already
    active, returns the existing trace id unchanged — callers that
    want a *new* trace regardless should reset the contextvar."""
    tid = _trace_id.get()
    if tid:
        return tid
    tid = str(uuid.uuid4())
    _trace_id.set(tid)
    # Optional synthetic root span so dashboards can see the trace
    # even if no explicit span() was opened. Very cheap.
    with _persist_root_span(tid, name or "trace", metadata):
        pass
    return tid


@contextmanager
def span(name: str, **metadata: Any) -> Iterator[Span]:
    """Open a span. On exit, writes one row to the `spans` table with
    duration + final metadata. If no trace is active, auto-opens one
    rooted at this span."""
    tid = _trace_id.get()
    owns_trace = False
    if tid is None:
        tid = str(uuid.uuid4())
        _trace_id.set(tid)
        owns_trace = True
    parent = _span_id.get()
    sp = Span(name=name, trace_id=tid, parent_id=parent)
    sp.update(**metadata)
    _span_id.set(sp.id)
    try:
        yield sp
    except Exception as exc:
        sp.status = "error"
        sp.error = f"{type(exc).__name__}: {exc}"[:500]
        _persist_span(sp)
        raise
    else:
        _persist_span(sp)
    finally:
        _span_id.set(parent)
        if owns_trace:
            _trace_id.set(None)


# ── Persistence ─────────────────────────────────────────────────────

def _persist_span(sp: Span) -> None:
    """Best-effort INSERT into `spans`. Never raises."""
    try:
        from sqlalchemy import text as _sql_text
        from sciknow.storage.db import get_session
        ended = datetime.now(timezone.utc)
        duration_ms = int((time.monotonic() - sp.started_monotonic) * 1000)
        with get_session() as session:
            session.execute(_sql_text("""
                INSERT INTO spans
                    (id, trace_id, parent_span_id, name, status,
                     started_at, ended_at, duration_ms, metadata_json, error)
                VALUES
                    (CAST(:id AS uuid), CAST(:tid AS uuid),
                     CAST(:pid AS uuid), :name, :status,
                     :started, :ended, :ms, CAST(:meta AS jsonb), :err)
            """), {
                "id": sp.id, "tid": sp.trace_id, "pid": sp.parent_id,
                "name": sp.name[:200], "status": sp.status,
                "started": sp.started_at, "ended": ended, "ms": duration_ms,
                "meta": json.dumps(sp.metadata, default=str)[:100_000],
                "err": sp.error,
            })
            session.commit()
    except Exception as exc:
        logger.debug("span persist failed for %s: %s", sp.name, exc)


@contextmanager
def _persist_root_span(tid: str, name: str, metadata: dict):
    """Write a zero-duration synthetic span so that `start_trace` alone
    shows up in the tail view even without nested spans below it."""
    try:
        from sqlalchemy import text as _sql_text
        from sciknow.storage.db import get_session
        now = datetime.now(timezone.utc)
        with get_session() as session:
            session.execute(_sql_text("""
                INSERT INTO spans (trace_id, name, status,
                                   started_at, ended_at, duration_ms,
                                   metadata_json)
                VALUES (CAST(:tid AS uuid), :name, 'ok',
                        :ts, :ts, 0, CAST(:meta AS jsonb))
            """), {
                "tid": tid, "name": (name or "trace")[:200], "ts": now,
                "meta": json.dumps(metadata or {}, default=str)[:100_000],
            })
            session.commit()
    except Exception as exc:
        logger.debug("root span persist failed: %s", exc)
    yield
