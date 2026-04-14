"""Phase 50.C — observability primitives.

Local structured tracing, modelled after Langfuse's span API but
persisted to our own PostgreSQL `spans` table (migration 0021)
instead of a service. Zero new dependencies, ~no runtime overhead
when the tracer is idle.
"""
from sciknow.observability.tracer import (
    Span,
    current_trace,
    current_span,
    span,
    start_trace,
)

__all__ = ["Span", "current_trace", "current_span", "span", "start_trace"]
