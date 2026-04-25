"""
sciknow v2 — autowrite engine (re-export shim).

The v2 spec §2 splits the 6.7 kLOC ``core/book_ops.py`` into:

* ``core/book_ops.py`` — CRUD-ish operations (chapters, sections,
  drafts, plans, sizing, classification).
* ``core/autowrite.py`` — the iteration engine (write → score → verify
  → revise → converge), plus the per-run logger, the CoVe verifier,
  and the chapter-wide loop.

The actual code-splitting is mechanical and risky: ~30 L1 contract
tests use ``inspect.getsource(book_ops.autowrite_section_stream)`` and
``book_ops._autowrite_section_body`` to assert the engine's wire
format. Moving the symbols at the same time as splitting the code
would silently flip those tests' source target — better to land the
public module first (this file) so callers can migrate to
``from sciknow.core.autowrite import …`` immediately, then move the
bodies in a follow-up commit when the tests are also moved.

Right now this file *only* re-exports the public surface from
``book_ops``. Callers should import from here going forward; the
``book_ops.X`` names are kept alive by Python's normal name binding
so existing imports keep working.
"""
from __future__ import annotations

# Re-export the public autowrite surface. The `# noqa: F401` keeps
# linters from flagging "imported but unused" — the whole point of
# this module is to expose them under a stable v2 name.
from sciknow.core.book_ops import (  # noqa: F401
    autowrite_section_stream,
    autowrite_chapter_all_sections_stream,
    _autowrite_section_body,
    _AutowriteLogger,
    _create_autowrite_run,
    _persist_autowrite_iteration,
    _persist_autowrite_retrievals,
    _finalize_autowrite_run,
    _score_draft_inner,
    _verify_draft_inner,
    _cove_verify,
    _cove_verify_streaming,
)

__all__ = [
    "autowrite_section_stream",
    "autowrite_chapter_all_sections_stream",
    "_autowrite_section_body",
    "_AutowriteLogger",
    "_create_autowrite_run",
    "_persist_autowrite_iteration",
    "_persist_autowrite_retrievals",
    "_finalize_autowrite_run",
    "_score_draft_inner",
    "_verify_draft_inner",
    "_cove_verify",
    "_cove_verify_streaming",
]
