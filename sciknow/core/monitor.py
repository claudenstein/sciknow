"""Phase 54.6.230 — unified monitor snapshot.

One aggregator shared by ``sciknow db monitor`` (CLI) and
``/api/monitor`` (web reader). Pulls together the pieces that used
to live in scattered places:

  * ``db stats``        — corpus counts per ingestion stage
  * ``db dashboard``    — stage timing + failures (54.6.229)
  * ``/api/stats``      — topic clusters + ingest source breakdown
  * ``ollama ps``       — currently loaded Ollama models + VRAM
  * ``nvidia-smi``      — per-GPU memory + process list
  * ``.last_refresh``   — most recent full refresh timestamp (54.6.210)
  * ``ingestion_jobs``  — last N entries for a live activity feed

Snapshot format (stable key contract — anyone calling
``collect_monitor_snapshot()`` via the CLI --json or the
``/api/monitor`` endpoint relies on this structure):

  {
    "project": {"slug", "data_dir", "pg_database"},
    "corpus": {
      "documents_total", "documents_complete", "chunks", "citations",
      "visuals", "kg_triples", "wiki_pages", "raptor_nodes",
      "status_breakdown": [{"status", "n"}],
    },
    "ingest_sources": [{"source", "n"}],
    "converter_backends": [{"backend", "n"}],
    "topic_clusters": [{"name", "n"}],
    "pipeline": {
      "stage_timing":   [{"stage", "n", "p50_ms", "p95_ms", "mean_ms"}],
      "stage_failures": [{"stage", "failed", "total", "failure_rate"}],
      "throughput":     [{"day", "docs", "jobs"}],
      "recent_activity":[{"created_at", "stage", "status", "doc_id"}],
    },
    "llm": {
      "usage_last_days": [{"operation", "model", "tokens", "seconds",
                            "calls"}],
      "loaded_models":   [{"name", "size_mb", "vram_mb", "expires_at"}],
    },
    "qdrant": [{"name", "points_count", "vectors_config"}],
    "gpu":    [{"index", "name", "memory_used_mb", "memory_total_mb",
                "utilization_pct"}],
    "last_refresh": "<iso timestamp or null>",
    "snapshotted_at": "<iso timestamp>",
  }

Read-only — every data source is a SELECT or an external read.
Safe to run alongside active ingestion.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_db(session, fn, *args, default=None, **kwargs):
    """Phase 54.6.237 — run a DB helper inside a savepoint.

    Each monitor sub-aggregator may hit a table that doesn't exist
    on fresh installs (`raptor_nodes` is the canonical example).
    Postgres aborts the outer transaction on any SQL error, and
    every subsequent SELECT returns "current transaction is
    aborted, commands ignored" until a ROLLBACK. That silent
    poisoning used to empty every aggregator below the first
    failing one.

    Wrapping each call in `session.begin_nested()` creates a
    savepoint; on failure we roll back just to the savepoint,
    leaving the outer transaction intact. Returned value on error
    is `default` (typically the helper's own "empty" return).
    """
    try:
        with session.begin_nested():
            return fn(session, *args, **kwargs)
    except Exception:
        try:
            session.rollback()
        except Exception:
            pass
        return default


def _project_info() -> dict[str, Any]:
    try:
        from sciknow.core.project import get_active_project
        from sciknow.config import settings
        p = get_active_project()
        return {
            "slug": p.slug,
            "data_dir": str(p.data_dir),
            "pg_database": settings.pg_database,
        }
    except Exception as exc:
        logger.debug("project info unavailable: %s", exc)
        return {"slug": None, "data_dir": None, "pg_database": None}


def _corpus_counts(session) -> dict[str, Any]:
    from sqlalchemy import text
    out: dict[str, Any] = {}
    # Simple COUNT(*) queries — all cheap on indexed tables.
    out["documents_total"] = _safe_int(session.execute(
        text("SELECT COUNT(*) FROM documents")
    ).scalar())
    out["documents_complete"] = _safe_int(session.execute(
        text("SELECT COUNT(*) FROM documents "
             "WHERE ingestion_status = 'complete'")
    ).scalar())
    out["chunks"] = _safe_int(session.execute(
        text("SELECT COUNT(*) FROM chunks")
    ).scalar())
    out["citations"] = _safe_int(session.execute(
        text("SELECT COUNT(*) FROM citations")
    ).scalar())
    # Optional tables — graceful degrade if they don't exist yet.
    for sql, key in [
        ("SELECT COUNT(*) FROM visuals", "visuals"),
        ("SELECT COUNT(*) FROM knowledge_graph", "kg_triples"),
        ("SELECT COUNT(*) FROM wiki_pages", "wiki_pages"),
        ("SELECT COUNT(*) FROM paper_institutions", "institutions"),
    ]:
        try:
            out[key] = _safe_int(session.execute(text(sql)).scalar())
        except Exception:
            out[key] = 0

    rows = session.execute(text("""
        SELECT ingestion_status, COUNT(*) FROM documents
        GROUP BY ingestion_status ORDER BY COUNT(*) DESC
    """)).fetchall()
    out["status_breakdown"] = [
        {"status": r[0], "n": _safe_int(r[1])} for r in rows
    ]
    return out


def _ingest_sources(session) -> list[dict]:
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT ingest_source, COUNT(*) FROM documents
        GROUP BY ingest_source ORDER BY COUNT(*) DESC
    """)).fetchall()
    return [
        {"source": r[0] or "unknown", "n": _safe_int(r[1])}
        for r in rows
    ]


def _converter_backends(session) -> list[dict]:
    """Phase 54.6.211 column — distinguishes pipeline-era from VLM-Pro-
    era chunks. Key migration visibility."""
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT COALESCE(converter_backend, '(pre-54.6.211)'),
               COUNT(*) FROM documents
        WHERE ingestion_status = 'complete'
        GROUP BY converter_backend ORDER BY COUNT(*) DESC
    """)).fetchall()
    return [
        {"backend": r[0], "n": _safe_int(r[1])} for r in rows
    ]


def _topic_clusters(session, limit: int = 20) -> list[dict]:
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT topic_cluster, COUNT(*) FROM paper_metadata
        WHERE topic_cluster IS NOT NULL AND topic_cluster != ''
        GROUP BY topic_cluster ORDER BY COUNT(*) DESC LIMIT :lim
    """), {"lim": limit}).fetchall()
    return [{"name": r[0], "n": _safe_int(r[1])} for r in rows]


def _pipeline_timing(session) -> list[dict]:
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT stage,
               COUNT(*) AS n,
               percentile_cont(0.5) WITHIN GROUP
                   (ORDER BY duration_ms) AS p50,
               percentile_cont(0.95) WITHIN GROUP
                   (ORDER BY duration_ms) AS p95,
               AVG(duration_ms) AS mean_ms
        FROM ingestion_jobs
        WHERE status IN ('completed', 'ok')
          AND duration_ms IS NOT NULL
        GROUP BY stage
        ORDER BY p95 DESC NULLS LAST
    """)).fetchall()
    return [
        {
            "stage": r[0],
            "n": _safe_int(r[1]),
            "p50_ms": float(r[2]) if r[2] is not None else None,
            "p95_ms": float(r[3]) if r[3] is not None else None,
            "mean_ms": float(r[4]) if r[4] is not None else None,
        }
        for r in rows
    ]


def _pipeline_failures(session) -> list[dict]:
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT stage,
               SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END),
               COUNT(*)
        FROM ingestion_jobs
        GROUP BY stage
        ORDER BY SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) DESC
    """)).fetchall()
    return [
        {
            "stage": r[0],
            "failed": _safe_int(r[1]),
            "total": _safe_int(r[2]),
            "failure_rate": (
                _safe_int(r[1]) / _safe_int(r[2])
                if _safe_int(r[2]) else 0.0
            ),
        }
        for r in rows
    ]


def _pipeline_throughput(session, days: int = 14) -> list[dict]:
    from sqlalchemy import text
    since_iso = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()
    rows = session.execute(text("""
        SELECT DATE(created_at) AS day,
               COUNT(DISTINCT document_id) AS docs,
               COUNT(*) AS jobs
        FROM ingestion_jobs
        WHERE created_at >= CAST(:since AS timestamptz)
        GROUP BY day
        ORDER BY day DESC
    """), {"since": since_iso}).fetchall()
    return [
        {
            "day": str(r[0]),
            "docs": _safe_int(r[1]),
            "jobs": _safe_int(r[2]),
        }
        for r in rows
    ]


def _corpus_growth_rate(session) -> dict:
    """Phase 54.6.237 — per-week corpus growth sparkline (last 14 weeks)
    + recent-ingest totals (last 24h, 7d, 30d).

    Sparkline counts documents by `created_at` bucketed weekly.
    Recent totals help the header read "42 docs this week" at a
    glance.
    """
    from sqlalchemy import text
    out = {
        "weekly_sparkline": [], "weeks_back": 14,
        "last_24h": 0, "last_7d": 0, "last_30d": 0,
    }
    try:
        # Weekly buckets via date_trunc — zero-filled via generate_series
        rows = session.execute(text("""
            WITH series AS (
                SELECT generate_series(
                    date_trunc('week', NOW()) - INTERVAL '13 weeks',
                    date_trunc('week', NOW()),
                    INTERVAL '1 week'
                ) AS week
            )
            SELECT s.week,
                   COALESCE(COUNT(d.id), 0)
            FROM series s
            LEFT JOIN documents d
              ON date_trunc('week', d.created_at) = s.week
            GROUP BY s.week
            ORDER BY s.week
        """)).fetchall()
        out["weekly_sparkline"] = [int(r[1] or 0) for r in rows]

        row24 = session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """)).fetchone()
        out["last_24h"] = int(row24[0] or 0)

        row7 = session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE created_at >= NOW() - INTERVAL '7 days'
        """)).fetchone()
        out["last_7d"] = int(row7[0] or 0)

        row30 = session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE created_at >= NOW() - INTERVAL '30 days'
        """)).fetchone()
        out["last_30d"] = int(row30[0] or 0)
    except Exception:
        pass
    return out


def _book_activity(session) -> dict:
    """Phase 54.6.237 — most-recently-updated book with chapter
    completion + word count. Silent when no books exist.

    "Completion" = chapters with at least one draft row /
    chapters total. Word count sums the latest-version draft per
    (chapter, section_type) tuple across all chapters.
    """
    from sqlalchemy import text
    out = {
        "title": None, "book_id": None, "book_type": None,
        "chapters_total": 0, "chapters_drafted": 0,
        "total_words": 0, "last_updated": None,
    }
    try:
        row = session.execute(text("""
            SELECT b.id::text, b.title, b.book_type, b.updated_at
            FROM books b
            ORDER BY b.updated_at DESC NULLS LAST
            LIMIT 1
        """)).fetchone()
        if not row:
            return out
        out["book_id"] = row[0]
        out["title"] = row[1]
        out["book_type"] = row[2]
        out["last_updated"] = (
            row[3].isoformat() if row[3] else None
        )

        out["chapters_total"] = int(session.execute(text("""
            SELECT COUNT(*) FROM book_chapters
            WHERE book_id = CAST(:bid AS uuid)
        """), {"bid": out["book_id"]}).scalar() or 0)

        out["chapters_drafted"] = int(session.execute(text("""
            SELECT COUNT(DISTINCT chapter_id)
            FROM drafts
            WHERE chapter_id IN (
                SELECT id FROM book_chapters
                WHERE book_id = CAST(:bid AS uuid)
            )
        """), {"bid": out["book_id"]}).scalar() or 0)

        # Sum latest-version draft words per (chapter, section_type).
        # Using a window-function variant keeps this one query.
        rows = session.execute(text("""
            WITH latest AS (
                SELECT chapter_id, section_type, word_count,
                       ROW_NUMBER() OVER (
                           PARTITION BY chapter_id, section_type
                           ORDER BY version DESC
                       ) AS rn
                FROM drafts
                WHERE chapter_id IN (
                    SELECT id FROM book_chapters
                    WHERE book_id = CAST(:bid AS uuid)
                )
            )
            SELECT COALESCE(SUM(word_count), 0)
            FROM latest WHERE rn = 1
        """), {"bid": out["book_id"]}).fetchone()
        out["total_words"] = int(rows[0] or 0)
    except Exception:
        pass
    return out


def _bench_quality_delta() -> dict:
    """Phase 54.6.237 — diff the two newest bench snapshots.

    Reads bench/snapshots/*.json sorted by mtime, picks the two
    newest, and computes per-metric Δ on every ok result that
    yields a numeric value. Surfaces the aggregate MRR / NDCG /
    recall shifts that the dashboard can show as "retrieval
    quality flat / up / down since last snapshot".

    Silent (empty result) when <2 snapshots exist. Cached for 60s
    in-process to avoid re-reading JSON every tick.
    """
    from sciknow.core.project import get_active_project
    try:
        active = get_active_project()
        snap_dir = active.data_dir / "bench" / "snapshots"
    except Exception:
        return {"have_pair": False}
    return _bench_quality_delta_for_dir(snap_dir)


_BENCH_DELTA_CACHE: dict = {}
_BENCH_DELTA_TTL = 60.0


def _bench_quality_delta_for_dir(snap_dir: Path) -> dict:
    import time as _time
    if not snap_dir.exists():
        return {"have_pair": False}
    key = str(snap_dir)
    now = _time.time()
    cached = _BENCH_DELTA_CACHE.get(key)
    if cached and (now - cached[0]) < _BENCH_DELTA_TTL:
        return cached[1]

    try:
        snaps = sorted(
            snap_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
    except Exception:
        snaps = []
    if len(snaps) < 2:
        result = {"have_pair": False, "count": len(snaps)}
        _BENCH_DELTA_CACHE[key] = (now, result)
        return result

    def _load(p):
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    new_snap = _load(snaps[0])
    old_snap = _load(snaps[1])
    if not new_snap or not old_snap:
        result = {"have_pair": False, "count": len(snaps)}
        _BENCH_DELTA_CACHE[key] = (now, result)
        return result

    def _index(snap: dict) -> dict[str, float]:
        idx = {}
        for r in snap.get("results") or []:
            if r.get("status") != "ok":
                continue
            fn = r.get("name") or ""
            for m in r.get("metrics") or []:
                v = m.get("value")
                if isinstance(v, (int, float)):
                    idx[f"{fn}:{m.get('name')}"] = float(v)
        return idx

    old_idx = _index(old_snap)
    new_idx = _index(new_snap)
    deltas = []
    for k in set(old_idx) & set(new_idx):
        old_v = old_idx[k]
        new_v = new_idx[k]
        if old_v == 0:
            continue
        pct = (new_v - old_v) / abs(old_v) * 100
        deltas.append({
            "metric": k,
            "old": old_v,
            "new": new_v,
            "delta_pct": pct,
        })
    # Surface top movers in each direction
    deltas.sort(key=lambda d: d["delta_pct"])
    worst = deltas[:3]
    deltas.sort(key=lambda d: d["delta_pct"], reverse=True)
    best = deltas[:3]

    result = {
        "have_pair": True,
        "count": len(snaps),
        "new_sha": (new_snap.get("git") or {}).get("sha"),
        "old_sha": (old_snap.get("git") or {}).get("sha"),
        "new_ts": new_snap.get("snapshotted_at"),
        "old_ts": old_snap.get("snapshotted_at"),
        "best": best,
        "worst": worst,
        "total_metrics": len(deltas),
    }
    _BENCH_DELTA_CACHE[key] = (now, result)
    return result


# Phase 54.6.237 — GPU temp/util trend ring buffer. Module-level so
# it persists across snapshot calls within one Python process
# (watch mode populates it over time). Cross-process state is a
# non-goal; each render session builds its own history.
_GPU_TREND_MAX = 60
_GPU_TREND: list[dict] = []


def _record_gpu_sample(gpus: list[dict]) -> None:
    """Append a timestamped sample of the primary GPU's temp +
    util to the in-process ring buffer. Called automatically by
    collect_monitor_snapshot — not intended for external use."""
    if not gpus:
        return
    import time as _time
    g = gpus[0]
    _GPU_TREND.append({
        "t": _time.time(),
        "temp": int(g.get("temperature_c") or 0),
        "util": int(g.get("utilization_pct") or 0),
    })
    if len(_GPU_TREND) > _GPU_TREND_MAX:
        del _GPU_TREND[:-_GPU_TREND_MAX]


def _gpu_trend_snapshot() -> dict:
    return {
        "temp_samples": [s["temp"] for s in _GPU_TREND],
        "util_samples": [s["util"] for s in _GPU_TREND],
        "sample_count": len(_GPU_TREND),
    }


def _model_assignments() -> dict:
    """Phase 54.6.236 — which LLM each stage uses.

    Reads from `Settings` (no DB hit). Surfaces the model-per-task
    mapping that the user configures via .env — useful for "wait,
    which model is autowrite using right now?" moments.

    Phase 54.6.244 — added per-role book writer / reviewer / autowrite
    scorer keys. Pre-244 the monitor showed only ``llm_main`` /
    ``llm_fast``, so the four overrides
    (``BOOK_WRITE_MODEL`` new in 54.6.243, ``BOOK_REVIEW_MODEL``,
    ``AUTOWRITE_SCORER_MODEL``, ``VISUALS_CAPTION_MODEL``) were
    invisible from the dashboard. Each key falls back to ``llm_main``
    in the consumer UI so an unset override renders explicitly
    rather than as ``null``.
    """
    try:
        from sciknow.config import settings
        return {
            "llm_main": settings.llm_model or None,
            "llm_fast": settings.llm_fast_model or None,
            # Per-role book pipeline overrides (None → inherits llm_main)
            "book_write": getattr(settings, "book_write_model", None),
            "book_review": getattr(settings, "book_review_model", None),
            "autowrite_scorer": getattr(
                settings, "autowrite_scorer_model", None,
            ),
            # Visual captioning
            "caption_vlm": (
                getattr(settings, "visuals_caption_model", None)
                or getattr(settings, "caption_vlm_model", None)
                or getattr(settings, "vlm_model", None)
            ),
            "embedder": settings.embedding_model or None,
            "reranker": (
                getattr(settings, "reranker_model", None)
            ),
            "pdf_backend": getattr(settings, "pdf_converter_backend", None),
            "mineru_vlm_backend": getattr(
                settings, "mineru_vlm_backend", None
            ),
            "mineru_vlm_model": getattr(settings, "mineru_vlm_model", None),
        }
    except Exception:
        return {}


def _llm_cost_totals(session, days: int = 30) -> dict:
    """Phase 54.6.236 — trailing LLM spend summary.

    Returns total tokens, total seconds, and distinct model count
    from `llm_usage_log` in the trailing window. Doesn't apply a
    dollar rate (that depends on the deployment; ours runs local
    Ollama, so tokens are "cost in GPU-time" not "cost in $").
    """
    from sqlalchemy import text
    from datetime import datetime, timedelta, timezone
    out = {
        "tokens": 0, "seconds": 0.0, "calls": 0, "models": 0,
        "window_days": days,
    }
    since_iso = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()
    try:
        row = session.execute(text("""
            SELECT COALESCE(SUM(tokens), 0),
                   COALESCE(SUM(duration_seconds), 0),
                   COUNT(*),
                   COUNT(DISTINCT model_name)
            FROM llm_usage_log
            WHERE started_at >= CAST(:since AS timestamptz)
        """), {"since": since_iso}).fetchone()
        if row:
            out["tokens"] = int(row[0] or 0)
            out["seconds"] = float(row[1] or 0.0)
            out["calls"] = int(row[2] or 0)
            out["models"] = int(row[3] or 0)
    except Exception:
        pass
    return out


def _visuals_coverage(session) -> dict:
    """Phase 54.6.236 — per-stage coverage on the visuals pipeline.

    Answers "did caption-visuals actually run on all the figures?
    did paraphrase-equations cover them all? which tables are still
    unparsed?" — the kind of thing the refresh command is supposed
    to handle idempotently, but worth monitoring because a stage
    that silently skipped work would otherwise go unnoticed.
    """
    from sqlalchemy import text
    out: dict = {
        "figures_total": 0, "figures_captioned": 0,
        "charts_total": 0, "charts_captioned": 0,
        "equations_total": 0, "equations_paraphrased": 0,
        "tables_total": 0, "tables_parsed": 0,
        "mentions_linked": 0, "mentions_total_eligible": 0,
    }
    try:
        rows = session.execute(text("""
            SELECT kind,
                   COUNT(*),
                   SUM(CASE WHEN ai_caption IS NOT NULL
                             AND length(ai_caption) >= 20
                            THEN 1 ELSE 0 END)
            FROM visuals
            GROUP BY kind
        """)).fetchall()
        for r in rows:
            kind = r[0]
            total = int(r[1] or 0)
            captioned = int(r[2] or 0)
            if kind == "figure":
                out["figures_total"] = total
                out["figures_captioned"] = captioned
            elif kind == "chart":
                out["charts_total"] = total
                out["charts_captioned"] = captioned
            elif kind == "equation":
                out["equations_total"] = total
                # For equations, "paraphrased" = ai_caption populated.
                out["equations_paraphrased"] = captioned
            elif kind == "table":
                out["tables_total"] = total
    except Exception:
        pass
    # table_summary is the 54.6.106 parse-tables signal; different
    # column than ai_caption.
    try:
        out["tables_parsed"] = int(session.execute(text("""
            SELECT COUNT(*) FROM visuals
            WHERE kind = 'table'
              AND table_summary IS NOT NULL
              AND length(table_summary) >= 20
        """)).scalar() or 0)
    except Exception:
        pass
    # Mention-paragraph linkage — 54.6.138
    try:
        total_eligible = int(session.execute(text("""
            SELECT COUNT(*) FROM visuals
            WHERE figure_num IS NOT NULL
        """)).scalar() or 0)
        linked = int(session.execute(text("""
            SELECT COUNT(*) FROM visuals
            WHERE figure_num IS NOT NULL
              AND mention_paragraphs IS NOT NULL
        """)).scalar() or 0)
        out["mentions_total_eligible"] = total_eligible
        out["mentions_linked"] = linked
    except Exception:
        pass
    return out


def _raptor_tree_shape(session) -> dict:
    """Phase 54.6.236 — shape of the RAPTOR hierarchical tree.

    Queries the topic_clusters/raptor_nodes tables if they exist;
    gracefully returns {} on schema drift."""
    from sqlalchemy import text
    out: dict = {"total_nodes": 0, "levels": [], "has_tree": False}
    try:
        # Levels breakdown — the table name in sciknow is
        # `raptor_nodes` (54.6.70-era). Each node carries a `level`.
        rows = session.execute(text("""
            SELECT level, COUNT(*) FROM raptor_nodes
            GROUP BY level ORDER BY level
        """)).fetchall()
        if rows:
            out["has_tree"] = True
            out["levels"] = [
                {"level": int(r[0]), "n": int(r[1])} for r in rows
            ]
            out["total_nodes"] = sum(l["n"] for l in out["levels"])
    except Exception:
        pass
    return out


def _section_coverage(session) -> dict:
    """Phase 54.6.282 — distribution of chunks across canonical
    section types.

    The chunker (sciknow/ingestion/chunker.py) classifies each
    section into one of the canonical types in ``_SECTION_PATTERNS``
    (abstract / introduction / methods / results / discussion /
    conclusion / related_work / appendix) or falls through to
    "unknown" when the heading doesn't match any pattern.  High
    ``unknown`` rates flag a chunker regression — either the PDF
    converter is dropping heading structure, or new heading styles
    aren't covered by the regex patterns.

    Returns::

        {
          "total": int,
          "unknown_pct": float,   # headline signal for the CLI
          "per_type": [
            {"type": "introduction", "n": 1740, "pct": 5.6},
            ...  # sorted descending by count
          ],
        }

    Silent (empty dict) when the chunks table is empty (fresh
    install).
    """
    from sqlalchemy import text
    try:
        rows = session.execute(text("""
            SELECT section_type, COUNT(*) AS n
            FROM chunks
            GROUP BY section_type
            ORDER BY n DESC
        """)).fetchall()
    except Exception:
        return {}
    total = sum(int(r[1] or 0) for r in rows)
    if total == 0:
        return {"total": 0, "unknown_pct": 0.0, "per_type": []}
    per_type = [
        {
            "type": r[0] or "unknown",
            "n": _safe_int(r[1]),
            "pct": round(int(r[1] or 0) / total * 100, 1),
        }
        for r in rows
    ]
    unknown_pct = next(
        (p["pct"] for p in per_type if p["type"] == "unknown"), 0.0
    )
    return {
        "total": total,
        "unknown_pct": round(unknown_pct, 1),
        "per_type": per_type,
    }


def _citation_graph(session) -> dict:
    """Phase 54.6.280 — citation graph connectivity metrics.

    Surfaces three operational signals the other panels don't cover:

      * **Internal coverage** — how many of the corpus's outgoing
        references land on papers we actually have. Low numbers mean
        `sciknow db expand` has room to run.
      * **Extraction coverage** — how many complete docs have had
        their reference list extracted at all. Papers with zero
        outgoing refs usually indicate converter drop-out (MinerU
        missing the References section) or an old pre-citations
        ingest.
      * **Orphan count** — complete docs with zero *incoming*
        internal citations. High orphan rates in a domain-focused
        corpus suggest we're ingesting papers that aren't referenced
        by the rest of the collection (stray downloads, off-topic
        expand results).

    Plus a top-5 most-cited leaderboard so the operator can
    eyeball the citation "spine" of the corpus at a glance.

    All counts are snapshot-fast (single aggregate per metric;
    `citations.citing_document_id` and `cited_document_id` are both
    indexed). Safe during active ingestion.
    """
    from sqlalchemy import text
    try:
        totals = session.execute(text("""
            SELECT
                COUNT(*) AS total_refs,
                COUNT(cited_document_id) AS internal_refs,
                COUNT(*) FILTER (WHERE is_self_cite = TRUE)
                    AS self_refs,
                COUNT(DISTINCT citing_document_id) AS citing_docs
            FROM citations
        """)).fetchone()
    except Exception:
        totals = None
    try:
        orphans = int(session.execute(text("""
            SELECT COUNT(*) FROM documents d
            WHERE d.ingestion_status = 'complete'
              AND NOT EXISTS (
                  SELECT 1 FROM citations c
                  WHERE c.cited_document_id = d.id
              )
        """)).scalar() or 0)
    except Exception:
        orphans = 0
    try:
        zero_out = int(session.execute(text("""
            SELECT COUNT(*) FROM documents d
            WHERE d.ingestion_status = 'complete'
              AND NOT EXISTS (
                  SELECT 1 FROM citations c
                  WHERE c.citing_document_id = d.id
              )
        """)).scalar() or 0)
    except Exception:
        zero_out = 0
    try:
        rows = session.execute(text("""
            SELECT d.id, pm.title, pm.year, COUNT(c.id) AS n
            FROM documents d
            JOIN citations c ON c.cited_document_id = d.id
            LEFT JOIN paper_metadata pm ON pm.document_id = d.id
            WHERE d.ingestion_status = 'complete'
            GROUP BY d.id, pm.title, pm.year
            ORDER BY n DESC
            LIMIT 5
        """)).fetchall()
    except Exception:
        rows = []
    total_refs = _safe_int(totals[0]) if totals else 0
    internal_refs = _safe_int(totals[1]) if totals else 0
    self_refs = _safe_int(totals[2]) if totals else 0
    citing_docs = _safe_int(totals[3]) if totals else 0
    avg_out = (total_refs / citing_docs) if citing_docs else 0.0
    coverage_pct = (
        (internal_refs / total_refs * 100.0) if total_refs else 0.0
    )
    self_cite_pct = (
        (self_refs / internal_refs * 100.0) if internal_refs else 0.0
    )
    return {
        "total_refs": total_refs,
        "internal_refs": internal_refs,
        "external_refs": max(0, total_refs - internal_refs),
        "self_refs": self_refs,
        "coverage_pct": round(coverage_pct, 1),
        "self_cite_pct": round(self_cite_pct, 1),
        "citing_docs": citing_docs,
        "avg_out_degree": round(avg_out, 1),
        "orphans": orphans,
        "zero_outgoing": zero_out,
        "top_cited": [
            {
                "document_id": str(r[0]),
                "title": (r[1] or "")[:120],
                "year": _safe_int(r[2]) if r[2] is not None else None,
                "n": _safe_int(r[3]),
            }
            for r in rows
        ],
    }


def _duplicate_hashes(session) -> int:
    """Phase 54.6.236 — count of file_hash collisions in documents.

    Should always be 0 on a healthy install (file_hash has a UNIQUE
    constraint, so collisions shouldn't be able to persist), but
    worth tracking as a belt-and-braces drift detector that catches
    schema migration bugs."""
    from sqlalchemy import text
    try:
        return int(session.execute(text("""
            SELECT COALESCE(SUM(c - 1), 0) FROM (
                SELECT COUNT(*) AS c FROM documents
                GROUP BY file_hash HAVING COUNT(*) > 1
            ) t
        """)).scalar() or 0)
    except Exception:
        return 0


def _ingest_funnel(session) -> list[dict]:
    """Phase 54.6.244 — document-count funnel across the canonical
    pipeline stages.

    Unlike `_ingest_queue_states` (which only counts non-terminal
    docs), this returns **all** stages in canonical pipeline order
    so a renderer can draw a funnel and the user sees where docs
    accumulate. Each entry: ``{stage, n}``. Stages that never
    appear in the DB come back with n=0 so the funnel is always
    shaped the same.
    """
    from sqlalchemy import text
    canonical = [
        "pending", "converting", "metadata_extraction",
        "chunking", "embedding", "complete", "failed",
    ]
    try:
        rows = session.execute(text("""
            SELECT ingestion_status, COUNT(*)
            FROM documents
            GROUP BY ingestion_status
        """)).fetchall()
        got = {r[0]: int(r[1] or 0) for r in rows}
    except Exception:
        got = {}
    ordered = [{"stage": s, "n": got.get(s, 0)} for s in canonical]
    # Surface any non-canonical stage values so schema drift is
    # visible rather than silently dropped.
    for stage, n in got.items():
        if stage not in canonical:
            ordered.append({"stage": stage, "n": n})
    return ordered


def _hourly_failures(session, hours: int = 24) -> list[int]:
    """Per-hour failure histogram over the last N hours. Same
    bucketing as `_hourly_throughput` so the two sparklines align
    one-to-one visually."""
    from sqlalchemy import text
    try:
        rows = session.execute(text(f"""
            WITH series AS (
                SELECT generate_series(
                    date_trunc('hour', NOW())
                      - INTERVAL '{int(hours) - 1} hours',
                    date_trunc('hour', NOW()),
                    INTERVAL '1 hour'
                ) AS hour
            )
            SELECT s.hour,
                   COALESCE(SUM(CASE WHEN ij.status = 'failed' THEN 1
                                     ELSE 0 END), 0)
            FROM series s
            LEFT JOIN ingestion_jobs ij
              ON date_trunc('hour', ij.created_at) = s.hour
            GROUP BY s.hour
            ORDER BY s.hour
        """)).fetchall()
        return [int(r[1] or 0) for r in rows]
    except Exception:
        return []


def _disk_free(data_dir: Path | None) -> dict:
    """Phase 54.6.244 — disk total / used / free for the data_dir's
    filesystem. Adds the context the size-only numbers in
    `_disk_usage` don't carry ("4.4G in data_dir" is meaningless
    without knowing total capacity). Silent on non-existent dirs.
    """
    out = {
        "total_mb": 0, "used_mb": 0, "free_mb": 0, "pct_used": 0.0,
    }
    if not data_dir:
        return out
    try:
        import shutil
        total, used, free = shutil.disk_usage(str(data_dir))
        out["total_mb"] = total // (1024 * 1024)
        out["used_mb"] = used // (1024 * 1024)
        out["free_mb"] = free // (1024 * 1024)
        out["pct_used"] = (used / total * 100) if total else 0.0
    except Exception:
        pass
    return out


def _inbox_pending(data_dir: Path | None) -> dict:
    """Phase 54.6.243 — how many PDFs are sitting in data/inbox/
    waiting to be ingested.  Phase 54.6.281 adds an age histogram.

    The inbox convention ("drop PDFs here, run ``sciknow ingest
    directory data/inbox/``") is informal — there is no queue
    table. This walk is recursive (``rglob`` — matches the
    cleanup-downloads scan behaviour from 54.6.273) so PDFs inside
    subfolders are counted too.

    Returns::

        {
          "count": int,
          "oldest_age_s": float | None,
          "age_buckets": {
            "fresh_24h":   int,   # mtime within last day — worth
                                  # ingesting soon; likely a new drop
            "week":        int,   # 1–7 days old
            "month":       int,   # 7–30 days old
            "stale":       int,   # >30 days old — probably forgotten
          },
        }

    Fresh / week drops signal active operator flow; stale counts
    flag "my inbox turned into a dumping ground".
    """
    out: dict = {
        "count": 0, "oldest_age_s": None,
        "age_buckets": {
            "fresh_24h": 0, "week": 0, "month": 0, "stale": 0,
        },
    }
    if not data_dir:
        return out
    inbox = Path(data_dir) / "inbox"
    if not inbox.is_dir():
        return out
    try:
        import time as _time
        # Recursive match: `sciknow ingest directory data/inbox/` walks
        # subfolders, and `sciknow db cleanup-downloads --include-inbox`
        # does too.  Stay consistent so the count doesn't diverge from
        # what the ingest command actually sees.
        pdfs = [
            p for p in inbox.rglob("*.pdf") if p.is_file()
        ]
        out["count"] = len(pdfs)
        if pdfs:
            now = _time.time()
            mtimes = [p.stat().st_mtime for p in pdfs]
            out["oldest_age_s"] = now - min(mtimes)
            buckets = out["age_buckets"]
            day = 24 * 3600
            for mt in mtimes:
                age = now - mt
                if age < day:
                    buckets["fresh_24h"] += 1
                elif age < 7 * day:
                    buckets["week"] += 1
                elif age < 30 * day:
                    buckets["month"] += 1
                else:
                    buckets["stale"] += 1
    except Exception:
        pass
    return out


def _corpus_quality_signals(session) -> dict:
    """Phase 54.6.243 — retrieval-quality signals that don't live in
    an existing helper.

      * abstract coverage — pct of complete docs with a non-empty
        abstract (feeds ColBERT's abstracts collection).
      * chunk length distribution — p50/p95 char count (too-short
        chunks = bad split, too-long = bad context).
      * KG density — triples per completed document (catches
        silent degradation in extract-kg output).
    """
    from sqlalchemy import text
    out = {
        "abstract_covered": 0, "abstract_eligible": 0,
        "abstract_pct": 0.0,
        "chunk_p50_chars": None, "chunk_p95_chars": None,
        "chunk_median_tokens": None,
        "kg_triples_per_doc": 0.0,
    }
    try:
        eligible = int(session.execute(text(
            "SELECT COUNT(*) FROM documents "
            "WHERE ingestion_status = 'complete'"
        )).scalar() or 0)
        covered = int(session.execute(text("""
            SELECT COUNT(*) FROM paper_metadata pm
            JOIN documents d ON d.id = pm.document_id
            WHERE d.ingestion_status = 'complete'
              AND pm.abstract IS NOT NULL
              AND length(pm.abstract) >= 100
        """)).scalar() or 0)
        out["abstract_eligible"] = eligible
        out["abstract_covered"] = covered
        out["abstract_pct"] = (covered / eligible * 100) if eligible else 0.0
    except Exception:
        pass
    try:
        row = session.execute(text("""
            SELECT
                percentile_cont(0.5) WITHIN GROUP
                    (ORDER BY length(content)),
                percentile_cont(0.95) WITHIN GROUP
                    (ORDER BY length(content))
            FROM chunks
        """)).fetchone()
        if row and row[0] is not None:
            out["chunk_p50_chars"] = int(row[0])
            out["chunk_p95_chars"] = int(row[1] or 0)
    except Exception:
        pass
    try:
        triples = int(session.execute(text(
            "SELECT COUNT(*) FROM knowledge_graph"
        )).scalar() or 0)
        docs = int(session.execute(text(
            "SELECT COUNT(*) FROM documents "
            "WHERE ingestion_status = 'complete'"
        )).scalar() or 0)
        out["kg_triples_per_doc"] = (triples / docs) if docs else 0.0
    except Exception:
        pass
    return out


def _wiki_materialization(session) -> dict:
    """Phase 54.6.243 — what fraction of the topic space is backed
    by a compiled wiki page.

    There is no explicit ``topics`` table; the universe is the set of
    distinct non-null ``paper_metadata.topic_cluster`` values, which
    is what `sciknow catalog cluster` populates. Numerator is
    ``wiki_pages``, populated by `sciknow wiki build`. Silent on
    schema drift.
    """
    from sqlalchemy import text
    out = {"topics_total": 0, "wiki_pages": 0, "pct": 0.0}
    try:
        out["topics_total"] = int(session.execute(text(
            "SELECT COUNT(DISTINCT topic_cluster) FROM paper_metadata "
            "WHERE topic_cluster IS NOT NULL AND topic_cluster != ''"
        )).scalar() or 0)
        out["wiki_pages"] = int(session.execute(text(
            "SELECT COUNT(*) FROM wiki_pages"
        )).scalar() or 0)
        if out["topics_total"]:
            out["pct"] = out["wiki_pages"] / out["topics_total"] * 100
    except Exception:
        pass
    return out


def _project_overview() -> list[dict]:
    """Phase 54.6.243 — cross-project inventory.

    Walks ``projects/`` and pulls a small summary per slug (doc
    count via direct SQL on that project's DB, plus active marker).
    Silent if the projects machinery isn't initialised. Cached per
    process for 30s — listing all projects hits N Postgres DBs
    which is a lot for a 5s tick.
    """
    import time as _time
    cache = _project_overview._cache  # type: ignore[attr-defined]
    now = _time.time()
    if cache and (now - cache[0]) < 30.0:
        return cache[1]
    try:
        from sciknow.core.project import list_projects, get_active_project
        active = None
        try:
            active = get_active_project().slug
        except Exception:
            pass
        out: list[dict] = []
        for p in list_projects():
            docs = 0
            try:
                from sqlalchemy import create_engine, text as _t
                from sciknow.config import settings
                url = (
                    f"postgresql+psycopg2://{settings.pg_user}:"
                    f"{settings.pg_password}@{settings.pg_host}:"
                    f"{settings.pg_port}/{p.pg_database}"
                )
                eng = create_engine(url, pool_pre_ping=True)
                with eng.connect() as conn:
                    docs = int(conn.execute(_t(
                        "SELECT COUNT(*) FROM documents"
                    )).scalar() or 0)
                eng.dispose()
            except Exception:
                docs = -1  # sentinel: DB unavailable or schema missing
            out.append({
                "slug": p.slug,
                "pg_database": p.pg_database,
                "docs": docs,
                "is_active": p.slug == active,
            })
        out.sort(key=lambda r: (not r["is_active"], -max(r["docs"], 0)))
        _project_overview._cache = (now, out)  # type: ignore[attr-defined]
        return out
    except Exception:
        return []


_project_overview._cache = None  # type: ignore[attr-defined]


def _build_alerts(snap: dict) -> list[dict]:
    """Phase 54.6.243 — consolidated alert banner feed.

    Takes the already-collected snapshot pieces and returns a list of
    ``{severity, code, message}`` records. Severities: ``error``,
    ``warn``, ``info`` — the CLI/web renderers pick palette based on
    severity. List is sorted error-first so the first entry is the
    worst.

    Kept in the core so both CLI and web render identical banners.
    """
    alerts: list[dict] = []

    stuck = snap.get("stuck_job") or {}
    if stuck.get("is_stuck"):
        age_s = stuck.get("last_age_s", 0) or 0
        alerts.append({
            "severity": "error",
            "code": "stuck_ingest",
            "message": (
                f"ingest STALLED — last job {age_s / 60:.1f}m ago, "
                f"{stuck.get('pending_docs', 0)} pending"
            ),
        })

    embed_cov = snap.get("embeddings_coverage") or {}
    if embed_cov.get("total") and embed_cov.get("pct", 100) < 95:
        alerts.append({
            "severity": "error",
            "code": "embed_drift",
            "message": (
                f"embeddings drift — {embed_cov['missing']:,} chunks "
                f"in PG without vectors ({embed_cov['pct']:.1f}% covered)"
            ),
        })

    dupe = snap.get("duplicate_hashes", 0) or 0
    if dupe > 0:
        alerts.append({
            "severity": "error",
            "code": "dupe_hashes",
            "message": f"{dupe} file_hash collisions detected",
        })

    bench_age = (snap.get("bench_freshness") or {}).get("newest_age_days")
    if bench_age is not None and bench_age > 14:
        alerts.append({
            "severity": "warn",
            "code": "bench_stale",
            "message": f"bench snapshot {bench_age:.0f}d stale",
        })

    # Phase 54.6.263 — snapshot-slow watchdog. When the collector
    # itself crosses 2s the dashboard has stopped being "live" in
    # any useful sense — usually a stuck DB session or slow Qdrant
    # probe. Warn-only; nothing's broken, it's just visibility.
    sd_ms = snap.get("snapshot_duration_ms")
    if isinstance(sd_ms, (int, float)) and sd_ms > 2000:
        alerts.append({
            "severity": "warn",
            "code": "snapshot_slow",
            "message": (
                f"monitor snapshot took {sd_ms:.0f}ms — investigate "
                "slow PG pool or stalled external probe"
            ),
        })

    # Phase 54.6.262 — services reachability. Postgres + Qdrant are
    # critical: ingest/retrieval won't run if either is down.
    # Ollama is warn-level (browse-only installs still work without
    # it). Messages include the latency budget so "slow but up"
    # reads as a distinct state from "unreachable".
    services = snap.get("services") or {}
    CRITICAL_SERVICES = {"postgres", "qdrant"}
    for name, info in services.items():
        if info.get("up"):
            continue
        is_critical = name in CRITICAL_SERVICES
        err_snippet = (info.get("error") or "").splitlines()[0][:80]
        alerts.append({
            "severity": "error" if is_critical else "warn",
            "code": "service_down",
            "message": (
                f"{name} unreachable — {err_snippet or 'no response'}"
            ),
            "action": (
                "systemctl --user status ollama" if name == "ollama" else
                "systemctl status postgresql" if name == "postgres" else
                "systemctl --user status qdrant" if name == "qdrant" else
                None
            ),
        })

    # Phase 54.6.261 — stuck-LLM-job watchdog. A job with a non-
    # zero elapsed time but 0 TPS has either hit a network/thinking
    # stall or the model has wedged. Threshold of 60s dodges the
    # legitimate "qwen3.6 spent 30s inside <think>" false positive.
    # Info-severity: it's suspicious but not always broken — some
    # CoT-heavy prompts spend >60s producing no tokens. Operators
    # can cancel via DELETE /api/jobs/{id} or the web cancel button;
    # action hint points at the web UI path.
    STUCK_THRESHOLD_S = 60
    for j in snap.get("active_jobs") or []:
        elapsed = float(j.get("elapsed_s") or 0)
        tps = float(j.get("tps") or 0)
        state = j.get("stream_state")
        if state != "streaming":
            continue
        if elapsed > STUCK_THRESHOLD_S and tps == 0:
            jid = str(j.get("id") or "?")[:8]
            model = j.get("model") or "?"
            alerts.append({
                "severity": "warn",
                "code": "stuck_llm_job",
                "message": (
                    f"job {jid} ({model}) at 0 tok/s for "
                    f"{elapsed:.0f}s — cancel via the web UI if "
                    "the model is wedged"
                ),
                "action": (
                    f"curl -X DELETE http://localhost:8080/api/jobs/{jid}"
                ),
            })

    # Phase 54.6.252 — config drift. Info-severity (it's ambiguous —
    # user may intend the override, or may have forgotten a stale
    # .env key after switching projects). Message enumerates the
    # first two overrides so the banner stays one line.
    drift = snap.get("config_drift") or []
    if drift:
        head = "; ".join(drift[:2])
        more = f" (+{len(drift) - 2} more)" if len(drift) > 2 else ""
        alerts.append({
            "severity": "info",
            "code": "config_drift",
            "message": (
                f".env overridden by active project: {head}{more} — "
                "drop those keys from .env to silence"
            ),
        })

    # Phase 54.6.250 — backup staleness. Warn at >7d (one week past
    # the default daily cron cadence), error at >30d (something's
    # definitely broken in the backup pipeline — crons stopped,
    # disk full, etc.). Silent when no backup has ever run —
    # that's a "clean install" state, not a regression.
    backup_age = (snap.get("backup_freshness") or {}).get("newest_age_days")
    if backup_age is not None:
        if backup_age > 30:
            alerts.append({
                "severity": "error",
                "code": "backup_stale",
                "message": (
                    f"newest backup {backup_age:.0f}d old — "
                    "run `sciknow backup run` or check the cron"
                ),
            })
        elif backup_age > 7:
            alerts.append({
                "severity": "warn",
                "code": "backup_stale",
                "message": f"newest backup {backup_age:.0f}d old",
            })

    mq = snap.get("meta_quality") or {}
    if mq.get("retracted"):
        # Phase 54.6.276 — demoted from warn → info. Retracted papers
        # are a tracked-but-not-excluded class per user policy; many
        # are in the corpus for legitimate reasons (data still useful,
        # retraction was political, corrigendum addressed the issue).
        # The dashboard just surfaces the count; the operator decides
        # per-paper whether to exclude via wiki-compile gates or book
        # filters, not by the monitor auto-downgrading retrieval.
        alerts.append({
            "severity": "info",
            "code": "retractions",
            "message": (
                f"{mq['retracted']} retracted papers in corpus "
                "(flagged only — not auto-excluded)"
            ),
        })

    gpus = snap.get("gpu") or []
    for g in gpus:
        t = g.get("temperature_c") or 0
        if t >= 85:
            alerts.append({
                "severity": "error",
                "code": "gpu_hot",
                "message": f"gpu #{g.get('index')} {t}°C (thermal limit)",
            })
        elif t >= 80:
            alerts.append({
                "severity": "warn",
                "code": "gpu_warm",
                "message": f"gpu #{g.get('index')} {t}°C",
            })

    host = snap.get("host") or {}
    if host.get("mem_pct", 0) >= 90:
        alerts.append({
            "severity": "warn",
            "code": "ram_pressure",
            "message": f"ram {host['mem_pct']:.0f}% used",
        })
    if host.get("cpu_count"):
        load_ratio = host.get("load_1m", 0) / max(host["cpu_count"], 1)
        if load_ratio >= 1.5:
            alerts.append({
                "severity": "warn",
                "code": "host_overload",
                "message": (
                    f"load {host['load_1m']:.1f} on {host['cpu_count']}c "
                    f"({load_ratio * 100:.0f}%)"
                ),
            })

    storage = snap.get("storage") or {}
    disk = storage.get("disk") or {}
    mineru_mb = disk.get("mineru_output_mb", 0) or 0
    if mineru_mb > 50_000:
        alerts.append({
            "severity": "info",
            "code": "mineru_big",
            "message": (
                f"mineru_output {mineru_mb / 1024:.1f}G — "
                "consider pruning processed outputs"
            ),
        })

    # Phase 54.6.244 — disk-free thresholds. Free space matters more
    # than absolute usage (a 2T drive at 4G used is fine; a 50G drive
    # at 47G used isn't). Error at <5% free, warn at <10%.
    disk_free = snap.get("disk_free") or {}
    total_mb = disk_free.get("total_mb") or 0
    free_mb = disk_free.get("free_mb") or 0
    if total_mb > 0:
        free_pct = free_mb / total_mb * 100
        if free_pct < 5:
            alerts.append({
                "severity": "error",
                "code": "disk_critical",
                "message": (
                    f"disk {free_mb / 1024:.1f}G free "
                    f"({free_pct:.1f}% of {total_mb / 1024:.0f}G)"
                ),
            })
        elif free_pct < 10:
            alerts.append({
                "severity": "warn",
                "code": "disk_low",
                "message": (
                    f"disk {free_mb / 1024:.1f}G free "
                    f"({free_pct:.1f}% of {total_mb / 1024:.0f}G)"
                ),
            })

    inbox = snap.get("inbox") or {}
    if inbox.get("count", 0) > 0:
        age_s = inbox.get("oldest_age_s") or 0
        age_str = (
            f"{age_s / 86400:.1f}d" if age_s >= 86400 else
            f"{age_s / 3600:.0f}h" if age_s >= 3600 else
            f"{age_s / 60:.0f}m"
        )
        alerts.append({
            "severity": "info",
            "code": "inbox_waiting",
            "message": (
                f"inbox {inbox['count']} PDF(s) waiting — oldest {age_str}"
            ),
        })

    # Phase 54.6.245 — missing-model alert. Cross-checks every
    # Ollama-served role in ``model_assignments`` against the list
    # of locally-pulled tags. A role that's configured but not
    # installed (`ollama pull` never ran) fails the first chat call
    # with a 404 — an operational class of bug that doesn't surface
    # anywhere else in the monitor. ``installed == None`` means
    # the list call itself failed (Ollama down) — skip silently
    # rather than emit N alerts when the server is just unreachable.
    installed = snap.get("ollama_installed_models")
    if isinstance(installed, (set, list)):
        installed_set = set(installed)
        models = snap.get("model_assignments") or {}
        # Only Ollama-served roles. `embedder` + `reranker` are HF,
        # `mineru_vlm_model` runs inside the MinerU VLM container
        # (not Ollama), `pdf_backend` is a string enum, not a model.
        ollama_roles: list[tuple[str, str]] = [
            ("LLM_MODEL",              models.get("llm_main")),
            ("LLM_FAST_MODEL",         models.get("llm_fast")),
            ("BOOK_WRITE_MODEL",       models.get("book_write")),
            ("BOOK_REVIEW_MODEL",      models.get("book_review")),
            ("AUTOWRITE_SCORER_MODEL", models.get("autowrite_scorer")),
            ("VISUALS_CAPTION_MODEL",  models.get("caption_vlm")),
        ]
        seen: set[str] = set()
        for role, name in ollama_roles:
            if not name or name in seen:
                continue
            seen.add(name)
            if name not in installed_set:
                alerts.append({
                    "severity": "error",
                    "code": "missing_model",
                    "message": (
                        f"{role}={name} not pulled in Ollama — run "
                        f"`ollama pull {name}` (or change the .env "
                        f"override)"
                    ),
                })

    # Phase 54.6.257 — attach a suggested-fix command to each alert
    # by code. The web modal renders this as a "📋 copy" affordance;
    # the CLI prints it dimly under the message. When a code needs
    # parameters we can't safely supply (e.g. missing_model needs
    # the tag name, which is already in the message), we leave
    # `action` None and rely on the inline hint.
    _ACTIONS: dict[str, str] = {
        "stuck_ingest":    "sciknow ingest directory ./data/inbox --resume",
        "embed_drift":     "sciknow db reingest --stage embedding",
        "dupe_hashes":     "sciknow db stats  # inspect hash dupes",
        "bench_stale":     "sciknow bench-snapshot",
        "backup_stale":    "sciknow backup run",
        "inbox_waiting":   "sciknow ingest directory ./data/inbox",
        "gpu_hot":         "nvidia-smi  # confirm temp and check airflow",
        "gpu_warm":        "nvidia-smi  # monitor temp",
        "ram_pressure":    "ps axu --sort=-%mem | head",
        "host_overload":   "uptime; top -b -n 1 | head -20",
        "mineru_big":      "du -sh data/mineru_output/  # consider pruning processed outputs",
        "disk_critical":   "df -h",
        "disk_low":        "df -h",
        "retractions":     "sciknow db stats  # flagged retractions",
    }
    for a in alerts:
        code = a.get("code")
        # Only fill action when the individual builder didn't set one.
        if code and a.get("action") is None and _ACTIONS.get(code):
            a["action"] = _ACTIONS[code]

    severity_rank = {"error": 0, "warn": 1, "info": 2}
    alerts.sort(key=lambda a: severity_rank.get(a["severity"], 3))
    return alerts


def _bench_freshness(data_dir: Path | None) -> dict:
    """Phase 54.6.236 — age of the newest bench-snapshot.

    Reads the bench/snapshots/ directory (populated by
    `sciknow bench-snapshot`, shipped in 54.6.224) and reports
    age-in-days of the most recent file. Lets the dashboard show
    "last bench 2d ago" and flag stale baselines (>14d) in red —
    a signal that a regression might have crept in since the last
    measurement.
    """
    import time as _time
    out = {"newest_age_days": None, "count": 0}
    if not data_dir:
        return out
    snap_dir = Path(data_dir) / "bench" / "snapshots"
    if not snap_dir.exists():
        return out
    try:
        snaps = list(snap_dir.glob("*.json"))
        if not snaps:
            return out
        out["count"] = len(snaps)
        newest_mtime = max(p.stat().st_mtime for p in snaps)
        age_s = _time.time() - newest_mtime
        out["newest_age_days"] = age_s / 86400
    except Exception:
        pass
    return out


def _ping_postgres() -> dict:
    """Phase 54.6.262 — Postgres reachability probe.

    Opens a fresh connection via the existing session factory, runs
    ``SELECT 1``, and times the round trip. Reports ``{up, latency_ms,
    error}``. Counts as "down" for any connect/query exception.
    """
    import time as _time
    t0 = _time.monotonic()
    try:
        from sqlalchemy import text
        from sciknow.storage.db import get_session
        with get_session() as s:
            s.execute(text("SELECT 1")).scalar_one()
        return {
            "up": True,
            "latency_ms": int((_time.monotonic() - t0) * 1000),
            "error": None,
        }
    except Exception as exc:
        return {
            "up": False,
            "latency_ms": int((_time.monotonic() - t0) * 1000),
            "error": str(exc)[:120],
        }


def _ping_qdrant() -> dict:
    """Phase 54.6.262 — Qdrant reachability probe. ``GET /healthz``
    via the existing client. Returns same shape as _ping_postgres."""
    import time as _time
    t0 = _time.monotonic()
    try:
        from sciknow.storage.qdrant import get_client
        client = get_client()
        # Qdrant client has a lightweight `get_collections` call;
        # prefer it over a raw HTTP ping so any auth/proxy
        # configuration inherited from the client still applies.
        client.get_collections()
        return {
            "up": True,
            "latency_ms": int((_time.monotonic() - t0) * 1000),
            "error": None,
        }
    except Exception as exc:
        return {
            "up": False,
            "latency_ms": int((_time.monotonic() - t0) * 1000),
            "error": str(exc)[:120],
        }


def _ping_ollama() -> dict:
    """Phase 54.6.262 — Ollama reachability probe. ``ollama.list()``
    via the client. Returns same shape as _ping_postgres."""
    import time as _time
    t0 = _time.monotonic()
    try:
        import ollama
        from sciknow.config import settings
        client = ollama.Client(host=settings.ollama_host)
        client.list()
        return {
            "up": True,
            "latency_ms": int((_time.monotonic() - t0) * 1000),
            "error": None,
        }
    except Exception as exc:
        return {
            "up": False,
            "latency_ms": int((_time.monotonic() - t0) * 1000),
            "error": str(exc)[:120],
        }


def _services_health() -> dict:
    """Phase 54.6.262 — PG + Qdrant + Ollama reachability probe.

    Returns a dict keyed by service name. Ollama is the only optional
    service (a read-only install without Ollama still works for
    retrieval + browsing), so its down-state is a warn rather than
    an error in the alert logic.

    Phase 54.6.264 — parallelised via ThreadPoolExecutor. Initial
    measurement showed Qdrant probe alone at ~350 ms (HTTPS TLS
    handshake on remote host), which serialised with the other two
    was most of the snapshot wall clock. Each probe is I/O-bound
    and holds no shared state, so threads are the right primitive.
    Worker count matches probe count (3) so there's no queueing.
    Timeout per probe = 3s — enough for remote Qdrant on a slow
    link but short enough to fail-fast without wedging the whole
    monitor.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _TO

    probes = {
        "postgres": _ping_postgres,
        "qdrant":   _ping_qdrant,
        "ollama":   _ping_ollama,
    }
    out: dict = {}
    with ThreadPoolExecutor(max_workers=len(probes)) as pool:
        futures = {name: pool.submit(fn) for name, fn in probes.items()}
        for name, fut in futures.items():
            try:
                out[name] = fut.result(timeout=3.0)
            except _TO:
                out[name] = {
                    "up": False,
                    "latency_ms": 3000,
                    "error": "probe timed out after 3s",
                }
            except Exception as exc:
                out[name] = {
                    "up": False,
                    "latency_ms": 0,
                    "error": f"probe raised: {type(exc).__name__}"[:120],
                }
    return out


def _log_tail(data_dir: Path | None, n: int = 20) -> dict:
    """Phase 54.6.260 — last ``n`` lines of ``<data_dir>/sciknow.log``.

    Reverse-read via ``seek`` so we don't slurp an 8 MB log when the
    operator just wants the last 20 lines. Returns::

        {
            "lines": list[str],
            "file_path": str | None,  # absolute path for UI hint
            "error_lines": int,       # count of ERROR/WARNING in tail
        }

    ``lines`` is empty when the log file doesn't exist (fresh
    install) — consumers render nothing.
    """
    out: dict = {"lines": [], "file_path": None, "error_lines": 0}
    if not data_dir:
        return out
    log_path = Path(data_dir) / "sciknow.log"
    if not log_path.exists():
        return out
    out["file_path"] = str(log_path)
    try:
        # 4KB per line × n + 50% overhead as the seek budget
        block = max(8192, n * 400)
        size = log_path.stat().st_size
        with log_path.open("rb") as f:
            start = max(0, size - block)
            f.seek(start)
            chunk = f.read()
        # Decode with latin-1 fallback so malformed bytes don't crash
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            text = chunk.decode("utf-8", errors="replace")
        lines = text.splitlines()
        # If we didn't read from the start, drop the first (likely
        # truncated) line so we return complete records.
        if start > 0 and lines:
            lines = lines[1:]
        tail = lines[-n:]
        out["lines"] = tail
        out["error_lines"] = sum(
            1 for ln in tail if "ERROR" in ln or "CRITICAL" in ln
        )
    except Exception as exc:
        logger.debug("log tail read failed: %s", exc)
    return out


def alerts_as_markdown(snap: dict) -> str:
    """Phase 54.6.268 — render the snapshot's alerts as a Markdown
    block suitable for pasting into Slack / Linear / a GitHub
    issue. Same helper is used by the CLI ``--alerts-md`` flag
    and the web "Copy as Markdown" button so the shape stays
    identical across both surfaces.

    Format:

        # sciknow alerts — <slug> @ <ISO-UTC>

        Health: NN/100 · Errors: X · Warn: Y · Info: Z

        - ❌ **code**: message · `sciknow …`
        - ⚠️ **code**: message
        - ℹ️ **code**: message

    Empty alerts list → "_No alerts at this snapshot._"
    """
    from datetime import datetime, timezone
    project = (snap.get("project") or {}).get("slug") or "(no project)"
    alerts = snap.get("alerts") or []
    health = (snap.get("health_score") or {}).get("score")
    ts = snap.get("snapshotted_at") or datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append(f"# sciknow alerts — {project} @ {ts}")
    lines.append("")
    counts = {
        "error": sum(1 for a in alerts if a.get("severity") == "error"),
        "warn":  sum(1 for a in alerts if a.get("severity") == "warn"),
        "info":  sum(1 for a in alerts if a.get("severity") == "info"),
    }
    header = []
    if health is not None:
        header.append(f"Health: {health}/100")
    header.append(f"Errors: {counts['error']}")
    header.append(f"Warn: {counts['warn']}")
    header.append(f"Info: {counts['info']}")
    lines.append(" · ".join(header))
    lines.append("")
    if not alerts:
        lines.append("_No alerts at this snapshot._")
        return "\n".join(lines)
    icon = {"error": "❌", "warn": "⚠️", "info": "ℹ️"}
    for a in alerts:
        sev = a.get("severity", "info")
        bullet = f"- {icon.get(sev, '•')} **{a.get('code', '?')}**: {a.get('message', '')}"
        act = a.get("action")
        if act:
            # Inline code span rather than fence so the line stays
            # compact in Slack / Linear preview.
            bullet += f" · `{act}`"
        lines.append(bullet)
    return "\n".join(lines)


def _compute_health_score(snap: dict) -> dict:
    """Phase 54.6.259 — composite 0-100 pipeline-health rollup.

    One number that answers "am I in good shape?" without requiring
    the operator to read every panel. Starts at 100; penalties:

      * -20 per error alert
      * -5 per warn alert
      * -(100 - embed_pct) / 2 when embeddings_coverage < 100 %
      * -10 if free disk < 10 %
      * -10 if any GPU ≥ 85 °C
      * -5 if RAM ≥ 90 %

    Clamps to [0, 100]. Returns ``{score, verdict, penalties}`` so
    both renderers can show a breakdown tooltip.

    Must run AFTER alerts have been built (it reads them). Pure
    function against the snapshot — no side effects.
    """
    score = 100
    penalties: list[str] = []

    alerts = snap.get("alerts") or []
    errs = sum(1 for a in alerts if a.get("severity") == "error")
    warns = sum(1 for a in alerts if a.get("severity") == "warn")
    if errs:
        score -= 20 * errs
        penalties.append(f"−{20 * errs} ({errs} error alert(s))")
    if warns:
        score -= 5 * warns
        penalties.append(f"−{5 * warns} ({warns} warn alert(s))")

    embed_cov = snap.get("embeddings_coverage") or {}
    if embed_cov.get("total") and embed_cov.get("pct", 100) < 100:
        drop = (100 - embed_cov["pct"]) / 2
        if drop > 0.5:
            score -= drop
            penalties.append(
                f"−{drop:.0f} (embeddings coverage {embed_cov['pct']:.1f}%)"
            )

    disk_free = snap.get("disk_free") or {}
    if disk_free.get("total_mb") and disk_free.get("free_mb") is not None:
        free_pct = disk_free["free_mb"] / disk_free["total_mb"] * 100
        if free_pct < 10:
            score -= 10
            penalties.append(f"−10 (disk {free_pct:.0f}% free)")

    for g in snap.get("gpu") or []:
        if (g.get("temperature_c") or 0) >= 85:
            score -= 10
            penalties.append(
                f"−10 (gpu #{g.get('index', '?')} {g['temperature_c']}°C)"
            )
            break  # cap the GPU penalty at 10 even with many cards

    host = snap.get("host") or {}
    if host.get("mem_pct", 0) >= 90:
        score -= 5
        penalties.append(f"−5 (ram {host['mem_pct']:.0f}%)")

    score = max(0, min(100, round(score)))
    if score >= 90:
        verdict = "healthy"
    elif score >= 60:
        verdict = "degraded"
    else:
        verdict = "critical"
    return {
        "score": int(score),
        "verdict": verdict,
        "penalties": penalties,
    }


def _config_drift() -> list[str]:
    """Phase 54.6.252 — config-drift list.

    Reads the list ``settings._env_overrides`` stashed by
    ``Settings._project_wins_over_env_overrides``. Each entry is a
    human-readable string like ``"pg_database 'sciknow' →
    'sciknow_global-cooling'"`` — one per .env key that the active
    project silently overrode.

    Empty list is the "clean" state. A non-empty list is a footgun
    signal: the user's .env doesn't match the actual runtime config,
    and anything they read from .env by eye will be misleading.
    """
    try:
        from sciknow.config import settings
        overrides = getattr(settings, "_env_overrides", None) or []
        # Always return a list (never None) for stable JSON shape.
        return [str(o) for o in overrides]
    except Exception:
        return []


def _backup_freshness() -> dict:
    """Phase 54.6.250 — age + count of the newest ``sciknow backup``.

    Reads ``archives/backups/.backup-state.json`` (the
    ``backup_retain_count``-capped ledger maintained by
    ``sciknow backup run``). Returns::

        {
            "newest_age_days": float | None,
            "count": int,
            "newest_timestamp": str | None,
            "total_bytes": int | None,
        }

    ``newest_age_days`` is ``None`` when no backup has run yet (or
    the ledger is unreadable) — consumers treat that as "no
    backup-stale alert" rather than "infinitely stale".

    Pulls from the repo-relative ``archives/backups/`` path (same
    as ``_backup_root()`` in sciknow/cli/backup.py), not the
    project data_dir, because backup sets span multiple projects
    and live at the repo level.
    """
    import time as _time
    import json as _json

    out: dict = {
        "newest_age_days": None,
        "count": 0,
        "newest_timestamp": None,
        "total_bytes": None,
    }
    try:
        from sciknow.core.project import _repo_root
        state_path = _repo_root() / "archives" / "backups" / ".backup-state.json"
        if not state_path.exists():
            return out
        state = _json.loads(state_path.read_text(encoding="utf-8"))
        backups = state.get("backups") or []
        if not backups:
            return out
        out["count"] = len(backups)
        # Newest = last appended (the runner appends chronologically)
        newest = backups[-1]
        out["newest_timestamp"] = newest.get("timestamp")
        out["total_bytes"] = newest.get("total_bytes")
        # Parse the timestamp into an age; format is
        # ``YYYYMMDDTHHMMSSZ``. Use mtime of the backup dir as a
        # sturdy fallback when the timestamp is malformed.
        ts = out["newest_timestamp"] or ""
        age_s: float | None = None
        if ts:
            try:
                from datetime import datetime
                dt = datetime.strptime(ts, "%Y%m%dT%H%M%SZ")
                age_s = _time.time() - dt.timestamp()
            except Exception:
                age_s = None
        if age_s is None:
            backup_dir = state_path.parent / (newest.get("dir") or ts)
            if backup_dir.exists():
                age_s = _time.time() - backup_dir.stat().st_mtime
        if age_s is not None:
            out["newest_age_days"] = max(0.0, age_s / 86400)
    except Exception:
        pass
    return out


def _year_histogram(session, since_year: int = 1980) -> list[dict]:
    """Phase 54.6.235 — papers-per-year from `since_year` through now.
    Zero-filled year-by-year so the sparkline shows continuity."""
    from sqlalchemy import text
    try:
        rows = session.execute(text("""
            SELECT year, COUNT(*) FROM paper_metadata
            WHERE year IS NOT NULL AND year >= :since
            GROUP BY year ORDER BY year
        """), {"since": since_year}).fetchall()
        raw = {int(r[0]): int(r[1]) for r in rows}
        # Find the actual year range in the data so we don't
        # always pad to 2026 on a corpus that stops at 2010.
        if not raw:
            return []
        max_year = max(raw.keys())
        out: list[dict] = []
        for y in range(since_year, max_year + 1):
            out.append({"year": y, "n": raw.get(y, 0)})
        return out
    except Exception:
        return []


def _embeddings_coverage(session) -> dict:
    """Phase 54.6.235 — catch the drift where chunks exist in
    PostgreSQL but never got an embedding point in Qdrant.
    Should always be ~0 on a healthy install; any positive number
    is a strong signal something silently broke."""
    from sqlalchemy import text
    out = {"total": 0, "embedded": 0, "missing": 0, "pct": 0.0}
    try:
        total = int(session.execute(text(
            "SELECT COUNT(*) FROM chunks"
        )).scalar() or 0)
        embedded = int(session.execute(text(
            "SELECT COUNT(*) FROM chunks "
            "WHERE qdrant_point_id IS NOT NULL"
        )).scalar() or 0)
        out["total"] = total
        out["embedded"] = embedded
        out["missing"] = total - embedded
        out["pct"] = (embedded / total * 100) if total else 0.0
    except Exception:
        pass
    return out


def _qdrant_disk_estimate(collection_info: dict,
                           default_dim: int = 1024) -> int:
    """Cheap disk-footprint estimate for a Qdrant collection in MB.

    Rough formula: points × named_vector_count × dim × 4 bytes
    (float32). Multi-vector (colbert) collections multiply by
    ~150 tokens per point. Excludes payload + HNSW graph overhead
    (add ~30% in real life); shown as "~X MB" to cue the reader.
    """
    points = collection_info.get("points_count", 0) or 0
    vectors = collection_info.get("vectors") or []
    bytes_total = 0
    for v in vectors:
        # ColBERT multi-vector: each point stores ~150 token vecs.
        per_point = default_dim * 4
        if v == "colbert":
            per_point *= 150
        bytes_total += points * per_point
    # Sparse vectors are small (~100 non-zeros × 8 bytes) — include
    # as a token contribution rather than ignoring.
    for _ in collection_info.get("sparse_vectors") or []:
        bytes_total += points * 100 * 8
    return bytes_total // (1024 * 1024)


def _read_refresh_pulse(data_dir: Path | None) -> dict | None:
    """Phase 54.6.238 — read the `sciknow refresh` pulse file.
    Returns None when no refresh is in flight or the pulse is
    ancient (`core.pulse` handles the staleness cutoff)."""
    try:
        from sciknow.core.pulse import read_pulse
        return read_pulse(data_dir, "refresh")
    except Exception:
        return None


def _read_web_jobs_pulse(data_dir: Path | None) -> list[dict]:
    """Phase 54.6.246 — read the active-web-jobs pulse.

    The web server writes this file from
    ``_write_web_jobs_pulse()`` on every job state transition. The
    CLI ``sciknow db monitor`` runs in a *different* process with no
    access to the web's in-memory ``_jobs`` dict, so without this
    pulse it can never show running autowrite / book-write / wiki
    compile jobs — a real gap for anyone babysitting a long run
    over SSH.

    Returns an empty list when:
      * no pulse file exists (no web server has started since the
        project was created)
      * the pulse is stale by ``core.pulse`` standards (web server
        crashed / was stopped; do not render imaginary jobs)
      * the pulse payload is malformed

    Staleness tagging: each job entry gets its own ``is_stale``
    flag from the *pulse-level* ``is_stale`` so the CLI renderer can
    render a "[STALE]" suffix.
    """
    try:
        from sciknow.core.pulse import read_pulse
        body = read_pulse(data_dir, "web_jobs")
        if not body:
            return []
        active = body.get("active") or []
        # Decorate with pulse-level staleness so consumers can tell
        # "this web job is probably zombie" without re-checking
        # pulse_at themselves.
        is_stale = bool(body.get("is_stale", False))
        for j in active:
            j["is_stale"] = is_stale
        return active
    except Exception:
        return []


def _host_load() -> dict:
    """Phase 54.6.234 — system RAM + load average from /proc.

    Portable Linux-only read; on non-Linux or read failure the
    returned dict has zeros so rendering degrades gracefully. Zero
    new dependencies — /proc/meminfo and /proc/loadavg are always
    present on Linux systems running sciknow.
    """
    import os
    out = {
        "mem_used_mb": 0, "mem_total_mb": 0, "mem_pct": 0.0,
        "load_1m": 0.0, "load_5m": 0.0, "load_15m": 0.0,
        "cpu_count": 0,
    }
    try:
        with open("/proc/meminfo") as f:
            info: dict[str, int] = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]
                    info[k] = int(v)  # kB
        total_kb = info.get("MemTotal", 0)
        # MemAvailable is the kernel's "how much can an app grab right
        # now" estimate — more useful than MemFree which misses
        # reclaimable cache.
        avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
        used_kb = max(total_kb - avail_kb, 0)
        out["mem_total_mb"] = total_kb // 1024
        out["mem_used_mb"] = used_kb // 1024
        out["mem_pct"] = (used_kb / total_kb * 100) if total_kb else 0.0
    except Exception:
        pass
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            if len(parts) >= 3:
                out["load_1m"] = float(parts[0])
                out["load_5m"] = float(parts[1])
                out["load_15m"] = float(parts[2])
    except Exception:
        pass
    try:
        out["cpu_count"] = os.cpu_count() or 0
    except Exception:
        pass
    return out


def _stuck_job(session) -> dict:
    """Phase 54.6.234 — detect a stalled ingest.

    Compares the newest ingestion_jobs row against "now" and the
    queue depth. Returns a dict:
      {"is_stuck": bool, "last_age_s": float | None,
       "pending_docs": int, "threshold_s": int}

    Stuck = queue has pending docs AND newest job is older than
    the threshold (5 min — convert p95 is the usual worst case,
    so 5 min with no new jobs strongly suggests a stall).
    """
    from sqlalchemy import text
    out = {
        "is_stuck": False, "last_age_s": None,
        "pending_docs": 0, "threshold_s": 300,
    }
    try:
        row = session.execute(text("""
            SELECT EXTRACT(EPOCH FROM NOW() - MAX(created_at))
            FROM ingestion_jobs
        """)).fetchone()
        if row and row[0] is not None:
            out["last_age_s"] = float(row[0])
        out["pending_docs"] = int(session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE ingestion_status NOT IN ('complete', 'failed')
        """)).scalar() or 0)
        if (out["last_age_s"] is not None
                and out["last_age_s"] > out["threshold_s"]
                and out["pending_docs"] > 0):
            out["is_stuck"] = True
    except Exception:
        pass
    return out


def _metadata_quality(session) -> dict:
    """Phase 54.6.234 — content-quality breakdown for the corpus
    panel. Counts by metadata source, paper type, language, and
    retraction flag — all from existing columns (no new queries
    per iteration, just aggregations).
    """
    from sqlalchemy import text
    out: dict = {
        "sources": [], "paper_types": [], "languages": [],
        "retracted": 0, "citations_crosslinked_pct": 0.0,
        "citations_total": 0, "citations_crosslinked": 0,
    }
    try:
        rows = session.execute(text("""
            SELECT metadata_source, COUNT(*) FROM paper_metadata
            GROUP BY metadata_source ORDER BY COUNT(*) DESC
        """)).fetchall()
        out["sources"] = [
            {"source": r[0] or "unknown", "n": int(r[1])}
            for r in rows
        ]
    except Exception:
        pass
    try:
        rows = session.execute(text("""
            SELECT paper_type, COUNT(*) FROM paper_metadata
            WHERE paper_type IS NOT NULL
            GROUP BY paper_type ORDER BY COUNT(*) DESC
        """)).fetchall()
        out["paper_types"] = [
            {"type": r[0], "n": int(r[1])} for r in rows
        ]
    except Exception:
        pass
    try:
        rows = session.execute(text("""
            SELECT language, COUNT(*) FROM documents
            WHERE ingestion_status = 'complete'
            GROUP BY language ORDER BY COUNT(*) DESC
        """)).fetchall()
        out["languages"] = [
            {"lang": r[0], "n": int(r[1])} for r in rows
        ]
    except Exception:
        pass
    try:
        out["retracted"] = int(session.execute(text("""
            SELECT COUNT(*) FROM paper_metadata
            WHERE retraction_status IN ('retracted', 'withdrawn')
        """)).scalar() or 0)
    except Exception:
        pass
    try:
        total = int(session.execute(text(
            "SELECT COUNT(*) FROM citations"
        )).scalar() or 0)
        xlinked = int(session.execute(text(
            "SELECT COUNT(*) FROM citations "
            "WHERE cited_document_id IS NOT NULL"
        )).scalar() or 0)
        out["citations_total"] = total
        out["citations_crosslinked"] = xlinked
        out["citations_crosslinked_pct"] = (
            (xlinked / total * 100) if total else 0.0
        )
    except Exception:
        pass
    return out


def _ingest_rates_and_eta(session) -> dict:
    """Phase 54.6.232 — trailing throughput + ETA to complete ingest.

    Measures docs completed (via the terminal `embedding` stage —
    the last step of the pipeline so one row per done paper) over
    two windows:

      * 1h  — most recent rate, sensitive to stalls
      * 4h  — smoother average for ETA math

    ETA prefers 4h rate when there's ≥3 samples; falls back to 1h
    to stay useful during the first hour of a fresh run.
    """
    from sqlalchemy import text
    out = {"rate_1h": 0.0, "rate_4h": 0.0,
           "pending_docs": 0, "eta_hours": None}
    try:
        row = session.execute(text("""
            SELECT COUNT(DISTINCT document_id)
            FROM ingestion_jobs
            WHERE status IN ('completed', 'ok')
              AND stage = 'embedding'
              AND created_at >= NOW() - INTERVAL '1 hour'
        """)).fetchone()
        out["rate_1h"] = float(row[0] or 0)

        row = session.execute(text("""
            SELECT COUNT(DISTINCT document_id)
            FROM ingestion_jobs
            WHERE status IN ('completed', 'ok')
              AND stage = 'embedding'
              AND created_at >= NOW() - INTERVAL '4 hours'
        """)).fetchone()
        out["rate_4h"] = (float(row[0] or 0)) / 4.0

        row = session.execute(text("""
            SELECT COUNT(*) FROM documents
            WHERE ingestion_status NOT IN ('complete', 'failed')
        """)).fetchone()
        out["pending_docs"] = int(row[0] or 0)

        # Prefer the smoother 4h rate when we have enough data,
        # else fall back to 1h. Both are in docs/hour.
        rate = out["rate_4h"] if out["rate_4h"] >= 3 else out["rate_1h"]
        if rate > 0 and out["pending_docs"] > 0:
            out["eta_hours"] = out["pending_docs"] / rate
    except Exception:
        pass
    return out


def _ingest_queue_states(session) -> dict:
    """Count of documents by non-complete ingestion_status."""
    from sqlalchemy import text
    try:
        rows = session.execute(text("""
            SELECT ingestion_status, COUNT(*)
            FROM documents
            WHERE ingestion_status NOT IN ('complete', 'failed')
            GROUP BY ingestion_status
        """)).fetchall()
        return {r[0]: int(r[1] or 0) for r in rows}
    except Exception:
        return {}


def _pending_downloads_count(session) -> int:
    from sqlalchemy import text
    try:
        return int(session.execute(text(
            "SELECT COUNT(*) FROM pending_downloads"
        )).scalar() or 0)
    except Exception:
        return 0


def _hourly_throughput(session, hours: int = 24) -> list[int]:
    """Per-hour docs-completed histogram over the last N hours.
    Zero-filled so the sparkline shows silence, not a gap."""
    from sqlalchemy import text
    try:
        rows = session.execute(text(f"""
            WITH series AS (
                SELECT generate_series(
                    date_trunc('hour', NOW())
                      - INTERVAL '{int(hours) - 1} hours',
                    date_trunc('hour', NOW()),
                    INTERVAL '1 hour'
                ) AS hour
            )
            SELECT s.hour,
                   COALESCE(COUNT(DISTINCT ij.document_id), 0)
            FROM series s
            LEFT JOIN ingestion_jobs ij
              ON date_trunc('hour', ij.created_at) = s.hour
             AND ij.stage = 'embedding'
             AND ij.status IN ('completed', 'ok')
            GROUP BY s.hour
            ORDER BY s.hour
        """)).fetchall()
        return [int(r[1] or 0) for r in rows]
    except Exception:
        return []


def _pg_database_size_mb(session) -> int:
    """Current DB size via pg_database_size(current_database())."""
    from sqlalchemy import text
    try:
        size = session.execute(text(
            "SELECT pg_database_size(current_database())"
        )).scalar() or 0
        return int(size) // (1024 * 1024)
    except Exception:
        return 0


def _top_failure_classes(
    session, since_hours: int = 24, limit: int = 3,
) -> list[dict]:
    """Top N (stage × error-prefix) failure classes in the trailing
    window. Small limit on purpose — this is a summary, full detail
    lives in `sciknow db failures`."""
    from sqlalchemy import text
    try:
        rows = session.execute(text(f"""
            SELECT stage,
                   COALESCE(
                       substring(details->>'error' from 1 for 60),
                       '(no error text)'
                   ) AS err_sig,
                   COUNT(*) AS n
            FROM ingestion_jobs
            WHERE status = 'failed'
              AND created_at >= NOW() - INTERVAL '{int(since_hours)} hours'
            GROUP BY stage, err_sig
            ORDER BY n DESC
            LIMIT :lim
        """), {"lim": limit}).fetchall()
        return [
            {"stage": r[0], "error": r[1], "count": int(r[2])}
            for r in rows
        ]
    except Exception:
        return []


# Per-process TTL cache — filesystem walks on mineru_output with
# thousands of per-paper subdirs are too slow to repeat every tick.
_DISK_CACHE: dict[str, tuple[float, int]] = {}
_DISK_TTL_SECONDS = 60.0


def _du_cached(path: Path) -> int:
    """Best-effort directory size in bytes. Skips symlinks and
    permission errors. Cached for 60s per path — a watching monitor
    at 5s refresh pays one walk per minute, not one per tick."""
    import time as _time
    key = str(path)
    now = _time.time()
    cached = _DISK_CACHE.get(key)
    if cached and (now - cached[0]) < _DISK_TTL_SECONDS:
        return cached[1]
    total = 0
    try:
        for dirpath, dirnames, filenames in _walk_no_symlinks(path):
            for f in filenames:
                try:
                    total += (Path(dirpath) / f).stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    _DISK_CACHE[key] = (now, total)
    return total


def _walk_no_symlinks(top: Path):
    """os.walk that never follows symlinks — guards against infinite
    loops from pathological symlink configurations."""
    import os
    if not top.exists():
        return
    for dirpath, dirnames, filenames in os.walk(
        str(top), followlinks=False,
    ):
        yield dirpath, dirnames, filenames


def _disk_usage(data_dir: Path | None) -> dict:
    """Key directory sizes under the project data_dir, in MB. Cheap
    thanks to _du_cached."""
    out = {
        "data_dir_mb": 0, "mineru_output_mb": 0, "processed_mb": 0,
        "downloads_mb": 0, "bench_mb": 0, "failed_mb": 0,
    }
    if not data_dir:
        return out
    data_path = Path(data_dir)
    out["data_dir_mb"] = _du_cached(data_path) // (1024 * 1024)
    for sub, key in [
        ("mineru_output", "mineru_output_mb"),
        ("processed", "processed_mb"),
        ("downloads", "downloads_mb"),
        ("bench", "bench_mb"),
        ("failed", "failed_mb"),
    ]:
        out[key] = _du_cached(data_path / sub) // (1024 * 1024)
    return out


def _recent_activity(session, limit: int = 15) -> list[dict]:
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT created_at, stage, status, document_id::text,
               duration_ms
        FROM ingestion_jobs
        ORDER BY created_at DESC
        LIMIT :lim
    """), {"lim": limit}).fetchall()
    return [
        {
            "created_at": r[0].isoformat() if r[0] else None,
            "stage": r[1],
            "status": r[2],
            "doc_id": r[3],
            "duration_ms": _safe_int(r[4]) if r[4] is not None else None,
        }
        for r in rows
    ]


def _llm_usage_by_day(session, days: int = 7) -> dict:
    """Phase 54.6.283 — per-day × per-operation LLM call grid for a
    heatmap rendering.

    Complements ``_llm_usage`` (which aggregates over the whole
    window).  Answers "did autowrite run today?  when did wiki
    compile stall?" at a glance.  Returns::

        {
          "days": ["2026-04-17", ..., "2026-04-23"],  # ascending
          "operations": ["autowrite_writer", ...],    # top N by total
          "grid":  {"autowrite_writer": {"2026-04-17": 12, ...}},
          "max_calls": int,   # for colour scaling in the renderer
        }

    Days with zero activity still appear in the ``days`` list so
    the heatmap has a consistent column count (a gap in activity
    is itself information).  Operations capped at top 12 to keep
    the grid legible — the long tail rolls up into ``(other)``.
    """
    from sqlalchemy import text
    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_iso = since.isoformat()
    try:
        rows = session.execute(text("""
            SELECT
                to_char(date_trunc('day', started_at), 'YYYY-MM-DD')
                    AS day,
                operation,
                COUNT(*) AS n
            FROM llm_usage_log
            WHERE started_at >= CAST(:since AS timestamptz)
            GROUP BY day, operation
            ORDER BY day ASC
        """), {"since": since_iso}).fetchall()
    except Exception:
        return {}
    # Build the continuous day axis (ascending, inclusive of today).
    day_axis: list[str] = []
    cursor = since.replace(hour=0, minute=0, second=0, microsecond=0)
    end = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    while cursor <= end:
        day_axis.append(cursor.strftime("%Y-%m-%d"))
        cursor = cursor + timedelta(days=1)

    # Accumulate.  op_totals rolls each op's counts so we can pick
    # the top N and merge the tail into a (other) bucket.
    op_totals: dict[str, int] = {}
    grid: dict[str, dict[str, int]] = {}
    for day, op, n in rows:
        op = op or "unknown"
        op_totals[op] = op_totals.get(op, 0) + int(n or 0)
        grid.setdefault(op, {})[day] = int(n or 0)

    TOP_N = 12
    sorted_ops = sorted(
        op_totals.items(), key=lambda kv: kv[1], reverse=True,
    )
    top_ops = [op for op, _ in sorted_ops[:TOP_N]]
    tail_ops = [op for op, _ in sorted_ops[TOP_N:]]
    if tail_ops:
        merged: dict[str, int] = {}
        for op in tail_ops:
            for day, n in grid.pop(op, {}).items():
                merged[day] = merged.get(day, 0) + n
        if merged:
            grid["(other)"] = merged
            top_ops.append("(other)")

    # Dense grid (zero-fill missing days) for renderer convenience.
    dense: dict[str, dict[str, int]] = {}
    max_calls = 0
    for op in top_ops:
        row = {}
        for day in day_axis:
            v = grid.get(op, {}).get(day, 0)
            row[day] = v
            if v > max_calls:
                max_calls = v
        dense[op] = row

    return {
        "days": day_axis,
        "operations": top_ops,
        "grid": dense,
        "max_calls": max_calls,
    }


def _llm_usage(session, days: int = 7) -> list[dict]:
    from sqlalchemy import text
    since_iso = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()
    try:
        rows = session.execute(text("""
            SELECT operation, model_name,
                   SUM(tokens), SUM(duration_seconds), COUNT(*)
            FROM llm_usage_log
            WHERE started_at >= CAST(:since AS timestamptz)
            GROUP BY operation, model_name
            ORDER BY SUM(tokens) DESC NULLS LAST
            LIMIT 20
        """), {"since": since_iso}).fetchall()
    except Exception:
        return []
    return [
        {
            "operation": r[0],
            "model": r[1],
            "tokens": _safe_int(r[2]),
            "seconds": float(r[3] or 0.0),
            "calls": _safe_int(r[4]),
        }
        for r in rows
    ]


def _ollama_installed_models() -> set[str]:
    """Phase 54.6.245 — names of every Ollama model pulled locally
    (``ollama.list()``). Distinct from ``_ollama_loaded_models`` —
    that only returns the currently-resident ones. We use *installed*
    for the ``missing_model`` alert that cross-checks the
    .env-configured assignments: a model that's configured but not
    pulled will fail on the first chat call, not just be slow.

    Returns ``None`` (not an empty set) when the Ollama client fails
    — e.g. Ollama isn't running. Downstream callers treat ``None``
    as "unknown; skip the missing-model check" rather than "every
    model is missing" so we don't spam alerts when the server is
    briefly down during monitor polling.
    """
    try:
        import ollama
        from sciknow.config import settings
        client = ollama.Client(host=settings.ollama_host)
        resp = client.list()
        names: set[str] = set()
        # The ollama python client evolved through several response
        # shapes (dict with "models" list, .models attribute, tag vs
        # name field). Handle all three defensively.
        raw = resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", None)
        for m in (raw or []):
            name = None
            if isinstance(m, dict):
                name = m.get("name") or m.get("model") or m.get("tag")
            else:
                name = (
                    getattr(m, "name", None)
                    or getattr(m, "model", None)
                    or getattr(m, "tag", None)
                )
            if name:
                names.add(str(name))
        return names
    except Exception as exc:
        logger.debug("ollama list failed: %s", exc)
        return None  # type: ignore[return-value]


def _ollama_loaded_models() -> list[dict]:
    """Currently-resident Ollama models via ``ollama.ps()``. Each
    entry carries the keep-alive expiry so we can tell "this will
    unload in 4 min" from "pinned forever"."""
    try:
        import ollama
        from sciknow.config import settings
        client = ollama.Client(host=settings.ollama_host)
        resp = client.ps()
        out: list[dict] = []
        for m in resp.models or []:
            expires_at = getattr(m, "expires_at", None)
            out.append({
                "name": getattr(m, "name", None) or getattr(m, "model", ""),
                "size_mb": _safe_int(getattr(m, "size", 0) or 0) // (1024 * 1024),
                "vram_mb": _safe_int(
                    getattr(m, "size_vram", 0) or 0
                ) // (1024 * 1024),
                "expires_at": (
                    expires_at.isoformat()
                    if expires_at and hasattr(expires_at, "isoformat")
                    else str(expires_at) if expires_at else None
                ),
            })
        return out
    except Exception as exc:
        logger.debug("ollama ps failed: %s", exc)
        return []


def _gpu_info() -> list[dict]:
    """Parse nvidia-smi CSV output into per-GPU dicts. No pynvml
    dependency — subprocess + CSV is portable and cheap."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return []
        out: list[dict] = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            out.append({
                "index": _safe_int(parts[0]),
                "name": parts[1],
                "memory_used_mb": _safe_int(parts[2]),
                "memory_total_mb": _safe_int(parts[3]),
                "utilization_pct": _safe_int(parts[4]),
                "temperature_c": _safe_int(parts[5]),
            })
        return out
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi unavailable: %s", exc)
        return []


def _qdrant_collections() -> list[dict]:
    """Per-collection row count + vector-field summary. Read-only."""
    try:
        from sciknow.storage.qdrant import get_client
        client = get_client()
        collections = client.get_collections().collections
        out: list[dict] = []
        for col in collections:
            try:
                info = client.get_collection(col.name)
                vectors = info.config.params.vectors or {}
                sparse = info.config.params.sparse_vectors or {}
                vec_names = list(vectors.keys()) if isinstance(
                    vectors, dict
                ) else []
                sparse_names = (
                    list(sparse.keys()) if isinstance(sparse, dict) else []
                )
                entry = {
                    "name": col.name,
                    "points_count": _safe_int(info.points_count or 0),
                    "vectors": vec_names,
                    "sparse_vectors": sparse_names,
                }
                # Phase 54.6.235 — cheap on-disk estimate in MB.
                entry["estimated_disk_mb"] = _qdrant_disk_estimate(entry)
                out.append(entry)
            except Exception:
                out.append({
                    "name": col.name, "points_count": 0,
                    "vectors": [], "sparse_vectors": [],
                    "estimated_disk_mb": 0,
                })
        return out
    except Exception as exc:
        logger.debug("qdrant collections query failed: %s", exc)
        return []


def _last_refresh(data_dir: Path | None) -> str | None:
    """Read the .last_refresh marker (54.6.210)."""
    if not data_dir:
        return None
    marker = Path(data_dir) / ".last_refresh"
    if not marker.exists():
        return None
    try:
        return marker.read_text(encoding="utf-8").strip() or None
    except Exception:
        return None


def collect_monitor_snapshot(
    *,
    topic_clusters_limit: int = 20,
    activity_limit: int = 15,
    throughput_days: int = 14,
    llm_usage_days: int = 7,
) -> dict:
    """Aggregate every observability source into one dict.

    Shape documented at module top. Read-only — every query is a
    SELECT or an external read (ollama ps / nvidia-smi / qdrant
    get_collections). Safe to call during active ingestion.

    Any single source that fails degrades to empty — we don't let a
    dead ollama daemon prevent the user from seeing corpus stats.
    """
    import time as _time
    from sciknow.storage.db import get_session

    # Phase 54.6.263 — time the collector itself; attached to the
    # snapshot as ``snapshot_duration_ms``.
    _snap_t0 = _time.monotonic()
    project = _project_info()
    data_dir = project.get("data_dir")

    with get_session() as session:
        # Each helper runs inside a savepoint via _safe_db — a
        # SQL failure in one (e.g. querying a missing table like
        # raptor_nodes on a fresh install) doesn't poison the
        # outer transaction for the others.
        corpus = _safe_db(session, _corpus_counts, default={})
        ingest_sources = _safe_db(session, _ingest_sources, default=[])
        converter_backends = _safe_db(
            session, _converter_backends, default=[]
        )
        topic_clusters = _safe_db(
            session, _topic_clusters,
            limit=topic_clusters_limit, default=[],
        )
        stage_timing = _safe_db(session, _pipeline_timing, default=[])
        stage_failures = _safe_db(session, _pipeline_failures, default=[])
        throughput = _safe_db(
            session, _pipeline_throughput,
            days=throughput_days, default=[],
        )
        activity = _safe_db(
            session, _recent_activity,
            limit=activity_limit, default=[],
        )
        llm_usage = _safe_db(
            session, _llm_usage, days=llm_usage_days, default=[],
        )
        # 54.6.283 — per-day × per-op grid for the heatmap renderer.
        llm_usage_by_day = _safe_db(
            session, _llm_usage_by_day,
            days=llm_usage_days, default={},
        )
        # 54.6.232 — operational additions
        rates = _safe_db(session, _ingest_rates_and_eta, default={})
        queue_states = _safe_db(session, _ingest_queue_states, default={})
        pending_downloads = _safe_db(
            session, _pending_downloads_count, default=0,
        )
        hourly_throughput = _safe_db(
            session, _hourly_throughput, hours=24, default=[],
        )
        pg_db_size_mb = _safe_db(
            session, _pg_database_size_mb, default=0,
        )
        top_failures = _safe_db(
            session, _top_failure_classes, default=[],
        )
        # 54.6.234 — host load + stuck-job + content quality
        stuck_job = _safe_db(session, _stuck_job, default={})
        meta_quality = _safe_db(session, _metadata_quality, default={})
        # 54.6.235 — year histogram + embeddings coverage
        year_hist = _safe_db(session, _year_histogram, default=[])
        embed_cov = _safe_db(session, _embeddings_coverage, default={})
        # 54.6.236 — config + coverage + cost + tree shape
        cost_totals = _safe_db(session, _llm_cost_totals, default={})
        visuals_cov = _safe_db(session, _visuals_coverage, default={})
        raptor_shape = _safe_db(session, _raptor_tree_shape, default={})
        dupe_hashes = _safe_db(session, _duplicate_hashes, default=0)
        # 54.6.280 — citation graph connectivity.
        citation_graph = _safe_db(session, _citation_graph, default={})
        # 54.6.282 — section-type coverage (chunker health signal).
        section_coverage = _safe_db(
            session, _section_coverage, default={},
        )
        # 54.6.237 — trend batch
        growth = _safe_db(session, _corpus_growth_rate, default={})
        book_act = _safe_db(session, _book_activity, default={})
        # 54.6.243 — retrieval quality + wiki materialization
        quality_sig = _safe_db(session, _corpus_quality_signals, default={})
        wiki_mat = _safe_db(session, _wiki_materialization, default={})
        # 54.6.244 — ingest funnel + hourly failure histogram
        funnel = _safe_db(session, _ingest_funnel, default=[])
        hourly_fails = _safe_db(
            session, _hourly_failures, hours=24, default=[],
        )

    # GPU sample recording happens here, outside the session ctx,
    # so the ring buffer gets one tick per snapshot call. CLI
    # watch mode populates over time; web server holds its own
    # rolling buffer per worker.
    gpu_info = _gpu_info()
    _record_gpu_sample(gpu_info)
    # 54.6.245 — cache the installed-tags list once per snapshot so
    # we don't round-trip to Ollama twice.
    _installed_ollama = _ollama_installed_models()

    snapshot = {
        "project": project,
        "corpus": corpus,
        "ingest_sources": ingest_sources,
        "converter_backends": converter_backends,
        "topic_clusters": topic_clusters,
        "pipeline": {
            "stage_timing": stage_timing,
            "stage_failures": stage_failures,
            "throughput": throughput,
            "recent_activity": activity,
            "throughput_days": throughput_days,
            # 54.6.232 additions
            "rates": rates,
            "queue_states": queue_states,
            "hourly_throughput": hourly_throughput,
            "top_failures": top_failures,
        },
        "pending_downloads": pending_downloads,
        # 54.6.234 additions
        "host": _host_load(),
        "stuck_job": stuck_job,
        "meta_quality": meta_quality,
        # 54.6.235 additions
        "year_histogram": year_hist,
        "embeddings_coverage": embed_cov,
        # 54.6.236 additions
        "model_assignments": _model_assignments(),
        "cost_totals": cost_totals,
        "visuals_coverage": visuals_cov,
        "raptor_shape": raptor_shape,
        "duplicate_hashes": dupe_hashes,
        # 54.6.280 — citation graph connectivity metrics. Populated
        # on every snapshot; empty dict when the citations table is
        # missing (fresh install).
        "citation_graph": citation_graph,
        # 54.6.282 — chunker/section-type coverage. High unknown %
        # flags a chunker regression or a new heading style the
        # regex patterns don't cover.
        "section_coverage": section_coverage,
        "bench_freshness": _bench_freshness(
            Path(data_dir) if data_dir else None
        ),
        # 54.6.250 — newest backup age + count, read from
        # archives/backups/.backup-state.json. Drives the
        # backup_stale alert (warn >7d, error >30d).
        "backup_freshness": _backup_freshness(),
        # 54.6.252 — list of active-project → .env overrides. Empty
        # when either there's no project or .env matches the project.
        # Non-empty means the user's .env is misleading at a glance.
        "config_drift": _config_drift(),
        # 54.6.260 — last 20 lines of data_dir/sciknow.log. Empty
        # when the log file doesn't exist (fresh install).
        "log_tail": _log_tail(
            Path(data_dir) if data_dir else None, n=20,
        ),
        # 54.6.262 — per-service reachability probes. Drives the
        # services_down alert + Services panel. Serial (<150ms total
        # on localhost); not cached so each snapshot call gives a
        # fresh reading.
        "services": _services_health(),
        # 54.6.237 additions
        "corpus_growth": growth,
        "book_activity": book_act,
        "bench_quality_delta": _bench_quality_delta(),
        # 54.6.238 — cross-process pulse from `sciknow refresh`.
        # Web process also overrides `active_jobs` at endpoint time;
        # from the CLI it stays an empty list.
        "refresh_pulse": _read_refresh_pulse(
            Path(data_dir) if data_dir else None
        ),
        # 54.6.246 — cross-process web-jobs pulse. CLI reads the
        # pulse file the web writes per state transition so `sciknow
        # db monitor --watch` over SSH sees active autowrite /
        # book-write / wiki-compile jobs without hitting the web
        # endpoint. The web endpoint still overrides this list at
        # /api/monitor time with its in-process direct read — which
        # is authoritative and fresher than the pulse.
        "active_jobs": _read_web_jobs_pulse(
            Path(data_dir) if data_dir else None
        ),
        "llm": {
            "usage_last_days": llm_usage,
            "usage_window_days": llm_usage_days,
            "loaded_models": _ollama_loaded_models(),
            # 54.6.283 — per-day × per-op grid for heatmap rendering.
            "usage_by_day": llm_usage_by_day,
        },
        "qdrant": _qdrant_collections(),
        "gpu": gpu_info,
        "gpu_trend": _gpu_trend_snapshot(),
        "storage": {
            "disk": _disk_usage(Path(data_dir) if data_dir else None),
            "pg_database_mb": pg_db_size_mb,
        },
        "last_refresh": _last_refresh(Path(data_dir) if data_dir else None),
        # 54.6.243 — quality signals + inbox + cross-project + alerts
        "quality_signals": quality_sig,
        "wiki_materialization": wiki_mat,
        "inbox": _inbox_pending(Path(data_dir) if data_dir else None),
        "projects_overview": _project_overview(),
        # 54.6.244 — funnel + failure sparkline + disk free
        "ingest_funnel": funnel,
        "pipeline_hourly_failures": hourly_fails,
        "disk_free": _disk_free(Path(data_dir) if data_dir else None),
        # 54.6.245 — installed Ollama tags, consumed by the
        # missing_model alert. ``None`` means the list call failed
        # (Ollama unreachable) — alert builder skips the check.
        "ollama_installed_models": (
            sorted(_installed_ollama)
            if _installed_ollama is not None else None
        ),
        "snapshotted_at": datetime.now(timezone.utc).isoformat(),
    }
    # Phase 54.6.263 — snapshot self-timing. Must be computed
    # BEFORE alerts so the snapshot_slow alert can inspect the
    # freshly-measured value. Rounded to ms for display.
    snapshot["snapshot_duration_ms"] = int(
        (_time.monotonic() - _snap_t0) * 1000
    )
    # Re-inject as a set for _build_alerts — it early-outs when the
    # value is None (Ollama down) and uses set-membership otherwise.
    # Store the sorted-list form in the snapshot so the JSON payload
    # is stable-ordered.
    snapshot["alerts"] = _build_alerts(snapshot)
    # Phase 54.6.259 — composite health score rolls up alerts +
    # coverage + hardware penalties into a single 0-100 number.
    # Must run AFTER alerts exist (_compute_health_score reads them).
    snapshot["health_score"] = _compute_health_score(snapshot)
    return snapshot
