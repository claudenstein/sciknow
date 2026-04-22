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
    """
    try:
        from sciknow.config import settings
        return {
            "llm_main": settings.llm_model or None,
            "llm_fast": settings.llm_fast_model or None,
            "caption_vlm": (
                getattr(settings, "caption_vlm_model", None)
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
    from sciknow.storage.db import get_session

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
        # 54.6.237 — trend batch
        growth = _safe_db(session, _corpus_growth_rate, default={})
        book_act = _safe_db(session, _book_activity, default={})

    # GPU sample recording happens here, outside the session ctx,
    # so the ring buffer gets one tick per snapshot call. CLI
    # watch mode populates over time; web server holds its own
    # rolling buffer per worker.
    gpu_info = _gpu_info()
    _record_gpu_sample(gpu_info)

    return {
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
        "bench_freshness": _bench_freshness(
            Path(data_dir) if data_dir else None
        ),
        # 54.6.237 additions
        "corpus_growth": growth,
        "book_activity": book_act,
        "bench_quality_delta": _bench_quality_delta(),
        "llm": {
            "usage_last_days": llm_usage,
            "usage_window_days": llm_usage_days,
            "loaded_models": _ollama_loaded_models(),
        },
        "qdrant": _qdrant_collections(),
        "gpu": gpu_info,
        "gpu_trend": _gpu_trend_snapshot(),
        "storage": {
            "disk": _disk_usage(Path(data_dir) if data_dir else None),
            "pg_database_mb": pg_db_size_mb,
        },
        "last_refresh": _last_refresh(Path(data_dir) if data_dir else None),
        "snapshotted_at": datetime.now(timezone.utc).isoformat(),
    }
