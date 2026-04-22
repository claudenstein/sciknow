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
        corpus = _corpus_counts(session)
        ingest_sources = _ingest_sources(session)
        converter_backends = _converter_backends(session)
        topic_clusters = _topic_clusters(
            session, limit=topic_clusters_limit
        )
        stage_timing = _pipeline_timing(session)
        stage_failures = _pipeline_failures(session)
        throughput = _pipeline_throughput(session, days=throughput_days)
        activity = _recent_activity(session, limit=activity_limit)
        llm_usage = _llm_usage(session, days=llm_usage_days)
        # Phase 54.6.232 — operational additions
        rates = _ingest_rates_and_eta(session)
        queue_states = _ingest_queue_states(session)
        pending_downloads = _pending_downloads_count(session)
        hourly_throughput = _hourly_throughput(session, hours=24)
        pg_db_size_mb = _pg_database_size_mb(session)
        top_failures = _top_failure_classes(session)
        # Phase 54.6.234 — host load + stuck-job + content quality
        stuck_job = _stuck_job(session)
        meta_quality = _metadata_quality(session)
        # Phase 54.6.235 — year histogram + embeddings coverage
        year_hist = _year_histogram(session)
        embed_cov = _embeddings_coverage(session)

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
        "llm": {
            "usage_last_days": llm_usage,
            "usage_window_days": llm_usage_days,
            "loaded_models": _ollama_loaded_models(),
        },
        "qdrant": _qdrant_collections(),
        "gpu": _gpu_info(),
        "storage": {
            "disk": _disk_usage(Path(data_dir) if data_dir else None),
            "pg_database_mb": pg_db_size_mb,
        },
        "last_refresh": _last_refresh(Path(data_dir) if data_dir else None),
        "snapshotted_at": datetime.now(timezone.utc).isoformat(),
    }
