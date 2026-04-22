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
                out.append({
                    "name": col.name,
                    "points_count": _safe_int(info.points_count or 0),
                    "vectors": vec_names,
                    "sparse_vectors": sparse_names,
                })
            except Exception:
                out.append({
                    "name": col.name, "points_count": 0,
                    "vectors": [], "sparse_vectors": [],
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
        },
        "llm": {
            "usage_last_days": llm_usage,
            "usage_window_days": llm_usage_days,
            "loaded_models": _ollama_loaded_models(),
        },
        "qdrant": _qdrant_collections(),
        "gpu": _gpu_info(),
        "last_refresh": _last_refresh(Path(data_dir) if data_dir else None),
        "snapshotted_at": datetime.now(timezone.utc).isoformat(),
    }
