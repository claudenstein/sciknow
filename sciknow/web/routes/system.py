"""``sciknow.web.routes.system`` — system-info + admin endpoints.

v2 Phase E (route split) — extracted from `web/app.py`. Groups
the small "tell me about the running server" endpoints that don't
fit anywhere else: stats, settings.models, monitor, monitor/alerts-md,
admin/release-vram.

Cross-module deps (via standard lazy `_app` shim):
  - /api/monitor reads `_jobs`, `_job_lock`, `_job_tps` for the
    active-jobs sidebar.
"""
from __future__ import annotations

import logging
import time as _time

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import text

from sciknow.config import settings
from sciknow.storage.db import get_session

logger = logging.getLogger("sciknow.web.routes.system")
router = APIRouter()


@router.get("/api/settings/models")
async def api_settings_models():
    """54.6.106 — effective model assignments, surfaced in the Book
    Settings → Models tab. Read-only snapshot of the Settings object.

    Phase 54.6.244 — exposes ``book_write_model`` (added alongside the
    qwen3.6:27b-dense writer-role split in 54.6.243). The Models tab
    renders it with an explicit "inherits LLM_MODEL" indicator when
    unset, so the user can tell "writer explicitly pinned to the
    global" apart from "writer role defaults to the global".
    """
    from sciknow.config import settings as _s
    return JSONResponse({
        "llm_model": _s.llm_model,
        "llm_fast_model": _s.llm_fast_model,
        "book_write_model": getattr(_s, "book_write_model", None),
        "book_outline_model": getattr(_s, "book_outline_model", None),
        "book_review_model": _s.book_review_model,
        "autowrite_scorer_model": _s.autowrite_scorer_model,
        "visuals_caption_model": _s.visuals_caption_model,
        "mineru_vlm_model": getattr(_s, "mineru_vlm_model", None),
        "embedding_model": _s.embedding_model,
        "reranker_model": _s.reranker_model,
    })


@router.get("/api/stats")
async def api_stats():
    """Aggregate stats for the enhanced dashboard panel.

    Mirrors a subset of `sciknow db stats` + `catalog raptor stats` +
    `catalog topics`. Cheap to compute — runs four counts plus two
    GROUP BYs against PostgreSQL, no Qdrant scrolls except for the
    RAPTOR level breakdown which uses an indexed payload filter so it's
    O(N_summary_nodes) not O(N_chunks).
    """
    out: dict = {}
    with get_session() as session:
        out["n_documents"] = session.execute(text(
            "SELECT COUNT(*) FROM documents"
        )).scalar() or 0
        out["n_completed"] = session.execute(text(
            "SELECT COUNT(*) FROM documents WHERE ingestion_status = 'complete'"
        )).scalar() or 0
        out["n_chunks"] = session.execute(text(
            "SELECT COUNT(*) FROM chunks"
        )).scalar() or 0
        out["n_citations"] = session.execute(text(
            "SELECT COUNT(*) FROM citations"
        )).scalar() or 0

        rows = session.execute(text("""
            SELECT ingest_source, COUNT(*) FROM documents
            GROUP BY ingest_source ORDER BY COUNT(*) DESC
        """)).fetchall()
        out["ingest_sources"] = [{"source": r[0] or "unknown", "n": r[1]} for r in rows]

        rows = session.execute(text("""
            SELECT topic_cluster, COUNT(*) FROM paper_metadata
            WHERE topic_cluster IS NOT NULL AND topic_cluster != ''
            GROUP BY topic_cluster ORDER BY COUNT(*) DESC LIMIT 20
        """)).fetchall()
        out["topic_clusters"] = [{"name": r[0], "n": r[1]} for r in rows]

        try:
            out["n_wiki_pages"] = session.execute(text(
                "SELECT COUNT(*) FROM wiki_pages"
            )).scalar() or 0
        except Exception:
            out["n_wiki_pages"] = 0

    raptor_levels: dict[str, int] = {}
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from sciknow.storage.qdrant import PAPERS_COLLECTION, get_client
        qdrant = get_client()
        for lvl in (0, 1, 2, 3, 4):
            try:
                info = qdrant.count(
                    collection_name=PAPERS_COLLECTION,
                    count_filter=Filter(must=[
                        FieldCondition(key="node_level", match=MatchValue(value=lvl))
                    ]),
                    exact=False,
                )
                n = info.count if hasattr(info, "count") else int(info)
                if n > 0:
                    raptor_levels[f"L{lvl}"] = n
            except Exception:
                pass
    except Exception:
        pass
    out["raptor_levels"] = raptor_levels

    return JSONResponse(out)


@router.get("/api/monitor/alerts-md")
async def api_monitor_alerts_md():
    """Phase 54.6.268 — return current alerts as a Markdown block.

    Shares ``core.monitor.alerts_as_markdown`` with the CLI
    ``sciknow db monitor --alerts-md`` so both UIs produce the same
    paste-ready format. Returns ``text/plain`` so copy-to-clipboard
    in the browser sees unescaped Markdown.
    """
    from sciknow.core.monitor import (
        collect_monitor_snapshot, alerts_as_markdown,
    )
    snap = collect_monitor_snapshot()
    return PlainTextResponse(alerts_as_markdown(snap))


@router.get("/api/monitor")
async def api_monitor(days: int = 14):
    """Phase 54.6.230 — unified monitor snapshot for the web reader.

    One endpoint, one dict — same shape as ``sciknow db monitor
    --json`` because both call ``core.monitor.collect_monitor_
    snapshot``. The web "System Monitor" modal polls this every
    5s. Read-only; safe during active ingestion.
    """
    from sciknow.core.monitor import collect_monitor_snapshot
    from sciknow.core.project import get_active_project
    from sciknow.core.pulse import read_pulse
    from sciknow.web import app as _app
    snap = collect_monitor_snapshot(
        throughput_days=max(1, int(days)),
        llm_usage_days=max(1, int(days)),
    )
    # Active web jobs — same process, direct read of _jobs dict.
    active: list[dict] = []
    now = _time.monotonic()
    with _app._job_lock:
        for jid, j in _app._jobs.items():
            if j.get("status") not in ("running", "starting"):
                continue
            started = j.get("started_at") or now
            active.append({
                "id": jid[:8],
                "type": j.get("task_desc") or j.get("job_type") or "?",
                "model": j.get("model_name") or None,
                "tokens": j.get("tokens", 0),
                "tps": round(_app._job_tps(j), 2),
                "elapsed_s": max(0, now - started),
                "target_words": j.get("target_words"),
                "stream_state": j.get("stream_state"),
            })
    snap["active_jobs"] = active
    try:
        active_project = get_active_project()
        snap["refresh_pulse"] = read_pulse(
            active_project.data_dir, "refresh",
        )
    except Exception:
        snap["refresh_pulse"] = None
    return JSONResponse(snap)


@router.post("/api/admin/release-vram")
async def api_admin_release_vram():
    """Phase 54.6.320 — runtime VRAM-eviction switch.

    Call this while an autowrite / verify job is running and decode
    tok/s has tanked because bge-m3 + reranker + ColBERT got reloaded
    by an iteration's retrieve step and are now squeezing Ollama's
    writer model into a partial GPU load (the classic "decode 4 t/s
    instead of 30 t/s" regression).

    Frees the retrieval models held by THIS process (sciknow web
    server) and reports the VRAM delta. Ollama doesn't auto-rebalance
    a partial-load on its own — we also issue a model-unload via
    Ollama's API so the next LLM call re-pages it with full GPU.
    """
    import shutil
    import subprocess
    from sciknow.core.book_ops import _release_gpu_models

    def _vram_used() -> tuple[int, int, int]:
        """(used_mib, free_mib, total_mib) on GPU 0; (-1,-1,-1) if nvidia-smi missing."""
        if not shutil.which("nvidia-smi"):
            return (-1, -1, -1)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=5,
            )
            parts = [int(x.strip()) for x in r.stdout.strip().split(",")]
            return (parts[0], parts[1], parts[2])
        except Exception:
            return (-1, -1, -1)

    before = _vram_used()
    _release_gpu_models()

    unloaded: list[str] = []
    try:
        import ollama as _ollama
        client = _ollama.Client(host=settings.ollama_host, timeout=10)
        try:
            ps = client.ps()
            loaded = [m.get("name") or m.get("model") for m in ps.get("models", [])]
        except Exception:
            loaded = []
        for name in loaded:
            if not name:
                continue
            try:
                client.generate(model=name, prompt="", keep_alive=0, stream=False)
                unloaded.append(name)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("ollama unload during release-vram failed: %s", exc)

    after = _vram_used()
    return JSONResponse({
        "ok": True,
        "vram_before_mib": {"used": before[0], "free": before[1], "total": before[2]},
        "vram_after_mib": {"used": after[0], "free": after[1], "total": after[2]},
        "ollama_unloaded": unloaded,
        "note": (
            "Retrieval models freed and Ollama models unloaded. The "
            "next LLM call (the running job's next phase) will re-page "
            "the writer model with the full GPU budget. Decode tok/s "
            "should jump from the partial-load value back to native."
        ),
    })
