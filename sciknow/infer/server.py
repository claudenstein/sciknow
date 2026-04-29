"""Subprocess lifecycle for llama-server (writer/embedder/reranker).

Each role runs as a managed subprocess on its own port. PID + log paths
live under ``data/infer/<role>/`` so multiple sciknow projects can share
one infer stack (the file layout is keyed by role, not by project).

This module is deliberately small and synchronous. A real production
deployment would use systemd-user units; for the local-first sciknow
case, a PID file + a managed subprocess is enough — and rolls back
cleanly on Ctrl-C because we register an atexit hook.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from sciknow.config import settings

logger = logging.getLogger("sciknow.infer.server")

# Roles that the manager knows about. Each role has a default port,
# default model, and a recommended set of llama-server flags. Profiles
# (default / low-vram / spec-dec) override these at runtime.
ROLE_DEFAULTS: dict[str, dict] = {
    "writer": {
        "port": 8090,
        "model": settings.writer_model_gguf,
        # Phase 55.V19 (2026-04-29 morning, 2nd attempt) — moving back
        # to ctx=262144 + q4_0 KV in a clean environment. The first
        # attempt at this config produced 0.12 scores, but that was
        # confounded with (a) the verifier-fix prompt tightening
        # (reverted in d208135), (b) a polluted corpus from expand-
        # thin-sections — 1532 chunks of off-topic stellar physics
        # (deleted), (c) the relevance filter being too lax (now
        # 0.55 → 0.75 + reranker pass). Re-trying the long-ctx
        # substrate in this clean state. Slate #1b measured
        # Q4_K_M + ctx=262144 + q4_0 KV at 22.4 GB peak.
        # REQUIRES WRITER_MODEL_GGUF=...Qwen3.6-27B-Q4_K_M.gguf in
        # .env (16.8 GB; UD-Q4_K_XL at 17.6 GB pushes total over 24).
        "ctx_size": 262144,
        "n_gpu_layers": 999,         # all layers on GPU
        "parallel": 1,
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
        ],
    },
    "embedder": {
        "port": 8091,
        # V2_FINAL Stage 3: --ctx-size is the *total* context across all
        # parallel slots. With --parallel 4 + --ctx-size 8192, each slot
        # gets only 2048 tokens — too tight for real corpus chunks (some
        # exceed 2048 tokens after tokenisation). Bumping to 32768 gives
        # each of 4 slots its own 8192 window, matching bge-m3's max.
        "model": settings.embedder_model_gguf,
        "ctx_size": 32768,
        "n_gpu_layers": 999,
        "parallel": 4,
        # --ubatch-size must be >= the longest single input or
        # llama-server returns 500 "input too large to process".
        # bge-m3 honours up to 8192 tokens; chunker.py already caps at
        # 8192, so matching ubatch to per-slot ctx is the safe default.
        "extra_flags": [
            "--embedding", "--pooling", "mean",
            "--batch-size", "8192",
            "--ubatch-size", "8192",
        ],
    },
    "reranker": {
        "port": 8092,
        "model": settings.reranker_model_gguf,
        # bge-reranker-v2-m3 supports 8192 max sequence length. We set
        # ctx_size, batch_size, and ubatch_size all to 8192 because the
        # reranker receives (query, document) pairs where the document
        # can be a RAPTOR summary node (full summary_text up to ~700
        # tokens) — the prior 4096/512 defaults rejected pairs > 512
        # tokens with `input is too large to process. increase the
        # physical batch size`. Same reasoning as the embedder role.
        "ctx_size": 8192,
        "n_gpu_layers": 999,
        "parallel": 1,
        "extra_flags": [
            "--reranking",
            "--batch-size", "8192",
            "--ubatch-size", "8192",
        ],
    },
    # v2.0 — visuals captioner. Loads a multimodal Qwen3-VL GGUF
    # + the mmproj sidecar that bridges the vision encoder into the
    # LM embedding space. Hot-swaps with the writer on a 3090 (both
    # ~17 GB) — `corpus caption-visuals` manages the down/up cycle
    # so users don't have to think about VRAM pressure during a
    # captioning batch.
    "vlm": {
        "port": 8093,
        "model": settings.vlm_model_gguf,
        "mmproj": settings.vlm_mmproj_gguf,
        # Phase 55.V19 — bumped 16384 → 32768. Adds ~0.75 GB KV
        # (Qwen3-VL is 48 layers × 4 kv_heads × 64 head_dim, fp16).
        # Total VLM footprint ~18.5 GB on a 24 GB 3090, still 5+ GB
        # headroom. Useful for figures with substantial OCR'd text
        # in the image + caption + mention paragraphs context.
        "ctx_size": 32768,
        "n_gpu_layers": 999,
        "parallel": 1,
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",
        ],
    },
    # Phase 55.S1 — autowrite scorer. Independent llama-server instance
    # loaded with a NON-Qwen GGUF so the score / rescore phases break
    # the same-family self-bias documented in arXiv:2506.22316 /
    # 2508.06709. Default candidate (when SCORER_MODEL_GGUF is set):
    # gemma-4-31B-it-Q4_1.gguf.
    #
    # Phase 55.V1 — ctx_size restored to 16384. The earlier 8192 cap
    # was a workaround for OOM-during-load with the embedder +
    # reranker still resident. Phase 55.V1 evicts them too (see
    # `_VRAM_CONFLICTS` below), so the scorer now gets the full
    # context that score / verify / CoVe prompts need on long
    # sections. The conflict map is what makes this safe; do NOT
    # raise ctx_size without keeping the eviction policy in place.
    #
    # Phase 55.V6 (2026-04-28) — bumped 16384 → 24576 to match the
    # writer. Verify prompts can be ~12K tokens (full draft + retrieved
    # evidence + claims template); ~16K headroom prevents the same
    # context-overflow that hit the writer's planning phase. Gemma
    # 4 31B Q4_1 @ 24K KV-cache peaks at ~22 GB on the 3090 — within
    # margin since gemma is alone in score phase.
    #
    # The role won't start unless SCORER_MODEL_GGUF is set; `infer up
    # --role scorer` no-ops with a clear error otherwise. On a 3090
    # the scorer cannot co-reside with the writer — the auto-swap in
    # `up()` handles this transparently.
    "scorer": {
        "port": 8094,
        "model": settings.scorer_model_gguf,
        # Phase 55.V19 (2026-04-29 morning, 2nd attempt) — moving to
        # ctx=131077 + q4_0 KV per user spec to mirror the writer's
        # long-ctx config. Slate #5 measured ctx=131072 + q4_0 KV at
        # 22.7 GB peak (0.7 GB headroom on the 3090) — 131077 is the
        # same in practice. Tight but fits. The 262144 row OOM'd in
        # slate #5 (gemma 18 GB + q4_0 KV at 262K + buffer ~25 GB).
        "ctx_size": 131077,
        "n_gpu_layers": 999,
        "parallel": 1,
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
        ],
    },
    # Phase 55.V18 — metadata extractor. A small ~6 GB GGUF
    # (Qwen3.5-9B-Q5_K_M by default) dedicated to ingestion's
    # rare metadata Layer 4 fallback. Co-resides with embedder +
    # reranker + MinerU during ingest (~14 GB total on a 24 GB
    # 3090) so the convert→embed loop never has to evict anything.
    # ctx_size=8192 is plenty: the prompt is the first ~3000
    # chars of the PDF + a short JSON-extraction template.
    "extractor": {
        "port": 8095,
        "model": settings.extractor_model_gguf,
        # Phase 55.V19 — bumped 8192 → 131077 with q8_0 KV.
        # Extractor handles ingestion's metadata Layer 4 + (Phase
        # 55.V19 fix) wiki concept extraction. Real prompts are
        # short (~3000 chars / ~750 tokens), so 131K is overkill
        # for capacity, but the spare ctx is free with q8_0 KV
        # (~2.1 GB reserved vs 4.2 GB at fp16). Co-resident with
        # embedder + reranker + MinerU during ingest:
        #   model 6 + KV 2 + buf 1 = ~9 GB extractor
        #   + embedder 1.1 + reranker 0.7 + MinerU 6 = ~16.8 GB
        #   on 24 GB → ~7 GB free. Healthy.
        # q8_0 over q4_0: short structured-output prompts (JSON
        # metadata) are precision-sensitive; q8_0's PPL delta is
        # negligible while q4_0 has the GGML #5932 GQA risk.
        "ctx_size": 131077,
        "n_gpu_layers": 999,
        "parallel": 1,
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
        ],
    },
}


@dataclass
class ProcInfo:
    """Snapshot of a managed llama-server subprocess."""
    role: str
    port: int
    pid: int | None
    model: str
    healthy: bool
    log_path: Path
    extra: dict = field(default_factory=dict)


def _state_dir(role: str) -> Path:
    """Per-role state dir. Lives outside the project tree so role state
    is shared across project switches (the writer model + KV cache
    don't care which project we ask questions of)."""
    base = Path(os.path.expanduser("~/.cache/sciknow/infer"))
    d = base / role
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pid_file(role: str) -> Path:
    return _state_dir(role) / "pid"


def _log_file(role: str) -> Path:
    return _state_dir(role) / "server.log"


def _read_pid(role: str) -> int | None:
    pf = _pid_file(role)
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text().strip())
    except (ValueError, OSError):
        return None
    # Verify the PID is actually running.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        # Exists but not ours — treat as alive (don't reap).
        return pid
    return pid


def _write_pid(role: str, pid: int) -> None:
    _pid_file(role).write_text(str(pid))


def _clear_pid(role: str) -> None:
    pf = _pid_file(role)
    if pf.exists():
        pf.unlink()


def _resolve_url(role: str) -> str:
    return {
        "writer": settings.infer_writer_url,
        "embedder": settings.infer_embedder_url,
        "reranker": settings.infer_reranker_url,
        "vlm": settings.infer_vlm_url,
        "scorer": settings.infer_scorer_url,
        "extractor": settings.infer_extractor_url,
    }[role]


def health(role: str, timeout: float = 1.0) -> bool:
    """Return True iff the role's HTTP endpoint responds 200 to /health."""
    url = _resolve_url(role).rstrip("/") + "/health"
    try:
        r = httpx.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def wait_healthy(role: str, timeout_s: float = 120.0) -> bool:
    """Poll /health until ready or timeout. Returns True on success."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if health(role, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def _build_argv(role: str, profile: str = "default") -> list[str]:
    """Compose the llama-server command line for a role under a profile."""
    cfg = dict(ROLE_DEFAULTS[role])
    port = cfg["port"]
    model = cfg["model"]
    if not model:
        raise RuntimeError(
            f"No model configured for role={role!r}. Set "
            f"{role.upper()}_MODEL_GGUF in .env or pass --model."
        )

    binary = settings.llama_server_binary
    if not Path(binary).exists():
        raise RuntimeError(
            f"llama-server binary not found at {binary!r}. "
            f"Set LLAMA_SERVER_BINARY in .env."
        )

    argv = [
        binary,
        "--model", str(model),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--ctx-size", str(cfg["ctx_size"]),
        "--n-gpu-layers", str(cfg["n_gpu_layers"]),
        "--parallel", str(cfg["parallel"]),
        "--log-colors", "off",
    ]
    # Multimodal projector — required for vision-capable roles.
    mmproj = cfg.get("mmproj")
    if mmproj:
        if not Path(str(mmproj)).exists():
            raise RuntimeError(
                f"Multimodal projector not found at {mmproj!r}. Set "
                f"{role.upper()}_MMPROJ_GGUF in .env."
            )
        argv += ["--mmproj", str(mmproj)]

    # Profile overrides
    if profile == "low-vram" and role in ("embedder", "reranker"):
        # Drop these to CPU; writer keeps GPU.
        argv = [a for a in argv if a != "999"]
        argv += ["--n-gpu-layers", "0"]
    if profile == "spec-dec" and role == "writer" and settings.draft_model_gguf:
        argv += [
            "--draft-model", str(settings.draft_model_gguf),
            "--draft-max", "16",
            "--draft-min", "2",
        ]

    argv += cfg["extra_flags"]
    return argv


# Phase 55.V1 — VRAM conflict map. Each role names the OTHER roles
# that cannot co-reside on a single 24 GB GPU.
#
# The three big-model roles (writer Qwen3.6-27B ~17 GB, scorer
# Gemma-4-31B ~18 GB, vlm Qwen3-VL-30B-A3B ~17 GB) each occupy enough
# VRAM that any pair would OOM. They also pin enough that even the
# small embedder (~1.1 GB bge-m3) + reranker (~700 MB bge-reranker-v2-m3)
# co-residency tips the scorer over the 24 GB ceiling at 16K KV cache
# (verified empirically 2026-04-27: compute_pp buffer alloc 522 MB
# failed at peak). So Phase 55.V1 makes embedder/reranker conflict
# with the big roles too: when running writer or scorer, evict
# everything else; when running retrieval, evict the big roles.
#
# This is bidirectional and intentionally aggressive — the user's
# directive (2026-04-27): "always when using either the writer or
# the scorer unload any other model from the vram". The retrieve→
# generate transition costs ~6 s of swap; that's accepted in exchange
# for full-quality (16K context, no KV cache hack) generation.
#
# When `up(role)` is called, we down every healthy conflicting role
# first and wait for VRAM to actually free (the CUDA context release
# lags process exit by ~0.5–1.5 s).
#
# DGX Spark note: with 128 GB unified memory, the conflict map can
# legitimately become empty. For now we keep the conservative 3090
# defaults; a future config flag (``vram_co_residence_ok=True``) lets
# Spark users skip the swap entirely. The conflict-resolver respects
# this flag — see `_should_evict_for_vram()`.
_VRAM_CONFLICTS: dict[str, set[str]] = {
    "writer":   {"scorer", "vlm", "extractor", "embedder", "reranker"},
    "scorer":   {"writer", "vlm", "extractor", "embedder", "reranker"},
    "vlm":      {"writer", "scorer", "extractor", "embedder", "reranker"},
    # Phase 55.V18 — extractor (~6 GB Qwen3.5-9B) co-resides with
    # the small retrieval roles + MinerU during ingest (sum
    # ~14 GB on a 24 GB 3090). It's only the big-model peers it
    # has to evict to start.
    "extractor": {"writer", "scorer", "vlm"},
    "embedder": {"writer", "scorer", "vlm"},
    "reranker": {"writer", "scorer", "vlm"},
}

# Empirical grace after llama-server SIGTERM exit before the GPU's
# VRAM is fully reclaimed. CUDA context teardown is async; if we
# start the next role before this lands, llama.cpp's cuMemAlloc may
# still see the old allocation and OOM. 1.5 s is enough on the 3090
# for the 17–18 GB models we route through here.
_VRAM_RELEASE_GRACE_S = 1.5

_swap_lock = threading.Lock()


def _should_evict_for_vram() -> bool:
    """Return True iff the conflict-driven eviction policy is in effect.

    Phase 55.V1 — DGX Spark and other big-VRAM hosts can co-reside
    every role; setting `VRAM_CO_RESIDENCE_OK=true` in the env makes
    `_free_vram_for` and `activate_phase` no-op. Default False (the
    24 GB 3090 case).
    """
    return not bool(getattr(settings, "vram_co_residence_ok", False))


def _free_vram_for(role: str, *, exclude: set[str] | None = None) -> list[str]:
    """Stop every healthy role that conflicts with ``role`` for VRAM.

    Returns the list of role names that were actually downed. A no-op
    when no conflict is healthy (the common steady-state case once a
    role has been hot for a while). Holds ``_swap_lock`` so two
    concurrent up() calls don't race the conflict resolution.

    ``exclude`` lets callers protect specific roles from eviction
    (used by `activate_phase` to keep peer roles co-resident — e.g.
    keep the embedder up when bringing the reranker up for retrieval).
    """
    if not _should_evict_for_vram():
        return []
    conflicts = _VRAM_CONFLICTS.get(role, set())
    if exclude:
        conflicts = conflicts - exclude
    if not conflicts:
        return []
    downed: list[str] = []
    with _swap_lock:
        for c in conflicts:
            try:
                if health(c, timeout=0.3):
                    logger.info(
                        "infer: role=%s starting → freeing VRAM by stopping "
                        "conflicting role=%s", role, c,
                    )
                    if down(c, timeout=5.0):
                        downed.append(c)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "infer: free_vram_for(%s) failed to down(%s): %s — "
                    "continuing; the start may OOM",
                    role, c, exc,
                )
        # CUDA context release lags process exit. Wait for VRAM to
        # actually free before letting the caller start the new role.
        if downed:
            time.sleep(_VRAM_RELEASE_GRACE_S)
            logger.info(
                "infer: freed %s for role=%s (waited %.1fs for CUDA release)",
                ",".join(downed), role, _VRAM_RELEASE_GRACE_S,
            )
    return downed


# Phase 55.V1 — phase → required-roles map. Each phase declares which
# roles it needs hot at the same time. `activate_phase(name)` ensures
# every required role is up and every other role (per `_VRAM_CONFLICTS`)
# is down. Callers pass through this single API instead of issuing
# `up(...)` calls one at a time, so the conflict resolver can reason
# about the full set rather than evicting peers one role at a time.
#
# Phase semantics:
#   "retrieve"  — embedder + reranker up; big models down. Used before
#                 every retrieval / lessons-fetch / RAPTOR / step-back.
#   "generate"  — writer up alone. Used for all writer-driven prose.
#   "score"     — scorer up alone (or writer when scorer is unconfigured;
#                 the rag.llm fallback handles model resolution).
#   "vlm"       — vlm up alone. Used for caption-visuals.
_PHASE_ROLES: dict[str, set[str]] = {
    "retrieve": {"embedder", "reranker"},
    "generate": {"writer"},
    "score":    {"scorer"},
    "vlm":      {"vlm"},
}


def activate_phase(phase: str, *, wait_healthy_s: float = 120.0) -> dict:
    """Bring up the roles required for ``phase`` and evict everything else.

    Returns a dict ``{"up": [...], "down": [...]}`` — actually-changed
    role names — so callers (and tests) can observe the swap. No-op
    when the substrate is already in the desired state, which is the
    happy path for back-to-back calls in the same phase.

    Phase 55.V1 — the autowrite engine and any code that bridges a
    retrieval step to a generation step should call this at the
    boundary (not before every llm call — once per phase transition
    is enough). Doing so once is the load-bearing fix for the
    long-standing "embedder + reranker linger in VRAM during writer
    generation" bug that tanked decode rate from 30 t/s to ~4 t/s
    pre-Phase 55.V1.

    On VRAM_CO_RESIDENCE_OK=True hosts (DGX Spark) eviction is skipped
    entirely; the function still up()s the required roles.
    """
    if phase not in _PHASE_ROLES:
        raise ValueError(
            f"Unknown phase: {phase!r}. Choose from {sorted(_PHASE_ROLES)}"
        )
    required = _PHASE_ROLES[phase]
    # Validate the required roles are actually configured (e.g.
    # don't try to bring up scorer if SCORER_MODEL_GGUF is empty).
    runnable = {r for r in required if ROLE_DEFAULTS.get(r, {}).get("model")}
    if not runnable:
        # The "score" phase falls through to writer when scorer isn't
        # configured — the rag.llm dispatch handles the rerouting.
        # Other phases with no runnable roles are a misconfiguration
        # but we'd rather no-op than crash here.
        if phase == "score":
            runnable = {"writer"}
        else:
            logger.warning(
                "activate_phase(%s): no runnable roles in %s; no-op",
                phase, sorted(required),
            )
            return {"up": [], "down": []}

    # Evict every conflicting role across the entire required set,
    # *excluding* the required roles themselves (so e.g. activating
    # "retrieve" doesn't evict the embedder when bringing up the
    # reranker, since they intentionally co-reside).
    out_down: list[str] = []
    out_up: list[str] = []
    if _should_evict_for_vram():
        with _swap_lock:
            # Compute the union of every conflict set, minus the
            # required roles.
            conflicts: set[str] = set()
            for r in runnable:
                conflicts |= _VRAM_CONFLICTS.get(r, set())
            conflicts -= runnable
            for c in conflicts:
                try:
                    if health(c, timeout=0.3):
                        if down(c, timeout=5.0):
                            out_down.append(c)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "activate_phase(%s): down(%s) failed: %s",
                        phase, c, exc,
                    )
            if out_down:
                time.sleep(_VRAM_RELEASE_GRACE_S)
                logger.info(
                    "activate_phase(%s): evicted %s (waited %.1fs)",
                    phase, ",".join(out_down), _VRAM_RELEASE_GRACE_S,
                )

    # Bring up the required roles. up() will internally call
    # _free_vram_for() too, but with `exclude=runnable` so it doesn't
    # tear down the peer we just brought up in the same phase.
    for r in runnable:
        try:
            if not health(r, timeout=0.3):
                up(r, wait=True)
                out_up.append(r)
            # else: already healthy, no-op
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "activate_phase(%s): up(%s) failed: %s",
                phase, r, exc,
            )
            raise

    return {"up": out_up, "down": out_down}


@contextlib.contextmanager
def hot_phase(phase: str):
    """Context-manager wrapper around activate_phase.

    Usage:
        with hot_phase("retrieve"):
            results = retrieve(...)
        with hot_phase("generate"):
            text = stream_writer(...)

    Doesn't tear down the phase on exit — the *next* phase activation
    handles the eviction. This avoids unnecessary up/down churn when
    consecutive blocks happen to share the same phase.
    """
    activate_phase(phase)
    try:
        yield
    finally:
        pass


def up(role: str, profile: str = "default", wait: bool = True) -> ProcInfo:
    """Start a llama-server for the role if not already healthy.

    Idempotent: returns the existing ProcInfo if a healthy process is
    already running on the configured port.

    Phase 55.S1 — before starting, downs any role that conflicts with
    this one for VRAM (see ``_VRAM_CONFLICTS``) so that loading the
    new model never OOMs against a stale residency.
    """
    if role not in ROLE_DEFAULTS:
        raise ValueError(f"Unknown role: {role!r}. Choose from {list(ROLE_DEFAULTS)}")

    if health(role, timeout=0.5):
        pid = _read_pid(role)
        cfg = ROLE_DEFAULTS[role]
        logger.info("infer up: role=%s already healthy on :%s pid=%s",
                    role, cfg["port"], pid)
        return ProcInfo(role=role, port=cfg["port"], pid=pid,
                        model=str(cfg["model"]), healthy=True,
                        log_path=_log_file(role))

    # Free VRAM by stopping any conflicting big-model role before we
    # spend ~5–10 s loading this one. Without this guard, an autowrite
    # iteration with USE_LLAMACPP_SCORER=true would CUDA-OOM the moment
    # it transitions writer → scorer (both ~17–18 GB on a 24 GB GPU).
    _free_vram_for(role)

    argv = _build_argv(role, profile=profile)
    log_path = _log_file(role)
    log_fp = open(log_path, "ab")
    log_fp.write(f"\n=== sciknow infer up role={role} profile={profile} "
                 f"ts={time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n".encode())
    log_fp.write(f"argv: {' '.join(argv)}\n".encode())
    log_fp.flush()

    proc = subprocess.Popen(
        argv,
        stdout=log_fp, stderr=log_fp,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_pid(role, proc.pid)
    logger.info("infer up: role=%s started pid=%s port=%s argv=%s",
                role, proc.pid, ROLE_DEFAULTS[role]["port"], " ".join(argv))

    info = ProcInfo(
        role=role, port=ROLE_DEFAULTS[role]["port"], pid=proc.pid,
        model=str(ROLE_DEFAULTS[role]["model"]),
        healthy=False, log_path=log_path,
    )
    if wait:
        info.healthy = wait_healthy(role, timeout_s=180.0)
        if not info.healthy:
            tail = _tail_log(log_path, n=40)
            raise RuntimeError(
                f"llama-server role={role} did not become healthy within 180s. "
                f"Last log lines:\n{tail}"
            )
    return info


def _find_pid_by_port(port: int) -> int | None:
    """Phase 55.V3 — fallback for the PID-file-out-of-sync case.

    When the PID file got cleared (manually rm'd, or killed by an
    earlier `down()` mid-startup, or a stale-grace race), but the
    process is still alive on its port, `_read_pid()` returns None
    and `down()` would refuse to act. This helper greps lsof / ss
    for the listening process so we can recover the PID and kill
    it. Returns None when no listener is found.
    """
    # Prefer ss (procfs-backed, present on every modern Linux).
    try:
        out = subprocess.run(
            ["ss", "-tlnp", "sport", f"= :{port}"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except Exception:  # noqa: BLE001
        out = ""
    # Output line format: ``LISTEN 0 0 127.0.0.1:8090 0.0.0.0:* users:(("llama-server",pid=12345,fd=3))``
    import re as _re
    m = _re.search(r"pid=(\d+)", out)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def down(role: str, timeout: float = 5.0) -> bool:
    """Stop the role's llama-server. Returns True if a process was killed.

    Phase 55.V3 — when the PID file is missing but the role's port is
    answering /health, fall back to ``_find_pid_by_port`` so the
    eviction still happens. Without this fallback, an out-of-sync
    state (PID file cleared but process alive) wedges the conflict
    map: every subsequent ``activate_phase`` would see the role as
    healthy, refuse to evict it, and the swap silently no-ops.
    """
    pid = _read_pid(role)
    if pid is None:
        # Fallback: maybe the PID file got cleared but the server is
        # still running. Probe the port.
        try:
            port = ROLE_DEFAULTS[role]["port"]
            recovered_pid = _find_pid_by_port(int(port))
        except Exception:  # noqa: BLE001
            recovered_pid = None
        if recovered_pid is None:
            return False
        logger.warning(
            "infer down: role=%s PID file missing but port %s answered "
            "with pid=%s — recovering and killing", role,
            ROLE_DEFAULTS[role]["port"], recovered_pid,
        )
        pid = recovered_pid
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            _clear_pid(role)
            return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            _clear_pid(role)
            logger.info("infer down: role=%s pid=%s stopped", role, pid)
            return True
        time.sleep(0.1)
    # Force.
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _clear_pid(role)
    logger.warning("infer down: role=%s pid=%s SIGKILLed", role, pid)
    return True


def status() -> list[ProcInfo]:
    """Return ProcInfo for every known role (running or not)."""
    out: list[ProcInfo] = []
    for role, cfg in ROLE_DEFAULTS.items():
        pid = _read_pid(role)
        out.append(ProcInfo(
            role=role, port=cfg["port"], pid=pid,
            model=str(cfg["model"]),
            healthy=health(role, timeout=0.5),
            log_path=_log_file(role),
        ))
    return out


def swap(role: str, model: str, profile: str = "default") -> ProcInfo:
    """Replace the role's running model. Down → reconfigure → up."""
    down(role)
    # Override the resolved model before re-up.
    ROLE_DEFAULTS[role]["model"] = model
    return up(role, profile=profile, wait=True)


def _tail_log(path: Path, n: int = 40) -> str:
    if not path.exists():
        return "(no log file)"
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 8192)
            f.seek(-chunk, os.SEEK_END)
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-n:])
    except Exception as exc:
        return f"(failed to tail: {exc})"


def tail_log(role: str, n: int = 40) -> str:
    return _tail_log(_log_file(role), n=n)


def pause_for_ingest(
    roles: list[str] | None = None,
    *,
    restart_on_exit: bool = True,
):
    """Context manager that stops the writer/embedder/reranker for the
    duration of an ingest call so MinerU / Marker get the GPU.

    Phase 54.6.x — the user's own complaint: if I'm ingesting, I'm not
    writing, so why is the writer holding 18 GB of VRAM and forcing
    MinerU onto CPU? This helper:

      1. Snapshots which configured roles currently have a healthy
         llama-server (writer / embedder / reranker by default).
      2. Stops each one (SIGTERM, fall back to SIGKILL).
      3. Yields.
      4. On exit, restarts every role that was up at entry — best-effort.
         A restart failure is logged but does NOT raise; the user can
         always re-up manually.

    Used by the PDF converter (when it detects GPU pressure caused by
    sciknow's own llama-server processes) and by the corpus / db
    ingest commands as a defensive wrap.

    Returns the list of (role, ProcInfo-at-entry) tuples for callers
    that want to log what was paused.
    """
    import contextlib
    import logging as _logging
    log = _logging.getLogger("sciknow.infer.server")

    if roles is None:
        # The roles that can pin VRAM. vlm intentionally NOT paused —
        # the converter MIGHT use it (the auto VLM-Pro path doesn't go
        # through the substrate today, but if it does in future, we
        # don't want to kill the very process the converter is about
        # to call). Phase 55.S1: scorer added — when the user has
        # USE_LLAMACPP_SCORER=true and the autowrite scorer was the
        # last loaded role, ingest needs to evict it too (~18 GB).
        roles = ["writer", "embedder", "reranker", "scorer"]

    @contextlib.contextmanager
    def _ctx():
        snapshots: list[tuple[str, "ProcInfo"]] = []
        for r in roles:
            try:
                pid = _read_pid(r)
                if pid and health(r, timeout=0.3):
                    cfg = ROLE_DEFAULTS.get(r) or {}
                    snapshots.append((r, ProcInfo(
                        role=r,
                        port=int(cfg.get("port", 0)),
                        pid=pid,
                        model=str(cfg.get("model", "")),
                        healthy=True,
                        log_path=_log_file(r),
                    )))
            except Exception as exc:  # noqa: BLE001
                log.debug("pause_for_ingest: snapshot %s failed: %s", r, exc)

        for r, _info in snapshots:
            try:
                down(r, timeout=5.0)
                log.info(
                    "pause_for_ingest: stopped role=%s to free GPU for ingest",
                    r,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "pause_for_ingest: down(%s) raised %s — continuing", r, exc,
                )

        try:
            yield snapshots
        finally:
            if not restart_on_exit:
                return
            for r, _info in snapshots:
                try:
                    up(r, wait=False)
                    log.info("pause_for_ingest: restarting role=%s", r)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "pause_for_ingest: up(%s) failed on restart: %s — "
                        "user can re-up manually with `sciknow infer up "
                        "--role %s`",
                        r, exc, r,
                    )

    return _ctx()


def running_role_pids() -> set[int]:
    """Return the set of PIDs that the substrate is currently
    managing (writer/embedder/reranker/vlm). Used by the PDF
    converter to decide whether GPU pressure is "our own" (safe to
    pause) vs. external (must fall back to CPU)."""
    pids: set[int] = set()
    for role in ROLE_DEFAULTS:
        try:
            pid = _read_pid(role)
            if pid:
                pids.add(int(pid))
        except Exception:  # noqa: BLE001
            pass
    return pids


# Best-effort: if this Python process started a writer subprocess and
# is exiting, leave the writer running. Users explicitly call
# `sciknow infer down` to stop. We do NOT register atexit shutdown —
# the whole point of the substrate is that it survives sciknow CLI
# invocations so prompt-cache stays warm.
