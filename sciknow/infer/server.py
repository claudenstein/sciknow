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
import json
import logging
import os
import signal
import subprocess
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
        "ctx_size": 16384,
        "n_gpu_layers": 999,         # all layers on GPU
        "parallel": 1,
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",
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
        "ctx_size": 4096,
        "n_gpu_layers": 999,
        "parallel": 1,
        "extra_flags": ["--reranking"],
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
        "ctx_size": 16384,
        "n_gpu_layers": 999,
        "parallel": 1,
        "extra_flags": [
            "--cont-batching",
            "--flash-attn", "on",
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


def up(role: str, profile: str = "default", wait: bool = True) -> ProcInfo:
    """Start a llama-server for the role if not already healthy.

    Idempotent: returns the existing ProcInfo if a healthy process is
    already running on the configured port.
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


def down(role: str, timeout: float = 5.0) -> bool:
    """Stop the role's llama-server. Returns True if a process was killed."""
    pid = _read_pid(role)
    if pid is None:
        return False
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


# Best-effort: if this Python process started a writer subprocess and
# is exiting, leave the writer running. Users explicitly call
# `sciknow infer down` to stop. We do NOT register atexit shutdown —
# the whole point of the substrate is that it survives sciknow CLI
# invocations so prompt-cache stays warm.
