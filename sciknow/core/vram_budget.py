"""Phase 54.6.290 — VRAM budget preflight + releaser registry.

Motivation
----------
The dual-embedder (54.6.279) put bge-m3 + Qwen3-4B in the embed
cohort at the same time as the MinerU-VLM subprocess from the
preceding convert stage.  On a 24 GB 3090 that stacks to ~23.8 GB
and OOMs.  The verification (54.6.285) had to fall back to the
pipeline backend to get past it.

Before loading any heavy model, call ``preflight(need_mb, reason)``.
If the observed free VRAM is below the requested amount, registered
releasers fire in order until either the budget is met or all
releasers have run.  Releasers are functions ``() -> int`` that drop
a specific cache (or kill a specific subprocess) and return the
approximate MB freed so the log is informative.

This is the **proactive** counterpart to 54.6.286's VRAM headroom
watchdog (which alerts reactively).  Proactive preflight prevents
the OOM; the watchdog tells you when something *else* is pressuring
VRAM (other user, stale vllm, etc).

Design notes
------------
* ``nvidia-smi`` parsed as CSV (same path as core.monitor._gpu_info)
  — no pynvml dependency, portable.
* Releasers register at module import.  Call order matters: cheapest-
  to-reload first so a preflight cascade stops early.
* Releasing a bge-m3 / Qwen3 embedder costs ~8 s to reload; releasing
  a MinerU VLM costs ~30 s.  So we release embedders *before* MinerU
  only if the caller is about to re-use MinerU (e.g. next convert
  stage); at the pipeline level the call sites pick the right order.
* ``psutil`` is already a transitive dep; we use it to kill vLLM
  subprocess children when the VLM backend is in play.  Safe on
  ImportError — releasers degrade to "drop in-process caches only".

NOT goals
---------
* Not a full VRAM scheduler — no reservations, no queueing.  Single-
  process pipeline; the caller owns the sequencing.
* Not a monitor — observability lives in core.monitor.  This module
  only *acts*.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Callable

logger = logging.getLogger(__name__)

# (priority, name, fn).  Lower priority fires first in a preflight
# cascade, so releasers whose cache is cheap to rebuild / safe to
# drop get a lower number:
#
#   10  ollama_llm       cheap unload (200 ms); reload 3-10 s
#   20  pdf_converter    subprocess kill; reload ~30 s
#   50  embedders        bge-m3 + Qwen3-4B; reload ~8-15 s, and
#                        they're the thing the embed stage *needs*
#                        so it's a last-resort release
#
# Registration is at import time from the owner module (embedder,
# pdf_converter, rag.llm) — each assigns its own priority.
_RELEASERS: list[tuple[int, str, Callable[[], int]]] = []


def register_releaser(
    name: str, fn: Callable[[], int], *, priority: int = 50,
) -> None:
    """Register a VRAM releaser.

    ``priority`` orders the cascade — lower fires first.  Default 50
    is for "last resort" caches that the current stage needs.

    ``fn`` returns the approximate MB freed (best-effort — the real
    number comes from re-reading nvidia-smi).  A releaser that has
    nothing to free must still return 0 without raising.
    """
    _RELEASERS.append((priority, name, fn))
    _RELEASERS.sort(key=lambda r: r[0])


def clear_releasers() -> None:
    """Remove every registered releaser.  Test-only."""
    _RELEASERS.clear()


def registered_releaser_names() -> list[str]:
    """Read-only view of what's registered right now, firing order."""
    return [n for _p, n, _fn in _RELEASERS]


def free_vram_mb(device: int = 0) -> int:
    """Return free VRAM on ``device`` in MB.  0 if nvidia-smi fails."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return 0
        return int(out.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return 0


def total_vram_mb(device: int = 0) -> int:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return 0
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def kill_our_gpu_children(
    name_filter: tuple[str, ...] = ("vllm", "mineru"),
    *,
    timeout_s: int = 5,
) -> int:
    """Kill child processes of this Python process that hold GPU
    memory and whose cmdline matches one of ``name_filter``.

    Returns the count of processes killed.  Used by release_mineru
    to take down the vLLM subprocess the ``mineru`` pip package
    spawns for VLM-Pro.

    Safe on missing psutil (returns 0 silently).  Never kills the
    current process, its parent, or an ollama / other unrelated
    subprocess.
    """
    try:
        import psutil
    except ImportError:
        return 0

    killed = 0
    our_pid = os.getpid()
    try:
        parent = psutil.Process(our_pid)
    except psutil.NoSuchProcess:
        return 0

    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0

    for child in children:
        try:
            cmdline = " ".join(child.cmdline()).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if not any(tag in cmdline for tag in name_filter):
            continue
        if child.pid == our_pid:
            continue
        try:
            logger.info(
                "vram preflight: terminating child pid=%d (%s)",
                child.pid, cmdline[:80],
            )
            child.terminate()
            try:
                child.wait(timeout=timeout_s)
            except psutil.TimeoutExpired:
                logger.warning(
                    "vram preflight: child pid=%d did not exit in %ds; "
                    "sending SIGKILL",
                    child.pid, timeout_s,
                )
                child.kill()
                try:
                    child.wait(timeout=timeout_s)
                except psutil.TimeoutExpired:
                    pass
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(
                "vram preflight: could not kill pid=%d: %s",
                child.pid, e,
            )
    return killed


def preflight(
    need_mb: int,
    reason: str = "",
    *,
    raise_on_fail: bool = True,
    release: bool = True,
    device: int = 0,
) -> int:
    """Ensure at least ``need_mb`` MB of free VRAM on ``device``.

    Returns the free VRAM observed after releases (if any).  Raises
    ``RuntimeError`` when the budget can't be met and
    ``raise_on_fail=True`` (the default).

    When ``release=False`` this is a pure observation — useful when
    the caller wants to know the gap without actually tearing down
    caches (e.g. for a dry-run log).
    """
    free = free_vram_mb(device)
    if free >= need_mb:
        logger.debug(
            "vram preflight OK: %d MB free ≥ %d MB needed for %s",
            free, need_mb, reason or "<unspecified>",
        )
        return free

    if not release or not _RELEASERS:
        if raise_on_fail:
            raise RuntimeError(
                f"vram preflight: {free} MB free, need {need_mb} MB "
                f"for {reason!r}"
            )
        return free

    logger.info(
        "vram preflight: need %d MB for %s, only %d MB free — "
        "firing releasers",
        need_mb, reason or "<unspecified>", free,
    )

    for _priority, name, fn in _RELEASERS:
        if free >= need_mb:
            break
        try:
            reported = int(fn() or 0)
        except Exception as exc:  # pragma: no cover — releaser bugs
            logger.warning(
                "vram preflight: releaser %s raised %s", name, exc,
            )
            reported = 0
        new_free = free_vram_mb(device)
        actual_delta = new_free - free
        logger.info(
            "vram preflight: %s -> reported=%d MB, actual=%+d MB "
            "(now %d MB free)",
            name, reported, actual_delta, new_free,
        )
        free = new_free

    if free < need_mb and raise_on_fail:
        raise RuntimeError(
            f"vram preflight: {free} MB free after all releasers, "
            f"need {need_mb} MB for {reason!r}"
        )
    return free
