"""Phase 54.6.238 — cross-process pulse files for the monitor.

Long-running processes (``sciknow refresh``, the FastAPI web
reader) write short JSON "heartbeat" files into
``<data_dir>/monitor/`` that ``sciknow db monitor`` /
``/api/monitor`` read to surface live progress.

Why files and not a shared DB table:

  * Each process already talks to the DB — adding writes there
    would create a DB-sized background noise for a few KB of
    state.
  * Files survive process crash (stale pulse shows `age > N`
    and the monitor can flag it).
  * Trivial to mmap, `inotify`, or just stat from any other
    language (operators can `cat` the pulse in bash too).

Schema: every pulse file carries
  {"pid": int, "pulse_at": "<iso-utc>", "type": "refresh|web", ...}

The monitor considers any pulse older than ``STALE_THRESHOLD_S``
(120s by default) as evidence the writing process died — it still
renders the data but tags it `STALE`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STALE_THRESHOLD_S: float = 120.0


def _monitor_dir(data_dir: Path | None) -> Path | None:
    if not data_dir:
        return None
    d = Path(data_dir) / "monitor"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_pulse(
    data_dir: Path | None, kind: str, payload: dict[str, Any],
) -> Path | None:
    """Append common fields + atomic-rename-write the payload.

    Atomic via tmp-file + rename so a partial write never exposes
    a half-serialised JSON to a concurrent reader. Silent no-op
    when data_dir is None (pulse is best-effort — never blocks
    the calling process).
    """
    mdir = _monitor_dir(data_dir)
    if not mdir:
        return None
    out_path = mdir / f"{kind}.json"
    tmp_path = mdir / f".{kind}.json.tmp"
    try:
        body = {
            "pid": os.getpid(),
            "pulse_at": datetime.now(timezone.utc).isoformat(),
            "type": kind,
            **payload,
        }
        tmp_path.write_text(
            json.dumps(body, default=str), encoding="utf-8",
        )
        tmp_path.replace(out_path)
        return out_path
    except Exception:
        return None


def read_pulse(
    data_dir: Path | None, kind: str,
) -> dict[str, Any] | None:
    """Return the parsed pulse payload, or None when absent /
    unreadable / older than STALE_THRESHOLD_S * 10 (ancient — the
    writer is definitively gone). Stale-but-recent pulses come
    back with `age_s` set so the caller can decide how to render.
    """
    mdir = _monitor_dir(data_dir)
    if not mdir:
        return None
    path = mdir / f"{kind}.json"
    if not path.exists():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Compute age in seconds for the caller
    try:
        pulsed_at = datetime.fromisoformat(
            body.get("pulse_at", "").replace("Z", "+00:00")
        )
        age_s = (
            datetime.now(timezone.utc) - pulsed_at
        ).total_seconds()
        body["age_s"] = age_s
        body["is_stale"] = age_s > STALE_THRESHOLD_S
    except Exception:
        body["age_s"] = None
        body["is_stale"] = False
    # Treat ancient pulses as gone.
    if body.get("age_s") is not None and body["age_s"] > \
            STALE_THRESHOLD_S * 10:
        return None
    return body


def clear_pulse(data_dir: Path | None, kind: str) -> None:
    """Remove the pulse file (e.g. on clean refresh completion)."""
    mdir = _monitor_dir(data_dir)
    if not mdir:
        return
    path = mdir / f"{kind}.json"
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
