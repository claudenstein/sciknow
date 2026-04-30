"""Phase 54.6.118 (Tier 4 #3) — corpus-drift detection.

After each expand round, compute how the corpus abstract-centroid
has shifted. Persistent log at ``<project data>/expand/drift.log``
plus a summary surfaced in ``sciknow db stats``.

The cosine-drop-to-previous-centroid is a fast proxy for "this
expansion changed what the corpus is about": 0.00 = identical,
0.05 = meaningful drift, 0.10+ = a new subtopic is being pulled in.
Might be intentional (user wanted to broaden); should be explicit.

See ``docs/research/EXPAND_ENRICH_RESEARCH_2.md`` §4.3.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_LOG_NAME = "drift.log"
_SNAPSHOT_NAME = "drift.latest.json"


def _drift_dir(project_root: Path) -> Path:
    d = Path(project_root) / "data" / "expand"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def record_drift(
    project_root: Path,
    *,
    tag: str = "",
    reason: str = "",
    also_size: int | None = None,
) -> dict[str, Any]:
    """Compute the current corpus centroid, compare to the last
    snapshot, append to the drift log, and overwrite the snapshot.

    Returns a summary dict:
      {timestamp, n_vectors, drift_cosine, drift_delta, tag, reason}
    ``drift_cosine`` is the cosine similarity of the new centroid to
    the previous one; ``drift_delta`` = 1 - drift_cosine. ``None`` when
    there's no prior snapshot yet.
    """
    from sciknow.retrieval.relevance import compute_corpus_centroid

    new_vec = compute_corpus_centroid()
    summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tag": tag,
        "reason": reason,
        "n_vectors": int(also_size) if also_size is not None else None,
        "drift_cosine": None,
        "drift_delta": None,
    }

    d = _drift_dir(project_root)
    snap_path = d / _SNAPSHOT_NAME
    prev_vec: np.ndarray | None = None
    if snap_path.exists():
        try:
            prev = json.loads(snap_path.read_text(encoding="utf-8"))
            vec = prev.get("centroid")
            if isinstance(vec, list):
                prev_vec = np.asarray(vec, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.debug("drift.latest.json unreadable: %s", exc)

    if prev_vec is not None and new_vec is not None:
        cos = _cosine(new_vec, prev_vec)
        if cos is not None:
            summary["drift_cosine"] = round(cos, 5)
            summary["drift_delta"] = round(1.0 - cos, 5)

    # Write a compact log line
    log_path = d / _LOG_NAME
    line = (
        f"{summary['timestamp']}\t"
        f"tag={tag or '-'}\t"
        f"reason={(reason or '-')[:80]}\t"
        f"n={summary['n_vectors']}\t"
        f"cosine={summary['drift_cosine']}\t"
        f"delta={summary['drift_delta']}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)

    # Persist the centroid itself for the NEXT diff
    if new_vec is not None:
        try:
            snap_path.write_text(
                json.dumps({
                    "timestamp": summary["timestamp"],
                    "tag": tag,
                    "n_vectors": summary["n_vectors"],
                    "centroid": new_vec.tolist(),
                }),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("drift.latest.json write failed: %s", exc)

    return summary


def read_recent(project_root: Path, n: int = 10) -> list[dict]:
    """Parse the tail of drift.log into structured records."""
    log_path = _drift_dir(project_root) / _LOG_NAME
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()[-n:]
    out: list[dict] = []
    for ln in lines:
        parts = ln.split("\t")
        rec: dict[str, Any] = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                rec[k] = v
            else:
                rec["timestamp"] = p
        out.append(rec)
    return out
