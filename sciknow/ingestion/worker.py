"""
Subprocess worker for directory ingestion.

Reads absolute PDF paths from stdin (one per line), processes each with
the full ingestion pipeline, and writes JSON status lines to stdout.

This isolation ensures that a native crash (glibc heap corruption, SIGABRT,
SIGSEGV) inside Marker or CUDA only kills this worker process — the parent
director detects the non-zero exit, marks the crashed file as failed, and
spawns a fresh worker for the remaining files.

Protocol:
  stdin  → one absolute path per line
  stdout → JSON lines, one per status event:
    {"file": "...", "status": "started"}
    {"file": "...", "status": "done",    "doc_id": "..."}
    {"file": "...", "status": "skipped"}
    {"file": "...", "status": "failed",  "error": "..."}
"""
import json
import sys
from pathlib import Path


def _emit(msg: dict) -> None:
    print(json.dumps(msg), flush=True)


def main() -> None:
    import os
    force = os.environ.get("SCIKNOW_FORCE_INGEST", "0") == "1"
    # Provenance tag recorded in documents.ingest_source on first insert.
    # Defaults to 'seed'; `sciknow corpus expand` sets it to 'expand' so the
    # resulting rows can be distinguished in db stats / future pruning.
    ingest_source = os.environ.get("SCIKNOW_INGEST_SOURCE", "seed")

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        path = Path(line)
        _emit({"file": str(path), "status": "started"})
        try:
            from sciknow.ingestion.pipeline import AlreadyIngested, PipelineError, ingest
            doc_id = ingest(path, force=force, ingest_source=ingest_source)
            _emit({"file": str(path), "status": "done", "doc_id": str(doc_id)})
        except AlreadyIngested:
            _emit({"file": str(path), "status": "skipped"})
        except PipelineError as exc:
            _emit({"file": str(path), "status": "failed", "error": str(exc)[:300]})
        except Exception as exc:
            _emit({"file": str(path), "status": "failed", "error": str(exc)[:300]})


if __name__ == "__main__":
    main()
