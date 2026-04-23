"""
Centralized logging for sciknow.

Sets up a rotating file logger at `data/sciknow.log` that captures:
  - Every CLI invocation (command + args)
  - Errors + full tracebacks
  - Warnings from all sciknow modules
  - LLM call summaries (model, tokens, duration)
  - Ingestion stage timing
  - Qdrant/Postgres connection issues

The log file rotates at 10 MB with 3 backups (data/sciknow.log,
data/sciknow.log.1, data/sciknow.log.2, data/sciknow.log.3).

Usage from any module:
    import logging
    logger = logging.getLogger("sciknow")
    logger.info("something happened")
    logger.error("something broke", exc_info=True)

The logger is configured once at CLI startup via setup_logging().
"""
import logging
import logging.handlers
from pathlib import Path


def setup_logging(log_dir: str | None = None, level: int = logging.DEBUG) -> None:
    """Configure the sciknow logger with a rotating file handler.

    Safe to call multiple times — skips if already configured.

    Phase 43d — when ``log_dir`` is ``None`` (the normal case), the log
    is written to the active project's ``data/sciknow.log``. Pass an
    explicit path to override (used by tests and for ad-hoc debugging).
    """
    logger = logging.getLogger("sciknow")
    if logger.handlers:
        return  # already configured

    logger.setLevel(level)

    if log_dir is None:
        # Lazy import — logging_config is imported early in the CLI
        # callback and we don't want a hard dep on settings at module
        # import time.
        from sciknow.config import settings
        log_base = Path(settings.data_dir)
    else:
        log_base = Path(log_dir)

    log_path = log_base / "sciknow.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def attach_uvicorn_to_sciknow_log() -> None:
    """Phase 54.6.308 — route uvicorn's loggers into data/sciknow.log.

    By default uvicorn writes access + error logs to its own stderr
    handlers, so ``sciknow book serve`` left the sciknow log silent
    on GUI clicks.  We can't fully debug GUI-triggered issues without
    seeing which endpoints the browser hit (especially the SSE
    endpoints that back long-running autowrite/review jobs).

    This helper attaches sciknow's rotating file handler to the three
    uvicorn loggers (``uvicorn``, ``uvicorn.access``, ``uvicorn.error``)
    so each HTTP request lands in ``<project>/data/sciknow.log`` with
    the same formatter as the rest of the CLI log — one file to grep.

    ``propagate=False`` is already the uvicorn default; we set it
    explicitly here to be defensive in case uvicorn's internals ever
    change.  Also sets level=INFO on uvicorn.access — anything less
    and the per-request lines would be filtered before the handler
    ever sees them.

    Idempotent: safe to call more than once.
    """
    import sys

    sciknow_logger = logging.getLogger("sciknow")
    if not sciknow_logger.handlers:
        # setup_logging() hasn't run yet — caller bug, but don't crash.
        return
    sk_handler = sciknow_logger.handlers[0]
    # Mirror to stderr too so the interactive ``sciknow book serve``
    # terminal still shows "Application startup complete", request
    # lines, and shutdown — passing ``log_config=None`` to uvicorn.run
    # suppresses its built-in stream handler, and without this the
    # user's terminal goes silent after the startup banner.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(sk_handler.formatter)
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        has_file = any(
            getattr(h, "baseFilename", None) == getattr(sk_handler, "baseFilename", None)
            for h in lg.handlers
        )
        if not has_file:
            lg.addHandler(sk_handler)
        has_stream = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in lg.handlers
        )
        if not has_stream:
            lg.addHandler(stream_handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False
