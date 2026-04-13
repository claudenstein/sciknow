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
