"""SQLAlchemy connection layer.

Phase 43b — project-aware. The module-level ``engine`` / ``SessionLocal``
still point at the active project's database (because ``settings.pg_url``
now reads through ``core.project.get_active_project().pg_database``), so
existing call sites keep working unchanged. A new ``get_engine(db_name)``
accessor lets cross-project operations (``sciknow project init``, the
one-shot migration) target arbitrary databases without hacking env vars.

Design notes:

- Engines are cached per URL. Creating a fresh engine every call is
  expensive (each one spins up its own connection pool); but we do need
  a separate engine per target DB, because PostgreSQL connections are
  bound to one database for their lifetime.
- ``get_admin_engine()`` connects to the ``postgres`` administrative
  database. That's where ``CREATE DATABASE`` / ``DROP DATABASE`` must
  run from — you can't create a database while connected to it, and
  you can't drop the one you're using.
"""
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from sciknow.config import settings


def _pg_url_for(db_name: str) -> str:
    """Build a PostgreSQL URL for an arbitrary database on the same host.

    All connection parameters (host, port, user, password) come from
    settings; only the database segment is swapped. Callers use this to
    target projects other than the currently active one, or the
    ``postgres`` admin DB for CREATE/DROP.
    """
    return (
        f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{db_name}"
    )


@lru_cache(maxsize=8)
def _engine_for_url(url: str) -> Engine:
    """Return a cached SQLAlchemy engine for a given URL.

    ``lru_cache`` keys on the full URL, so swapping the password or the
    host produces a distinct engine. The cache size is a soft cap: in
    practice a single process rarely talks to more than 2-3 databases
    (active project + admin + maybe one migration target).
    """
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=settings.pg_pool_size,
        max_overflow=settings.pg_max_overflow,
    )


def get_engine(db_name: str | None = None) -> Engine:
    """Return a cached engine targeting ``db_name``.

    When ``db_name`` is ``None`` (the common case) the active project's
    database is used — same as the legacy module-level ``engine`` below,
    now served out of the cache so repeated calls share the pool.

    Pass an explicit ``db_name`` when you need to reach into a project
    other than the active one (``sciknow project init`` creates a
    database, the migration command dumps from one DB and restores into
    another, etc.).
    """
    if db_name is None:
        url = settings.pg_url
    else:
        url = _pg_url_for(db_name)
    return _engine_for_url(url)


def get_admin_engine() -> Engine:
    """Engine bound to the ``postgres`` admin database.

    ``CREATE DATABASE`` and ``DROP DATABASE`` must run from a session
    that isn't itself using the target database. The admin DB is the
    conventional neutral ground.
    """
    return get_engine("postgres")


# Legacy module-level handles — retained for backwards compatibility
# with call sites that import ``engine`` / ``SessionLocal`` directly.
# These point at the active project's DB because ``settings.pg_url``
# is project-aware (Phase 43a).
engine = get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session(
    db_name: str | None = None,
) -> Generator[Session, None, None]:
    """Yield a session bound to the active project's DB (default) or to
    an explicit ``db_name``.

    Usage unchanged for the 99% case: ``with get_session() as session:``
    continues to target the active project. The ``db_name`` parameter
    is the Phase 43b escape hatch for cross-project operations.
    """
    if db_name is None:
        factory = SessionLocal
    else:
        factory = sessionmaker(bind=get_engine(db_name), expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(db_name: str | None = None) -> bool:
    """``SELECT 1`` smoke test for the target database.

    Used by the root CLI's preflight to fail fast with a clear error if
    PostgreSQL is down, and by ``sciknow project show`` to report per-
    project health.
    """
    try:
        with get_engine(db_name).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
