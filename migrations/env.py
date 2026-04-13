"""Alembic migration environment.

Phase 43b — project-aware. Two ways to target a specific database:

1. **Default** (most common): reads ``settings.pg_url``, which via
   Phase 43a delegates to ``core.project.get_active_project().pg_database``.
   Every alembic invocation honours ``SCIKNOW_PROJECT`` / ``--project`` /
   ``.active-project`` the same way the rest of the CLI does.

2. **Explicit override**: ``uv run alembic -x db_name=sciknow_foo upgrade head``
   forces migrations to run against ``sciknow_foo`` regardless of the
   active project. Used by ``sciknow project init``'s migration step to
   target the brand-new database it just created (before that DB is
   even a "project" on disk).
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from sciknow.config import settings
from sciknow.storage.models import Base

config = context.config

# Phase 43b — respect `-x db_name=...` if passed. Falls back to the
# active project's pg_url via settings. The -x arg is set with e.g.
#   alembic -x db_name=sciknow_foo upgrade head
_x_args = context.get_x_argument(as_dictionary=True)
_db_override = (_x_args.get("db_name") or "").strip()
if _db_override:
    # Rebuild the URL with the overridden DB segment while keeping the
    # host / port / user / password from settings.
    _url = (
        f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{_db_override}"
    )
else:
    _url = settings.pg_url
config.set_main_option("sqlalchemy.url", _url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
