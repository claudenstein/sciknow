"""`sciknow library` — database + infrastructure lifecycle.

Spec §5.1: this is the v2 home for `init`, `reset`, `stats`, `migrate`,
`validate`, `snapshot` (and the operational helpers backup / restore /
doctor / monitor / dashboard / failures / drift / provenance /
audit-sidecar that live alongside them).

The implementations remain in ``sciknow.cli.db`` for now; this module
re-exports the command callables under the renamed subapp so the v2
verb structure is live immediately. ``cli/main.py`` registers a
``sciknow db`` deprecation shim that prints a one-shot warning then
dispatches to the same callables — users get a smooth migration window.

Phase F continuation: physically moving the command bodies out of
db.py is a separate cleanup commit that requires moving the helper
functions too (~3 kLOC of supporting code). Re-registration is the
contract change.
"""
from __future__ import annotations

import typer

from sciknow.cli import db as _db

app = typer.Typer(
    name="library",
    help="Database + infrastructure lifecycle (init, reset, stats, "
         "migrate, validate, snapshot, backup, doctor, monitor).",
    no_args_is_help=True,
)

# ── core lifecycle (spec verbs) ─────────────────────────────────────────
app.command(name="init")(_db.init)
app.command(name="reset")(_db.reset)
app.command(name="stats")(_db.stats)

# ── operational helpers ─────────────────────────────────────────────────
app.command(name="backup")(_db.backup)
app.command(name="restore")(_db.restore)
app.command(name="failures")(_db.failures)
app.command(name="doctor")(_db.doctor)
app.command(name="monitor")(_db.monitor)
app.command(name="dashboard")(_db.dashboard)

# ── audits / drift detection ────────────────────────────────────────────
app.command(name="audit-sidecar")(_db.audit_sidecar_cmd)
app.command(name="drift")(_db.drift_cmd)
app.command(name="provenance")(_db.provenance_cmd)


@app.command(name="migrate")
def migrate_cmd():
    """Run pending Alembic migrations against the active project's DB.

    Equivalent to ``uv run alembic upgrade head`` — wrapped here so
    library lifecycle stays a one-stop shop.
    """
    import subprocess
    import sys
    rc = subprocess.call(["uv", "run", "alembic", "upgrade", "head"])
    sys.exit(rc)


@app.command(name="validate")
def validate_cmd():
    """Validate that the SQLAlchemy models match the Alembic head.

    Equivalent to ``alembic check`` — fails non-zero if drift is
    detected (i.e. someone added a column to models.py without an
    accompanying migration).
    """
    import subprocess
    import sys
    rc = subprocess.call(["uv", "run", "alembic", "check"])
    sys.exit(rc)


@app.command(name="snapshot")
def snapshot_cmd():
    """One-shot pg_dump + Qdrant snapshot of the active project.

    Spec §5.1 names this verb explicitly; for now it delegates to the
    existing ``library backup`` flow.
    """
    # The v1 'backup' helper is the snapshot facility. Surfacing it
    # under a stable name now means future Phase G `import-v1` can
    # call `library snapshot` cleanly.
    from sciknow.cli.db import backup as _backup
    # `_backup` reads its args from the typer context — but it has
    # defaults that produce a complete tarball. Invoke directly.
    _backup()
