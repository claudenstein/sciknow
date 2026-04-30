"""Phase 43a — per-project scoping for sciknow.

Sciknow originally ran as a single tenant: one PostgreSQL database
(``sciknow``), one set of Qdrant collections (``papers`` / ``abstracts`` /
``wiki``), one ``data/`` directory. Phase 43 introduces *projects* so a
single install can host multiple independent knowledge bases (e.g. a
"global-cooling" climate book and a "materials-science" one) without
cross-contamination.

This module is the single source of truth for resolving which project
is currently active. It has **no runtime dependencies on settings** so
``sciknow/config.py`` can safely import it without a circular import.

See ``docs/reference/PROJECTS.md`` for the full design (sharing policy, one-shot
migration path, CLI surface).

## Precedence

``get_active_project()`` resolves in this order:

1. ``--project <slug>`` CLI flag (implemented by exporting
   ``SCIKNOW_PROJECT`` in the root Typer callback, Phase 43g).
2. ``SCIKNOW_PROJECT`` environment variable (for scripts / CI).
3. ``.active-project`` file at repo root (stateful, set by
   ``sciknow project use <slug>`` — Phase 43e).
4. **Legacy fallback**: if no ``projects/`` directory exists, return the
   ``default`` project whose paths match today's single-tenant layout
   (``data/``, ``sciknow`` DB, unprefixed Qdrant collections). Ensures
   pre-migration installs keep working on first run of new code.
5. If ``projects/`` exists but none is selected, warn + pick
   alphabetically first.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("sciknow.project")

# Slug charset: lowercase alphanumerics + hyphens, must start and end
# alphanumeric. Mirrors the user-facing validation rule accepted by
# `sciknow project init`. Kept strict because the slug becomes part of a
# PostgreSQL database name and a Qdrant collection name, both of which
# tolerate a much narrower alphabet than arbitrary user strings.
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

# Sentinel slug for the legacy (pre-Phase-43) layout. Not a valid
# user-facing slug — underscore rules out the regex above, so no user
# project can ever collide with it.
_DEFAULT_SLUG = "default"


def _repo_root() -> Path:
    """Walk up from this file to the repository root.

    ``sciknow/core/project.py`` → ``sciknow/core`` → ``sciknow`` → repo.
    """
    return Path(__file__).resolve().parents[2]


def validate_slug(slug: str) -> str:
    """Normalise + validate a project slug.

    Raises ``ValueError`` if the slug doesn't match ``[a-z0-9-]``.
    Accepts the ``default`` sentinel for internal use (but the CLI
    refuses to create a project with that name).
    """
    if slug == _DEFAULT_SLUG:
        return slug
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"Project slug must match [a-z0-9-] (start and end "
            f"alphanumeric, no leading/trailing hyphen). Got: {slug!r}"
        )
    return slug


def _slug_to_sqlsafe(slug: str) -> str:
    """Hyphens in slugs are fine for the filesystem but breakafunction for PostgreSQL
    (unquoted identifiers can't contain ``-``) and are awkward for Qdrant
    payload filters. Convert to underscores for any downstream identifier.
    """
    return slug.replace("-", "_")


@dataclass(frozen=True)
class Project:
    """A resolved project identity.

    Instances are cheap to create and fully determined by ``slug``
    (plus the repo root, which we pin at construction time so worktree
    moves don't break running code). All derived paths / DB names /
    collection names are computed on demand.

    The ``default`` project is a legacy shim: it represents the
    single-tenant layout that existed before Phase 43. Its paths and
    names are intentionally the same as pre-migration values, so the
    CLI keeps working on a system that hasn't run
    ``sciknow project init`` yet.
    """

    slug: str
    repo_root: Path

    @classmethod
    def default(cls) -> "Project":
        """Legacy project — pre-multi-project paths + DB name."""
        return cls(slug=_DEFAULT_SLUG, repo_root=_repo_root())

    @property
    def is_default(self) -> bool:
        return self.slug == _DEFAULT_SLUG

    # ── Filesystem ──────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Project root directory.

        For the legacy ``default`` project this is the repo root itself
        (so ``data/`` sits directly under the repo, matching pre-Phase-43
        behaviour). For real projects it's ``projects/<slug>/``.
        """
        if self.is_default:
            return self.repo_root
        return self.repo_root / "projects" / self.slug

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def env_overlay_path(self) -> Path:
        """Optional per-project .env that layers on top of the root .env.

        Created empty by ``sciknow project init``. Users can override
        per-project settings (e.g. a different ``LLM_MODEL``) here
        without touching the global .env.
        """
        return self.root / ".env.overlay"

    # ── PostgreSQL ──────────────────────────────────────────────────

    @property
    def pg_database(self) -> str:
        """PostgreSQL database name for this project.

        ``default`` → ``sciknow`` (legacy). Real projects →
        ``sciknow_<sql_safe_slug>`` so the DB name is a valid unquoted
        PostgreSQL identifier regardless of the slug's hyphens.
        """
        if self.is_default:
            return "sciknow"
        return f"sciknow_{_slug_to_sqlsafe(self.slug)}"

    # ── Qdrant ──────────────────────────────────────────────────────

    @property
    def qdrant_prefix(self) -> str:
        """Prefix applied to every Qdrant collection name.

        Empty for ``default`` (so the legacy collections ``papers`` /
        ``abstracts`` / ``wiki`` keep working without migration). Real
        projects get ``<sql_safe_slug>_``.
        """
        if self.is_default:
            return ""
        return f"{_slug_to_sqlsafe(self.slug)}_"

    @property
    def papers_collection(self) -> str:
        return f"{self.qdrant_prefix}papers"

    @property
    def abstracts_collection(self) -> str:
        return f"{self.qdrant_prefix}abstracts"

    @property
    def wiki_collection(self) -> str:
        return f"{self.qdrant_prefix}wiki"

    @property
    def visuals_collection(self) -> str:
        return f"{self.qdrant_prefix}visuals"

    # ── Existence check ─────────────────────────────────────────────

    def exists(self) -> bool:
        """True when the project's directory exists on disk.

        The ``default`` project is always considered to exist (it's
        the legacy pre-migration layout). For real projects, existence
        is ``projects/<slug>/`` being a directory.
        """
        if self.is_default:
            return True
        return self.root.is_dir()


# ── Discovery + persistence ────────────────────────────────────────────


def _projects_root() -> Path:
    return _repo_root() / "projects"


def _active_project_file() -> Path:
    return _repo_root() / ".active-project"


def list_projects() -> list[Project]:
    """Return all projects in ``projects/`` in alphabetical slug order."""
    root = _projects_root()
    if not root.exists():
        return []
    out: list[Project] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and _SLUG_RE.match(p.name):
            out.append(Project(slug=p.name, repo_root=_repo_root()))
    return out


def read_active_slug_from_file() -> str | None:
    """Read the persisted active-project slug, if any.

    Returns ``None`` when ``.active-project`` is missing, empty, or
    whitespace-only. The caller is responsible for validating the slug.
    """
    f = _active_project_file()
    if not f.exists():
        return None
    s = f.read_text().strip()
    return s or None


def write_active_slug(slug: str) -> None:
    """Persist the active-project slug to ``.active-project``.

    Used by ``sciknow project use <slug>`` (Phase 43e). Validates the
    slug before writing so a malformed file can't be left behind.

    Phase 54.6.21 — write atomically via a tempfile + replace so a
    concurrent reader can never see a partially-written file (would
    fail ``validate_slug`` on read) and two concurrent writers can't
    interleave their bytes. ``Path.replace`` is atomic on POSIX and
    Windows ≥ 10.
    """
    validate_slug(slug)
    f = _active_project_file()
    tmp = f.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(slug + "\n")
    tmp.replace(f)


def get_active_project() -> Project:
    """Resolve the active project.

    Resolution order (highest priority first):

    1. ``SCIKNOW_PROJECT`` environment variable (set by the root Typer
       callback for ``--project`` flag, or by the user directly).
    2. ``.active-project`` file at repo root (stateful).
    3. Legacy fallback: no ``projects/`` dir → ``default`` project.
    4. Projects exist but none selected → alphabetically first, with a
       one-time warning.

    This function is deliberately cheap so it can be called from inside
    ``settings``' ``default_factory`` without load-time overhead or
    circular import risk.
    """
    env_slug = os.environ.get("SCIKNOW_PROJECT")
    if env_slug and env_slug.strip():
        slug = validate_slug(env_slug.strip())
        return Project(slug=slug, repo_root=_repo_root())

    file_slug = read_active_slug_from_file()
    if file_slug:
        slug = validate_slug(file_slug)
        return Project(slug=slug, repo_root=_repo_root())

    projects = list_projects()
    if not projects:
        # Pre-Phase-43 layout: no projects/ directory has been created
        # yet. Return the legacy singleton so the CLI keeps working on
        # a clean install until `sciknow project init` is run.
        return Project.default()

    # Projects exist but none was chosen. Warn and default to the
    # alphabetically first — deterministic, and the user can silence
    # this by running `sciknow project use <slug>` once.
    logger.warning(
        "No active project set (SCIKNOW_PROJECT unset, .active-project "
        "missing). Falling back to %r. Run `sciknow project use <slug>` "
        "to select a different one explicitly.",
        projects[0].slug,
    )
    return projects[0]
