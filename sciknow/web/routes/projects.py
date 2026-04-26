"""``sciknow.web.routes.projects`` — project lifecycle endpoints.

v2 Phase E (route split): extracted from `web/app.py` so the app
module stays close to mount points + middleware + lifespan.
Behaviour unchanged — every handler is still the same generator,
takes the same Request body shape, and returns the same JSONResponse
schema. The only edit is `@app.<m>(...)` → `@router.<m>(...)`.

7 routes (matches MIGRATION.md tally for /api/projects + the venues
sub-tree):
  GET  /api/projects                  list with health
  GET  /api/projects/{slug}           show details + counts
  POST /api/projects/use              set active (writes .active-project)
  POST /api/projects/init             create new (DB + Qdrant + dir)
  GET  /api/projects/{slug}/venues    venue allow/block list
  POST /api/projects/{slug}/venues    upsert venue rule
  POST /api/projects/destroy          drop DB + collections + data dir
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

logger = logging.getLogger("sciknow.web.routes.projects")
router = APIRouter()


@router.get("/api/projects")
async def api_projects_list():
    """List all projects with their active marker + health."""
    import os as _os
    from sciknow.cli.project import _pg_database_exists
    from sciknow.core.project import (
        get_active_project, list_projects, read_active_slug_from_file,
    )
    active_slug = (
        _os.environ.get("SCIKNOW_PROJECT")
        or read_active_slug_from_file()
        or get_active_project().slug
    )
    out = []
    for p in list_projects():
        try:
            db_ok = _pg_database_exists(p.pg_database)
        except Exception:
            db_ok = False
        out.append({
            "slug": p.slug,
            "active": p.slug == active_slug,
            "pg_database": p.pg_database,
            "data_dir": str(p.data_dir),
            "papers_collection": p.papers_collection,
            "abstracts_collection": p.abstracts_collection,
            "wiki_collection": p.wiki_collection,
            "status": "ok" if (p.exists() and db_ok) else "incomplete",
            "is_default": p.is_default,
        })
    # Running-process view — what the web reader itself is currently
    # using (may differ from .active-project if the server was started
    # with --project / SCIKNOW_PROJECT and that has since changed).
    running = get_active_project()
    return JSONResponse({
        "projects": out,
        "active_slug": active_slug,
        "running_slug": running.slug,
    })


@router.get("/api/projects/{slug}")
async def api_projects_show(slug: str):
    """Show details for a single project (counts + collection names)."""
    from sciknow.cli.project import _pg_database_exists
    from sciknow.core.project import Project, get_active_project, validate_slug
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    db_ok = _pg_database_exists(project.pg_database)
    payload = {
        "slug": project.slug,
        "root": str(project.root),
        "data_dir": str(project.data_dir),
        "data_dir_exists": project.data_dir.exists(),
        "pg_database": project.pg_database,
        "pg_database_exists": db_ok,
        "qdrant_prefix": project.qdrant_prefix,
        "papers_collection": project.papers_collection,
        "abstracts_collection": project.abstracts_collection,
        "wiki_collection": project.wiki_collection,
        "env_overlay_path": str(project.env_overlay_path),
        "env_overlay_exists": project.env_overlay_path.exists(),
        "is_default": project.is_default,
    }
    if db_ok:
        try:
            from sciknow.storage.db import get_session as _gs
            with _gs(db_name=project.pg_database) as session:
                payload["n_documents"] = session.execute(text("SELECT COUNT(*) FROM documents")).scalar() or 0
                payload["n_chunks"]    = session.execute(text("SELECT COUNT(*) FROM chunks")).scalar() or 0
                payload["n_books"]     = session.execute(text("SELECT COUNT(*) FROM books")).scalar() or 0
                payload["n_drafts"]    = session.execute(text("SELECT COUNT(*) FROM drafts")).scalar() or 0
        except Exception as exc:
            payload["counts_error"] = str(exc)
    return JSONResponse(payload)


@router.post("/api/projects/use")
async def api_projects_use(request: Request):
    """Set the active project (writes ``.active-project``).

    Does NOT hot-swap the running server's DB connection. Returns a
    ``restart_required`` flag so the frontend can tell the user to
    restart ``sciknow book serve`` in a book that exists in the target
    project.
    """
    from sciknow.core.project import (
        Project, get_active_project, list_projects, validate_slug,
        write_active_slug,
    )
    body = await request.json()
    slug = (body or {}).get("slug", "").strip()
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if slug != "default":
        candidate = Project(slug=slug, repo_root=get_active_project().repo_root)
        if not candidate.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Project {slug!r} not found. Available: "
                       + ", ".join(p.slug for p in list_projects()),
            )
    write_active_slug(slug)
    running = get_active_project().slug
    return JSONResponse({
        "ok": True,
        "active_slug": slug,
        "running_slug": running,
        "restart_required": slug != running,
        "message": (
            f"Active project set to {slug!r}. Restart `sciknow book serve` "
            f"in a book that belongs to {slug!r} to work on that corpus."
            if slug != running else
            f"Active project is already {slug!r}."
        ),
    })


@router.post("/api/projects/init")
async def api_projects_init(request: Request):
    """Create a new empty project. Runs synchronously (fast: ~3–5 s).

    Does NOT accept ``--from-existing`` — that's a one-shot migration
    that should be done from the CLI where the user has a shell for
    monitoring. The GUI only creates fresh empty projects.
    """
    from sciknow.cli.project import (
        _create_pg_database, _ensure_init_dirs, _init_qdrant_collections_for_project,
        _pg_database_exists, _run_alembic_upgrade,
    )
    from sciknow.core.project import (
        Project, get_active_project, validate_slug,
    )
    body = await request.json()
    slug = (body or {}).get("slug", "").strip()
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if slug == "default":
        raise HTTPException(status_code=400, detail="'default' is reserved for the legacy layout.")
    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    if project.root.exists():
        raise HTTPException(status_code=409, detail=f"Project directory already exists: {project.root}")
    if _pg_database_exists(project.pg_database):
        raise HTTPException(status_code=409, detail=f"PG database already exists: {project.pg_database}")

    try:
        _ensure_init_dirs(project)
        _create_pg_database(project.pg_database)
        _run_alembic_upgrade(project.pg_database)
        _init_qdrant_collections_for_project(project)
    except Exception as exc:
        logger.exception("project init failed")
        raise HTTPException(status_code=500, detail=f"init failed: {exc}")
    return JSONResponse({
        "ok": True,
        "slug": slug,
        "pg_database": project.pg_database,
        "papers_collection": project.papers_collection,
    })


@router.get("/api/projects/{slug}/venues")
async def api_project_venues_get(slug: str):
    """Phase 54.6.112 — return the project's venue_config.json as JSON."""
    from sciknow.core.project import Project, get_active_project
    from sciknow.core import venue_config as _vc
    if slug == "active":
        project = get_active_project()
    else:
        project = Project(slug=slug, repo_root=get_active_project().repo_root)
    cfg = _vc.load(project.root)
    return JSONResponse({
        "slug": project.slug,
        "path": str(_vc.path_for(project.root)),
        "blocklist": cfg.blocklist,
        "allowlist": cfg.allowlist,
    })


@router.post("/api/projects/{slug}/venues")
async def api_project_venues_post(slug: str, request: Request):
    """Mutate the venue block/allow lists. Body: ``{action, kind, pattern}``.

    action ∈ {add, remove}. kind ∈ {block, allow}.
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    action = body.get("action")
    kind = body.get("kind")
    pattern = (body.get("pattern") or "").strip()
    if action not in ("add", "remove") or kind not in ("block", "allow") or not pattern:
        raise HTTPException(400, "Body must be {action:'add'|'remove', kind:'block'|'allow', pattern:'...'}")

    from sciknow.core.project import Project, get_active_project
    from sciknow.core import venue_config as _vc
    if slug == "active":
        project = get_active_project()
    else:
        project = Project(slug=slug, repo_root=get_active_project().repo_root)

    if action == "add":
        _, changed = _vc.add_pattern(project.root, pattern, kind=kind)
    else:
        _, changed = _vc.remove_pattern(project.root, pattern, kind=kind)
    cfg = _vc.load(project.root)
    return JSONResponse({
        "slug": project.slug,
        "changed": changed,
        "blocklist": cfg.blocklist,
        "allowlist": cfg.allowlist,
    })


@router.post("/api/projects/destroy")
async def api_projects_destroy(request: Request):
    """Drop a project's DB + collections + data dir. Requires ``confirm``
    matching the slug to prevent accidental clicks."""
    import shutil as _shutil
    from sciknow.cli.project import (
        _delete_qdrant_collections_for_project, _drop_pg_database,
    )
    from sciknow.core.project import (
        Project, _active_project_file, get_active_project,
        read_active_slug_from_file, validate_slug,
    )
    body = await request.json()
    slug    = (body or {}).get("slug", "").strip()
    confirm = (body or {}).get("confirm", "").strip()
    try:
        validate_slug(slug)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if slug == "default":
        raise HTTPException(status_code=400, detail="Refusing to destroy the legacy 'default' project.")
    if confirm != slug:
        raise HTTPException(status_code=400, detail="Confirmation slug does not match.")
    if slug == get_active_project().slug:
        raise HTTPException(
            status_code=409,
            detail="Cannot destroy the project the web reader is currently running against. "
                   "Restart `sciknow book serve` in a different project first.",
        )

    project = Project(slug=slug, repo_root=get_active_project().repo_root)
    errors: list[str] = []
    try:
        _drop_pg_database(project.pg_database)
    except Exception as exc:
        errors.append(f"pg drop: {exc}")
    try:
        n_dropped = _delete_qdrant_collections_for_project(project)
    except Exception as exc:
        n_dropped = 0
        errors.append(f"qdrant: {exc}")
    try:
        if project.root.exists():
            _shutil.rmtree(project.root)
    except Exception as exc:
        errors.append(f"data dir: {exc}")
    # If destroying the on-disk active project, clear .active-project
    active = read_active_slug_from_file()
    if active == slug:
        f = _active_project_file()
        if f.exists():
            f.unlink()
    return JSONResponse({"ok": not errors, "errors": errors, "qdrant_dropped": n_dropped})
