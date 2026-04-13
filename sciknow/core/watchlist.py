"""Repo watchlist — track upstream research repos for new commits / releases.

Phase 45 — the project already borrows ideas from karpathy/autoresearch,
SakanaAI/AI-Scientist, aiming-lab/AutoResearchClaw, and analemma FARS.
Those are all actively evolving. Rather than remembering to check them
by hand, this module keeps a small list of URLs and periodically fetches
the GitHub API to surface what's changed since the last check.

Minimal design — no dependencies beyond httpx (already used for
Crossref). Watchlist state lives in a small JSONL file per project
root so different projects can track different repos if the user
decides to. The CLI in ``sciknow/cli/watch.py`` is the entry point.

Data layout
-----------

    {repo_root}/data/watchlist.jsonl

Each line = one JSON object with the schema below. Append-only writes
for history; a single-row-per-repo index is rebuilt from the log on
read.

    {
      "ts":              "2026-04-13T16:45:12Z",     # when this entry was written
      "kind":            "add" | "check" | "note" | "remove",
      "owner":           "karpathy",
      "repo":            "autoresearch",
      "url":             "https://github.com/karpathy/autoresearch",
      "note":            "why we care",             # free-form, optional
      "last_default_sha":"abc123...",              # HEAD at last check
      "last_pushed_at":  "2026-03-26T...",          # from the API
      "stars":           71500,
      "topics":          ["autonomous-research"],
      "new_commits":      12                        # since last_default_sha
    }

There is deliberately no background scheduler — periodic checking is
the user's job (a cron line in README, or their shell history). This
module only provides the *data plane* for "what changed"; telling the
user lives in the CLI layer.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("sciknow.watchlist")


# ── Known seed repos (sciknow has already learned from these) ─────────
#
# These are pre-populated into the watchlist on first use. Users can
# add/remove freely; this list is read once, never written.

SEED_REPOS: list[dict] = [
    {
        "url": "https://github.com/karpathy/autoresearch",
        "note": "Karpathy's agent harness for single-GPU LLM pretraining. "
                "Source of sciknow's 'program.md' + wall-clock-budget ideas.",
    },
    {
        "url": "https://github.com/SakanaAI/AI-Scientist",
        "note": "End-to-end automated ML-paper pipeline (idea → experiment → "
                "write → review). Source of two-stage citation loop + "
                "NeurIPS-rubric ensemble review ideas.",
    },
    {
        "url": "https://github.com/aiming-lab/AutoResearchClaw",
        "note": "23-stage pipeline with HITL gates + external citation "
                "verification (arXiv/Crossref/OpenAlex) + cross-run "
                "'MetaClaw' lesson store.",
    },
    {
        "url": "https://github.com/open-fars/openfars",
        "note": "Open re-implementation of analemma FARS. Reference for "
                "per-project filesystem reference cache + typed task_plan.json "
                "contract.",
    },
    {
        "url": "https://github.com/WecoAI/aideml",
        "note": "AIDE — the tree-search ML-engineering agent that AI-Scientist "
                "v2's BFTS is derived from. Relevant if we ever want to add "
                "agentic tree search to autowrite.",
    },
]


# ── Data types ─────────────────────────────────────────────────────────


@dataclass
class WatchedRepo:
    owner: str
    repo: str
    url: str
    note: str = ""
    last_default_sha: str | None = None
    last_pushed_at: str | None = None
    stars: int = 0
    topics: list[str] = field(default_factory=list)
    new_commits_since_last_check: int = 0
    last_checked_at: str | None = None

    @property
    def key(self) -> str:
        return f"{self.owner}/{self.repo}"

    def to_row(self) -> dict:
        return {
            "key": self.key,
            "owner": self.owner,
            "repo": self.repo,
            "url": self.url,
            "note": self.note,
            "last_default_sha": self.last_default_sha,
            "last_pushed_at": self.last_pushed_at,
            "stars": self.stars,
            "topics": self.topics,
            "new_commits": self.new_commits_since_last_check,
            "last_checked_at": self.last_checked_at,
        }


# ── URL parsing ────────────────────────────────────────────────────────


_GITHUB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[A-Za-z0-9_.\-]+)/"
    r"(?P<repo>[A-Za-z0-9_.\-]+?)(?:\.git)?/?$"
)


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a github URL. Raises ValueError on bad input."""
    m = _GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"not a github.com/owner/repo URL: {url!r}. "
            "Non-GitHub watchlist entries aren't supported yet."
        )
    return m.group("owner"), m.group("repo")


# ── Persistence (append-only JSONL log + replay-on-read index) ────────


def _log_path() -> Path:
    """Where the watchlist log lives for the active project."""
    from sciknow.config import settings
    p = Path(settings.data_dir) / "watchlist.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(entry: dict) -> None:
    entry = {"ts": _ts(), **entry}
    with _log_path().open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _replay() -> dict[str, WatchedRepo]:
    """Rebuild the current index from the append-only log.

    The log is cheap to scan (one entry per watch operation) and easy
    to audit. We never rewrite old entries — retracting a repo is a
    ``remove`` event, not a delete.
    """
    index: dict[str, WatchedRepo] = {}
    path = _log_path()
    if not path.exists():
        return index
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            kind = ev.get("kind") or ""
            owner = ev.get("owner") or ""
            repo  = ev.get("repo")  or ""
            if not owner or not repo:
                continue
            key = f"{owner}/{repo}"
            if kind == "add":
                index[key] = WatchedRepo(
                    owner=owner, repo=repo,
                    url=ev.get("url") or f"https://github.com/{key}",
                    note=ev.get("note") or "",
                )
            elif kind == "remove":
                index.pop(key, None)
            elif kind == "check" and key in index:
                r = index[key]
                r.last_default_sha = ev.get("last_default_sha") or r.last_default_sha
                r.last_pushed_at   = ev.get("last_pushed_at")   or r.last_pushed_at
                r.stars            = ev.get("stars", r.stars)
                r.topics           = ev.get("topics") or r.topics
                r.new_commits_since_last_check = ev.get("new_commits", 0)
                r.last_checked_at  = ev.get("ts")
            elif kind == "note" and key in index:
                index[key].note = ev.get("note", index[key].note)
    return index


def seed_if_empty() -> int:
    """Pre-populate the watchlist with SEED_REPOS on first use.

    Returns the number of repos added. No-op if the log already exists
    with any content. Safe to call at CLI startup.
    """
    index = _replay()
    if index:
        return 0
    added = 0
    for s in SEED_REPOS:
        try:
            owner, repo = parse_github_url(s["url"])
            _append({
                "kind": "add",
                "owner": owner, "repo": repo,
                "url": s["url"], "note": s.get("note", ""),
            })
            added += 1
        except Exception as exc:
            logger.debug("seed failed for %s: %s", s.get("url"), exc)
    return added


# ── Public API ────────────────────────────────────────────────────────


def list_watched() -> list[WatchedRepo]:
    """Current watchlist, sorted by (last_pushed_at desc, key)."""
    index = _replay()
    return sorted(
        index.values(),
        key=lambda r: (r.last_pushed_at or "", r.key),
        reverse=True,
    )


def add(url: str, note: str = "") -> WatchedRepo:
    """Add a repo. Idempotent — re-adding updates the note."""
    owner, repo = parse_github_url(url)
    _append({
        "kind": "add",
        "owner": owner, "repo": repo,
        "url": f"https://github.com/{owner}/{repo}",
        "note": note,
    })
    return WatchedRepo(owner=owner, repo=repo,
                       url=f"https://github.com/{owner}/{repo}", note=note)


def remove(url_or_key: str) -> bool:
    """Remove a repo by URL or owner/repo key. Returns True if it was present."""
    if "/" in url_or_key and "github.com" not in url_or_key:
        owner, repo = url_or_key.split("/", 1)
    else:
        owner, repo = parse_github_url(url_or_key)
    index = _replay()
    key = f"{owner}/{repo}"
    if key not in index:
        return False
    _append({"kind": "remove", "owner": owner, "repo": repo})
    return True


def check(url_or_key: str, *, github_token: str | None = None) -> WatchedRepo:
    """Hit the GitHub API and record the current HEAD sha + activity.

    Uses the anonymous v3 REST API (60 requests/hour) by default. Pass
    ``github_token`` or set ``GITHUB_TOKEN`` in the environment to raise
    the rate limit to 5000/hour. No authentication beyond that.

    Writes a ``check`` event to the log; the ``new_commits_since_last_check``
    field is set to the number of commits between the previously stored
    ``last_default_sha`` and the current HEAD (or 0 on first check).
    """
    import httpx
    if "/" in url_or_key and "github.com" not in url_or_key:
        owner, repo = url_or_key.split("/", 1)
    else:
        owner, repo = parse_github_url(url_or_key)

    token = github_token or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = f"https://api.github.com/repos/{owner}/{repo}"

    with httpx.Client(headers=headers, timeout=10.0) as client:
        r = client.get(base)
        r.raise_for_status()
        meta = r.json()
        default_branch = meta.get("default_branch") or "main"
        r2 = client.get(f"{base}/commits/{default_branch}")
        r2.raise_for_status()
        head = r2.json()
        head_sha = head.get("sha")

        # Count commits since last known sha — the compare endpoint
        # returns {"total_commits": N} directly when a baseline exists.
        index = _replay()
        key = f"{owner}/{repo}"
        prior_sha = index[key].last_default_sha if key in index else None
        new_commits = 0
        if prior_sha and prior_sha != head_sha:
            rc = client.get(f"{base}/compare/{prior_sha}...{head_sha}")
            if rc.status_code == 200:
                new_commits = (rc.json() or {}).get("total_commits", 0)

    ev = {
        "kind": "check",
        "owner": owner, "repo": repo,
        "last_default_sha": head_sha,
        "last_pushed_at": meta.get("pushed_at"),
        "stars": meta.get("stargazers_count", 0),
        "topics": meta.get("topics") or [],
        "new_commits": new_commits,
    }
    _append(ev)

    return WatchedRepo(
        owner=owner, repo=repo,
        url=meta.get("html_url") or f"https://github.com/{owner}/{repo}",
        note=(index[key].note if key in index else ""),
        last_default_sha=head_sha,
        last_pushed_at=meta.get("pushed_at"),
        stars=meta.get("stargazers_count", 0),
        topics=meta.get("topics") or [],
        new_commits_since_last_check=new_commits,
        last_checked_at=_ts(),
    )


def check_all(*, github_token: str | None = None) -> Iterable[WatchedRepo]:
    """Yield a WatchedRepo per watched repo after fetching. Best-effort —
    errors on individual repos are logged and skipped."""
    for r in list_watched():
        try:
            yield check(r.key, github_token=github_token)
        except Exception as exc:
            logger.warning("check %s failed: %s", r.key, exc)
