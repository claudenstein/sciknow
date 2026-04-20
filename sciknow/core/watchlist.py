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
from datetime import datetime, timedelta, timezone
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
    # Phase 46 audit surfaced these three as more substantive than FARS.
    {
        "url": "https://github.com/zhu-minjun/Researcher",
        "note": "CycleResearcher (arXiv:2411.00816, ICLR 2025). Ships a "
                "fine-tuned CycleReviewer judge model (MAE 26.89% below human "
                "reviewers on OpenReview triples). Main porting target when "
                "DGX Spark arrives — swap in as the autowrite scorer.",
    },
    {
        "url": "https://github.com/ResearAI/DeepScientist",
        "note": "DeepScientist (arXiv:2509.26603, ICLR 2026 top-10). "
                "Findings Memory + Bayesian optimization + hierarchical "
                "fidelity tiers (cheap → expensive promotion gates). "
                "Candidate inspiration for a `book autowrite --tier` flag.",
    },
]


# Phase 46.G — HuggingFace benchmark leaderboards.
#
# Each entry is an HF dataset slug that acts as a benchmark. The check
# path hits /api/datasets/<slug> for lastModified + sha (which moves
# whenever a new model is evaluated or the README is updated) and
# best-effort-parses the README for a top-N model ranking. HF does
# not expose the leaderboard via a structured JSON API, so
# README-table scraping is the reliable signal.
SEED_BENCHMARKS: list[dict] = [
    {
        "dataset": "allenai/olmOCR-bench",
        "note": "Third-party OCR benchmark from AllenAI, independent of "
                "OmniDocBench's self-scoring. sciknow's MinerU 2.5 is NOT "
                "on this leaderboard (only MinerU 1.3.10 at 61.5). Top "
                "today: Infinity-Parser2-Pro 86.7, Chandra-2 85.9. See "
                "docs/INGESTION.md for the audit.",
    },
    {
        "dataset": "opendatalab/OmniDocBench",
        "note": "OpenDataLab (MinerU team) — self-scored numbers. "
                "MinerU 2.5 pipeline = 86.2 (v1.5), MinerU 2.5-Pro VLM = "
                "95.69 (v1.6). Trust with caveats — author-run — but "
                "still the primary scientific-paper parsing benchmark.",
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


@dataclass
class WatchedBenchmark:
    """Phase 46.G — an HF benchmark leaderboard watched for regime changes.

    Identified by the HF dataset slug (``owner/dataset``). ``top_models``
    is a best-effort parse of the README's leaderboard table — each
    entry is ``{"rank": int, "name": str, "score": float | None}``.
    ``last_modified`` + ``sha`` come straight from the HF API; either
    changing means *something* moved, even if the README parse fails.
    """
    dataset: str
    url: str
    note: str = ""
    last_modified: str | None = None
    sha: str | None = None
    likes: int = 0
    downloads: int = 0
    top_models: list[dict] = field(default_factory=list)
    prev_top_models: list[dict] = field(default_factory=list)
    last_checked_at: str | None = None

    @property
    def key(self) -> str:
        return self.dataset

    @property
    def top_model_name(self) -> str | None:
        return (self.top_models[0].get("name")
                if self.top_models else None)

    @property
    def top_changed_since_last_check(self) -> bool:
        """True iff the #1 model differs between the last two checks.

        Regime-change signal — a new entrant just took the top spot.
        False when prev_top_models is empty (first check) or when the
        top name is unchanged.
        """
        if not self.top_models or not self.prev_top_models:
            return False
        return (self.top_models[0].get("name")
                != self.prev_top_models[0].get("name"))


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


# ── HF dataset slug parsing + README leaderboard scraping (Phase 46.G) ──

_HF_SLUG_RE = re.compile(
    r"(?:https?://huggingface\.co/datasets/)?"
    r"(?P<slug>[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)/?$"
)


def parse_hf_dataset_slug(url_or_slug: str) -> str:
    """Accept ``owner/dataset`` OR ``https://huggingface.co/datasets/owner/dataset``.

    Returns the normalized ``owner/dataset`` slug. Raises ValueError
    on malformed input.
    """
    m = _HF_SLUG_RE.match((url_or_slug or "").strip())
    if not m:
        raise ValueError(
            f"not a valid HF dataset slug or URL: {url_or_slug!r}. "
            "Expected 'owner/dataset' or "
            "'https://huggingface.co/datasets/owner/dataset'."
        )
    return m.group("slug")


# Match a pipe-delimited Markdown table row in a README: looks for any
# row with a cell that resembles a model name + a numeric score cell.
# We deliberately keep this lenient — leaderboard table shapes vary
# across READMEs — and validate the extracted rows downstream.
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|\s*(.+?)\s*\|\s*$", re.MULTILINE)
_MODEL_NAME_HINT = re.compile(
    r"[A-Za-z][A-Za-z0-9._\-]*(?:/[A-Za-z0-9._\-]+)?"
)


def _parse_readme_leaderboard(readme: str, *, top_n: int = 5) -> list[dict]:
    """Best-effort extraction of a top-N ranked model list from a README.

    Heuristic: locate the LAST pipe-delimited table (leaderboards tend
    to live near the bottom of HF dataset READMEs) whose header row
    mentions ``model`` AND the first cell of its data rows looks like
    a model name. Score = the LAST numeric column on the same row, on
    the assumption that most OCR benchmarks put the "Overall" score on
    the right. Returns ``[{"rank", "name", "score"}]`` with up to
    ``top_n`` entries, sorted by score desc.

    Fails gracefully: any parsing surprise produces an empty list, not
    an exception. The dataset-info sha change is still a useful signal
    on its own when this returns nothing.
    """
    if not readme or "|" not in readme:
        return []

    # Split into candidate tables (contiguous runs of 2+ pipe-rows)
    lines = readme.splitlines()
    tables: list[list[str]] = []
    cur: list[str] = []
    for ln in lines:
        if ln.lstrip().startswith("|"):
            cur.append(ln)
        else:
            if len(cur) >= 3:   # header + separator + ≥1 data row
                tables.append(cur)
            cur = []
    if len(cur) >= 3:
        tables.append(cur)
    if not tables:
        return []

    # Prefer the last table whose header mentions 'model' (case-insens).
    leaderboard_table: list[str] | None = None
    for tbl in reversed(tables):
        header = tbl[0].lower()
        if "model" in header or "method" in header or "system" in header:
            leaderboard_table = tbl
            break
    if leaderboard_table is None:
        leaderboard_table = tables[-1]

    # Parse rows. Skip separator rows (|---|---|).
    parsed: list[tuple[str, float]] = []
    for ln in leaderboard_table[2:]:      # skip header + separator
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if not cells or all(c.startswith(":") or c in {"-", "---", ""}
                            or set(c) <= set("-: ") for c in cells):
            continue
        name_cell = cells[0]
        # Strip markdown emphasis (**X**, *X*, `X`, [X](url))
        name = re.sub(r"\*\*|\*|`", "", name_cell)
        m_link = re.match(r"\[([^\]]+)\]\([^)]+\)", name)
        if m_link:
            name = m_link.group(1)
        name = name.strip()
        if not name or not _MODEL_NAME_HINT.search(name):
            continue
        # Score: walk right-to-left for the last numeric cell.
        score: float | None = None
        for c in reversed(cells[1:]):
            # Strip confidence intervals like "76.3 ± 1.1"
            c_clean = c.split("±")[0].strip().replace(",", "")
            c_clean = re.sub(r"\*\*|\*|`", "", c_clean)
            try:
                score = float(c_clean)
                break
            except ValueError:
                continue
        if score is None:
            continue
        parsed.append((name, score))

    if not parsed:
        return []
    parsed.sort(key=lambda x: x[1], reverse=True)
    return [
        {"rank": i + 1, "name": name, "score": score}
        for i, (name, score) in enumerate(parsed[:top_n])
    ]


# ── Persistence (append-only JSONL log + replay-on-read index) ────────


def _log_path() -> Path:
    """Where the watchlist log lives for the active project."""
    from sciknow.config import settings
    p = Path(settings.data_dir) / "watchlist.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Rate limit ─────────────────────────────────────────────────────────
#
# GitHub's anonymous REST API allows 60 requests/hour — easy to burn
# through if a script or shell loop hits `watch check` repeatedly. Worse,
# daily-level insight (what got pushed overnight) is all we need for
# a research-watch use case. The 24h guard below is enforced on
# ``check()`` by default; callers pass ``force=True`` to override.

CHECK_COOLDOWN_HOURS = 24.0


def _hours_since(iso_ts: str | None) -> float | None:
    """Hours since the given ISO-8601 UTC timestamp, or None if unparseable."""
    if not iso_ts:
        return None
    try:
        # Accept both "…Z" and "…+00:00"
        s = iso_ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    return delta.total_seconds() / 3600.0


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
            # Phase 46.G — benchmark events share the same log; branch
            # by event kind. Benchmark events begin with "bench_".
            if kind.startswith("bench_"):
                continue   # handled by _replay_benchmarks

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


def _replay_benchmarks() -> dict[str, WatchedBenchmark]:
    """Phase 46.G — rebuild the benchmark index from the same log."""
    index: dict[str, WatchedBenchmark] = {}
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
            if not kind.startswith("bench_"):
                continue
            slug = ev.get("dataset") or ""
            if not slug:
                continue
            if kind == "bench_add":
                index[slug] = WatchedBenchmark(
                    dataset=slug,
                    url=ev.get("url") or f"https://huggingface.co/datasets/{slug}",
                    note=ev.get("note") or "",
                )
            elif kind == "bench_remove":
                index.pop(slug, None)
            elif kind == "bench_check" and slug in index:
                b = index[slug]
                # Preserve last-known top_models so the next check can
                # compare for regime changes.
                b.prev_top_models  = list(b.top_models) if b.top_models else list(b.prev_top_models)
                b.last_modified    = ev.get("last_modified")    or b.last_modified
                b.sha              = ev.get("sha")              or b.sha
                b.likes            = int(ev.get("likes", b.likes) or 0)
                b.downloads        = int(ev.get("downloads", b.downloads) or 0)
                if ev.get("top_models") is not None:
                    b.top_models   = ev.get("top_models") or []
                b.last_checked_at  = ev.get("ts")
            elif kind == "bench_note" and slug in index:
                index[slug].note = ev.get("note", index[slug].note)
    return index


def seed_if_empty() -> int:
    """Pre-populate the watchlist with SEED_REPOS on first use.

    Returns the number of repos + benchmarks added. No-op if the log
    already exists with any content. Safe to call at CLI startup.
    """
    index = _replay()
    bench_index = _replay_benchmarks()
    if index or bench_index:
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
    for b in SEED_BENCHMARKS:
        try:
            slug = parse_hf_dataset_slug(b["dataset"])
            _append({
                "kind": "bench_add",
                "dataset": slug,
                "url": f"https://huggingface.co/datasets/{slug}",
                "note": b.get("note", ""),
            })
            added += 1
        except Exception as exc:
            logger.debug("seed bench failed for %s: %s", b.get("dataset"), exc)
    return added


def seed_benchmarks_if_missing() -> int:
    """Phase 46.G — add SEED_BENCHMARKS to the existing log if they're
    not already present. Safe to call repeatedly. Useful for existing
    sciknow installs whose log predates the benchmark feature.
    """
    bench_index = _replay_benchmarks()
    added = 0
    for b in SEED_BENCHMARKS:
        try:
            slug = parse_hf_dataset_slug(b["dataset"])
        except Exception:
            continue
        if slug in bench_index:
            continue
        _append({
            "kind": "bench_add",
            "dataset": slug,
            "url": f"https://huggingface.co/datasets/{slug}",
            "note": b.get("note", ""),
        })
        added += 1
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


class RateLimited(Exception):
    """Raised when ``check`` is called inside the cooldown window and
    the caller didn't pass ``force=True``.

    The exception carries the ``hours_remaining`` until the next check
    is allowed, and the cached WatchedRepo from the last successful
    check, so callers can show stale-but-useful data without a new
    API call.
    """
    def __init__(self, hours_remaining: float, cached: WatchedRepo):
        self.hours_remaining = hours_remaining
        self.cached = cached
        super().__init__(
            f"checked {round(CHECK_COOLDOWN_HOURS - hours_remaining, 1)}h ago; "
            f"next check in {hours_remaining:.1f}h. Pass force=True to override."
        )


def check(
    url_or_key: str,
    *,
    github_token: str | None = None,
    force: bool = False,
    cooldown_hours: float | None = None,
) -> WatchedRepo:
    """Hit the GitHub API and record the current HEAD sha + activity.

    Uses the anonymous v3 REST API (60 requests/hour) by default. Pass
    ``github_token`` or set ``GITHUB_TOKEN`` in the environment to raise
    the rate limit to 5000/hour. No authentication beyond that.

    **Daily rate-limit**: by default a repo is re-checked only if its
    most recent ``check`` log entry is older than ``CHECK_COOLDOWN_HOURS``
    (default 24h). Pass ``force=True`` to override, or tighten/loosen
    via ``cooldown_hours``. This is a defensive guard against script
    loops burning through the anon-API budget; the upstream repos we
    watch don't ship day-over-day, so hourly checks wouldn't tell us
    anything a daily check doesn't.

    Raises ``RateLimited`` if skipped; the exception carries the cached
    ``WatchedRepo`` so callers can render stale-but-useful data without
    a new API call.

    Writes a ``check`` event to the log; the ``new_commits_since_last_check``
    field is set to the number of commits between the previously stored
    ``last_default_sha`` and the current HEAD (or 0 on first check).
    """
    import httpx
    if "/" in url_or_key and "github.com" not in url_or_key:
        owner, repo = url_or_key.split("/", 1)
    else:
        owner, repo = parse_github_url(url_or_key)

    # Cooldown guard — only when we have a prior successful check.
    index = _replay()
    key = f"{owner}/{repo}"
    cached = index.get(key)
    cooldown = cooldown_hours if cooldown_hours is not None else CHECK_COOLDOWN_HOURS
    if not force and cached is not None and cached.last_checked_at:
        hrs = _hours_since(cached.last_checked_at)
        if hrs is not None and hrs < cooldown:
            raise RateLimited(
                hours_remaining=round(cooldown - hrs, 2),
                cached=cached,
            )

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
        prior_sha = cached.last_default_sha if cached is not None else None
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
        note=(cached.note if cached is not None else ""),
        last_default_sha=head_sha,
        last_pushed_at=meta.get("pushed_at"),
        stars=meta.get("stargazers_count", 0),
        topics=meta.get("topics") or [],
        new_commits_since_last_check=new_commits,
        last_checked_at=_ts(),
    )


def check_all(
    *,
    github_token: str | None = None,
    force: bool = False,
    cooldown_hours: float | None = None,
) -> Iterable[tuple[WatchedRepo, str]]:
    """Yield ``(repo, status)`` per watched repo.

    ``status`` is one of ``"checked"`` (hit the API, fresh data),
    ``"cached"`` (inside the cooldown, returned last-known data), or
    ``"error"`` (network / HTTP failure).

    The signature changed in Phase 46.D to carry the status explicitly;
    the CLI surface uses it to distinguish "just checked" from "within
    24h window". Errors on individual repos are still best-effort:
    they log a warning and yield ``(cached_or_empty, "error")``.
    """
    for r in list_watched():
        try:
            got = check(r.key, github_token=github_token,
                        force=force, cooldown_hours=cooldown_hours)
            yield got, "checked"
        except RateLimited as rl:
            yield rl.cached, "cached"
        except Exception as exc:
            logger.warning("check %s failed: %s", r.key, exc)
            yield r, "error"


# ── Benchmark public API (Phase 46.G) ─────────────────────────────────


def list_watched_benchmarks() -> list[WatchedBenchmark]:
    """Current benchmark watchlist, sorted by (last_modified desc, key)."""
    index = _replay_benchmarks()
    return sorted(
        index.values(),
        key=lambda b: (b.last_modified or "", b.key),
        reverse=True,
    )


def add_benchmark(dataset: str, note: str = "") -> WatchedBenchmark:
    """Add a benchmark (HF dataset slug). Idempotent — re-adding updates the note."""
    slug = parse_hf_dataset_slug(dataset)
    _append({
        "kind": "bench_add",
        "dataset": slug,
        "url":  f"https://huggingface.co/datasets/{slug}",
        "note": note,
    })
    return WatchedBenchmark(
        dataset=slug,
        url=f"https://huggingface.co/datasets/{slug}",
        note=note,
    )


def remove_benchmark(dataset: str) -> bool:
    """Remove a benchmark. Returns True if it was present."""
    try:
        slug = parse_hf_dataset_slug(dataset)
    except ValueError:
        return False
    if slug not in _replay_benchmarks():
        return False
    _append({"kind": "bench_remove", "dataset": slug})
    return True


def check_benchmark(
    dataset: str,
    *,
    force: bool = False,
    cooldown_hours: float | None = None,
) -> WatchedBenchmark:
    """Hit the HF dataset API + fetch README; record current state.

    Anonymous HF API has no strict published cap (~soft limits at
    several hundred/min), so the same daily cooldown applies for
    politeness + to keep the watchlist cheap.

    Writes a ``bench_check`` event with the current ``lastModified``,
    ``sha``, ``likes``, ``downloads``, and the parsed top-5 ranked
    model list (empty if the README shape didn't match). Raises
    ``RateLimited`` if called inside the cooldown window.
    """
    import httpx
    slug = parse_hf_dataset_slug(dataset)

    index  = _replay_benchmarks()
    cached = index.get(slug)
    cooldown = cooldown_hours if cooldown_hours is not None else CHECK_COOLDOWN_HOURS
    if not force and cached is not None and cached.last_checked_at:
        hrs = _hours_since(cached.last_checked_at)
        if hrs is not None and hrs < cooldown:
            raise RateLimited(
                hours_remaining=round(cooldown - hrs, 2),
                cached=cached,  # type: ignore[arg-type]
            )

    ua = "sciknow/0.1 (+https://github.com/claudenstein/sciknow)"
    with httpx.Client(timeout=15.0, headers={"User-Agent": ua}) as client:
        meta_r = client.get(f"https://huggingface.co/api/datasets/{slug}")
        meta_r.raise_for_status()
        meta = meta_r.json()

        # Fetch the README — best-effort; if it 404s the sha-delta
        # remains the signal.
        readme_text = ""
        try:
            readme_r = client.get(
                f"https://huggingface.co/datasets/{slug}/raw/main/README.md"
            )
            if readme_r.status_code == 200:
                readme_text = readme_r.text
        except Exception as exc:
            logger.debug("bench_check readme fetch failed: %s", exc)

    top_models = _parse_readme_leaderboard(readme_text, top_n=5)

    ev = {
        "kind":          "bench_check",
        "dataset":       slug,
        "last_modified": meta.get("lastModified"),
        "sha":           meta.get("sha"),
        "likes":         meta.get("likes", 0),
        "downloads":     meta.get("downloads", 0),
        "top_models":    top_models,
    }
    _append(ev)

    prev = list(cached.top_models) if cached and cached.top_models else []
    return WatchedBenchmark(
        dataset=slug,
        url=f"https://huggingface.co/datasets/{slug}",
        note=(cached.note if cached is not None else ""),
        last_modified=meta.get("lastModified"),
        sha=meta.get("sha"),
        likes=int(meta.get("likes") or 0),
        downloads=int(meta.get("downloads") or 0),
        top_models=top_models,
        prev_top_models=prev,
        last_checked_at=_ts(),
    )


def check_all_benchmarks(
    *,
    force: bool = False,
    cooldown_hours: float | None = None,
) -> Iterable[tuple[WatchedBenchmark, str]]:
    """Mirror of ``check_all`` for benchmarks; yields ``(bench, status)``."""
    for b in list_watched_benchmarks():
        try:
            got = check_benchmark(
                b.dataset, force=force, cooldown_hours=cooldown_hours,
            )
            yield got, "checked"
        except RateLimited as rl:
            yield rl.cached, "cached"   # type: ignore[misc]
        except Exception as exc:
            logger.warning("bench_check %s failed: %s", b.key, exc)
            yield b, "error"


# ══════════════════════════════════════════════════════════════════════
# Phase 54.6.137 — velocity queries (scheduled OpenAlex semantic watch).
#
# A third watched-entity kind, sharing the same append-only log + replay
# pattern as repos and benchmarks. Purpose: surface *new* papers in the
# last N days that match a stored semantic query, ranked by citation
# velocity, so researchers don't forget to re-expand their corpus when
# a topic starts moving.
# ══════════════════════════════════════════════════════════════════════


@dataclass
class WatchedVelocityQuery:
    """A stored OpenAlex semantic query watched for new high-velocity papers.

    ``query`` is the primary key (normalised: trimmed + case preserved
    since OpenAlex's ``search`` is case-insensitive, but we keep the
    original casing for display). ``window_days`` controls how far back
    each ``check`` looks; ``top_k`` caps the number of papers surfaced
    per check.

    Per-check output lives in the log; the dataclass only carries the
    *current* snapshot so the ``list`` command is cheap.
    """
    query: str
    note: str = ""
    window_days: int = 180
    top_k: int = 20
    # Rolling state
    last_checked_at: str | None = None
    # DOIs (lowercased) surfaced on the previous check, used to compute
    # the delta on the next check.
    last_seen_dois: list[str] = field(default_factory=list)
    # Top-K paper snapshots from the last check — each is
    # {doi, title, year, cited_by_count, velocity, authors (≤3)}.
    last_top_papers: list[dict] = field(default_factory=list)
    new_since_last_check: int = 0

    @property
    def key(self) -> str:
        """Stable primary key. Normalised for dedup: lowercased, whitespace-collapsed."""
        return " ".join(self.query.lower().split())


def _normalise_velocity_key(q: str) -> str:
    """Normalise a query string for velocity lookup: lowercase + collapse whitespace."""
    return " ".join((q or "").lower().split())


def _replay_velocity_queries() -> dict[str, WatchedVelocityQuery]:
    """Rebuild the velocity-query index from the shared append-only log."""
    index: dict[str, WatchedVelocityQuery] = {}
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
            if not kind.startswith("velo_"):
                continue
            q = ev.get("query") or ""
            if not q:
                continue
            key = _normalise_velocity_key(q)
            if kind == "velo_add":
                index[key] = WatchedVelocityQuery(
                    query=q,
                    note=ev.get("note") or "",
                    window_days=int(ev.get("window_days") or 180),
                    top_k=int(ev.get("top_k") or 20),
                )
            elif kind == "velo_remove":
                index.pop(key, None)
            elif kind == "velo_check" and key in index:
                w = index[key]
                w.last_checked_at     = ev.get("ts")
                w.last_seen_dois      = list(ev.get("seen_dois") or [])
                w.last_top_papers     = list(ev.get("top_papers") or [])
                w.new_since_last_check = int(ev.get("new_count") or 0)
            elif kind == "velo_note" and key in index:
                index[key].note = ev.get("note", index[key].note)
            elif kind == "velo_config" and key in index:
                w = index[key]
                if ev.get("window_days") is not None:
                    w.window_days = int(ev["window_days"])
                if ev.get("top_k") is not None:
                    w.top_k = int(ev["top_k"])
    return index


def list_watched_velocity_queries() -> list[WatchedVelocityQuery]:
    """Current velocity-query watchlist, sorted by last-check-time desc."""
    index = _replay_velocity_queries()
    return sorted(
        index.values(),
        key=lambda w: (w.last_checked_at or "", w.key),
        reverse=True,
    )


def add_velocity_query(
    query: str,
    *,
    note: str = "",
    window_days: int = 180,
    top_k: int = 20,
) -> WatchedVelocityQuery:
    """Register a velocity query. Idempotent — re-adding updates the note
    and the window/top_k parameters without resetting the rolling state."""
    q = (query or "").strip()
    if not q:
        raise ValueError("velocity query must not be empty")
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    key = _normalise_velocity_key(q)
    index = _replay_velocity_queries()
    existing = index.get(key)
    if existing is None:
        _append({
            "kind": "velo_add",
            "query": q,
            "note": note,
            "window_days": window_days,
            "top_k": top_k,
        })
        return WatchedVelocityQuery(
            query=q, note=note, window_days=window_days, top_k=top_k,
        )
    # Update path: emit a velo_config event to change params without
    # wiping the rolling state.
    _append({
        "kind": "velo_config",
        "query": existing.query,
        "window_days": window_days,
        "top_k": top_k,
    })
    if note and note != existing.note:
        _append({"kind": "velo_note", "query": existing.query, "note": note})
        existing.note = note
    existing.window_days = window_days
    existing.top_k = top_k
    return existing


def remove_velocity_query(query: str) -> bool:
    """Remove a velocity query. Returns True if it was present."""
    key = _normalise_velocity_key(query)
    index = _replay_velocity_queries()
    if key not in index:
        return False
    _append({"kind": "velo_remove", "query": index[key].query})
    return True


def _velocity_score(cited_by_count: int, publication_year: int | None) -> float:
    """Compute citations-per-active-year for a candidate paper.

    Denominator is clamped to 0.25 so brand-new papers with any citations
    get a large but finite velocity rather than exploding. This intentionally
    rewards recently-cited papers: that's the signal the watcher exists
    to surface.
    """
    if cited_by_count <= 0:
        return 0.0
    now_year = datetime.now(timezone.utc).year
    if publication_year is None or publication_year < 1900 or publication_year > now_year + 1:
        years = 1.0
    else:
        years = max(now_year - publication_year + 1, 0.25)  # include current year
    return float(cited_by_count) / years


def check_velocity_query(
    query: str,
    *,
    force: bool = False,
    cooldown_hours: float | None = None,
) -> WatchedVelocityQuery:
    """Hit OpenAlex for papers published in the last ``window_days`` matching
    the stored query, rank by citation velocity, diff against the previous
    check's DOI set, and record the result.

    The politeness cooldown (default 24h) applies — the anonymous OpenAlex
    API is lenient but cooperative usage is the documented norm.
    """
    import httpx
    key = _normalise_velocity_key(query)
    index = _replay_velocity_queries()
    cached = index.get(key)
    if cached is None:
        raise ValueError(
            f"velocity query not on the watchlist: {query!r}. "
            f"Run `sciknow watch add-velocity {query!r}` first."
        )

    cooldown = cooldown_hours if cooldown_hours is not None else CHECK_COOLDOWN_HOURS
    if not force and cached.last_checked_at:
        hrs = _hours_since(cached.last_checked_at)
        if hrs is not None and hrs < cooldown:
            raise RateLimited(
                hours_remaining=round(cooldown - hrs, 2),
                cached=cached,  # type: ignore[arg-type]
            )

    from_date = (
        datetime.now(timezone.utc) - timedelta(days=cached.window_days)
    ).strftime("%Y-%m-%d")

    # Re-use the project's polite-pool Crossref email — OpenAlex routes
    # requests with a `mailto` to a separate rate-limit pool. Mirrors
    # how `core.expand_ops` builds its OpenAlex params.
    from sciknow.config import settings as _settings
    mailto = (getattr(_settings, "crossref_email", "") or "").strip()
    params = {
        "search": cached.query,
        "filter": f"from_publication_date:{from_date}",
        "sort": "cited_by_count:desc",
        "per-page": "50",
    }
    if mailto:
        params["mailto"] = mailto

    ua = "sciknow/0.1 (+https://github.com/claudenstein/sciknow)"
    raw: list[dict] = []
    with httpx.Client(timeout=20.0, headers={"User-Agent": ua}) as client:
        r = client.get("https://api.openalex.org/works", params=params)
        r.raise_for_status()
        data = r.json()
        for w in (data.get("results") or []):
            doi = (w.get("doi") or "").lower().replace("https://doi.org/", "")
            title = (w.get("display_name") or "").strip()
            pub_year = w.get("publication_year")
            cited = int(w.get("cited_by_count") or 0)
            authors = []
            for a in (w.get("authorships") or [])[:3]:
                name = ((a.get("author") or {}).get("display_name") or "").strip()
                if name:
                    authors.append(name)
            raw.append({
                "doi": doi or None,
                "title": title,
                "year": pub_year,
                "cited_by_count": cited,
                "velocity": round(_velocity_score(cited, pub_year), 3),
                "authors": authors,
            })

    # Velocity re-rank; OpenAlex returned in cited_by_count order, we want
    # citations-per-active-year (favours recent-and-hot over old-and-ubiquitous).
    raw.sort(key=lambda p: p["velocity"], reverse=True)
    top = raw[: cached.top_k]

    # Diff against the previous check's DOI set.
    prev_set = {d for d in (cached.last_seen_dois or []) if d}
    current_dois = [p["doi"] for p in top if p.get("doi")]
    new_count = sum(1 for d in current_dois if d and d not in prev_set)

    ev = {
        "kind": "velo_check",
        "query": cached.query,
        "seen_dois": current_dois,
        "top_papers": top,
        "new_count": new_count,
    }
    _append(ev)

    return WatchedVelocityQuery(
        query=cached.query,
        note=cached.note,
        window_days=cached.window_days,
        top_k=cached.top_k,
        last_checked_at=_ts(),
        last_seen_dois=current_dois,
        last_top_papers=top,
        new_since_last_check=new_count,
    )


def check_all_velocity_queries(
    *,
    force: bool = False,
    cooldown_hours: float | None = None,
) -> Iterable[tuple[WatchedVelocityQuery, str]]:
    """Mirror of ``check_all`` for velocity queries; yields ``(query, status)``."""
    for w in list_watched_velocity_queries():
        try:
            got = check_velocity_query(
                w.query, force=force, cooldown_hours=cooldown_hours,
            )
            yield got, "checked"
        except RateLimited as rl:
            yield rl.cached, "cached"   # type: ignore[misc]
        except Exception as exc:
            logger.warning("velo_check %s failed: %s", w.key, exc)
            yield w, "error"
