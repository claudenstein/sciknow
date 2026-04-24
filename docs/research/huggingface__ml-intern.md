# huggingface/ml-intern — research memo

**URL**: <https://github.com/huggingface/ml-intern>
**Shared by user**: 2026-04-24
**Clone scratch**: `data/research/huggingface__ml-intern/`
**Stack**: Python + TypeScript/JavaScript, `pyproject.toml`, Dockerfile, `uv.lock`

## One-line summary

Hugging Face's autonomous "ML intern" CLI + backend agent that researches,
writes, and ships ML code using the HF ecosystem. Built on LiteLLM for
multi-provider LLM routing, shipped as `ml-intern` uv-tool-installable
CLI, with a companion FastAPI backend for session/quota management and a
React/Vite frontend (`frontend/`).

## Architecture at a glance

- **Tri-split repo**: `agent/` (smolagents-pattern LLM loop + tools),
  `backend/` (FastAPI for sessions, KPI scheduler, user quotas, routes),
  `frontend/` (Vite/React UI). Dockerfile covers the whole stack.
- **Agent core modules** (`agent/core/`):
  - `agent_loop.py` (1294 LOC) — the big submission loop.
  - `session.py` (355 LOC) — events + event types, async `send_event`
    pump.
  - `tools.py` (376 LOC) — `ToolRouter` dispatcher over ~12 tools.
  - `effort_probe.py` (229 LOC) — 1-token ping cascade to auto-discover
    per-model reasoning-effort support (max → xhigh → high → medium →
    low) so LiteLLM calls never fail on "unsupported effort" after the
    first probe. Result cached on the session.
  - `model_switcher.py` (228 LOC) — interactive `/model` command with
    routing policy suffixes (`:fastest` / `:cheapest` / `:preferred` /
    `:<provider>`).
  - `hf_router_catalog.py` — enumerates the HF router inference catalog.
  - `llm_params.py` (200 LOC) — param-resolution logic distinct from the
    effort probe.
  - `prompt_caching.py` — Anthropic-style cache-breakpoint management.
  - `redact.py` — secret redaction before telemetry export.
  - `telemetry.py` (289 LOC) — OpenTelemetry-shaped event export.
- **Tool inventory** (`agent/tools/`):
  - `plan_tool.py` — TODO list with `pending` / `in_progress` /
    `completed` statuses. Emits a `plan_update` session event. Matches
    1:1 the `TaskCreate` / `TaskUpdate` harness we already use.
  - `papers_tool.py` — HF Papers + arXiv HTML + Semantic Scholar with
    `trending` / `search` / `paper_details` / `read_paper` /
    `find_datasets` / `find_models` / `find_collections` /
    `citation_graph` / `snippet_search` / `recommend` operations. The
    `_s2_cache` dict at module scope implements a 500-entry shared
    response cache across sessions.
  - `docs_tools.py`, `github_find_examples.py`, `github_list_repos.py`,
    `github_read_file.py`, `hf_repo_files_tool.py`,
    `hf_repo_git_tool.py`, `jobs_tool.py`, `dataset_tools.py`,
    `private_hf_repo_tools.py`, `edit_utils.py`, `local_tools.py`.
- **System prompts** live in YAML (`agent/prompts/system_prompt*.yaml`)
  with v2 + v3 revisions checked in — version-controlled prompt edits
  the same way we version `drafts`.
- **REVIEW.md** at repo root is a PR-review rubric: P0 / P1 / P2
  severity, "rigor over speed" as the default bias, explicit anti-
  nitpicking guardrails. Written specifically to override default
  reviewer habits on this open-source repo.

## What sciknow could port / adapt

1. **Effort probe + cascade (`agent/core/effort_probe.py`).** Our
   Ollama-only writer doesn't feel this pain today, but the moment we
   support a second backend (ik_llama.cpp, vLLM, LiteLLM), a 1-token
   probe cascade that auto-discovers the supported reasoning-effort /
   max-tokens level avoids brittle per-model config tables. The pattern
   is small (229 LOC) and the cascade structure (`_EFFORT_CASCADE` dict
   keyed by user preference) transfers as-is.

2. **Papers tool + `_s2_cache` shared cache.** We already have
   Semantic Scholar layers in `sciknow/ingestion/enrich_sources.py` and
   `sciknow/ingestion/metadata.py`. Adopting the `papers_tool.py`
   caching pattern — a module-scope dict capped at 500 entries, keyed
   by `(path, params_tuple)` — would cut a lot of redundant Crossref /
   OpenAlex / S2 traffic in bulk `db enrich` passes without introducing
   a persistent cache layer. Currently our per-request retry logic
   re-hits the wire every time.

3. **Plan tool event pattern.** `agent/tools/plan_tool.py` emits
   `plan_update` events through `session.send_event` so any UI can
   subscribe. Our Corkboard + Plan modal already render plans from a
   DB, but a live-streaming "plan is mutating right now" flow is
   missing — adopting this would give us a real-time plan-editing view
   during autowrite runs.

4. **REVIEW.md severity rubric (P0 / P1 / P2 + explicit skip list).**
   sciknow's CLAUDE.md is directive but doesn't enumerate review
   severity. Lifting this rubric verbatim as `docs/REVIEW.md` and
   reference-linking it from `docs/PHASE_LOG.md` would standardize how
   we describe pre-merge gates in phase entries and monitor alerts.

5. **Session / telemetry split.** `agent/core/session.py` +
   `agent/core/telemetry.py` separate event generation from export
   with a `Event(event_type, data)` shape that mirrors our web-job SSE
   pattern. Their redaction pass (`agent/core/redact.py`) is a clean
   reference for masking API tokens before they hit the span export —
   something our `sciknow/observability/` pipeline doesn't do yet.

6. **Routing policy suffixes.** `ml-intern` lets users type
   `moonshotai/Kimi-K2.6:fastest` to pin a routing preference. We
   don't have multi-provider routing today, but when we add a second
   LLM backend, this suffix syntax is a lightweight way to expose it
   without a new flag.

## What NOT to port

- The LiteLLM dependency (`import litellm`) — they use it for multi-
  provider routing. We're Ollama-first; adding LiteLLM would pull in a
  heavy cross-provider abstraction layer we don't need.
- The HF-specific tools (`hf_repo_*`, `jobs_tool`,
  `private_hf_repo_*`) — our papers come from PDFs on disk + Crossref /
  OpenAlex, not HF.
- The React/Vite frontend — sciknow's f-string template UI is
  deliberately simpler; their stack has a multi-page router with auth.

## Relevance verdict

- [x] **medium** — useful reference, especially for the effort-probe
  cascade and the shared-cache pattern. No immediate port target, but
  both are filed for the inevitable second-LLM-backend phase.

## Next actions

- [ ] Apply the `papers_tool` caching pattern to `enrich_sources.py`'s
  S2 layer (cap-500 dict keyed by `(path, params_tuple)`) — ~20 LOC,
  cuts repeat Crossref calls during large `db enrich` runs.
- [ ] Write `docs/REVIEW.md` using the P0/P1/P2 skeleton from the
  `ml-intern` rubric.
- [ ] Watch for updates: `sciknow watch check huggingface/ml-intern`
  — the effort-probe module is evolving (v3 system prompt is a recent
  commit).
