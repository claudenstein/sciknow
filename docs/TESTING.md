# Testing Protocol

[&larr; Back to README](../README.md)

---

sciknow has a layered smoke-test harness, **not** a pytest suite. The CLI command is `sciknow test`. Every PR that touches synthesis, retrieval, ingestion, or storage should pass at least L1 before merging. Bigger PRs (a "Phase" feature drop, an infrastructure change, a model swap) should also pass L2 and ideally L3.

> **Performance + quality measurements** live in a separate harness: `sciknow bench` (`sciknow/testing/bench.py`, Phase 44). `test` answers "is the code still correct?"; `bench` answers "how fast is it and how good are the outputs?". See [BENCHMARKS.md](BENCHMARKS.md) for the bench layers and how to add a metric.

## Why a smoke harness instead of pytest?

This repo prioritises correctness over coverage metrics. The failure modes worth catching are:

1. A new prompt rule silently dropped from `WRITE_V2_SYSTEM`
2. A required kwarg removed from `autowrite_section_stream`
3. A CLI command quietly de-registered
4. An import that crashes the whole CLI on a fresh checkout
5. A Qdrant payload index that doesn't exist on the live collection
6. A response shape from Ollama that no longer parses

These are all checkable with ~10 lines of straight-line Python per test. A pytest setup would add structure (fixtures, parametrisation, plugins, conftest) that doesn't help you catch any of those failures faster — and the project explicitly avoids hidden complexity. The harness is a single file (`sciknow/testing/protocol.py`) that anyone can read end-to-end.

## The three layers

| Layer | Speed | Dependencies | What it covers | When to run |
|---|---|---|---|---|
| **L1 — Static** | seconds | none | imports, prompts, function signatures, CLI registration, JSON shapes, doc consistency | Every PR. Should be in your pre-push muscle memory. |
| **L2 — Live integration** | ~10–60 s | PostgreSQL + Qdrant up | hybrid_search returns results, Qdrant collections exist, ensure_node_level_index is idempotent, db queries work | Before shipping a "Phase" feature drop, after touching `storage/`, `retrieval/`, or `ingestion/` |
| **L3 — End-to-end** | minutes | PG + Qdrant + Ollama | Ollama answers, one tiny `complete()` call returns text, bge-m3 model loads and embeds | After touching `rag/llm.py`, `embedder.py`, `reranker.py`, or after a model swap |

L2 and L3 are **skipped automatically** when their service prerequisites aren't reachable — so on a laptop without Qdrant running, `sciknow test --layer all` reports L2/L3 as `~ skipped` instead of failing hard.

## How to run it

```bash
# Default — L1 only, fast (under 10 seconds typically)
uv run sciknow test

# Live integration (needs PG + Qdrant up)
uv run sciknow test --layer L2

# Everything
uv run sciknow test --layer all

# Stop on first failure
uv run sciknow test -l all --fail-fast
```

The output is a Rich table per layer with green ✓ / yellow ~ (skipped) / red ✗ markers, per-test wall time in milliseconds, and a one-line note for tests that report data (e.g. `docs=2752  meta=2752  chunks=102268`). Exit code is `0` on full pass and `1` on any failure.

## When each layer should run

### L1 — every PR

Anything that touches Python source. L1 catches:

- Module imports break (most common failure mode after a refactor)
- A prompt's hedging-fidelity rule got dropped
- The PDTB discourse-relation enum is missing a label
- `_save_draft` no longer accepts `custom_metadata`
- A new CLI command is defined but not registered on the Typer app
- The web reader template lost its `OVERSTATED` CSS class
- `docs/RESEARCH.md` doesn't reference a new phase

L1 is fast enough (~8 seconds with the heavy `umap` import) that there's no excuse to skip it. If a check is slower than 1 second, it should probably move to L2.

### L2 — before shipping a Phase or after infrastructure changes

L2 needs PG + Qdrant. Run it when:

- You changed anything in `sciknow/storage/`
- You changed `hybrid_search.py`, `context_builder.py`, or `reranker.py`
- You added a new Qdrant payload index
- You're about to push a "Phase N" feature drop
- You're about to bump the embedding model dimension

L2 is the layer that catches **drift between code and the live database/collection state**. Example: if a new payload field is added in `_hydrate` but the actual Qdrant points don't have it, L2's hybrid_search smoke will surface that.

### L3 — after model or LLM-path changes

L3 needs Ollama too. Run it when:

- You changed `sciknow/rag/llm.py`
- You changed the embedder or reranker
- You swapped the main or fast LLM model
- You changed the Ollama host
- You made a release-tagging commit

L3 is the slowest layer because the first LLM call typically loads a model into VRAM (~3 minutes for a 30B model). Once the model is hot, subsequent calls are fast.

## How to add a new check

A test is a function. Open `sciknow/testing/protocol.py`, write the function in the appropriate layer section, and append it to the layer's list at the bottom of the file. That's it.

### Example: add an L1 test that a new scoring dimension is in the rubric

```python
def l1_prompts_phase_X_my_new_dim() -> None:
    """Phase X — my_new_dim is in score_draft."""
    from sciknow.rag import prompts
    sys_s, _ = prompts.score_draft("results", "x", "draft", [])
    assert "my_new_dim" in sys_s, "scorer missing my_new_dim"
```

Then append it to `L1_TESTS` at the bottom of the file:

```python
L1_TESTS: list[Callable] = [
    ...,
    l1_prompts_phase_X_my_new_dim,
]
```

`uv run sciknow test` will pick it up automatically.

### Example: add an L2 test that exercises a new helper

```python
def l2_my_new_helper_works() -> None:
    """The new my_new_helper returns a non-empty list against the live DB."""
    from sciknow.retrieval.my_module import my_new_helper
    from sciknow.storage.db import get_session
    with get_session() as session:
        result = my_new_helper(session)
    assert isinstance(result, list)
    return TestResult.ok(
        name="l2_my_new_helper_works",
        message=f"helper returned {len(result)} items",
    )
```

Note: tests can return a `TestResult` (from `sciknow.testing.protocol`) when they want to attach an informational message. Otherwise they return `None` and the harness builds a default OK result with the elapsed time.

### Conventions for test functions

- **Naming:** `l{1,2,3}_what_it_checks`. The `l1_` / `l2_` / `l3_` prefix is the layer hint (the harness doesn't enforce this; it's for grep-ability).
- **Docstrings:** one line, explains what's being verified. Read like a test report row.
- **Assertions:** plain `assert` with a clear message on failure. The harness catches `AssertionError` and records the message.
- **Don't catch exceptions yourself.** Let the harness do it. Catching exceptions inside a test function turns a real failure into a silent pass.
- **Skip gracefully.** If the test needs data that may not exist (e.g. RAPTOR nodes), check first and return `TestResult.skip(...)` instead of asserting.
- **Speed budgets:** L1 tests should be under 1 second (the umap import is the noisy exception). L2 under 10 seconds. L3 has no budget (LLM calls are inherently slow on a cold model).

## Reading a failing report

When a test fails, the report shows a one-line message that combines the assertion text with the last few traceback lines. Example:

```
  ✗  l1_prompts_phase11_cove   3ms   AssertionError: cove_answer missing NOT_IN_SOURCES verdict
```

The message is enough to identify the regression. To debug, open `protocol.py` and run the test function directly:

```bash
uv run python -c "from sciknow.testing.protocol import l1_prompts_phase11_cove; l1_prompts_phase11_cove()"
```

That gives you the full traceback.

## What's NOT in the harness (and why)

These intentionally aren't tested by `sciknow test`:

- **Prompt output quality** — there's no automated way to check that an LLM-generated summary is "good". The `book autowrite-bench` command (Phase 13) is the right tool for this; it measures *score variance* across runs, not absolute quality.
- **Long-running ingestion** — running MinerU on a real PDF takes minutes. The pipeline tests are smoke-only at L1; for real ingestion verification, run `sciknow ingest file <small-paper.pdf>` manually.
- **Cross-model regressions** — if you swap the main LLM, scoring numbers will drift. That's not a code bug, it's a model change. Use `book draft compare --rescore` to measure the drift.
- **RAPTOR end-to-end build** — the build is a one-off batch op that takes 5–30 minutes. L1 verifies the surface is intact and `_cluster_with_gmm_bic` works on synthetic data; the live build is verified by inspecting `sciknow catalog raptor stats` after running it.

## Protocol for shipping a new feature

1. **Implement.**
2. **Run L1.** `uv run sciknow test`. Must pass.
3. **Add a new L1 test** for whatever you just shipped. Re-run L1.
4. **If you touched storage / retrieval / ingestion, run L2.** `uv run sciknow test --layer L2`.
5. **If you touched anything in the LLM path, run L3.** `uv run sciknow test --layer L3`.
6. **Commit and push.** Include the test result counts in the commit body if it's a big change.
7. **For "Phase" feature drops:** run `--layer all` and include the full output (or a summary) in the PR description / commit message.

This protocol exists so that future-you (or future-Claude) can ship a complex feature change with confidence, knowing that the regression-catching net is in place.

## Phase 32 — QA module overhaul

Phase 32 added a shared `sciknow/testing/helpers.py` module so new tests don't have to re-derive the inline boilerplate that grew up around the protocol. It exposes:

- `get_test_client()` — cached `fastapi.testclient.TestClient` with the global book wired up. Used by L2 endpoint shape tests.
- `a_book_id()` / `a_chapter_id()` / `a_draft_id()` — fetch identifiers for the first record of each kind from the live DB. Cached.
- `inspect_handler_source(name)` — return the source of a named function from `sciknow.web.app` without 5 lines of `import inspect` boilerplate.
- `web_app_full_source()` — full module source as one cached string for tests that grep across the whole module.
- `rendered_template_static()` — render the TEMPLATE f-string with placeholder values. **No DB access** — L1-safe.
- `rendered_template_with_data()` — render the full HTML with real DB data. L2 only.
- `js_function_definitions()` / `js_onclick_handlers()` — parse the module source and return the set of defined JS functions and the set of onclick-attribute references. Used by the integrity test.
- `all_app_routes()` / `find_route(path)` — endpoint inventory introspection (unions methods across multiple route entries with the same path).

On top of these, Phase 32 added:

**L1 (no service deps):**
- `l1_phase32_qa_helpers_module` — sanity that helpers import cleanly and `rendered_template_static()` works without a DB.
- `l1_phase32_endpoint_inventory` — verifies every expected `(method, path)` is registered. Catalog is hand-maintained at the top of the test; adding a new endpoint to `web/app.py` means adding one line here too.
- `l1_phase32_js_handler_integrity` — every `onclick`/`oninput`/`onchange`/etc handler resolves to a defined JS function (with a JS-builtins/keywords exclusion list to avoid false positives like `alert(...)` or `if(event.target===this)...`).
- `l1_phase32_render_helpers_escape_chain` — every `_render_*` helper that builds HTML via f-strings uses `_esc()` or `_md_to_html()`. XSS audit.
- `l1_phase32_no_global_state_leak` — guards against new mutable module-level globals creeping into `web/app.py`.
- `l1_phase32_endpoint_handler_signatures_consistent` — every `@app.{get,post,put,delete}` handler is `async def` (catches accidental sync handlers that block the event loop).

**L2 (PG required):**
- `l2_phase32_endpoint_shapes` — TestClient hits the major read-only endpoints and asserts status 200 + expected JSON keys. Catches handler exceptions and schema drift.
- `l2_phase32_data_invariants` — DB-level invariants the GUI relies on (no orphaned drafts, no dangling chapter_id, no nameless chapters, no comments on deleted drafts).

The total Phase 32 surface is **6 new L1 + 2 new L2 tests**, taking the harness from 83 to 91 tests.

## File map

```
sciknow/
└── testing/
    ├── __init__.py        # re-exports protocol API
    ├── helpers.py         # shared TestClient + JS parsers + route inventory (Phase 32)
    └── protocol.py        # the harness + all test functions

sciknow/cli/main.py        # `sciknow test` command (wraps the harness)
docs/TESTING.md            # this document
```
