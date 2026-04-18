"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Builds a hierarchical tree of summary embeddings on top of the existing
chunk index. The tree's leaves are the chunks already in Qdrant. Each
non-leaf level is built by:

  1. Fetching the parent-level vectors from Qdrant
  2. UMAP-reducing them to ~10 dimensions
  3. GMM clustering with BIC-selected k (RAPTOR's paper specifically
     argues for soft clustering — chunks can contribute to multiple
     summaries by probability)
  4. For each cluster, fetching the source content (from PostgreSQL for
     level 0, from Qdrant payload for higher levels) and summarising
     with the LLM via the RAPTOR_SUMMARY prompt
  5. Embedding the summary with bge-m3 and upserting it into the same
     `papers` Qdrant collection with `node_level: N`

At retrieval time, no special handling is needed — the existing hybrid
search just returns a mix of leaves and summary nodes, which the reranker
scores against the user's query. `_hydrate` in hybrid_search.py recognises
RAPTOR nodes by their `node_level` payload field and reads their content
from the payload directly (they have no row in the chunks table).

Source: Sarthi et al., "RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval", ICLR 2024 (arXiv:2401.18059).
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Iterator
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from sciknow.storage.qdrant import PAPERS_COLLECTION

logger = logging.getLogger("sciknow.raptor")

Event = dict


# ── Helpers ──────────────────────────────────────────────────────────────────


def _scroll_papers(
    qdrant: QdrantClient,
    *,
    node_level: int | None = None,
    limit_total: int | None = None,
    with_vectors: bool = True,
) -> list:
    """Scroll all (or up to `limit_total`) points from the papers collection.

    If `node_level` is given, filter by that level. With `node_level=None`
    we return EVERYTHING in the collection — use sparingly.
    """
    points = []
    offset = None
    qdrant_filter = None
    if node_level is not None:
        qdrant_filter = Filter(must=[
            FieldCondition(key="node_level", match=MatchValue(value=node_level))
        ])
    while True:
        result, next_offset = qdrant.scroll(
            collection_name=PAPERS_COLLECTION,
            scroll_filter=qdrant_filter,
            limit=500,
            offset=offset,
            with_vectors=["dense"] if with_vectors else False,
            with_payload=True,
        )
        points.extend(result)
        if limit_total is not None and len(points) >= limit_total:
            return points[:limit_total]
        if next_offset is None:
            break
        offset = next_offset
    return points


def _backfill_leaf_node_level(qdrant: QdrantClient) -> int:
    """Set `node_level: 0` on every existing point that is missing the field.

    Idempotent. Returns the number of points updated. Safe to call multiple
    times — points that already have node_level set will not be touched.

    We do this in the most surgical way Qdrant supports: scroll all points,
    filter to those without node_level locally, then bulk set_payload by id.
    Qdrant's set_payload doesn't have a server-side "where missing" filter,
    so we have to scan.
    """
    from qdrant_client.models import PayloadSelectorInclude

    n_set = 0
    offset = None
    batch_ids: list[str] = []
    while True:
        result, next_offset = qdrant.scroll(
            collection_name=PAPERS_COLLECTION,
            limit=2000,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )
        for pt in result:
            payload = pt.payload or {}
            if "node_level" not in payload:
                batch_ids.append(str(pt.id))

        if len(batch_ids) >= 500:
            qdrant.set_payload(
                collection_name=PAPERS_COLLECTION,
                payload={"node_level": 0},
                points=batch_ids,
                wait=True,
            )
            n_set += len(batch_ids)
            batch_ids.clear()

        if next_offset is None:
            break
        offset = next_offset

    if batch_ids:
        qdrant.set_payload(
            collection_name=PAPERS_COLLECTION,
            payload={"node_level": 0},
            points=batch_ids,
            wait=True,
        )
        n_set += len(batch_ids)

    return n_set


def _delete_existing_summary_levels(qdrant: QdrantClient) -> int:
    """Delete every point with node_level >= 1 (used by --rebuild)."""
    qdrant.delete(
        collection_name=PAPERS_COLLECTION,
        points_selector=Filter(must=[
            FieldCondition(
                key="node_level",
                range=Range(gte=1),
            )
        ]),
        wait=True,
    )
    # Qdrant doesn't return a delete count from this call, so we re-scroll
    # to confirm. Cheap because deleted points are gone.
    leftover = _scroll_papers(qdrant, node_level=None, with_vectors=False, limit_total=1)
    return len(leftover)  # informational only


def _count_summary_nodes_per_level(qdrant: QdrantClient) -> dict[int, int]:
    """Scroll every summary point (node_level >= 1) and count per level.

    Phase 54.6.60 — powers the "already built" skip's informative log.
    Cheap in practice because summary nodes are O(100–1000) even on
    large corpora (level-1 is the widest, bounded by the number of
    leaf clusters).
    """
    from collections import Counter
    counts: Counter = Counter()
    offset = None
    while True:
        result, next_offset = qdrant.scroll(
            collection_name=PAPERS_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="node_level", range=Range(gte=1))
            ]),
            limit=500,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )
        for pt in result:
            lvl = (pt.payload or {}).get("node_level")
            if isinstance(lvl, int) and lvl >= 1:
                counts[lvl] += 1
        if next_offset is None:
            break
        offset = next_offset
    return dict(sorted(counts.items()))


def _has_existing_summary_nodes(qdrant: QdrantClient) -> bool:
    """Cheap check: are there any points with node_level >= 1?"""
    result, _ = qdrant.scroll(
        collection_name=PAPERS_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="node_level", range=Range(gte=1))
        ]),
        limit=1,
        with_vectors=False,
        with_payload=False,
    )
    return len(result) > 0


# ── Clustering ───────────────────────────────────────────────────────────────


def _cluster_with_gmm_bic(
    embeddings: np.ndarray,
    *,
    max_k: int = 50,
    random_state: int = 42,
    n_components_umap: int = 10,
    n_neighbors: int = 15,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Reduce with UMAP, cluster with GMM, select k by BIC.

    Returns:
        labels: hard-assignment cluster labels (np.ndarray of shape (n,))
        best_k: the BIC-selected number of clusters
        proba: GMM membership probabilities of shape (n, best_k)

    The hard labels are used for downstream summarisation (each chunk
    contributes to its argmax cluster). The proba matrix is returned in
    case future work wants to use soft assignment — RAPTOR's paper
    technically allows a chunk to contribute to multiple cluster summaries
    when its membership probability is above a threshold, but for v1 we
    use hard assignment for simplicity.
    """
    n = embeddings.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), 1, np.ones((n, 1), dtype=np.float32)

    # UMAP-reduce — cosine metric matches bge-m3's training.
    import umap
    target_dim = min(n_components_umap, max(2, n - 2))
    reducer = umap.UMAP(
        n_components=target_dim,
        n_neighbors=min(n_neighbors, n - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    X_reduced = reducer.fit_transform(embeddings)

    # GMM with BIC selection. Search a reasonable range of k values; in
    # practice the elbow is usually well below n / 5 for scientific corpora.
    from sklearn.mixture import GaussianMixture

    upper_k = min(max_k, max(2, n // 4))
    if upper_k < 2:
        return np.zeros(n, dtype=int), 1, np.ones((n, 1), dtype=np.float32)

    best_bic = np.inf
    best_k = 2
    best_gmm = None
    for k in range(2, upper_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=random_state,
                n_init=1,
                max_iter=100,
            )
            gmm.fit(X_reduced)
            bic = gmm.bic(X_reduced)
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_gmm = gmm
        except Exception as exc:
            logger.warning("GMM k=%d failed: %s", k, exc)
            continue

    if best_gmm is None:
        # Fallback: single cluster
        return np.zeros(n, dtype=int), 1, np.ones((n, 1), dtype=np.float32)

    labels = best_gmm.predict(X_reduced)
    proba = best_gmm.predict_proba(X_reduced)
    return labels.astype(int), best_k, proba.astype(np.float32)


# ── Per-cluster summarisation ────────────────────────────────────────────────


def _fetch_chunk_texts(
    session, qdrant_point_ids: list[str],
) -> dict[str, str]:
    """Look up the full content of leaf chunks by qdrant_point_id from PostgreSQL."""
    if not qdrant_point_ids:
        return {}
    from sqlalchemy import text
    placeholders = ", ".join(f":id{i}" for i in range(len(qdrant_point_ids)))
    params = {f"id{i}": pid for i, pid in enumerate(qdrant_point_ids)}
    rows = session.execute(
        text(f"""
            SELECT qdrant_point_id::text, content
            FROM chunks
            WHERE qdrant_point_id::text IN ({placeholders})
        """),
        params,
    ).fetchall()
    return {row[0]: row[1] for row in rows if row[1]}


def _summarise_cluster(
    cluster_chunk_ids: list[str],
    chunk_texts: dict[str, str],
    payloads: dict[str, dict],
    level: int,
    model: str | None,
    max_chars_per_chunk: int = 1500,
    max_total_chars: int = 24000,
) -> str | None:
    """Build the LLM prompt for one cluster and return the summary text.

    For level 1, `chunk_texts` contains the original chunk content from PG.
    For level >=2, the parent nodes are themselves summaries — their content
    is in `payloads[node_id]["summary_text"]`.
    """
    from sciknow.config import settings
    from sciknow.rag import prompts as rag_prompts
    from sciknow.rag.llm import complete as llm_complete

    parts = []
    total = 0
    for cid in cluster_chunk_ids:
        # Higher levels: read summary_text from the payload directly.
        text_content = chunk_texts.get(cid)
        if text_content is None:
            payload = payloads.get(cid) or {}
            text_content = payload.get("summary_text") or payload.get("content_preview", "")
        if not text_content:
            continue

        text_content = text_content[:max_chars_per_chunk]

        # Tag with a small header so the LLM can attribute consistently.
        payload = payloads.get(cid) or {}
        title = payload.get("title") or "Untitled"
        year = payload.get("year") or payload.get("year_max") or "n.d."
        section = payload.get("section_type") or "text"
        header = f"--- excerpt [{section}] from \"{title}\" ({year}) ---"
        block = f"{header}\n{text_content}"

        if total + len(block) > max_total_chars:
            break
        parts.append(block)
        total += len(block)

    if not parts:
        return None

    chunks_text = "\n\n".join(parts)
    sys_p, usr_p = rag_prompts.raptor_summary(chunks_text, n=len(parts))

    try:
        # Use the FAST model by default — RAPTOR is a one-shot batch op
        # and the summary quality is fine with a 7B model. The user can
        # override with --model.
        chosen_model = model or settings.llm_fast_model
        raw = llm_complete(
            sys_p, usr_p,
            model=chosen_model,
            temperature=0.2,
            num_ctx=16384,
        )
        # Strip thinking blocks and surrounding fences.
        raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return raw or None
    except Exception as exc:
        logger.warning("RAPTOR summary call failed (level=%d): %s", level, exc)
        return None


# ── One level build ──────────────────────────────────────────────────────────


def _build_level(
    *,
    session,
    qdrant: QdrantClient,
    parent_level: int,
    target_level: int,
    min_cluster_size: int,
    model: str | None,
    dry_run: bool,
) -> Iterator[Event]:
    """Build one level of the RAPTOR tree from the points at `parent_level`.

    Yields progress events:
        {"type": "level_start", "level": N, "parent_count": M}
        {"type": "cluster_progress", "level": N, "i": int, "of": int}
        {"type": "level_done", "level": N, "n_clusters": int, "n_summaries": int, "elapsed": float}
    """
    t0 = time.monotonic()

    # Fetch the parent-level points (with vectors).
    yield {"type": "progress", "stage": f"fetch L{parent_level}",
           "detail": f"Loading level-{parent_level} vectors from Qdrant..."}
    parents = _scroll_papers(qdrant, node_level=parent_level, with_vectors=True)
    if len(parents) < min_cluster_size * 2:
        yield {"type": "level_skipped", "level": target_level,
               "parent_count": len(parents),
               "reason": f"too few parent nodes ({len(parents)})"}
        return

    yield {"type": "level_start", "level": target_level,
           "parent_count": len(parents)}

    # Build the embedding matrix and the parallel id list.
    parent_ids: list[str] = []
    vectors: list[list[float]] = []
    payloads: dict[str, dict] = {}
    for pt in parents:
        pid = str(pt.id)
        parent_ids.append(pid)
        vectors.append(pt.vector["dense"])
        payloads[pid] = pt.payload or {}
    X = np.array(vectors, dtype=np.float32)

    # Cluster.
    yield {"type": "progress", "stage": f"cluster L{target_level}",
           "detail": f"Clustering {len(parent_ids)} nodes (UMAP + GMM/BIC)..."}
    try:
        labels, best_k, proba = _cluster_with_gmm_bic(X)
    except Exception as exc:
        logger.exception("Clustering failed at level %d", target_level)
        yield {"type": "error", "level": target_level,
               "message": f"Clustering failed: {exc}"}
        return

    # Group parent ids by cluster label (hard assignment, Phase 12).
    clusters: dict[int, list[str]] = defaultdict(list)
    for pid, lbl in zip(parent_ids, labels):
        clusters[int(lbl)].append(pid)

    # Phase 34 — soft RAPTOR clustering: also assign chunks to secondary
    # clusters where their GMM membership probability exceeds the
    # threshold. This lets one chunk contribute to multiple cluster
    # summaries so its information is "visible" through more entry
    # points at retrieval time — better recall for queries that approach
    # the topic from different angles. See docs/ROADMAP.md §2 and
    # docs/RESEARCH.md for the research basis.
    from sciknow.config import settings
    soft_threshold = getattr(settings, "raptor_soft_threshold", 0.15)
    n_soft_added = 0
    if soft_threshold > 0 and proba is not None and proba.shape[1] == best_k:
        for i, pid in enumerate(parent_ids):
            primary = int(labels[i])
            for k in range(best_k):
                if k == primary:
                    continue  # already in via hard assignment
                if float(proba[i, k]) >= soft_threshold:
                    if pid not in clusters[k]:
                        clusters[k].append(pid)
                        n_soft_added += 1
    if n_soft_added > 0:
        logger.info(
            "RAPTOR L%d soft clustering: %d extra assignments (threshold=%.2f)",
            target_level, n_soft_added, soft_threshold,
        )

    # Drop clusters smaller than min_cluster_size.
    clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
    if not clusters:
        yield {"type": "level_skipped", "level": target_level,
               "parent_count": len(parents),
               "reason": "no cluster met min_cluster_size"}
        return

    yield {"type": "progress", "stage": f"cluster L{target_level}",
           "detail": f"BIC selected k={best_k}, kept {len(clusters)} clusters "
                     f">= {min_cluster_size} members"}

    if dry_run:
        # Show the cluster sizes and stop.
        sizes = sorted((len(v) for v in clusters.values()), reverse=True)
        yield {"type": "dry_run_summary", "level": target_level,
               "cluster_sizes": sizes,
               "k_selected": best_k}
        yield {"type": "level_done", "level": target_level,
               "n_clusters": len(clusters), "n_summaries": 0,
               "elapsed": time.monotonic() - t0}
        return

    # For level 1, we need full chunk text from PG. For higher levels,
    # we can read summary_text directly from the payloads we already have.
    chunk_texts: dict[str, str] = {}
    if parent_level == 0:
        all_leaf_ids = [pid for ids in clusters.values() for pid in ids]
        chunk_texts = _fetch_chunk_texts(session, all_leaf_ids)

    n_summaries_made = 0
    n_clusters_total = len(clusters)
    from sciknow.ingestion.embedder import embed_summary_text

    for i, (label, cluster_ids) in enumerate(clusters.items(), 1):
        yield {"type": "cluster_progress", "level": target_level,
               "i": i, "of": n_clusters_total, "size": len(cluster_ids)}

        summary = _summarise_cluster(
            cluster_ids,
            chunk_texts=chunk_texts,
            payloads=payloads,
            level=target_level,
            model=model,
        )
        if not summary:
            yield {"type": "cluster_skipped", "level": target_level,
                   "label": label, "reason": "empty summary"}
            continue

        # Build the payload for the new summary node. Aggregate metadata
        # from the children so the node carries useful filter info.
        doc_ids: set[str] = set()
        years: list[int] = []
        section_types: set[str] = set()
        topic_clusters: set[str] = set()
        for cid in cluster_ids:
            p = payloads.get(cid) or {}
            d = p.get("document_id")
            if d:
                doc_ids.add(d)
            for d in (p.get("document_ids") or []):
                doc_ids.add(d)
            y = p.get("year") or p.get("year_max")
            if isinstance(y, int):
                years.append(y)
            elif isinstance(p.get("year_min"), int) and isinstance(p.get("year_max"), int):
                years.extend([p["year_min"], p["year_max"]])
            st = p.get("section_type")
            if st:
                section_types.add(st)
            tc = p.get("topic_cluster")
            if tc:
                topic_clusters.add(tc)

        new_payload = {
            "node_level": target_level,
            "summary_text": summary,
            "child_chunk_ids": cluster_ids,
            "child_count": len(cluster_ids),
            "document_ids": sorted(doc_ids),
            "n_documents": len(doc_ids),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "section_types": sorted(section_types),
            "topic_clusters": sorted(topic_clusters),
            # Display fields used by _hydrate
            "title": f"RAPTOR L{target_level} synthesis ({len(doc_ids)} papers)",
            "section_type": f"raptor_l{target_level}",
            "section_title": f"Cluster of {len(cluster_ids)} {'leaves' if parent_level == 0 else f'L{parent_level} nodes'}",
            "content_preview": summary[:200],
            # Mark for citation_boost: empty document_id excludes from boost.
            "document_id": "",
        }

        try:
            embed_summary_text(summary, new_payload, qdrant)
            n_summaries_made += 1
        except Exception as exc:
            logger.warning("Embed summary failed at level %d: %s", target_level, exc)

    yield {"type": "level_done", "level": target_level,
           "n_clusters": len(clusters),
           "n_summaries": n_summaries_made,
           "elapsed": time.monotonic() - t0}


# ── Top-level build entrypoint ───────────────────────────────────────────────


def build_raptor_tree(
    *,
    max_levels: int = 4,
    min_cluster_size: int = 2,
    min_top_level: int = 4,
    model: str | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
) -> Iterator[Event]:
    """
    Build the RAPTOR tree on top of the existing chunk index.

    Steps:
      0. Ensure the node_level payload index exists.
      1. (If --rebuild) delete every existing point with node_level >= 1.
      2. Backfill node_level=0 on every leaf chunk that's missing it.
      3. For level in 1..max_levels:
         - Cluster level-(N-1) nodes
         - Summarise each cluster with the LLM
         - Embed and upsert as level-N nodes
         - Stop early if fewer than `min_top_level` parent nodes remain.

    Yields events as it goes (consumed by `sciknow catalog raptor build`).
    Generators don't open DB sessions themselves — the caller is expected
    to construct a Session inside the loop or wrap calls in a session
    context manager.

    This function operates with `min_cluster_size=2` by default because
    on small corpora the BIC-selected k can be high and individual clusters
    will be small. Bump it up on large corpora.
    """
    from sciknow.storage.db import get_session
    from sciknow.storage.qdrant import ensure_node_level_index, get_client

    qdrant = get_client()

    # Step 0: index.
    yield {"type": "progress", "stage": "init",
           "detail": "Ensuring node_level payload index..."}
    ensure_node_level_index(qdrant)

    # Step 1: optional wipe.
    if rebuild and not dry_run:
        if _has_existing_summary_nodes(qdrant):
            yield {"type": "progress", "stage": "rebuild",
                   "detail": "Deleting existing RAPTOR summary nodes..."}
            _delete_existing_summary_levels(qdrant)
    elif not rebuild and _has_existing_summary_nodes(qdrant) and not dry_run:
        # Phase 54.6.60 — "already built" is a valid terminal state for
        # a one-time-batch build (see module-level docstring). Return
        # it as a benign skip event instead of a hard error so the
        # idempotent-resume contract holds — `sciknow refresh` can be
        # re-run safely and the RAPTOR step reports "nothing to do"
        # rather than exit 1. Users who want to absorb newly ingested
        # papers into the tree must opt in with --rebuild (documented
        # in the CLI help).
        yield {"type": "already_built",
               "per_level_counts": _count_summary_nodes_per_level(qdrant)}
        return

    # Step 2: backfill node_level=0 on leaves.
    yield {"type": "progress", "stage": "backfill",
           "detail": "Setting node_level=0 on existing leaf chunks..."}
    n_set = _backfill_leaf_node_level(qdrant)
    yield {"type": "progress", "stage": "backfill",
           "detail": f"Backfilled node_level=0 on {n_set} leaf points "
                     f"(0 means everything was already tagged)."}

    # Step 3: build levels.
    last_level_size = None
    with get_session() as session:
        for level in range(1, max_levels + 1):
            for ev in _build_level(
                session=session,
                qdrant=qdrant,
                parent_level=level - 1,
                target_level=level,
                min_cluster_size=min_cluster_size,
                model=model,
                dry_run=dry_run,
            ):
                yield ev
                if ev.get("type") == "level_done":
                    last_level_size = ev.get("n_summaries", 0)

            if last_level_size is None or last_level_size < min_top_level:
                yield {"type": "tree_complete",
                       "stopped_at_level": level,
                       "reason": (
                           "fewer than min_top_level summaries at this level"
                           if last_level_size is not None
                           else "level skipped"
                       )}
                return

    yield {"type": "tree_complete", "stopped_at_level": max_levels,
           "reason": "reached max_levels"}
