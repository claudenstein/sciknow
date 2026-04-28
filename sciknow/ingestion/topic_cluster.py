"""
BERTopic-style embedding-based topic clustering.

Uses the bge-m3 abstract embeddings already stored in Qdrant — no extra
LLM calls for clustering.  Pipeline:

  1. Fetch dense vectors from the ``abstracts`` Qdrant collection
  2. UMAP dimensionality reduction (1024 → 5 dims)
  3. HDBSCAN density-based clustering (auto-determines cluster count)
  4. c-TF-IDF per cluster for interpretable topic keywords
  5. One LLM call per cluster to generate a human-readable name

10-50x faster than the old LLM-batch approach and deterministic.
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict

import numpy as np

logger = logging.getLogger("sciknow.cluster")


def cluster_papers(
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    n_components: int = 5,
    n_neighbors: int = 15,
    model: str | None = None,
) -> dict:
    """
    Run BERTopic-style clustering on all paper abstracts.

    Returns::

        {
            "clusters": [{"id": 0, "name": "Solar Irradiance", "keywords": [...], "count": N}, ...],
            "assignments": {doc_id: cluster_id, ...},
            "noise_count": N,   # papers HDBSCAN couldn't assign (-1 label)
        }
    """
    from sciknow.storage.qdrant import ABSTRACTS_COLLECTION, get_client

    qdrant = get_client()

    # ── Step 1: Fetch all abstract vectors + metadata from Qdrant ────────
    logger.info("Fetching abstract vectors from Qdrant...")

    all_points = []
    offset = None
    while True:
        result, next_offset = qdrant.scroll(
            collection_name=ABSTRACTS_COLLECTION,
            limit=500,
            offset=offset,
            with_vectors=["dense"],
            with_payload=True,
        )
        all_points.extend(result)
        if next_offset is None:
            break
        offset = next_offset

    if len(all_points) < min_cluster_size * 2:
        raise ValueError(
            f"Too few abstracts ({len(all_points)}) for clustering. "
            f"Need at least {min_cluster_size * 2}."
        )

    logger.info("Fetched %d abstract vectors", len(all_points))

    doc_ids = []
    vectors = []
    abstracts = []
    titles = []
    for pt in all_points:
        doc_id = pt.payload.get("document_id", "")
        doc_ids.append(doc_id)
        vectors.append(pt.vector["dense"])
        abstracts.append(pt.payload.get("content_preview", ""))
        titles.append(pt.payload.get("title", ""))

    X = np.array(vectors, dtype=np.float32)

    # ── Step 2: UMAP dimensionality reduction ────────────────────────────
    logger.info("UMAP reduction: %d dims → %d dims...", X.shape[1], n_components)

    import umap
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    X_reduced = reducer.fit_transform(X)

    # ── Step 3: HDBSCAN clustering ───────────────────────────────────────
    logger.info("HDBSCAN clustering (min_cluster_size=%d)...", min_cluster_size)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    logger.info("Found %d clusters, %d noise points", n_clusters, noise_count)

    # ── Step 4: c-TF-IDF for topic keywords ──────────────────────────────
    logger.info("Computing c-TF-IDF keywords per cluster...")

    # Build per-cluster document text
    cluster_docs: dict[int, list[str]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            text = f"{titles[i]} {abstracts[i]}"
            cluster_docs[label].append(text)

    # Simple c-TF-IDF: term frequency in cluster / inverse frequency across clusters
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    cluster_ids_sorted = sorted(cluster_docs.keys())
    joined_docs = [" ".join(cluster_docs[cid]) for cid in cluster_ids_sorted]

    vectorizer = CountVectorizer(
        max_features=10000,
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
    )
    tf = vectorizer.fit_transform(joined_docs)
    tfidf = TfidfTransformer().fit_transform(tf)

    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords: dict[int, list[str]] = {}
    for idx, cid in enumerate(cluster_ids_sorted):
        scores = tfidf[idx].toarray().flatten()
        top_indices = scores.argsort()[-10:][::-1]
        cluster_keywords[cid] = [feature_names[i] for i in top_indices if scores[i] > 0]

    # ── Step 5: LLM naming (one call per cluster) ────────────────────────
    logger.info("Generating cluster names with LLM...")

    cluster_names = _name_clusters(cluster_keywords, cluster_docs, model=model)

    # ── Build result ─────────────────────────────────────────────────────
    assignments: dict[str, int] = {}
    for i, label in enumerate(labels):
        if label >= 0:
            assignments[doc_ids[i]] = int(label)

    clusters = []
    for cid in cluster_ids_sorted:
        clusters.append({
            "id": cid,
            "name": cluster_names.get(cid, f"Topic {cid}"),
            "keywords": cluster_keywords.get(cid, [])[:8],
            "count": len(cluster_docs[cid]),
        })

    return {
        "clusters": clusters,
        "assignments": assignments,
        "noise_count": noise_count,
    }


def _name_clusters(
    keywords: dict[int, list[str]],
    docs: dict[int, list[str]],
    model: str | None = None,
) -> dict[int, str]:
    """Use a single LLM call to name all clusters from their keywords."""
    from sciknow.rag.llm import complete as llm_complete
    # Phase 55.V3 — single writer call; evict retrieval roles so the
    # writer claims the full GPU. Cluster naming runs after the
    # embedding-driven cluster pass (which left embedder hot), so
    # this swap genuinely reclaims VRAM.
    from sciknow.core.book_ops import _swap_to_phase
    _swap_to_phase("generate")

    lines = []
    for cid in sorted(keywords.keys()):
        kw = ", ".join(keywords[cid][:6])
        # Add 2 sample titles for context
        sample_titles = [d.split(" ", 1)[0][:80] for d in docs[cid][:2]]
        samples = "; ".join(sample_titles)
        lines.append(f"Cluster {cid}: keywords=[{kw}] samples=[{samples}]")

    system = (
        "You are a scientific librarian. Given topic clusters with their keywords "
        "and sample paper titles, assign a short name (2-4 words) to each cluster.\n"
        "Respond ONLY with valid JSON: {\"0\": \"Name\", \"1\": \"Name\", ...}"
    )
    user = "Clusters:\n" + "\n".join(lines) + "\n\nName each cluster."

    try:
        raw = llm_complete(system, user, model=model, temperature=0.0, num_ctx=8192)
        # Strip thinking blocks
        raw = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        # Clean JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last > first:
            raw = raw[first:last + 1]

        import json
        names = json.loads(raw, strict=False)
        return {int(k): str(v) for k, v in names.items()}
    except Exception as exc:
        logger.warning("LLM cluster naming failed: %s — using keyword fallback", exc)
        return {cid: " ".join(kw[:2]).title() for cid, kw in keywords.items()}
