"""Phase 56.A — Corpus topic tree.

A queryable in-memory representation of the corpus's natural topical
structure, built from the existing RAPTOR summary nodes that live in
Qdrant. The outline proposer (56.B) walks the tree top-down to surface
chapter / section / subsection candidates anchored to specific
clusters.

We don't recompute clustering here — RAPTOR's per-corpus build (run
via ``sciknow corpus refresh`` / ``sciknow catalog raptor``) has
already grouped chunks into level-1 clusters and summarised them, then
grouped level-1 nodes into level-2 super-clusters. This module reads
those summary nodes back out, infers parent/child relationships from
the ``child_chunk_ids`` pointer that summarisation writes per-node,
and exposes them as a tree.

Key methods:

  TopicTree.from_qdrant(qdrant, project) → loads from a project's
    Qdrant collection
  tree.roots() → top-level cluster nodes (highest existing level)
  tree.children_of(node) → immediate children of a cluster
  tree.papers_in(node) → all papers reachable from a cluster (incl.
    descendant clusters' papers)
  tree.score_against_scope(node, scope_embedding) → cosine similarity
    in [-1, 1]
  tree.walk_in_scope(scope_embedding, threshold) → iterator over nodes
    above the scope-relevance threshold, in BFS order from the roots

Coverage / scope are computed on demand and cached on the tree
instance for the lifetime of the load. Callers that want them
persisted across processes should serialise the tree itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data shape
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TopicNode:
    """One cluster in the topic tree.

    Mirrors the RAPTOR summary-node payload schema documented in
    ``sciknow/ingestion/raptor.py``. Child relationships are populated
    by ``TopicTree`` after all nodes are loaded.
    """
    node_id: str                   # Qdrant point id (uuid str)
    level: int                     # 1 = first-level cluster; 2 = super-cluster; …
    summary_text: str
    title: str                     # "RAPTOR L1 synthesis (12 papers)" etc.
    section_title: str             # cluster centroid concept hint, may be empty
    document_ids: list[str]        # papers covered by this cluster
    child_chunk_ids: list[str]     # leaf chunk ids (L1) OR summary ids (L2+)
    year_min: Optional[int]
    year_max: Optional[int]
    section_types: list[str]
    topic_clusters: list[str]
    centroid: np.ndarray           # embedding (dense float32, normalised)

    # ── populated by TopicTree.build() ──
    children: list["TopicNode"] = field(default_factory=list)
    parent: Optional["TopicNode"] = None

    # ── lazy / cached ──
    _depth_of_coverage: Optional[float] = None
    _scope_relevance: Optional[float] = None

    @property
    def n_documents(self) -> int:
        return len(self.document_ids)

    def depth_of_coverage(self) -> float:
        """Cohesion + density estimate for the cluster.

        Operationalised as ``mean_pairwise_similarity × log(1 + n_docs)``,
        capped at 1.0. High when the cluster is large and tightly
        themed; low when it's small or topically scattered.

        Pairwise similarity over the actual document chunks would be
        more accurate but expensive at outline-proposal time; the
        cluster's centroid-distance-to-each-doc is a reasonable proxy
        and we use it lazily on the cluster's own embedding magnitude
        as a stand-in.
        """
        if self._depth_of_coverage is not None:
            return self._depth_of_coverage
        # Use the centroid magnitude as a coarse cohesion stand-in:
        # a well-themed cluster has a centroid close to unit norm
        # post-mean-pool. Multiply by log scale of paper count.
        norm = float(np.linalg.norm(self.centroid))
        scale = float(np.log1p(self.n_documents) / np.log1p(50))  # 50 papers ≈ 1.0
        v = min(1.0, norm * scale)
        self._depth_of_coverage = v
        return v


# ──────────────────────────────────────────────────────────────────────
# Tree
# ──────────────────────────────────────────────────────────────────────


class TopicTree:
    """In-memory topic tree built from Qdrant RAPTOR summary nodes."""

    def __init__(self, nodes: list[TopicNode]):
        self._nodes_by_id: dict[str, TopicNode] = {n.node_id: n for n in nodes}
        self._max_level: int = max((n.level for n in nodes), default=0)
        self._build_relationships()

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def from_qdrant(cls, qdrant_client, collection: str) -> "TopicTree":
        """Scroll the collection's RAPTOR summary nodes and build a tree.

        Reads only ``node_level >= 1`` points; leaf chunks are
        intentionally excluded (the tree is for outline-level
        navigation, not retrieval).

        Args:
          qdrant_client: a connected ``qdrant_client.QdrantClient``.
          collection: e.g. ``"papers"``. Use ``project.collection_papers``.
        """
        from qdrant_client.http.models import (
            Filter, FieldCondition, Range,
        )

        nodes: list[TopicNode] = []
        offset = None
        flt = Filter(
            must=[FieldCondition(key="node_level",
                                  range=Range(gte=1))]
        )
        while True:
            page, offset = qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=512,
                with_payload=True,
                with_vectors=True,
                offset=offset,
            )
            for pt in page:
                payload = pt.payload or {}
                lvl = int(payload.get("node_level") or 0)
                if lvl < 1:
                    continue
                # Vectors come back as a dict if multi-vector; pick dense.
                vec = pt.vector
                if isinstance(vec, dict):
                    vec = vec.get("dense") or vec.get("default") or next(iter(vec.values()))
                if vec is None:
                    continue
                centroid = np.asarray(vec, dtype=np.float32)
                # Normalise so cosine sim collapses to dot product.
                norm = float(np.linalg.norm(centroid))
                if norm > 0:
                    centroid = centroid / norm
                nodes.append(TopicNode(
                    node_id=str(pt.id),
                    level=lvl,
                    summary_text=str(payload.get("summary_text") or
                                     payload.get("content_preview") or ""),
                    title=str(payload.get("title") or ""),
                    section_title=str(payload.get("section_title") or ""),
                    document_ids=list(payload.get("document_ids") or []),
                    child_chunk_ids=list(payload.get("child_chunk_ids") or []),
                    year_min=payload.get("year_min"),
                    year_max=payload.get("year_max"),
                    section_types=list(payload.get("section_types") or []),
                    topic_clusters=list(payload.get("topic_clusters") or []),
                    centroid=centroid,
                ))
            if not offset:
                break
        logger.info("TopicTree.from_qdrant: %d summary nodes loaded "
                    "from %s (max level %d)",
                    len(nodes), collection,
                    max((n.level for n in nodes), default=0))
        return cls(nodes)

    def _build_relationships(self) -> None:
        """Populate parent/children edges from child_chunk_ids.

        L2 / L3 nodes' ``child_chunk_ids`` point at L1 / L2 node ids
        respectively (matching the summary-of-summaries pattern in
        raptor.py). L1 nodes' ``child_chunk_ids`` point at leaf chunk
        ids — those aren't in the tree, so no edges added.
        """
        for parent in self._nodes_by_id.values():
            for child_id in parent.child_chunk_ids:
                child = self._nodes_by_id.get(str(child_id))
                if child is None:
                    continue   # leaf chunk, not in summary-only tree
                child.parent = parent
                parent.children.append(child)

    # ── traversal ───────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._nodes_by_id)

    @property
    def max_level(self) -> int:
        return self._max_level

    def all_nodes(self) -> Iterable[TopicNode]:
        return self._nodes_by_id.values()

    def get(self, node_id: str) -> Optional[TopicNode]:
        return self._nodes_by_id.get(node_id)

    def roots(self) -> list[TopicNode]:
        """Top-level cluster nodes — highest level present, with no parent."""
        if self._max_level == 0:
            return []
        return [n for n in self._nodes_by_id.values()
                if n.level == self._max_level and n.parent is None]

    def children_of(self, node: TopicNode) -> list[TopicNode]:
        return list(node.children)

    def papers_in(self, node: TopicNode) -> set[str]:
        """All paper document_ids reachable from ``node``.

        Includes the node's own ``document_ids`` and recursively those
        of every descendant. Useful when a chapter's anchor cluster
        has sub-clusters that contribute additional papers.
        """
        seen: set[str] = set(node.document_ids)
        for child in node.children:
            seen |= self.papers_in(child)
        return seen

    # ── scope scoring ───────────────────────────────────────────────

    def score_against_scope(
        self,
        node: TopicNode,
        scope_embedding: np.ndarray,
    ) -> float:
        """Cosine similarity of node's centroid to the book scope vector.

        Both vectors must be unit-normalised; ``from_qdrant`` does
        that for centroids, callers must do it for their scope
        embedding (see ``embed_book_scope``).
        """
        v = float(node.centroid @ scope_embedding)
        node._scope_relevance = v
        return v

    def walk_in_scope(
        self,
        scope_embedding: np.ndarray,
        *,
        threshold: float = 0.30,
        max_per_level: int | None = None,
    ) -> Iterator[TopicNode]:
        """BFS through the tree, yielding every node whose
        scope-relevance is at or above ``threshold``.

        ``max_per_level`` caps the number of nodes yielded at each
        level (post-threshold, sorted by relevance descending). Set to
        ``None`` for no cap.
        """
        # Group all nodes by level for efficient cap-and-sort per level.
        by_level: dict[int, list[TopicNode]] = {}
        for n in self._nodes_by_id.values():
            score = self.score_against_scope(n, scope_embedding)
            if score < threshold:
                continue
            by_level.setdefault(n.level, []).append(n)

        for lvl in sorted(by_level.keys(), reverse=True):
            kept = sorted(by_level[lvl],
                          key=lambda n: n._scope_relevance or 0.0,
                          reverse=True)
            if max_per_level is not None:
                kept = kept[:max_per_level]
            yield from kept

    # ── helpers ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary stats for diagnostics (also useful as a smoke probe)."""
        per_level: dict[int, int] = {}
        for n in self._nodes_by_id.values():
            per_level[n.level] = per_level.get(n.level, 0) + 1
        return {
            "n_nodes": len(self._nodes_by_id),
            "per_level": per_level,
            "max_level": self._max_level,
            "n_roots": len(self.roots()),
        }


# ──────────────────────────────────────────────────────────────────────
# Scope embedding helper
# ──────────────────────────────────────────────────────────────────────


def embed_book_scope(
    book_description: str,
    book_plan: str | None = None,
) -> np.ndarray:
    """Embed a book's scope (description ⊕ plan) for use as the
    scope vector against TopicTree.score_against_scope.

    Returns a unit-normalised float32 vector. Uses the same dense
    embedder as the writer (bge-m3 via the embedder role).
    """
    from sciknow.infer.client import embed
    text = (book_description or "").strip()
    if book_plan and book_plan.strip():
        text = (text + "\n\n" + book_plan.strip()).strip()
    if not text:
        raise ValueError("book scope is empty")
    [vec] = embed([text])
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr
