# Knowledge Graph — design research (Phase 48)

Scope: node-link visualization for entity-relationship triples extracted
from our corpus. Current impl (Phase 48) is pure SVG + vanilla JS with a
3D force simulation, orbit camera, drag-and-zoom, and a theme-preset row.
This doc captures the research that justifies what's shipped and what's
deferred — written so the next session can pick the right next move
without re-running the analysis.

## Status (Phase 48b)

### Shipped
- 3D orbit camera + drag-to-reposition + wheel zoom (Phase 48)
- Seven theme presets + one-click Invert (Phase 48)
- **Louvain community detection**, per-cluster Okabe-Ito coloring, and
  per-community gravity wells that pre-position nodes
- **ForceAtlas2-derived physics**: log-weighted attraction (linLog),
  degree-scaled repulsion (dissuade-hubs), weak origin centering
- Duplicate-triple merging: same (s, t) pair → one curved edge with a
  count badge and a Set of predicate families
- Curved quadratic-Bézier edges with bundle offsets so A→B and B→A (or
  multiple same-direction edges) don't overlap
- **Hover-dim + 1-hop highlight**: dim non-neighbors on node hover
- **Right-click context menu** for node / edge / background — pin,
  hide, ego-expand (via new `any_side` API param), center view,
  copy label/triple, show source paper, filter by predicate
- **Live search box** that dims non-matches and tween-centers on the
  first match
- **Spacebar freeze** / resume of the physics loop
- **Center-on-click camera tween** (cubic ease-in-out)
- **PNG export** (SVG → canvas → 2× retina download)
- **Color-by dropdown**: cluster / predicate family / plain (theme edge)
- **Label size slider** + **Max-degree slider** (hides hub nodes above N)
- **Predicate family coloring** (causal / measurement / taxonomic /
  compositional / citational / other) using Okabe-Ito colorblind-safe
  categorical palette

### Still in the backlog (in rough priority order)
1. **"Explain this edge" popover** with the source *sentence* (not just
   paper title). Requires schema migration to store sentence offsets
   at extraction time — biggest feature remaining, biggest lift.
2. **Cached layout per filter** (localStorage by filter-hash) for warm
   starts when users alternate filters.
3. **Shareable URL** encoding camera + filter + pinned set in the hash.
4. **Ego expansion at depth ≥ 2** (currently depth 1).
5. **Barnes-Hut** only once we hit n ≈ 800 or profile shows > 8 ms/frame.
6. **Lasso selection**, **keyboard graph walk**, **minimap** — all
   explicitly parked as low-ROI for this tool.

## Axis-by-axis notes

### 1. Graph structure / readability
- **Edge bundling** (Holten 2006; Holten & van Wijk 2009): clutter-reducer,
  but only pays off above ~500 edges and hides which exact pair connects.
  **Skip** for triple browsing.
- **Community detection**: Louvain (Blondel 2008) → color + per-cluster
  gravity wells. **Implement.** Leiden (Traag 2019) fixes Louvain's
  disconnected-community bug — use only if a Louvain cluster is visibly
  broken.
- **Ego network extraction at depth N**: right-click a node → "expand
  depth 1/2". **Implement** — most-used affordance in every KG tool that
  has it.
- **Hub handling**: scientific KGs have vicious hubs ("temperature",
  "model"). Options: (a) collapse > K edges per hub into a "+27 more" stub
  that expands on click (Scholia's trick); (c) a degree-percentile slider
  hiding top-N% hubs. **Implement (a)+(c); skip anti-gravity hubs
  — it fights users' mental model.**
- **Curved edges + bidirectional merging**: quadratic Bézier with small
  perpendicular offset per (u,v) pair so A→B and B→A don't overlap; merge
  parallel same-predicate edges into one thicker edge with a count badge.
  **Implement both.**
- **Arrowheads**: SVG `<marker>`; always show for directed predicates.
- **Edge labels**: always-on is unreadable above ~40 edges. Use hover-only +
  a global "show all labels" toggle. **Implement.**
- **Focus+context / fisheye** (Sarkar & Brown 1992): disorienting in 3D
  with an orbit camera. **Skip.**

### 2. Interaction
- **Hover highlight with dim**: set opacity 0.15 on non-neighbors.
  **Implement first.**
- **Search box with live highlight**: debounced substring match; pulse
  matches; auto-center camera on first match. **Implement.**
- **Right-click context menu**: "Expand neighbors / Pin / Hide / Open source
  paper / Copy triple". **Implement.**
- **Double-click → open source paper**: minimum "why is this edge here"
  affordance. **Implement.**
- **Pin-on-drag + shift-click-to-unpin**: Gephi idiom; users discover it
  immediately. **Implement.** (Our current impl releases pin on mouseup —
  easy to flip.)
- **Center-on-node animation**: 400ms ease-in-out-cubic tween of camera
  target. **Implement.**
- **Keyboard walk, lasso selection, undo stack**: **Skip** unless there's
  an explicit a11y or power-user ask.

### 3. Layout & physics
- **ForceAtlas2**: port of Jacomy 2014, ~150 lines JS. `linLogMode=true`
  + `strongGravityMode=false` wins on real citation data.
- **Edge weight = log(1 + triple_count)** — prevents a single high-count
  edge from collapsing the graph.
- **Per-cluster gravity wells** — weak attractor at each Louvain centroid.
  Clusters become visually obvious.
- **Physics freeze** (spacebar) — non-negotiable. Users need this the
  moment they want to read labels.
- **Cached layout per filter** — hash filter state → localStorage node
  positions → warm-start next time. Big perceived-perf win.
- **Barnes-Hut (θ≈0.9)**: only matters above n≈800. **Skip until profiled.**

### 4. Prior art for scientific KGs
- **Connected Papers**: co-citation similarity, force layout, year-gradient
  node color, size by citations, *no predicates*. For a pure "which papers
  relate" view, predicates are noise.
- **Scholia** (Wikidata-backed): typed edges with inline labels; readable
  up to ~30 edges then collapses. Confirms: always-on predicate labels
  don't scale.
- **Inciteful / Litmaps / ResearchRabbit**: all primary-navigate via ego
  expansion. *Expansion beats global view.* A "start from one node, grow
  by click" flow may beat showing top-100 triples unfiltered.
- **OpenAlex graph explorer**: color by concept → maps cleanly to Louvain
  clusters.
- **Predicate coloring** — use a *categorical* palette keyed by predicate
  *family*, not per-predicate:
  - causal (CAUSES, INCREASES, DECREASES) → warm reds/oranges
  - measurement (MEASURES, OBSERVES) → blues
  - taxonomic (IS-A, PART-OF) → greens
  - citational (CITES, EVIDENCE-FOR) → gray
  RDF-viz convention (VOWL / WebVOWL). Cap at ~8 distinguishable hues;
  group rarer predicates into "other" gray.

### 5. Performance at 100–500 nodes
- n=100: O(n²) = 10k pair computations, negligible.
- n=500: 250k/frame ≈ 5ms JS, fine.
- **Stay with O(n²) until n≈800 or profiler shows >8ms/frame.**
- When it's time: Barnes-Hut quadtree (2D) / octree (3D) with θ=0.9.
  Refs: Barnes & Hut 1986 (Nature); Mike Bostock's `d3-force-3d`.
- SVG itself caps out around ~2000 visible elements before GC pauses.
  **Cap rendered nodes at 500** and paginate/filter above that.

### 6. Colorblind-safe palettes
- **Okabe-Ito (8 colors)** — gold standard for categorical; designed for
  protanopia/deuteranopia/tritanopia.
  `#000000 #E69F00 #56B4E9 #009E73 #F0E442 #0072B2 #D55E00 #CC79A7`.
  Use for **predicate families**.
- **ColorBrewer Set2** — good for cluster coloring up to 12 categories,
  muted, works on dark backgrounds.
- **Viridis** — continuous, perceptually uniform, colorblind-safe. Use for
  year-of-publication or confidence gradients.
- Dark vs light: on dark backgrounds, saturate less / lighten more
  (Okabe-Ito yellow becomes blinding — use Paul Tol's "bright" or
  "vibrant" variants, SRON tech note 2021).
- Never encode with color alone: pair **color + glyph + dash pattern**
  for predicate type.

### 7. Tiny features that punch above their weight
- **Freeze/unfreeze (spacebar)** — ship today.
- **Download PNG**: serialize SVG → canvas → `toBlob`. ~20 lines.
- **Shareable URL with camera + filter + pinned state** (base64 in hash).
  Zero server change.
- **FPS counter** gated on `?debug=1`.
- **Label-size slider** (10–20 px), localStorage-persisted.
- **Degree-based node size** — already present; switch to `radius = base +
  k·√degree` rather than linear to prevent hub gigantism.
- **Minimap in 3D**: confusing (which projection?). **Skip.**

## Sources
- Holten 2006 (InfoVis); Holten & van Wijk 2009 (EuroVis).
- Blondel et al. 2008 (J. Stat. Mech.) — Louvain.
- Traag et al. 2019 (Sci. Reports) — Leiden.
- Jacomy et al. 2014 (PLOS ONE) — ForceAtlas2.
- Noack 2009 — LinLog.
- Barnes & Hut 1986 (Nature).
- Sarkar & Brown 1992 (CHI) — fisheye.
- Okabe & Ito 2008 (J*FLY); Paul Tol SRON/EPS/TN/09-002 (2021).
- Lohmann et al. WebVOWL (SWJ 2016).
- Connected Papers blog (Eitan & Mordo, 2020).
- Pike, Stasko, Chang, O'Connell 2009 (Info. Vis.) — sense-making.
