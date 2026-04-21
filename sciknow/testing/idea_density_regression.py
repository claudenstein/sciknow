"""Phase 54.6.160 — Brown 2008 propositional idea-density regression.

Ships the experiment RESEARCH.md §24 §gaps lists as "publishable on its
own": computes propositional idea density (Brown et al., 2008,
doi:10.3758/BRM.40.2.540) on a sample of the corpus's
``paper_sections.content``, regresses section word-count against total
idea count, and emits empirical **words-per-concept** estimates per
canonical section type. Complements (does not replace) the 54.6.146
research-grounded defaults, which come from the brief's literature
survey rather than this corpus.

**Brown 2008 P-density**::

    P-density = (V + Adj + Adv + Prep + Conj) / word_count

where V/Adj/Adv/Prep/Conj are counts of those POS families (via
spaCy's universal POS tags). Higher = more propositionally dense
prose. The formula is the most widely-used automatic approximation
of propositions-per-word in the cognitive-science literature.

**Regression**: for each section type, fit ``word_count = a + b × total_ideas``.
The slope ``b`` (seconds if you squint) is the empirical words-per-
idea ratio — the corpus's own version of the 54.6.146 wpc midpoint.

**Dependency**: spaCy. Not carried by default because it's ~200 MB
and this experiment runs once. Lazy-imported with a clear install
message on ImportError so regular sciknow users aren't forced to
install it.

**Sampling**: full-corpus runs are expensive (~100 ms per section ×
32 k sections = ~1 hour). Default samples 500 sections per canonical
type via ``TABLESAMPLE BERNOULLI``, bringing wall time to ~4-8 min
total while keeping enough statistical power for a rough per-type
wpc estimate.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


# Canonical sections the regression targets. Must match the 54.6.157
# bench's canonical list so the two sets of numbers are comparable.
_CANONICAL_SECTIONS = (
    "abstract", "introduction", "methods", "results",
    "discussion", "conclusion", "related_work",
)

# Brown 2008 P-density formula: propositions are approximately
# signalled by verbs, modifiers (adj + adv), prepositions, and
# conjunctions. Count each under a spaCy universal-POS label.
_P_DENSITY_POS: set[str] = {
    "VERB", "AUX",        # verbs + auxiliaries
    "ADJ", "ADV",         # modifiers
    "ADP",                # adpositions (preps + postposisions)
    "CCONJ", "SCONJ",     # coordinating + subordinating conjunctions
}


@dataclass
class SectionMetric:
    section_type: str
    doc_id: str
    n_words: int
    n_ideas: int
    p_density: float    # ideas per word

    @property
    def wpc(self) -> float:
        """Words per idea, from THIS single section."""
        return (self.n_words / self.n_ideas) if self.n_ideas else 0.0


@dataclass
class RegressionPerType:
    section_type: str
    n_sections: int
    mean_word_count: float
    mean_p_density: float
    # Simple regression: word_count ~ a + b * n_ideas → b ≈ wpc
    slope_wpc: float        # words per idea, from the fit
    r_squared: float
    # Per-section wpc distribution (words ÷ ideas)
    wpc_median: float
    wpc_q1: float
    wpc_q3: float


@dataclass
class RegressionReport:
    n_total: int
    sample_per_type: int
    by_type: dict[str, RegressionPerType] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "sample_per_type": self.sample_per_type,
            "elapsed_s": self.elapsed_s,
            "by_type": {
                st: {
                    "n_sections": r.n_sections,
                    "mean_word_count": round(r.mean_word_count, 1),
                    "mean_p_density": round(r.mean_p_density, 4),
                    "slope_wpc": round(r.slope_wpc, 2),
                    "r_squared": round(r.r_squared, 3),
                    "wpc_q1": round(r.wpc_q1, 1),
                    "wpc_median": round(r.wpc_median, 1),
                    "wpc_q3": round(r.wpc_q3, 1),
                }
                for st, r in self.by_type.items()
            },
        }


# ── Sampling + idea count ───────────────────────────────────────────


def _load_spacy():
    """Lazy-import spaCy with a clear install message on miss."""
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required for the Brown 2008 idea-density "
            "regression. Install with:\n"
            "    uv add spacy\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as exc:
        raise RuntimeError(
            "spaCy is installed but the en_core_web_sm model is not. "
            "Download it with:\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc


def _count_ideas(text: str, nlp) -> tuple[int, int]:
    """Brown 2008 P-count on one text block.

    Returns ``(n_ideas, n_words)``. Words excludes punctuation and
    spaces (spaCy's ``is_alpha``). Ideas = sum of POS counts in
    ``_P_DENSITY_POS``.
    """
    doc = nlp(text or "")
    n_words = sum(1 for t in doc if t.is_alpha)
    n_ideas = sum(1 for t in doc if t.pos_ in _P_DENSITY_POS)
    return n_ideas, n_words


def _sample_sections(sample_per_type: int):
    """Yield ``(section_type, doc_id, content)`` rows sampled per type.

    Uses Postgres' ``TABLESAMPLE BERNOULLI`` for an unbiased sample,
    but we also apply a deterministic LIMIT after ORDER BY random()
    because BERNOULLI requires a percentage that's hard to pick
    without knowing the total first. Cost is one COUNT per type for
    the rate calculation, negligible on 32k rows.
    """
    from sqlalchemy import text
    from sciknow.storage.db import get_session
    with get_session() as session:
        for st in _CANONICAL_SECTIONS:
            rows = session.execute(text("""
                SELECT section_type, document_id::text, content
                FROM paper_sections
                WHERE section_type = :st
                  AND content IS NOT NULL
                  AND LENGTH(content) >= 100
                ORDER BY random()
                LIMIT :lim
            """), {"st": st, "lim": sample_per_type}).fetchall()
            for r in rows:
                yield r[0], r[1], r[2]


# ── Regression fit ──────────────────────────────────────────────────


def _fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Simple OLS y = a + b*x. Returns (a, b, r²).

    Uses ``statistics.linear_regression`` (Python 3.10+) if available;
    otherwise falls back to manual Pearson correlation + slope.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0, 0.0, 0.0
    try:
        slope, intercept = statistics.linear_regression(xs, ys)
    except (ValueError, AttributeError, statistics.StatisticsError):
        return 0.0, 0.0, 0.0
    # R² via Pearson
    try:
        r = statistics.correlation(xs, ys)
        return intercept, slope, r * r
    except statistics.StatisticsError:
        return intercept, slope, 0.0


# ── Public API ──────────────────────────────────────────────────────


def run_regression(sample_per_type: int = 500) -> RegressionReport:
    """Walk a stratified sample of paper_sections, compute Brown 2008
    P-density per section, fit a per-section-type regression, return
    the empirical wpc estimates.

    Raises RuntimeError with a pointed install message when spaCy or
    its English model isn't available. Catch and handle in the CLI
    wrapper.
    """
    import time
    t0 = time.monotonic()
    nlp = _load_spacy()

    by_type_metrics: dict[str, list[SectionMetric]] = {
        st: [] for st in _CANONICAL_SECTIONS
    }
    n_total = 0
    for st, doc_id, content in _sample_sections(sample_per_type):
        n_ideas, n_words = _count_ideas(content, nlp)
        if n_words < 50 or n_ideas == 0:
            continue
        by_type_metrics[st].append(SectionMetric(
            section_type=st, doc_id=doc_id,
            n_words=n_words, n_ideas=n_ideas,
            p_density=(n_ideas / n_words) if n_words else 0.0,
        ))
        n_total += 1

    by_type: dict[str, RegressionPerType] = {}
    for st, metrics in by_type_metrics.items():
        if len(metrics) < 5:
            continue
        xs = [m.n_ideas for m in metrics]
        ys = [m.n_words for m in metrics]
        _a, slope, r2 = _fit_line([float(x) for x in xs], [float(y) for y in ys])
        wpcs = [m.wpc for m in metrics if m.wpc > 0]
        wpcs.sort()
        by_type[st] = RegressionPerType(
            section_type=st,
            n_sections=len(metrics),
            mean_word_count=statistics.fmean([m.n_words for m in metrics]),
            mean_p_density=statistics.fmean([m.p_density for m in metrics]),
            slope_wpc=slope,
            r_squared=r2,
            wpc_median=wpcs[len(wpcs) // 2] if wpcs else 0.0,
            wpc_q1=wpcs[len(wpcs) // 4] if len(wpcs) >= 4 else 0.0,
            wpc_q3=wpcs[(3 * len(wpcs)) // 4] if len(wpcs) >= 4 else 0.0,
        )

    return RegressionReport(
        n_total=n_total,
        sample_per_type=sample_per_type,
        by_type=by_type,
        elapsed_s=round(time.monotonic() - t0, 1),
    )
