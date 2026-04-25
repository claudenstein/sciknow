"""sciknow v2 — typed event schema for the SSE/CLI/MCP wire protocol.

Spec §5.3: every generator under ``sciknow.core.book_ops`` /
``sciknow.core.wiki_ops`` yields events tagged with ``"type"``. v2
formalises that into a Pydantic discriminated union so:

* The web SSE serialiser can call ``event.model_dump_json()`` without
  guessing keys.
* The CLI Rich renderer can switch on ``event.type`` with an explicit
  match expression.
* The contract test in ``sciknow/testing/contracts/`` can assert that
  every yielded dict is parseable as ``SciknowEvent``.

The union is **permissive** by design (``extra="allow"`` on the base
class) so the migration from ad-hoc dicts to Pydantic instances can
land incrementally — call sites that still emit raw dicts continue to
validate, and migration to ``model_dump`` happens per-call-site.

Author note (2026-04-25): the v1 enumeration scraped from book_ops +
wiki_ops sources lists 32 distinct event tags. v2 keeps every tag the
roadmap calls out as "the 16 autowrite/book SSE protocol events"
(§1.4) plus the 16 wiki-only tags. New event tags MUST be added here
first — that's the spec's "test the contract, not the implementation"
discipline.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class _BaseEvent(BaseModel):
    """Permissive parent: extra keys are allowed during the v1→v2 grace
    period so partially-typed yields still validate."""
    model_config = ConfigDict(extra="allow")


# ── streaming primitives ────────────────────────────────────────────────


class TokenEvent(_BaseEvent):
    type: Literal["token"]
    text: str
    phase: str | None = None


class ProgressEvent(_BaseEvent):
    type: Literal["progress"]
    stage: str | None = None
    detail: str | None = None
    pct: float | None = None
    tokens: int | None = None
    eta_seconds: float | None = None


class ErrorEvent(_BaseEvent):
    type: Literal["error"]
    message: str
    code: str | None = None


class CompletedEvent(_BaseEvent):
    type: Literal["completed"]
    draft_id: str | None = None
    feedback: Any = None


# ── autowrite scoring + verification ────────────────────────────────────


class ScoresEvent(_BaseEvent):
    """Scorer rubric output — one per autowrite iteration."""
    type: Literal["scores"]
    iteration: int | None = None
    grounded: float | None = None
    coherent: float | None = None
    complete: float | None = None
    concise: float | None = None
    rationale: str | None = None
    scores: dict | None = None  # legacy aggregate field


class VerificationEvent(_BaseEvent):
    """Atomic-claim verification result. Holds CoVe decisions when batched
    (v1 phase 54.6.319 batched the per-question CoV calls)."""
    type: Literal["verification"]
    iteration: int | None = None
    grounded: float | None = None
    data: Any = None
    cove_decisions: list | None = None


class CoveVerificationEvent(_BaseEvent):
    type: Literal["cove_verification"]
    data: Any = None


class IterationStartEvent(_BaseEvent):
    type: Literal["iteration_start"]
    iteration: int | None = None


class ConvergedEvent(_BaseEvent):
    type: Literal["converged"]
    reason: str | None = None
    iteration: int | None = None


# ── plan / outline ──────────────────────────────────────────────────────


class PlanEvent(_BaseEvent):
    type: Literal["plan"]
    content: str | None = None


class TreePlanEvent(_BaseEvent):
    type: Literal["tree_plan"]
    data: Any = None
    raw: str | None = None


class PlanCoverageEvent(_BaseEvent):
    type: Literal["plan_coverage"]
    data: Any = None


class LengthTargetEvent(_BaseEvent):
    type: Literal["length_target"]
    target_words: int | None = None


class ModelInfoEvent(_BaseEvent):
    type: Literal["model_info"]
    writer_model: str | None = None
    scorer_model: str | None = None


# ── checkpointing / persistence ─────────────────────────────────────────


class CheckpointEvent(_BaseEvent):
    type: Literal["checkpoint"]
    draft_id: str | None = None
    stage: str | None = None
    content: str | None = None


class LogPathEvent(_BaseEvent):
    type: Literal["log_path"]
    path: str | None = None


# ── citations ───────────────────────────────────────────────────────────


class CitationAlignEvent(_BaseEvent):
    type: Literal["citation_align"]


class CitationCandidatesEvent(_BaseEvent):
    type: Literal["citation_candidates"]


class CitationNeedsEvent(_BaseEvent):
    type: Literal["citation_needs"]


class CitationInsertedEvent(_BaseEvent):
    type: Literal["citation_inserted"]


class CitationSelectedEvent(_BaseEvent):
    type: Literal["citation_selected"]


class CitationSkippedEvent(_BaseEvent):
    type: Literal["citation_skipped"]


class FindingEvent(_BaseEvent):
    type: Literal["finding"]
    data: Any = None


# ── meta-review / wiki ──────────────────────────────────────────────────


class MetaReviewStartEvent(_BaseEvent):
    type: Literal["meta_review_start"]


class CompileStartEvent(_BaseEvent):
    type: Literal["compile_start"]


class PaperStartEvent(_BaseEvent):
    type: Literal["paper_start"]


class PaperDoneEvent(_BaseEvent):
    type: Literal["paper_done"]


class ReviewerDoneEvent(_BaseEvent):
    type: Literal["reviewer_done"]


class ConsensusEvent(_BaseEvent):
    type: Literal["consensus"]


class LintIssueEvent(_BaseEvent):
    type: Literal["lint_issue"]


class RevisionVerdictEvent(_BaseEvent):
    type: Literal["revision_verdict"]


class VisualCitationEvent(_BaseEvent):
    type: Literal["visual_citation"]


# ── chapter-autowrite + section-level events ────────────────────────────


class ChapterAutowriteStartEvent(_BaseEvent):
    type: Literal["chapter_autowrite_start"]
    chapter_id: str | None = None


class AllSectionsCompleteEvent(_BaseEvent):
    type: Literal["all_sections_complete"]


class SectionStartEvent(_BaseEvent):
    type: Literal["section_start"]
    section: str | None = None


class SectionDoneEvent(_BaseEvent):
    type: Literal["section_done"]
    section: str | None = None


class SectionErrorEvent(_BaseEvent):
    type: Literal["section_error"]
    section: str | None = None
    message: str | None = None


class SectionLengthWarningEvent(_BaseEvent):
    type: Literal["section_length_warning"]
    section: str | None = None


class RefinementGateEvent(_BaseEvent):
    type: Literal["refinement_gate"]


class ResumeInfoEvent(_BaseEvent):
    type: Literal["resume_info"]


class RetrievalDensityAdjustEvent(_BaseEvent):
    type: Literal["retrieval_density_adjust"]


class LintSummaryEvent(_BaseEvent):
    type: Literal["lint_summary"]


# ── union (discriminator: "type") ───────────────────────────────────────

SciknowEvent = Annotated[
    Union[
        TokenEvent, ProgressEvent, ErrorEvent, CompletedEvent,
        ScoresEvent, VerificationEvent, CoveVerificationEvent,
        IterationStartEvent, ConvergedEvent,
        PlanEvent, TreePlanEvent, PlanCoverageEvent,
        LengthTargetEvent, ModelInfoEvent,
        CheckpointEvent, LogPathEvent,
        CitationAlignEvent, CitationCandidatesEvent, CitationNeedsEvent,
        CitationInsertedEvent, CitationSelectedEvent, CitationSkippedEvent,
        FindingEvent,
        MetaReviewStartEvent,
        CompileStartEvent, PaperStartEvent, PaperDoneEvent,
        ReviewerDoneEvent, ConsensusEvent, LintIssueEvent,
        RevisionVerdictEvent, VisualCitationEvent,
        ChapterAutowriteStartEvent, AllSectionsCompleteEvent,
        SectionStartEvent, SectionDoneEvent, SectionErrorEvent,
        SectionLengthWarningEvent, RefinementGateEvent,
        ResumeInfoEvent, RetrievalDensityAdjustEvent, LintSummaryEvent,
    ],
    Field(discriminator="type"),
]


KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    # Keep this in sync with the union above. Used by tests + the SSE
    # filter to silently drop legacy/unknown event types instead of
    # leaking them onto the wire.
    "token", "progress", "error", "completed",
    "scores", "verification", "cove_verification",
    "iteration_start", "converged",
    "plan", "tree_plan", "plan_coverage",
    "length_target", "model_info",
    "checkpoint", "log_path",
    "citation_align", "citation_candidates", "citation_needs",
    "citation_inserted", "citation_selected", "citation_skipped",
    "finding",
    "meta_review_start",
    "compile_start", "paper_start", "paper_done",
    "reviewer_done", "consensus", "lint_issue",
    "revision_verdict", "visual_citation",
    "chapter_autowrite_start", "all_sections_complete",
    "section_start", "section_done", "section_error",
    "section_length_warning", "refinement_gate",
    "resume_info", "retrieval_density_adjust", "lint_summary",
})


def parse_event(raw: dict) -> _BaseEvent:
    """Validate ``raw`` against the union; raise pydantic.ValidationError
    on unknown ``type``. Used by contract tests; production code can
    keep emitting raw dicts during the migration."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(SciknowEvent)
    return adapter.validate_python(raw)


__all__ = [
    "SciknowEvent",
    "KNOWN_EVENT_TYPES",
    "parse_event",
    # Concrete classes (export for callers that want to construct them).
    "TokenEvent", "ProgressEvent", "ErrorEvent", "CompletedEvent",
    "ScoresEvent", "VerificationEvent", "CoveVerificationEvent",
    "IterationStartEvent", "ConvergedEvent",
    "PlanEvent", "TreePlanEvent", "PlanCoverageEvent",
    "LengthTargetEvent", "ModelInfoEvent",
    "CheckpointEvent", "LogPathEvent",
    "CitationAlignEvent", "CitationCandidatesEvent", "CitationNeedsEvent",
    "CitationInsertedEvent", "CitationSelectedEvent", "CitationSkippedEvent",
    "FindingEvent",
    "MetaReviewStartEvent",
    "CompileStartEvent", "PaperStartEvent", "PaperDoneEvent",
    "ReviewerDoneEvent", "ConsensusEvent", "LintIssueEvent",
    "RevisionVerdictEvent", "VisualCitationEvent",
    "ChapterAutowriteStartEvent", "AllSectionsCompleteEvent",
    "SectionStartEvent", "SectionDoneEvent", "SectionErrorEvent",
    "SectionLengthWarningEvent", "RefinementGateEvent",
    "ResumeInfoEvent", "RetrievalDensityAdjustEvent", "LintSummaryEvent",
]
