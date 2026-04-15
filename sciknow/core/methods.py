"""Phase 54.6.14 — method catalogues adapted from BMAD-METHOD (MIT).

Two small collections of named cognitive methods that can be
injected as a preamble to existing LLM prompts, steering the model's
approach without rewriting the whole prompt:

* **Elicitation methods** (24) — structured ways to pull deeper
  analysis / critique out of a draft or proposal. Used by the book
  plan / outline generators so the user can say "generate this via
  Tree of Thoughts" instead of a generic request.

* **Brainstorming methods** (24) — structured ways to generate new
  angles on a problem. Used by the book-gaps flow and the Visualize
  modal's consensus tab when the user wants a wider net.

The methods themselves are adapted verbatim from bmad-code-org/
BMAD-METHOD's src/core-skills/bmad-advanced-elicitation/methods.csv
and bmad-brainstorming/brain-methods.csv (MIT-licensed, 2024-25).

Usage:

    from sciknow.core.methods import get_method, list_methods
    m = get_method("elicitation", "Tree of Thoughts")
    preamble = method_preamble(m)  # inject into the LLM user prompt
"""
from __future__ import annotations

from typing import Any


# ── Elicitation methods (from BMAD advanced-elicitation) ────────────────

ELICITATION_METHODS: list[dict[str, str]] = [
    # Collaboration (multi-perspective)
    {"category": "collaboration", "name": "Stakeholder Round Table",
     "description": "Convene multiple personas to contribute diverse perspectives — "
                    "essential for requirements gathering and finding balanced "
                    "solutions across competing interests."},
    {"category": "collaboration", "name": "Expert Panel Review",
     "description": "Assemble domain experts for deep specialized analysis — ideal "
                    "when technical depth and peer review quality are needed."},
    {"category": "collaboration", "name": "Debate Club Showdown",
     "description": "Two personas argue opposing positions while a moderator scores "
                    "points — great for exploring controversial decisions and "
                    "finding middle ground."},
    {"category": "collaboration", "name": "Time Traveler Council",
     "description": "Past-you and future-you advise present-you on decisions — "
                    "powerful for gaining perspective on long-term consequences "
                    "vs short-term pressures."},
    {"category": "collaboration", "name": "Mentor and Apprentice",
     "description": "Senior expert teaches junior while junior asks naive questions "
                    "— surfaces hidden assumptions through teaching."},
    {"category": "collaboration", "name": "Good Cop Bad Cop",
     "description": "Supportive persona and critical persona alternate — finds "
                    "both strengths to build on and weaknesses to address."},
    # Advanced reasoning
    {"category": "advanced", "name": "Tree of Thoughts",
     "description": "Explore multiple reasoning paths simultaneously then "
                    "evaluate and select the best — perfect for complex problems "
                    "with multiple valid approaches."},
    {"category": "advanced", "name": "Graph of Thoughts",
     "description": "Model reasoning as an interconnected network of ideas to "
                    "reveal hidden relationships — ideal for systems thinking "
                    "and discovering emergent patterns."},
    {"category": "advanced", "name": "Self-Consistency Validation",
     "description": "Generate multiple independent approaches then compare for "
                    "consistency — crucial for high-stakes decisions where "
                    "verification matters."},
    {"category": "advanced", "name": "Meta-Prompting Analysis",
     "description": "Step back to analyze the approach structure and methodology "
                    "itself — valuable for optimizing prompts and improving "
                    "problem-solving."},
    {"category": "advanced", "name": "Reasoning via Planning",
     "description": "Build a reasoning tree guided by world models and goal "
                    "states — excellent for strategic planning and sequential "
                    "decision-making."},
    # Competitive / adversarial
    {"category": "competitive", "name": "Red Team vs Blue Team",
     "description": "Adversarial attack-defend analysis to find vulnerabilities "
                    "— critical for security testing and building robust "
                    "solutions."},
    {"category": "competitive", "name": "Pre-mortem",
     "description": "Imagine the project failed a year from now — work backwards "
                    "to identify the likely causes. Surfaces risks that optimism "
                    "bias normally hides."},
    {"category": "competitive", "name": "Shark Tank Pitch",
     "description": "Entrepreneur pitches to skeptical investors who poke holes "
                    "— stress-tests business viability and forces clarity on "
                    "value proposition."},
    # Critical / philosophical
    {"category": "critical", "name": "Socratic Method",
     "description": "Progressively deeper questions that push the author to "
                    "examine assumptions and reach clearer definitions. Each "
                    "answer triggers a sharper follow-up."},
    {"category": "critical", "name": "First Principles",
     "description": "Strip away assumptions to rebuild from fundamental truths "
                    "— essential for breakthrough innovation. 'What do we know "
                    "for certain? What are the fundamental truths?'"},
    {"category": "critical", "name": "Five Whys",
     "description": "Drill down through layers of causation to uncover root "
                    "causes — essential for solving problems at source rather "
                    "than symptoms."},
    {"category": "critical", "name": "Assumption Reversal",
     "description": "Challenge and flip core assumptions to rebuild from new "
                    "foundations — essential for paradigm shifts."},
    # Technical / architectural
    {"category": "technical", "name": "Architecture Decision Records",
     "description": "Propose and debate architectural / structural choices with "
                    "explicit trade-offs — ensures decisions are well-reasoned "
                    "and documented."},
    {"category": "technical", "name": "Rubber Duck Evolved",
     "description": "Explain the content to progressively more technical ducks "
                    "until you find the hole — forces clarity at multiple "
                    "abstraction levels."},
    {"category": "technical", "name": "Algorithm Olympics",
     "description": "Multiple approaches compete on the same problem with "
                    "benchmarks — finds optimal solution through direct "
                    "comparison."},
    # Scientific writing-specific
    {"category": "scientific", "name": "Peer Review Simulation",
     "description": "Imagine three referees (methodological, empirical, "
                    "conceptual) each write a review. What do they each flag? "
                    "What do they disagree on?"},
    {"category": "scientific", "name": "Strong Inference (Platt)",
     "description": "Devise an alternative hypothesis, design a crucial "
                    "experiment that would disprove each, and identify which "
                    "alternative the data currently rules out."},
    {"category": "scientific", "name": "Inference to the Best Explanation",
     "description": "List all plausible explanations for the observed pattern, "
                    "rank by prior probability × fit, and identify which "
                    "evidence would discriminate between them."},
]


# ── Brainstorming methods (from BMAD bmad-brainstorming) ────────────────

BRAINSTORMING_METHODS: list[dict[str, str]] = [
    # Collaborative
    {"category": "collaborative", "name": "Yes And Building",
     "description": "Build momentum through positive additions where each idea "
                    "becomes a launching pad — use 'Yes and we could also…' "
                    "to create energetic collaborative flow."},
    {"category": "collaborative", "name": "Brain Writing Round Robin",
     "description": "Silent idea generation followed by building on others' "
                    "written concepts — gives quieter voices equal contribution."},
    {"category": "collaborative", "name": "Random Stimulation",
     "description": "Use random words / images as creative catalysts to force "
                    "unexpected connections — breaks through mental blocks."},
    {"category": "collaborative", "name": "Role Playing",
     "description": "Generate solutions from multiple stakeholder perspectives — "
                    "embody different roles and ask what they want, how they'd "
                    "approach the problem, what matters most to them."},
    # Creative / lateral
    {"category": "creative", "name": "What If Scenarios",
     "description": "Explore radical possibilities by questioning all constraints "
                    "and assumptions — 'What if we had unlimited resources? "
                    "What if the opposite were true?'"},
    {"category": "creative", "name": "Analogical Thinking",
     "description": "Find creative solutions by drawing parallels to other "
                    "domains — 'This is like what? How is this similar to…?'"},
    {"category": "creative", "name": "Reversal Inversion",
     "description": "Deliberately flip problems upside down to reveal hidden "
                    "assumptions — 'What if we did the opposite? How could we "
                    "make this worse?'"},
    {"category": "creative", "name": "Forced Relationships",
     "description": "Connect unrelated concepts to spark innovative bridges — "
                    "take two unrelated things, find connections between them."},
    {"category": "creative", "name": "Time Shifting",
     "description": "Explore solutions across different time periods — 'How "
                    "would this work in 1900? 2100? What era-specific "
                    "constraints or opportunities apply?'"},
    {"category": "creative", "name": "Metaphor Mapping",
     "description": "Use extended metaphors as thinking tools — transforms "
                    "abstract challenges into tangible narratives. Map every "
                    "element to discover insights."},
    {"category": "creative", "name": "Cross-Pollination",
     "description": "Transfer solutions from completely different industries or "
                    "domains — 'How would a biologist solve this? What patterns "
                    "work in field Y?'"},
    {"category": "creative", "name": "Concept Blending",
     "description": "Merge two or more existing concepts to create entirely new "
                    "categories — goes beyond simple combination to genuine "
                    "innovation."},
    {"category": "creative", "name": "Reverse Brainstorming",
     "description": "Generate problems instead of solutions — 'What could go "
                    "wrong? How could we make this fail?' to reveal solution "
                    "insights by negation."},
    # Deep / analytical
    {"category": "deep", "name": "Five Whys",
     "description": "Drill down through layers of causation — ask 'Why did this "
                    "happen?' repeatedly until reaching fundamental drivers."},
    {"category": "deep", "name": "Morphological Analysis",
     "description": "Systematically explore all possible parameter combinations "
                    "for complex systems — identify key parameters, list options "
                    "for each, try combinations."},
    {"category": "deep", "name": "Provocation Technique",
     "description": "Use deliberately provocative statements to extract useful "
                    "ideas from seemingly absurd starting points — catalyses "
                    "breakthrough thinking."},
    {"category": "deep", "name": "Assumption Reversal",
     "description": "Challenge and flip core assumptions to rebuild from new "
                    "foundations — essential for paradigm shifts."},
    {"category": "deep", "name": "Question Storming",
     "description": "Generate questions before seeking answers — ensures you're "
                    "solving the right problem. Only questions, no answers yet."},
    {"category": "deep", "name": "Constraint Mapping",
     "description": "Identify and visualize all constraints — 'Which are real "
                    "vs imagined? How do we work around or eliminate barriers?'"},
    {"category": "deep", "name": "Failure Analysis",
     "description": "Study successful failures to extract valuable insights and "
                    "avoid common pitfalls — 'What went wrong, why did it fail, "
                    "what lessons emerge?'"},
    {"category": "deep", "name": "Emergent Thinking",
     "description": "Allow solutions to emerge organically without forcing "
                    "linear progression — 'What patterns emerge? What wants to "
                    "happen naturally?'"},
    # Scientific-writing specific
    {"category": "scientific", "name": "Missing Control",
     "description": "For every claim, ask what the null / baseline / placebo "
                    "case would look like and whether the author addressed it."},
    {"category": "scientific", "name": "Scope Boundaries",
     "description": "For every claim, list the spatial / temporal / population "
                    "/ regime / instrument conditions under which it would fail. "
                    "Each boundary = a potential gap."},
    {"category": "scientific", "name": "Benchmark-Hunting",
     "description": "What would a gold-standard benchmark for this claim look "
                    "like? Does the literature have one? If not, what would it "
                    "take to build?"},
]


def list_methods(kind: str) -> list[dict[str, str]]:
    """Return the full list for a given kind (``"elicitation"`` or
    ``"brainstorming"``)."""
    if kind == "elicitation":
        return list(ELICITATION_METHODS)
    if kind == "brainstorming":
        return list(BRAINSTORMING_METHODS)
    raise ValueError(f"unknown method kind {kind!r}")


def get_method(kind: str, name: str) -> dict[str, str] | None:
    """Look up one method by name (case-insensitive)."""
    if not name:
        return None
    needle = name.strip().lower()
    for m in list_methods(kind):
        if m["name"].lower() == needle:
            return m
    return None


def method_preamble(method: dict[str, Any] | None) -> str:
    """Render a method into a short instruction preamble to inject
    at the top of an LLM user prompt. Returns an empty string if the
    method is None so callers can unconditionally concatenate.
    """
    if not method or not method.get("name"):
        return ""
    return (
        f"Apply the **{method['name']}** technique for this task. "
        f"Method: {method['description']}\n\n"
        f"Proceed with the main task below using this framing.\n\n"
        "---\n\n"
    )
