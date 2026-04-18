"""Phase 54.6.78 (#11) — natural-language paraphrase of extracted equations.

bge-m3 embeds raw LaTeX poorly: the tokenizer fragments commands like
``\\frac`` into character-level pieces, and the resulting embedding
drifts from the equation's semantic meaning. A one-sentence prose
paraphrase (*"The slope of outgoing longwave radiation with respect
to global surface temperature, 2.93 ± 0.3 W/m²·K"*) embeds far
better. This module generates those paraphrases via the fast text LLM
using the equation's LaTeX body + its surrounding prose for context.

Storage: reuses the ``visuals.ai_caption`` column added by Phase
54.6.72. Disambiguated by ``kind`` — VLM-generated image captions
live on ``kind in (figure, chart)``; text-LLM equation paraphrases
live on ``kind='equation'``. The ``ai_caption_model`` column records
which model produced the paraphrase, so re-runs can target stale rows.

Out of scope here: re-embedding the paraphrase into Qdrant. That
would be a separate phase — the paraphrases become retrieval-visible
once a new embedding pass runs.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════


PARAPHRASE_SYSTEM = (
    "You are rewriting a mathematical equation from a scientific paper "
    "as one natural-language sentence suitable for retrieval indexing. "
    "Your paraphrase will be embedded and used to surface this equation "
    "when a researcher queries the corpus in prose.\n\n"
    "Rules:\n"
    "  - ONE sentence, 15-50 words.\n"
    "  - Describe what the equation expresses (relationship between "
    "variables) using the domain terms visible in the surrounding text.\n"
    "  - Include any numeric values with their units (e.g. "
    "'2.93 ± 0.3 W/m²·K').\n"
    "  - Keep variable names from the equation (e.g. 'the derivative "
    "dOLR/dT') so a reader can match paraphrase to source.\n"
    "  - NO preamble like 'This equation expresses...' — just state "
    "the relationship directly.\n"
    "  - NO LaTeX syntax in the output — translate \\frac, \\sqrt, "
    "subscripts, etc. into prose.\n"
    "  - If the surrounding text doesn't disambiguate the variables "
    "(e.g. 'r²=r₀²+a²' with no context), write a generic but honest "
    "paraphrase naming the variables and their mathematical form."
)


PARAPHRASE_USER = (
    "Equation (LaTeX):\n{latex}\n\n"
    "Surrounding text from the paper:\n{context}\n\n"
    "Paraphrase:"
)


# ════════════════════════════════════════════════════════════════════════
# LaTeX cleanup (pre-processing)
# ════════════════════════════════════════════════════════════════════════


def _clean_latex(raw: str) -> str:
    """Trim MinerU's ``$$ ... $$`` wrappers + ``\\tag{N}`` suffixes so
    the prompt to the LLM is maximally clean.

    MinerU's content_list.json preserves the original display-math
    wrapper; strip it so the LLM doesn't echo ``$$`` into the output.
    Also strip the equation-number tag ``\\tag{3.13}`` since it's
    context we add to the prompt separately if useful."""
    text = (raw or "").strip()
    # Remove leading + trailing $$ pairs (display math)
    text = re.sub(r"^\$+\s*", "", text)
    text = re.sub(r"\s*\$+$", "", text)
    # Remove \tag{...}
    text = re.sub(r"\\tag\s*\{[^}]*\}", "", text)
    return text.strip()


def _is_trivial_equation(latex: str) -> bool:
    """A 2-character equation like ``a=b`` isn't worth paraphrasing;
    its embedding is already essentially random. Skip rather than
    clutter the ai_caption column with LLM confabulations."""
    body = _clean_latex(latex)
    # Strip whitespace + backslash commands + braces + operators.
    tokens = re.sub(r"[\\{}\s=+\-*/,.()^_]", "", body)
    return len(tokens) < 3


# ════════════════════════════════════════════════════════════════════════
# One-equation paraphrase
# ════════════════════════════════════════════════════════════════════════


def paraphrase_equation(
    latex: str,
    surrounding_text: str = "",
    model: str | None = None,
    *,
    num_ctx: int = 4096,
) -> str | None:
    """Return a one-sentence prose paraphrase, or None if the equation
    is too trivial to paraphrase meaningfully (``_is_trivial_equation``)
    or the LLM returned empty output."""
    body = _clean_latex(latex)
    if not body or _is_trivial_equation(body):
        return None

    from sciknow.rag.llm import complete as _complete

    # Trim surrounding_text so the prompt stays small. 800 chars is
    # usually enough to disambiguate the variables.
    ctx = (surrounding_text or "").strip()[:800] or "(no surrounding text)"
    user = PARAPHRASE_USER.format(latex=body, context=ctx)

    try:
        raw = _complete(
            PARAPHRASE_SYSTEM, user,
            model=model, temperature=0.2, num_ctx=num_ctx, keep_alive=-1,
        )
    except Exception as exc:
        logger.warning("paraphrase LLM call failed: %s", exc)
        return None

    # Clean: strip quote wrappers, truncate to a single sentence.
    text = (raw or "").strip().strip('"\'')
    if not text:
        return None
    # If the model returned multiple sentences, keep only the first.
    text = re.sub(r"\s+", " ", text)
    m = re.search(r"^(.+?[.!?])(?:\s|$)", text)
    if m:
        text = m.group(1)
    # Minimum length — shorter than 10 words is probably a refusal or
    # a degenerate "This equation describes X." non-answer.
    if len(text.split()) < 6:
        return None
    return text.strip()
