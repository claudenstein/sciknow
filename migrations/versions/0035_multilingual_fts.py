"""Phase 54.6.207 — multilingual FTS.

Revision ID: 0035
Revises: 0034
Create Date: 2026-04-22

Fulfils roadmap item 3.2.5 — language detection + language-aware FTS.

Adds
  * ``documents.language text DEFAULT 'en'`` — ISO 639-1 code (``en``,
    ``es``, ``de``, ``fr``, ``pt``, ``it``, ``ru``, ``nl``, ``tr``,
    …). Populated at ingest time by py3langid on the first ~500
    chars of extracted text. Defaults to ``'en'`` so existing rows
    don't suddenly look stateless.
  * ``chunks.tsvector_lang text`` — Postgres ``regconfig`` name
    (``english``, ``spanish``, ``german``, …). Denormalised from
    ``documents.language`` because tsvector generation is per-row
    and needs the config locally, and GENERATED columns cannot
    JOIN to another table. Defaults to ``'english'``.
  * ``sk_lang_config(text) -> regconfig`` IMMUTABLE helper. Maps
    2-letter ISO codes (en/es/fr/de/pt/it/ru/nl/tr/sv/no/da/fi/hu)
    and full config names to the matching Postgres builtin
    dictionary. Unknown → ``english`` (safe fallback).
  * Replaces ``chunks.search_vector`` GENERATED column with a
    language-aware version using ``sk_lang_config(tsvector_lang)``
    so each row tokenises with its own language dictionary.
  * GIN index recreated on the new expression.

English-only corpora remain functionally identical: all default
values route to the ``english`` config, so the stored tsvectors
are byte-equivalent.

Why a denormalised ``chunks.tsvector_lang`` rather than referring
to ``documents.language`` directly
  GENERATED ALWAYS columns can only reference columns of the same
  row and IMMUTABLE functions. A JOIN isn't allowed, and a
  volatile sub-query can't appear in a generation expression. So
  the document's language must live locally on the chunk row.
  The ingest pipeline populates both in one pass.
"""
from typing import Sequence, Union

from alembic import op


revision: str = "0035"
down_revision: Union[str, None] = "0034"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Mapping from ISO 639-1 codes (or full English names) to the
# Postgres built-in text-search config name. English is the
# catch-all for unknown codes — safer than raising at tsvector
# generation time.
_LANG_MAP = {
    "en": "english",    "english": "english",
    "es": "spanish",    "spanish": "spanish",
    "de": "german",     "german": "german",
    "fr": "french",     "french": "french",
    "pt": "portuguese", "portuguese": "portuguese",
    "it": "italian",    "italian": "italian",
    "ru": "russian",    "russian": "russian",
    "nl": "dutch",      "dutch": "dutch",
    "tr": "turkish",    "turkish": "turkish",
    "sv": "swedish",    "swedish": "swedish",
    "no": "norwegian",  "norwegian": "norwegian",
    "da": "danish",     "danish": "danish",
    "fi": "finnish",    "finnish": "finnish",
    "hu": "hungarian",  "hungarian": "hungarian",
}


def _when_clauses() -> str:
    # Build one CASE arm per language. IMMUTABLE requires literal
    # casts to regconfig, so every arm has ::regconfig.
    arms = []
    for k, v in _LANG_MAP.items():
        arms.append(f"WHEN '{k}' THEN '{v}'::regconfig")
    return "\n    ".join(arms)


def upgrade() -> None:
    # 1. documents.language (ISO code)
    op.execute("""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS language text NOT NULL DEFAULT 'en'
    """)

    # 2. chunks.tsvector_lang (Postgres regconfig name — denormalised)
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS tsvector_lang text NOT NULL DEFAULT 'english'
    """)

    # 3. IMMUTABLE helper: ISO code / config name → regconfig.
    case_arms = _when_clauses()
    op.execute(f"""
        CREATE OR REPLACE FUNCTION sk_lang_config(lang text)
        RETURNS regconfig
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        AS $$
        SELECT CASE lower(COALESCE(lang, ''))
            {case_arms}
            ELSE 'english'::regconfig
        END
        $$
    """)

    # 4. Rebuild chunks.search_vector on the new expression. GENERATED
    # columns can't be altered in place; drop index + column, add
    # back with the language-aware expression.
    op.execute("DROP INDEX IF EXISTS idx_chunks_search_vector")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector")
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector(sk_lang_config(tsvector_lang), coalesce(content, ''))
        ) STORED
    """)
    op.create_index(
        "idx_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_search_vector")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector")
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(content, ''))
        ) STORED
    """)
    op.create_index(
        "idx_chunks_search_vector",
        "chunks",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.execute("DROP FUNCTION IF EXISTS sk_lang_config(text)")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS tsvector_lang")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS language")
