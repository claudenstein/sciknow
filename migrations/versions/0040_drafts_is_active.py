"""v2 Phase C — promote drafts.is_active from JSON metadata to a real column.

Revision ID: 0040
Revises: 0039
Create Date: 2026-04-25

The v1 path stored the active-draft marker as
``custom_metadata->>'is_active' = 'true'`` and resolved the active
version through a three-tier picker (active flag → highest-version-with-
content → first version). Spec §4.1: v2 promotes that to a real
``is_active BOOLEAN`` column with a partial-unique index ensuring at
most one active version per ``(chapter_id, section_type)``.

Migration is data-preserving:

  1. Add the nullable column with a server default of FALSE.
  2. Backfill from the existing JSON marker.
  3. For any (chapter_id, section_type) group where the JSON marker
     never named one, fall back to the highest-version row with
     non-empty content — same rule the v1 picker used (`bridge:
     active_version_picker_v3`).
  4. Add the partial-unique index after the data is sane.

The column stays nullable to preserve INSERT compatibility with code
paths that don't yet set it; FALSE is the safe default. Downgrade
drops the index then the column.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0040"
down_revision: Union[str, None] = "0039"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "drafts",
        sa.Column(
            "is_active", sa.Boolean(), nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )

    # 1. Backfill from JSON marker (v1 wrote 'true'/'false' as text).
    op.execute(
        """
        UPDATE drafts
        SET is_active = TRUE
        WHERE custom_metadata ? 'is_active'
          AND lower(custom_metadata->>'is_active') = 'true';
        """
    )

    # 2. Resolve groups without any active row by promoting the highest-
    # version row with non-empty content. Mirror of the v1 active-version
    # picker rule. Excludes drafts not bound to a chapter (loose docs).
    op.execute(
        """
        WITH groups AS (
          SELECT chapter_id, section_type
          FROM drafts
          WHERE chapter_id IS NOT NULL
            AND section_type IS NOT NULL
          GROUP BY chapter_id, section_type
          HAVING NOT bool_or(is_active)
        ),
        candidates AS (
          SELECT DISTINCT ON (d.chapter_id, d.section_type)
                 d.id, d.chapter_id, d.section_type
          FROM drafts d
          JOIN groups g
            ON d.chapter_id = g.chapter_id
           AND d.section_type = g.section_type
          WHERE length(coalesce(d.content, '')) > 0
          ORDER BY d.chapter_id, d.section_type, d.version DESC
        )
        UPDATE drafts d
        SET is_active = TRUE
        FROM candidates c
        WHERE d.id = c.id;
        """
    )

    # 3. Partial-unique index. Restricted to drafts bound to a chapter
    # so loose drafts (no chapter_id) don't fight for the slot.
    op.create_index(
        "ux_drafts_active_per_section",
        "drafts",
        ["chapter_id", "section_type"],
        unique=True,
        postgresql_where=sa.text(
            "is_active = TRUE AND chapter_id IS NOT NULL "
            "AND section_type IS NOT NULL"
        ),
    )

    # Helpful supporting index for "find active draft by section" lookups
    # the v2 picker will issue (no need for partial-uniqueness here, just
    # speed).
    op.create_index(
        "idx_drafts_chapter_section_active",
        "drafts",
        ["chapter_id", "section_type", "is_active"],
        postgresql_where=sa.text(
            "chapter_id IS NOT NULL AND section_type IS NOT NULL"
        ),
    )


def downgrade() -> None:
    """Restore the JSON marker so v1 readers keep working, then drop
    the column. Idempotent on re-application."""
    # Restore the JSON marker before dropping the column so v1 readers
    # don't lose the active-draft info.
    op.execute(
        """
        UPDATE drafts
        SET custom_metadata = jsonb_set(
              coalesce(custom_metadata, '{}'::jsonb),
              '{is_active}',
              to_jsonb(is_active)
            )
        WHERE is_active = TRUE;
        """
    )

    op.drop_index("idx_drafts_chapter_section_active", table_name="drafts")
    op.drop_index("ux_drafts_active_per_section", table_name="drafts")
    op.drop_column("drafts", "is_active")
