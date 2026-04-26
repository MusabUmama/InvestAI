"""Create recommendation explanations

Revision ID: 0003_recommendation_explanations
Revises: 0002_recommendation_runs
Create Date: 2026-04-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_recommendation_explanations"
down_revision = "0002_recommendation_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "recommendation_explanations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recommendation_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("provider", sa.String(length=32), nullable=False, server_default="openrouter"),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False, server_default="v1"),
        sa.Column("content_type", sa.String(length=32), nullable=False, server_default="text/markdown"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("request", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.create_index("ix_reco_expl_run_id", "recommendation_explanations", ["run_id"])
    op.create_index("ix_reco_expl_created_at", "recommendation_explanations", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_reco_expl_created_at", table_name="recommendation_explanations")
    op.drop_index("ix_reco_expl_run_id", table_name="recommendation_explanations")
    op.drop_table("recommendation_explanations")

