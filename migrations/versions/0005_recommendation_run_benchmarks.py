"""Add benchmarks to recommendation runs

Revision ID: 0005_recommendation_run_benchmarks
Revises: 0004_ml_models
Create Date: 2026-04-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_recommendation_run_benchmarks"
down_revision = "0004_ml_models"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "recommendation_runs",
        sa.Column("benchmarks", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )


def downgrade() -> None:
    op.drop_column("recommendation_runs", "benchmarks")

