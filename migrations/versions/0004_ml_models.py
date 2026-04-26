"""Create ML model registry

Revision ID: 0004_ml_models
Revises: 0003_recommendation_explanations
Create Date: 2026-04-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_ml_models"
down_revision = "0003_recommendation_explanations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ml_models",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=64), nullable=False),
        sa.Column("data_frequency", sa.String(length=16), nullable=False, server_default="monthly"),
        sa.Column("feature_schema", sa.String(length=32), nullable=False, server_default="monthly_v2"),
        sa.Column("symbols", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("train_params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_ml_models_created_at", "ml_models", ["created_at"])
    op.create_index("ix_ml_models_name", "ml_models", ["name"])


def downgrade() -> None:
    op.drop_index("ix_ml_models_name", table_name="ml_models")
    op.drop_index("ix_ml_models_created_at", table_name="ml_models")
    op.drop_table("ml_models")

