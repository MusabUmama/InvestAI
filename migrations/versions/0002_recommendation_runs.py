"""Create recommendation runs + backtest outputs

Revision ID: 0002_recommendation_runs
Revises: 0001_symbols_and_price_bars
Create Date: 2026-04-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_recommendation_runs"
down_revision = "0001_symbols_and_price_bars"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Needed for gen_random_uuid()
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "recommendation_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("data_source", sa.String(length=16), nullable=False, server_default="file"),
        sa.Column("data_frequency", sa.String(length=16), nullable=False, server_default="monthly"),
        sa.Column("model_path", sa.String(length=512), nullable=True),
        sa.Column("request", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("symbols", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latest_weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    )
    op.create_index("ix_recommendation_runs_created_at", "recommendation_runs", ["created_at"])

    op.create_table(
        "recommendation_backtest_points",
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recommendation_runs.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("date", sa.Date(), primary_key=True, nullable=False),
        sa.Column("portfolio_value", sa.Numeric(18, 6), nullable=False),
        sa.Column("portfolio_return", sa.Numeric(18, 6), nullable=False),
    )
    op.create_index("ix_reco_points_date", "recommendation_backtest_points", ["date"])

    op.create_table(
        "recommendation_weight_history",
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("recommendation_runs.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("rebalance_date", sa.Date(), primary_key=True, nullable=False),
        sa.Column("weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    )
    op.create_index("ix_reco_weight_rebalance_date", "recommendation_weight_history", ["rebalance_date"])


def downgrade() -> None:
    op.drop_index("ix_reco_weight_rebalance_date", table_name="recommendation_weight_history")
    op.drop_table("recommendation_weight_history")
    op.drop_index("ix_reco_points_date", table_name="recommendation_backtest_points")
    op.drop_table("recommendation_backtest_points")
    op.drop_index("ix_recommendation_runs_created_at", table_name="recommendation_runs")
    op.drop_table("recommendation_runs")

