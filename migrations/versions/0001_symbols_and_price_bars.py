"""Create symbols and price_bars

Revision ID: 0001_symbols_and_price_bars
Revises:
Create Date: 2026-04-18
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_symbols_and_price_bars"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "symbols",
        sa.Column("symbol", sa.String(length=16), primary_key=True),
        sa.Column("name", sa.String(length=256), nullable=True),
        sa.Column("asset_class", sa.String(length=32), nullable=False),
        sa.Column("region", sa.String(length=64), nullable=False),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "price_bars",
        sa.Column("symbol", sa.String(length=16), sa.ForeignKey("symbols.symbol", ondelete="CASCADE"), primary_key=True),
        sa.Column("date", sa.Date(), primary_key=True),
        sa.Column("open", sa.Numeric(18, 6), nullable=False),
        sa.Column("high", sa.Numeric(18, 6), nullable=False),
        sa.Column("low", sa.Numeric(18, 6), nullable=False),
        sa.Column("close", sa.Numeric(18, 6), nullable=False),
        sa.Column("adjusted_close", sa.Numeric(18, 6), nullable=True),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("dividend_amount", sa.Numeric(18, 6), nullable=True),
        sa.Column("split_coefficient", sa.Numeric(18, 6), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False, server_default="alphavantage"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_price_bars_date", "price_bars", ["date"])


def downgrade() -> None:
    op.drop_index("ix_price_bars_date", table_name="price_bars")
    op.drop_table("price_bars")
    op.drop_table("symbols")

