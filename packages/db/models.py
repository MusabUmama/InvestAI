from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import uuid

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Symbol(Base):
    __tablename__ = "symbols"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    asset_class: Mapped[str] = mapped_column(String(32), nullable=False)
    region: Mapped[str] = mapped_column(String(64), nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    price_bars: Mapped[list["PriceBar"]] = relationship(back_populates="symbol_rel", cascade="all, delete-orphan")


class PriceBar(Base):
    __tablename__ = "price_bars"

    symbol: Mapped[str] = mapped_column(String(16), ForeignKey("symbols.symbol", ondelete="CASCADE"), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)

    open: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)

    adjusted_close: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    volume: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    dividend_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    split_coefficient: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False, default="alphavantage")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    symbol_rel: Mapped[Symbol] = relationship(back_populates="price_bars")


class RecommendationRun(Base):
    __tablename__ = "recommendation_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    data_source: Mapped[str] = mapped_column(String(16), nullable=False, default="file")
    data_frequency: Mapped[str] = mapped_column(String(16), nullable=False, default="monthly")
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    request: Mapped[dict] = mapped_column(JSONB, nullable=False)
    symbols: Mapped[list] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    latest_weights: Mapped[dict] = mapped_column(JSONB, nullable=False)
    benchmarks: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    backtest_points: Mapped[list["RecommendationBacktestPoint"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    weight_history: Mapped[list["RecommendationWeightHistory"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    explanations: Mapped[list["RecommendationExplanation"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RecommendationBacktestPoint(Base):
    __tablename__ = "recommendation_backtest_points"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendation_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    portfolio_value: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    portfolio_return: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)

    run: Mapped[RecommendationRun] = relationship(back_populates="backtest_points")


class RecommendationWeightHistory(Base):
    __tablename__ = "recommendation_weight_history"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendation_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    rebalance_date: Mapped[date] = mapped_column(Date, primary_key=True)
    weights: Mapped[dict] = mapped_column(JSONB, nullable=False)

    run: Mapped[RecommendationRun] = relationship(back_populates="weight_history")


class RecommendationExplanation(Base):
    __tablename__ = "recommendation_explanations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="openrouter")
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    content_type: Mapped[str] = mapped_column(String(32), nullable=False, default="text/markdown")
    content: Mapped[str] = mapped_column(String, nullable=False)
    request: Mapped[dict] = mapped_column(JSONB, nullable=False)
    response: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    run: Mapped[RecommendationRun] = relationship(back_populates="explanations")


class MlModel(Base):
    __tablename__ = "ml_models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    name: Mapped[str] = mapped_column(String(64), nullable=False)
    task: Mapped[str] = mapped_column(String(64), nullable=False)
    data_frequency: Mapped[str] = mapped_column(String(16), nullable=False, default="monthly")
    feature_schema: Mapped[str] = mapped_column(String(32), nullable=False, default="monthly_v2")

    symbols: Mapped[list] = mapped_column(JSONB, nullable=False)
    train_params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)

    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
