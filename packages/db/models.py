from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, Numeric, String, func
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

