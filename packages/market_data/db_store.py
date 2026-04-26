from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select

from packages.core.db import db_session
from packages.db.models import PriceBar


@dataclass(frozen=True)
class DbPriceStore:
    """
    Reads price bars from Postgres (price_bars table).
    Returns a DataFrame shaped like LocalPriceStore.read_symbol_bars().
    """

    def read_symbol_bars(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        with db_session() as session:
            rows = session.execute(
                select(
                    PriceBar.date,
                    PriceBar.open,
                    PriceBar.high,
                    PriceBar.low,
                    PriceBar.close,
                    PriceBar.adjusted_close,
                    PriceBar.volume,
                    PriceBar.dividend_amount,
                    PriceBar.split_coefficient,
                ).where(PriceBar.symbol == sym)
            ).all()

        if not rows:
            raise FileNotFoundError(f"No DB price bars for {sym}. Ingest data first.")

        df = pd.DataFrame(
            rows,
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "dividend_amount",
                "split_coefficient",
            ],
        )
        df["symbol"] = sym
        # Ensure consistent dtypes
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.sort_values("date")

