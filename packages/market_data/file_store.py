from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LocalPriceStore:
    root_dir: Path

    @staticmethod
    def default() -> "LocalPriceStore":
        return LocalPriceStore(root_dir=Path("data/processed/price_bars"))

    def path_for_symbol(self, symbol: str) -> Path:
        return self.root_dir / f"{symbol.upper()}.csv"

    def write_symbol_bars(self, symbol: str, df: pd.DataFrame) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for_symbol(symbol)

        # Expected columns: date, open, high, low, close, adjusted_close, volume, dividend_amount, split_coefficient
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        df.to_csv(path, index=False)
        return path

    def read_symbol_bars(self, symbol: str) -> pd.DataFrame:
        path = self.path_for_symbol(symbol)
        if not path.exists():
            raise FileNotFoundError(f"Missing price file for {symbol}: {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["symbol"] = symbol.upper()
        return df

