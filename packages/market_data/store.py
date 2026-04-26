from __future__ import annotations

from typing import Protocol

import pandas as pd


class PriceStore(Protocol):
    def read_symbol_bars(self, symbol: str) -> pd.DataFrame: ...

