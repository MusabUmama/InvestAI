from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


@dataclass(frozen=True)
class AlphaVantageClient:
    api_key: str
    timeout_seconds: float = 30.0

    @staticmethod
    def from_env() -> "AlphaVantageClient":
        key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ALPHAVANTAGE_API_KEY is not set")
        return AlphaVantageClient(api_key=key)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def get_daily_adjusted(self, symbol: str, outputsize: str = "full") -> dict[str, Any]:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.get(ALPHAVANTAGE_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        # Alpha Vantage uses "Note" for rate limits and "Error Message" for invalid symbols.
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
        return data

