from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(RuntimeError):
    pass


class AlphaVantageRateLimitError(AlphaVantageError):
    pass


class AlphaVantagePremiumError(AlphaVantageError):
    pass


class AlphaVantageInvalidRequestError(AlphaVantageError):
    pass


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

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, AlphaVantageRateLimitError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def get_time_series(self, *, function: str, symbol: str, outputsize: str | None = None) -> dict[str, Any]:
        params: dict[str, str] = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        if outputsize:
            params["outputsize"] = outputsize
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.get(ALPHAVANTAGE_BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        # Alpha Vantage uses these keys for common failure modes.
        if "Error Message" in data:
            raise AlphaVantageInvalidRequestError(f"{symbol}: {data['Error Message']}")
        if "Note" in data:
            raise AlphaVantageRateLimitError(data["Note"])
        if "Information" in data:
            msg = str(data["Information"])
            if "premium" in msg.lower():
                raise AlphaVantagePremiumError(msg)
            raise AlphaVantageInvalidRequestError(msg)
        return data

    def get_daily(self, symbol: str, outputsize: str = "compact") -> dict[str, Any]:
        return self.get_time_series(function="TIME_SERIES_DAILY", symbol=symbol, outputsize=outputsize)

    def get_monthly(self, symbol: str) -> dict[str, Any]:
        return self.get_time_series(function="TIME_SERIES_MONTHLY", symbol=symbol)
