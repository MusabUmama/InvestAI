from __future__ import annotations

import argparse
import os
import time
from datetime import date
from decimal import Decimal
from typing import Any, Iterable

from dotenv import load_dotenv
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from packages.core.db import db_session
from packages.db.models import PriceBar, Symbol
from packages.domain.universe import DEFAULT_ETF_UNIVERSE, DEFAULT_ETF_SYMBOLS
from packages.market_data.file_store import LocalPriceStore
from packages.market_data.providers.alpha_vantage.client import AlphaVantageClient


def _parse_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return Decimal(value)


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    return int(Decimal(value))


def _extract_time_series(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    # Alpha Vantage uses one of these keys depending on endpoint.
    for key in ("Time Series (Daily)", "Time Series (Daily Adjusted)"):
        if key in payload and isinstance(payload[key], dict):
            return payload[key]
    keys = ", ".join(payload.keys())
    raise RuntimeError(f"Unexpected Alpha Vantage response keys: {keys}")


def _iter_default_symbols(symbols: Iterable[str] | None) -> list[str]:
    if symbols is None:
        return list(DEFAULT_ETF_SYMBOLS)
    out: list[str] = []
    for sym in symbols:
        sym = sym.strip().upper()
        if sym:
            out.append(sym)
    return out


def _upsert_symbols(session, symbols: list[str]) -> None:
    universe_by_symbol = {e.symbol: e for e in DEFAULT_ETF_UNIVERSE}
    rows = []
    for sym in symbols:
        etf = universe_by_symbol.get(sym)
        rows.append(
            {
                "symbol": sym,
                "name": None if etf is None else etf.name,
                "asset_class": "Equity" if etf is None else etf.asset_class,
                "region": "Unknown" if etf is None else etf.region,
                "currency": "USD" if etf is None else etf.currency,
                "is_active": True,
            }
        )

    stmt = insert(Symbol.__table__).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol"],
        set_={
            "name": stmt.excluded.name,
            "asset_class": stmt.excluded.asset_class,
            "region": stmt.excluded.region,
            "currency": stmt.excluded.currency,
            "is_active": stmt.excluded.is_active,
        },
    )
    session.execute(stmt)


def _chunked(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _upsert_price_bars(session, symbol: str, time_series: dict[str, dict[str, str]]) -> int:
    rows: list[dict[str, Any]] = []
    for day, values in time_series.items():
        rows.append(
            {
                "symbol": symbol,
                "date": date.fromisoformat(day),
                "open": _parse_decimal(values.get("1. open")) or Decimal("0"),
                "high": _parse_decimal(values.get("2. high")) or Decimal("0"),
                "low": _parse_decimal(values.get("3. low")) or Decimal("0"),
                "close": _parse_decimal(values.get("4. close")) or Decimal("0"),
                "adjusted_close": _parse_decimal(values.get("5. adjusted close")),
                "volume": _parse_int(values.get("6. volume")),
                "dividend_amount": _parse_decimal(values.get("7. dividend amount")),
                "split_coefficient": _parse_decimal(values.get("8. split coefficient")),
                "source": "alphavantage",
            }
        )

    inserted = 0
    for batch in _chunked(rows, size=500):
        stmt = insert(PriceBar.__table__).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "adjusted_close": stmt.excluded.adjusted_close,
                "volume": stmt.excluded.volume,
                "dividend_amount": stmt.excluded.dividend_amount,
                "split_coefficient": stmt.excluded.split_coefficient,
                "source": stmt.excluded.source,
            },
        )
        result = session.execute(stmt)
        inserted += result.rowcount or 0
    return inserted


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest daily adjusted ETF prices from Alpha Vantage into Postgres.")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols. Default: built-in ETF universe.",
    )
    parser.add_argument("--outputsize", choices=("compact", "full"), default="full")
    parser.add_argument(
        "--store",
        choices=("db", "file"),
        default="db",
        help="Where to store results: 'db' (Postgres) or 'file' (data/processed/price_bars/*.csv).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.getenv("ALPHAVANTAGE_SLEEP_SECONDS", "15")),
        help="Sleep between API calls to respect rate limits (default 15s).",
    )
    args = parser.parse_args()

    symbols = _iter_default_symbols(args.symbols.split(",") if args.symbols.strip() else None)

    client = AlphaVantageClient.from_env()

    if args.store == "file":
        store = LocalPriceStore.default()
        for idx, sym in enumerate(symbols, start=1):
            payload = client.get_daily_adjusted(sym, outputsize=args.outputsize)
            if "Note" in payload:
                time.sleep(max(args.sleep_seconds, 60.0))
                payload = client.get_daily_adjusted(sym, outputsize=args.outputsize)

            series = _extract_time_series(payload)
            rows: list[dict[str, Any]] = []
            for day, values in series.items():
                rows.append(
                    {
                        "date": day,
                        "open": values.get("1. open"),
                        "high": values.get("2. high"),
                        "low": values.get("3. low"),
                        "close": values.get("4. close"),
                        "adjusted_close": values.get("5. adjusted close"),
                        "volume": values.get("6. volume"),
                        "dividend_amount": values.get("7. dividend amount"),
                        "split_coefficient": values.get("8. split coefficient"),
                    }
                )
            df = pd.DataFrame(rows)
            path = store.write_symbol_bars(sym, df)
            print(f"[{idx}/{len(symbols)}] {sym}: wrote {len(df)} bars -> {path}")

            if idx < len(symbols):
                time.sleep(args.sleep_seconds)
    else:
        with db_session() as session:
            _upsert_symbols(session, symbols)
            session.commit()

            for idx, sym in enumerate(symbols, start=1):
                payload = client.get_daily_adjusted(sym, outputsize=args.outputsize)
                if "Note" in payload:
                    # Rate limit note; pause longer and retry once.
                    time.sleep(max(args.sleep_seconds, 60.0))
                    payload = client.get_daily_adjusted(sym, outputsize=args.outputsize)

                series = _extract_time_series(payload)
                upserted = _upsert_price_bars(session, sym, series)
                session.commit()
                print(f"[{idx}/{len(symbols)}] {sym}: upserted {upserted} bars")

                if idx < len(symbols):
                    time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
