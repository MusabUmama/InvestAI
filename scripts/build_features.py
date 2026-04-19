from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from packages.domain.universe import DEFAULT_ETF_SYMBOLS
from packages.market_data.file_store import LocalPriceStore
from packages.ml.features import add_basic_features, to_monthly_dataset


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build features and a monthly training dataset from local price files.")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols; default is built-in universe.")
    parser.add_argument(
        "--out",
        default="data/features/monthly_dataset.csv",
        help="Output CSV path for the merged monthly dataset.",
    )
    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols.strip()
        else list(DEFAULT_ETF_SYMBOLS)
    )

    store = LocalPriceStore.default()
    monthly_parts: list[pd.DataFrame] = []

    for sym in symbols:
        daily = store.read_symbol_bars(sym)
        daily = add_basic_features(daily)
        monthly = to_monthly_dataset(daily)
        monthly_parts.append(monthly)

    merged = pd.concat(monthly_parts, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

