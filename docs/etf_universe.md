# ETF Universe (South Asia-Friendly)

This project targets a practical ETF-only universe that is commonly available to investors in South Asia via
international brokers (US-listed, USD-denominated, large and liquid, widely covered by data providers).

The goal is global diversification with simple building blocks:

- US equities: broad market + growth tilt
- International equities: developed + emerging markets
- India tilt: optional dedicated India exposure
- Bonds: US investment-grade aggregate + US Treasuries + inflation-protected
- Alternatives: gold + listed real estate

## Default Symbols

- `VT`  Total World Stock ETF
- `VTI` US Total Stock Market ETF
- `QQQ` NASDAQ-100 (US large-cap growth tilt)
- `VXUS` International equities ex-US
- `VWO` Emerging markets equities
- `INDA` India equities (tilt)
- `BND` US total bond market
- `IEF` US 7-10Y Treasury bonds
- `TIP` US inflation-protected bonds (TIPS)
- `GLD` Gold
- `VNQ` US REITs

## Notes

- Keep the universe small early. It makes data ingestion cheaper and model training/evaluation clearer.
- For v1, we optimize for Sharpe under long-only + max-weight constraints; we do not use leverage.
- This is decision-support software. It is not personalized financial advice.

