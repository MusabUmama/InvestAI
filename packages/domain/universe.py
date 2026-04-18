from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Etf:
    symbol: str
    name: str
    asset_class: str
    region: str
    currency: str = "USD"


# South Asia-friendly default universe: US-listed, liquid, globally diversified.
DEFAULT_ETF_UNIVERSE: tuple[Etf, ...] = (
    Etf("VT", "Vanguard Total World Stock ETF", "Equity", "Global"),
    Etf("VTI", "Vanguard Total Stock Market ETF", "Equity", "US"),
    Etf("QQQ", "Invesco QQQ Trust (NASDAQ-100)", "Equity", "US"),
    Etf("VXUS", "Vanguard Total International Stock ETF", "Equity", "Global ex-US"),
    Etf("VWO", "Vanguard FTSE Emerging Markets ETF", "Equity", "Emerging Markets"),
    Etf("INDA", "iShares MSCI India ETF", "Equity", "India"),
    Etf("BND", "Vanguard Total Bond Market ETF", "Bond", "US"),
    Etf("IEF", "iShares 7-10 Year Treasury Bond ETF", "Bond", "US"),
    Etf("TIP", "iShares TIPS Bond ETF", "Bond", "US"),
    Etf("GLD", "SPDR Gold Shares", "Commodity", "Global"),
    Etf("VNQ", "Vanguard Real Estate ETF", "REIT", "US"),
)


DEFAULT_ETF_SYMBOLS: tuple[str, ...] = tuple(etf.symbol for etf in DEFAULT_ETF_UNIVERSE)

