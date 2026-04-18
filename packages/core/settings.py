from __future__ import annotations

import os


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


OPENROUTER_API_KEY = env("OPENROUTER_API_KEY")
ALPHAVANTAGE_API_KEY = env("ALPHAVANTAGE_API_KEY")

# Database wiring comes in later steps; keep the variable name stable from day 1.
DATABASE_URL = env("DATABASE_URL")

