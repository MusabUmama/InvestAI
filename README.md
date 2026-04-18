# InvestAI
A full-stack AI-powered Robo-Advisor that analyzes user profiles, financial goals, and risk tolerance to recommend personalized investment portfolios, run backtests, and generate forecasts using market data.

## Status

This repo is being rebuilt into a production-grade end-to-end system:

- ETFs only (real market data via Alpha Vantage)
- Sharpe-optimized portfolio construction (long-only, constrained)
- Flask frontend + FastAPI API
- Dockerized local/dev environment
- OpenRouter LLM for grounded explanations (not allocation decisions)

## Project Layout (New)

- `apps/web` Flask UI
- `apps/api` FastAPI service
- `packages` shared domain + ML/quant code
- `docs/etf_universe.md` default ETF universe
- `scripts/synthetic` legacy synthetic generators (reference only)

## Quickstart (Data Foundation)

1. Install deps:

   - `pip install -r requirements.txt`

2. Set environment:

   - Copy `.env.example` to `.env` and fill `DATABASE_URL` and `ALPHAVANTAGE_API_KEY`

3. Create tables:

   - `alembic upgrade head`

4. Backfill ETF prices (daily adjusted):

   - `python scripts/ingest_alpha_vantage_daily.py --outputsize full`

Notes:
- Alpha Vantage free tier is rate-limited; the script sleeps between symbols (`ALPHAVANTAGE_SLEEP_SECONDS`, default 15s).
