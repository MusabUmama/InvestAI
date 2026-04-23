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

## Quickstart (Data Foundation)

### Option A: Docker Compose (recommended)

1. Create `.env` (based on `.env.example`) and set:

   - `ALPHAVANTAGE_API_KEY=...`

2. Start services:

   - `docker compose up -d --build`

3. Run migrations (creates tables in Postgres):

   - `docker compose run --rm api alembic upgrade head`

4. Backfill ETF prices (daily adjusted) into Postgres:

   - `docker compose run --rm api python scripts/ingest_alpha_vantage_daily.py --outputsize full`

Your `DATABASE_URL` in Docker Compose is already set to:

- `postgresql+psycopg://postgres:postgres@db:5432/investai`

### Option B: Local Python

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

## No-DB Dev Path (Skip Migrations)

If Docker/Postgres isn’t ready yet, you can continue with ML development using file-based storage:

1. Ingest prices to CSV files:
   - `python scripts/ingest_alpha_vantage_daily.py --store file --frequency monthly`
   - Writes to `data/processed/price_bars/*.csv`

2. Build features + monthly training dataset:
   - `python scripts/build_features.py`
   - Writes to `data/features/monthly_dataset.csv`

3. Train a baseline return model:
   - `python scripts/train_return_model.py`
   - Writes to `artifacts/return_model/`

4. Run Sharpe-optimized recommendation + walk-forward backtest (file-based):
   - `python scripts/recommend_and_backtest.py`
   - Writes to `artifacts/backtests/latest/`
