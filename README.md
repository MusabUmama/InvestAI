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
