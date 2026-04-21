from __future__ import annotations

import os

import httpx
from flask import Flask
from flask import render_template, request


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/")
    def index() -> str:
        return render_template("recommend.html", result=None, error=None)

    @app.get("/recommend")
    def recommend_get() -> str:
        return render_template("recommend.html", result=None, error=None)

    @app.post("/recommend")
    def recommend_post() -> str:
        api_url = os.getenv("INVESTAI_API_URL", "http://localhost:8000").rstrip("/")
        symbols_csv = request.form.get("symbols_csv", "").strip()
        include_equity = request.form.get("include_equity") == "on"

        payload = {
            "symbols_csv": symbols_csv,
            "include_equity": include_equity,
        }

        try:
            resp = httpx.post(f"{api_url}/recommendations/backtest", json=payload, timeout=60.0)
            resp.raise_for_status()
            result = resp.json()
            return render_template("recommend.html", result=result, error=None)
        except Exception as e:
            return render_template("recommend.html", result=None, error=str(e))

    return app
