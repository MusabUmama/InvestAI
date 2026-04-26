from __future__ import annotations

import os

import httpx
from flask import Flask
from flask import render_template, request


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

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

    @app.get("/runs")
    def runs() -> str:
        api_url = os.getenv("INVESTAI_API_URL", "http://localhost:8000").rstrip("/")
        try:
            resp = httpx.get(f"{api_url}/recommendations/runs", params={"limit": 50}, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            # FastAPI returns a plain list; PowerShell formatting sometimes shows {value, Count} but JSON is list.
            runs_list = data if isinstance(data, list) else data.get("value", [])
            return render_template("runs.html", runs=runs_list, error=None)
        except Exception as e:
            return render_template("runs.html", runs=[], error=str(e))

    @app.get("/runs/<run_id>")
    def run_detail(run_id: str) -> str:
        api_url = os.getenv("INVESTAI_API_URL", "http://localhost:8000").rstrip("/")
        try:
            run_resp = httpx.get(f"{api_url}/recommendations/runs/{run_id}", timeout=20.0)
            run_resp.raise_for_status()
            run_data = run_resp.json()
        except Exception as e:
            return render_template("run_detail.html", run=None, explanation=None, error=str(e), run_id=run_id)

        explanation = None
        try:
            expl_resp = httpx.get(f"{api_url}/recommendations/runs/{run_id}/explanation", timeout=20.0)
            if expl_resp.status_code == 200:
                explanation = expl_resp.json()
        except Exception:
            explanation = None

        return render_template("run_detail.html", run=run_data, explanation=explanation, error=None, run_id=run_id)

    @app.post("/runs/<run_id>/explain")
    def run_explain(run_id: str) -> str:
        api_url = os.getenv("INVESTAI_API_URL", "http://localhost:8000").rstrip("/")
        force = request.form.get("force") == "on"
        try:
            resp = httpx.post(
                f"{api_url}/recommendations/runs/{run_id}/explain",
                json={"force": force},
                timeout=90.0,
            )
            resp.raise_for_status()
        except Exception as e:
            return render_template("run_detail.html", run=None, explanation=None, error=str(e), run_id=run_id)
        return run_detail(run_id)

    return app
