from __future__ import annotations

import os

import bleach
import httpx
from flask import Flask
from flask import redirect, render_template, request
import markdown as md


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.template_filter("render_markdown")
    def render_markdown(text: str) -> str:
        html = md.markdown(text or "", extensions=["fenced_code", "tables"])
        return bleach.clean(
            html,
            tags=[
                "p",
                "pre",
                "code",
                "strong",
                "em",
                "ul",
                "ol",
                "li",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "blockquote",
                "br",
                "hr",
                "table",
                "thead",
                "tbody",
                "tr",
                "th",
                "td",
                "a",
            ],
            attributes={"a": ["href", "title", "rel", "target"]},
            protocols=["http", "https"],
            strip=True,
        )

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
            if result and result.get("run_id"):
                return redirect(f"/runs/{result['run_id']}")
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

        series = None
        try:
            s_resp = httpx.get(f"{api_url}/recommendations/runs/{run_id}/series", timeout=20.0)
            if s_resp.status_code == 200:
                series = s_resp.json()
        except Exception:
            series = None

        explanation = None
        try:
            expl_resp = httpx.get(f"{api_url}/recommendations/runs/{run_id}/explanation", timeout=20.0)
            if expl_resp.status_code == 200:
                explanation = expl_resp.json()
        except Exception:
            explanation = None

        return render_template(
            "run_detail.html",
            run=run_data,
            series=series,
            explanation=explanation,
            error=None,
            run_id=run_id,
        )

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
