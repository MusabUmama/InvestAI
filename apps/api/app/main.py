from __future__ import annotations

from fastapi import FastAPI

from .routers.recommendations import router as recommendations_router


def create_app() -> FastAPI:
    app = FastAPI(title="InvestAI API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(recommendations_router)
    return app


app = create_app()
