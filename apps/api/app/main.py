from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi import Request, Response

from .routers.recommendations import router as recommendations_router
from .routers.run_series import router as run_series_router


def create_app() -> FastAPI:
    app = FastAPI(title="InvestAI API", version="0.1.0")

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        # Lightweight readiness check: can we open a DB session if configured?
        try:
            from packages.core.db import get_database_url, db_session
            from sqlalchemy import text

            _ = get_database_url()
            with db_session() as session:
                session.execute(text("SELECT 1"))
            return {"status": "ready"}
        except Exception as e:
            return {"status": f"not_ready: {e}"}

    app.include_router(recommendations_router)
    app.include_router(run_series_router)
    return app


app = create_app()
