"""FastAPI application factory for the AutoResearch control plane."""
from __future__ import annotations

from fastapi import FastAPI

from backend.config import ui_url
from backend.routes import control, dashboard
from backend.storage import JsonStore


def create_app() -> FastAPI:
    """Create an app with a local JSON-backed store attached to app state."""
    app = FastAPI(title="AutoResearch Chess Lab", version="0.1.0")
    app.state.store = JsonStore()
    app.include_router(control.router)
    app.include_router(dashboard.router)

    @app.get("/")
    def root() -> dict:
        """Return a compact service index for operators and smoke checks."""
        return {
            "name": "AutoResearch Chess Lab API",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "dashboard": "/dashboard",
            "experiments": "/experiments",
            "ui": ui_url(),
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return liveness status for Docker and local scripts."""
        return {"status": "ok"}

    return app


app = create_app()
