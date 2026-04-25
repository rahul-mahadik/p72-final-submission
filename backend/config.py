"""Environment-driven path and dashboard configuration."""
from __future__ import annotations

import os
from pathlib import Path


def experiments_root() -> Path:
    """Directory containing durable per-experiment state."""
    return Path(os.getenv("AUTORESEARCH_EXPERIMENTS_DIR", "experiments"))


def artifacts_root() -> Path:
    """Directory containing exported artifacts, evals, PGNs, and logs."""
    return Path(os.getenv("AUTORESEARCH_ARTIFACTS_DIR", "artifacts"))


def ui_url() -> str:
    """Public dashboard URL shown by the API root."""
    return os.getenv("AUTORESEARCH_DASHBOARD_URL", "/dashboard")
