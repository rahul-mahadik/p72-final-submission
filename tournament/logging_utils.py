"""Small JSONL logging helpers for generated evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import artifacts_root


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append a timestamped JSON object beneath the artifacts directory."""
    raw_path = Path(path)
    target = raw_path if raw_path.is_absolute() else artifacts_root() / raw_path
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {"created_at": datetime.now(timezone.utc).isoformat(), **payload}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
