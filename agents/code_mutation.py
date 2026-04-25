"""Bounded code-mutation helpers for live child pods.

The live orchestrator lets Anthropic propose full-file replacements, but this
module keeps the dangerous parts small and auditable:

- paths must be relative and under an allowed prefix
- empty replacements are rejected
- original file content is kept in memory so failed checks can be reverted
- artifacts receive hashes and paths, not raw secrets
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_CONTEXT_FILES = (
    "engine/common/interface.py",
    "engine/common/eval.py",
    "engine/common/registry.py",
    "tests/test_engines.py",
    "tests/test_tournament.py",
)


@dataclass
class AppliedChange:
    """In-memory record needed to revert one accepted replacement."""

    path: Path
    relative_path: str
    existed_before: bool
    original_text: str


def repo_root() -> Path:
    """Return the project root from this module location."""
    return Path(__file__).resolve().parents[1]


def sha256_text(text: str) -> str:
    """Return a short content hash for audit summaries."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def candidate_allowed_prefixes(kind: str) -> tuple[str, ...]:
    """Return writable path prefixes for one candidate-improvement task."""
    return (
        f"engine/candidates/{kind}/",
        "engine/common/",
        "tests/",
    )


def collect_candidate_source_context(kind: str, root: Path | None = None) -> list[dict[str, str]]:
    """Collect compact source files for a candidate-development prompt."""
    root = root or repo_root()
    paths = [f"engine/candidates/{kind}/engine.py", *SOURCE_CONTEXT_FILES]
    files: list[dict[str, str]] = []
    for rel in paths:
        path = root / rel
        if path.exists() and path.is_file():
            files.append({"path": rel, "content": path.read_text(encoding="utf-8")})
    return files


def extract_file_replacements(content: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize model output into replacement specs.

    Accepted shapes:

    - {"file_replacements": [{"path": "...", "content": "..."}]}
    - {"files": [{"path": "...", "content": "..."}]}
    """
    raw = content.get("file_replacements") or content.get("files") or []
    if not isinstance(raw, list):
        return []
    replacements = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        new_content = item.get("content")
        if isinstance(path, str) and isinstance(new_content, str):
            replacements.append({"path": path, "content": new_content})
    return replacements


def _safe_target(root: Path, rel_path: str, allowed_prefixes: tuple[str, ...]) -> tuple[Path | None, str | None]:
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    if not rel_path or rel_path.startswith("../") or "/../" in rel_path:
        return None, "path traversal is not allowed"
    if not any(rel_path.startswith(prefix) for prefix in allowed_prefixes):
        return None, f"path is outside allowed prefixes: {', '.join(allowed_prefixes)}"
    target = (root / rel_path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None, "resolved path escapes repo root"
    return target, None


def apply_file_replacements(
    replacements: list[dict[str, str]],
    *,
    root: Path | None = None,
    allowed_prefixes: tuple[str, ...],
    max_file_bytes: int = 120_000,
) -> tuple[dict[str, Any], list[AppliedChange]]:
    """Apply safe full-file replacements and return an audit summary."""
    root = root or repo_root()
    applied_changes: list[AppliedChange] = []
    summary: dict[str, Any] = {"applied": [], "rejected": []}

    for item in replacements:
        rel_path = item["path"].replace("\\", "/").lstrip("/")
        new_content = item["content"]
        target, error = _safe_target(root, rel_path, allowed_prefixes)
        if error:
            summary["rejected"].append({"path": rel_path, "reason": error})
            continue
        if not new_content.strip():
            summary["rejected"].append({"path": rel_path, "reason": "empty replacement rejected"})
            continue
        if len(new_content.encode("utf-8")) > max_file_bytes:
            summary["rejected"].append({"path": rel_path, "reason": "replacement exceeds max_file_bytes"})
            continue

        assert target is not None
        existed_before = target.exists()
        original_text = target.read_text(encoding="utf-8") if existed_before else ""
        if existed_before and original_text == new_content:
            summary["rejected"].append({"path": rel_path, "reason": "replacement is identical to current file"})
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        applied_changes.append(AppliedChange(target, rel_path, existed_before, original_text))
        summary["applied"].append({
            "path": rel_path,
            "before_hash": sha256_text(original_text) if existed_before else None,
            "after_hash": sha256_text(new_content),
            "bytes": len(new_content.encode("utf-8")),
        })

    return summary, applied_changes


def revert_changes(changes: list[AppliedChange]) -> None:
    """Restore files changed by ``apply_file_replacements``."""
    for change in reversed(changes):
        if change.existed_before:
            change.path.write_text(change.original_text, encoding="utf-8")
        elif change.path.exists():
            change.path.unlink()


def json_dumps_compact(value: Any) -> str:
    """Serialize prompt context deterministically for debugging/tests."""
    return json.dumps(value, sort_keys=True, default=str)
