"""Small structured payloads used by the agent layer."""
from __future__ import annotations

from pydantic import BaseModel, Field


class LLMResult(BaseModel):
    """Normalized response returned by any LLM adapter."""

    content: dict
    estimated_tokens: int
    actual_tokens: int | None = None
    artifacts: list[dict] = Field(default_factory=list)


class PodConfig(BaseModel):
    """Runtime configuration for one logical child pod."""

    pod_id: str
    api_key_id: str
    api_key_env: str | None = None
    backend_url: str = "http://127.0.0.1:8000"
    mock_mode: bool = True
