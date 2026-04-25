"""LLM adapter layer used by parent and child logical pods.

The rest of the framework depends only on the small ``LLMAdapter`` protocol.
Mock mode returns deterministic structured outputs for local testing; live mode
uses Anthropic and records returned token usage when available.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from agents.schemas import LLMResult


ANTHROPIC_STRONG_MODEL = "claude-opus-4-1-20250805"
ANTHROPIC_MEDIUM_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_WEAK_MODEL = "claude-3-5-haiku-20241022"


class LLMAdapter:
    """Role-oriented interface for swappable LLM providers."""

    def parent_plan(self, context: dict[str, Any]) -> LLMResult:
        """Plan or steer the experiment from the parent-pod role."""
        raise NotImplementedError

    def child_develop(self, context: dict[str, Any]) -> LLMResult:
        """Produce development guidance or patch proposals for a child pod."""
        raise NotImplementedError

    def test_or_perf(self, context: dict[str, Any]) -> LLMResult:
        """Produce test or performance-analysis output for a pod task."""
        raise NotImplementedError

    def summarize(self, context: dict[str, Any]) -> LLMResult:
        """Summarize completed artifacts into a compact report payload."""
        raise NotImplementedError


def resolve_api_key(
    *env_names: str,
    explicit: str | None = None,
) -> str | None:
    """Resolve an Anthropic key from role-specific env vars, then shared key."""
    if explicit:
        return explicit
    for name in env_names:
        value = os.getenv(name)
        if value:
            return value
    return os.getenv("ANTHROPIC_API_KEY")


def choose_anthropic_model(agent_role: str, context: dict[str, Any]) -> tuple[str, str]:
    """Choose a Claude model with simple quality-biased heuristics.

    Parent planning/reporting and high-impact child work use the strongest
    configured model. Routine child exploration still defaults to Sonnet rather
    than a tiny model because this experiment values research quality.
    """
    role = agent_role.lower()
    context_text = json.dumps(context, sort_keys=True, default=str).lower()
    if "parent" in role or "summary" in role:
        return (
            os.getenv("ANTHROPIC_PARENT_MODEL", os.getenv("ANTHROPIC_STRONG_MODEL", ANTHROPIC_STRONG_MODEL)),
            "parent/report role uses strongest configured model",
        )

    high_impact_markers = (
        "final",
        "heldout",
        "prune",
        "promote",
        "serious",
        "illegal",
        "crash",
        "regression",
        "tournament",
        "stockfish",
    )
    if any(marker in context_text for marker in high_impact_markers):
        return (
            os.getenv("ANTHROPIC_STRONG_CHILD_MODEL", os.getenv("ANTHROPIC_STRONG_MODEL", ANTHROPIC_STRONG_MODEL)),
            "high-impact child task uses stronger configured model",
        )

    if "test" in role or "perf" in role:
        return (
            os.getenv("ANTHROPIC_TEST_MODEL", os.getenv("ANTHROPIC_WEAK_MODEL", ANTHROPIC_WEAK_MODEL)),
            "testing/perf task uses weak/fast configured model",
        )

    return (
        os.getenv("ANTHROPIC_CHILD_MODEL", os.getenv("ANTHROPIC_MEDIUM_MODEL", ANTHROPIC_MEDIUM_MODEL)),
        "routine child task uses medium configured model",
    )


def create_llm_adapter(
    mock_mode: bool | None = None,
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> LLMAdapter:
    """Build either the mock adapter or the configured live adapter."""
    if mock_mode is None:
        mock_mode = os.getenv("MOCK_MODE", "true").lower() not in {"0", "false", "no"}
    if mock_mode:
        return MockLLMAdapter()
    env_names = [api_key_env] if api_key_env else []
    return AnthropicMessagesAdapter(api_key=resolve_api_key(*env_names, explicit=api_key))


class MockLLMAdapter(LLMAdapter):
    """Deterministic no-network adapter for tests and demos."""

    def _tokens(self, context: dict[str, Any], floor: int = 120) -> int:
        text = json.dumps(context, sort_keys=True, default=str)
        return floor + max(20, len(text) // 4)

    def _digest(self, context: dict[str, Any]) -> str:
        return hashlib.sha1(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()[:8]

    def parent_plan(self, context: dict[str, Any]) -> LLMResult:
        """Return deterministic mock parent planning output."""
        return LLMResult(
            content={"decision": "continue", "notes": "Mock planner created bounded implementation/eval tasks.", "digest": self._digest(context)},
            estimated_tokens=self._tokens(context, 220),
        )

    def child_develop(self, context: dict[str, Any]) -> LLMResult:
        """Return deterministic mock child development output."""
        candidate_id = context.get("candidate_id")
        return LLMResult(
            content={
                "patch_summary": f"Mock development validated candidate {candidate_id}; built-in engine implementation is available.",
                "checks": ["select_move interface", "legal move smoke"],
                "digest": self._digest(context),
            },
            estimated_tokens=self._tokens(context, 320),
        )

    def test_or_perf(self, context: dict[str, Any]) -> LLMResult:
        """Return deterministic mock testing/performance output."""
        return LLMResult(
            content={"test_summary": "Mock role requested local smoke/eval execution.", "digest": self._digest(context)},
            estimated_tokens=self._tokens(context, 180),
        )

    def summarize(self, context: dict[str, Any]) -> LLMResult:
        """Return deterministic mock final-summary output."""
        return LLMResult(
            content={"summary": "Mock final report generated from stored artifacts and budget ledger.", "digest": self._digest(context)},
            estimated_tokens=self._tokens(context, 200),
        )


class AnthropicMessagesAdapter(LLMAdapter):
    """Anthropic Messages API adapter with heuristic per-call model routing."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = resolve_api_key(explicit=api_key)
        if not self.api_key:
            raise RuntimeError(
                "An Anthropic API key is required when MOCK_MODE=false. "
                "Set ANTHROPIC_API_KEY or role-specific "
                "keys such as PARENT_ANTHROPIC_API_KEY, CHILD_A_ANTHROPIC_API_KEY, "
                "or CHILD_B_ANTHROPIC_API_KEY."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("Install the anthropic package or run `pip install -r requirements.txt`.") from exc
        self.client = Anthropic(api_key=self.api_key)

    def parent_plan(self, context: dict[str, Any]) -> LLMResult:
        """Call Anthropic for parent planning/reporting output."""
        return self._call(
            role="parent_research_manager",
            context=context,
            instructions=(
                "You are the parent research manager for AutoResearch Chess Lab. "
                "Use careful, high-quality reasoning. Return compact JSON with "
                "decision, rationale, next_tasks, prune_or_promote, model_choice, "
                "and caveats. Do not include raw secrets."
            ),
            max_output_tokens=int(os.getenv("ANTHROPIC_PARENT_MAX_OUTPUT_TOKENS", "1200")),
        )

    def child_develop(self, context: dict[str, Any]) -> LLMResult:
        """Call Anthropic for child development or patch-planning output."""
        return self._call(
            role="child_development_pod",
            context=context,
            instructions=(
                "You are a fresh-context child development pod. Return compact "
                "JSON with patch_summary, candidate_assessment, files_to_touch, "
                "checks_to_run, risk_notes, completion_criteria, and optionally "
                "file_replacements. If source_files and allowed_write_prefixes are "
                "provided, you may propose full-file replacements as "
                "{\"file_replacements\":[{\"path\":\"relative/path.py\",\"content\":\"complete file\"}]}. "
                "Only replace files under allowed_write_prefixes. Preserve public "
                "interfaces and do not include markdown fences."
            ),
            max_output_tokens=int(os.getenv("ANTHROPIC_CHILD_MAX_OUTPUT_TOKENS", "6000")),
        )

    def test_or_perf(self, context: dict[str, Any]) -> LLMResult:
        """Call Anthropic for test and performance analysis."""
        return self._call(
            role="testing_or_perf_pod",
            context=context,
            instructions=(
                "You are a testing/performance pod. Return compact JSON with "
                "test_plan, expected_metrics, failure_signals, and recommended_next_step."
            ),
            max_output_tokens=int(os.getenv("ANTHROPIC_TEST_MAX_OUTPUT_TOKENS", "900")),
        )

    def summarize(self, context: dict[str, Any]) -> LLMResult:
        """Call Anthropic for compact final report content."""
        return self._call(
            role="summary_reporter",
            context=context,
            instructions=(
                "You summarize AutoResearch experiment artifacts. Return compact "
                "JSON with summary, metrics, caveats, and recommended_followups."
            ),
            max_output_tokens=int(os.getenv("ANTHROPIC_SUMMARY_MAX_OUTPUT_TOKENS", "1200")),
        )

    def _call(self, role: str, context: dict[str, Any], instructions: str, max_output_tokens: int) -> LLMResult:
        model, model_reason = choose_anthropic_model(role, context)
        prompt = {
            "role": role,
            "context_packet": context,
            "model_policy": {"selected_model": model, "reason": model_reason},
            "output_contract": "Return one JSON object only. No markdown fences.",
        }
        response = self._create_message(model, max_output_tokens, instructions, prompt)
        text = self._message_text(response)
        content = self._parse_json(text)
        usage = getattr(response, "usage", None)
        actual_tokens = self._usage_total_tokens(usage)
        estimated_tokens = actual_tokens or self._estimate_tokens(prompt, text)
        content.setdefault("_adapter", "anthropic_messages")
        content.setdefault("_model", model)
        content.setdefault("_model_reason", model_reason)
        content.setdefault("_response_id", getattr(response, "id", None))
        return LLMResult(content=content, estimated_tokens=estimated_tokens, actual_tokens=actual_tokens)

    def _create_message(self, model: str, max_output_tokens: int, instructions: str, prompt: dict[str, Any]) -> Any:
        """Create a message, streaming when Anthropic requires it for long calls."""
        payload = {
            "model": model,
            "max_tokens": max_output_tokens,
            "system": instructions,
            "messages": [{"role": "user", "content": json.dumps(prompt, sort_keys=True, default=str)}],
        }
        if max_output_tokens >= int(os.getenv("ANTHROPIC_STREAM_MIN_OUTPUT_TOKENS", "8000")):
            return self._stream_message(payload)
        try:
            return self.client.messages.create(**payload)
        except ValueError as exc:
            if "Streaming is required" in str(exc):
                return self._stream_message(payload)
            raise

    def _stream_message(self, payload: dict[str, Any]) -> Any:
        """Use Anthropic streaming and return the final Message object."""
        with self.client.messages.stream(**payload) as stream:
            for _event in stream:
                pass
            return stream.get_final_message()

    def _message_text(self, response: Any) -> str:
        chunks: list[str] = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    def _parse_json(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"items": parsed}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    return parsed if isinstance(parsed, dict) else {"items": parsed}
                except json.JSONDecodeError:
                    pass
        return {"raw_text": text}

    def _usage_total_tokens(self, usage: Any) -> int | None:
        if not usage:
            return None
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        if input_tokens or output_tokens:
            return int(input_tokens + output_tokens)
        return None

    def _estimate_tokens(self, prompt: dict[str, Any], text: str) -> int:
        return max(1, (len(json.dumps(prompt, default=str)) + len(text)) // 4)
