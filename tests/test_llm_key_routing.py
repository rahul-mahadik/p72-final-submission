from __future__ import annotations

from agents.llm_adapter import choose_anthropic_model, resolve_api_key
from agents.pod_worker import default_api_key_env


def test_resolve_api_key_prefers_explicit(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared")
    monkeypatch.setenv("CHILD_A_ANTHROPIC_API_KEY", "child-a")
    assert resolve_api_key("CHILD_A_ANTHROPIC_API_KEY", explicit="direct") == "direct"


def test_resolve_api_key_prefers_role_specific_before_shared(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared")
    monkeypatch.setenv("CHILD_A_ANTHROPIC_API_KEY", "child-a")
    assert resolve_api_key("CHILD_A_ANTHROPIC_API_KEY") == "child-a"


def test_resolve_api_key_falls_back_to_shared(monkeypatch) -> None:
    monkeypatch.delenv("CHILD_B_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared")
    assert resolve_api_key("CHILD_B_ANTHROPIC_API_KEY") == "shared"


def test_default_pod_api_key_env() -> None:
    assert default_api_key_env("child-a") == "CHILD_A_ANTHROPIC_API_KEY"
    assert default_api_key_env("pod_b") == "POD_B_ANTHROPIC_API_KEY"


def test_anthropic_model_policy_is_quality_biased(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_PARENT_MODEL", "claude-opus-4-1-20250805")
    monkeypatch.setenv("ANTHROPIC_STRONG_CHILD_MODEL", "claude-opus-4-1-20250805")
    monkeypatch.setenv("ANTHROPIC_CHILD_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.setenv("ANTHROPIC_TEST_MODEL", "claude-3-5-haiku-20241022")

    parent_model, _ = choose_anthropic_model("parent_research_manager", {"objective": "plan"})
    child_model, _ = choose_anthropic_model("child_development_pod", {"objective": "routine explore"})
    high_model, _ = choose_anthropic_model("child_development_pod", {"objective": "final heldout regression"})
    test_model, _ = choose_anthropic_model("testing_or_perf_pod", {"objective": "quick smoke"})

    assert parent_model == "claude-opus-4-1-20250805"
    assert child_model == "claude-sonnet-4-20250514"
    assert high_model == "claude-opus-4-1-20250805"
    assert test_model == "claude-3-5-haiku-20241022"
