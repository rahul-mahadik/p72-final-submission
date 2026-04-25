from __future__ import annotations

from tournament.runner import evaluate_engine, head_to_head


def test_evaluate_engine_writes_summary(tmp_path) -> None:
    summary = evaluate_engine("alphabeta", move_budget_ms=20, games_per_position=3, output_dir=tmp_path)
    assert summary["games"] == 12
    assert 0.0 <= summary["win_rate"] <= 1.0
    assert summary["artifact_files"]


def test_evaluate_engine_supports_nnue_baseline_kind(tmp_path) -> None:
    summary = evaluate_engine(
        "alphabeta",
        baseline_kind="nnue_lite",
        move_budget_ms=20,
        games_per_position=1,
        output_dir=tmp_path,
    )
    assert summary["baseline_kind"] == "nnue_lite"
    assert summary["reference_engine"] == "nnue_lite_value"
    assert summary["games"] == 4
    assert summary["artifact_files"]


def test_head_to_head_writes_summary(tmp_path) -> None:
    summary = head_to_head(
        "alphabeta",
        "policy_guided",
        move_budget_ms=20,
        games_per_position=3,
        output_dir=tmp_path,
    )
    assert summary["games"] == 12
    assert 0.0 <= summary["score_for_first"] <= 1.0
    assert 0.0 <= summary["score_for_second"] <= 1.0
    assert summary["score_for_first"] + summary["score_for_second"] == 1.0
    assert summary["winner"] in {"alphabeta", "policy_guided", "tie"}
    assert summary["artifact_files"]


def test_head_to_head_same_kind_scores_are_complementary(tmp_path) -> None:
    summary = head_to_head(
        "alphabeta",
        "alphabeta",
        move_budget_ms=20,
        games_per_position=2,
        output_dir=tmp_path,
    )
    assert summary["games"] == 8
    assert summary["score_for_first"] + summary["score_for_second"] == 1.0
