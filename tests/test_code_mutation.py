from __future__ import annotations

from agents.code_mutation import (
    apply_file_replacements,
    candidate_allowed_prefixes,
    collect_candidate_source_context,
    extract_file_replacements,
    revert_changes,
)


def test_extract_file_replacements_accepts_expected_shape() -> None:
    content = {"file_replacements": [{"path": "engine/common/eval.py", "content": "x = 1\n"}]}
    assert extract_file_replacements(content) == [{"path": "engine/common/eval.py", "content": "x = 1\n"}]


def test_apply_file_replacements_rejects_unsafe_paths(tmp_path) -> None:
    summary, changes = apply_file_replacements(
        [
            {"path": "../escape.py", "content": "x = 1\n"},
            {"path": "README.md", "content": "x = 1\n"},
            {"path": "engine/candidates/alphabeta/empty.py", "content": ""},
        ],
        root=tmp_path,
        allowed_prefixes=candidate_allowed_prefixes("alphabeta"),
    )

    assert not changes
    assert len(summary["rejected"]) == 3


def test_apply_and_revert_file_replacement(tmp_path) -> None:
    target = tmp_path / "engine/candidates/alphabeta/engine.py"
    target.parent.mkdir(parents=True)
    target.write_text("old = True\n", encoding="utf-8")

    summary, changes = apply_file_replacements(
        [{"path": "engine/candidates/alphabeta/engine.py", "content": "new = True\n"}],
        root=tmp_path,
        allowed_prefixes=candidate_allowed_prefixes("alphabeta"),
    )

    assert summary["applied"][0]["path"] == "engine/candidates/alphabeta/engine.py"
    assert target.read_text(encoding="utf-8") == "new = True\n"

    revert_changes(changes)

    assert target.read_text(encoding="utf-8") == "old = True\n"


def test_collect_candidate_source_context_contains_candidate_engine() -> None:
    files = collect_candidate_source_context("alphabeta")
    paths = {row["path"] for row in files}
    assert "engine/candidates/alphabeta/engine.py" in paths
    assert "engine/common/interface.py" in paths
