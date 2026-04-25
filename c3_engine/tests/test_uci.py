"""UCI protocol tests: drive the engine with a script and check output."""

from __future__ import annotations

import io

from engine.uci import UCIEngine


def _drive(commands: list[str]) -> list[str]:
    out = io.StringIO()
    engine = UCIEngine(in_stream=None, out_stream=out)
    engine.run(lines=iter(commands + ["quit"]))
    return out.getvalue().strip().splitlines()


def test_uci_handshake():
    out = _drive(["uci"])
    assert any(line.startswith("id name") for line in out)
    assert any(line.startswith("id author") for line in out)
    assert "uciok" in out


def test_isready():
    out = _drive(["isready"])
    assert "readyok" in out


def test_position_and_go_outputs_bestmove():
    out = _drive([
        "ucinewgame",
        "position startpos",
        "go depth 2",
    ])
    bestmoves = [l for l in out if l.startswith("bestmove")]
    assert len(bestmoves) == 1
    parts = bestmoves[0].split()
    assert len(parts) >= 2
    move_str = parts[1]
    # Must be a 4 or 5 character UCI move.
    assert 4 <= len(move_str) <= 5


def test_position_with_moves_then_go():
    out = _drive([
        "ucinewgame",
        "position startpos moves e2e4 e7e5",
        "go depth 2",
    ])
    info_lines = [l for l in out if l.startswith("info")]
    assert len(info_lines) >= 1
    bestmoves = [l for l in out if l.startswith("bestmove")]
    assert len(bestmoves) == 1


def test_position_fen():
    fen = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
    out = _drive([
        "ucinewgame",
        f"position fen {fen}",
        "go depth 2",
    ])
    bestmoves = [l for l in out if l.startswith("bestmove")]
    assert len(bestmoves) == 1


def test_go_movetime():
    out = _drive([
        "ucinewgame",
        "position startpos",
        "go movetime 100",
    ])
    bestmoves = [l for l in out if l.startswith("bestmove")]
    assert len(bestmoves) == 1


def test_info_line_format():
    out = _drive([
        "ucinewgame",
        "position startpos",
        "go depth 3",
    ])
    info_lines = [l for l in out if l.startswith("info")]
    assert info_lines, "expected at least one info line"
    last = info_lines[-1]
    for token in ("depth", "score", "nodes", "nps", "time", "pv"):
        assert token in last, f"missing {token} in info line: {last}"
