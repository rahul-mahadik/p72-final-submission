"""Evaluator-level tests."""

from __future__ import annotations

import chess

from engine.eval import Evaluator


def test_evaluate_returns_int():
    ev = Evaluator()
    board = chess.Board()
    v = ev.evaluate(board)
    assert isinstance(v, int)


def test_evaluate_clamped():
    ev = Evaluator()
    board = chess.Board()
    v = ev.evaluate(board)
    assert -30000 <= v <= 30000


def test_apply_undo_roundtrip_complex_sequence():
    """Push several moves with apply_move + undo_move; final state should
    equal a fresh refresh."""
    import numpy as np
    ev = Evaluator()
    board = chess.Board()
    ev.reset(board)
    seq = ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "d7d5",
           "b1c3", "f8b4", "c1g5"]
    snaps = []
    for uci in seq:
        m = chess.Move.from_uci(uci)
        snaps.append(ev.apply_move(board, m))
        board.push(m)
    # Now unwind.
    while snaps:
        board.pop()
        ev.undo_move(snaps.pop())
    # Should match start position.
    ev_fresh = Evaluator(net=ev.net)
    ev_fresh.reset(chess.Board())
    assert np.allclose(ev.acc.white, ev_fresh.acc.white, atol=1e-4)
    assert np.allclose(ev.acc.black, ev_fresh.acc.black, atol=1e-4)


def test_evaluator_used_after_search():
    """Search should leave the evaluator in a consistent state for a fresh
    evaluate() on the original board."""
    from engine.search import Searcher, SearchLimits
    s = Searcher()
    board = chess.Board()
    pre = s.eval.evaluate(board)  # initialize and eval
    s.search(board, SearchLimits(depth=2))
    post = s.eval.evaluate(board)
    # Eval of the same position should be the same.
    assert pre == post
