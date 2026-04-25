"""Tests for the NNUE-lite evaluator and accumulator."""

from __future__ import annotations

import numpy as np
import chess
import pytest

from engine.nnue import NNUELite, Accumulator, feature_index, N_FEATURES, H


def test_feature_index_in_range():
    for pt in range(1, 7):
        for color in (chess.WHITE, chess.BLACK):
            for sq in range(64):
                for persp in (chess.WHITE, chess.BLACK):
                    idx = feature_index(pt, color, sq, persp)
                    assert 0 <= idx < N_FEATURES


def test_feature_index_us_them_split():
    # White piece from white perspective is "us" => index < 384.
    assert feature_index(chess.PAWN, chess.WHITE, chess.E2, chess.WHITE) < 384
    # White piece from black perspective is "them" => index >= 384.
    assert feature_index(chess.PAWN, chess.WHITE, chess.E2, chess.BLACK) >= 384


def test_accumulator_refresh_initial():
    net = NNUELite()
    acc = Accumulator(net)
    board = chess.Board()
    acc.refresh(board)
    # Bias has shape H; accumulator should be a length-H vector now.
    assert acc.white.shape == (H,)
    assert acc.black.shape == (H,)
    # Initial position is fully symmetric: both perspectives see "my pieces
    # in starting setup vs their pieces in starting setup", so the PSQT sums
    # are equal (and ~0 because each side's pieces cancel the enemy's).
    assert abs(acc.psqt_white - acc.psqt_black) < 1e-3
    assert abs(acc.psqt_white) < 1e-3


def test_evaluate_initial_position_near_zero():
    net = NNUELite()
    acc = Accumulator(net)
    board = chess.Board()
    acc.refresh(board)
    # Start position is symmetric: eval should be small.
    assert abs(acc.evaluate(chess.WHITE)) < 50.0


def test_incremental_matches_refresh_simple_capture():
    """After a sequence of incremental moves, accumulator should equal a
    fresh refresh from the same position."""
    from engine.eval import Evaluator
    ev = Evaluator()
    board = chess.Board()
    ev.reset(board)
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    for uci in moves:
        m = chess.Move.from_uci(uci)
        ev.apply_move(board, m)
        board.push(m)
    inc_white = ev.acc.white.copy()
    inc_black = ev.acc.black.copy()
    inc_psqt_w = ev.acc.psqt_white
    inc_psqt_b = ev.acc.psqt_black
    # Refresh and compare.
    ev2 = Evaluator(net=ev.net)
    ev2.reset(board)
    assert np.allclose(inc_white, ev2.acc.white, atol=1e-4)
    assert np.allclose(inc_black, ev2.acc.black, atol=1e-4)
    assert abs(inc_psqt_w - ev2.acc.psqt_white) < 1e-2
    assert abs(inc_psqt_b - ev2.acc.psqt_black) < 1e-2


def test_incremental_handles_castling():
    from engine.eval import Evaluator
    ev = Evaluator()
    board = chess.Board()
    ev.reset(board)
    # A short sequence ending in white kingside castle.
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]
    for uci in moves:
        m = chess.Move.from_uci(uci)
        ev.apply_move(board, m)
        board.push(m)
    ev2 = Evaluator(net=ev.net)
    ev2.reset(board)
    assert np.allclose(ev.acc.white, ev2.acc.white, atol=1e-4)
    assert np.allclose(ev.acc.black, ev2.acc.black, atol=1e-4)


def test_incremental_handles_en_passant():
    from engine.eval import Evaluator
    ev = Evaluator()
    board = chess.Board()
    ev.reset(board)
    moves = ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]  # ep capture
    for uci in moves:
        m = chess.Move.from_uci(uci)
        assert m in board.legal_moves, f"Move {uci} not legal"
        ev.apply_move(board, m)
        board.push(m)
    ev2 = Evaluator(net=ev.net)
    ev2.reset(board)
    assert np.allclose(ev.acc.white, ev2.acc.white, atol=1e-4)
    assert np.allclose(ev.acc.black, ev2.acc.black, atol=1e-4)


def test_incremental_handles_promotion():
    from engine.eval import Evaluator
    ev = Evaluator()
    # Position with white pawn on 7th about to promote.
    board = chess.Board("8/P7/8/8/8/8/k7/7K w - - 0 1")
    ev.reset(board)
    m = chess.Move.from_uci("a7a8q")
    assert m in board.legal_moves
    ev.apply_move(board, m)
    board.push(m)
    ev2 = Evaluator(net=ev.net)
    ev2.reset(board)
    assert np.allclose(ev.acc.white, ev2.acc.white, atol=1e-4)
    assert np.allclose(ev.acc.black, ev2.acc.black, atol=1e-4)


def test_undo_restores_accumulator():
    from engine.eval import Evaluator
    ev = Evaluator()
    board = chess.Board()
    ev.reset(board)
    pre_w = ev.acc.white.copy()
    pre_b = ev.acc.black.copy()
    m = chess.Move.from_uci("e2e4")
    snap = ev.apply_move(board, m)
    board.push(m)
    # Mutated.
    assert not np.allclose(ev.acc.white, pre_w)
    board.pop()
    ev.undo_move(snap)
    assert np.allclose(ev.acc.white, pre_w)
    assert np.allclose(ev.acc.black, pre_b)
