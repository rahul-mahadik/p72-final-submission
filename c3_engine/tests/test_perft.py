"""Perft sanity check via python-chess.

Validates the move-generation foundation we rely on for search correctness.
Numbers from the well-known perft table for the start position and Kiwipete.
"""

from __future__ import annotations

import chess


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


def test_perft_startpos_depth_3():
    board = chess.Board()
    assert perft(board, 3) == 8902


def test_perft_startpos_depth_4():
    board = chess.Board()
    assert perft(board, 4) == 197281


def test_perft_kiwipete_depth_2():
    # Kiwipete: classic perft test position with castling, ep, promotions.
    board = chess.Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    assert perft(board, 2) == 2039
