"""Self-play smoke test: engine plays itself for N plies at fixed depth.

Verifies the engine can play a complete game without crashing, makes
only legal moves, and reaches a sensible terminal state or move limit.
"""

from __future__ import annotations

import time
import chess

from engine.search import Searcher, SearchLimits


def play_game(max_plies: int = 80, depth: int = 3) -> dict:
    s = Searcher()
    board = chess.Board()
    moves: list[str] = []
    t0 = time.monotonic()
    while len(moves) < max_plies and not board.is_game_over(claim_draw=True):
        move, info = s.search(board, SearchLimits(depth=depth, movetime_ms=300))
        if move is None or move not in board.legal_moves:
            return {"ok": False, "reason": "no legal move", "moves": moves}
        moves.append(move.uci())
        board.push(move)
    elapsed = time.monotonic() - t0
    return {
        "ok": True,
        "plies": len(moves),
        "result": board.result(claim_draw=True),
        "moves": moves,
        "time_s": round(elapsed, 2),
        "final_fen": board.fen(),
    }


if __name__ == "__main__":
    r = play_game()
    print(f"plies: {r['plies']}, result: {r['result']}, time: {r['time_s']}s")
    print(f"first 20 moves: {' '.join(r['moves'][:20])}")
    print(f"final fen: {r['final_fen']}")
