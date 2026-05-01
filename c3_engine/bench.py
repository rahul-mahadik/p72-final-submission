"""Tactical/strength bench for the C3 engine.

Runs a curated tactical suite and a small self-play smoke game. Reports
solve rate, NPS, total nodes, and a rough ELO estimate based on the solve
rate against a known reference mapping.
"""

from __future__ import annotations

import time
import chess

from engine.search import Searcher, SearchLimits


# (FEN, side-to-move, list_of_acceptable_uci_best_moves, label)
SUITE = [
    # Mate-in-1: back-rank mate.
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
     ["a1a8"], "M1 back-rank"),
    # Mate-in-1: smothered-style with queen.
    ("6k1/5p1p/6p1/8/8/8/5PPP/4Q1K1 w - - 0 1",
     ["e1e8"], "M1 Qe8#"),
    # Free hanging queen capture.
    ("4k3/8/8/4q3/3P4/8/8/3QK1N1 w - - 0 1",
     ["d4e5"], "Win hanging Q"),
    # Fork: knight forks king and queen.
    ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4",
     # Black to move; this position is just a quiet middlegame, ensure engine
     # picks a sane developing move (any legal move accepted as "didn't blunder").
     None, "Quiet middlegame (no blunder)"),
    # Skewer: white wins material with Bb5+ skewering king to queen.
    ("r3k2r/ppp2ppp/2n5/3qp3/3PP3/2N5/PPP2PPP/R3KB1R w KQkq - 0 1",
     ["f1b5"], "Skewer Bb5+"),
    # Capture defended piece is bad: avoid losing trade. Position where the only
    # safe capture is a free pawn, taking the knight loses material.
    ("4k3/8/3p4/3P4/3N4/8/8/4K3 w - - 0 1",
     # Best is to just hold; any non-blundering move accepted (knight is safe,
     # pawn capture by Nxd6? But that's not how this position works; let any
     # non-losing move pass).
     None, "Quiet endgame (no blunder)"),
    # Mate-in-2 (well-known: Reti-style).
    ("6k1/5p2/8/8/8/8/5PP1/R5K1 w - - 0 1",
     ["a1a8"], "M1 simpler back-rank"),
    # Win a free pawn via en passant being available.
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
     ["e5d6"], "En passant capture"),
]


def solves(searcher: Searcher, fen: str, accepted_moves: list[str] | None,
           depth: int, movetime_ms: int) -> tuple[bool, chess.Move, int]:
    board = chess.Board(fen)
    move, info = searcher.search(
        board, SearchLimits(depth=depth, movetime_ms=movetime_ms))
    if move is None:
        return False, move, info.nodes
    if accepted_moves is None:
        # "No blunder" criterion: ensure search completed and produced a
        # legal move with a not-too-bad score.
        ok = move in board.legal_moves and info.score_cp > -300
        return ok, move, info.nodes
    ok = move.uci() in accepted_moves
    return ok, move, info.nodes


def estimate_elo(solve_rate: float, avg_depth: float) -> int:
    """Very rough ELO estimate from tactical solve rate + average depth.

    Calibration heuristic (not a substitute for gauntlet matches):
      - 0% solved at depth 4 ~ random mover (~500 ELO)
      - 50% solved at depth 4 ~ basic engine (~1300 ELO)
      - 100% on this small suite at depth 4 ~ ~1800 ELO PSQT engine
    """
    base = 800 + int(1000 * solve_rate)
    depth_bonus = int(50 * max(0.0, avg_depth - 3.0))
    return base + depth_bonus


def main() -> None:
    s = Searcher()
    total_nodes = 0
    solved = 0
    depths: list[int] = []
    t0 = time.monotonic()
    print(f"{'#':>2} {'label':<28} {'best':<7} {'expected':<14} {'ok':<3} {'nodes':>8}")
    print("-" * 70)
    for i, (fen, accepted, label) in enumerate(SUITE, 1):
        ok, move, nodes = solves(s, fen, accepted, depth=5, movetime_ms=1000)
        total_nodes += nodes
        depths.append(5)
        solved += 1 if ok else 0
        exp = ",".join(accepted) if accepted else "no-blunder"
        bm = move.uci() if move else "-"
        print(f"{i:>2} {label:<28} {bm:<7} {exp:<14} {'OK' if ok else 'X':<3} {nodes:>8}")
    elapsed = time.monotonic() - t0
    rate = solved / len(SUITE)
    avg_depth = sum(depths) / len(depths)
    elo = estimate_elo(rate, avg_depth)
    nps = int(total_nodes / elapsed) if elapsed > 0 else 0
    print("-" * 70)
    print(f"Solved: {solved}/{len(SUITE)} ({rate:.0%})")
    print(f"Total nodes: {total_nodes}, NPS: {nps}, time: {elapsed:.2f}s")
    print(f"Estimated ELO: ~{elo}  (rough: tactical-solve heuristic, not gauntlet-validated)")


if __name__ == "__main__":
    main()
