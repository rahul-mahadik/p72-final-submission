"""Alpha-beta search with iterative deepening.

Features:
    - Negamax framework with alpha-beta cutoffs.
    - Iterative deepening from depth 1 upward.
    - Transposition table cutoffs and best-move ordering.
    - Quiescence search over captures (and promotions) to mitigate horizon effect.
    - Move ordering: TT-move first, then MVV-LVA captures, then killer moves,
      then remaining quiets.
    - Mate scoring relative to root distance: MATE - ply.
    - Time and node-budget aware; soft stop between nodes via a deadline check.

The search uses the Evaluator's incremental accumulator: each node calls
evaluator.apply_move() before board.push(), then evaluator.undo_move() after
board.pop().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import chess
import chess.polyglot

from .eval import Evaluator
from .tt import TranspositionTable, EXACT, LOWER, UPPER


MATE = 30000
MATE_IN_MAX = MATE - 1000  # threshold above which we treat scores as mate


PIECE_ORDER_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}


@dataclass
class SearchLimits:
    depth: int = 64
    nodes: int | None = None
    movetime_ms: int | None = None  # hard wall-clock budget
    # Time-control fields (UCI go wtime/btime/winc/binc/movestogo).
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int = 0
    binc_ms: int = 0
    movestogo: int | None = None


@dataclass
class SearchInfo:
    nodes: int = 0
    depth: int = 0
    seldepth: int = 0
    score_cp: int = 0
    mate: int | None = None
    time_ms: int = 0
    pv: list[chess.Move] = field(default_factory=list)
    nps: int = 0


class Searcher:
    def __init__(self, evaluator: Evaluator | None = None,
                 tt: TranspositionTable | None = None) -> None:
        self.eval = evaluator or Evaluator()
        self.tt = tt or TranspositionTable()
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(128)]
        self.nodes = 0
        self._deadline: float | None = None
        self._stop_flag = False
        self._info_cb: Callable[[SearchInfo], None] | None = None

    def stop(self) -> None:
        self._stop_flag = True

    # --- Time management for UCI go ---
    @staticmethod
    def _budget_ms(limits: SearchLimits, side_to_move: bool) -> int | None:
        if limits.movetime_ms is not None:
            # leave a small safety margin
            return max(1, limits.movetime_ms - 5)
        remaining = (limits.wtime_ms if side_to_move == chess.WHITE
                     else limits.btime_ms)
        inc = (limits.winc_ms if side_to_move == chess.WHITE
               else limits.binc_ms)
        if remaining is None:
            return None
        moves_to_go = limits.movestogo or 30
        # Use a fraction of remaining time plus most of the increment.
        budget = remaining // moves_to_go + (inc * 3) // 4
        # Never spend more than half the clock or less than 5 ms.
        budget = max(5, min(budget, max(5, remaining // 2)))
        return budget

    # --- Move ordering ---
    def _score_move(self, board: chess.Board, move: chess.Move,
                    tt_move: chess.Move | None, ply: int) -> int:
        if tt_move is not None and move == tt_move:
            return 1_000_000
        score = 0
        if board.is_capture(move):
            # En passant: victim is a pawn at ep square.
            if board.is_en_passant(move):
                victim_val = PIECE_ORDER_VALUE[chess.PAWN]
            else:
                cap = board.piece_at(move.to_square)
                victim_val = PIECE_ORDER_VALUE[cap.piece_type] if cap else 0
            attacker = board.piece_at(move.from_square)
            attacker_val = PIECE_ORDER_VALUE[attacker.piece_type] if attacker else 0
            score = 100_000 + victim_val * 16 - attacker_val
        elif move.promotion:
            score = 90_000 + PIECE_ORDER_VALUE.get(move.promotion, 0)
        else:
            killers = self.killers[ply] if ply < len(self.killers) else (None, None)
            if killers[0] == move:
                score = 80_000
            elif killers[1] == move:
                score = 79_000
        return score

    def _order_moves(self, board: chess.Board, moves: list[chess.Move],
                     tt_move: chess.Move | None, ply: int) -> list[chess.Move]:
        return sorted(moves, key=lambda m: -self._score_move(board, m, tt_move, ply))

    # --- Time check ---
    def _time_up(self) -> bool:
        if self._stop_flag:
            return True
        if self._deadline is not None and time.monotonic() >= self._deadline:
            return True
        return False

    # --- Quiescence search ---
    def _qsearch(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        # Periodic time check (cheap modulo).
        if (self.nodes & 4095) == 0 and self._time_up():
            return 0
        stand_pat = self.eval.evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate captures and promotions only.
        moves = [m for m in board.legal_moves
                 if board.is_capture(m) or m.promotion]
        moves = self._order_moves(board, moves, None, min(ply, len(self.killers) - 1))

        for move in moves:
            snap = self.eval.apply_move(board, move)
            board.push(move)
            score = -self._qsearch(board, -beta, -alpha, ply + 1)
            board.pop()
            self.eval.undo_move(snap)
            if self._time_up():
                return 0
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    # --- Main negamax ---
    def _search(self, board: chess.Board, depth: int, alpha: int, beta: int,
                ply: int) -> int:
        # Repetition / 50-move draw.
        if ply > 0 and (board.is_repetition(2) or board.halfmove_clock >= 100):
            return 0

        original_alpha = alpha
        key = chess.polyglot.zobrist_hash(board)

        tt_move: chess.Move | None = None
        tt_entry = self.tt.probe(key)
        if tt_entry is not None:
            tt_move = tt_entry.move
            if tt_entry.depth >= depth and ply > 0:
                v = tt_entry.value
                if tt_entry.bound == EXACT:
                    return v
                if tt_entry.bound == LOWER and v >= beta:
                    return v
                if tt_entry.bound == UPPER and v <= alpha:
                    return v

        if depth <= 0:
            return self._qsearch(board, alpha, beta, ply)

        self.nodes += 1
        if (self.nodes & 4095) == 0 and self._time_up():
            return 0

        legal = list(board.legal_moves)
        if not legal:
            if board.is_check():
                return -MATE + ply  # checkmate; closer to root = worse for us
            return 0  # stalemate

        legal = self._order_moves(board, legal, tt_move, min(ply, len(self.killers) - 1))

        best_score = -MATE - 1
        best_move: chess.Move | None = None
        for move in legal:
            snap = self.eval.apply_move(board, move)
            board.push(move)
            score = -self._search(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            self.eval.undo_move(snap)

            if self._time_up():
                return 0

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
            if alpha >= beta:
                # Killer update on quiet beta-cutoff move.
                if not board.is_capture(move) and not move.promotion:
                    if ply < len(self.killers):
                        k = self.killers[ply]
                        if k[0] != move:
                            k[1] = k[0]
                            k[0] = move
                break

        # Store in TT.
        if best_score <= original_alpha:
            bound = UPPER
        elif best_score >= beta:
            bound = LOWER
        else:
            bound = EXACT
        self.tt.store(key, depth, best_score, bound, best_move)
        return best_score

    # --- Iterative deepening / public API ---
    def search(self, board: chess.Board, limits: SearchLimits,
               info_cb: Callable[[SearchInfo], None] | None = None
               ) -> tuple[chess.Move | None, SearchInfo]:
        self.eval.reset(board)
        self.nodes = 0
        self._stop_flag = False
        self._info_cb = info_cb
        budget_ms = self._budget_ms(limits, board.turn)
        start = time.monotonic()
        if budget_ms is not None:
            self._deadline = start + budget_ms / 1000.0
        else:
            self._deadline = None

        best_move: chess.Move | None = None
        info = SearchInfo()
        max_depth = max(1, min(limits.depth, 64))

        for depth in range(1, max_depth + 1):
            score = self._search(board, depth, -MATE - 1, MATE + 1, 0)
            if self._time_up() and depth > 1:
                break

            # Reconstruct PV from TT.
            pv = self._extract_pv(board, depth)
            elapsed = time.monotonic() - start
            elapsed_ms = max(1, int(elapsed * 1000))
            info = SearchInfo(
                nodes=self.nodes,
                depth=depth,
                seldepth=depth,
                score_cp=score,
                mate=self._mate_in(score),
                time_ms=elapsed_ms,
                pv=pv,
                nps=int(self.nodes * 1000 / elapsed_ms),
            )
            if pv:
                best_move = pv[0]
            elif best_move is None:
                # First-iteration fallback: use TT root move.
                key = chess.polyglot.zobrist_hash(board)
                e = self.tt.probe(key)
                if e and e.move is not None:
                    best_move = e.move
            if info_cb is not None:
                info_cb(info)
            if limits.nodes is not None and self.nodes >= limits.nodes:
                break
            # Stop if we've found a forced mate.
            if info.mate is not None:
                break
            # Soft stop: don't start a new iteration if we've used >50% of budget.
            if budget_ms is not None and (time.monotonic() - start) * 1000 >= budget_ms * 0.5:
                break

        # Final fallback: ensure a legal best move (for very short budgets).
        if best_move is None:
            legal = list(board.legal_moves)
            if legal:
                best_move = legal[0]

        return best_move, info

    @staticmethod
    def _mate_in(score: int) -> int | None:
        if score > MATE_IN_MAX:
            plies = MATE - score
            return (plies + 1) // 2
        if score < -MATE_IN_MAX:
            plies = MATE + score
            return -((plies + 1) // 2)
        return None

    def _extract_pv(self, board: chess.Board, depth: int) -> list[chess.Move]:
        pv: list[chess.Move] = []
        pushed = 0
        for _ in range(depth):
            key = chess.polyglot.zobrist_hash(board)
            entry = self.tt.probe(key)
            if entry is None or entry.move is None:
                break
            move = entry.move
            if move not in board.legal_moves:
                break
            pv.append(move)
            board.push(move)
            pushed += 1
        for _ in range(pushed):
            board.pop()
        return pv
