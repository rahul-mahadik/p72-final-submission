"""Position evaluator: NNUE-lite + incremental accumulator updates.

This module wraps NNUELite and Accumulator with chess-move semantics:
    - apply_move(board, move): call BEFORE board.push(move). Returns a snapshot
      that callers must keep until they unmake the move.
    - undo_move(snapshot): restore the previous accumulator state.
    - evaluate(board): centipawn score from side-to-move POV.

We use snapshot+restore for the unmake path (cheap: ~64 floats copied) and apply
the move incrementally during make. This matches Stockfish's NNUE flow in spirit:
the accumulator is updated by adding/subtracting a small number of feature
columns per move rather than recomputing from scratch.
"""

from __future__ import annotations

import chess

from .nnue import NNUELite, Accumulator


class Evaluator:
    def __init__(self, net: NNUELite | None = None) -> None:
        self.net = net or NNUELite()
        self.acc = Accumulator(self.net)
        self._initialized = False

    def reset(self, board: chess.Board) -> None:
        self.acc.refresh(board)
        self._initialized = True

    def evaluate(self, board: chess.Board) -> int:
        """Return integer centipawn eval from side-to-move POV.

        Always refreshes from the board; safe to call standalone (e.g. from
        tests). Callers in the search loop should rely on the incremental
        path via apply_move/undo_move and then evaluate() will use the live
        accumulator state — but we re-refresh here as a defensive measure
        only when not yet initialized.
        """
        if not self._initialized:
            self.acc.refresh(board)
            self._initialized = True
        cp = self.acc.evaluate(board.turn)
        # Clamp to a sane integer centipawn range.
        if cp > 30000:
            cp = 30000
        elif cp < -30000:
            cp = -30000
        return int(round(cp))

    # --- Incremental make/unmake. Snapshot is opaque to callers. ---

    def apply_move(self, board: chess.Board, move: chess.Move):
        """Update accumulator for `move` BEFORE board.push(move).

        Returns a snapshot to be passed to undo_move() after board.pop().
        """
        snap = self.acc.snapshot()
        # Handle castling specially (king + rook both move).
        if board.is_castling(move):
            color = board.turn
            # King moves
            self.acc.remove_piece(chess.KING, color, move.from_square)
            self.acc.add_piece(chess.KING, color, move.to_square)
            # Rook moves
            if board.is_kingside_castling(move):
                if color == chess.WHITE:
                    rook_from, rook_to = chess.H1, chess.F1
                else:
                    rook_from, rook_to = chess.H8, chess.F8
            else:  # queenside
                if color == chess.WHITE:
                    rook_from, rook_to = chess.A1, chess.D1
                else:
                    rook_from, rook_to = chess.A8, chess.D8
            self.acc.remove_piece(chess.ROOK, color, rook_from)
            self.acc.add_piece(chess.ROOK, color, rook_to)
            return snap

        src = board.piece_at(move.from_square)
        # Remove source piece from its square.
        self.acc.remove_piece(src.piece_type, src.color, move.from_square)

        # Handle capture (regular or en passant).
        if board.is_en_passant(move):
            cap_sq = move.to_square + (-8 if src.color == chess.WHITE else 8)
            self.acc.remove_piece(chess.PAWN, not src.color, cap_sq)
        else:
            cap = board.piece_at(move.to_square)
            if cap is not None:
                self.acc.remove_piece(cap.piece_type, cap.color, move.to_square)

        # Add piece at destination (promotion changes piece type).
        if move.promotion:
            self.acc.add_piece(move.promotion, src.color, move.to_square)
        else:
            self.acc.add_piece(src.piece_type, src.color, move.to_square)

        return snap

    def undo_move(self, snapshot) -> None:
        self.acc.restore(snapshot)
