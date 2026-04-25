"""Baseline opponent used for cheap local evaluations."""
from __future__ import annotations

import random

import chess

from engine.common.interface import ChessEngine, safe_legal_move


class RandomBaseline(ChessEngine):
    """Random legal-move player for sanity checks and toy win rates."""

    name = "random_baseline"

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        """Choose a random legal move, falling back safely if none exists."""
        return safe_legal_move(position, random.choice(list(position.legal_moves)) if position.legal_moves.count() else None)
