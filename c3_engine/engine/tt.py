"""Transposition table.

Standard depth-preferred replacement with bound types:
    EXACT: value is the true score
    LOWER: value is a lower bound (failed high in search)
    UPPER: value is an upper bound (failed low in search)
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

EXACT = 0
LOWER = 1
UPPER = 2


@dataclass
class TTEntry:
    key: int
    depth: int
    value: int
    bound: int
    move: chess.Move | None


class TranspositionTable:
    def __init__(self, max_entries: int = 1 << 18) -> None:
        # Cap so RAM doesn't explode. Replacement is by key % max_entries with
        # depth-preferred policy: keep the entry with greater depth.
        self.max_entries = max_entries
        self._table: dict[int, TTEntry] = {}

    def clear(self) -> None:
        self._table.clear()

    def probe(self, key: int) -> TTEntry | None:
        e = self._table.get(key & (self.max_entries - 1))
        if e is not None and e.key == key:
            return e
        return None

    def store(self, key: int, depth: int, value: int, bound: int,
              move: chess.Move | None) -> None:
        slot = key & (self.max_entries - 1)
        existing = self._table.get(slot)
        if existing is not None and existing.key == key and existing.depth > depth:
            return  # keep deeper entry
        self._table[slot] = TTEntry(key=key, depth=depth, value=value,
                                    bound=bound, move=move)

    def __len__(self) -> int:
        return len(self._table)
