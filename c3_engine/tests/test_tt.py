"""Transposition table tests."""

from __future__ import annotations

import chess

from engine.tt import TranspositionTable, EXACT, LOWER, UPPER


def test_probe_unknown_returns_none():
    tt = TranspositionTable(max_entries=1024)
    assert tt.probe(0xDEADBEEF) is None


def test_store_then_probe():
    tt = TranspositionTable(max_entries=1024)
    move = chess.Move.from_uci("e2e4")
    tt.store(0x1234, depth=4, value=42, bound=EXACT, move=move)
    e = tt.probe(0x1234)
    assert e is not None
    assert e.depth == 4
    assert e.value == 42
    assert e.bound == EXACT
    assert e.move == move


def test_depth_preferred_replacement():
    tt = TranspositionTable(max_entries=1024)
    tt.store(0x42, depth=5, value=10, bound=EXACT, move=None)
    # Shallower entry with same key should NOT overwrite.
    tt.store(0x42, depth=2, value=99, bound=EXACT, move=None)
    e = tt.probe(0x42)
    assert e.value == 10
    # Deeper entry should overwrite.
    tt.store(0x42, depth=6, value=77, bound=EXACT, move=None)
    e = tt.probe(0x42)
    assert e.value == 77


def test_clear():
    tt = TranspositionTable(max_entries=1024)
    tt.store(0x1, 1, 1, EXACT, None)
    assert len(tt) > 0
    tt.clear()
    assert len(tt) == 0
