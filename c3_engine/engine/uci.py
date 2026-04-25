"""UCI protocol implementation.

Supports the subset required for play and tournament use:
    uci, isready, ucinewgame, position [startpos|fen ...] [moves ...],
    go [depth N | nodes N | movetime N | wtime/btime/winc/binc/movestogo],
    stop, quit, setoption (no-op for unknown options).
Outputs `info` lines with depth/seldepth/score/nodes/nps/time/pv and a final
`bestmove` line as required by UCI.
"""

from __future__ import annotations

import sys
import threading
from typing import Iterable, IO

import chess

from .search import Searcher, SearchLimits, SearchInfo, MATE_IN_MAX, MATE


ENGINE_NAME = "C3-NNUE-lite"
ENGINE_VERSION = "0.1"
ENGINE_AUTHOR = "Claude (Opus 4.7)"


def _format_score(score_cp: int, mate: int | None) -> str:
    if mate is not None:
        return f"score mate {mate}"
    return f"score cp {score_cp}"


def _format_pv(pv: list[chess.Move]) -> str:
    return " ".join(m.uci() for m in pv)


class UCIEngine:
    def __init__(self, in_stream: IO[str] | None = None,
                 out_stream: IO[str] | None = None) -> None:
        self.in_stream = in_stream or sys.stdin
        self.out_stream = out_stream or sys.stdout
        self.board = chess.Board()
        self.searcher = Searcher()
        self._search_thread: threading.Thread | None = None

    # --- I/O helpers ---
    def _write(self, line: str) -> None:
        self.out_stream.write(line + "\n")
        self.out_stream.flush()

    def _info(self, info: SearchInfo) -> None:
        line = (
            f"info depth {info.depth} seldepth {info.seldepth} "
            f"{_format_score(info.score_cp, info.mate)} "
            f"nodes {info.nodes} nps {info.nps} time {info.time_ms} "
            f"pv {_format_pv(info.pv)}"
        )
        self._write(line)

    # --- Command handlers ---
    def _cmd_uci(self) -> None:
        self._write(f"id name {ENGINE_NAME} {ENGINE_VERSION}")
        self._write(f"id author {ENGINE_AUTHOR}")
        # Minimal options block (extensible).
        self._write("option name Hash type spin default 16 min 1 max 1024")
        self._write("uciok")

    def _cmd_isready(self) -> None:
        self._write("readyok")

    def _cmd_ucinewgame(self) -> None:
        self.searcher.tt.clear()
        self.board = chess.Board()

    def _cmd_position(self, args: list[str]) -> None:
        if not args:
            return
        i = 0
        if args[0] == "startpos":
            self.board = chess.Board()
            i = 1
        elif args[0] == "fen":
            # FEN is 6 fields.
            fen = " ".join(args[1:7])
            try:
                self.board = chess.Board(fen)
            except ValueError:
                return
            i = 7
        else:
            return
        if i < len(args) and args[i] == "moves":
            for uci in args[i + 1:]:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    break
                if move not in self.board.legal_moves:
                    break
                self.board.push(move)

    def _parse_go_args(self, args: list[str]) -> SearchLimits:
        limits = SearchLimits()
        i = 0
        while i < len(args):
            tok = args[i]
            if tok == "depth" and i + 1 < len(args):
                limits.depth = int(args[i + 1]); i += 2
            elif tok == "nodes" and i + 1 < len(args):
                limits.nodes = int(args[i + 1]); i += 2
            elif tok == "movetime" and i + 1 < len(args):
                limits.movetime_ms = int(args[i + 1]); i += 2
            elif tok == "wtime" and i + 1 < len(args):
                limits.wtime_ms = int(args[i + 1]); i += 2
            elif tok == "btime" and i + 1 < len(args):
                limits.btime_ms = int(args[i + 1]); i += 2
            elif tok == "winc" and i + 1 < len(args):
                limits.winc_ms = int(args[i + 1]); i += 2
            elif tok == "binc" and i + 1 < len(args):
                limits.binc_ms = int(args[i + 1]); i += 2
            elif tok == "movestogo" and i + 1 < len(args):
                limits.movestogo = int(args[i + 1]); i += 2
            elif tok == "infinite":
                limits.depth = 64; i += 1
            else:
                i += 1
        return limits

    def _cmd_go(self, args: list[str]) -> None:
        limits = self._parse_go_args(args)

        def run():
            move, _info = self.searcher.search(self.board, limits, self._info)
            uci = move.uci() if move is not None else "0000"
            self._write(f"bestmove {uci}")

        # Synchronous when running under a programmatic harness; in real UCI
        # we'd thread, but a simple synchronous run is fine for tests too.
        self._search_thread = threading.Thread(target=run, daemon=True)
        self._search_thread.start()

    def _cmd_stop(self) -> None:
        self.searcher.stop()
        if self._search_thread is not None:
            self._search_thread.join(timeout=5.0)

    # --- Main loop ---
    def run(self, lines: Iterable[str] | None = None) -> None:
        """Run the UCI loop. If `lines` is provided, drive from that iterable
        instead of stdin (used by tests)."""
        source = lines if lines is not None else iter(self.in_stream)
        for raw in source:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0]
            args = parts[1:]
            if cmd == "uci":
                self._cmd_uci()
            elif cmd == "isready":
                self._cmd_isready()
            elif cmd == "ucinewgame":
                self._cmd_ucinewgame()
            elif cmd == "position":
                self._cmd_position(args)
            elif cmd == "go":
                self._cmd_go(args)
                # Wait for search to finish so output is ordered.
                if self._search_thread is not None:
                    self._search_thread.join()
            elif cmd == "stop":
                self._cmd_stop()
            elif cmd == "quit":
                self._cmd_stop()
                break
            elif cmd == "setoption":
                # Acknowledge but no-op for now.
                pass
            else:
                # Unknown commands are ignored per UCI spec.
                pass


def main() -> None:
    UCIEngine().run()


if __name__ == "__main__":
    main()
