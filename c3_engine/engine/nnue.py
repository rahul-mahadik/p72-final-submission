"""NNUE-lite evaluator.

Architecture (lite variant of Stockfish-style NNUE):
  - 768 input features = 12 piece types x 64 squares (color encoded into piece).
  - Two perspectives: white-relative and black-relative (mirrored).
  - Feature transformer: 768 -> H=32 (per perspective).
  - Accumulator: concatenated [stm_persp, opp_persp] -> 2H=64.
  - Head: 64 -> 16 (clipped ReLU) -> 16 (clipped ReLU) -> 1.
  - PSQT side-channel: linear sum of per-feature scalars (Stockfish has this too).
  - Final eval = PSQT_sum + NN_head_output  (both in centipawns, side-to-move POV).

Incremental updates: the accumulator (dim 32 per perspective) is maintained
across make/unmake by adding/subtracting feature column vectors of W_ft.
A full refresh from the board position is also provided for correctness.

Weights are bootstrapped from piece-square tables so the engine plays
reasonable chess immediately without training. The NN correction is initialized
near zero, so it acts as a learnable residual on top of PSQT.
"""

from __future__ import annotations

import numpy as np
import chess

# --- Piece-square tables (centipawns, white POV, a1 = index 0, h8 = 63) ---
# Standard simplified evaluation function PSQT.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # king material excluded; mate handled by search
}

# PSQT in white POV, indexed by square (a1=0..h8=63). Black side mirrors vertically.
_PAWN_PSQT = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_KNIGHT_PSQT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
_BISHOP_PSQT = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
_ROOK_PSQT = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_QUEEN_PSQT = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
_KING_PSQT = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
]

_PSQT_BY_PIECE = {
    chess.PAWN: _PAWN_PSQT,
    chess.KNIGHT: _KNIGHT_PSQT,
    chess.BISHOP: _BISHOP_PSQT,
    chess.ROOK: _ROOK_PSQT,
    chess.QUEEN: _QUEEN_PSQT,
    chess.KING: _KING_PSQT,
}

# Network shape constants.
N_FEATURES = 768  # 12 piece-color combos * 64 squares
H = 32            # feature transformer hidden width
H1 = 16           # head hidden width
CLIP = 127        # clipped-ReLU saturation (NNUE convention; here floats but same idea)


def feature_index(piece_type: int, color: bool, square: int, perspective: bool) -> int:
    """Map (piece, color, square) into 0..767 index from the given perspective.

    From perspective `perspective`:
      - "us" pieces (color == perspective) occupy indices 0..383
      - "them" pieces                     occupy indices 384..767
      - within each: piece_type-1 in 0..5, square mirrored if perspective is BLACK.
    """
    is_us = (color == perspective)
    pt_idx = piece_type - 1  # 0..5
    sq = square if perspective == chess.WHITE else chess.square_mirror(square)
    base = 0 if is_us else 384
    return base + pt_idx * 64 + sq


def _build_psqt_vector() -> np.ndarray:
    """Build a length-768 vector of PSQT values, one perspective.

    Indexed in perspective-local coordinates c = 0..63 (perspective player's
    back rank is rank 1). For each piece type:
      - us slot at coord c:   +mat + psqt[c]
        (our piece sitting on perspective-local square c)
      - them slot at coord c: -(mat + psqt[mirror(c)])
        (enemy piece sitting on our square c, evaluated from their POV: their
        own rank-2 pawn looks like white's rank-2 pawn, hence the mirror)

    This makes the perspective evaluation symmetric: a symmetric board has
    psqt_white == psqt_black, and an advantage for white shows as
    psqt_white > 0 and psqt_black < 0.
    """
    v = np.zeros(N_FEATURES, dtype=np.float32)
    for pt in range(1, 7):
        psqt = _PSQT_BY_PIECE[pt]
        mat = PIECE_VALUES[pt]
        for c in range(64):
            us_idx = (pt - 1) * 64 + c
            them_idx = 384 + (pt - 1) * 64 + c
            v[us_idx] = mat + psqt[c]
            v[them_idx] = -(mat + psqt[chess.square_mirror(c)])
    return v


class NNUELite:
    """Small NNUE network with PSQT side-channel and incremental accumulator.

    Architecture summary:
        features (768) ---> W_ft (768xH) + b_ft  ---> accumulator (H per perspective)
        concat(stm, opp) --> L1 (2H -> H1) --clipped ReLU
                          --> L2 (H1 -> H1)  --clipped ReLU
                          --> L3 (H1 -> 1)   = NN correction (centipawns)
        psqt_score = sum of psqt vector entries for active features (one perspective)
        eval_cp = psqt_score (stm POV) + nn_correction
    """

    def __init__(self, seed: int = 0x5C3) -> None:
        rng = np.random.default_rng(seed)
        # Feature transformer: small random init so NN correction stays near zero.
        self.W_ft = (rng.standard_normal((N_FEATURES, H)).astype(np.float32) * 0.01)
        self.b_ft = np.zeros(H, dtype=np.float32)
        # Head layers.
        self.W1 = (rng.standard_normal((2 * H, H1)).astype(np.float32) * 0.05)
        self.b1 = np.zeros(H1, dtype=np.float32)
        self.W2 = (rng.standard_normal((H1, H1)).astype(np.float32) * 0.05)
        self.b2 = np.zeros(H1, dtype=np.float32)
        self.W3 = (rng.standard_normal((H1, 1)).astype(np.float32) * 0.05)
        self.b3 = np.zeros(1, dtype=np.float32)
        # PSQT side-channel.
        self.psqt = _build_psqt_vector()

    # --- Pure forward pass from a feature index list (used for refresh) ---
    def _ft_forward(self, indices: list[int]) -> np.ndarray:
        acc = self.b_ft.copy()
        if indices:
            acc += self.W_ft[indices].sum(axis=0)
        return acc

    def _psqt_score(self, indices_white_persp: list[int]) -> float:
        if not indices_white_persp:
            return 0.0
        return float(self.psqt[indices_white_persp].sum())

    @staticmethod
    def _crelu(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, float(CLIP))

    def head(self, acc_stm: np.ndarray, acc_opp: np.ndarray) -> float:
        x = np.concatenate([acc_stm, acc_opp])
        x = self._crelu(x @ self.W1 + self.b1)
        x = self._crelu(x @ self.W2 + self.b2)
        out = float((x @ self.W3 + self.b3)[0])
        return out


class Accumulator:
    """Maintains per-perspective feature transformer activations incrementally.

    Two arrays of shape (H,): one from white's perspective, one from black's.
    Plus the PSQT running sums (scalar each).
    """

    __slots__ = ("white", "black", "psqt_white", "psqt_black", "net")

    def __init__(self, net: NNUELite) -> None:
        self.net = net
        self.white = net.b_ft.copy()
        self.black = net.b_ft.copy()
        self.psqt_white = 0.0
        self.psqt_black = 0.0

    def refresh(self, board: chess.Board) -> None:
        """Recompute accumulator and PSQT from full board state."""
        idx_w: list[int] = []
        idx_b: list[int] = []
        for sq, piece in board.piece_map().items():
            idx_w.append(feature_index(piece.piece_type, piece.color, sq, chess.WHITE))
            idx_b.append(feature_index(piece.piece_type, piece.color, sq, chess.BLACK))
        self.white = self.net._ft_forward(idx_w)
        self.black = self.net._ft_forward(idx_b)
        self.psqt_white = self.net._psqt_score(idx_w)
        self.psqt_black = self.net._psqt_score(idx_b)

    def add_piece(self, piece_type: int, color: bool, sq: int) -> None:
        iw = feature_index(piece_type, color, sq, chess.WHITE)
        ib = feature_index(piece_type, color, sq, chess.BLACK)
        self.white += self.net.W_ft[iw]
        self.black += self.net.W_ft[ib]
        self.psqt_white += float(self.net.psqt[iw])
        self.psqt_black += float(self.net.psqt[ib])

    def remove_piece(self, piece_type: int, color: bool, sq: int) -> None:
        iw = feature_index(piece_type, color, sq, chess.WHITE)
        ib = feature_index(piece_type, color, sq, chess.BLACK)
        self.white -= self.net.W_ft[iw]
        self.black -= self.net.W_ft[ib]
        self.psqt_white -= float(self.net.psqt[iw])
        self.psqt_black -= float(self.net.psqt[ib])

    def evaluate(self, side_to_move: bool) -> float:
        """Return centipawn evaluation from the side-to-move's POV."""
        if side_to_move == chess.WHITE:
            stm, opp = self.white, self.black
            psqt_stm = self.psqt_white
        else:
            stm, opp = self.black, self.white
            psqt_stm = self.psqt_black
        nn = self.net.head(stm, opp)
        return psqt_stm + nn

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Cheap copy for make/unmake stack (numpy copy + 2 floats)."""
        return (self.white.copy(), self.black.copy(),
                self.psqt_white, self.psqt_black)

    def restore(self, snap: tuple[np.ndarray, np.ndarray, float, float]) -> None:
        self.white, self.black, self.psqt_white, self.psqt_black = (
            snap[0], snap[1], snap[2], snap[3]
        )
