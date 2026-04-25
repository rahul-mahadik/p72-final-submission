"""Handcrafted static evaluation helpers shared by simple engines."""
from __future__ import annotations

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables for positional evaluation
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,  0,  5,  5,  0, -20, -40,
    -30,  5, 10, 15, 15, 10,  5, -30,
    -30,  0, 15, 20, 20, 15,  0, -30,
    -30,  5, 15, 20, 20, 15,  5, -30,
    -30,  0, 10, 15, 15, 10,  0, -30,
    -40, -20,  0,  0,  0,  0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5, 10, 10,  5,  0, -10,
    -10,  5,  5, 10, 10,  5,  5, -10,
    -10,  0, 10, 10, 10, 10,  0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10,  5,  0,  0,  0,  0,  5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]

ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10,  0,  0,  0,  0,  0,  0, -10,
    -10,  0,  5,  5,  5,  5,  0, -10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0, -10,
    -10,  0,  5,  0,  0,  0,  0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
]

KING_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

KING_ENDGAME_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,  0,  0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30,  0,  0,  0,  0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
]


def evaluate_material(board: chess.Board) -> int:
    """Return material balance from White's perspective."""
    if board.is_checkmate():
        return -100_000 if board.turn == chess.WHITE else 100_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def evaluate_handcrafted(board: chess.Board) -> int:
    """Return a comprehensive evaluation with piece-square tables and strategic features."""
    score = evaluate_material(board)
    
    # Check if we're in endgame (simplified: few pieces remaining)
    total_pieces = len(board.piece_map())
    is_endgame = total_pieces <= 14
    
    # Apply piece-square tables
    for square in board.pieces(chess.PAWN, chess.WHITE):
        score += PAWN_TABLE[square]
    for square in board.pieces(chess.PAWN, chess.BLACK):
        score -= PAWN_TABLE[chess.square_mirror(square)]
    
    for square in board.pieces(chess.KNIGHT, chess.WHITE):
        score += KNIGHT_TABLE[square]
    for square in board.pieces(chess.KNIGHT, chess.BLACK):
        score -= KNIGHT_TABLE[chess.square_mirror(square)]
    
    for square in board.pieces(chess.BISHOP, chess.WHITE):
        score += BISHOP_TABLE[square]
    for square in board.pieces(chess.BISHOP, chess.BLACK):
        score -= BISHOP_TABLE[chess.square_mirror(square)]
    
    for square in board.pieces(chess.ROOK, chess.WHITE):
        score += ROOK_TABLE[square]
    for square in board.pieces(chess.ROOK, chess.BLACK):
        score -= ROOK_TABLE[chess.square_mirror(square)]
    
    for square in board.pieces(chess.QUEEN, chess.WHITE):
        score += QUEEN_TABLE[square]
    for square in board.pieces(chess.QUEEN, chess.BLACK):
        score -= QUEEN_TABLE[chess.square_mirror(square)]
    
    # Use appropriate king table based on game phase
    king_table = KING_ENDGAME_TABLE if is_endgame else KING_TABLE
    for square in board.pieces(chess.KING, chess.WHITE):
        score += king_table[square]
    for square in board.pieces(chess.KING, chess.BLACK):
        score -= king_table[chess.square_mirror(square)]
    
    # Mobility evaluation
    turn = board.turn
    board.turn = chess.WHITE
    white_mobility = board.legal_moves.count()
    board.turn = chess.BLACK
    black_mobility = board.legal_moves.count()
    board.turn = turn
    score += 2 * (white_mobility - black_mobility)
    
    # Check penalty
    if board.is_check():
        score += -25 if board.turn == chess.WHITE else 25
    
    # Bishop pair bonus
    white_bishops = board.pieces(chess.BISHOP, chess.WHITE)
    black_bishops = board.pieces(chess.BISHOP, chess.BLACK)
    if len(white_bishops) >= 2:
        score += 50
    if len(black_bishops) >= 2:
        score -= 50
    
    # Passed pawn bonus
    for square in board.pieces(chess.PAWN, chess.WHITE):
        if is_passed_pawn(board, square, chess.WHITE):
            rank = chess.square_rank(square)
            score += 20 * rank  # More valuable as they advance
    
    for square in board.pieces(chess.PAWN, chess.BLACK):
        if is_passed_pawn(board, square, chess.BLACK):
            rank = 7 - chess.square_rank(square)
            score -= 20 * rank
    
    # Doubled pawn penalty
    for file in range(8):
        white_pawns_on_file = sum(1 for sq in board.pieces(chess.PAWN, chess.WHITE) 
                                  if chess.square_file(sq) == file)
        black_pawns_on_file = sum(1 for sq in board.pieces(chess.PAWN, chess.BLACK) 
                                  if chess.square_file(sq) == file)
        if white_pawns_on_file > 1:
            score -= 10 * (white_pawns_on_file - 1)
        if black_pawns_on_file > 1:
            score += 10 * (black_pawns_on_file - 1)
    
    return score


def is_passed_pawn(board: chess.Board, square: int, color: chess.Color) -> bool:
    """Check if a pawn is passed (no enemy pawns can stop it)."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    # Check files adjacent and same file
    files_to_check = [f for f in [file - 1, file, file + 1] if 0 <= f <= 7]
    
    if color == chess.WHITE:
        # Check ranks ahead
        for r in range(rank + 1, 8):
            for f in files_to_check:
                if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.BLACK):
                    return False
    else:
        # Check ranks behind (for black, moving down)
        for r in range(rank - 1, -1, -1):
            for f in files_to_check:
                if board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, chess.WHITE):
                    return False
    
    return True


def score_for_side(board: chess.Board) -> int:
    """Return the handcrafted score from the side-to-move perspective."""
    score = evaluate_handcrafted(board)
    return score if board.turn == chess.WHITE else -score
