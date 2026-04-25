"""NNUE-lite candidate using a tiny feature-based value function."""
from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import chess

from engine.common.interface import ChessEngine, safe_legal_move


class NNUELiteEngine(ChessEngine):
    """Enhanced iterative deepening search with quiescence and improved evaluation."""

    name = "nnue_lite_value"

    weights = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    # Piece-square tables (from white's perspective)
    pawn_table = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]

    knight_table = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ]

    bishop_table = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ]

    king_midgame_table = [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]

    def __init__(self):
        """Initialize search tables used by the lightweight value engine."""
        self.transposition_table: Dict[str, Tuple[float, int, str]] = {}
        self.max_table_size = 50000
        self.history_table: Dict[Tuple[int, int], int] = {}
        self.killer_moves: Dict[int, list] = {}
        self.node_count = 0
        self.null_move_allowed = True

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        """Run iterative deepening with a feature evaluator and return a move."""
        deadline = time.perf_counter() + max(time_budget_ms, 20) / 1000
        
        # Clean transposition table more intelligently
        if len(self.transposition_table) > self.max_table_size:
            # Keep the most recent half of entries
            entries = list(self.transposition_table.items())
            self.transposition_table = dict(entries[len(entries)//2:])
        
        self.history_table.clear()
        self.killer_moves.clear()
        self.node_count = 0
        
        best_move: chess.Move | None = None
        best_score = -math.inf
        
        # Initialize aspiration window
        alpha_window = 50
        beta_window = 50
        aspiration_alpha = -math.inf
        aspiration_beta = math.inf
        
        # Iterative deepening
        for depth in range(1, 12):
            if time.perf_counter() >= deadline - 0.001:
                break
            
            current_best_move = None
            current_best_score = -math.inf
            
            # Use aspiration windows after depth 3
            if depth > 3 and best_score > -9000:
                aspiration_alpha = best_score - alpha_window
                aspiration_beta = best_score + beta_window
            else:
                aspiration_alpha = -math.inf
                aspiration_beta = math.inf
            
            # Try search with aspiration window
            fail_high = False
            fail_low = False
            
            while True:
                alpha = aspiration_alpha
                beta = aspiration_beta
                
                # Order moves with best move from previous iteration first
                moves = self._ordered_moves(position, depth, best_move)
                
                for move in moves:
                    if time.perf_counter() >= deadline:
                        break
                    
                    position.push(move)
                    self.node_count += 1
                    
                    # Search with alpha-beta
                    score = -self._alphabeta(position, depth - 1, -beta, -alpha, deadline, True)
                    
                    position.pop()
                    
                    if score > current_best_score:
                        current_best_score = score
                        current_best_move = move
                    
                    alpha = max(alpha, score)
                    
                    # Time check
                    if time.perf_counter() >= deadline:
                        break
                
                # Check for aspiration window failures
                if depth > 3 and best_score > -9000:
                    if current_best_score <= aspiration_alpha:
                        fail_low = True
                        # Widen window on the low side
                        aspiration_alpha = -math.inf
                        if time.perf_counter() >= deadline:
                            break
                    elif current_best_score >= aspiration_beta:
                        fail_high = True
                        # Widen window on the high side
                        aspiration_beta = math.inf
                        if time.perf_counter() >= deadline:
                            break
                    else:
                        # Search succeeded within window
                        break
                else:
                    break
                
                # Don't retry if we're running out of time
                if fail_high and fail_low:
                    break
                if time.perf_counter() >= deadline - 0.001:
                    break
            
            # Only update best move if we completed this depth
            if current_best_move is not None and time.perf_counter() < deadline:
                best_move = current_best_move
                best_score = current_best_score
                
                # Adjust aspiration windows for next iteration
                if depth > 2:
                    if fail_low:
                        alpha_window *= 2
                    elif fail_high:
                        beta_window *= 2
                    else:
                        # Narrow windows if search succeeded
                        alpha_window = max(25, alpha_window * 3 // 4)
                        beta_window = max(25, beta_window * 3 // 4)
        
        return safe_legal_move(position, best_move)

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                   deadline: float, allow_null: bool = True) -> float:
        """Alpha-beta minimax with null move pruning and quiescence search at leaves."""
        # Check time
        if time.perf_counter() >= deadline:
            return self._evaluate(board)
        
        # Check transposition table
        fen = board.fen()
        if fen in self.transposition_table:
            cached_score, cached_depth, bound_type = self.transposition_table[fen]
            if cached_depth >= depth:
                if bound_type == 'exact':
                    return cached_score
                elif bound_type == 'lower' and cached_score >= beta:
                    return cached_score
                elif bound_type == 'upper' and cached_score <= alpha:
                    return cached_score
        
        # Terminal node checks
        if board.is_checkmate():
            return -10000.0 + board.ply()
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves():
            return 0.0
        
        # Leaf node - use quiescence search
        if depth <= 0:
            return self._quiescence(board, alpha, beta, deadline, 0)
        
        self.node_count += 1
        
        # Null move pruning
        if (allow_null and depth >= 3 and not board.is_check() and 
            self._has_non_pawn_material(board, board.turn)):
            
            # Make null move
            board.push(chess.Move.null())
            
            # Search with reduced depth
            null_reduction = 2 if depth >= 6 else 2
            score = -self._alphabeta(board, depth - 1 - null_reduction, -beta, -beta + 1, 
                                    deadline, False)
            
            board.pop()
            
            # Cutoff if null move is good enough
            if score >= beta:
                return beta
        
        best_score = -math.inf
        bound_type = 'upper'
        
        moves = self._ordered_moves(board, depth)
        move_count = 0
        
        for move in moves:
            if time.perf_counter() >= deadline:
                break
            
            board.push(move)
            move_count += 1
            
            # Late move reduction for quiet moves
            reduction = 0
            if move_count > 4 and not board.is_capture(move) and not board.is_check():
                reduction = 1 if depth >= 3 else 0
            
            # Principal variation search for moves after the first
            if move_count == 1:
                score = -self._alphabeta(board, depth - 1, -beta, -alpha, deadline, True)
            else:
                # Search with null window first
                score = -self._alphabeta(board, depth - 1 - reduction, -alpha - 1, -alpha, 
                                        deadline, True)
                
                # Re-search if it improves alpha
                if score > alpha and score < beta:
                    score = -self._alphabeta(board, depth - 1, -beta, -alpha, deadline, True)
            
            board.pop()
            
            if score > best_score:
                best_score = score
                
                # Update history for good quiet moves
                if not board.is_capture(move):
                    self.history_table[(move.from_square, move.to_square)] = \
                        self.history_table.get((move.from_square, move.to_square), 0) + depth * depth
                
                if score >= beta:
                    # Beta cutoff - this is a killer move
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop()
                    
                    bound_type = 'lower'
                    break
                
                if score > alpha:
                    alpha = score
                    bound_type = 'exact'
        
        # Store in transposition table
        self.transposition_table[fen] = (best_score, depth, bound_type)
        
        return best_score

    def _quiescence(self, board: chess.Board, alpha: float, beta: float, deadline: float, depth: int) -> float:
        """Quiescence search to avoid horizon effect."""
        if time.perf_counter() >= deadline or depth > 5:
            return self._evaluate(board)
        
        self.node_count += 1
        stand_pat = self._evaluate(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Only search captures and checks
        moves = [m for m in board.legal_moves if board.is_capture(m) or board.gives_check(m)]
        moves.sort(key=lambda m: self._mvv_lva_score(board, m), reverse=True)
        
        for move in moves:
            if time.perf_counter() >= deadline:
                break
            
            # Futility pruning in quiescence
            if board.is_capture(move) and not board.gives_check(move):
                captured = board.piece_at(move.to_square)
                if captured and stand_pat + self.weights[captured.piece_type] + 200 < alpha:
                    continue
            
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, deadline, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha

    def _has_non_pawn_material(self, board: chess.Board, color: chess.Color) -> bool:
        """Check if the given side has non-pawn material for null move pruning."""
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            if board.pieces(piece_type, color):
                return True
        return False

    def _evaluate(self, board: chess.Board) -> float:
        """Enhanced evaluation with piece-square tables and positional features."""
        if board.is_checkmate():
            return -10000.0 + board.ply()
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        else:
            # Cache piece locations to avoid repeated lookups
            white_pawns = list(board.pieces(chess.PAWN, chess.WHITE))
            black_pawns = list(board.pieces(chess.PAWN, chess.BLACK))
            white_knights = list(board.pieces(chess.KNIGHT, chess.WHITE))
            black_knights = list(board.pieces(chess.KNIGHT, chess.BLACK))
            white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
            black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))
            white_rooks = list(board.pieces(chess.ROOK, chess.WHITE))
            black_rooks = list(board.pieces(chess.ROOK, chess.BLACK))
            white_queens = list(board.pieces(chess.QUEEN, chess.WHITE))
            black_queens = list(board.pieces(chess.QUEEN, chess.BLACK))
            white_king = list(board.pieces(chess.KING, chess.WHITE))
            black_king = list(board.pieces(chess.KING, chess.BLACK))
            
            # Material balance
            material = 0.0
            material += (len(white_pawns) - len(black_pawns)) * self.weights[chess.PAWN]
            material += (len(white_knights) - len(black_knights)) * self.weights[chess.KNIGHT]
            material += (len(white_bishops) - len(black_bishops)) * self.weights[chess.BISHOP]
            material += (len(white_rooks) - len(black_rooks)) * self.weights[chess.ROOK]
            material += (len(white_queens) - len(black_queens)) * self.weights[chess.QUEEN]
            
            # Piece-square tables
            positional = 0.0
            for square in white_pawns:
                positional += self.pawn_table[square]
            for square in black_pawns:
                positional -= self.pawn_table[chess.square_mirror(square)]
            
            for square in white_knights:
                positional += self.knight_table[square]
            for square in black_knights:
                positional -= self.knight_table[chess.square_mirror(square)]
            
            for square in white_bishops:
                positional += self.bishop_table[square]
            for square in black_bishops:
                positional -= self.bishop_table[chess.square_mirror(square)]
            
            for square in white_king:
                positional += self.king_midgame_table[square]
            for square in black_king:
                positional -= self.king_midgame_table[chess.square_mirror(square)]
            
            # Optimized mobility calculation - avoid turn switching
            mobility = 0.0
            if not board.is_check():
                # Approximate mobility using attack maps which is more efficient
                for square in chess.SQUARES:
                    white_attackers = len(board.attackers(chess.WHITE, square))
                    black_attackers = len(board.attackers(chess.BLACK, square))
                    mobility += (white_attackers - black_attackers) * 0.5
            
            # King safety
            king_safety = self._king_safety_fast(board, white_king, black_king)
            
            # Center control bonus
            center_control = self._center_control(board)
            
            # Pawn structure evaluation using cached pawn lists
            pawn_structure = self._pawn_structure(board, white_pawns, black_pawns)
            
            score = material + positional + mobility + king_safety + center_control + pawn_structure
            
            # Check penalty/bonus
            if board.is_check():
                score += -30 if board.turn == chess.WHITE else 30
            
            # Convert to side-to-move perspective
            return score if board.turn == chess.WHITE else -score

    def _king_safety_fast(self, board: chess.Board, white_king: list, black_king: list) -> float:
        """Evaluate king safety based on attackers and pawn shield."""
        score = 0.0
        
        # White king safety
        if white_king:
            king_sq = white_king[0]
            attackers = board.attackers(chess.BLACK, king_sq)
            score -= len(attackers) * 15
            
            # Bonus for pawn shield
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            if king_rank < 2:
                shield_bonus = 0
                for file_offset in [-1, 0, 1]:
                    shield_file = king_file + file_offset
                    if 0 <= shield_file <= 7:
                        shield_sq = chess.square(shield_file, king_rank + 1)
                        piece = board.piece_at(shield_sq)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                            shield_bonus += 5
                score += shield_bonus
        
        # Black king safety
        if black_king:
            king_sq = black_king[0]
            attackers = board.attackers(chess.WHITE, king_sq)
            score += len(attackers) * 15
            
            # Bonus for pawn shield
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            if king_rank > 5:
                shield_bonus = 0
                for file_offset in [-1, 0, 1]:
                    shield_file = king_file + file_offset
                    if 0 <= shield_file <= 7:
                        shield_sq = chess.square(shield_file, king_rank - 1)
                        piece = board.piece_at(shield_sq)
                        if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                            shield_bonus += 5
                score -= shield_bonus
        
        return score

    def _center_control(self, board: chess.Board) -> float:
        """Bonus for controlling central squares."""
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        score = 0.0
        for sq in center_squares:
            piece = board.piece_at(sq)
            if piece:
                value = self.weights.get(piece.piece_type, 0) * 0.02
                score += value if piece.color == chess.WHITE else -value
            # Control bonus
            white_control = len(board.attackers(chess.WHITE, sq))
            black_control = len(board.attackers(chess.BLACK, sq))
            score += (white_control - black_control) * 3
        
        for sq in extended_center:
            piece = board.piece_at(sq)
            if piece:
                value = self.weights.get(piece.piece_type, 0) * 0.01
                score += value if piece.color == chess.WHITE else -value
        
        return score

    def _pawn_structure(self, board: chess.Board, white_pawns: list, black_pawns: list) -> float:
        """Evaluate pawn structure using cached pawn lists."""
        score = 0.0
        
        # Track pawn positions by file
        white_pawns_by_file = [[] for _ in range(8)]
        black_pawns_by_file = [[] for _ in range(8)]
        
        for square in white_pawns:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            white_pawns_by_file[file].append(rank)
        
        for square in black_pawns:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            black_pawns_by_file[file].append(rank)
        
        # Evaluate each file
        for file in range(8):
            # Doubled pawns penalty
            if len(white_pawns_by_file[file]) > 1:
                score -= 10 * (len(white_pawns_by_file[file]) - 1)
            if len(black_pawns_by_file[file]) > 1:
                score += 10 * (len(black_pawns_by_file[file]) - 1)
            
            # Isolated pawns penalty
            adjacent_files = [f for f in [file - 1, file + 1] if 0 <= f <= 7]
            
            if white_pawns_by_file[file]:
                has_support = any(white_pawns_by_file[f] for f in adjacent_files)
                if not has_support:
                    score -= 15
            
            if black_pawns_by_file[file]:
                has_support = any(black_pawns_by_file[f] for f in adjacent_files)
                if not has_support:
                    score += 15
            
            # Passed pawns bonus
            for rank in white_pawns_by_file[file]:
                is_passed = True
                for check_file in [file - 1, file, file + 1]:
                    if 0 <= check_file <= 7:
                        for black_rank in black_pawns_by_file[check_file]:
                            if black_rank > rank:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    score += 20 * rank  # More valuable as they advance
            
            for rank in black_pawns_by_file[file]:
                is_passed = True
                for check_file in [file - 1, file, file + 1]:
                    if 0 <= check_file <= 7:
                        for white_rank in white_pawns_by_file[check_file]:
                            if white_rank < rank:
                                is_passed = False
                                break
                    if not is_passed:
                        break
                if is_passed:
                    score -= 20 * (7 - rank)  # More valuable as they advance
        
        return score

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """MVV-LVA (Most Valuable Victim - Least Valuable Attacker) scoring."""
        if board.is_en_passant(move):
            return 105  # Pawn takes pawn
        
        victim = board.piece_at(move.to_square)
        if victim is None:
            return 0
        
        attacker = board.piece_at(move.from_square)
        if attacker is None:
            return 0
        
        victim_value = self.weights.get(victim.piece_type, 0)
        attacker_value = self.weights.get(attacker.piece_type, 0)
        
        # Higher score for capturing valuable pieces with less valuable pieces
        return victim_value * 10 - attacker_value

    def _ordered_moves(self, board: chess.Board, depth: int = 0, best_move: chess.Move | None = None) -> list[chess.Move]:
        """Order moves with best move first, then MVV-LVA, killers, history, then others."""
        captures = []
        killers = []
        others = []
        
        # Get killer moves for this depth
        killer_list = self.killer_moves.get(depth, [])
        
        for move in board.legal_moves:
            # Best move from previous iteration goes first
            if best_move and move == best_move:
                continue
            
            if board.is_capture(move):
                captures.append((move, self._mvv_lva_score(board, move)))
            elif move in killer_list:
                killers.append(move)
            else:
                # Score by history
                history_score = self.history_table.get((move.from_square, move.to_square), 0)
                others.append((move, history_score))
        
        # Sort captures by MVV-LVA score
        captures.sort(key=lambda x: x[1], reverse=True)
        capture_moves = [m for m, _ in captures]
        
        # Sort others by history score
        others.sort(key=lambda x: x[1], reverse=True)
        other_moves = [m for m, _ in others]
        
        # Best move first if provided
        result = []
        if best_move and best_move in board.legal_moves:
            result.append(best_move)
        
        return result + capture_moves + killers + other_moves
