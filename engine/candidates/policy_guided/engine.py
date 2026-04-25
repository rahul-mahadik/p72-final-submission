"""Policy-guided candidate that searches a heuristic top-k move set."""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import chess

from engine.common.eval import evaluate_handcrafted
from engine.common.interface import ChessEngine, safe_legal_move


class PolicyGuidedEngine(ChessEngine):
    """Use policy priors for move filtering, then shallow tactical search."""

    name = "policy_guided_search"

    def __init__(self):
        """Initialize policy-guided search tables and move-ordering memory."""
        self.transposition_table = {}
        self.tt_generation = 0
        # Two-slot killer moves for better coverage
        self.killer_moves = [[None, None] for _ in range(128)]
        self.history_table = {}
        self.counter_moves = {}

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        """Filter likely moves with policy priors, then search tactically."""
        deadline = time.perf_counter() + max(time_budget_ms, 20) / 1000
        soft_deadline = time.perf_counter() + max(time_budget_ms * 0.4, 10) / 1000
        
        # Increment generation for TT aging
        self.tt_generation += 1
        
        # Only clear tables periodically to preserve learned information
        if self.tt_generation % 10 == 0:
            # Keep only recent entries
            self.transposition_table = {k: v for k, v in self.transposition_table.items() 
                                       if v[4] >= self.tt_generation - 5}
            # Decay history scores
            self.history_table = {k: v // 2 for k, v in self.history_table.items() if v > 100}
        
        best_move: chess.Move | None = None
        best_score = -math.inf
        aspiration_window = 50
        
        # Iterative deepening with policy-guided move selection
        for depth in range(1, 12):
            if time.perf_counter() >= soft_deadline and depth > 2:
                break
                
            top_moves = self._policy_top_k(position, k=max(5, 10 - depth // 2))
            depth_best_move = None
            depth_best_score = -math.inf
            
            # Aspiration window search
            if depth > 2 and best_score > -10000:
                alpha = best_score - aspiration_window
                beta = best_score + aspiration_window
            else:
                alpha = -math.inf
                beta = math.inf
            
            for i, move in enumerate(top_moves):
                if time.perf_counter() >= deadline:
                    break
                position.push(move)
                
                # Late move reduction for non-promising moves
                reduction = 0
                if depth >= 3 and i >= 2 and not position.is_check():
                    if not position.is_capture(position.peek()):
                        reduction = 1
                        if i >= 4 and depth >= 4:
                            reduction = 2
                        if i >= 6 and depth >= 5:
                            reduction = 3
                
                score = -self._search(position, depth - 1 - reduction, -beta, -alpha, 1, deadline, False)
                
                # Re-search if reduced search beats alpha
                if reduction > 0 and score > alpha:
                    score = -self._search(position, depth - 1, -beta, -alpha, 1, deadline, i == 0)
                
                position.pop()
                
                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
                    if score > alpha:
                        alpha = score
            
            # Re-search with wider window if score falls outside aspiration window
            if depth > 2 and depth_best_move and (depth_best_score <= best_score - aspiration_window or 
                                                  depth_best_score >= best_score + aspiration_window):
                if time.perf_counter() < soft_deadline:
                    position.push(depth_best_move)
                    depth_best_score = -self._search(position, depth - 1, -math.inf, math.inf, 1, deadline, True)
                    position.pop()
                    aspiration_window = min(500, aspiration_window * 3)
            else:
                aspiration_window = max(25, aspiration_window * 2 // 3)
            
            if depth_best_move and time.perf_counter() < deadline:
                best_move = depth_best_move
                best_score = depth_best_score
                
        return safe_legal_move(position, best_move)

    def _policy_top_k(self, board: chess.Board, k: int) -> list[chess.Move]:
        def prior(move: chess.Move) -> int:
            """Score a move as a cheap policy prior for top-k filtering."""
            score = 0
            # MVV-LVA for captures
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10000 + 100 * victim.piece_type - attacker.piece_type
            # Checks are often forcing
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            # Promotion is usually strong
            if move.promotion:
                score += 8000 if move.promotion == chess.QUEEN else 7000
            # Killer moves bonus (check both slots)
            if self.killer_moves[0][0] == move:
                score += 2000
            elif self.killer_moves[0][1] == move:
                score += 1900
            # Counter move bonus
            if board.move_stack:
                last_move = board.peek()
                if self.counter_moves.get(last_move) == move:
                    score += 1500
            # History heuristic bonus
            score += min(1000, self.history_table.get(move, 0) // 32)
            # Center control bonus
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            if to_file in (3, 4) and to_rank in (3, 4):
                score += 15
            elif to_file in (2, 5) and to_rank in (2, 5):
                score += 8
            # Castling for king safety
            if board.is_castling(move):
                score += 150
            # Develop minor pieces early
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                if chess.square_rank(move.from_square) in (0, 7):
                    score += 20
            return score

        moves = sorted(list(board.legal_moves), key=prior, reverse=True)
        return moves[:k] if moves else []

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int, deadline: float, is_pv: bool) -> float:
        if time.perf_counter() >= deadline:
            return self._relative(board)
            
        # Transposition table lookup
        zobrist = board.zobrist_hash() if hasattr(board, 'zobrist_hash') else hash(board.fen())
        tt_entry = self.transposition_table.get(zobrist)
        tt_move = None
        if tt_entry:
            tt_depth, tt_score, tt_type, tt_move, tt_gen = tt_entry
            if tt_depth >= depth and not is_pv:  # Don't use TT cutoffs in PV nodes
                if tt_type == 'exact':
                    return tt_score
                elif tt_type == 'lower' and tt_score >= beta:
                    return beta
                elif tt_type == 'upper' and tt_score <= alpha:
                    return alpha
        
        if depth <= 0 or board.is_game_over():
            score = self._quiesce(board, alpha, beta, deadline)
            self.transposition_table[zobrist] = (0, score, 'exact', None, self.tt_generation)
            return score
        
        # Null move pruning (not in PV nodes)
        if not is_pv and depth >= 3 and not board.is_check() and ply > 0:
            # Don't use null move in endgame positions (risk of zugzwang)
            material = sum(len(board.pieces(pt, board.turn)) * v for pt, v in 
                         [(chess.QUEEN, 9), (chess.ROOK, 5), (chess.BISHOP, 3), (chess.KNIGHT, 3)])
            if material > 10:  # Not endgame
                board.push(chess.Move.null())
                reduction = 3 if depth >= 6 else 2
                null_score = -self._search(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, deadline, False)
                board.pop()
                if null_score >= beta:
                    return beta
        
        # Extended futility pruning
        if not is_pv and depth <= 3 and not board.is_check():
            stand_pat = self._relative(board)
            margin = 200 * depth
            if stand_pat + margin < alpha:
                max_score = stand_pat + margin
                # Check if any capture can raise the score enough
                for move in board.legal_moves:
                    if board.is_capture(move):
                        victim = board.piece_at(move.to_square)
                        if victim:
                            victim_value = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                                          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}[victim.piece_type]
                            max_score = max(max_score, stand_pat + victim_value)
                if max_score < alpha:
                    return alpha
        
        best_score = -math.inf
        best_move = None
        moves = self._order_moves(board, ply, tt_move)
        moves_searched = 0
        raised_alpha = False
        
        for move in moves:
            if time.perf_counter() >= deadline:
                break
            
            # Futility pruning for quiet moves
            if not is_pv and depth == 1 and moves_searched >= 4:
                if not board.is_capture(move) and not board.gives_check(move):
                    continue
            
            # Late move reduction
            reduction = 0
            if depth >= 3 and moves_searched >= 3 and not board.gives_check(move):
                if not board.is_capture(move):
                    reduction = 1
                    if not is_pv and moves_searched >= 6:
                        reduction = 2
                    if not is_pv and moves_searched >= 12 and depth >= 4:
                        reduction = 3
            
            board.push(move)
            
            if moves_searched == 0:
                # Full window for first move
                score = -self._search(board, depth - 1, -beta, -alpha, ply + 1, deadline, is_pv)
            else:
                # Null window search for other moves
                score = -self._search(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, deadline, False)
                if score > alpha and reduction > 0:
                    # Re-search with full depth if reduced search beats alpha
                    score = -self._search(board, depth - 1, -alpha - 1, -alpha, ply + 1, deadline, False)
                if score > alpha and score < beta:
                    # Re-search with full window if within bounds
                    score = -self._search(board, depth - 1, -beta, -alpha, ply + 1, deadline, is_pv)
            
            board.pop()
            moves_searched += 1
            
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    raised_alpha = True
                    
                    # Update killer moves (two-slot system)
                    if ply < 128 and not board.is_capture(move):
                        if self.killer_moves[ply][0] != move:
                            self.killer_moves[ply][1] = self.killer_moves[ply][0]
                            self.killer_moves[ply][0] = move
                    
                    # Update counter moves
                    if board.move_stack and ply > 0:
                        self.counter_moves[board.peek()] = move
                    
                    # Update history heuristic
                    if not board.is_capture(move):
                        bonus = depth * depth
                        self.history_table[move] = min(10000, self.history_table.get(move, 0) + bonus)
                    
                    if alpha >= beta:
                        # Beta cutoff - update history for moves that weren't searched
                        penalty = -depth * depth // 2
                        for not_searched in moves[moves_searched:]:
                            if not board.is_capture(not_searched):
                                self.history_table[not_searched] = max(-10000, 
                                    self.history_table.get(not_searched, 0) + penalty)
                        break
        
        # Store result in transposition table with replacement scheme
        if best_score > -math.inf:
            if best_score >= beta:
                tt_type = 'lower'
            elif raised_alpha:
                tt_type = 'exact'
            else:
                tt_type = 'upper'
            
            # Replace if: 1) Empty slot, 2) Older generation, or 3) Shallower depth
            should_replace = True
            if tt_entry:
                old_depth, _, _, _, old_gen = tt_entry
                if old_gen == self.tt_generation and old_depth > depth:
                    should_replace = False
            
            if should_replace:
                self.transposition_table[zobrist] = (depth, best_score, tt_type, best_move, self.tt_generation)
        
        return best_score if best_score > -math.inf else self._relative(board)

    def _quiesce(self, board: chess.Board, alpha: float, beta: float, deadline: float, depth: int = 0) -> float:
        if time.perf_counter() >= deadline or depth > 10:
            return self._relative(board)
            
        stand_pat = self._relative(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        # Only search captures and checks in quiescence
        for move in self._order_captures(board):
            if time.perf_counter() >= deadline:
                break
            
            # Delta pruning with margin
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                if victim:
                    victim_value = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                                  chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}[victim.piece_type]
                    # Skip if capture can't raise alpha (with margin for tactics)
                    if stand_pat + victim_value + 200 < alpha and move.promotion != chess.QUEEN:
                        continue
            
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, deadline, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    def _order_moves(self, board: chess.Board, ply: int, tt_move: chess.Move = None) -> list[chess.Move]:
        moves = list(board.legal_moves)
        
        def move_priority(move: chess.Move) -> Tuple[int, int, int, int]:
            """Build a stable tuple key for tactical move ordering."""
            # Priority 1: Hash move from transposition table
            hash_score = 1000000 if tt_move == move else 0
            
            # Priority 2: Captures (MVV-LVA)
            capture_score = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    capture_score = 100000 + 100 * victim.piece_type - attacker.piece_type
            
            # Priority 3: Promotions
            promo_score = 50000 if move.promotion == chess.QUEEN else (40000 if move.promotion else 0)
            
            # Priority 4: Killer moves (two slots)
            killer_score = 0
            if ply < 128:
                if self.killer_moves[ply][0] == move:
                    killer_score = 30000
                elif self.killer_moves[ply][1] == move:
                    killer_score = 29000
            
            # Priority 5: Counter moves
            counter_score = 0
            if board.move_stack:
                if self.counter_moves.get(board.peek()) == move:
                    counter_score = 20000
            
            # Priority 6: History heuristic
            history_score = min(10000, max(-10000, self.history_table.get(move, 0)))
            
            # Priority 7: Checks
            check_score = 1000 if board.gives_check(move) else 0
            
            return (-hash_score - capture_score - promo_score - killer_score - counter_score - check_score - history_score, 
                   move.from_square, move.to_square, move.promotion or 0)
        
        return sorted(moves, key=move_priority)

    def _order_captures(self, board: chess.Board) -> list[chess.Move]:
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move) or move.promotion:
                captures.append(move)
        
        def capture_priority(move: chess.Move) -> int:
            """Rank captures and forcing quiescence moves."""
            score = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    # MVV-LVA
                    score = 1000 * victim.piece_type - attacker.piece_type
            if move.promotion:
                score += 500 if move.promotion == chess.QUEEN else 400
            if board.gives_check(move):
                score += 100
            return -score
        
        return sorted(captures, key=capture_priority)

    def _relative(self, board: chess.Board) -> int:
        score = evaluate_handcrafted(board)
        return score if board.turn == chess.WHITE else -score
