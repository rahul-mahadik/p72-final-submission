"""Classical alpha-beta candidate with handcrafted evaluation."""
from __future__ import annotations

import math
import time
from typing import Optional

import chess

from engine.common.eval import evaluate_handcrafted
from engine.common.interface import ChessEngine, safe_legal_move


class AlphaBetaEngine(ChessEngine):
    """Iterative deepening alpha-beta with quiescence, killer moves, and PV tracking."""

    name = "classical_alphabeta"

    def __init__(self):
        """Initialize search heuristics and bounded in-memory caches."""
        self.killer_moves = {}  # depth -> [move1, move2]
        self.pv_table = {}  # depth -> best_move
        self.history_table = {}  # (piece_type, to_square) -> score
        self.capture_history = {}  # (attacker_type, victim_type, to_square) -> score
        self.countermove_table = {}  # (from_square, to_square) -> move
        self.butterfly_table = {}  # (from_square, to_square) -> score
        self.nodes_searched = 0
        self.transposition_table = {}  # position_hash -> (depth, score, flag, best_move, age)
        self.search_age = 0
        self.eval_cache = {}  # position_hash -> eval_score
        self.max_history_value = 10000
        self.threat_moves = {}  # depth -> [threat_move1, threat_move2]

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        """Return the best legal move found before the wall-clock deadline."""
        # Ensure minimum time
        time_budget_ms = max(time_budget_ms, 20)
        start_time = time.perf_counter()
        
        # Adaptive time management
        hard_deadline = start_time + time_budget_ms / 1000
        
        self.nodes_searched = 0
        self.search_age += 1
        self.killer_moves.clear()
        self.pv_table.clear()
        self.threat_moves.clear()
        self._decay_history_tables()
        self.countermove_table.clear()
        self.eval_cache.clear()
        
        # Smart TT cleanup - only clean when getting large
        if len(self.transposition_table) > 500000:
            self._cleanup_transposition_table()
        
        best_move: Optional[chess.Move] = None
        best_score = -math.inf
        
        # Aspiration window bounds
        aspiration_alpha = -math.inf
        aspiration_beta = math.inf
        aspiration_delta = 25
        
        # Analyze position complexity
        legal_moves = list(position.legal_moves)
        num_moves = len(legal_moves)
        
        # Check for single legal move
        if num_moves == 1:
            return legal_moves[0]
        
        # Calculate dynamic time allocation
        num_captures = sum(1 for m in legal_moves if position.is_capture(m))
        num_checks = sum(1 for m in legal_moves if position.gives_check(m))
        in_check = position.is_check()
        piece_count = len(position.piece_map())
        is_endgame = piece_count <= 14
        
        # Enhanced tactical factor calculation
        tactical_factor = 1.0
        if in_check:
            tactical_factor = 1.35
        elif num_captures > num_moves * 0.4:
            tactical_factor = 1.25
        elif num_checks > 3:
            tactical_factor = 1.2
        elif num_captures > num_moves * 0.25 and num_checks > 1:
            tactical_factor = 1.15
        
        # Endgame time bonus
        if is_endgame:
            tactical_factor *= 1.1
        
        # Adjust soft deadline based on complexity
        base_soft_fraction = 0.75
        if num_moves > 40:
            base_soft_fraction = 0.6  # Many moves, be more conservative
        elif num_moves < 10:
            base_soft_fraction = 0.9  # Few moves, can search deeper
        elif num_moves < 20:
            base_soft_fraction = 0.85
        
        soft_deadline = start_time + (time_budget_ms * base_soft_fraction * tactical_factor) / 1000
        soft_deadline = min(soft_deadline, hard_deadline - 0.01)
        
        # Iterative deepening
        depth_completed = 0
        prev_iteration_time = 0.001
        stable_iterations = 0
        winning_threshold = 500  # Score threshold for winning positions
        last_score_change = 0
        
        for depth in range(1, 50):
            iteration_start = time.perf_counter()
            
            # Enhanced time prediction
            if depth > 3:
                # Dynamic branching factor based on position and depth
                base_branching = 2.5 if depth < 10 else 2.0
                if best_score > winning_threshold:
                    base_branching = 1.7  # Search less when winning
                elif abs(best_score) < 50:
                    base_branching *= 1.1  # More time in balanced positions
                
                # Adjust for position complexity
                complexity_factor = 1 + (num_moves - 20) * 0.01
                branching_factor = base_branching * max(0.8, min(1.3, complexity_factor))
                
                predicted_next_time = prev_iteration_time * branching_factor
                if iteration_start + predicted_next_time > soft_deadline:
                    break
            
            if time.perf_counter() >= soft_deadline:
                break
            
            current_best: Optional[chess.Move] = None
            current_score = -math.inf
            
            # Use aspiration windows after depth 5
            if depth > 5 and abs(best_score) < 90000:
                aspiration_alpha = best_score - aspiration_delta
                aspiration_beta = best_score + aspiration_delta
            else:
                aspiration_alpha = -math.inf
                aspiration_beta = math.inf
            
            # Aspiration search with re-search on fail
            for window_attempt in range(3):
                alpha = aspiration_alpha
                beta = aspiration_beta
                
                moves = self._ordered_moves(position, 0, best_move, None)
                
                for i, move in enumerate(moves):
                    if time.perf_counter() >= hard_deadline - 0.002:
                        break
                    
                    position.push(move)
                    
                    # PV search for first move, scout search for others
                    if i == 0:
                        score = -self._search(position, depth - 1, -beta, -alpha, hard_deadline, 1, move, True)
                    else:
                        # Zero-window scout search
                        score = -self._search(position, depth - 1, -alpha - 1, -alpha, hard_deadline, 1, move, False)
                        if alpha < score < beta:
                            # Re-search with full window
                            score = -self._search(position, depth - 1, -beta, -alpha, hard_deadline, 1, move, False)
                    
                    position.pop()
                    
                    if score > current_score:
                        current_score = score
                        current_best = move
                        alpha = max(alpha, score)
                
                # Check aspiration window result
                if current_score <= aspiration_alpha:
                    aspiration_alpha = -math.inf
                    aspiration_delta = min(aspiration_delta * 2, 500)
                elif current_score >= aspiration_beta:
                    aspiration_beta = math.inf
                    aspiration_delta = min(aspiration_delta * 2, 500)
                else:
                    break
            
            # Only update best move if we completed this depth
            if current_best is not None and time.perf_counter() < hard_deadline:
                # Check for move stability
                if current_best == best_move:
                    stable_iterations += 1
                else:
                    stable_iterations = 0
                
                # Track score changes
                if abs(current_score - best_score) > 50:
                    last_score_change = depth
                
                best_move = current_best
                best_score = current_score
                self.pv_table[0] = best_move
                depth_completed = depth
                
                # Early exit on checkmate
                if best_score > 90000:
                    break
                
                # More aggressive early exit when winning clearly
                if best_score > winning_threshold * 2 and stable_iterations >= 2:
                    break
                
                # Exit if score stable for many iterations
                if depth - last_score_change >= 4 and time.perf_counter() - start_time > time_budget_ms * 0.5 / 1000:
                    break
                
                # Early exit if move is stable and we used enough time
                if stable_iterations >= 3 and time.perf_counter() - start_time > time_budget_ms * 0.4 / 1000:
                    break
            
            prev_iteration_time = time.perf_counter() - iteration_start
        
        return safe_legal_move(position, best_move)

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, deadline: float, ply: int, last_move: Optional[chess.Move], is_pv: bool) -> float:
        self.nodes_searched += 1
        
        if time.perf_counter() >= deadline:
            return self._relative_eval(board)
        
        if board.is_game_over():
            if board.is_checkmate():
                return -100000 + ply  # Prefer faster checkmates
            return 0  # Draw
        
        # Transposition table lookup
        position_hash = board.zobrist_hash() if hasattr(board, 'zobrist_hash') else hash(board.fen())
        tt_entry = self.transposition_table.get(position_hash)
        tt_move = None
        
        if tt_entry:
            tt_depth, tt_score, tt_flag, tt_move, tt_age = tt_entry
            if tt_depth >= depth and not is_pv:
                if tt_flag == 'exact':
                    return tt_score
                elif tt_flag == 'lower' and tt_score >= beta:
                    return tt_score
                elif tt_flag == 'upper' and tt_score <= alpha:
                    return tt_score
        
        if depth <= 0:
            return self._quiescence(board, alpha, beta, deadline, ply)
        
        in_check = board.is_check()
        static_eval = self._relative_eval(board) if not in_check else -math.inf
        
        # Enhanced reverse futility pruning
        if not is_pv and not in_check and depth <= 6 and abs(beta) < 90000:
            margin = 100 * depth + 20 * depth * depth // 2
            if static_eval - margin >= beta:
                return static_eval - margin
        
        # Razoring
        if not is_pv and not in_check and depth <= 3 and static_eval + 300 * depth < alpha:
            razor_score = self._quiescence(board, alpha, beta, deadline, ply)
            if razor_score <= alpha:
                return razor_score
        
        # Null move pruning with improved conditions
        total_pieces = len(board.piece_map())
        is_endgame = total_pieces <= 14
        
        # More careful null move in endgames
        null_move_min_pieces = 7 if is_endgame else 0
        
        if (not is_pv and not in_check and depth >= 3 and 
            abs(beta) < 90000 and static_eval >= beta and 
            self._has_non_pawn_material(board) and total_pieces > null_move_min_pieces):
            
            # Dynamic reduction based on eval and depth
            eval_margin = max(0, static_eval - beta)
            R = 3 + depth // 6 + min(3, eval_margin // 200)
            
            # Reduce reduction in endgame
            if is_endgame:
                R = max(2, R - 1)
            
            board.push(chess.Move.null())
            null_score = -self._search(board, depth - R, -beta, -beta + 1, deadline, ply + 1, None, False)
            board.pop()
            
            if null_score >= beta:
                # Verification search for deep nodes
                if depth > 11 and abs(null_score) < 90000:
                    v_score = self._search(board, depth - R, beta - 1, beta, deadline, ply, last_move, False)
                    if v_score >= beta:
                        return null_score
                else:
                    return null_score
        
        # Enhanced futility pruning
        futility_margin = math.inf
        if depth <= 3 and not in_check and not is_pv:
            futility_margin = 150 + 100 * depth
        
        # Internal iterative deepening
        if not tt_move and is_pv and depth >= 6:
            self._search(board, depth - 2, alpha, beta, deadline, ply, last_move, True)
            tt_entry = self.transposition_table.get(position_hash)
            if tt_entry:
                tt_move = tt_entry[3]
        
        best_score = -math.inf
        best_move = None
        moves = self._ordered_moves(board, ply, tt_move or self.pv_table.get(ply), last_move)
        moves_searched = 0
        raised_alpha = False
        quiet_moves_searched = 0
        
        # Multi-cut pruning
        if not is_pv and depth >= 3 and not in_check:
            mc_count = 0
            for i, move in enumerate(moves[:6]):
                board.push(move)
                score = -self._search(board, depth - 2, -beta, -alpha, deadline, ply + 1, move, False)
                board.pop()
                if score >= beta:
                    mc_count += 1
                    if mc_count >= 3:
                        return beta
        
        for i, move in enumerate(moves):
            is_quiet = not board.is_capture(move) and not move.promotion and not board.gives_check(move)
            
            # Enhanced futility pruning
            if is_quiet and not in_check and i > 0:
                if static_eval + futility_margin <= alpha:
                    continue
            
            # Enhanced late move pruning
            if not is_pv and not in_check and depth <= 4 and is_quiet:
                lmp_threshold = 3 + depth * depth
                if quiet_moves_searched >= lmp_threshold:
                    continue
            
            board.push(move)
            moves_searched += 1
            if is_quiet:
                quiet_moves_searched += 1
            
            # Enhanced late move reductions
            reduction = 0
            if depth >= 3 and moves_searched > 3 and not in_check and not board.is_check():
                if is_quiet:
                    # Base reduction
                    reduction = int(0.5 + math.log(depth) * math.log(moves_searched) / 2.0)
                    
                    # Adjust based on move quality
                    if move in self.killer_moves.get(ply, []):
                        reduction = max(0, reduction - 1)
                    
                    # Adjust based on evaluation
                    if static_eval > alpha + 100:
                        reduction = max(0, reduction - 1)
                    elif static_eval < alpha - 100:
                        reduction += 1
                    
                    # Countermove bonus
                    if last_move:
                        countermove = self.countermove_table.get((last_move.from_square, last_move.to_square))
                        if countermove and move == countermove:
                            reduction = max(0, reduction - 1)
                    
                    reduction = min(reduction, depth - 1)
            
            # PVS search
            if moves_searched == 1:
                score = -self._search(board, depth - 1, -beta, -alpha, deadline, ply + 1, move, is_pv)
            else:
                if reduction > 0:
                    score = -self._search(board, depth - 1 - reduction, -alpha - 1, -alpha, deadline, ply + 1, move, False)
                else:
                    score = -self._search(board, depth - 1, -alpha - 1, -alpha, deadline, ply + 1, move, False)
                
                if score > alpha and score < beta:
                    score = -self._search(board, depth - 1, -beta, -alpha, deadline, ply + 1, move, is_pv)
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            if score >= beta:
                # Update history tables with bounded increments
                if not board.is_capture(move):
                    self._add_killer(ply, move)
                    self._add_threat(ply, move)
                    piece = board.piece_at(move.from_square)
                    if piece:
                        self._update_history(piece.piece_type, move.to_square, depth)
                        self._update_butterfly(move.from_square, move.to_square, depth)
                    if last_move:
                        self.countermove_table[(last_move.from_square, last_move.to_square)] = move
                else:
                    self._update_capture_history(board, move, depth)
                
                # Store in TT
                self.transposition_table[position_hash] = (depth, beta, 'lower', best_move, self.search_age)
                return beta
            
            if score > alpha:
                alpha = score
                raised_alpha = True
        
        # Store in transposition table
        if best_score <= alpha:
            self.transposition_table[position_hash] = (depth, best_score, 'upper', best_move, self.search_age)
        else:
            self.transposition_table[position_hash] = (depth, best_score, 'exact', best_move, self.search_age)
        
        if best_move and raised_alpha:
            self.pv_table[ply] = best_move
        
        return best_score if best_score > -math.inf else alpha

    def _quiescence(self, board: chess.Board, alpha: float, beta: float, deadline: float, ply: int) -> float:
        self.nodes_searched += 1
        
        if time.perf_counter() >= deadline or ply > 50:
            return self._relative_eval(board)
        
        in_check = board.is_check()
        
        # Don't stand pat when in check
        if not in_check:
            stand_pat = self._relative_eval(board)
            
            if stand_pat >= beta:
                return beta
            
            # Enhanced delta pruning
            big_delta = 900
            # Check for potential promotions
            if board.pawns & board.occupied_co[board.turn] & (chess.BB_RANK_7 if board.turn else chess.BB_RANK_2):
                big_delta = 1750
            
            if stand_pat + big_delta < alpha:
                return alpha
            
            alpha = max(alpha, stand_pat)
        
        # Generate moves
        if in_check:
            moves = list(board.legal_moves)
        else:
            moves = []
            for m in board.legal_moves:
                if board.is_capture(m):
                    # Include capture if SEE is positive or close to positive
                    see_value = self._see_score(board, m)
                    if see_value >= -50:
                        moves.append(m)
                elif m.promotion and m.promotion == chess.QUEEN:
                    moves.append(m)
        
        # Order moves
        moves.sort(key=lambda m: self._quiescence_move_score(board, m), reverse=True)
        
        # Search limit
        max_moves = 999 if in_check else min(10, len(moves))
        
        for i, move in enumerate(moves[:max_moves]):
            # Enhanced SEE pruning
            if not in_check:
                see_value = self._see_score(board, move)
                
                # Prune bad captures more aggressively
                if see_value < -100 and i > 0:
                    continue
                
                # Deep futility pruning
                if i > 3:
                    material_gain = see_value
                    if stand_pat + material_gain + 250 < alpha:
                        continue
            
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, deadline, ply + 1)
            board.pop()
            
            if score >= beta:
                if board.is_capture(move):
                    self._update_capture_history(board, move, 1)
                return beta
            
            alpha = max(alpha, score)
        
        return alpha

    def _has_non_pawn_material(self, board: chess.Board) -> bool:
        """Check if the side to move has non-pawn material."""
        color = board.turn
        return (len(board.pieces(chess.KNIGHT, color)) > 0 or
                len(board.pieces(chess.BISHOP, color)) > 0 or
                len(board.pieces(chess.ROOK, color)) > 0 or
                len(board.pieces(chess.QUEEN, color)) > 0)

    def _see_score(self, board: chess.Board, move: chess.Move) -> int:
        """Enhanced static exchange evaluation."""
        if not board.is_capture(move):
            return 0 if not move.promotion else 800
        
        piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                       chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000}
        
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if not victim or not attacker:
            return 0
        
        # Start with the value of the captured piece
        gain = [piece_values[victim.piece_type]]
        
        # Account for promotion
        if move.promotion:
            gain[0] += piece_values.get(move.promotion, 0) - piece_values[chess.PAWN]
        
        # Simulate the capture sequence
        board_copy = board.copy(stack=False)
        board_copy.push(move)
        
        # Get all attackers to the square
        attackers = []
        for color in [not board_copy.turn, board_copy.turn]:
            color_attackers = []
            for attacker_square in board_copy.attackers(color, move.to_square):
                piece = board_copy.piece_at(attacker_square)
                if piece:
                    color_attackers.append((piece_values[piece.piece_type], attacker_square))
            color_attackers.sort()  # Sort by piece value
            attackers.append(color_attackers)
        
        # Simulate captures
        side_to_move = 0  # 0 for original attacker's opponent, 1 for original attacker
        last_attacker_value = piece_values[attacker.piece_type]
        
        while attackers[side_to_move]:
            value, square = attackers[side_to_move].pop(0)
            
            # Add/subtract the value of the piece being captured
            if side_to_move == 0:
                gain.append(last_attacker_value - gain[-1])
            else:
                gain.append(last_attacker_value - gain[-1])
            
            last_attacker_value = value
            
            # Stop if capturing with the king and opponent can recapture
            if value == piece_values[chess.KING] and attackers[1 - side_to_move]:
                break
            
            side_to_move = 1 - side_to_move
        
        # Negamax the gain list
        while len(gain) > 1:
            gain[-2] = max(gain[-2], -gain[-1])
            gain.pop()
        
        return gain[0]

    def _mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """Enhanced MVV-LVA scoring with special case handling."""
        score = 0
        
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            if victim and attacker:
                victim_value = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, 
                               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000}[victim.piece_type]
                attacker_value = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                                 chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 10000}[attacker.piece_type]
                
                score = victim_value * 100 - attacker_value
                
                # Bonus for capturing with pawns
                if attacker.piece_type == chess.PAWN:
                    score += 50
        
        if move.promotion:
            promotion_value = {chess.QUEEN: 900, chess.ROOK: 500, 
                             chess.BISHOP: 330, chess.KNIGHT: 320}.get(move.promotion, 0)
            score += promotion_value * 100
        
        return score

    def _quiescence_move_score(self, board: chess.Board, move: chess.Move) -> int:
        """Enhanced scoring for quiescence search moves."""
        score = self._mvv_lva_score(board, move)
        
        # Add SEE score as tie-breaker
        see_value = self._see_score(board, move)
        score += see_value * 10
        
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            if attacker and victim:
                key = (attacker.piece_type, victim.piece_type, move.to_square)
                score += self.capture_history.get(key, 0) // 100
        
        return score

    def _update_capture_history(self, board: chess.Board, move: chess.Move, depth: int) -> None:
        """Update capture history."""
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)
            if attacker and victim:
                key = (attacker.piece_type, victim.piece_type, move.to_square)
                current = self.capture_history.get(key, 0)
                bonus = depth * depth * 4
                # Bounded update to prevent overflow
                new_value = current + bonus - (current * bonus) // self.max_history_value
                self.capture_history[key] = min(self.max_history_value, new_value)

    def _update_history(self, piece_type: int, to_square: int, depth: int) -> None:
        """Update history heuristic with aging and bounded increment."""
        key = (piece_type, to_square)
        current = self.history_table.get(key, 0)
        bonus = depth * depth * 4
        # Bounded update formula to prevent saturation
        new_value = current + bonus - (current * bonus) // self.max_history_value
        self.history_table[key] = min(self.max_history_value, new_value)

    def _update_butterfly(self, from_square: int, to_square: int, depth: int) -> None:
        """Update butterfly table for move ordering."""
        key = (from_square, to_square)
        current = self.butterfly_table.get(key, 0)
        bonus = depth * depth * 4
        # Bounded update
        new_value = current + bonus - (current * bonus) // self.max_history_value
        self.butterfly_table[key] = min(self.max_history_value, new_value)

    def _decay_history_tables(self) -> None:
        """Decay history values to prevent overflow and adapt to position changes."""
        # More aggressive decay to prevent saturation
        decay_factor = 2  # Divide by 2 instead of multiply by 3/4
        
        # Decay history table
        for key in list(self.history_table.keys()):
            self.history_table[key] = self.history_table[key] // decay_factor
            if self.history_table[key] < 10:
                del self.history_table[key]
        
        # Decay capture history
        for key in list(self.capture_history.keys()):
            self.capture_history[key] = self.capture_history[key] // decay_factor
            if self.capture_history[key] < 10:
                del self.capture_history[key]
        
        # Decay butterfly table
        for key in list(self.butterfly_table.keys()):
            self.butterfly_table[key] = self.butterfly_table[key] // decay_factor
            if self.butterfly_table[key] < 10:
                del self.butterfly_table[key]

    def _cleanup_transposition_table(self) -> None:
        """Enhanced transposition table cleanup with better replacement policy."""
        # Calculate replacement scores
        entries_with_score = []
        current_age = self.search_age
        
        for k, v in self.transposition_table.items():
            depth, score, flag, move, age = v
            age_diff = current_age - age
            
            # Enhanced scoring: recent, deep, and exact entries are valuable
            replacement_score = depth * 15 - age_diff * 5
            
            # Bonus for exact scores
            if flag == 'exact':
                replacement_score += 10
            
            # Bonus for entries with moves
            if move:
                replacement_score += 5
            
            # Penalty for very old entries
            if age_diff > 10:
                replacement_score -= age_diff * 2
            
            entries_with_score.append((replacement_score, k))
        
        # Sort and keep best 60%
        entries_with_score.sort(reverse=True)
        keep_count = len(entries_with_score) * 3 // 5
        
        new_tt = {}
        for i, (_, k) in enumerate(entries_with_score[:keep_count]):
            new_tt[k] = self.transposition_table[k]
        
        self.transposition_table = new_tt

    def _relative_eval(self, board: chess.Board) -> int:
        """Cached evaluation from side-to-move perspective."""
        position_hash = board.zobrist_hash() if hasattr(board, 'zobrist_hash') else hash(board.fen())
        if position_hash in self.eval_cache:
            return self.eval_cache[position_hash]
        
        score = evaluate_handcrafted(board)
        result = score if board.turn == chess.WHITE else -score
        
        # Limit cache size
        if len(self.eval_cache) > 100000:
            self.eval_cache.clear()
        
        self.eval_cache[position_hash] = result
        return result

    def _ordered_moves(self, board: chess.Board, ply: int, pv_move: Optional[chess.Move] = None, last_move: Optional[chess.Move] = None) -> list[chess.Move]:
        moves = list(board.legal_moves)
        
        def move_score(move: chess.Move) -> int:
            """Rank legal moves so alpha-beta sees forcing lines first."""
            score = 0
            
            # PV move highest priority
            if pv_move and move == pv_move:
                return 1000000
            
            # Winning captures
            if board.is_capture(move):
                see_value = self._see_score(board, move)
                if see_value > 0:
                    score += 200000 + see_value * 100
                elif see_value == 0:
                    score += 100000
                else:
                    # Losing captures go after quiet moves
                    score += 5000 + see_value
                
                # Capture history bonus
                attacker = board.piece_at(move.from_square)
                victim = board.piece_at(move.to_square)
                if attacker and victim:
                    key = (attacker.piece_type, victim.piece_type, move.to_square)
                    score += min(10000, self.capture_history.get(key, 0))
            
            # Promotions
            if move.promotion:
                promotion_bonus = {chess.QUEEN: 180000, chess.ROOK: 90000,
                                 chess.BISHOP: 60000, chess.KNIGHT: 60000}.get(move.promotion, 0)
                score += promotion_bonus
            
            # Castling
            if board.is_castling(move):
                score += 70000
            
            # Killer moves
            killers = self.killer_moves.get(ply, [])
            if move in killers:
                score += 65000 - killers.index(move) * 1000
            
            # Threat moves
            threats = self.threat_moves.get(ply, [])
            if move in threats:
                score += 60000 - threats.index(move) * 500
            
            # Countermove
            if last_move and not board.is_capture(move):
                countermove = self.countermove_table.get((last_move.from_square, last_move.to_square))
                if countermove and move == countermove:
                    score += 55000
            
            # Checks (but not in deep plies to avoid check spam)
            if ply < 8 and board.gives_check(move):
                score += 40000
            
            # History and butterfly heuristics for quiet moves
            if not board.is_capture(move) and not move.promotion:
                piece = board.piece_at(move.from_square)
                if piece:
                    # History heuristic
                    key = (piece.piece_type, move.to_square)
                    history_score = self.history_table.get(key, 0)
                    score += min(history_score, 30000)
                    
                    # Butterfly heuristic
                    butterfly_key = (move.from_square, move.to_square)
                    butterfly_score = self.butterfly_table.get(butterfly_key, 0)
                    score += min(butterfly_score, 20000)
                
                # Enhanced piece-square bonus
                if piece:
                    to_file = chess.square_file(move.to_square)
                    to_rank = chess.square_rank(move.to_square)
                    from_file = chess.square_file(move.from_square)
                    from_rank = chess.square_rank(move.from_square)
                    
                    center_distance = abs(to_file - 3.5) + abs(to_rank - 3.5)
                    
                    if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        # Encourage centralization
                        score += int((7 - center_distance) * 15)
                    elif piece.piece_type == chess.PAWN:
                        # Encourage pawn advancement
                        if piece.color == chess.WHITE:
                            score += (to_rank - from_rank) * 10
                        else:
                            score += (from_rank - to_rank) * 10
                    elif piece.piece_type == chess.ROOK:
                        # Encourage rooks on open files and 7th rank
                        if piece.color == chess.WHITE and to_rank == 6:
                            score += 20
                        elif piece.color == chess.BLACK and to_rank == 1:
                            score += 20
            
            return score
        
        return sorted(moves, key=move_score, reverse=True)

    def _add_killer(self, ply: int, move: chess.Move) -> None:
        """Add a killer move."""
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        killers = self.killer_moves[ply]
        
        if move in killers:
            killers.remove(move)
        
        killers.insert(0, move)
        self.killer_moves[ply] = killers[:2]

    def _add_threat(self, ply: int, move: chess.Move) -> None:
        """Add a threat move."""
        if ply not in self.threat_moves:
            self.threat_moves[ply] = []
        
        threats = self.threat_moves[ply]
        
        if move not in threats:
            threats.insert(0, move)
            self.threat_moves[ply] = threats[:2]
