"""MCTS/UCT-style candidate for architectural diversity."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import chess

from engine.common.eval import score_for_side
from engine.common.interface import ChessEngine, safe_legal_move


@dataclass
class Node:
    """Tree node storing visit counts and rollout value."""

    move: chess.Move | None
    parent: "Node | None" = None
    visits: int = 0
    value: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    untried: list[chess.Move] = field(default_factory=list)
    fully_expanded: bool = False


class MCTSEngine(ChessEngine):
    """Tiny UCT search with heuristic expansion and static rollout value."""

    name = "mcts_puct_lite"

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        """Search with UCT until the budget expires and return a legal move."""
        root = Node(move=None)
        self._initialize_node(root, position)
        deadline = time.perf_counter() + max(time_budget_ms, 20) / 1000
        iterations = 0
        
        while time.perf_counter() < deadline:
            board = position.copy(stack=False)
            node = self._select(root, board, iterations)
            value = self._simulate(board)
            self._backup(node, value)
            iterations += 1
            
        if not root.children:
            return safe_legal_move(position, None)
        
        # Select most visited child
        best = max(root.children.values(), key=lambda child: child.visits)
        return safe_legal_move(position, best.move)

    def _initialize_node(self, node: Node, board: chess.Board) -> None:
        """Initialize node with sorted moves using move ordering heuristics."""
        moves = list(board.legal_moves)
        if not moves:
            node.untried = []
            node.fully_expanded = True
            return
            
        # Sort moves by priority
        scored_moves = []
        for move in moves:
            score = 0
            if board.is_capture(move):
                score += 100
                if board.is_en_passant(move):
                    score += 20
            if board.gives_check(move):
                score += 50
            if move.promotion:
                score += 200 if move.promotion == chess.QUEEN else 50
            # Penalize moving to attacked squares
            board.push(move)
            if board.is_attacked_by(not board.turn, move.to_square):
                score -= 25
            board.pop()
            scored_moves.append((score, random.random(), move))
        
        scored_moves.sort(reverse=True)
        node.untried = [m for _, _, m in scored_moves]
        node.fully_expanded = False

    def _select(self, node: Node, board: chess.Board, iterations: int) -> Node:
        """Tree policy with progressive widening."""
        while not board.is_game_over():
            # Progressive widening: limit children based on visits
            max_children = min(len(node.untried) + len(node.children), 
                              int(2 + math.sqrt(node.visits + 1)))
            
            if len(node.children) < max_children and node.untried:
                # Expand
                move = node.untried.pop(0)
                board.push(move)
                child = Node(move=move, parent=node)
                self._initialize_node(child, board)
                node.children[move] = child
                if not node.untried:
                    node.fully_expanded = True
                return child
                
            if not node.children:
                # Terminal node
                return node
                
            # Select best child using UCT
            exploration = self._get_exploration_constant(iterations)
            node = max(node.children.values(), 
                      key=lambda c: self._uct(c, exploration))
            board.push(node.move)
            
        return node

    def _uct(self, child: Node, exploration: float) -> float:
        """Upper confidence bound for trees."""
        if child.visits == 0:
            return float('inf')
        parent_visits = child.parent.visits if child.parent else 1
        exploitation = child.value / child.visits
        exploration_term = exploration * math.sqrt(2 * math.log(parent_visits) / child.visits)
        return exploitation + exploration_term

    def _get_exploration_constant(self, iterations: int) -> float:
        """Dynamic exploration constant that decreases over time."""
        return max(0.5, 1.4 - 0.0002 * iterations)

    def _simulate(self, board: chess.Board) -> float:
        """Enhanced simulation with quiescence and better evaluation."""
        if board.is_checkmate():
            return -1.0
        if board.is_game_over():
            return 0.0
            
        # Run short quiescence search on captures
        depth = 0
        max_depth = 3
        while depth < max_depth and not board.is_game_over():
            captures = [m for m in board.legal_moves if board.is_capture(m)]
            if not captures:
                break
            # Pick best capture by MVV-LVA
            best_capture = None
            best_score = -1000
            for move in captures:
                victim = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                if victim and attacker:
                    score = victim - attacker / 10
                    if score > best_score:
                        best_score = score
                        best_capture = move
            if best_capture and best_score > 0:
                board.push(best_capture)
                depth += 1
            else:
                break
                
        # Evaluate final position
        score = score_for_side(board)
        # Undo quiescence moves
        for _ in range(depth):
            board.pop()
            score = -score
            
        return max(-1.0, min(1.0, score / 1000))

    def _backup(self, node: Node, value: float) -> None:
        """Backpropagate value through tree."""
        while node:
            node.visits += 1
            node.value += value
            value = -value
            node = node.parent
