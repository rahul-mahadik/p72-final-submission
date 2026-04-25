"""Factory for constructing candidate engines by architecture key."""
from __future__ import annotations

from engine.candidates.alphabeta.engine import AlphaBetaEngine
from engine.candidates.mcts.engine import MCTSEngine
from engine.candidates.nnue_lite.engine import NNUELiteEngine
from engine.candidates.policy_guided.engine import PolicyGuidedEngine


ENGINE_FACTORIES = {
    "alphabeta": AlphaBetaEngine,
    "mcts": MCTSEngine,
    "nnue_lite": NNUELiteEngine,
    "policy_guided": PolicyGuidedEngine,
}

ENGINE_KINDS = tuple(ENGINE_FACTORIES)


def create_engine(kind: str):
    """Instantiate one of the supported candidate engine families."""
    if kind not in ENGINE_FACTORIES:
        raise KeyError(f"Unknown engine kind: {kind}")
    return ENGINE_FACTORIES[kind]()
