"""Small Elo helper used for rough win-rate-to-Elo reporting."""
from __future__ import annotations

import math


def approximate_elo_delta(score_rate: float) -> float:
    """Convert a score rate to an approximate Elo delta.

    This is only a convenient monotonic summary for hackathon comparisons; it
    is not a statistically rigorous rating estimate.
    """
    score_rate = min(0.99, max(0.01, score_rate))
    return -400 * math.log10(1 / score_rate - 1)
