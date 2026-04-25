# Experiment Protocol

AutoResearch Chess Lab compares research allocation strategies under a fixed LLM token budget.

## Arms

- `autoresearch_minimal`: explores all candidate families, runs cheap evals, prunes weak candidates, promotes a survivor, then performs held-out eval.
- `single_best`: selects one candidate upfront from candidate descriptions and spends the same token cap on that candidate.
- `autoresearch_frequent`: supported as an arm name and UI mode for human-heavy review gates.

## Candidate Families

- C1 Classical Alpha-Beta: alpha-beta pruning plus handcrafted material, mobility, and simple piece-square terms.
- C2 MCTS / PUCT-Lite: tree search with UCT selection and heuristic value cutoffs.
- C3 NNUE-Lite: shallow search using a tiny hand-coded learned-value-style feature evaluator.
- C4 Policy-Guided Search: heuristic policy prior selects a top-k move set before shallow evaluation.

All candidates implement:

```python
select_move(position, time_budget_ms) -> chess.Move
```

## Run Order

1. Run AutoResearch minimal-human in mock or real mode.
2. Record total tokens used by that arm.
3. Run Single-Best with the same token budget.
4. Optionally run frequent-human AutoResearch with the same cap.
5. Compare held-out eval artifacts and final reports.

## Metrics

Primary:

- Held-out win rate and approximate Elo delta.

Secondary:

- Win rate per token
- Raw win rate
- Crash count
- Illegal move count
- Smoke/regression pass rate
- Average move latency
- Wall-clock runtime
- Accepted patches
- Tokens per accepted improvement
- Tokens by candidate
- Candidate pruning path

## Audit Trail

Each run writes:

- `experiments/<experiment_id>/experiment.json`
- `experiments/<experiment_id>/tasks.json`
- `experiments/<experiment_id>/artifacts.json`
- `experiments/<experiment_id>/budget.json` and `budget.jsonl`
- `experiments/<experiment_id>/events.jsonl`
- `artifacts/<experiment_id>/*.json`
- `artifacts/evals/*.json` and `*.pgn`

