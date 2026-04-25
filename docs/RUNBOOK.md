# Runbook

This is the local-first path for running AutoResearch Chess Lab and inspecting a result. Mock mode is best for offline validation; live mode spends real API tokens through Anthropic.

## Install And Test

```bash
./p72 install
./p72 test
```

## Start The Backend

```bash
./p72 ui
```

Open:

```text
http://127.0.0.1:8001/dashboard
http://127.0.0.1:8001/docs
```

The dashboard is served by FastAPI and polls the backend for the latest experiment, research trace, leaderboard, artifacts, and budget ledger.

## Run The Full Mock Experiment

```bash
python3 -m agents.orchestrator --backend-url http://127.0.0.1:8001 --full-mock
```

This runs the complete comparison:

1. Creates one experiment.
2. Runs `autoresearch_minimal` across all four candidates.
3. Records AutoResearch token estimates.
4. Runs `single_best` with the same token cap.
5. Writes a final comparison report.

Mock mode still runs real local chess evaluations. Only LLM planning, development, test-summary, and final-summary text is mocked by `MockLLMAdapter`.

## Redo Elo Over Time Against NNUE

```bash
python3 scripts/elo_over_time.py
```

The default curve compares `alphabeta` against `nnue_lite`, treating NNUE-lite as Elo 1000 at each checkpoint. Checkpoints are increasing per-move search budgets, and outputs are written to:

```text
docs/ELO_OVER_TIME_NNUE_BASELINE.md
docs/elo_over_time_nnue_baseline.csv
```

The underlying JSON and PGN evaluation artifacts are generated under `artifacts/elo_over_time/` and remain ignored by git.

## Run Anthropic Live Logical Pods

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./p72 live-ui
```

To pause at major gates for terminal approval/reject prompts:

```bash
./p72 live-hitl-ui
```

The code also supports role-specific keys:

```text
PARENT_ANTHROPIC_API_KEY
CHILD_A_ANTHROPIC_API_KEY
CHILD_B_ANTHROPIC_API_KEY
```

Unset role-specific keys fall back to `ANTHROPIC_API_KEY`. The budget ledger records logical key labels such as `key-1`, `key-2`, and `key-3`, never raw secrets.

## Clean Runtime State

```bash
./p72 clean
```

Generated outputs live under `experiments/`, `artifacts/`, and `.run/`. They are intentionally ignored by git so the final submission contains source, tests, docs, and seed data only.

## Inspect Outputs

Useful files after a run:

```text
experiments/<experiment_id>/experiment.json
experiments/<experiment_id>/candidates.json
experiments/<experiment_id>/budget.json
experiments/<experiment_id>/events.jsonl
experiments/<experiment_id>/artifacts.json
artifacts/<experiment_id>_autoresearch_minimal_final_report.json
artifacts/<experiment_id>_single_best_final_report.json
artifacts/<experiment_id>_comparison_report.json
artifacts/evals/<engine>_<dev|heldout>_<timestamp>.json
artifacts/evals/<engine>_<dev|heldout>_<timestamp>.pgn
```
