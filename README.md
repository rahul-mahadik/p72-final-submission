# AutoResearch Chess Lab

AutoResearch Chess Lab is a local-first framework for comparing two LLM research-allocation strategies under the same token budget.

- **Single-Best:** pick one candidate upfront, then spend the full token budget improving it.
- **AutoResearch:** explore multiple candidates cheaply, evaluate, prune, and reallocate budget to survivors.

The toy domain is chess engine development. The goal is not a world-class chess engine; it is an auditable loop for comparing research strategy choices.

## What Is Included

- FastAPI control plane for experiments, tasks, artifacts, budget ledger entries, events, candidates, and human intercepts.
- Lightweight FastAPI dashboard at `/dashboard` for trace, leaderboard, artifact, and budget inspection.
- Four candidate engine families behind the same `select_move(position, time_budget_ms)` interface.
- Tournament runner that writes JSON summaries and PGN logs.
- Anthropic live adapter plus an explicitly marked mock adapter for offline testing.
- Guarded live code mutation from model-proposed full-file replacements.
- Pytest coverage for backend lifecycle, candidate legality, tournament output, code mutation safety, key routing, and human gates.

## What Is Mocked

Mock mode is the default because it makes the project runnable without external API keys. Mocked pieces are intentionally named and commented in code:

- `MockLLMAdapter` in `agents/llm_adapter.py` returns deterministic no-network planning, development, testing, and summary payloads.
- Orchestrator methods with `mock` in their names use those deterministic LLM outputs while still running real local chess evaluations.
- Mock mode records estimated token counts, not provider-reported token usage.

Live logical-pod mode uses Anthropic for planning/development summaries and records provider token usage when the API returns it.

## Quickstart

```bash
./p72 install
./p72 test
./p72 ui
```

Open:

```text
http://127.0.0.1:8001/dashboard
```

Run the full offline comparison:

```bash
python3 -m agents.orchestrator --backend-url http://127.0.0.1:8001 --full-mock
```

Generate the Elo-over-time report with NNUE-lite as the comparison engine:

```bash
python3 scripts/elo_over_time.py
```

The report is written to `docs/ELO_OVER_TIME_NNUE_BASELINE.md`, with CSV data in `docs/elo_over_time_nnue_baseline.csv`.

Run live with one Anthropic key shared by logical parent/child pods:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./p72 live-ui
```

For terminal approval prompts at major gates:

```bash
./p72 live-hitl-ui
```

Useful commands:

```bash
./p72 status
./p72 stop
./p72 clean
./p72 doctor-live
```

## Docker

```bash
cp .env.example .env
docker compose up --build backend
```

Open:

```text
http://localhost:8000/dashboard
http://localhost:8000/docs
```

Run a mock experiment:

```bash
docker compose exec backend python3 -m agents.orchestrator --backend-url http://localhost:8000 --full-mock
```

## Project Layout

```text
agents/       LLM adapters, orchestrator, pod worker, guarded code mutation
backend/      FastAPI app, Pydantic models, file-backed storage, routes
data/         Dev and held-out FEN positions plus tactical smoke tests
docs/         Protocol, runbook, deployment notes, assumptions
engine/       Candidate chess engines and common interface
tournament/   Game runner, position loading, rough Elo estimate, log writing
tests/        Backend, engine, tournament, mutation, routing, and HITL tests
```

Generated runtime outputs under `artifacts/`, `experiments/`, `.run/`, and Python caches are ignored by git and are not part of the final submission.

## Candidate Engines

- `alphabeta`: classical alpha-beta with handcrafted evaluation.
- `mcts`: UCT tree search with heuristic expansion and value cutoff.
- `nnue_lite`: tiny feature-based value evaluator with shallow move selection.
- `policy_guided`: policy-prior top-k move filtering plus shallow search.

## Documentation

- [Experiment Protocol](docs/EXPERIMENT.md)
- [Runbook](docs/RUNBOOK.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Assumptions](docs/ASSUMPTIONS.md)
