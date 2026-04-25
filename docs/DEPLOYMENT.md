# Deployment Readiness

The project is deployable as a lightweight FastAPI service. Mock mode needs no external LLM keys; live logical-pod mode needs Anthropic credentials.

## Local Docker

Create `.env` from the template:

```bash
cp .env.example .env
```

Start the backend:

```bash
docker compose up --build backend
```

Open:

```text
http://localhost:8000/dashboard
http://localhost:8000/docs
```

Run a full mock experiment from another shell:

```bash
docker compose exec backend python3 -m agents.orchestrator --backend-url http://localhost:8000 --full-mock
```

Artifacts and experiments persist to local `./artifacts` and `./experiments` through mounted volumes. Those directories are runtime outputs and are ignored by git.

## Optional Worker Containers

The full orchestrator can run without separate worker containers. If you create tasks manually and want polling child pods:

```bash
EXPERIMENT_ID=<experiment_id> docker compose --profile workers up worker-a worker-b
```

## Keys

Mock mode needs no keys. In mock mode, `MockLLMAdapter` returns deterministic structured placeholders and records estimated tokens.

Live mode can use one physical Anthropic key while preserving separate logical labels:

```bash
ANTHROPIC_API_KEY=sk-ant-...
MOCK_MODE=false
```

Then run:

```bash
python3 -m agents.orchestrator --backend-url http://localhost:8000 --live-logical-pods
```

The budget ledger records labels only:

- `parent` / `key-1`
- `child-a` / `key-2`
- `child-b` / `key-3`

For stricter runs, set `PARENT_ANTHROPIC_API_KEY`, `CHILD_A_ANTHROPIC_API_KEY`, and `CHILD_B_ANTHROPIC_API_KEY`. Raw keys are read from environment variables and are never written to artifacts.

## Mutation Guardrails

Live logical-pod mode can apply model-proposed full-file replacements. Mutation is guarded:

- writes must be relative paths under the candidate package, `engine/common/`, or `tests/`
- empty replacements and path traversal are rejected
- local checks run after accepted replacements
- failed checks automatically revert changed files

Candidate engines, tournament evals, backend events, artifacts, and budget ledgers are real local execution paths in both mock and live modes.
