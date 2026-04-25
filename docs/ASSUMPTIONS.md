# Assumptions

- This MVP is local-first and file-backed. State is stored under `experiments/` and artifact payloads are mirrored under `artifacts/`.
- Mock mode is the default. It records estimated token usage and structured artifacts without requiring external LLM keys.
- Candidate engines are toy implementations intended to compare research allocation workflows, not to maximize chess strength.
- The primary fairness constraint is total LLM tokens. Wall-clock time, latency, crash rate, and illegal moves are logged as secondary metrics.
- Stockfish evaluation is not required for acceptance. The runner can evaluate against either the random legal-move baseline or another registered engine such as `nnue_lite`.
- Human intercepts are implemented in the backend and visible in the FastAPI dashboard. The current mock orchestrator performs high-impact decisions directly and records prune/promotion artifacts; stricter approval gating can be added on top of the same intercept endpoints.
- Single-Best uses alpha-beta as the upfront prior choice in mock mode because it is the most robust small deterministic engine family.
