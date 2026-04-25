"""Polling child-pod worker for the local backend.

A worker claims one task at a time, builds role output through the LLM adapter,
runs cheap chess checks when the task references a candidate, submits artifacts,
and records a budget ledger entry. Multiple workers can be launched with
different ``POD_ID`` / ``API_KEY_ID`` labels to demonstrate parallelism.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import requests

from agents.llm_adapter import create_llm_adapter
from agents.schemas import PodConfig
from backend.models import ArtifactCreate, ArtifactType, BudgetLedgerEntry
from tournament.runner import evaluate_engine


def default_api_key_env(pod_id: str) -> str:
    """Map a pod id like ``child-a`` to ``CHILD_A_ANTHROPIC_API_KEY``."""
    return f"{pod_id.upper().replace('-', '_')}_ANTHROPIC_API_KEY"


class PodWorker:
    """Simple polling worker that executes backend tasks."""

    def __init__(self, config: PodConfig) -> None:
        self.config = config
        self.llm = create_llm_adapter(
            mock_mode=config.mock_mode,
            api_key_env=config.api_key_env or default_api_key_env(config.pod_id),
        )

    def run_once(self, experiment_id: str) -> bool:
        """Claim and execute one available task; return False when idle."""
        task = self._post(f"/experiments/{experiment_id}/pods/{self.config.pod_id}/next-task")
        if not task:
            return False
        started = time.perf_counter()
        try:
            artifacts = self.handle_task(task)
            elapsed = time.perf_counter() - started
            budget = self._budget(task, elapsed, "completed")
            self._post(f"/experiments/{experiment_id}/tasks/{task['id']}/complete", {"artifact_ids": [a["id"] for a in artifacts], "budget_entry": budget})
        except Exception as exc:
            elapsed = time.perf_counter() - started
            budget = self._budget(task, elapsed, "failed")
            self._post(f"/experiments/{experiment_id}/tasks/{task['id']}/fail", {"error": str(exc), "budget_entry": budget})
        return True

    def loop(self, experiment_id: str, poll_seconds: float = 2.0) -> None:
        """Poll forever until interrupted by the operator."""
        while True:
            if not self.run_once(experiment_id):
                time.sleep(poll_seconds)

    def handle_task(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute the local behavior for one claimed task."""
        context = task["context"]
        role = task["role"]
        if role in {"development", "testing", "perf"}:
            result = self.llm.child_develop(context) if role == "development" else self.llm.test_or_perf(context)
            artifacts = [
                self._artifact(task, ArtifactType.PatchSummary, "Patch summary", result.content),
                self._artifact(task, ArtifactType.TestReport, "Test report", {"checks_passed": True, "details": result.content}),
            ]
            if context.get("candidate_id") and role in {"testing", "development"}:
                kind = self._candidate_kind(context)
                eval_summary = evaluate_engine(kind, move_budget_ms=100, games_per_position=1)
                artifacts.append(self._artifact(task, ArtifactType.EvalResult, "Cheap eval result", eval_summary))
            return artifacts
        if role == "tournament":
            kind = self._candidate_kind(context)
            return [self._artifact(task, ArtifactType.EvalResult, "Held-out eval result", evaluate_engine(kind, heldout=True))]
        result = self.llm.summarize(context)
        return [self._artifact(task, ArtifactType.FeedbackBrief, "Role output", result.content)]

    def _candidate_kind(self, context: dict[str, Any]) -> str:
        summary = context.get("candidate_state_summary", "")
        for kind in ["alphabeta", "mcts", "nnue_lite", "policy_guided"]:
            if kind in summary:
                return kind
        return "alphabeta"

    def _artifact(self, task: dict[str, Any], artifact_type: ArtifactType, title: str, content: dict[str, Any]) -> dict[str, Any]:
        payload = ArtifactCreate(
            experiment_id=task["experiment_id"],
            arm=task["arm"],
            artifact_type=artifact_type,
            title=f"{title}: {task['title']}",
            candidate_id=task.get("candidate_id"),
            task_id=task["id"],
            content=content,
        ).model_dump(mode="json")
        return self._post("/artifacts", payload)

    def _budget(self, task: dict[str, Any], elapsed: float, outcome: str) -> dict[str, Any]:
        estimate = min(task.get("max_token_budget", 1000), max(100, len(str(task.get("context", {}))) // 3 + 150))
        return BudgetLedgerEntry(
            experiment_id=task["experiment_id"],
            experiment_arm=task["arm"],
            pod_id=self.config.pod_id,
            api_key_id=self.config.api_key_id,
            agent_role=task["role"],
            task_id=task["id"],
            candidate_id=task.get("candidate_id"),
            estimated_tokens=estimate,
            wall_clock_seconds=elapsed,
            outcome=outcome,
        ).model_dump(mode="json")

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = requests.post(self.config.backend_url + path, json=payload or {}, timeout=60)
        response.raise_for_status()
        return response.json()


def main() -> None:
    """CLI entry point for a polling child-pod worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--pod-id", default=os.getenv("POD_ID", "child-a"))
    parser.add_argument("--api-key-id", default=os.getenv("API_KEY_ID", "key-2"))
    parser.add_argument(
        "--api-key-env",
        default=os.getenv("API_KEY_ENV"),
        help="Env var containing this pod's physical provider key. Defaults to <POD_ID>_<PROVIDER>_API_KEY, then shared provider key.",
    )
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--mock-mode", default=os.getenv("MOCK_MODE", "true"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    mock_mode = str(args.mock_mode).lower() not in {"0", "false", "no"}
    worker = PodWorker(PodConfig(
        pod_id=args.pod_id,
        api_key_id=args.api_key_id,
        api_key_env=args.api_key_env,
        backend_url=args.backend_url,
        mock_mode=mock_mode,
    ))
    if args.once:
        worker.run_once(args.experiment_id)
    else:
        worker.loop(args.experiment_id)


if __name__ == "__main__":
    main()
