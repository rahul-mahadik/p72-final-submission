"""Experiment-level coordinator for AutoResearch Chess Lab.

The orchestrator is intentionally small and auditable: it talks to the
FastAPI backend, asks the LLM adapter for compact role outputs, runs local
chess evaluations, records token/budget entries, and writes final reports.
Mock mode exercises the whole control plane without spending API credits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import requests

from agents.code_mutation import (
    apply_file_replacements,
    candidate_allowed_prefixes,
    collect_candidate_source_context,
    extract_file_replacements,
    repo_root,
    revert_changes,
)
from backend.config import artifacts_root
from agents.llm_adapter import create_llm_adapter
from backend.models import (
    ArtifactCreate,
    ArtifactType,
    BudgetLedgerEntry,
    Candidate,
    CandidateStatus,
    ContextPacket,
    Event,
    ExperimentCreate,
    ExperimentMode,
    InterceptCreate,
    InterceptStatus,
    TaskCreate,
)
from tournament.runner import evaluate_engine, head_to_head

CANDIDATES = [
    ("C1 Classical Alpha-Beta", "alphabeta", "engine.candidates.alphabeta.engine"),
    ("C2 MCTS / PUCT-Style", "mcts", "engine.candidates.mcts.engine"),
    ("C3 NNUE-Lite Value", "nnue_lite", "engine.candidates.nnue_lite.engine"),
    ("C4 Policy-Guided Search", "policy_guided", "engine.candidates.policy_guided.engine"),
]


class HumanRejectedStep(RuntimeError):
    """Raised when a human rejects a gated experiment step."""


class Orchestrator:
    """High-level driver for AutoResearch and Single-Best experiment arms."""

    def __init__(
        self,
        backend_url: str = "http://127.0.0.1:8000",
        mock_mode: bool | None = None,
        human_in_loop: bool = False,
        autoresearch_iterations: int = 1,
        dev_games_per_position: int = 1,
        heldout_games_per_position: int = 2,
        dev_move_budget_ms: int = 100,
        heldout_move_budget_ms: int = 200,
    ) -> None:
        self.backend_url = backend_url
        self.mock_mode = os.getenv("MOCK_MODE", "true").lower() not in {"0", "false", "no"} if mock_mode is None else mock_mode
        self.human_in_loop = human_in_loop
        self.autoresearch_iterations = max(1, autoresearch_iterations)
        self.dev_games_per_position = max(1, dev_games_per_position)
        self.heldout_games_per_position = max(1, heldout_games_per_position)
        self.dev_move_budget_ms = max(1, dev_move_budget_ms)
        self.heldout_move_budget_ms = max(1, heldout_move_budget_ms)
        self.repo_root = repo_root()
        self.parent_llm = create_llm_adapter(mock_mode=self.mock_mode, api_key_env=self._key_env("parent"))
        self.child_a_llm = create_llm_adapter(mock_mode=self.mock_mode, api_key_env=self._key_env("child-a"))
        self.child_b_llm = create_llm_adapter(mock_mode=self.mock_mode, api_key_env=self._key_env("child-b"))
        self.llm = self.parent_llm

    def create_experiment(self, name: str, mode: ExperimentMode, token_budget_cap: int = 20_000) -> dict:
        """Create the durable experiment record in the backend."""
        return self._post("/experiments", ExperimentCreate(name=name, mode=mode, token_budget_cap=token_budget_cap).model_dump(mode="json"))

    def bootstrap_candidates(self, experiment_id: str, arm: str) -> list[dict]:
        """Create the four frozen starting candidates for one experiment arm."""
        candidates = []
        for name, arch, module in CANDIDATES:
            candidate = Candidate(experiment_id=experiment_id, arm=arm, name=name, architecture=arch, module_path=module)
            created = self._post("/candidates", candidate.model_dump(mode="json"))
            self._artifact(experiment_id, arm, ArtifactType.CandidateCard, f"CandidateCard: {name}", created, created["id"])
            candidates.append(created)
        return candidates

    def create_candidate_tasks(self, candidates: list[dict], arm: str, per_candidate_budget: int = 1200) -> None:
        """Create one compact development task per candidate."""
        for candidate in candidates:
            context = ContextPacket(
                role="development",
                objective=f"Validate and improve {candidate['name']} while preserving select_move(position, time_budget_ms).",
                experiment_arm=arm,
                candidate_id=candidate["id"],
                candidate_state_summary=f"{candidate['architecture']} candidate {candidate['name']} status={candidate['status']}",
                constraints=["Use python-chess", "Return legal moves", "Run smoke checks before completion"],
                allowed_files=[f"engine/candidates/{candidate['architecture']}", "tests"],
                max_token_budget=per_candidate_budget,
                expected_output_artifact_type=ArtifactType.PatchSummary,
            )
            self._post("/tasks", TaskCreate(
                experiment_id=candidate["experiment_id"],
                arm=arm,
                title=f"Develop {candidate['name']}",
                role="development",
                objective=context.objective,
                candidate_id=candidate["id"],
                context=context,
                max_token_budget=per_candidate_budget,
            ).model_dump(mode="json"))

    def run_autoresearch_mock(self, experiment_id: str, frequent_human: bool = False) -> dict:
        """Run the AutoResearch arm with logical pods and local evaluations."""
        arm = "autoresearch_frequent" if frequent_human else "autoresearch_minimal"
        self._post(f"/experiments/{experiment_id}/start")
        self._trace(experiment_id, arm, "research.arm_started", "Started AutoResearch arm", {"mode": arm})
        plan_started = time.perf_counter()
        plan = self.parent_llm.parent_plan({"experiment_id": experiment_id, "arm": arm, "objective": "Plan initial AutoResearch exploration across four chess engine candidates."})
        self._human_gate(
            experiment_id,
            arm,
            "Approve parent AutoResearch plan",
            {"step": "parent_plan", "plan": plan.content, "tokens": plan.actual_tokens or plan.estimated_tokens},
            high_impact=True,
        )
        self._budget(
            experiment_id,
            arm,
            "parent",
            os.getenv("PARENT_API_KEY_ID", "key-1"),
            "planning",
            None,
            plan.actual_tokens or plan.estimated_tokens,
            "completed",
            "initial autoresearch plan",
            wall_clock_seconds=time.perf_counter() - plan_started,
            token_source="actual" if plan.actual_tokens else "estimated",
        )
        self._artifact(experiment_id, arm, ArtifactType.FeedbackBrief, "Parent planning output", plan.content)
        self._trace(experiment_id, arm, "research.parent_plan_completed", "Parent planner produced initial AutoResearch plan", {"plan": plan.content, "tokens": plan.actual_tokens or plan.estimated_tokens, "mock_mode": self.mock_mode})
        candidates = self.bootstrap_candidates(experiment_id, arm)
        self._trace(
            experiment_id,
            arm,
            "research.candidates_initialized",
            "Created four candidate cards from the frozen starting state",
            {"candidate_ids": [c["id"] for c in candidates], "architectures": [c["architecture"] for c in candidates]},
        )
        self.create_candidate_tasks(candidates, arm)
        active_candidates = candidates
        survivors: list[tuple[dict, dict]] = []
        mutation_records: list[dict[str, Any]] = []
        for iteration in range(1, self.autoresearch_iterations + 1):
            self._trace(
                experiment_id,
                arm,
                "research.iteration_started",
                f"Iteration {iteration}: explore/evaluate active candidates",
                {
                    "iteration": iteration,
                    "active_candidate_ids": [c["id"] for c in active_candidates],
                    "dev_games_per_position": self.dev_games_per_position,
                    "move_budget_ms": self.dev_move_budget_ms,
                },
            )
            evals = []
            for candidate in active_candidates:
                child_started = time.perf_counter()
                logical_pod = "child-a" if len(evals) % 2 == 0 else "child-b"
                allowed_prefixes = candidate_allowed_prefixes(candidate["architecture"])
                child_result = self._child_llm(logical_pod).child_develop(
                    {
                        "experiment_id": experiment_id,
                        "arm": arm,
                        "iteration": iteration,
                        "candidate_id": candidate["id"],
                        "candidate_name": candidate["name"],
                        "architecture": candidate["architecture"],
                        "objective": (
                            "Improve this candidate engine under the shared select_move(position, time_budget_ms) interface. "
                            "If you can make a safe code improvement from the provided source_files, return full-file replacements."
                        ),
                        "source_files": collect_candidate_source_context(candidate["architecture"], self.repo_root),
                        "allowed_write_prefixes": list(allowed_prefixes),
                        "output_schema": {
                            "patch_summary": "short explanation",
                            "file_replacements": [{"path": "relative/path.py", "content": "complete file content"}],
                            "checks_to_run": ["pytest tests/test_engines.py tests/test_tournament.py -q"],
                            "risk_notes": ["..."],
                        },
                        "constraints": [
                            "Use the configured key slot for this logical pod.",
                            "Treat pod identity as auditable.",
                            "Do not read other arms' artifacts.",
                            "Only propose full-file replacements under allowed_write_prefixes.",
                            "Preserve select_move(position, time_budget_ms) and python-chess legality.",
                        ],
                    }
                )
                self._human_gate(
                    experiment_id,
                    arm,
                    f"Approve {logical_pod} proposal for {candidate['name']} iteration {iteration}",
                    {
                        "step": "child_development_output",
                        "iteration": iteration,
                        "candidate_id": candidate["id"],
                        "candidate_name": candidate["name"],
                        "logical_pod": logical_pod,
                        "output": child_result.content,
                    },
                )
                mutation = self._apply_live_candidate_mutation(candidate, child_result.content, allowed_prefixes)
                mutation_records.append(
                    {
                        "iteration": iteration,
                        "candidate_id": candidate["id"],
                        "candidate_name": candidate["name"],
                        "architecture": candidate["architecture"],
                        "applied_count": len(mutation.get("applied", []) or []),
                        "rejected_count": len(mutation.get("rejected", []) or []),
                        "checks_ok": mutation.get("checks", {}).get("ok"),
                        "checks_skipped": mutation.get("checks", {}).get("skipped"),
                        "checks_reason": mutation.get("checks", {}).get("reason"),
                    }
                )
                if mutation.get("applied") or mutation.get("rejected"):
                    self._human_gate(
                        experiment_id,
                        arm,
                        f"Review mutation result for {candidate['name']} iteration {iteration}",
                        {"step": "mutation_result", "iteration": iteration, "candidate_id": candidate["id"], "mutation": mutation},
                        high_impact=bool(mutation.get("applied")),
                    )
                self._budget(
                    experiment_id,
                    arm,
                    logical_pod,
                    os.getenv("CHILD_A_API_KEY_ID" if logical_pod == "child-a" else "CHILD_B_API_KEY_ID", "key-2" if logical_pod == "child-a" else "key-3"),
                    "development",
                    candidate["id"],
                    child_result.actual_tokens or child_result.estimated_tokens,
                    "completed",
                    f"logical pod development analysis iteration {iteration}",
                    wall_clock_seconds=time.perf_counter() - child_started,
                    token_source="actual" if child_result.actual_tokens else "estimated",
                )
                self._artifact(
                    experiment_id,
                    arm,
                    ArtifactType.PatchSummary,
                    f"Logical pod patch i{iteration}: {candidate['name']}",
                    {"iteration": iteration, "llm_output": child_result.content, "mutation": mutation},
                    candidate["id"],
                )
                self._trace(
                    experiment_id,
                    arm,
                    "research.logical_pod_completed",
                    f"{logical_pod} completed iteration {iteration} analysis for {candidate['name']}",
                    {
                        "iteration": iteration,
                        "candidate_id": candidate["id"],
                        "logical_pod": logical_pod,
                        "output": child_result.content,
                        "mutation": mutation,
                        "tokens": child_result.actual_tokens or child_result.estimated_tokens,
                    },
                )
                self._trace(
                    experiment_id,
                    arm,
                    "research.eval_started",
                    f"Dev eval started for {candidate['name']} iteration {iteration}",
                    {
                        "iteration": iteration,
                        "candidate_id": candidate["id"],
                        "architecture": candidate["architecture"],
                        "move_budget_ms": self.dev_move_budget_ms,
                        "games_per_position": self.dev_games_per_position,
                    },
                )
                summary = evaluate_engine(
                    candidate["architecture"],
                    move_budget_ms=self.dev_move_budget_ms,
                    games_per_position=self.dev_games_per_position,
                )
                self._human_gate(
                    experiment_id,
                    arm,
                    f"Review dev eval for {candidate['name']} iteration {iteration}",
                    {"step": "dev_eval", "iteration": iteration, "candidate_id": candidate["id"], "summary": summary},
                )
                evals.append((candidate, summary))
                self._artifact(experiment_id, arm, ArtifactType.EvalResult, f"Dev eval i{iteration}: {candidate['name']}", summary, candidate["id"])
                self._budget(experiment_id, arm, "parent", os.getenv("PARENT_API_KEY_ID", "key-1"), "tournament", candidate["id"], 250 * self.dev_games_per_position, "completed", f"dev eval iteration {iteration}")
                self._trace(
                    experiment_id,
                    arm,
                    "research.eval_completed",
                    f"Dev eval completed for {candidate['name']} iteration {iteration}: win_rate={summary['win_rate']:.2f}, illegal={summary['illegal_moves']}, crashes={summary['crashes']}",
                    {
                        "iteration": iteration,
                        "candidate_id": candidate["id"],
                        "architecture": candidate["architecture"],
                        "win_rate": summary["win_rate"],
                        "elo_delta": summary["elo_delta"],
                        "illegal_moves": summary["illegal_moves"],
                        "crashes": summary["crashes"],
                        "avg_move_latency_ms": summary["avg_move_latency_ms"],
                    },
                )
            ranked = sorted(evals, key=lambda pair: (pair[1]["illegal_moves"], pair[1]["crashes"], -pair[1]["win_rate"]))
            survivor_count = min(len(ranked), 2)
            survivors = ranked[:survivor_count]
            pruned = ranked[survivor_count:] if iteration == 1 else []
            self._trace(
                experiment_id,
                arm,
                "research.iteration_decision",
                f"Iteration {iteration} decision: keep top {survivor_count} active candidates",
                {
                    "iteration": iteration,
                    "survivors": [{"candidate_id": c["id"], "name": c["name"], "win_rate": s["win_rate"]} for c, s in survivors],
                    "pruned": [{"candidate_id": c["id"], "name": c["name"], "win_rate": s["win_rate"]} for c, s in pruned],
                },
            )
            self._human_gate(
                experiment_id,
                arm,
                f"Approve iteration {iteration} prune/survivor decision",
                {
                    "step": "prune_decision",
                    "iteration": iteration,
                    "survivors": [{"candidate_id": c["id"], "name": c["name"], "win_rate": s["win_rate"]} for c, s in survivors],
                    "pruned": [{"candidate_id": c["id"], "name": c["name"], "win_rate": s["win_rate"]} for c, s in pruned],
                },
                high_impact=True,
            )
            for candidate, summary in pruned:
                self._patch_candidate(experiment_id, candidate["id"], {"status": CandidateStatus.pruned, "prune_reason": f"Pruned after iteration {iteration}: win_rate={summary['win_rate']:.2f}, crashes={summary['crashes']}"})
                self._artifact(experiment_id, arm, ArtifactType.PruneDecision, f"Pruned {candidate['name']}", {"iteration": iteration, "reason": "behind dev-eval leaders", "eval": summary}, candidate["id"])
                self._trace(
                    experiment_id,
                    arm,
                    "research.pruned",
                    f"Pruned {candidate['name']}",
                    {"iteration": iteration, "candidate_id": candidate["id"], "architecture": candidate["architecture"], "reason": f"win_rate={summary['win_rate']:.2f}, crashes={summary['crashes']}"},
                )
            active_candidates = [candidate for candidate, _summary in survivors]
        best_candidate, best_eval = max(survivors, key=lambda pair: pair[1]["win_rate"])
        self._trace(
            experiment_id,
            arm,
            "research.promotion_selected",
            f"Selected survivor for held-out eval: {best_candidate['name']}",
            {"candidate_id": best_candidate["id"], "architecture": best_candidate["architecture"], "dev_win_rate": best_eval["win_rate"]},
        )
        heldout = evaluate_engine(
            best_candidate["architecture"],
            heldout=True,
            move_budget_ms=self.heldout_move_budget_ms,
            games_per_position=self.heldout_games_per_position,
        )
        self._human_gate(
            experiment_id,
            arm,
            f"Approve held-out final eval for {best_candidate['name']}",
            {"step": "heldout_eval", "candidate_id": best_candidate["id"], "heldout": heldout},
            high_impact=True,
        )
        best_candidate = self._patch_candidate(experiment_id, best_candidate["id"], {"status": CandidateStatus.final, "eval_summary": heldout, "tokens_spent": self.total_tokens(experiment_id, arm)})
        self._artifact(experiment_id, arm, ArtifactType.PromotionDecision, f"Promoted {best_candidate['name']}", {"dev_eval": best_eval, "heldout_eval": heldout}, best_candidate["id"])
        self._trace(
            experiment_id,
            arm,
            "research.heldout_completed",
            f"Held-out eval completed for {best_candidate['name']}: win_rate={heldout['win_rate']:.2f}",
            {"candidate_id": best_candidate["id"], "heldout": heldout},
        )
        mutation_summary = {
            "development_calls": len(mutation_records),
            "accepted_file_replacements": sum(item["applied_count"] for item in mutation_records),
            "rejected_file_replacements": sum(item["rejected_count"] for item in mutation_records),
            "records": mutation_records,
        }
        report = self.final_report(
            experiment_id,
            arm,
            best_candidate,
            heldout,
            metadata={"iterations": self.autoresearch_iterations, "mutation_summary": mutation_summary},
        )
        self._trace(experiment_id, arm, "research.arm_completed", "Completed AutoResearch arm", {"tokens": self.total_tokens(experiment_id, arm), "final_candidate_id": best_candidate["id"], "mutation_summary": mutation_summary})
        return {
            "arm": arm,
            "winner": best_candidate,
            "heldout": heldout,
            "report": report,
            "tokens": self.total_tokens(experiment_id, arm),
            "iterations": self.autoresearch_iterations,
            "mutation_summary": mutation_summary,
        }

    def run_single_best_mock(self, experiment_id: str, token_cap: int) -> dict:
        """Run the Wiggums-style Single-Best arm against a fixed token cap."""
        arm = "single_best"
        self._trace(experiment_id, arm, "research.arm_started", "Started Single-Best arm", {"token_cap": token_cap})
        candidates = self.bootstrap_candidates(experiment_id, arm)
        selector_started = time.perf_counter()
        selector = self.parent_llm.parent_plan(
            {
                "experiment_id": experiment_id,
                "arm": arm,
                "objective": "Choose one candidate upfront using only candidate descriptions, without exploratory evals.",
                "candidates": [{"name": c["name"], "architecture": c["architecture"]} for c in candidates],
            }
        )
        self._human_gate(
            experiment_id,
            arm,
            "Approve Wiggums/Single-Best upfront selector",
            {"step": "single_best_selector", "selector": selector.content, "tokens": selector.actual_tokens or selector.estimated_tokens},
            high_impact=True,
        )
        self._budget(
            experiment_id,
            arm,
            "parent",
            os.getenv("PARENT_API_KEY_ID", "key-1"),
            "single_best_selection",
            None,
            selector.actual_tokens or selector.estimated_tokens,
            "completed",
            "single-best selector reasoning",
            wall_clock_seconds=time.perf_counter() - selector_started,
            token_source="actual" if selector.actual_tokens else "estimated",
        )
        self._artifact(experiment_id, arm, ArtifactType.FeedbackBrief, "Single-best selector output", selector.content)
        selected = next(c for c in candidates if c["architecture"] == "alphabeta")
        self._artifact(experiment_id, arm, ArtifactType.PromotionDecision, "Single-best upfront selection", {"selected": selected, "reason": "Prior: alpha-beta is robust for small deterministic chess engines."}, selected["id"])
        self._trace(
            experiment_id,
            arm,
            "research.single_best_selected",
            f"Single-Best selected upfront: {selected['name']}",
            {"candidate_id": selected["id"], "reason": "Prior: alpha-beta is robust for small deterministic chess engines."},
        )
        spend_remaining = max(0, token_cap - self.total_tokens(experiment_id, arm))
        self._budget(experiment_id, arm, "child-a", os.getenv("CHILD_A_API_KEY_ID", "key-2"), "development", selected["id"], spend_remaining, "completed", "spent on selected candidate")
        heldout = evaluate_engine(
            selected["architecture"],
            heldout=True,
            move_budget_ms=self.heldout_move_budget_ms,
            games_per_position=self.heldout_games_per_position,
        )
        self._human_gate(
            experiment_id,
            arm,
            f"Approve Single-Best held-out eval for {selected['name']}",
            {"step": "single_best_heldout_eval", "candidate_id": selected["id"], "heldout": heldout},
            high_impact=True,
        )
        selected = self._patch_candidate(experiment_id, selected["id"], {"status": CandidateStatus.final, "eval_summary": heldout, "tokens_spent": self.total_tokens(experiment_id, arm)})
        report = self.final_report(experiment_id, arm, selected, heldout)
        self._trace(experiment_id, arm, "research.arm_completed", "Completed Single-Best arm", {"tokens": self.total_tokens(experiment_id, arm), "final_candidate_id": selected["id"], "heldout_win_rate": heldout["win_rate"]})
        return {"arm": arm, "winner": selected, "heldout": heldout, "report": report, "tokens": self.total_tokens(experiment_id, arm)}

    def comparison_report(self, experiment_id: str, autoresearch: dict, single_best: dict) -> dict:
        """Store the final side-by-side comparison for the two experiment arms."""
        auto_eval = autoresearch["heldout"]
        single_eval = single_best["heldout"]
        direct_match = head_to_head(
            autoresearch["winner"]["architecture"],
            single_best["winner"]["architecture"],
            heldout=True,
            move_budget_ms=self.heldout_move_budget_ms,
            games_per_position=self.heldout_games_per_position,
        )
        self._artifact(
            experiment_id,
            "comparison",
            ArtifactType.EvalResult,
            "Final head-to-head: AutoResearch vs Single-Best",
            {
                "autoresearch_candidate_id": autoresearch["winner"]["id"],
                "single_best_candidate_id": single_best["winner"]["id"],
                "match": direct_match,
            },
        )
        report = {
            "experiment_id": experiment_id,
            "arms": {
                "autoresearch": {
                    "arm": autoresearch["arm"],
                    "tokens": autoresearch["tokens"],
                    "final_candidate": autoresearch["winner"],
                    "heldout_win_rate": auto_eval["win_rate"],
                    "elo_delta": auto_eval["elo_delta"],
                    "reference_elo": auto_eval.get("reference_elo", 1000),
                    "estimated_elo": auto_eval.get("estimated_elo"),
                    "illegal_moves": auto_eval["illegal_moves"],
                    "crashes": auto_eval["crashes"],
                    "avg_move_latency_ms": auto_eval["avg_move_latency_ms"],
                    "iterations": autoresearch.get("iterations"),
                    "mutation_summary": autoresearch.get("mutation_summary"),
                },
                "single_best": {
                    "arm": single_best["arm"],
                    "tokens": single_best["tokens"],
                    "final_candidate": single_best["winner"],
                    "heldout_win_rate": single_eval["win_rate"],
                    "elo_delta": single_eval["elo_delta"],
                    "reference_elo": single_eval.get("reference_elo", 1000),
                    "estimated_elo": single_eval.get("estimated_elo"),
                    "illegal_moves": single_eval["illegal_moves"],
                    "crashes": single_eval["crashes"],
                    "avg_move_latency_ms": single_eval["avg_move_latency_ms"],
                },
            },
            "direct_head_to_head": {
                "autoresearch_kind": autoresearch["winner"]["architecture"],
                "single_best_kind": single_best["winner"]["architecture"],
                "autoresearch_score": direct_match["score_for_first"],
                "single_best_score": direct_match["score_for_second"],
                "winner": (
                    "autoresearch"
                    if direct_match["winner"] == autoresearch["winner"]["architecture"]
                    else "single_best"
                    if direct_match["winner"] == single_best["winner"]["architecture"]
                    else "tie"
                ),
                "games": direct_match["games"],
                "elo_delta_autoresearch_vs_single_best": direct_match["elo_delta_first_vs_second"],
                "estimated_autoresearch_elo_if_single_best_1000": direct_match.get("estimated_first_elo_if_second_1000"),
                "artifact_files": direct_match.get("artifact_files", []),
            },
            "winner_by_win_rate": "autoresearch" if auto_eval["win_rate"] > single_eval["win_rate"] else "single_best" if single_eval["win_rate"] > auto_eval["win_rate"] else "tie",
            "winner_by_head_to_head": (
                "autoresearch"
                if direct_match["winner"] == autoresearch["winner"]["architecture"]
                else "single_best"
                if direct_match["winner"] == single_best["winner"]["architecture"]
                else "tie"
            ),
            "token_budget_matched": autoresearch["tokens"] == single_best["tokens"],
            "caveats": [
                "Baseline win rates can be stochastic because baseline play is random.",
                "Direct head-to-head is more relevant when both final engines are available.",
                "Use larger held-out samples for more stable estimates.",
            ],
        }
        self._artifact(experiment_id, "comparison", ArtifactType.FinalReport, "Final comparison report", report)
        artifacts_root().mkdir(parents=True, exist_ok=True)
        (artifacts_root() / f"{experiment_id}_comparison_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._trace(experiment_id, "comparison", "research.comparison_completed", f"Final comparison completed: {report['winner_by_win_rate']}", report)
        return report

    def _human_gate(
        self,
        experiment_id: str,
        arm: str,
        reason: str,
        original_object: dict[str, Any],
        high_impact: bool = False,
    ) -> dict[str, Any]:
        """Create an audit intercept and optionally pause for CLI approval.

        This is intentionally lightweight for the MVP. The dashboard can show
        every intercept, while `--human-in-loop` adds a blocking terminal prompt.
        Steering text is recorded as the resolution note but is not applied.
        """
        intercept = self._post(
            "/intercepts",
            InterceptCreate(
                experiment_id=experiment_id,
                arm=arm,
                reason=reason,
                original_object=original_object,
                high_impact=high_impact,
            ).model_dump(mode="json"),
        )
        if not self.human_in_loop:
            resolved = self._resolve_intercept(
                experiment_id,
                intercept["id"],
                InterceptStatus.approved,
                "auto-approved; HITL disabled",
            )
            self._trace(
                experiment_id,
                arm,
                "research.human_gate_auto_approved",
                reason,
                {"intercept_id": intercept["id"], "high_impact": high_impact},
            )
            return resolved

        print("\n=== Human gate ===")
        print(f"Reason: {reason}")
        print(json.dumps(original_object, indent=2, default=str)[:6000])
        print("Choose: [a]pprove / [r]eject / [s]teer facade")
        choice = input("> ").strip().lower()[:1] or "a"
        if choice == "r":
            note = input("Rejection note: ").strip() or "Rejected by operator"
            resolved = self._resolve_intercept(experiment_id, intercept["id"], InterceptStatus.rejected, note)
            self._trace(
                experiment_id,
                arm,
                "research.human_gate_rejected",
                reason,
                {"intercept_id": intercept["id"], "note": note, "high_impact": high_impact},
            )
            raise HumanRejectedStep(f"Human rejected step: {reason}")
        if choice == "s":
            note = input("Steering note (recorded only for MVP): ").strip() or "Steering note recorded; facade only"
            resolved = self._resolve_intercept(
                experiment_id,
                intercept["id"],
                InterceptStatus.approved,
                note,
                modified_object={"operator_note": note, "facade_only": True},
            )
            self._trace(
                experiment_id,
                arm,
                "research.human_gate_steered",
                reason,
                {"intercept_id": intercept["id"], "note": note, "facade_only": True, "high_impact": high_impact},
            )
            return resolved

        resolved = self._resolve_intercept(experiment_id, intercept["id"], InterceptStatus.approved, "Approved by operator")
        self._trace(
            experiment_id,
            arm,
            "research.human_gate_approved",
            reason,
            {"intercept_id": intercept["id"], "high_impact": high_impact},
        )
        return resolved

    def _resolve_intercept(
        self,
        experiment_id: str,
        intercept_id: str,
        action: InterceptStatus,
        note: str,
        modified_object: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            f"/experiments/{experiment_id}/intercepts/{intercept_id}/resolve",
            {
                "action": action.value,
                "note": note,
                "modified_object": modified_object,
            },
        )

    def _child_llm(self, logical_pod: str):
        """Return the live/mock adapter assigned to a logical child pod."""
        return self.child_a_llm if logical_pod == "child-a" else self.child_b_llm

    def _key_env(self, logical_pod: str) -> str:
        """Return Anthropic key env var for a logical pod."""
        prefix = logical_pod.upper().replace("-", "_")
        return f"{prefix}_ANTHROPIC_API_KEY"

    def _apply_live_candidate_mutation(
        self,
        candidate: dict,
        llm_content: dict[str, Any],
        allowed_prefixes: tuple[str, ...],
    ) -> dict[str, Any]:
        """Apply model-proposed file replacements, test them, and revert on failure."""
        if self.mock_mode:
            return {"mode": "mock", "applied": [], "rejected": [], "checks": {"skipped": True}}

        replacements = extract_file_replacements(llm_content)
        if not replacements:
            return {"mode": "live", "applied": [], "rejected": [], "checks": {"skipped": True, "reason": "no file_replacements"}}

        patch_summary, changes = apply_file_replacements(
            replacements,
            root=self.repo_root,
            allowed_prefixes=allowed_prefixes,
        )
        if not changes:
            return {"mode": "live", **patch_summary, "checks": {"skipped": True, "reason": "no accepted replacements"}}

        checks = self._run_quality_checks(candidate["architecture"])
        if not checks["ok"]:
            revert_changes(changes)
            checks["reverted"] = True
        return {"mode": "live", **patch_summary, "checks": checks}

    def _run_quality_checks(self, architecture: str) -> dict[str, Any]:
        """Run local checks after a live code mutation."""
        started = time.perf_counter()
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_engines.py",
            "tests/test_tournament.py",
            "-q",
        ]
        proc = subprocess.run(
            cmd,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=int(os.getenv("AUTORESEARCH_CHECK_TIMEOUT_SECONDS", "90")),
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        return {
            "ok": proc.returncode == 0,
            "architecture": architecture,
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "duration_seconds": time.perf_counter() - started,
            "output_tail": output[-4000:],
        }

    def final_report(self, experiment_id: str, arm: str, candidate: dict, heldout: dict, metadata: dict[str, Any] | None = None) -> dict:
        """Store a per-arm final report artifact and JSON export."""
        budget = self._get(f"/experiments/{experiment_id}/budget")
        relevant_budget = [b for b in budget if b["experiment_arm"] == arm]
        report = {
            "experiment_id": experiment_id,
            "arm": arm,
            "final_candidate": candidate,
            "heldout_eval": heldout,
            "tokens_used": sum(b["actual_tokens"] or b["estimated_tokens"] for b in relevant_budget),
            "budget_entries": relevant_budget,
            "metadata": metadata or {},
            "caveats": ["Mock LLM mode uses estimated tokens.", "Toy engines are intentionally simple.", "Win rates use small hackathon eval samples."],
        }
        self._artifact(experiment_id, arm, ArtifactType.FinalReport, f"Final report: {arm}", report, candidate["id"])
        artifacts_root().mkdir(parents=True, exist_ok=True)
        (artifacts_root() / f"{experiment_id}_{arm}_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def total_tokens(self, experiment_id: str, arm: str) -> int:
        """Return actual tokens when available, otherwise estimates, for one arm."""
        rows = self._get(f"/experiments/{experiment_id}/budget")
        return sum(row["actual_tokens"] or row["estimated_tokens"] for row in rows if row["experiment_arm"] == arm)

    def _artifact(self, experiment_id: str, arm: str, artifact_type: ArtifactType, title: str, content: dict[str, Any], candidate_id: str | None = None) -> dict:
        return self._post("/artifacts", ArtifactCreate(experiment_id=experiment_id, arm=arm, artifact_type=artifact_type, title=title, candidate_id=candidate_id, content=content).model_dump(mode="json"))

    def _budget(
        self,
        experiment_id: str,
        arm: str,
        pod_id: str,
        api_key_id: str,
        role: str,
        candidate_id: str | None,
        tokens: int,
        outcome: str,
        result: str,
        wall_clock_seconds: float = 0.0,
        token_source: str = "estimated",
    ) -> dict:
        actual_tokens = tokens if token_source == "actual" else None
        estimated_tokens = tokens
        return self._post(
            "/budget",
            BudgetLedgerEntry(
                experiment_id=experiment_id,
                experiment_arm=arm,
                pod_id=pod_id,
                api_key_id=api_key_id,
                agent_role=role,
                candidate_id=candidate_id,
                estimated_tokens=estimated_tokens,
                actual_tokens=actual_tokens,
                token_source=token_source,
                wall_clock_seconds=wall_clock_seconds,
                outcome=outcome,
                downstream_result=result,
            ).model_dump(mode="json"),
        )

    def _trace(self, experiment_id: str, arm: str, event_type: str, message: str, payload: dict[str, Any]) -> dict:
        return self._post("/events", Event(experiment_id=experiment_id, event_type=event_type, message=message, payload={"arm": arm, **payload}).model_dump(mode="json"))

    def _patch_candidate(self, experiment_id: str, candidate_id: str, patch: dict) -> dict:
        response = requests.patch(f"{self.backend_url}/experiments/{experiment_id}/candidates/{candidate_id}", json={k: (v.value if hasattr(v, "value") else v) for k, v in patch.items()}, timeout=60)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        response = requests.post(self.backend_url + path, json=payload or {}, timeout=120)
        response.raise_for_status()
        return response.json()

    def _get(self, path: str) -> Any:
        response = requests.get(self.backend_url + path, timeout=60)
        response.raise_for_status()
        return response.json()


def main() -> None:
    """CLI entry point for running mock or live logical-pod experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--full-mock", action="store_true")
    parser.add_argument("--live-logical-pods", action="store_true")
    parser.add_argument(
        "--human-in-loop",
        action="store_true",
        help="Pause at major gates for terminal approve/reject/steering facade prompts.",
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--token-budget", type=int, default=20_000)
    parser.add_argument("--autoresearch-iterations", type=int, default=int(os.getenv("AUTORESEARCH_ITERATIONS", "1")))
    parser.add_argument("--dev-games-per-position", type=int, default=int(os.getenv("DEV_GAMES_PER_POSITION", "1")))
    parser.add_argument("--heldout-games-per-position", type=int, default=int(os.getenv("HELDOUT_GAMES_PER_POSITION", "2")))
    parser.add_argument("--dev-move-budget-ms", type=int, default=int(os.getenv("DEV_MOVE_BUDGET_MS", "100")))
    parser.add_argument("--heldout-move-budget-ms", type=int, default=int(os.getenv("HELDOUT_MOVE_BUDGET_MS", "200")))
    args = parser.parse_args()
    orch = Orchestrator(
        args.backend_url,
        mock_mode=False if args.live_logical_pods else None,
        human_in_loop=args.human_in_loop,
        autoresearch_iterations=args.autoresearch_iterations,
        dev_games_per_position=args.dev_games_per_position,
        heldout_games_per_position=args.heldout_games_per_position,
        dev_move_budget_ms=args.dev_move_budget_ms,
        heldout_move_budget_ms=args.heldout_move_budget_ms,
    )
    exp = {"id": args.experiment_id} if args.experiment_id else orch.create_experiment("AutoResearch Chess Lab", ExperimentMode.minimal_human, args.token_budget)
    if args.full_mock or args.live_logical_pods:
        auto = orch.run_autoresearch_mock(exp["id"])
        single = orch.run_single_best_mock(exp["id"], auto["tokens"])
        comparison = orch.comparison_report(exp["id"], auto, single)
        print(json.dumps({"experiment_id": exp["id"], "autoresearch": auto, "single_best": single, "comparison": comparison}, indent=2))
    else:
        print(json.dumps(exp, indent=2))


if __name__ == "__main__":
    main()
