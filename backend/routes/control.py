"""REST endpoints for experiments, tasks, artifacts, budget, and events."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from backend.config import artifacts_root
from backend.models import (
    Artifact,
    ArtifactCreate,
    BudgetLedgerEntry,
    Candidate,
    CompleteTaskRequest,
    Event,
    Experiment,
    ExperimentCreate,
    ExperimentStatus,
    FailTaskRequest,
    Intercept,
    InterceptCreate,
    InterceptResolveRequest,
    InterceptStatus,
    Task,
    TaskCreate,
    TaskStatus,
    now_iso,
)
from backend.storage import JsonStore

router = APIRouter()


def store(request: Request) -> JsonStore:
    """Return the request-scoped app store."""
    return request.app.state.store


@router.post("/experiments", response_model=Experiment)
def create_experiment(payload: ExperimentCreate, request: Request) -> Experiment:
    """Create and persist a new experiment record."""
    exp = Experiment(**payload.model_dump())
    return store(request).create_experiment(exp)


@router.get("/experiments", response_model=list[Experiment])
def list_experiments(request: Request) -> list[Experiment]:
    """Return all experiments known to the local store."""
    return store(request).list_experiments()


@router.get("/experiments/{experiment_id}", response_model=Experiment)
def get_experiment(experiment_id: str, request: Request) -> Experiment:
    """Return a single experiment or a 404 when it is missing."""
    try:
        return store(request).get_experiment(experiment_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Experiment not found") from exc


@router.post("/experiments/{experiment_id}/start", response_model=Experiment)
def start_experiment(experiment_id: str, request: Request) -> Experiment:
    """Mark an experiment running and log the transition."""
    s = store(request)
    exp = s.get_experiment(experiment_id)
    exp.status = ExperimentStatus.running
    s.log_event(Event(experiment_id=experiment_id, event_type="experiment.started", message="Experiment started"))
    return s.save_experiment(exp)


@router.post("/experiments/{experiment_id}/pause", response_model=Experiment)
def pause_experiment(experiment_id: str, request: Request) -> Experiment:
    """Pause an experiment and record an audit event."""
    s = store(request)
    exp = s.get_experiment(experiment_id)
    exp.status = ExperimentStatus.paused
    s.log_event(Event(experiment_id=experiment_id, event_type="experiment.paused", message="Experiment paused"))
    return s.save_experiment(exp)


@router.post("/experiments/{experiment_id}/resume", response_model=Experiment)
def resume_experiment(experiment_id: str, request: Request) -> Experiment:
    """Resume an experiment and record an audit event."""
    s = store(request)
    exp = s.get_experiment(experiment_id)
    exp.status = ExperimentStatus.running
    s.log_event(Event(experiment_id=experiment_id, event_type="experiment.resumed", message="Experiment resumed"))
    return s.save_experiment(exp)


@router.post("/candidates", response_model=Candidate)
def create_candidate(candidate: Candidate, request: Request) -> Candidate:
    """Create a candidate engine record for an experiment arm."""
    return store(request).add_candidate(candidate)


@router.get("/experiments/{experiment_id}/candidates", response_model=list[Candidate])
def list_candidates(experiment_id: str, request: Request) -> list[Candidate]:
    """List candidate engines for one experiment."""
    return store(request).list_candidates(experiment_id)


@router.patch("/experiments/{experiment_id}/candidates/{candidate_id}", response_model=Candidate)
def update_candidate(experiment_id: str, candidate_id: str, patch: dict, request: Request) -> Candidate:
    """Patch candidate metadata such as status, tokens, or eval summary."""
    s = store(request)
    for candidate in s.list_candidates(experiment_id):
        if candidate.id == candidate_id:
            data = candidate.model_dump()
            data.update(patch)
            return s.update_candidate(Candidate.model_validate(data))
    raise HTTPException(status_code=404, detail="Candidate not found")


@router.post("/tasks", response_model=Task)
def create_task(payload: TaskCreate, request: Request) -> Task:
    """Create a pod task, optionally holding it for review."""
    task = Task(**payload.model_dump())
    if payload.requires_review:
        task.status = TaskStatus.pending_user_review
    return store(request).add_task(task)


@router.get("/experiments/{experiment_id}/tasks", response_model=list[Task])
def list_tasks(experiment_id: str, request: Request) -> list[Task]:
    """List all tasks for one experiment."""
    return store(request).list_tasks(experiment_id)


@router.post("/experiments/{experiment_id}/pods/{pod_id}/next-task", response_model=Task | None)
def next_task(experiment_id: str, pod_id: str, request: Request) -> Task | None:
    """Atomically claim the next pending task for a pod."""
    return store(request).next_task_for_pod(experiment_id, pod_id)


@router.post("/experiments/{experiment_id}/tasks/{task_id}/claim", response_model=Task)
def claim_task(experiment_id: str, task_id: str, pod_id: str, request: Request) -> Task:
    """Claim a specific task for a pod if it is claimable."""
    s = store(request)
    task = s.get_task(experiment_id, task_id)
    if task.status not in {TaskStatus.pending, TaskStatus.claimed}:
        raise HTTPException(status_code=409, detail=f"Task is {task.status}")
    task.status = TaskStatus.claimed
    task.claimed_by = pod_id
    s.log_event(Event(experiment_id=experiment_id, event_type="task.claimed", message=f"{pod_id} claimed {task.title}", payload={"task_id": task.id}))
    return s.update_task(task)


@router.post("/experiments/{experiment_id}/tasks/{task_id}/complete", response_model=Task)
def complete_task(experiment_id: str, task_id: str, payload: CompleteTaskRequest, request: Request) -> Task:
    """Complete a task and attach artifacts and optional budget data."""
    s = store(request)
    task = s.get_task(experiment_id, task_id)
    task.status = TaskStatus.completed
    task.result_artifact_ids.extend(payload.artifact_ids)
    if payload.budget_entry:
        s.add_budget(payload.budget_entry)
    s.log_event(Event(experiment_id=experiment_id, event_type="task.completed", message=task.title, payload={"task_id": task.id, "artifacts": payload.artifact_ids}))
    return s.update_task(task)


@router.post("/experiments/{experiment_id}/tasks/{task_id}/fail", response_model=Task)
def fail_task(experiment_id: str, task_id: str, payload: FailTaskRequest, request: Request) -> Task:
    """Mark a task failed and persist the failure reason."""
    s = store(request)
    task = s.get_task(experiment_id, task_id)
    task.status = TaskStatus.failed
    task.error = payload.error
    if payload.budget_entry:
        s.add_budget(payload.budget_entry)
    s.log_event(Event(experiment_id=experiment_id, event_type="task.failed", message=task.title, payload={"task_id": task.id, "error": payload.error}))
    return s.update_task(task)


@router.post("/artifacts", response_model=Artifact)
def create_artifact(payload: ArtifactCreate, request: Request) -> Artifact:
    """Persist a structured artifact and mirror it to the artifacts tree."""
    return store(request).add_artifact(Artifact(**payload.model_dump()))


@router.get("/experiments/{experiment_id}/artifacts", response_model=list[Artifact])
def list_artifacts(experiment_id: str, request: Request, candidate_id: str | None = None) -> list[Artifact]:
    """List artifacts for one experiment, optionally filtered by candidate."""
    return store(request).list_artifacts(experiment_id, candidate_id)


@router.get("/experiments/{experiment_id}/artifact-summaries")
def list_artifact_summaries(experiment_id: str, request: Request, candidate_id: str | None = None) -> list[dict]:
    """Return artifact metadata without large content payloads."""
    return [
        {
            "id": artifact.id,
            "experiment_id": artifact.experiment_id,
            "arm": artifact.arm,
            "artifact_type": artifact.artifact_type,
            "title": artifact.title,
            "candidate_id": artifact.candidate_id,
            "task_id": artifact.task_id,
            "created_at": artifact.created_at,
            "content_preview": _preview_json(artifact.content),
        }
        for artifact in store(request).list_artifacts(experiment_id, candidate_id)
    ]


@router.get("/experiments/{experiment_id}/artifacts/{artifact_id}", response_model=Artifact)
def get_artifact(experiment_id: str, artifact_id: str, request: Request) -> Artifact:
    """Return one full artifact by id."""
    for artifact in store(request).list_artifacts(experiment_id):
        if artifact.id == artifact_id:
            return artifact
    raise HTTPException(status_code=404, detail="Artifact not found")


@router.get("/experiments/{experiment_id}/budget", response_model=list[BudgetLedgerEntry])
def list_budget(experiment_id: str, request: Request) -> list[BudgetLedgerEntry]:
    """Return budget ledger entries for an experiment."""
    return store(request).list_budget(experiment_id)


@router.post("/budget", response_model=BudgetLedgerEntry)
def add_budget(entry: BudgetLedgerEntry, request: Request) -> BudgetLedgerEntry:
    """Append a budget ledger entry."""
    return store(request).add_budget(entry)


@router.get("/experiments/{experiment_id}/events", response_model=list[Event])
def list_events(experiment_id: str, request: Request, limit: int = 200) -> list[Event]:
    """Return recent audit events for an experiment."""
    return store(request).list_events(experiment_id, limit)


@router.post("/events", response_model=Event)
def create_event(event: Event, request: Request) -> Event:
    """Append an audit event."""
    return store(request).log_event(event)


@router.get("/experiments/{experiment_id}/research-trace", response_model=list[Event])
def research_trace(experiment_id: str, request: Request, limit: int = 200) -> list[Event]:
    """Return explicit trace events, or synthesize them for older runs."""
    s = store(request)
    events = s.list_events(experiment_id, limit)
    trace = [event for event in events if event.event_type.startswith("research.")]
    if trace:
        return trace
    return _synthesize_research_trace(s, experiment_id, events)


@router.post("/intercepts", response_model=Intercept)
def create_intercept(payload: InterceptCreate, request: Request) -> Intercept:
    """Create a human/operator approval intercept."""
    return store(request).add_intercept(Intercept(**payload.model_dump()))


@router.get("/experiments/{experiment_id}/intercepts", response_model=list[Intercept])
def list_intercepts(experiment_id: str, request: Request) -> list[Intercept]:
    """List operator intercepts for an experiment."""
    return store(request).list_intercepts(experiment_id)


@router.post("/experiments/{experiment_id}/intercepts/{intercept_id}/resolve", response_model=Intercept)
def resolve_intercept(experiment_id: str, intercept_id: str, payload: InterceptResolveRequest, request: Request) -> Intercept:
    """Resolve an operator intercept and log the decision."""
    s = store(request)
    for intercept in s.list_intercepts(experiment_id):
        if intercept.id == intercept_id:
            intercept.status = payload.action
            intercept.modified_object = payload.modified_object
            intercept.resolution_note = payload.note
            intercept.resolved_at = now_iso()
            s.log_event(Event(experiment_id=experiment_id, event_type="intercept.resolved", message=f"{payload.action}: {intercept.reason}", payload={"intercept_id": intercept_id}))
            return s.update_intercept(intercept)
    raise HTTPException(status_code=404, detail="Intercept not found")


@router.get("/experiments/{experiment_id}/leaderboard")
def leaderboard(experiment_id: str, request: Request) -> list[dict]:
    """Return candidates sorted by latest score/win rate."""
    candidates = store(request).list_candidates(experiment_id)
    return sorted(
        [c.model_dump(mode="json") | {"score": c.eval_summary.get("score", c.eval_summary.get("win_rate", 0.0))} for c in candidates],
        key=lambda row: row["score"],
        reverse=True,
    )


@router.get("/logs/eval-runs")
def eval_run_logs(limit: int = 100) -> list[dict]:
    """Return recent evaluation run log rows."""
    return _read_jsonl_tail(artifacts_root() / "logs/eval_runs.jsonl", limit)


@router.get("/logs/engine-checks")
def engine_check_logs(limit: int = 100) -> list[dict]:
    """Return recent engine-check log rows."""
    return _read_jsonl_tail(artifacts_root() / "logs/engine_checks.jsonl", limit)


def _read_jsonl_tail(path: Path, limit: int) -> list[dict]:
    """Read the newest JSONL rows for UI log panels."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    rows = []
    for line in lines:
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _preview_json(value: object, max_chars: int = 600) -> str:
    """Small JSON preview for REST dashboards."""
    text = json.dumps(value, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + " ..."


def _synthesize_research_trace(s: JsonStore, experiment_id: str, events: list[Event]) -> list[Event]:
    """Backfill iteration-level trace rows from artifacts and candidate state."""
    candidates = s.list_candidates(experiment_id)
    artifacts = s.list_artifacts(experiment_id)
    budget = s.list_budget(experiment_id)
    trace: list[Event] = []

    if not candidates and not artifacts:
        return []

    arms = sorted({candidate.arm for candidate in candidates} | {artifact.arm for artifact in artifacts})
    for arm in arms:
        if arm == "comparison":
            continue
        arm_candidates = [candidate for candidate in candidates if candidate.arm == arm]
        trace.append(
            Event(
                experiment_id=experiment_id,
                event_type="research.synthetic.arm_seen",
                message=f"Observed historical arm: {arm}",
                payload={"arm": arm, "candidate_count": len(arm_candidates), "synthetic": True},
            )
        )
        if arm_candidates:
            trace.append(
                Event(
                    experiment_id=experiment_id,
                    event_type="research.synthetic.candidates_initialized",
                    message=f"{len(arm_candidates)} candidates existed for {arm}",
                    payload={
                        "arm": arm,
                        "synthetic": True,
                        "candidates": [
                            {"candidate_id": c.id, "name": c.name, "architecture": c.architecture, "status": c.status}
                            for c in arm_candidates
                        ],
                    },
                )
            )

        eval_artifacts = [artifact for artifact in artifacts if artifact.arm == arm and artifact.artifact_type == "EvalResult"]
        for artifact in eval_artifacts:
            content = artifact.content
            trace.append(
                Event(
                    experiment_id=experiment_id,
                    event_type="research.synthetic.eval_completed",
                    message=(
                        f"{artifact.title}: win_rate={content.get('win_rate', 'n/a')}, "
                        f"illegal={content.get('illegal_moves', 'n/a')}, crashes={content.get('crashes', 'n/a')}"
                    ),
                    payload={"arm": arm, "synthetic": True, "artifact_id": artifact.id, "candidate_id": artifact.candidate_id, "eval": content},
                    created_at=artifact.created_at,
                )
            )

        for candidate in arm_candidates:
            if candidate.status == "pruned":
                trace.append(
                    Event(
                        experiment_id=experiment_id,
                        event_type="research.synthetic.pruned",
                        message=f"Pruned {candidate.name}",
                        payload={"arm": arm, "synthetic": True, "candidate_id": candidate.id, "reason": candidate.prune_reason},
                        created_at=candidate.updated_at,
                    )
                )
            if candidate.status == "final":
                trace.append(
                    Event(
                        experiment_id=experiment_id,
                        event_type="research.synthetic.final_selected",
                        message=f"Final candidate selected: {candidate.name}",
                        payload={"arm": arm, "synthetic": True, "candidate_id": candidate.id, "eval_summary": candidate.eval_summary},
                        created_at=candidate.updated_at,
                    )
                )

        tokens = sum((entry.actual_tokens or entry.estimated_tokens) for entry in budget if entry.experiment_arm == arm)
        if tokens:
            trace.append(
                Event(
                    experiment_id=experiment_id,
                    event_type="research.synthetic.budget_used",
                    message=f"{arm} used {tokens} estimated/actual tokens",
                    payload={"arm": arm, "synthetic": True, "tokens": tokens},
                )
            )

    comparison = [artifact for artifact in artifacts if artifact.arm == "comparison" or artifact.title == "Final comparison report"]
    for artifact in comparison:
        trace.append(
            Event(
                experiment_id=experiment_id,
                event_type="research.synthetic.comparison_completed",
                message=f"Historical comparison artifact: {artifact.title}",
                payload={"arm": "comparison", "synthetic": True, "artifact_id": artifact.id, "content": artifact.content},
                created_at=artifact.created_at,
            )
        )

    existing_by_type = {event.event_type for event in events}
    if "experiment.created" in existing_by_type:
        first = next(event for event in events if event.event_type == "experiment.created")
        trace.insert(
            0,
            Event(
                experiment_id=experiment_id,
                event_type="research.synthetic.experiment_created",
                message="Historical experiment existed before detailed trace logging was enabled",
                payload={"synthetic": True},
                created_at=first.created_at,
            ),
        )

    return sorted(trace, key=lambda event: event.created_at)
