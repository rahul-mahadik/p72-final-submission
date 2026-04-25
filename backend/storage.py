"""Thread-safe JSON/JSONL persistence for local hackathon runs."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar

from pydantic import BaseModel

from backend.config import artifacts_root, experiments_root
from backend.models import (
    Artifact,
    BudgetLedgerEntry,
    Candidate,
    Event,
    Experiment,
    Intercept,
    Task,
    TaskStatus,
    now_iso,
)

T = TypeVar("T", bound=BaseModel)


class JsonStore:
    """Small durable store backed by per-experiment JSON files."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else experiments_root()
        self.artifacts_root = artifacts_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def _exp_dir(self, experiment_id: str) -> Path:
        path = self.root / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, experiment_id: str, name: str) -> Path:
        return self._exp_dir(experiment_id) / name

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _append_jsonl(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, sort_keys=True) + "\n")

    def _list_file(self, experiment_id: str, name: str) -> Path:
        return self._path(experiment_id, name)

    def _load_list(self, experiment_id: str, name: str, cls: type[T]) -> list[T]:
        rows = self._read_json(self._list_file(experiment_id, name), [])
        return [cls.model_validate(row) for row in rows]

    def _save_list(self, experiment_id: str, name: str, rows: list[BaseModel]) -> None:
        self._write_json(self._list_file(experiment_id, name), [r.model_dump(mode="json") for r in rows])

    def create_experiment(self, exp: Experiment) -> Experiment:
        """Create experiment files and initialize empty run collections."""
        with self.lock:
            self._write_json(self._path(exp.id, "experiment.json"), exp.model_dump(mode="json"))
            for name in ["tasks.json", "artifacts.json", "candidates.json", "budget.json", "intercepts.json"]:
                self._write_json(self._path(exp.id, name), [])
            self.log_event(Event(experiment_id=exp.id, event_type="experiment.created", message=f"Created {exp.name}"))
            return exp

    def get_experiment(self, experiment_id: str) -> Experiment:
        """Load one experiment by id."""
        return Experiment.model_validate(self._read_json(self._path(experiment_id, "experiment.json"), {}))

    def list_experiments(self) -> list[Experiment]:
        """Load all experiments with durable metadata."""
        experiments: list[Experiment] = []
        for path in sorted(self.root.glob("*/experiment.json")):
            experiments.append(Experiment.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return experiments

    def save_experiment(self, exp: Experiment) -> Experiment:
        """Persist updated experiment metadata."""
        exp.updated_at = now_iso()
        with self.lock:
            self._write_json(self._path(exp.id, "experiment.json"), exp.model_dump(mode="json"))
        return exp

    def log_event(self, event: Event) -> Event:
        """Append an audit event to the experiment JSONL stream."""
        exp_id = event.experiment_id or "_global"
        with self.lock:
            self._append_jsonl(self._path(exp_id, "events.jsonl"), event.model_dump(mode="json"))
        return event

    def list_events(self, experiment_id: str, limit: int = 200) -> list[Event]:
        """Return the newest audit events for an experiment."""
        path = self._path(experiment_id, "events.jsonl")
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [Event.model_validate(json.loads(line)) for line in lines if line.strip()]

    def add_task(self, task: Task) -> Task:
        """Persist a task and log its creation."""
        with self.lock:
            tasks = self._load_list(task.experiment_id, "tasks.json", Task)
            tasks.append(task)
            self._save_list(task.experiment_id, "tasks.json", tasks)
            self.log_event(Event(experiment_id=task.experiment_id, event_type="task.created", message=task.title, payload={"task_id": task.id, "status": task.status}))
        return task

    def list_tasks(self, experiment_id: str) -> list[Task]:
        """Return all tasks for an experiment."""
        return self._load_list(experiment_id, "tasks.json", Task)

    def update_task(self, task: Task) -> Task:
        """Replace one task in its experiment task list."""
        task.updated_at = now_iso()
        with self.lock:
            tasks = self._load_list(task.experiment_id, "tasks.json", Task)
            tasks = [task if row.id == task.id else row for row in tasks]
            self._save_list(task.experiment_id, "tasks.json", tasks)
        return task

    def get_task(self, experiment_id: str, task_id: str) -> Task:
        """Return one task or raise KeyError."""
        for task in self.list_tasks(experiment_id):
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def next_task_for_pod(self, experiment_id: str, pod_id: str) -> Task | None:
        """Claim and return the next pending task available to a pod."""
        with self.lock:
            tasks = self._load_list(experiment_id, "tasks.json", Task)
            for task in tasks:
                if task.status == TaskStatus.pending and task.pod_id in (None, pod_id):
                    task.status = TaskStatus.claimed
                    task.claimed_by = pod_id
                    task.updated_at = now_iso()
                    self._save_list(experiment_id, "tasks.json", tasks)
                    self.log_event(Event(experiment_id=experiment_id, event_type="task.claimed", message=f"{pod_id} claimed {task.title}", payload={"task_id": task.id, "pod_id": pod_id}))
                    return task
        return None

    def add_artifact(self, artifact: Artifact) -> Artifact:
        """Persist artifact metadata and mirror full content under artifacts."""
        with self.lock:
            rows = self._load_list(artifact.experiment_id, "artifacts.json", Artifact)
            rows.append(artifact)
            self._save_list(artifact.experiment_id, "artifacts.json", rows)
            artifact_path = self.artifacts_root / artifact.experiment_id / f"{artifact.id}.json"
            self._write_json(artifact_path, artifact.model_dump(mode="json"))
            self.log_event(Event(experiment_id=artifact.experiment_id, event_type="artifact.created", message=artifact.title, payload={"artifact_id": artifact.id, "type": artifact.artifact_type}))
        return artifact

    def list_artifacts(self, experiment_id: str, candidate_id: str | None = None) -> list[Artifact]:
        """Return artifacts for an experiment with optional candidate filtering."""
        rows = self._load_list(experiment_id, "artifacts.json", Artifact)
        if candidate_id:
            rows = [row for row in rows if row.candidate_id == candidate_id]
        return rows

    def add_candidate(self, candidate: Candidate) -> Candidate:
        """Persist a candidate and log its creation."""
        with self.lock:
            rows = self._load_list(candidate.experiment_id, "candidates.json", Candidate)
            rows.append(candidate)
            self._save_list(candidate.experiment_id, "candidates.json", rows)
            self.log_event(Event(experiment_id=candidate.experiment_id, event_type="candidate.created", message=candidate.name, payload={"candidate_id": candidate.id, "architecture": candidate.architecture}))
        return candidate

    def list_candidates(self, experiment_id: str) -> list[Candidate]:
        """Return all candidates for an experiment."""
        return self._load_list(experiment_id, "candidates.json", Candidate)

    def update_candidate(self, candidate: Candidate) -> Candidate:
        """Replace one candidate in its experiment candidate list."""
        candidate.updated_at = now_iso()
        with self.lock:
            rows = self._load_list(candidate.experiment_id, "candidates.json", Candidate)
            rows = [candidate if row.id == candidate.id else row for row in rows]
            self._save_list(candidate.experiment_id, "candidates.json", rows)
        return candidate

    def add_budget(self, entry: BudgetLedgerEntry) -> BudgetLedgerEntry:
        """Append a budget entry to JSON and JSONL ledgers."""
        with self.lock:
            rows = self._load_list(entry.experiment_id, "budget.json", BudgetLedgerEntry)
            rows.append(entry)
            self._save_list(entry.experiment_id, "budget.json", rows)
            self._append_jsonl(self._path(entry.experiment_id, "budget.jsonl"), entry.model_dump(mode="json"))
        return entry

    def list_budget(self, experiment_id: str) -> list[BudgetLedgerEntry]:
        """Return all budget ledger entries for an experiment."""
        return self._load_list(experiment_id, "budget.json", BudgetLedgerEntry)

    def add_intercept(self, intercept: Intercept) -> Intercept:
        """Persist an operator intercept and log its creation."""
        with self.lock:
            rows = self._load_list(intercept.experiment_id, "intercepts.json", Intercept)
            rows.append(intercept)
            self._save_list(intercept.experiment_id, "intercepts.json", rows)
            self.log_event(Event(experiment_id=intercept.experiment_id, event_type="intercept.created", message=intercept.reason, payload={"intercept_id": intercept.id}))
        return intercept

    def list_intercepts(self, experiment_id: str) -> list[Intercept]:
        """Return operator intercepts for an experiment."""
        return self._load_list(experiment_id, "intercepts.json", Intercept)

    def update_intercept(self, intercept: Intercept) -> Intercept:
        """Replace one intercept after an operator resolution."""
        with self.lock:
            rows = self._load_list(intercept.experiment_id, "intercepts.json", Intercept)
            rows = [intercept if row.id == intercept.id else row for row in rows]
            self._save_list(intercept.experiment_id, "intercepts.json", rows)
        return intercept
