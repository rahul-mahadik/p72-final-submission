"""Pydantic schemas shared by the backend, agents, and UI."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def now_iso() -> str:
    """Return a timezone-aware UTC timestamp for audit records."""
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    """Create a short prefixed identifier suitable for local artifacts."""
    return f"{prefix}_{uuid4().hex[:12]}"


class ExperimentMode(StrEnum):
    """Supported high-level experiment execution modes."""

    minimal_human = "minimal_human"
    frequent_human = "frequent_human"
    single_best = "single_best"


class ExperimentStatus(StrEnum):
    """Lifecycle states for an experiment record."""

    draft = "draft"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"


class TaskStatus(StrEnum):
    """Lifecycle states for work dispatched to pods."""

    draft = "draft"
    pending_user_review = "pending_user_review"
    pending = "pending"
    claimed = "claimed"
    running = "running"
    completed = "completed"
    failed = "failed"
    rejected = "rejected"
    cancelled = "cancelled"


class ArtifactType(StrEnum):
    """Structured artifact categories written by agents and evaluators."""

    CandidateCard = "CandidateCard"
    TaskSpec = "TaskSpec"
    PatchSummary = "PatchSummary"
    TestReport = "TestReport"
    EvalResult = "EvalResult"
    FeedbackBrief = "FeedbackBrief"
    BudgetLedgerEntry = "BudgetLedgerEntry"
    PruneDecision = "PruneDecision"
    PromotionDecision = "PromotionDecision"
    FinalReport = "FinalReport"


class CandidateStatus(StrEnum):
    """Lifecycle states for candidate chess engines within an arm."""

    active = "active"
    pruned = "pruned"
    promoted = "promoted"
    final = "final"


class InterceptStatus(StrEnum):
    """Operator decision states for high-impact intercepts."""

    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    revision_requested = "revision_requested"


class ContextPacket(BaseModel):
    """Compact task context passed to fresh-context pods."""

    role: str
    objective: str
    experiment_arm: str
    candidate_id: str | None = None
    relevant_artifact_ids: list[str] = Field(default_factory=list)
    candidate_state_summary: str = ""
    constraints: list[str] = Field(default_factory=list)
    allowed_files: list[str] = Field(default_factory=list)
    max_token_budget: int = 0
    expected_output_artifact_type: ArtifactType | None = None


class ExperimentCreate(BaseModel):
    """Request body for creating an experiment."""

    name: str = "AutoResearch Chess Lab"
    mode: ExperimentMode = ExperimentMode.minimal_human
    token_budget_cap: int = 20_000
    move_budget_ms: int = 200


class Experiment(BaseModel):
    """Durable experiment metadata and current run status."""

    id: str = Field(default_factory=lambda: new_id("exp"))
    name: str
    mode: ExperimentMode
    status: ExperimentStatus = ExperimentStatus.draft
    token_budget_cap: int = 20_000
    move_budget_ms: int = 200
    current_arm: str = "autoresearch_minimal"
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    """A chess engine candidate tracked inside one experiment arm."""

    id: str = Field(default_factory=lambda: new_id("cand"))
    experiment_id: str
    arm: str
    name: str
    architecture: Literal["alphabeta", "mcts", "nnue_lite", "policy_guided"]
    module_path: str
    parent_candidate_id: str | None = None
    status: CandidateStatus = CandidateStatus.active
    tokens_spent: int = 0
    eval_summary: dict[str, Any] = Field(default_factory=dict)
    key_artifact_ids: list[str] = Field(default_factory=list)
    prune_reason: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class TaskCreate(BaseModel):
    """Request body for creating pod work."""

    experiment_id: str
    arm: str
    title: str
    role: str
    objective: str
    candidate_id: str | None = None
    pod_id: str | None = None
    context: ContextPacket
    max_token_budget: int = 1_000
    requires_review: bool = False


class Task(BaseModel):
    """Durable work item claimed and completed by a logical pod."""

    id: str = Field(default_factory=lambda: new_id("task"))
    experiment_id: str
    arm: str
    title: str
    role: str
    objective: str
    candidate_id: str | None = None
    pod_id: str | None = None
    claimed_by: str | None = None
    status: TaskStatus = TaskStatus.pending
    context: ContextPacket
    max_token_budget: int = 1_000
    result_artifact_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class ArtifactCreate(BaseModel):
    """Request body for storing a structured run artifact."""

    experiment_id: str
    arm: str
    artifact_type: ArtifactType
    title: str
    candidate_id: str | None = None
    task_id: str | None = None
    content: dict[str, Any]


class Artifact(BaseModel):
    """Stored output from planning, mutation, evaluation, or reporting."""

    id: str = Field(default_factory=lambda: new_id("art"))
    experiment_id: str
    arm: str
    artifact_type: ArtifactType
    title: str
    candidate_id: str | None = None
    task_id: str | None = None
    content: dict[str, Any]
    created_at: str = Field(default_factory=now_iso)


class BudgetLedgerEntry(BaseModel):
    """Token, cost, wall-clock, and outcome record for one unit of work."""

    id: str = Field(default_factory=lambda: new_id("bud"))
    experiment_id: str
    experiment_arm: str
    pod_id: str
    api_key_id: str
    agent_role: str
    task_id: str | None = None
    candidate_id: str | None = None
    estimated_tokens: int = 0
    actual_tokens: int | None = None
    token_source: Literal["estimated", "actual"] = "estimated"
    wall_clock_seconds: float = 0.0
    outcome: str = "completed"
    downstream_result: str | None = None
    created_at: str = Field(default_factory=now_iso)


class Event(BaseModel):
    """Timestamped audit event for dashboards and run reconstruction."""

    id: str = Field(default_factory=lambda: new_id("evt"))
    experiment_id: str | None = None
    event_type: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)


class InterceptCreate(BaseModel):
    """Request body for creating an operator approval gate."""

    experiment_id: str
    arm: str
    reason: str
    original_object: dict[str, Any]
    high_impact: bool = False


class Intercept(BaseModel):
    """Durable operator gate and its final resolution."""

    id: str = Field(default_factory=lambda: new_id("int"))
    experiment_id: str
    arm: str
    reason: str
    original_object: dict[str, Any]
    modified_object: dict[str, Any] | None = None
    high_impact: bool = False
    status: InterceptStatus = InterceptStatus.pending
    resolution_note: str | None = None
    created_at: str = Field(default_factory=now_iso)
    resolved_at: str | None = None


class CompleteTaskRequest(BaseModel):
    """Request body for completing a task with artifacts and budget data."""

    artifact_ids: list[str] = Field(default_factory=list)
    budget_entry: BudgetLedgerEntry | None = None


class FailTaskRequest(BaseModel):
    """Request body for failing a task with an optional budget entry."""

    error: str
    budget_entry: BudgetLedgerEntry | None = None


class InterceptResolveRequest(BaseModel):
    """Request body for approving, rejecting, or revising an intercept."""

    action: InterceptStatus
    modified_object: dict[str, Any] | None = None
    note: str | None = None
