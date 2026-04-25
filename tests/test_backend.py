from __future__ import annotations

from backend.app import create_app
from backend.models import ArtifactType, ContextPacket
from fastapi.testclient import TestClient


def test_backend_task_artifact_budget_lifecycle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)

    exp = client.post("/experiments", json={"name": "test", "mode": "minimal_human", "token_budget_cap": 1000}).json()
    context = ContextPacket(role="development", objective="do work", experiment_arm="arm", max_token_budget=100).model_dump(mode="json")
    task = client.post("/tasks", json={
        "experiment_id": exp["id"],
        "arm": "arm",
        "title": "task",
        "role": "development",
        "objective": "do work",
        "context": context,
        "max_token_budget": 100,
    }).json()

    claimed = client.post(f"/experiments/{exp['id']}/pods/child-a/next-task").json()
    assert claimed["id"] == task["id"]
    artifact = client.post("/artifacts", json={
        "experiment_id": exp["id"],
        "arm": "arm",
        "artifact_type": ArtifactType.TestReport,
        "title": "report",
        "task_id": task["id"],
        "content": {"ok": True},
    }).json()
    budget = {
        "experiment_id": exp["id"],
        "experiment_arm": "arm",
        "pod_id": "child-a",
        "api_key_id": "key-2",
        "agent_role": "development",
        "task_id": task["id"],
        "estimated_tokens": 123,
        "wall_clock_seconds": 0.1,
    }
    completed = client.post(f"/experiments/{exp['id']}/tasks/{task['id']}/complete", json={"artifact_ids": [artifact["id"]], "budget_entry": budget}).json()
    assert completed["status"] == "completed"
    assert client.get(f"/experiments/{exp['id']}/budget").json()[0]["estimated_tokens"] == 123

