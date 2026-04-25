from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agents.orchestrator import HumanRejectedStep, Orchestrator
from backend.app import create_app
from backend.models import ExperimentMode


def test_human_gate_auto_approves_when_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)
    exp = client.post("/experiments", json={"name": "hitl", "mode": "minimal_human"}).json()

    orch = Orchestrator(backend_url="http://testserver", mock_mode=True, human_in_loop=False)
    orch._post = lambda path, payload=None: client.post(path, json=payload or {}).json()

    resolved = orch._human_gate(exp["id"], "autoresearch_minimal", "Approve thing", {"step": "x"}, high_impact=True)

    assert resolved["status"] == "approved"
    assert resolved["resolution_note"] == "auto-approved; HITL disabled"
    intercepts = client.get(f"/experiments/{exp['id']}/intercepts").json()
    assert len(intercepts) == 1
    assert intercepts[0]["high_impact"] is True


def test_human_gate_reject_raises_and_records_rejection(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app()
    client = TestClient(app)
    exp = client.post("/experiments", json={"name": "hitl", "mode": ExperimentMode.minimal_human}).json()

    answers = iter(["r", "nope"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    orch = Orchestrator(backend_url="http://testserver", mock_mode=True, human_in_loop=True)
    orch._post = lambda path, payload=None: client.post(path, json=payload or {}).json()

    with pytest.raises(HumanRejectedStep):
        orch._human_gate(exp["id"], "autoresearch_minimal", "Reject thing", {"step": "x"})

    intercepts = client.get(f"/experiments/{exp['id']}/intercepts").json()
    assert intercepts[0]["status"] == "rejected"
    assert intercepts[0]["resolution_note"] == "nope"
