import sys
from pathlib import Path

from flask import Flask

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from routes import training_routes  # noqa: E402


class DummyTrainer:
    def __init__(self):
        self.started_with = None

    def start_training(self, data):
        self.started_with = data
        return "run-123"

    def get_status(self):
        return {"queue_position": 1, "queue_length": 1}

    def cancel_training(self):
        return None


def _client(monkeypatch):
    app = Flask(__name__)
    dummy = DummyTrainer()
    monkeypatch.setattr(training_routes, "trainer", dummy)
    app.register_blueprint(training_routes.bp)
    return app.test_client(), dummy


def test_start_training_requires_model_and_datasets(monkeypatch):
    client, _ = _client(monkeypatch)
    res = client.post("/api/training/start", json={})
    assert res.status_code == 400


def test_start_training_returns_queue_details(monkeypatch):
    client, dummy = _client(monkeypatch)
    payload = {"model_id": "test-model", "datasets": ["dataset-1"]}
    res = client.post("/api/training/start", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["run_id"] == "run-123"
    assert data["queue_position"] == 1
    assert data["queue_length"] == 1
    assert dummy.started_with == payload


def test_training_status_returns_success(monkeypatch):
    client, _ = _client(monkeypatch)
    res = client.get("/api/training/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["queue_length"] == 1
