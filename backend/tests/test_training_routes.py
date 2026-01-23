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


def test_save_model_requires_model_id_and_save_name(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch)
    monkeypatch.setattr(training_routes, "MODELS_DIR", tmp_path / "models")
    
    res = client.post("/api/training/save", json={})
    assert res.status_code == 400
    assert "required" in res.get_json()["message"].lower()


def test_save_model_accepts_convert_to_gguf_parameter(monkeypatch, tmp_path):
    import shutil
    
    client, _ = _client(monkeypatch)
    
    models_dir = tmp_path / "models"
    saved_dir = tmp_path / "saved"
    models_dir.mkdir()
    saved_dir.mkdir()
    
    model_dir = models_dir / "test_model" / "finetuned"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"test": "data"}')
    
    monkeypatch.setattr(training_routes, "MODELS_DIR", models_dir)
    monkeypatch.setattr(training_routes, "SAVED_MODELS_DIR", saved_dir)
    
    def mock_convert(model_dir, output_path, base_model_id, skip_conversion=False):
        if not skip_conversion:
            output_path.write_text("gguf_data")
    
    monkeypatch.setattr(training_routes, "convert_model_to_gguf", mock_convert)
    
    res = client.post("/api/training/save", json={
        "model_id": "test_model",
        "save_name": "test-save",
        "convert_to_gguf": False
    })
    
    assert res.status_code == 200
    assert res.get_json()["status"] == "success"
    
    saved_model_dir = saved_dir / "test-save"
    assert saved_model_dir.exists()
    assert not (saved_model_dir / "model.gguf").exists()


def test_save_model_converts_to_gguf_by_default(monkeypatch, tmp_path):
    import shutil
    
    client, _ = _client(monkeypatch)
    
    models_dir = tmp_path / "models"
    saved_dir = tmp_path / "saved"
    models_dir.mkdir()
    saved_dir.mkdir()
    
    model_dir = models_dir / "test_model" / "finetuned"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"test": "data"}')
    
    monkeypatch.setattr(training_routes, "MODELS_DIR", models_dir)
    monkeypatch.setattr(training_routes, "SAVED_MODELS_DIR", saved_dir)
    
    def mock_convert(model_dir, output_path, base_model_id, skip_conversion=False):
        if not skip_conversion:
            output_path.write_text("gguf_data")
    
    monkeypatch.setattr(training_routes, "convert_model_to_gguf", mock_convert)
    
    res = client.post("/api/training/save", json={
        "model_id": "test_model",
        "save_name": "test-save-default"
    })
    
    assert res.status_code == 200
    assert res.get_json()["status"] == "success"
    
    saved_model_dir = saved_dir / "test-save-default"
    assert saved_model_dir.exists()
    assert (saved_model_dir / "model.gguf").exists()
    assert (saved_model_dir / "model.gguf").read_text() == "gguf_data"


def test_save_model_returns_conflict_when_model_exists(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch)
    
    models_dir = tmp_path / "models"
    saved_dir = tmp_path / "saved"
    models_dir.mkdir()
    saved_dir.mkdir()
    
    model_dir = models_dir / "test_model" / "finetuned"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"test": "data"}')
    
    existing_model = saved_dir / "existing-model"
    existing_model.mkdir()
    (existing_model / "existing.txt").write_text("existing")
    
    monkeypatch.setattr(training_routes, "MODELS_DIR", models_dir)
    monkeypatch.setattr(training_routes, "SAVED_MODELS_DIR", saved_dir)
    
    def mock_convert(model_dir, output_path, base_model_id, skip_conversion=False):
        pass
    
    monkeypatch.setattr(training_routes, "convert_model_to_gguf", mock_convert)
    
    res = client.post("/api/training/save", json={
        "model_id": "test_model",
        "save_name": "existing-model"
    })
    
    assert res.status_code == 409
    data = res.get_json()
    assert data["status"] == "error"
    assert "already exists" in data["message"]
    assert data["requires_confirmation"] is True


def test_save_model_overwrites_when_confirmed(monkeypatch, tmp_path):
    client, _ = _client(monkeypatch)
    
    models_dir = tmp_path / "models"
    saved_dir = tmp_path / "saved"
    models_dir.mkdir()
    saved_dir.mkdir()
    
    model_dir = models_dir / "test_model" / "finetuned"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"test": "new"}')
    
    existing_model = saved_dir / "existing-model"
    existing_model.mkdir()
    (existing_model / "old.txt").write_text("old data")
    
    monkeypatch.setattr(training_routes, "MODELS_DIR", models_dir)
    monkeypatch.setattr(training_routes, "SAVED_MODELS_DIR", saved_dir)
    
    def mock_convert(model_dir, output_path, base_model_id, skip_conversion=False):
        if not skip_conversion:
            output_path.write_text("new gguf")
    
    monkeypatch.setattr(training_routes, "convert_model_to_gguf", mock_convert)
    
    res = client.post("/api/training/save", json={
        "model_id": "test_model",
        "save_name": "existing-model",
        "overwrite": True
    })
    
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    
    saved_model_dir = saved_dir / "existing-model"
    assert saved_model_dir.exists()
    assert not (saved_model_dir / "old.txt").exists()
    assert (saved_model_dir / "config.json").exists()
    assert (saved_model_dir / "config.json").read_text() == '{"test": "new"}'
