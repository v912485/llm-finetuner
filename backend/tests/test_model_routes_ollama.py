import sys
from pathlib import Path

from flask import Flask

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from routes import model_routes  # noqa: E402


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.ok = 200 <= status_code < 300

    def json(self):
        return self._json_data


def _client(monkeypatch, tmp_path):
    app = Flask(__name__)
    saved_dir = tmp_path / "saved_models"
    saved_dir.mkdir()
    monkeypatch.setattr(model_routes, "SAVED_MODELS_DIR", saved_dir)
    monkeypatch.setattr(model_routes, "_load_app_config", lambda: {"ollama": {"host": "http://localhost", "port": 11434}})
    app.register_blueprint(model_routes.bp)
    return app.test_client(), saved_dir


def test_deploy_to_ollama_requires_gguf(monkeypatch, tmp_path):
    client, saved_dir = _client(monkeypatch, tmp_path)
    model_dir = saved_dir / "model-one"
    model_dir.mkdir()

    res = client.post("/api/models/ollama/deploy", json={"saved_model_name": "model-one"})
    assert res.status_code == 404
    assert res.get_json()["message"] == "No GGUF file found in saved model"


def test_deploy_to_ollama_rejects_multiple_gguf(monkeypatch, tmp_path):
    client, saved_dir = _client(monkeypatch, tmp_path)
    model_dir = saved_dir / "model-two"
    model_dir.mkdir()
    (model_dir / "a.gguf").write_text("x")
    (model_dir / "b.gguf").write_text("y")

    res = client.post("/api/models/ollama/deploy", json={"saved_model_name": "model-two"})
    assert res.status_code == 409
    assert res.get_json()["message"].startswith("Multiple GGUF files")


def test_deploy_to_ollama_success(monkeypatch, tmp_path):
    client, saved_dir = _client(monkeypatch, tmp_path)
    model_dir = saved_dir / "model-three"
    model_dir.mkdir()
    gguf_path = model_dir / "model.gguf"
    gguf_path.write_text("gguf")

    captured = {}

    def fake_head(url, timeout=None):
        captured["head_url"] = url
        return DummyResponse(status_code=404, json_data={})

    def fake_post(url, json=None, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        if data is not None:
            return DummyResponse(status_code=201, json_data={"status": "uploaded"})
        return DummyResponse(status_code=200, json_data={"status": "success"})

    monkeypatch.setattr(model_routes.requests, "head", fake_head)
    monkeypatch.setattr(model_routes.requests, "post", fake_post)

    res = client.post("/api/models/ollama/deploy", json={"saved_model_name": "model-three"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert captured["url"].endswith("/api/create")
    assert gguf_path.name in captured["json"]["files"]
