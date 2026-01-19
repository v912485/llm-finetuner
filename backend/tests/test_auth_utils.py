import sys
from pathlib import Path

from flask import Flask, jsonify

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.auth import is_admin_configured, is_request_authenticated, require_admin  # noqa: E402


def test_is_admin_configured_tracks_admin_token(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert is_admin_configured() is False

    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    assert is_admin_configured() is True


def test_is_request_authenticated_matches_bearer_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    app = Flask(__name__)

    with app.test_request_context(headers={"Authorization": "Bearer secret"}):
        assert is_request_authenticated() is True

    with app.test_request_context(headers={"Authorization": "Bearer wrong"}):
        assert is_request_authenticated() is False


def test_require_admin_enforces_configured_token(monkeypatch):
    app = Flask(__name__)
    called = {"value": False}

    @app.route("/protected")
    @require_admin
    def protected():
        called["value"] = True
        return jsonify({"ok": True})

    client = app.test_client()

    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    res = client.get("/protected")
    assert res.status_code == 500

    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    res = client.get("/protected")
    assert res.status_code == 401

    res = client.get("/protected", headers={"Authorization": "Bearer secret"})
    assert res.status_code == 200
    assert called["value"] is True
