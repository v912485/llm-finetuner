import os
import sys
from pathlib import Path


def _import_app():
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    import app  # noqa: F401
    return app


def test_health_is_public():
    os.environ["ADMIN_TOKEN"] = "testtoken"
    app = _import_app()
    client = app.app.test_client()

    res = client.get("/api/health")
    assert res.status_code == 200


def test_api_requires_admin_token():
    os.environ["ADMIN_TOKEN"] = "testtoken"
    app = _import_app()
    client = app.app.test_client()

    res = client.get("/api/models/available")
    assert res.status_code == 401

    res2 = client.get("/api/models/available", headers={"Authorization": "Bearer testtoken"})
    assert res2.status_code == 200







