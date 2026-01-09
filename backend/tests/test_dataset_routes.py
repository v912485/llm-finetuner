import os
import sys
from pathlib import Path


def _client():
    os.environ["ADMIN_TOKEN"] = "testtoken"
    backend_dir = Path(__file__).resolve().parents[1]
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    import app  # noqa: F401
    return app.app.test_client()


def test_structure_requires_dataset_id():
    client = _client()
    res = client.post(
        "/api/datasets/structure",
        json={"file_path": "/etc/passwd"},
        headers={"Authorization": "Bearer testtoken"},
    )
    assert res.status_code == 400


def test_structure_rejects_invalid_dataset_id():
    client = _client()
    res = client.post(
        "/api/datasets/structure",
        json={"dataset_id": "../../etc/passwd"},
        headers={"Authorization": "Bearer testtoken"},
    )
    assert res.status_code == 400







