import sys
from pathlib import Path

import pytest

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.datasets import load_json_dataset, resolve_dataset_path  # noqa: E402


def test_resolve_dataset_path_prefers_prefixed_id(tmp_path):
    dataset_id = "a" * 32
    dataset_path = tmp_path / f"{dataset_id}_data.json"
    dataset_path.write_text('[{"text": "hello"}]')

    resolved = resolve_dataset_path(tmp_path, dataset_id)
    assert resolved == dataset_path


def test_resolve_dataset_path_rejects_invalid_id(tmp_path):
    with pytest.raises(ValueError):
        resolve_dataset_path(tmp_path, "../etc/passwd")


def test_load_json_dataset_reads_jsonl(tmp_path):
    dataset_path = tmp_path / "sample.jsonl"
    dataset_path.write_text('{"text": "a"}\n{"text": "b"}\n')

    records = load_json_dataset(dataset_path)
    assert records == [{"text": "a"}, {"text": "b"}]


def test_load_json_dataset_rejects_empty_jsonl(tmp_path):
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.write_text("")

    with pytest.raises(ValueError):
        load_json_dataset(dataset_path)
