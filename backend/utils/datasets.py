from __future__ import annotations

import json
from pathlib import Path


def _is_safe_id(value: str) -> bool:
    if not value:
        return False
    for c in value:
        if not (c.isalnum() or c in ("-", "_")):
            return False
    return True


def resolve_dataset_path(datasets_dir: Path, dataset_id: str) -> Path:
    """
    Resolve a dataset file from an opaque dataset_id without allowing path traversal.

    Supports:
    - New uploads stored as: <dataset_id>_<original_filename>
    - Legacy datasets stored as: <dataset_id>.<ext> (where dataset_id is the stem)
    """
    if not _is_safe_id(dataset_id):
        raise ValueError("Invalid dataset_id")

    prefixed = list(datasets_dir.glob(f"{dataset_id}_*"))
    if len(prefixed) == 1 and prefixed[0].is_file():
        return prefixed[0]

    legacy = list(datasets_dir.glob(f"{dataset_id}.*"))
    if len(legacy) == 1 and legacy[0].is_file():
        return legacy[0]

    raise FileNotFoundError("Dataset not found")


def split_dataset_filename(filename: str) -> tuple[str, str]:
    """
    Return (dataset_id, display_name) for a stored dataset filename.

    - For <dataset_id>_<original_name>: returns (dataset_id, original_name)
    - For legacy: returns (stem, filename)
    """
    if "_" in filename:
        dataset_id, rest = filename.split("_", 1)
        if _is_safe_id(dataset_id) and rest:
            return dataset_id, rest

    stem = Path(filename).stem
    return stem, filename


def _load_jsonl(path: Path) -> list:
    records: list = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e
    if not records:
        raise ValueError("No valid JSON entries found in file")
    return records


def load_json_dataset(path: Path) -> list:
    """
    Load a JSON dataset as a list of records.

    Supports:
    - .json: a JSON array, or a single JSON object (returned as a single-item list)
    - .jsonl: JSON Lines (one JSON object per line)
    - Mislabelled JSONL stored with .json extension (auto-detected on JSONDecodeError)
    """
    ext = path.suffix.lower()

    if ext == ".jsonl":
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                first = f.read(4096).lstrip()
            if first.startswith("["):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    obj = json.load(f)
                return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            pass
        return _load_jsonl(path)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        return _load_jsonl(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("data", "items", "records", "examples"):
            value = obj.get(key)
            if isinstance(value, list):
                return value
        return [obj]

    return [obj]


