import sys
from pathlib import Path


backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.datasets import split_dataset_filename  # noqa: E402


def test_split_dataset_filename_does_not_split_legacy_names_with_underscores():
    dataset_id, display_name = split_dataset_filename("cvs_text_cleaned.json")
    assert dataset_id == "cvs_text_cleaned"
    assert display_name == "cvs text cleaned"


def test_split_dataset_filename_splits_new_upload_uuid_prefix():
    dataset_id, display_name = split_dataset_filename("0123456789abcdef0123456789abcdef_mydata.json")
    assert dataset_id == "0123456789abcdef0123456789abcdef"
    assert display_name == "mydata.json"


