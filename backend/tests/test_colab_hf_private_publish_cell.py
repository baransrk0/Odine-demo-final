"""Static safety checks for the private Hugging Face publishing cell."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_PATH = REPO_ROOT / "deliverables" / "colab_publish_hf_private_cell.py"


def test_publish_cell_keeps_repo_private_and_avoids_row_level_data():
    source = CELL_PATH.read_text(encoding="utf-8")
    ast.parse(source, filename=str(CELL_PATH))

    assert 'userdata.get("HF_TOKEN")' in source
    assert "private=True" in source
    assert 'assert model_info.private is True' in source
    assert "test_predictions.csv" in source
    assert "dataset/" in source
    assert "upload_large_folder(" in source
    assert 'print("[1/7]' in source
    assert "flush=True" in source
    assert '"pip",' not in source
