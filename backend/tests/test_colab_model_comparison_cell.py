"""Static checks for the Colab base-versus-fine-tuned comparison cell."""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_PATH = REPO_ROOT / "deliverables" / "colab_mdeberta_base_vs_finetuned_cell.py"


def test_comparison_cell_uses_same_four_labels_and_turkish_hypotheses():
    tree = ast.parse(CELL_PATH.read_text(encoding="utf-8"), filename=str(CELL_PATH))
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"LABELS", "VERBALIZATIONS", "HYPOTHESIS_TEMPLATE"}:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)

    assert assignments["LABELS"] == ["medikal", "savaş yönergeleri", "matematik", "sohbet"]
    assert set(assignments["VERBALIZATIONS"]) == set(assignments["LABELS"])
    assert assignments["HYPOTHESIS_TEMPLATE"] == "Bu metnin konusu {}."

