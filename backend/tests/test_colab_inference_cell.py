"""Static behavior checks for the single-cell Colab inference helper."""

import ast
from pathlib import Path
import re
import unicodedata


REPO_ROOT = Path(__file__).resolve().parents[2]
CELL_PATH = REPO_ROOT / "deliverables" / "colab_mdeberta_4class_inference_cell.py"


def _clock_namespace():
    tree = ast.parse(CELL_PATH.read_text(encoding="utf-8"), filename=str(CELL_PATH))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if set(names) & {"_CLOCK_PREFIX", "_CLOCK_SUFFIX", "_CLOCK_BODIES", "_CLOCK_PATTERN"}:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"normalize_text", "matches_clock"}:
            selected.append(node)
    namespace = {"re": re, "unicodedata": unicodedata}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CELL_PATH), "exec"), namespace)
    return namespace


def test_inference_cell_keeps_clock_rule_narrow():
    namespace = _clock_namespace()
    normalize = namespace["normalize_text"]
    matches_clock = namespace["matches_clock"]

    assert matches_clock(normalize("Şu anda saat kaç?"))
    assert matches_clock(normalize("Saati söyler misin?"))
    assert not matches_clock(normalize("Altmış kilometreyi kaç saatte giderim?"))
    assert not matches_clock(normalize("Nöbet saati kaçta başlıyor?"))

