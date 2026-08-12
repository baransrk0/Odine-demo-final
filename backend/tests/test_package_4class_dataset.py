"""Tests for rebuilding the four-class intent fine-tune package."""

from collections import Counter
import csv
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "package_4class_intent_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("package_4class_intent_dataset", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_package_maps_labels_excludes_clock_and_resplits(tmp_path):
    module = _load_module()
    old_csv = tmp_path / "old.csv"
    new_jsonl = tmp_path / "new.jsonl"
    out_dir = tmp_path / "package"

    rows = []
    for label in (
        "ilk yardım",
        "telsiz ve raporlama",
        "matematik",
        "sohbet",
        "saat",
    ):
        rows.extend({"soru": f"{label} eski soru {index}?", "sinif": label} for index in range(20))
    rows[0]["soru"] = "Arazide kolu kanayan yaralıya temiz bezle ilk ne yapayım?"
    rows[1]["soru"] = "Komutanım arazide kolu kanayan yaralıya temiz bezle ilk ne yapayım?"
    with old_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["soru", "sinif"])
        writer.writeheader()
        writer.writerows(rows)

    new_jsonl.write_text(
        "".join(
            f'{{"soru":"{label} yeni soru {index}?","sinif":"{label}","alt_sinif":"ornek"}}\n'
            for label in module.LABELS
            for index in range(10)
        ),
        encoding="utf-8",
    )

    summary = module.build_package(
        existing_csv=old_csv,
        added_jsonl=new_jsonl,
        output_dir=out_dir,
        seed=0,
        created_on="2026-08-12",
    )

    assert summary["selected_rows"] == 120
    assert summary["excluded_clock_rows"] == 20
    assert summary["split_counts"] == {"train": 84, "validation": 18, "test": 18}
    assert summary["label_counts"] == {
        "medikal": 30,
        "savaş yönergeleri": 30,
        "matematik": 30,
        "sohbet": 30,
    }
    assert Counter(row["source"] for row in module.read_audit(out_dir / "audit" / "review_audit.csv")) == {
        "existing_package": 80,
        "provided_jsonl": 40,
    }
    audit_rows = module.read_audit(out_dir / "audit" / "review_audit.csv")
    near_splits = {
        row["split"] for row in audit_rows if "arazide kolu kanayan yaralıya" in row["soru"].casefold()
    }
    assert len(near_splits) == 1
    assert (out_dir / "SHA256SUMS").is_file()
