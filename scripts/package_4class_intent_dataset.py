#!/usr/bin/env python3
"""Build an auditable four-class intent fine-tuning package."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata


LABELS = ("medikal", "savaş yönergeleri", "matematik", "sohbet")
MILITARY_LABELS = {
    "telsiz ve raporlama",
    "nöbet ve emniyet",
    "harita ve intikal",
    "mevzi ve gizlenme",
    "kbrn korunma",
    "angajman ve esir hukuku",
}
LABEL_MAP = {
    "ilk yardım": "medikal",
    "medikal": "medikal",
    **{label: "savaş yönergeleri" for label in MILITARY_LABELS},
    "savaş yönergeleri": "savaş yönergeleri",
    "matematik": "matematik",
    "sohbet": "sohbet",
    "saat": None,
}
SPLITS = ("train", "validation", "test")
RATIOS = (0.70, 0.15, 0.15)
AUDIT_FIELDS = (
    "id",
    "soru",
    "sinif",
    "alt_sinif",
    "split",
    "source",
    "original_sinif",
    "karar",
    "gerekce",
    "family_id",
    "recipe_id",
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: geçersiz JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: JSON nesnesi bekleniyor")
        rows.append(row)
    return rows


def read_audit(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def _global_allocation(label_counts: dict[str, int]) -> dict[str, dict[str, int]]:
    total = sum(label_counts.values())
    raw_targets = {split: total * ratio for split, ratio in zip(SPLITS, RATIOS, strict=True)}
    split_targets = {split: int(raw_targets[split]) for split in SPLITS}
    for split in sorted(
        SPLITS,
        key=lambda name: (-(raw_targets[name] - split_targets[name]), SPLITS.index(name)),
    )[: total - sum(split_targets.values())]:
        split_targets[split] += 1

    allocation = {
        label: {split: int(count * ratio) for split, ratio in zip(SPLITS, RATIOS, strict=True)}
        for label, count in label_counts.items()
    }
    label_deficits = {
        label: label_counts[label] - sum(allocation[label].values()) for label in label_counts
    }
    split_deficits = {
        split: split_targets[split] - sum(allocation[label][split] for label in label_counts)
        for split in SPLITS
    }
    candidates = sorted(
        (
            (label_counts[label] * RATIOS[SPLITS.index(split)] - allocation[label][split], label, split)
            for label in label_counts
            for split in SPLITS
        ),
        key=lambda item: (-item[0], LABELS.index(item[1]), SPLITS.index(item[2])),
    )
    used_cells = set()
    while any(label_deficits.values()):
        choice = next(
            (
                (label, split)
                for _, label, split in candidates
                if label_deficits[label] > 0
                and split_deficits[split] > 0
                and (label, split) not in used_cells
            ),
            None,
        )
        if choice is None:
            raise RuntimeError("Stratified split kotaları yuvarlanamadı")
        label, split = choice
        allocation[label][split] += 1
        label_deficits[label] -= 1
        split_deficits[split] -= 1
        used_cells.add(choice)
    return allocation


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _near_duplicate_groups(rows: list[dict[str, str]], threshold: float = 0.80) -> list[list[dict[str, str]]]:
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _token_jaccard(rows[left]["soru"], rows[right]["soru"]) >= threshold:
                union(left, right)

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    return list(grouped.values())


def _split_rows(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["sinif"]].append(row)

    allocations = _global_allocation({label: len(by_label[label]) for label in LABELS})
    split_rows = {split: [] for split in SPLITS}
    for label in LABELS:
        groups = _near_duplicate_groups(by_label[label])
        groups.sort(
            key=lambda group: (
                -len(group),
                hashlib.sha256(
                    f"{seed}\0{label}\0{normalize_text(group[0]['soru'])}".encode("utf-8")
                ).hexdigest(),
            )
        )
        allocation = allocations[label]
        remaining = dict(allocation)
        for group in groups:
            group_size = len(group)
            split = min(
                SPLITS,
                key=lambda name: (
                    max(group_size - remaining[name], 0),
                    -(remaining[name] / allocation[name] if allocation[name] else 0),
                    hashlib.sha256(
                        f"{seed}\0{label}\0{normalize_text(group[0]['soru'])}\0{name}".encode("utf-8")
                    ).hexdigest(),
                ),
            )
            for row in group:
                row["split"] = split
                split_rows[split].append(row)
            remaining[split] -= group_size

    for split in SPLITS:
        split_rows[split].sort(key=lambda row: (LABELS.index(row["sinif"]), row["id"]))
    return split_rows


def _write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_manifest(output_dir: Path) -> None:
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(f"{digest}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _write_docs(output_dir: Path, summary: dict) -> None:
    counts = summary["label_counts"]
    split_counts = summary["per_label_split_counts"]
    table = ["| Sınıf | Toplam | Train | Validation | Test |", "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        table.append(
            f"| {label} | {counts[label]} | {split_counts[label]['train']} | "
            f"{split_counts[label]['validation']} | {split_counts[label]['test']} |"
        )
    table.append(
        f"| **Toplam** | **{summary['selected_rows']}** | **{summary['split_counts']['train']}** | "
        f"**{summary['split_counts']['validation']}** | **{summary['split_counts']['test']}** |"
    )
    report = f"""# Odine dört sınıflı intent fine-tune veri seti

## Kısa özet

Bu paket, önceki 700 soruluk paketin dört sınıfa eşlenen kayıtları ile kullanıcı tarafından sağlanan 400 yeni sorunun birleşimidir. `saat` sınıfındaki {summary['excluded_clock_rows']} kayıt, repoda deterministik kural tarafından işlendiği için sınıflandırıcı eğitiminden çıkarıldı.

Toplam {summary['selected_rows']} benzersiz soru, seed {summary['split_seed']} ile sınıf bazında deterministik olarak yeniden train/validation/test splitlerine ayrıldı.

## Etiket eşlemesi

- `ilk yardım` → `medikal`
- Telsiz, nöbet, harita, mevzi, KBRN ve angajman alt sınıfları → `savaş yönergeleri`
- `matematik` → `matematik`
- `sohbet` → `sohbet`
- `saat` → eğitim dışında, kural tabanlı yol

## Dağılım

{chr(10).join(table)}

## Kaynak ve kalite notları

- Eski paketten {summary['source_counts']['existing_package']} kayıt, yeni JSONL dosyasından {summary['source_counts']['provided_jsonl']} kayıt alındı.
- Yeni 400 kayıtta parse hatası, boş soru, normalize tekrar veya eski paketle normalize-birebir çakışma bulunmadı.
- Eski audit kararları korundu: {summary['decision_counts'].get('keep', 0)} `keep`, {summary['decision_counts'].get('revise', 0)} `revise`. Yeni kayıtlar bağımsız insan/LLM incelemesinden geçirilmedi ve `provided` olarak işaretlendi.
- Public CSV dosyaları yalnızca `soru,sinif` kolonlarını içerir. Alt sınıf, kaynak ve karar bilgileri `audit/review_audit.csv` içindedir.
- Splitler yeniden üretildi; eski test splitinin üyeliği korunmadı.
- Test seti eğitim veya model seçimi için kullanılmamalıdır.
"""
    (output_dir / "RAPOR_TR.md").write_text(report, encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Odine dört sınıflı intent fine-tune paketi\n\n"
        "Colab eğitimi için `dataset/train.csv`, `dataset/validation.csv` ve "
        "`dataset/test.csv` dosyalarını kullanın. Ayrıntılar `RAPOR_TR.md`, kayıt bazındaki "
        "iz `audit/review_audit.csv` içindedir.\n",
        encoding="utf-8",
    )


def build_package(
    *,
    existing_csv: Path,
    added_jsonl: Path,
    output_dir: Path,
    seed: int = 17,
    created_on: str,
    existing_audit: Path | None = None,
) -> dict:
    existing_rows = read_csv(existing_csv)
    added_rows = read_jsonl(added_jsonl)
    audit_by_question = {}
    if existing_audit and existing_audit.is_file():
        audit_by_question = {normalize_text(row["soru"]): row for row in read_audit(existing_audit)}

    combined = []
    excluded_clock_rows = 0
    unknown_labels = set()

    for index, source_row in enumerate(existing_rows, 1):
        question = str(source_row.get("soru", "")).strip()
        original_label = normalize_text(source_row.get("sinif", ""))
        if original_label not in LABEL_MAP:
            unknown_labels.add(original_label)
            continue
        mapped_label = LABEL_MAP[original_label]
        if mapped_label is None:
            excluded_clock_rows += 1
            continue
        prior = audit_by_question.get(normalize_text(question), {})
        combined.append({
            "id": prior.get("id") or f"existing-{index:04d}",
            "soru": question,
            "sinif": mapped_label,
            "alt_sinif": original_label,
            "source": "existing_package",
            "original_sinif": original_label,
            "karar": prior.get("karar", "existing"),
            "gerekce": prior.get("gerekce", ""),
            "family_id": prior.get("family_id", ""),
            "recipe_id": prior.get("recipe_id", ""),
        })

    for index, source_row in enumerate(added_rows, 1):
        question = str(source_row.get("soru", "")).strip()
        original_label = normalize_text(source_row.get("sinif", ""))
        if original_label not in LABEL_MAP or LABEL_MAP[original_label] is None:
            unknown_labels.add(original_label)
            continue
        combined.append({
            "id": f"provided-{index:04d}",
            "soru": question,
            "sinif": LABEL_MAP[original_label],
            "alt_sinif": str(source_row.get("alt_sinif", "")).strip(),
            "source": "provided_jsonl",
            "original_sinif": original_label,
            "karar": "provided",
            "gerekce": "not_reviewed",
            "family_id": "",
            "recipe_id": "",
        })

    if unknown_labels:
        raise ValueError(f"Bilinmeyen veya eğitim dışı yeni etiketler: {sorted(unknown_labels)}")
    if any(not normalize_text(row["soru"]) for row in combined):
        raise ValueError("Boş soru bulundu")
    if any("\n" in row["soru"] or "\r" in row["soru"] for row in combined):
        raise ValueError("Çok satırlı soru bulundu")

    by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in combined:
        by_question[normalize_text(row["soru"])].append(row)
    duplicates = {key: rows for key, rows in by_question.items() if len(rows) > 1}
    if duplicates:
        raise ValueError(f"Normalize tekrar bulundu: {len(duplicates)}")

    label_counts = Counter(row["sinif"] for row in combined)
    missing_labels = set(LABELS) - set(label_counts)
    if missing_labels:
        raise ValueError(f"Eksik sınıflar: {sorted(missing_labels)}")

    split_rows = _split_rows(combined, seed)
    all_rows = [row for split in SPLITS for row in split_rows[split]]

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "dataset").mkdir(parents=True)
    (output_dir / "audit").mkdir(parents=True)
    (output_dir / "source").mkdir(parents=True)

    for split in SPLITS:
        _write_csv(output_dir / "dataset" / f"{split}.csv", split_rows[split], ("soru", "sinif"))
    _write_csv(output_dir / "dataset" / "all.csv", all_rows, ("soru", "sinif"))
    _write_csv(output_dir / "audit" / "review_audit.csv", all_rows, AUDIT_FIELDS)
    shutil.copy2(added_jsonl, output_dir / "source" / "provided_questions.jsonl")

    per_label_split_counts = {
        label: {split: sum(row["sinif"] == label for row in split_rows[split]) for split in SPLITS}
        for label in LABELS
    }
    summary = {
        "package": output_dir.name,
        "created_on": created_on,
        "source_files": [str(existing_csv), str(added_jsonl)],
        "selected_rows": len(all_rows),
        "excluded_clock_rows": excluded_clock_rows,
        "label_counts": dict(label_counts),
        "source_counts": dict(Counter(row["source"] for row in all_rows)),
        "decision_counts": dict(Counter(row["karar"] for row in all_rows)),
        "split_seed": seed,
        "split_method": "label-stratified deterministic near-duplicate-group split with globally balanced largest-remainder targets",
        "target_ratios": dict(zip(SPLITS, RATIOS, strict=True)),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "per_label_split_counts": per_label_split_counts,
        "quality_checks": {
            "unique_normalized_questions": len(by_question),
            "empty_questions": 0,
            "multiline_questions": 0,
            "unknown_labels": [],
            "cross_split_normalized_overlap": 0,
        },
        "notes": [
            "Saat records are excluded because runtime handles current-clock queries with a deterministic rule.",
            "Existing keep/revise audit decisions are preserved; provided JSONL rows are marked provided/not_reviewed.",
            "All splits were regenerated; prior test membership was not preserved.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_docs(output_dir, summary)
    _sha256_manifest(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-csv", type=Path, required=True)
    parser.add_argument("--existing-audit", type=Path)
    parser.add_argument("--added-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-on", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    summary = build_package(
        existing_csv=args.existing_csv,
        existing_audit=args.existing_audit,
        added_jsonl=args.added_jsonl,
        output_dir=args.output_dir,
        seed=args.seed,
        created_on=args.created_on,
    )
    archive = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"ZIP: {archive}")


if __name__ == "__main__":
    main()
