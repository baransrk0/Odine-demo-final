# Colab A100: bu dosyanin tamamini tek hucreye yapistirip calistirin.
import os
import sys
import json
import math
import random
import shutil
import hashlib
import zipfile
import subprocess
import unicodedata
import re
from pathlib import Path

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "transformers==4.57.6",
    "datasets==4.3.0",
    "accelerate==1.11.0",
    "sentencepiece==0.2.1",
    "scikit-learn==1.7.2",
])

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from google.colab import files
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------- Ayarlar ---------------------------------------
MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SEED = 17
MAX_LENGTH = 96
EPOCHS = 8
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
OUTPUT_ROOT = Path("/content/odine_mdeberta_4class")
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
ARTIFACT_DIR = OUTPUT_ROOT / "artifact"
INPUT_DIR = OUTPUT_ROOT / "input"

LABELS = ["medikal", "savaş yönergeleri", "matematik", "sohbet"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

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
    # Saat sınıflandırıcıya verilmez; repoda deterministik kural ile çözülür.
    "saat": None,
}


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def safe_extract(archive_path, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Guvenli olmayan ZIP yolu: {member.filename}")
        archive.extractall(destination)


def read_table(path):
    path = Path(path)
    if path.suffix.casefold() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path, encoding="utf-8-sig")


def canonicalize_frame(frame, source_name, selected_questions=None):
    required = {"soru", "sinif"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_name}: eksik kolonlar: {sorted(missing)}")

    clean = frame.copy()
    clean["soru"] = clean["soru"].astype(str).str.strip()
    clean["sinif"] = clean["sinif"].astype(str).map(normalize_text)
    clean["question_key"] = clean["soru"].map(normalize_text)

    if clean["question_key"].eq("").any():
        raise ValueError(f"{source_name}: bos soru bulundu")

    unknown = sorted(set(clean["sinif"]) - set(LABEL_MAP))
    if unknown:
        raise ValueError(f"{source_name}: bilinmeyen siniflar: {unknown}")

    clean["sinif"] = clean["sinif"].map(LABEL_MAP)
    clock_count = int(clean["sinif"].isna().sum())
    clean = clean.dropna(subset=["sinif"]).copy()
    if clock_count:
        print(f"{source_name}: {clock_count} saat kaydi kural tabanli yol icin egitimden cikarildi.")

    if selected_questions is not None:
        before = len(clean)
        clean = clean[clean["question_key"].isin(selected_questions)].copy()
        print(f"{source_name}: audit filtresiyle {before - len(clean)} secilmemis kayit cikarildi.")

    conflicting = clean.groupby("question_key")["sinif"].nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        examples = clean[clean["question_key"].isin(conflicting.index)][["soru", "sinif"]].head(10)
        raise ValueError("Ayni soru birden fazla sinifa atanmis:\n" + examples.to_string(index=False))

    duplicate_count = int(clean.duplicated("question_key").sum())
    if duplicate_count:
        print(f"{source_name}: {duplicate_count} normalize-birebir tekrar kaldirildi.")
        clean = clean.drop_duplicates("question_key", keep="first")

    clean["label"] = clean["sinif"].map(LABEL2ID).astype(int)
    return clean[["soru", "sinif", "label", "question_key"]].reset_index(drop=True)


def find_named_file(paths, filename):
    matches = [path for path in paths if path.name.casefold() == filename.casefold()]
    if len(matches) > 1:
        print(f"Uyari: birden fazla {filename} bulundu; kullanilan: {matches[0]}")
    return matches[0] if matches else None


def validate_split(frame, split_name):
    if frame.empty:
        raise ValueError(f"{split_name} split'i bos")
    missing_labels = sorted(set(LABELS) - set(frame["sinif"]))
    if missing_labels:
        raise ValueError(f"{split_name} split'inde eksik siniflar: {missing_labels}")


def print_distribution(split_frames):
    table = pd.DataFrame({
        name: frame["sinif"].value_counts().reindex(LABELS, fill_value=0)
        for name, frame in split_frames.items()
    })
    table["toplam"] = table.sum(axis=1)
    print("\nSinif dagilimi:\n", table.to_string())


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------- Veri yükleme ----------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("GPU bulunamadi. Colab > Runtime > Change runtime type > A100 GPU secin.")

print("GPU:", torch.cuda.get_device_name(0))
if "A100" not in torch.cuda.get_device_name(0).upper():
    print("Uyari: secili GPU A100 degil; kod yine calisir fakat ayarlar A100 icin yapildi.")

if OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)
INPUT_DIR.mkdir(parents=True)
CHECKPOINT_DIR.mkdir(parents=True)
ARTIFACT_DIR.mkdir(parents=True)

print(
    "ZIP paketini, train/validation/test CSV'lerini veya tek bir CSV/JSONL dosyasini yukleyin.\n"
    "Tek dosya yuklenirse 70/15/15 stratified split otomatik olusturulur."
)
uploaded = files.upload()
if not uploaded:
    raise ValueError("Dosya yuklenmedi")

uploaded_paths = []
for filename, payload in uploaded.items():
    target = INPUT_DIR / Path(filename).name
    target.write_bytes(payload)
    uploaded_paths.append(target)

for archive_path in [path for path in uploaded_paths if path.suffix.casefold() == ".zip"]:
    extract_dir = INPUT_DIR / archive_path.stem
    extract_dir.mkdir(exist_ok=True)
    safe_extract(archive_path, extract_dir)

all_paths = [path for path in INPUT_DIR.rglob("*") if path.is_file()]
train_path = find_named_file(all_paths, "train.csv")
validation_path = find_named_file(all_paths, "validation.csv")
test_path = find_named_file(all_paths, "test.csv")
audit_path = find_named_file(all_paths, "review_audit.csv")

selected_questions = None
if audit_path is not None:
    audit = pd.read_csv(audit_path, encoding="utf-8-sig")
    if {"soru", "karar"}.issubset(audit.columns):
        accepted_decisions = {"keep", "revise", "provided", "existing"}
        selected_questions = set(
            audit.loc[
                audit["karar"].astype(str).str.casefold().isin(accepted_decisions), "soru"
            ].map(normalize_text)
        )
        print(f"Audit kullaniliyor: {len(selected_questions)} egitim icin secili soru.")

if train_path and validation_path and test_path:
    split_frames = {
        "train": canonicalize_frame(read_table(train_path), str(train_path), selected_questions),
        "validation": canonicalize_frame(read_table(validation_path), str(validation_path), selected_questions),
        "test": canonicalize_frame(read_table(test_path), str(test_path), selected_questions),
    }
    split_names = list(split_frames)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1:]:
            overlap = set(split_frames[left_name]["question_key"]) & set(split_frames[right_name]["question_key"])
            if overlap:
                raise ValueError(f"{left_name} ile {right_name} arasinda {len(overlap)} normalize-birebir tekrar var")
else:
    candidate_files = [
        path for path in all_paths
        if path.suffix.casefold() in {".csv", ".jsonl"}
        and path.name.casefold() not in {"review_audit.csv", "candidates.csv"}
        and "audit" not in {part.casefold() for part in path.parts}
    ]
    all_path = find_named_file(candidate_files, "all.csv")
    if all_path is None:
        if len(candidate_files) != 1:
            raise ValueError(
                "Uc split dosyasi bulunamadi. Tek CSV/JSONL yukleyin veya dosyayi all.csv olarak adlandirin. "
                f"Adaylar: {[str(path) for path in candidate_files]}"
            )
        all_path = candidate_files[0]

    full_frame = canonicalize_frame(read_table(all_path), str(all_path), selected_questions)
    class_counts = full_frame["sinif"].value_counts().reindex(LABELS, fill_value=0)
    if int(class_counts.min()) < 7:
        raise ValueError(f"70/15/15 split icin her sinifta en az 7 kayit gerekli:\n{class_counts}")

    train_frame, remainder = train_test_split(
        full_frame,
        test_size=0.30,
        random_state=SEED,
        stratify=full_frame["label"],
    )
    validation_frame, test_frame = train_test_split(
        remainder,
        test_size=0.50,
        random_state=SEED,
        stratify=remainder["label"],
    )
    split_frames = {
        "train": train_frame.reset_index(drop=True),
        "validation": validation_frame.reset_index(drop=True),
        "test": test_frame.reset_index(drop=True),
    }

for split_name, frame in split_frames.items():
    validate_split(frame, split_name)
print_distribution(split_frames)

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Kullanilan split'leri denetlenebilir bicimde kaydet.
for split_name, frame in split_frames.items():
    frame[["soru", "sinif"]].to_csv(ARTIFACT_DIR / f"{split_name}.csv", index=False, encoding="utf-8")

# ---------------------------- Model ve eğitim -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
    problem_type="single_label_classification",
)


def tokenize_batch(batch):
    return tokenizer(batch["soru"], truncation=True, max_length=MAX_LENGTH)


def to_dataset(frame):
    dataset = Dataset.from_pandas(frame[["soru", "label"]], preserve_index=False)
    return dataset.map(tokenize_batch, batched=True, remove_columns=["soru"])


tokenized = {name: to_dataset(frame) for name, frame in split_frames.items()}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

train_counts = split_frames["train"]["label"].value_counts().reindex(range(len(LABELS)), fill_value=0)
class_weights = len(split_frames["train"]) / (len(LABELS) * train_counts.to_numpy(dtype=np.float32))
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("Class weights:", {ID2LABEL[i]: round(float(weight), 4) for i, weight in enumerate(class_weights)})


class ClassWeightedTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
            label_smoothing=0.05,
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_prediction):
    logits, references = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        references, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        references, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(references, predictions),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


bf16_enabled = bool(torch.cuda.is_bf16_supported())
print(f"BF16: {bf16_enabled} | FP16: False")

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.10,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,
    bf16=bf16_enabled,
    fp16=False,
    tf32=True,
    dataloader_num_workers=2,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)

trainer = ClassWeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)],
    class_weights=class_weights,
)

train_result = trainer.train()
validation_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="validation")
test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")

# ---------------------------- Rapor ve çıktı --------------------------------
test_logits = test_output.predictions
test_references = test_output.label_ids
test_predictions = np.argmax(test_logits, axis=-1)
test_probabilities = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

report = classification_report(
    test_references,
    test_predictions,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    zero_division=0,
    output_dict=True,
)
matrix = confusion_matrix(test_references, test_predictions, labels=list(range(len(LABELS))))

predictions_frame = split_frames["test"][["soru", "sinif"]].copy()
predictions_frame["tahmin"] = [ID2LABEL[int(index)] for index in test_predictions]
predictions_frame["guven"] = test_probabilities.max(axis=1)
predictions_frame["dogru_mu"] = predictions_frame["sinif"] == predictions_frame["tahmin"]
predictions_frame.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False, encoding="utf-8")

pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
    ARTIFACT_DIR / "confusion_matrix.csv", encoding="utf-8"
)

import matplotlib.pyplot as plt

figure, axis = plt.subplots(figsize=(8, 6))
image = axis.imshow(matrix, cmap="Blues")
axis.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
axis.set_yticks(range(len(LABELS)), LABELS)
axis.set_xlabel("Tahmin")
axis.set_ylabel("Gercek")
axis.set_title("Test confusion matrix")
for row_index in range(len(LABELS)):
    for column_index in range(len(LABELS)):
        axis.text(column_index, row_index, int(matrix[row_index, column_index]), ha="center", va="center")
figure.colorbar(image, ax=axis)
figure.tight_layout()
figure.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=180)
plt.close(figure)

final_model_dir = ARTIFACT_DIR / "model"
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

summary = {
    "base_model": MODEL_ID,
    "labels": LABELS,
    "label2id": LABEL2ID,
    "seed": SEED,
    "max_length": MAX_LENGTH,
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_validation_metric": trainer.state.best_metric,
    "split_counts": {name: len(frame) for name, frame in split_frames.items()},
    "split_label_counts": {
        name: frame["sinif"].value_counts().reindex(LABELS, fill_value=0).to_dict()
        for name, frame in split_frames.items()
    },
    "class_weights": {ID2LABEL[i]: float(weight) for i, weight in enumerate(class_weights)},
    "train_metrics": train_result.metrics,
    "validation_metrics": validation_metrics,
    "test_metrics": test_output.metrics,
    "test_classification_report": report,
    "versions": {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    },
}
with (ARTIFACT_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
    json.dump(json_ready(summary), handle, ensure_ascii=False, indent=2)
with (ARTIFACT_DIR / "label_mapping.json").open("w", encoding="utf-8") as handle:
    json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, handle, ensure_ascii=False, indent=2)

manifest = []
for path in sorted(ARTIFACT_DIR.rglob("*")):
    if path.is_file():
        manifest.append({
            "path": str(path.relative_to(ARTIFACT_DIR)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        })
with (ARTIFACT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, ensure_ascii=False, indent=2)

zip_path = Path(shutil.make_archive("/content/odine_mdeberta_4class", "zip", ARTIFACT_DIR))

print("\nValidation metrikleri:")
print(json.dumps(json_ready(validation_metrics), ensure_ascii=False, indent=2))
print("\nTest metrikleri:")
print(json.dumps(json_ready(test_output.metrics), ensure_ascii=False, indent=2))
print("\nTest sinif raporu:")
print(pd.DataFrame(report).transpose().round(4).to_string())
print(f"\nModel paketi: {zip_path} ({zip_path.stat().st_size / 1024**2:.1f} MB)")
files.download(str(zip_path))
