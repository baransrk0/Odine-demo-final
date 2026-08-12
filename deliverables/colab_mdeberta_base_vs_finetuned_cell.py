# Colab A100: eğitimden sonra bunu AYRI bir hücreye yapıştırıp çalıştırın.
# Her soru önce base zero-shot modele, sonra fine-tune modele verilir.
import json
import re
import subprocess
import sys
import time
import unicodedata
import zipfile
from pathlib import Path

try:
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.57.6",
        "sentencepiece==0.2.1",
        "pandas",
    ])
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


BASE_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LABELS = ["medikal", "savaş yönergeleri", "matematik", "sohbet"]
VERBALIZATIONS = {
    "medikal": "yaralıya yapılan ilk yardım",
    "savaş yönergeleri": "askerî görev ve saha yönergeleri",
    "matematik": "sayılarla yapılan bir hesaplama",
    "sohbet": "günlük sohbet ve selamlaşma",
}
HYPOTHESIS_TEMPLATE = "Bu metnin konusu {}."
MAX_LENGTH = 96


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


_CLOCK_PREFIX = r"(?:(?:lütfen|bana|acaba|peki|bi|bir)\s+)*"
_CLOCK_SUFFIX = r"(?:\s+(?:oldu|acaba|lütfen|biliyor\s+musun(?:uz)?))*"
_CLOCK_BODIES = (
    r"saat",
    r"(?:şu\s+an(?:da)?\s+)?saat(?:in|i)?\s+kaç(?:tır|ta|te)?",
    r"saat(?:in|i)?\s+söyle(?:r|yebilir)?(?:\s+mi(?:sin|siniz))?",
    r"(?:şu\s+an(?:da)?\s+)?saat\s+ne(?:dir)?",
)
_CLOCK_PATTERN = re.compile(
    rf"^{_CLOCK_PREFIX}(?:{'|'.join(_CLOCK_BODIES)}){_CLOCK_SUFFIX}$"
)


def matches_clock(normalized_text):
    return bool(normalized_text) and _CLOCK_PATTERN.fullmatch(normalized_text) is not None


def safe_extract(zip_path, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Güvenli olmayan ZIP yolu: {member.filename}")
        archive.extractall(destination)


def is_finetuned_model_dir(path):
    config_path = path / "config.json"
    if not config_path.is_file() or not (path / "tokenizer_config.json").is_file():
        return False
    if not ((path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configured_labels = {str(value) for value in config.get("id2label", {}).values()}
    except (OSError, ValueError, TypeError):
        return False
    return configured_labels == set(LABELS)


def find_finetuned_model_dir():
    preferred = [
        Path("/content/odine_mdeberta_4class/artifact/model"),
        Path("/content/odine_mdeberta_4class/model"),
        Path("/content/model"),
    ]
    for path in preferred:
        if is_finetuned_model_dir(path):
            return path

    candidates = sorted(
        {config.parent for config in Path("/content").rglob("config.json") if is_finetuned_model_dir(config.parent)},
        key=lambda path: (len(path.parts), str(path)),
    )
    if candidates:
        return candidates[0]

    from google.colab import files

    print("Fine-tune model bulunamadı. Eğitim sonunda indirilen odine_mdeberta_4class.zip dosyasını yükleyin.")
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.casefold().endswith(".zip")]
    if len(zip_names) != 1:
        raise ValueError("Tam olarak bir fine-tune model ZIP dosyası yükleyin; dataset ZIP'ini yüklemeyin.")
    zip_path = Path("/content") / Path(zip_names[0]).name
    zip_path.write_bytes(uploaded[zip_names[0]])
    destination = Path("/content/odine_comparison_finetuned")
    destination.mkdir(parents=True, exist_ok=True)
    safe_extract(zip_path, destination)
    candidates = sorted(
        {config.parent for config in destination.rglob("config.json") if is_finetuned_model_dir(config.parent)},
        key=lambda path: (len(path.parts), str(path)),
    )
    if not candidates:
        raise FileNotFoundError("ZIP içinde dört sınıflı fine-tune model bulunamadı.")
    return candidates[0]


if not torch.cuda.is_available():
    raise RuntimeError("GPU bulunamadı. Colab runtime'ında GPU seçin.")

DEVICE = torch.device("cuda")
FT_MODEL_DIR = find_finetuned_model_dir()

print("Fine-tune model yükleniyor:", FT_MODEL_DIR)
ft_tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_DIR, local_files_only=True)
ft_model = AutoModelForSequenceClassification.from_pretrained(
    FT_MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
).to(DEVICE)
ft_model.eval()

print("Base zero-shot model yükleniyor:", BASE_MODEL_ID)
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
).to(DEVICE)
base_model.eval()
base_classifier = pipeline(
    "zero-shot-classification",
    model=base_model,
    tokenizer=base_tokenizer,
    device=0,
)

verbalization_to_label = {verbalization: label for label, verbalization in VERBALIZATIONS.items()}


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def base_tahmin(soru):
    synchronize_gpu()
    started = time.perf_counter()
    output = base_classifier(
        soru,
        candidate_labels=[VERBALIZATIONS[label] for label in LABELS],
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    synchronize_gpu()
    elapsed_ms = (time.perf_counter() - started) * 1000
    scores = {
        verbalization_to_label[verbalization]: float(score)
        for verbalization, score in zip(output["labels"], output["scores"], strict=True)
    }
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    predicted = next(iter(scores))
    return {"sinif": predicted, "guven": scores[predicted], "skorlar": scores, "sure_ms": elapsed_ms}


@torch.inference_mode()
def finetune_tahmin(soru):
    encoded = ft_tokenizer(
        soru,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(DEVICE)
    synchronize_gpu()
    started = time.perf_counter()
    logits = ft_model(**encoded).logits[0]
    synchronize_gpu()
    elapsed_ms = (time.perf_counter() - started) * 1000
    probabilities = torch.softmax(logits.float(), dim=-1).cpu()
    scores = {
        ft_model.config.id2label[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    predicted = next(iter(scores))
    return {"sinif": predicted, "guven": scores[predicted], "skorlar": scores, "sure_ms": elapsed_ms}


def modelleri_kiyasla(soru, skorları_göster=True):
    soru = str(soru).strip()
    if not soru:
        raise ValueError("Soru boş olamaz.")

    if matches_clock(normalize_text(soru)):
        print("\nBu ifade saat kuralıyla yakalandı; iki modele de gönderilmedi.")
        return {
            "soru": soru,
            "base": {"sinif": "saat", "guven": 1.0, "skorlar": {"saat": 1.0}, "sure_ms": 0.0},
            "finetune": {"sinif": "saat", "guven": 1.0, "skorlar": {"saat": 1.0}, "sure_ms": 0.0},
            "ayni_tahmin": True,
            "kaynak": "kural",
        }

    base_result = base_tahmin(soru)
    finetune_result = finetune_tahmin(soru)
    result = {
        "soru": soru,
        "base": base_result,
        "finetune": finetune_result,
        "ayni_tahmin": base_result["sinif"] == finetune_result["sinif"],
        "kaynak": "modeller",
    }

    comparison = pd.DataFrame(
        [
            {
                "model": "Base zero-shot",
                "sinif": base_result["sinif"],
                "guven_%": round(base_result["guven"] * 100, 2),
                "sure_ms": round(base_result["sure_ms"], 2),
            },
            {
                "model": "Fine-tune",
                "sinif": finetune_result["sinif"],
                "guven_%": round(finetune_result["guven"] * 100, 2),
                "sure_ms": round(finetune_result["sure_ms"], 2),
            },
        ]
    )
    print(f"\nSoru: {soru}")
    print(comparison.to_string(index=False))
    print("Karar:", "AYNI" if result["ayni_tahmin"] else "FARKLI")

    if skorları_göster:
        score_table = pd.DataFrame(
            {
                "sinif": LABELS,
                "base_%": [round(base_result["skorlar"][label] * 100, 2) for label in LABELS],
                "finetune_%": [round(finetune_result["skorlar"][label] * 100, 2) for label in LABELS],
            }
        )
        print("\nTüm sınıf skorları:")
        print(score_table.to_string(index=False))
    return result


# İlk çağrılardaki CUDA/kernel ısınma maliyetini süre karşılaştırmasından ayır.
print("Modeller ısıtılıyor...")
_ = base_tahmin("Merhaba")
_ = finetune_tahmin("Merhaba")
print("Hazır. Aynı soru iki modele sırayla verilecek. Bitirmek için 'çıkış' yazın.\n")

while True:
    try:
        soru = input("Soru: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBitti.")
        break
    if normalize_text(soru) in {"çıkış", "cikis", "quit", "exit"}:
        print("Bitti.")
        break
    if not soru:
        continue
    modelleri_kiyasla(soru)
