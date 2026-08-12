# Colab: eğitim hücresinden SONRA bunu ayrı bir hücreye yapıştırıp çalıştırın.
import json
import re
import subprocess
import sys
import unicodedata
import zipfile
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==4.57.6",
        "sentencepiece==0.2.1",
    ])
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


EXPECTED_LABELS = {"medikal", "savaş yönergeleri", "matematik", "sohbet"}
MAX_LENGTH = 96


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


# Repodaki saat kuralının Colab karşılığı. "Kaç saatte giderim?" gibi matematik
# sorularını yanlışlıkla saat sınıfına almamak için tüm ifadeye sabitlenmiştir.
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


def is_trained_model_dir(path):
    config_path = path / "config.json"
    if not config_path.is_file() or not (path / "tokenizer_config.json").is_file():
        return False
    if not ((path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        labels = {str(value) for value in config.get("id2label", {}).values()}
    except (OSError, ValueError, TypeError):
        return False
    return labels == EXPECTED_LABELS


def find_model_dir():
    preferred = [
        Path("/content/odine_mdeberta_4class/artifact/model"),  # Aynı eğitim oturumu
        Path("/content/odine_mdeberta_4class/model"),
        Path("/content/model"),
    ]
    for path in preferred:
        if is_trained_model_dir(path):
            return path

    candidates = sorted(
        {config.parent for config in Path("/content").rglob("config.json") if is_trained_model_dir(config.parent)},
        key=lambda path: (len(path.parts), str(path)),
    )
    if candidates:
        return candidates[0]

    from google.colab import files

    print("Eğitilmiş model bulunamadı. Eğitim sonunda indirilen odine_mdeberta_4class.zip dosyasını yükleyin.")
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.casefold().endswith(".zip")]
    if len(zip_names) != 1:
        raise ValueError("Tam olarak bir eğitilmiş model ZIP dosyası yükleyin.")
    zip_path = Path("/content") / Path(zip_names[0]).name
    zip_path.write_bytes(uploaded[zip_names[0]])
    destination = Path("/content/odine_inference_model")
    destination.mkdir(parents=True, exist_ok=True)
    safe_extract(zip_path, destination)
    candidates = sorted(
        {config.parent for config in destination.rglob("config.json") if is_trained_model_dir(config.parent)},
        key=lambda path: (len(path.parts), str(path)),
    )
    if not candidates:
        raise FileNotFoundError("ZIP içinde dört sınıflı eğitilmiş model bulunamadı.")
    return candidates[0]


MODEL_DIR = find_model_dir()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(DEVICE)
model.eval()


@torch.inference_mode()
def intent_sor(soru, skorları_göster=True):
    soru = str(soru).strip()
    if not soru:
        raise ValueError("Soru boş olamaz.")

    if matches_clock(normalize_text(soru)):
        sonuc = {
            "soru": soru,
            "sinif": "saat",
            "guven": 1.0,
            "kaynak": "kural",
            "skorlar": {"saat": 1.0},
        }
    else:
        encoded = tokenizer(
            soru,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(DEVICE)
        logits = model(**encoded).logits[0]
        probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        scores = {
            model.config.id2label[index]: float(probabilities[index])
            for index in range(len(probabilities))
        }
        predicted_class = max(scores, key=scores.get)
        sonuc = {
            "soru": soru,
            "sinif": predicted_class,
            "guven": scores[predicted_class],
            "kaynak": "model",
            "skorlar": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
        }

    print(f"\nSınıf  : {sonuc['sinif']}")
    print(f"Güven  : %{sonuc['guven'] * 100:.2f}")
    print(f"Kaynak : {sonuc['kaynak']}")
    if skorları_göster:
        print("Skorlar:")
        for label, score in sonuc["skorlar"].items():
            print(f"  {label:<20} %{score * 100:6.2f}")
    return sonuc


print(f"Model hazır: {MODEL_DIR}")
print(f"Cihaz: {DEVICE}")
print("Soru yazın. Bitirmek için 'çıkış' yazın.\n")

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
    intent_sor(soru)
