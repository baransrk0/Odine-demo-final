# Colab: eğitim hücresinden sonra bunu AYRI bir hücreye yapıştırıp çalıştırın.
# Önce Colab sol paneli > anahtar simgesi (Secrets) > HF_TOKEN ekleyin.
# Token, Hugging Face'te bu namespace için write yetkisine sahip olmalıdır.
print("[1/7] Mevcut Colab paketleri yükleniyor...", flush=True)
import json
import re
import shutil
import zipfile
from pathlib import Path

from google.colab import userdata
from huggingface_hub import HfApi, __version__ as hub_version, create_repo, update_repo_settings

print(f"[1/7] huggingface_hub {hub_version} hazır.", flush=True)


# İstersen yalnızca bu adı değiştir. Namespace kullanıcı hesabından otomatik alınır.
REPO_NAME = "odine-mdeberta-v3-intent-tr-4class"
BASE_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
EXPECTED_LABELS = {"medikal", "savaş yönergeleri", "matematik", "sohbet"}
ARTIFACT_ROOT = Path("/content/odine_mdeberta_4class/artifact")
STAGING_DIR = Path("/content/odine_hf_publish")


def safe_extract(zip_path, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Güvenli olmayan ZIP yolu: {member.filename}")
        archive.extractall(destination)


def is_model_dir(path):
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


def find_artifact():
    preferred_model = ARTIFACT_ROOT / "model"
    if is_model_dir(preferred_model):
        return ARTIFACT_ROOT, preferred_model

    model_candidates = sorted(
        {config.parent for config in Path("/content").rglob("config.json") if is_model_dir(config.parent)},
        key=lambda path: (len(path.parts), str(path)),
    )
    if model_candidates:
        model_dir = model_candidates[0]
        artifact_root = model_dir.parent if model_dir.name == "model" else model_dir
        return artifact_root, model_dir

    from google.colab import files

    print("Eğitilmiş model bulunamadı. odine_mdeberta_4class.zip model paketini yükleyin; dataset ZIP'ini değil.")
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.casefold().endswith(".zip")]
    if len(zip_names) != 1:
        raise ValueError("Tam olarak bir model ZIP dosyası yükleyin.")
    zip_path = Path("/content") / Path(zip_names[0]).name
    zip_path.write_bytes(uploaded[zip_names[0]])
    extracted = Path("/content/odine_hf_upload_source")
    extracted.mkdir(parents=True, exist_ok=True)
    safe_extract(zip_path, extracted)
    model_candidates = sorted(
        {config.parent for config in extracted.rglob("config.json") if is_model_dir(config.parent)},
        key=lambda path: (len(path.parts), str(path)),
    )
    if not model_candidates:
        raise FileNotFoundError("ZIP içinde dört sınıflı fine-tune model bulunamadı.")
    model_dir = model_candidates[0]
    artifact_root = model_dir.parent if model_dir.name == "model" else extracted
    return artifact_root, model_dir


print("[2/7] Colab Secret HF_TOKEN kontrol ediliyor...", flush=True)
HF_TOKEN = userdata.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "Colab Secrets'a HF_TOKEN ekleyin ve bu notebook için erişimi açın. "
        "Tokenı hücre içine yazmayın."
    )

if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,95}", REPO_NAME):
    raise ValueError(f"Geçersiz REPO_NAME: {REPO_NAME}")

print("[3/7] Hugging Face hesabı doğrulanıyor...", flush=True)
api = HfApi(token=HF_TOKEN)
account = api.whoami()
namespace = account["name"]
REPO_ID = f"{namespace}/{REPO_NAME}"
print(f"[3/7] Hesap doğrulandı: {namespace}", flush=True)

print("[4/7] Model artefaktı bulunuyor ve staging alanı hazırlanıyor...", flush=True)
artifact_root, model_dir = find_artifact()
if STAGING_DIR.exists():
    shutil.rmtree(STAGING_DIR)
shutil.copytree(model_dir, STAGING_DIR)

# Yalnızca toplu değerlendirme artefaktlarını ekle. Satır bazındaki özel veri yüklenmez.
for filename in ("metrics.json", "label_mapping.json", "confusion_matrix.csv", "confusion_matrix.png"):
    source = artifact_root / filename
    if source.is_file():
        shutil.copy2(source, STAGING_DIR / filename)

# Bilinçli dışlamalar: dataset/ ve test_predictions.csv model reposuna yüklenmez.
for forbidden in (STAGING_DIR / "dataset", STAGING_DIR / "test_predictions.csv"):
    if forbidden.is_dir():
        shutil.rmtree(forbidden)
    elif forbidden.exists():
        forbidden.unlink()

model_card = f"""---
language:
- tr
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: {BASE_MODEL_ID}
base_model_relation: finetune
tags:
- deberta-v3
- intent-classification
- turkish
- odine
metrics:
- accuracy
- f1
---

# Odine Turkish Intent Classifier, 4 Classes

`{BASE_MODEL_ID}` taban modelinin dört sınıflı Türkçe intent sınıflandırma için fine-tune edilmiş sürümüdür.

## Sınıflar

- `medikal`
- `savaş yönergeleri`
- `matematik`
- `sohbet`

Güncel saat sorguları modelin eğitim kapsamı dışındadır ve uygulamada deterministik bir kuralla işlenir.

## Değerlendirme

Donmuş 154 örneklik test splitinde:

| Metrik | Sonuç |
|---|---:|
| Accuracy | 0.9610 |
| Macro F1 | 0.9600 |
| Weighted F1 | 0.9614 |

Sınıf bazındaki test F1 değerleri: medikal 0.9259, savaş yönergeleri 0.9615, matematik 1.0000, sohbet 0.9524.

Bu skorlar aynı kontrollü üretim sürecinden gelen dataset-içi split üzerinde ölçülmüştür. Gerçek STT trafiği veya bağımsız saha genellemesi kanıtı değildir.

## Eğitim verisi

Toplam 1.028 Türkçe soru kullanıldı: 720 train, 154 validation, 154 test. Dağılım medikal 167, savaş yönergeleri 528, matematik 182, sohbet 151. Veri seti ve kayıt bazındaki tahminler bu model reposuna yüklenmemiştir.

Veri kalite sınırları: önceki paketten 99 kayıt `revise` durumunu korur; sonradan sağlanan 400 kayıt bağımsız insan veya LLM incelemesinden geçirilmemiştir. Normalize tekrarlar ve splitler arası yüksek sözcüksel benzerlik kontrol edilmiştir.

## Kullanım

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="{REPO_ID}",
    token=True,
)
print(classifier("Yaralının kolu kanıyor, ne yapmalıyım?"))
```

Private repo erişimi için istemci ortamında Hugging Face tokenıyla giriş yapılmalıdır.
"""
(STAGING_DIR / "README.md").write_text(model_card, encoding="utf-8")

required_files = {"config.json", "tokenizer_config.json", "README.md"}
staged_files = {path.name for path in STAGING_DIR.iterdir() if path.is_file()}
if not required_files.issubset(staged_files):
    raise FileNotFoundError(f"Eksik yayın dosyaları: {sorted(required_files - staged_files)}")
if not ({"model.safetensors", "pytorch_model.bin"} & staged_files):
    raise FileNotFoundError("Model ağırlık dosyası bulunamadı.")
if "test_predictions.csv" in staged_files or (STAGING_DIR / "dataset").exists():
    raise RuntimeError("Satır bazındaki veri staging alanında bulundu; upload iptal edildi.")

staging_bytes = sum(path.stat().st_size for path in STAGING_DIR.rglob("*") if path.is_file())
print(f"[4/7] Staging hazır: {staging_bytes / 1024**2:.1f} MB", flush=True)
print("Yüklenecek dosyalar:", sorted(path.name for path in STAGING_DIR.iterdir() if path.is_file()), flush=True)

# exist_ok mevcut bir repoyu private yapmaz; bu yüzden görünürlük ayrıca zorlanır.
print(f"[5/7] Private repo hazırlanıyor: {REPO_ID}", flush=True)
create_repo(repo_id=REPO_ID, token=HF_TOKEN, private=True, exist_ok=True, repo_type="model")
update_repo_settings(repo_id=REPO_ID, private=True, token=HF_TOKEN, repo_type="model")
print("[5/7] Repo private olarak hazır.", flush=True)

print("[6/7] Model Hub'a yükleniyor. Bu adım ağ hızına bağlıdır ve yeniden sürdürülebilir...", flush=True)
api.upload_large_folder(
    folder_path=str(STAGING_DIR),
    repo_id=REPO_ID,
    repo_type="model",
)
print("[6/7] Upload çağrısı tamamlandı.", flush=True)

print("[7/7] Private görünürlük ve uzak dosyalar doğrulanıyor...", flush=True)
model_info = api.model_info(REPO_ID, files_metadata=True)
assert model_info.private is True, "Gizlilik doğrulaması başarısız: repo private değil."
remote_files = set(api.list_repo_files(REPO_ID, repo_type="model"))
expected_remote = {path.name for path in STAGING_DIR.iterdir() if path.is_file()}
missing_remote = expected_remote - remote_files
if missing_remote:
    raise RuntimeError(f"Hub upload eksik dosyalar içeriyor: {sorted(missing_remote)}")
if "test_predictions.csv" in remote_files or any(path.startswith("dataset/") for path in remote_files):
    raise RuntimeError("Hub reposunda satır bazındaki özel veri bulundu.")

print("\n[7/7] Yayın tamamlandı ve private olduğu doğrulandı:", flush=True)
print(f"https://huggingface.co/{REPO_ID}", flush=True)
print("Hub dosya sayısı:", len(remote_files), flush=True)
