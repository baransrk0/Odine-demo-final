# Orin Türkçe Sesli Asistan

Tek kullanıcılı demo; tarayıcıdan alınan Türkçe sesi Orin üzerinde
STT → `llama-server` → cümle bazlı TTS zincirinden geçirir. İlk tamamlanan
cümle, LLM yanıtının tamamı beklenmeden sentezlenir ve oynatılır.

Çalışan Orin deployment'ını Mac veya Windows üzerinden terminal terminal
başlatmak için [RUNBOOK.md](RUNBOOK.md) dosyasını kullanın.

Model kimlikleri bilinçli olarak repoda boştur. Uygulama RAG, Qdrant,
embedding, kimlik doğrulama veya kalıcı konuşma geçmişi içermez.

## Gereksinimler

- Jetson Orin ve seçili runtime'lar için yeterli GPU/RAM
- Python 3.10+, Node.js/npm ve ffmpeg/ffprobe
- OpenAI uyumlu `/v1/chat/completions` sunan sağlıklı `llama-server`
- VPN/yerel ağ erişimi; public internet yayını yok
- Mikrofon için güvenli tarayıcı bağlamı: HTTPS veya tarayıcının güvenli
  kabul ettiği yerel origin

## Kurulum

Frontend'i üretin:

```bash
cd frontend
npm ci
npm run build
```

Backend ortamını kurup test edin:

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -v
```

Yerel `uvicorn` çalıştırması için `.env.example` dosyasını
`backend/.env` konumuna kopyalayın. Varsayılan Hugging Face runtime'ları için
model kimlikleri boş kalır. Orin'deki mevcut servisler için aşağıdaki harici
runtime ayarlarını kullanın:

```dotenv
LLAMA_CPP_BASE_URL=http://127.0.0.1:8090
LLAMA_CPP_MODEL=gemma-4-26B-A4B-it
STT_BACKEND=whisper_cpp
WHISPER_CPP_BASE_URL=http://127.0.0.1:8080
WHISPER_CPP_INFERENCE_PATH=/inference
TTS_BACKEND=piper
PIPER_BINARY=/home/odine/.local/bin/piper
PIPER_MODEL_PATH=/home/odine/piper-models/tr_TR.onnx
```

Piper modelinin yanında aynı ada sahip
`/home/odine/piper-models/tr_TR.onnx.json` yapılandırma dosyası da bulunmalıdır.
Harici modda backend Torch veya Transformers import etmez. Hugging Face
STT/TTS kullanmak isterseniz `STT_BACKEND=huggingface` ve
`TTS_BACKEND=huggingface` ayarlayın; bağımlılıkları ayrıca
`pip install -e '.[huggingface]'` ile kurun.

## Preflight

Gerçek deployment ortam değişkenlerini yükledikten sonra:

```bash
./scripts/preflight.sh
```

Kontrol; seçili runtime'ın ayarlarını, disk/RAM durumunu, ffmpeg/ffprobe,
whisper.cpp ve yerel `llama-server` sağlığını doğrular. Piper seçiliyse geçici
bir WAV üreterek binary, model ve model yapılandırmasını birlikte sınar.
Harici whisper.cpp/Piper modunda `nvcc` veya backend içinde görünür CUDA
aranmaz. Hugging Face runtime seçildiğinde Python bağımlılıkları, yalnız
cihazı `cuda` olan seçili runtime için de CUDA görünürlüğü kontrol edilir.
Değerlerin kendisini, prompt içeriğini veya erişim bilgilerini yazdırmaz.

Orin'de özellikle `/`, Hugging Face cache ve geçici ses alanında boş disk
bulundurun. CUDA OOM görülürse eşzamanlı LLM/TTS yükünü, dtype ayarını ve model
bellek bütçesini yeniden ölçün. Tekrarlayan timeout'ta önce GPU/RAM baskısını,
sonra ilgili `*_TIMEOUT_SECONDS` değerini inceleyin. `llama-server`
erişilemiyorsa uygulamadan önce yapılandırılan `LLAMA_CPP_BASE_URL` adresinin
health endpoint'ini düzeltin.

## Yerel çalıştırma

```bash
cd backend
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`0.0.0.0` yalnızca Orin host firewall'u portu VPN/LAN ile sınırlıyorsa
kullanılmalıdır. `CORS_ORIGINS`, Mac tarayıcısındaki gerçek UI origin'ine
ayarlanmalıdır. SSH/VPN adresleri, anahtarlar ve token'lar environment
dosyasına veya loglara yazılmaz.

Tarayıcı otomatik oynatmayı engellerse UI aynı ses parçasını korur ve
`Sesi oynat` düğmesini gösterir.

## RF / I2S input and local headset playback

The default `AUDIO_INPUT_MODE=browser` keeps the browser microphone flow.
For the Jetson-connected RF receiver, set the following deployment-only
environment values:

```dotenv
AUDIO_INPUT_MODE=rf_i2s
RF_MIC_DEVICE=hw:APE,0
RF_MIC_SAMPLE_RATE=8000
RF_MIC_CHANNELS=2
RF_CAPTURE_MAX_SECONDS=30
RF_PTT_FRAME_TIMEOUT_MS=300
RF_PTT_MIN_SECONDS=0.30
APE_CARD=APE
APE_I2S_PORT=I2S2
LOCAL_AUDIO_PLAYBACK=true
SPEAKER_DEVICE=plughw:2,0
```

In this mode the RF receiver feeds Jetson I2S2, APE routes it to `hw:APE,0`,
and PTT clock loss completes one utterance. The backend converts the captured
8 kHz stereo PCM to the existing 16 kHz mono STT input. Each generated TTS
chunk is played in order through the configured USB headset. This mode does
not transmit TTS back over RF.

Before starting the service on the target Orin, inspect the real ALSA names:

```bash
arecord -l
aplay -l
amixer -c APE controls
```

The deployment user must be allowed to access the ALSA devices. Do not assume
the reference card numbers are portable across Orin images.

## systemd

Dedicated kullanıcı ve yazılabilir alanı hazırlayın:

```bash
sudo useradd --system --home /var/lib/orin-voice-assistant \
  --shell /usr/sbin/nologin orin-voice
sudo install -d -o orin-voice -g orin-voice \
  /var/lib/orin-voice-assistant/audio \
  /var/lib/orin-voice-assistant/huggingface
sudo install -m 0644 systemd/orin-voice-assistant.service \
  /etc/systemd/system/orin-voice-assistant.service
sudo install -m 0600 /path/to/deployment.env \
  /etc/orin-voice-assistant.env
sudo systemctl daemon-reload
sudo systemctl enable --now orin-voice-assistant
```

`llama-server` bu servisten ayrı yönetilir ve voice-assistant başlamadan önce
sağlıklı olmalıdır. Unit dosyasında secret bulunmaz. `PrivateTmp=false`,
`TMPDIR=/var/lib/orin-voice-assistant/audio` alanının kullanılabilmesi içindir.

## Smoke testi

İçeriği hassas olmayan, izinli bir Türkçe ses fixture'ı ile:

```bash
python scripts/smoke_turn.py fixture.wav http://ORIN-VPN-ADRESI:8000
```

Script tur oluşturur, SSE akışını izler, her ses parçasını indirir, sıra
boşluklarını ve metrikleri doğrular. Varsayılan çıktı transcript veya yanıt
içeriğini göstermez. Yalnızca kontrollü yerel teşhis için `--show-content`
kullanın.

Cold-start kabul kaydında yalnızca şunları saklayın:

```text
tarih/saat, git commit, sonuç, güvenli device/dtype özeti,
upload/STT/LLM/TTS/toplam gecikme,
ilk cümle ve ilk oynatma gecikmesi,
parça sayısı ve nullable token telemetrisi
```

## Kabul kontrolü

Cold start sonrasında:

1. Mac tarayıcısından VPN üzerinden mikrofon kaydını başlatıp durdurun.
2. Türkçe transcript, metin yanıtı ve sıralı ses oynatmayı doğrulayın.
3. STT, LLM, TTS, toplam, ilk cümle ve ilk oynatma metriklerini doğrulayın.
4. İlk sesin LLM tamamlanmadan başladığını doğrulayın.
5. Aktif tur sırasında ikinci isteğin `409` aldığını doğrulayın.
6. Kontrollü bir STT/LLM/TTS hatasından sonra yeni turun başlayabildiğini
   doğrulayın.
7. `rg -n -i 'qdrant|rag|embedding' backend frontend scripts systemd`
   çıktısında runtime bağımlılığı olmadığını doğrulayın.
8. `smoke_turn.py` ile en az bir başarılı cold-start turunu belgeleyin.

Ham ses, transcript ve yanıt varsayılan olarak kalıcı tutulmaz. Son metrikler
yalnızca süreç belleğinde sınırlı sayıda saklanır; restart ile silinir.
