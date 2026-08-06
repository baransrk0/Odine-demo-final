# Orin Türkçe Sesli Asistan Çalıştırma Runbook'u

Bu doküman, çalışan demo terminalleri kapatıldıktan sonra sistemi yeniden
başlatmak için gereken kesin komutları içerir. Hem browser-mikrofon hem de
RF/I²S (cihaz mikrofonu) akışı **aynı üç terminali** kullanır; giriş kaynağı
(tarayıcı ↔ RF/I²S) ve ses çıkışı (tarayıcı ↔ USB kulaklık) artık UI'dan,
backend'i yeniden başlatmadan seçilir (bkz. "Ses girişi ve çıkışı seçimi").

| Terminal | Nerede? | Görevi |
|---|---|---|
| 1 | Orin SSH | Gemma 4 `llama-server` (`127.0.0.1:8090`) |
| 2 | Orin SSH | FastAPI backend ve statik UI (`127.0.0.1:8001`) |
| 3 | Mac veya Windows | Yerel `localhost:8001` SSH tüneli |

> **Port notu:** Bu kurulumda backend/UI portu `8001`'dir. Tüm `curl`, tünel ve
> browser adresleri `8001` kullanır. `llama-server` `8090`, Whisper `8080`,
> niyet servisi `6006` portlarında kalır.

Whisper.cpp mevcut Orin kurulumunda terminalden bağımsız olarak
`0.0.0.0:8080` üzerinde çalışır. Piper ayrı servis değildir; backend her
tamamlanan cümle için Piper binary'sini çağırır.

## Bilinen çalışan deployment

```text
Orin SSH adresi:       odine@192.168.1.29
Deployment dizini:    /home/odine/pipeline/uysm-odine-demo-external
Backend environment:  /home/odine/pipeline/uysm-odine-demo-external/backend/.env
Whisper.cpp:           http://127.0.0.1:8080
Gemma llama-server:    http://127.0.0.1:8090
Backend/UI:            http://127.0.0.1:8001
Host browser:          http://localhost:8001
```

Gemma modeli:

```text
/home/odine/models/llama.cpp/models--lmstudio-community--gemma-4-26B-A4B-it-GGUF/snapshots/f11c185920b2c7b202e07e4a9768bf1bea0111d1/gemma-4-26B-A4B-it-Q4_K_M.gguf
```

Piper artifact'ları:

```text
/home/odine/.local/bin/piper
/home/odine/piper-models/tr_TR.onnx
/home/odine/piper-models/tr_TR.onnx.json
```

## Bir defalık kontrol

Orin'e bağlanın:

```bash
ssh odine@192.168.1.29
```

Deployment ve runtime dosyalarını doğrulayın:

```bash
test -d /home/odine/pipeline/uysm-odine-demo-external
test -x /home/odine/pipeline/uysm-odine-demo-external/backend/.venv/bin/uvicorn
test -x /home/odine/.local/bin/piper
test -f /home/odine/piper-models/tr_TR.onnx
test -f /home/odine/piper-models/tr_TR.onnx.json
test -x /home/odine/llama.cpp/build/bin/llama-server
```

`backend/.env` içinde en az şu değerler bulunmalıdır:

```dotenv
LLAMA_CPP_BASE_URL=http://127.0.0.1:8090
LLAMA_CPP_MODEL=gemma-4-26B-A4B-it
STT_BACKEND=whisper_cpp
WHISPER_CPP_BASE_URL=http://127.0.0.1:8080
WHISPER_CPP_INFERENCE_PATH=/inference
TTS_BACKEND=piper
PIPER_BINARY=/home/odine/.local/bin/piper
PIPER_MODEL_PATH=/home/odine/piper-models/tr_TR.onnx
CORS_ORIGINS=http://localhost:8001
```

Kontrol komutu:

```bash
grep -E '^(LLAMA_CPP_BASE_URL|LLAMA_CPP_MODEL|STT_BACKEND|WHISPER_CPP_BASE_URL|TTS_BACKEND|PIPER_BINARY|PIPER_MODEL_PATH|CORS_ORIGINS)=' \
  /home/odine/pipeline/uysm-odine-demo-external/backend/.env
```

## Her açılışta çalıştırma

### Ön kontrol: Whisper.cpp

Mac/Windows üzerinde yeni bir terminalden Orin'e bağlanın:

```bash
ssh odine@192.168.1.29
```

Whisper servisinin ayakta olduğunu doğrulayın:

```bash
ss -ltnp | grep -E ':8080\b'
curl --fail --silent --show-error http://127.0.0.1:8080/health
```

Beklenen process adı `whisper-server`dır. Bu iki komuttan biri başarısızsa
Gemma/backend başlatılmamalıdır. Whisper'ın boot-time servis tanımı bu repoda
yönetilmediği için, tam Orin reboot'u öncesinde mevcut Whisper autostart
mekanizması ayrıca belgelenmelidir. Yalnız terminaller kapatıldıysa mevcut
Whisper process'i çalışmaya devam eder.

Bu ön kontrol SSH oturumundan çıkın:

```bash
exit
```

### Terminal 1: Gemma 4

Mac/Windows üzerinde bir terminal açın:

```bash
ssh odine@192.168.1.29
```

Ollama'nın bellekte tuttuğu demo dışı modeli boşaltın:

```bash
ollama stop gemma3:12b
```

Gemma 4 server'ını başlatın:

```bash
/home/odine/llama.cpp/build/bin/llama-server \
  --model /home/odine/models/llama.cpp/models--lmstudio-community--gemma-4-26B-A4B-it-GGUF/snapshots/f11c185920b2c7b202e07e4a9768bf1bea0111d1/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --alias gemma-4-26B-A4B-it \
  --host 127.0.0.1 \
  --port 8090 \
  --ctx-size 8192 \
  --parallel 1 \
  --reasoning off
```

`--ctx-size 8192` zorunludur. Sistem promptu, onaylanmış 100 soru-cevap
referansını içerir ve yaklaşık 17.700 karakterdir. `4096` ile başlatılırsa
prompt bağlama sığmaz ve turlar kesilir. Referans bloğu sabittir; llama-server
prefix cache'i sayesinde yalnız ilk turda prefill edilir.

Şu satırlar görülmeden devam etmeyin:

```text
model loaded
listening on http://127.0.0.1:8090
```

`--reasoning off` zorunludur. Kaldırılırsa model token bütçesini düşünme
aşamasında tüketebilir ve UI metinsiz bir tur görebilir. Terminal 1 açık
kalmalıdır.

### Terminal 2: Preflight ve backend

Mac/Windows üzerinde ikinci bir terminal açın:

```bash
ssh odine@192.168.1.29
```

Deployment dizinine girip environment'ı yükleyin:

```bash
cd /home/odine/pipeline/uysm-odine-demo-external
set -a
source backend/.env
set +a
```

Preflight çalıştırın:

```bash
./scripts/preflight.sh
```

Başarılı çıktı şu bölümleri ve son satırı içermelidir:

```text
== whisper.cpp ==
== Piper cold synthesis ==
== llama-server ==
Preflight başarılı.
```

Piper cold synthesis sırasında ONNX Runtime'ın
`/sys/class/drm/card1/device/vendor` dosyasıyla ilgili GPU discovery uyarısı
görülebilir. Preflight `Preflight başarılı.` ile bitiyorsa bu uyarı non-fataldır.

Backend'i başlatın:

```bash
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Bu komuta `--ctx-size` eklenmez. `--ctx-size` yalnız Terminal 1'deki
`llama-server` komutuna aittir; uvicorn bu seçeneği tanımaz ve
`No such option '--ctx-size'` hatasıyla çıkar.

Şunları bekleyin:

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8001
```

Terminal 2 açık kalmalıdır.

### Terminal 3: Host SSH tüneli

#### macOS

Mac Terminal'de:

```bash
ssh -N -L 8001:127.0.0.1:8001 odine@192.168.1.29
```

Şifre girildikten sonra terminalin sessiz kalması normaldir. Terminal 3 açık
kalmalıdır.

#### Windows

Windows Terminal veya PowerShell'de aynı komut kullanılır:

```powershell
ssh -N -L 8001:127.0.0.1:8001 odine@192.168.1.29
```

Windows'ta WSL, Python veya Node.js gerekmez. Yalnız OpenSSH Client ve VPN
erişimi gerekir. `ssh` komutu bulunamazsa yönetici PowerShell'de:

```powershell
Get-WindowsCapability -Online | Where-Object Name -Like 'OpenSSH.Client*'
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
```

Kurulumdan sonra yeni bir PowerShell açın.

### Browser

Mac veya Windows browser'ında yalnız şu adresi açın:

```text
http://localhost:8001
```

Orin IP'sini browser'a doğrudan yazmayın. `localhost` hem SSH tünelini
kullanır hem de browser mikrofonu için güvenli context kabul edilir.

UI'da şu dört gösterge yeşil olmalıdır:

```text
Backend Hazır
STT Hazır
TTS Hazır
LLM Hazır
```

`Kaydı başlat` düğmesine basın, mikrofon iznini verin, 2–5 saniyelik net bir
Türkçe cümle söyleyin ve kaydı bitirin. Başarılı turda:

1. Transcript `Siz` alanında görünür.
2. Gemma yanıtı `Asistan` alanında canlı birikir.
3. İlk tamamlanan cümle tüm yanıt beklenmeden Piper'a gönderilir.
4. Ses parçaları browser'da sırayla oynar.
5. `Tur tamamlandı` görünür ve gecikme metrikleri dolar.

## Ses girişi ve çıkışı seçimi (UI'dan, yeniden başlatmadan)

Backend başladıktan sonra giriş kaynağı ve ses çıkışı UI'dan canlı değiştirilir;
`.env` düzenlemek veya uvicorn'u kapatmak gerekmez.

**Ses girişi** (sağ sütun, "Ses girişi" kartı):

- **Tarayıcı girişi**: bilgisayar mikrofonu; `Kaydı başlat` ile manuel kayıt.
- **RF/I²S girişi**: Orin'e bağlı RF alıcısı. Seçildiğinde backend APE I2S
  yönlendirmesini yapar ve `hw:APE,0` üzerinden yakalamayı başlatır.
  Başlatılamazsa kartta hata görünür ve mod tarayıcıda kalır.

**Ses çıkışı** ("Ses çıkışı" kartı):

- **Tarayıcı**: yanıt sesi browser'da çalınır.
- **Cihaz (USB kulaklık)**: yanıt sesi Orin'deki seçili ALSA cihazında çalınır;
  açılır listeden USB kulaklık seçilir. Değişiklik bir sonraki cümleden itibaren
  geçerlidir.

İki mod da serbestçe, ileri-geri değiştirilebilir. Yakalama cihazını
(`hw:APE,0`) başka bir process tutmamalıdır (ör. arka planda çalışan `arecord`
VU metre); aksi halde RF'e geçiş `RF girişi başlatılamadı` hatası verir.

**Dinleme pini (GPIO):** RF/I²S modunda RF kartındaki "Dinleme pini (GPIO)"
satırı yakalama sırasında `HIGH`, boştayken `LOW` gösterir; aynı sinyal donanım
pinini de sürer (bkz. `GPIO_*` ayarları). Rozet ayrıca boşta `PTT bekleniyor`,
yakalarken yeşil `Dinleniyor` gösterir.

## RF/I²S ile hızlı başlangıç (yeni kullanıcı)

RF alıcısı Orin'e bağlıyken, sistemi hiç bilmeyen biri şu adımlarla kullanabilir:

1. **Terminal 1–3'ü** "Her açılışta çalıştırma" bölümündeki gibi başlatın
   (Gemma, backend `--port 8001`, SSH tüneli). Backend'i **RF override olmadan**
   başlatın; `RF_MIC_DEVICE` `hw:APE,0` olmalı ya da tanımsız kalmalı:

   ```bash
   cd /home/odine/pipeline/uysm-odine-demo-external/backend
   .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001
   ```

   (`grep RF_MIC_DEVICE .env` → `hw:APE,0` veya çıktı yoksa tanımsız; her ikisi de
   doğru.)
2. Browser'da `http://localhost:8001` açın; dört gösterge yeşil olmalı.
3. **Ses girişi** kartında **RF/I²S girişi**'ni seçin.
4. **Ses çıkışı** kartında **Cihaz (USB kulaklık)**'ı seçip listeden Orin'e bağlı
   USB kulaklığı seçin.
5. RF alıcısında konuşun (gerekiyorsa PTT'ye basın). Beklenen:
   - "Dinleme pini (GPIO): HIGH" ve yeşil "Dinleniyor",
   - `Siz` alanında transcript, `Asistan` alanında yanıt,
   - yanıt sesi USB kulaklıkta çalınır, `Tur tamamlandı`.

Giriş türü otomatik algılanmaz; RF alıcısını fiziksel bağlamak tek başına modu
değiştirmez — UI'dan **RF/I²S girişi** seçilmelidir. Seçildikten sonra iki mod
arasında ve çıkış hedefleri arasında yeniden başlatmadan geçilebilir.

## RF/I²S .env ayarları

Bu değerler backend başlarken okunur. Mod (tarayıcı/RF) ve çıkış hedefi UI'dan
canlı değişir; ancak aşağıdaki **cihaz ve format** değerleri değişirse backend
yeniden başlatılmalıdır.

```dotenv
# Başlangıç giriş modu (opsiyonel; UI seçimi bunu geçersiz kılar)
AUDIO_INPUT_MODE=browser
# RF yakalama cihazı ve formatı
RF_MIC_DEVICE=hw:APE,0
RF_MIC_SAMPLE_RATE=8000
RF_MIC_CHANNELS=2
APE_CARD=APE
APE_I2S_PORT=I2S2
# Yerel (cihaz) ses çıkışı için varsayılan ALSA cihazı
SPEAKER_DEVICE=plughw:2,0
# Dinleme pini (GPIO) — Jetson.GPIO; varsayılan kapalı
GPIO_LISTENING_ENABLED=false
GPIO_LISTENING_PIN=0
GPIO_MODE=BOARD
```

`AUDIO_INPUT_MODE` yalnız açılıştaki başlangıç modunu belirler; satırı yorumda
bırakırsanız varsayılan `browser`'dır ve UI'dan RF'e geçebilirsiniz.
`GPIO_LISTENING_ENABLED=true` yaparsanız `GPIO_LISTENING_PIN`'i geçerli bir BOARD
pin numarasına ayarlayın ve process'in GPIO iznine sahip olduğundan emin olun
(kullanıcıyı `gpio` grubuna ekleyin ve Jetson.GPIO udev kurallarını kurun),
aksi halde pin sessizce devre dışı kalır ama UI'daki HIGH/LOW göstergesi yine
çalışır.

### Hangi pin HIGH olur?

Sabit/gömülü bir pin **yoktur**; HIGH olan pin `GPIO_LISTENING_PIN` ile seçilir.
Varsayılan değeri `0` ve `GPIO_LISTENING_ENABLED=false` olduğu için kutudan
çıktığı hâliyle hiçbir fiziksel pin sürülmez (UI'daki HIGH/LOW göstergesi yine
çalışır). Numara `GPIO_MODE` ile yorumlanır; varsayılan **BOARD**'dır, yani
40-pinli başlıktaki **fiziksel pin numarası** (ör. `GPIO_LISTENING_PIN=7` =
başlık pini 7, BCM/SoC numarası değil). SoC numaralandırması için `GPIO_MODE=BCM`
yapın.

- Seçtiğiniz pin, yakalama **aktif dinlerken HIGH**, boştayken **LOW** sürülür
  (yani `Dinleniyor` / pin HIGH ile aynı sinyal).
- İsteğe bağlı ikinci hat `GPIO_ARMED_PIN` (varsayılan `0` = kullanılmaz), RF
  yakalama döngüsü ayakta olduğu sürece HIGH kalır.
- Pini seçerken RF alıcısının kullandığı I2S pinleriyle (BCLK/LRCK/SDIN)
  çakışmayan boş bir başlık pini seçin.

**Nereden değiştirilir:** Bu değerler Orin'deki backend `.env` dosyasında
tutulur:

```text
/home/odine/pipeline/uysm-odine-demo-external/backend/.env
```

Dosyayı düzenleyin (ör. `nano`) veya komutla ayarlayın; ardından **backend'i
yeniden başlatın** — GPIO ayarları yalnız açılışta okunur, UI'dan canlı
değişmez:

```bash
cd /home/odine/pipeline/uysm-odine-demo-external/backend
# Örnek: dinleme pinini başlık pini 7 yap ve etkinleştir
sed -i 's|^GPIO_LISTENING_ENABLED=.*|GPIO_LISTENING_ENABLED=true|' .env || echo 'GPIO_LISTENING_ENABLED=true' >> .env
sed -i 's|^GPIO_LISTENING_PIN=.*|GPIO_LISTENING_PIN=7|' .env || echo 'GPIO_LISTENING_PIN=7' >> .env
# Terminal 2'deki uvicorn'u Ctrl+C ile durdurup yeniden başlatın:
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Değeri doğrulamak için: `grep -E '^GPIO_' .env`.

## RF/I²S donanımını doğrulama ve test

### Gerçek RF alıcısı bağlıyken

Backend'i durdurup (cihazı tutmasın) sinyal ve saatleri doğrulayın:

```bash
# Sinyal geliyor mu ve seviyesi makul mü (VU metre)
arecord -D hw:APE,0 -f S16_LE -c 2 -r 8000 -V mono /dev/null
# I2S saatleri canlı mı
watch -n 0.5 "sudo cat /sys/kernel/debug/clk/clk_summary | grep -iE 'i2s|ahub|admaif'"
```

VU metre RF yayınına tepki veriyorsa donanım tarafı hazırdır. Backend'i tekrar
başlatıp UI'dan RF/I²S girişini seçin. Aynı anda yalnız bir process `hw:APE,0`'ı
açabilir; VU metre açıkken backend RF modunda başlatılamaz.

### RF alıcısı olmadan (yerine koyma)

Tüm RF yolunu donanımsız denemek için yakalamayı bir yerine-koyma kaynağına
yönlendirin (8 kHz / 2 kanal):

```bash
# Yazılım loopback
sudo modprobe snd-aloop
RF_MIC_DEVICE=plughw:Loopback,1 .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001
# Başka bir terminalde ses besleyin:
speaker-test -D plughw:Loopback,0 -c 2 -r 8000 -t sine -l 3
```

Veya Orin'e bağlı bir USB mikrofonu `RF_MIC_DEVICE=plughw:<kart>,0` ile kullanın.
UI'dan RF/I²S girişini seçin ve "Dinleniyor"/pin HIGH ile bir turun oluştuğunu
doğrulayın. Tam sesli yanıt için STT/LLM/TTS servisleri açık olmalıdır.

### API'den doğrulama

```bash
curl -s http://127.0.0.1:8001/api/audio/input           # {"mode":"browser"} | {"mode":"rf_i2s"}
curl -s http://127.0.0.1:8001/api/audio/outputs         # mevcut çıkış + ALSA cihaz listesi
curl -sN http://127.0.0.1:8001/api/rf/listening/events  # ses gelince: event: listening / data:{"listening":true}
```

## Windows'a özgü ek kontroller

Windows'ta uygulama veya backend kurulmaz; yalnız browser ve SSH tüneli
çalışır. Aşağıdaki izinleri kontrol edin:

1. **Ayarlar → Gizlilik ve güvenlik → Mikrofon** altında mikrofon erişimini ve
   masaüstü uygulamalarının mikrofon erişimini açın.
2. Chrome/Edge site izinlerinde `http://localhost:8001` için mikrofonu
   `İzin ver` yapın.
3. Browser'da doğru fiziksel mikrofonu seçin; yanlış webcam/headset mikrofonu
   Whisper hallucination'ına neden olabilir.
4. Windows Defender Firewall'da özel bir `8001` inbound kuralı gerekmez;
   browser yerel `localhost:8001` portuna, SSH ise VPN üzerinden Orin'in
   `22` portuna bağlanır.
5. VPN route'u doğrulamak için:

   ```powershell
   Test-NetConnection 192.168.1.29 -Port 22
   ```

   `TcpTestSucceeded : True` beklenir.

Port `8001` Windows'ta başka bir uygulama tarafından kullanılıyorsa önce:

```powershell
Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue
```

Mevcut uygulamayı kapatmak tercih edilir. Tüneli `18000` gibi başka porta
taşımak mümkündür fakat bu durumda `backend/.env` içindeki `CORS_ORIGINS`
değeri de `http://localhost:18000` içerecek şekilde değiştirilip backend
yeniden başlatılmalıdır.

## Niyet motoru (intent engine)

Backend, STT ile LLM arasında bir niyet katmanı çalıştırır. Kayıt
`atbk_knowledge_base.json` içindeki `etiketler` listesine göre yönlendirilir.
Şu an on sınıf vardır: `ilk yardım`, `telsiz ve raporlama`,
`nöbet ve emniyet`, `harita ve intikal`, `mevzi ve gizlenme`, `kbrn korunma`,
`angajman ve esir hukuku`, `matematik`, `sohbet`, `saat`.

Sınıf sayısı ajan sayısına eşit değildir. Ajan her kaydın `beklenen_ajan`
alanından gelir: yukarıdaki altı muharebe sınıfı tek bir `savaş yönergeleri`
ajanına, `ilk yardım` ise `medikal` ajanına yönlenir. Toplam beş ajan vardır:
`medikal`, `savaş yönergeleri`, `matematik`, `sohbet`, `saat`. Sınıf ekleyip
çıkarmak yeni prompt gerektirmez; arayüz her turda hem sınıfı hem ajanı
`İlk yardım → Medikal · %83` biçiminde gösterir.

- `saat` sınıfı değerlendirme setinde `fonksiyon_cagrisi` olarak işaretlidir.
  Bu turlar LLM'e hiç gitmez; cevap `Şu an saat HH:MM` olarak yerel saatten
  üretilir ve doğrudan TTS'e verilir. "Saat kaç" gibi bilinen kalıplar
  sınıflandırıcıya bile sorulmaz; farklı sorulan saat soruları
  sınıflandırıcıdan `saat` etiketiyle dönüp yine aynı yerel yola girer.
- Diğer sınıflar kendi ajan promptuna yönlendirilir. Her prompt yalnız kendi
  etiketinin referans cevaplarını taşır, bu yüzden `llama-server` her ajan için
  ayrı ve sabit bir prefix cache tutar.
- Güven eşiği setin kendi `guven_esigi` değeridir (0.25). Altında kalan turlar
  varsayılan `sohbet` ajanına düşer.

Sınıflandırıcı `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` servisidir ve
Orin üzerinde 6006 portunda çalışır. Servisin başlatılması bu runbook'un
dışındadır; backend yalnız ona bağlanır.

Backend `POST /classify` uç noktasını kullanır. Servis yalnız `/health` ve
`/embed` sunuyorsa niyet yönlendirmesi çalışmaz: gömme vektörlerinden NLI
entailment skoru üretilemez. Uç noktanın sözleşmesi ve tek model kopyasıyla
nasıl ekleneceği `docs/intent-classify-endpoint.md` içindedir.

Servis kapalıysa tur düşmez: saat kuralları çalışmaya devam eder, geri kalan
sorular varsayılan ajanla yanıtlanır. `/api/health` bunu `intent_ready: false`
ile bildirir ama `status` `ok` kalır, çünkü hat hâlâ tur tamamlayabilir.
Arayüzde ilgili tur `sınıflandırıcı yok` etiketiyle görünür.

Kapatmak için `INTENT_ENABLED=false`; bu durumda tek ortak prompt'a ve
`classifying` aşaması olmayan eski akışa dönülür.

### Niyet doğruluğunu ölçme

Değerlendirme seti 100 kayıt içerir ve her kaydın `stt_varyanti` alanı Whisper
çıktısını taklit eder. Orin üzerinde, backend'in venv'i ile:

```bash
cd /home/odine/pipeline/uysm-odine-demo-external
backend/.venv/bin/python scripts/eval_intent.py --output intent_results.json
```

Çıktı; genel doğruluk, etiket başına recall/precision, karışan sınıf çiftleri,
p50/p95 gecikme ve turların hangi yolla (`rule`, `classifier`, `low_confidence`,
`unavailable`) yönlendirildiğini verir.

Hipotez metinlerini veya eşiği ayarlarken modelin ham davranışını görmek için
kuralları ve eşiği devre dışı bırakın:

```bash
backend/.venv/bin/python scripts/eval_intent.py --classifier-only
```

## Hızlı sağlık kontrolleri

Orin üzerinde:

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8090/health
curl --fail http://127.0.0.1:6006/health
curl --fail http://127.0.0.1:8001/api/health
```

Host üzerinde, SSH tüneli açıkken:

```bash
curl --fail http://localhost:8001/api/health
```

Windows PowerShell'de `curl` alias farklarından kaçınmak için:

```powershell
Invoke-RestMethod http://localhost:8001/api/health
```

## Sık hatalar

### `/api/rf/... 409 Conflict`

Backend tarayıcı modundadır; RF uç noktaları (`/api/rf/turns/events`,
`/api/rf/listening/events`) yalnız RF/I²S modunda açıktır. UI'dan **RF/I²S
girişi**'ni seçin. Mod değiştirdikten sonra browser'ı tam yenileyin: eski RF
aboneliği açık kalıp tek seferlik 409 üretebilir. Tarayıcı modunda bu 409'lar
beklenendir, hata değildir.

### `RF girişi başlatılamadı` (503)

RF'e geçerken APE yönlendirmesi veya `arecord` başlatılamadı. `hw:APE,0`'ı başka
bir process tutuyor olabilir (arka plandaki VU metre'yi kapatın) ya da cihaz/ALSA
adları eşleşmiyordur. `arecord -l` ile kartı, `arecord -D hw:APE,0 -f S16_LE -c 2
-r 8000 /dev/null` ile açılabilirliği doğrulayın.

### RF seçili ama ses gelmiyor / tur oluşmuyor

`RF_MIC_DEVICE` gerçek alıcıya (`hw:APE,0`) işaret etmeli. Test için
loopback/USB override kullandıysanız o env var olmadan yeniden başlatın
(`grep RF_MIC_DEVICE .env`). Sinyali VU metre ile, saatleri `watch clk_summary`
ile doğrulayın. Yakalama 8 kHz/2 kanaldır; kaynak farklı format veriyorsa
`plughw:` kullanın. Rozet sürekli `PTT bekleniyor`'da kalıyorsa yakalama hiç
çerçeve almıyordur.

### `127.0.0.1:8090 Connection refused`

Terminal 1 kapalıdır veya Gemma henüz yüklenmemiştir. Terminal 1 komutunu
yeniden çalıştırın ve `listening` satırını bekleyin.

### `localhost:8001` açılmıyor

Terminal 2 backend'ini ve Terminal 3 SSH tünelini kontrol edin. Mac/Windows
host üzerinde:

```bash
ssh -v -N -L 8001:127.0.0.1:8001 odine@192.168.1.29
```

### `Bu kaynaktan erişime izin verilmiyor`

Orin'de:

```bash
cd /home/odine/pipeline/uysm-odine-demo-external/backend
sed -i 's|^CORS_ORIGINS=.*|CORS_ORIGINS=http://localhost:8001|' .env
set -a
source .env
set +a
```

Ardından Terminal 2 backend'ini yeniden başlatın ve browser'da tam yenileme
yapın.

### Asistan metni boş ama tur tamamlandı

Gemma server komutunda `--reasoning off` eksiktir. Terminal 1'i durdurup bu
runbook'taki komutla yeniden başlatın.

### Her soru `sohbet` ajanına gidiyor

Arayüzde tur `sınıflandırıcı yok` diyorsa 6006 portundaki servis kapalıdır;
`curl --fail http://127.0.0.1:6006/health` ile doğrulayın. `varsayılan` diyorsa
servis çalışıyor ama skorlar 0.25 eşiğinin altında kalıyor demektir; önce
`scripts/eval_intent.py --classifier-only` ile ham doğruluğu ölçün, eşiği
`INTENT_CONFIDENCE_THRESHOLD` ile ancak ondan sonra değiştirin.

Servis çalışıyor ama her istek hata veriyorsa büyük ihtimalle `/classify`
sözleşmesi tutmuyordur. `INTENT_CLASSIFY_PATH` ile yolu düzeltin; istek ve yanıt
gövdesi farklıysa `backend/app/runtimes/intent.py` içindeki `_build_request` ve
`_parse_scores` fonksiyonları değiştirilecek tek yerdir.

### Saat sorusuna `HH:MM` diye cevap veriliyor

Saat turu LLM'e gitmiş demektir. Yerel yolda cevap `Şu an saat 14:05` gibi
gerçek saatle üretilir; harfi harfine `HH:MM` görülüyorsa referans cevap bir
ajan promptuna sızmıştır. `INTENT_ENABLED` değerini ve `saat` etiketinin
`fonksiyon_cagrisi` olarak işaretli kaldığını kontrol edin.

### Whisper anlamsız kısa metin üretiyor

Browser'ın doğru mikrofonu kullandığını ve işletim sistemi giriş seviyesini
kontrol edin. Piper sesi oynarken yeni kayıt başlatmayın. Mikrofona yakın,
2–5 saniyelik net bir cümleyle tekrar deneyin.

### Piper GPU discovery uyarısı

Piper ses üretiyor ve preflight başarıyla bitiyorsa uyarı non-fataldır. Piper
bu deployment'ta backend tarafından CLI üzerinden çalıştırılır.

### Disk kullanımı yüksek

Canlı doğrulamada root disk `%97` kullanım ve yaklaşık `16 GB` boş alan
göstermiştir. Yeni model indirmeden veya build cache oluşturmadan önce:

```bash
df -h /
```

Disk temizliği ayrı, inspect-first bir bakım işi olarak yapılmalıdır.

## Sistemi kapatma

Kontrollü kapatma sırası:

1. Aktif ses turunun tamamlanmasını bekleyin.
2. Terminal 3'te `Ctrl+C` ile SSH tünelini kapatın.
3. Terminal 2'de `Ctrl+C` ile Uvicorn'u kapatın.
4. Terminal 1'de `Ctrl+C` ile Gemma `llama-server`ı kapatın.

Whisper.cpp mevcut kurulumda arka plan servisi olarak bırakılır. Orin'i
tamamen kapatmak gerekiyorsa sistem yönetim prosedürünü kullanın; model veya
deployment dizinlerini silmeyin.

## Kod güncellemesini Orin'e taşıma

### macOS

Repo kökünde:

```bash
rsync -az --progress \
  --exclude '.git' \
  --exclude '.env' \
  --exclude 'backend/.venv' \
  --exclude 'frontend/node_modules' \
  /Users/baransarak/Projects/uysm-odine-demo/ \
  odine@192.168.1.29:/home/odine/pipeline/uysm-odine-demo-external/
```

### Windows

Günlük çalıştırma için rsync gerekmez. Kod güncellemesi taşınacaksa en güvenli
eşdeğer WSL içinden aynı `rsync` komutunu çalıştırmaktır. Git Bash/WSL yoksa
`scp -r` kullanılabilir ancak `.env`, `.venv` ve `node_modules` exclude
edilemediği için deployment dizinini gereksiz veya yanlış dosyalarla
ezmemeye dikkat edilmelidir.

---

> **⚠️ Uyarı — komutları kopyalayanlar için:** Bu dokümandaki yollar ve adresler
> bu kuruluma özeldir (ör. `odine@192.168.1.29`, Orin'de
> `/home/odine/pipeline/uysm-odine-demo-external/...`, Mac'te
> `/Users/baransarak/Projects/uysm-odine-demo/`, backend/UI portu `8001`).
> Komutları
> olduğu gibi kopyalamadan önce **kendi SSH adresinizi, dosya yollarınızı ve
> port numaranızı** kendi ortamınıza göre değiştirin. Aksi halde komutlar
> başka bir makinenin yollarını hedefler ve çalışmaz ya da yanlış dizine yazar.
