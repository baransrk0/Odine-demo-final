# Orin Türkçe Sesli Asistan Çalıştırma Runbook'u

Bu doküman, çalışan demo terminalleri kapatıldıktan sonra sistemi yeniden
başlatmak için gereken kesin komutları içerir. Günlük browser-mikrofon akışı
üç açık terminal kullanır:

| Terminal | Nerede? | Görevi |
|---|---|---|
| 1 | Orin SSH | Gemma 4 `llama-server` (`127.0.0.1:8090`) |
| 2 | Orin SSH | FastAPI backend ve statik UI (`127.0.0.1:8000`) |
| 3 | Mac veya Windows | Yerel `localhost:8000` SSH tüneli |

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
Backend/UI:            http://127.0.0.1:8000
Host browser:          http://localhost:8000
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
CORS_ORIGINS=http://localhost:8000
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
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Bu komuta `--ctx-size` eklenmez. `--ctx-size` yalnız Terminal 1'deki
`llama-server` komutuna aittir; uvicorn bu seçeneği tanımaz ve
`No such option '--ctx-size'` hatasıyla çıkar.

Şunları bekleyin:

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

Terminal 2 açık kalmalıdır.

### Terminal 3: Host SSH tüneli

#### macOS

Mac Terminal'de:

```bash
ssh -N -L 8000:127.0.0.1:8000 odine@192.168.1.29
```

Şifre girildikten sonra terminalin sessiz kalması normaldir. Terminal 3 açık
kalmalıdır.

#### Windows

Windows Terminal veya PowerShell'de aynı komut kullanılır:

```powershell
ssh -N -L 8000:127.0.0.1:8000 odine@192.168.1.29
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
http://localhost:8000
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

## Windows'a özgü ek kontroller

Windows'ta uygulama veya backend kurulmaz; yalnız browser ve SSH tüneli
çalışır. Aşağıdaki izinleri kontrol edin:

1. **Ayarlar → Gizlilik ve güvenlik → Mikrofon** altında mikrofon erişimini ve
   masaüstü uygulamalarının mikrofon erişimini açın.
2. Chrome/Edge site izinlerinde `http://localhost:8000` için mikrofonu
   `İzin ver` yapın.
3. Browser'da doğru fiziksel mikrofonu seçin; yanlış webcam/headset mikrofonu
   Whisper hallucination'ına neden olabilir.
4. Windows Defender Firewall'da özel bir `8000` inbound kuralı gerekmez;
   browser yerel `localhost:8000` portuna, SSH ise VPN üzerinden Orin'in
   `22` portuna bağlanır.
5. VPN route'u doğrulamak için:

   ```powershell
   Test-NetConnection 192.168.1.29 -Port 22
   ```

   `TcpTestSucceeded : True` beklenir.

Port `8000` Windows'ta başka bir uygulama tarafından kullanılıyorsa önce:

```powershell
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
```

Mevcut uygulamayı kapatmak tercih edilir. Tüneli `18000` gibi başka porta
taşımak mümkündür fakat bu durumda `backend/.env` içindeki `CORS_ORIGINS`
değeri de `http://localhost:18000` içerecek şekilde değiştirilip backend
yeniden başlatılmalıdır.

## Niyet motoru (intent engine)

Backend, STT ile LLM arasında bir niyet katmanı çalıştırır. Kayıt
`atbk_knowledge_base.json` içindeki beş sınıfa göre yönlendirilir:
`medikal`, `savaş yönergeleri`, `matematik`, `sohbet`, `saat`.

- `saat` sınıfı değerlendirme setinde `fonksiyon_cagrisi` olarak işaretlidir.
  Bu turlar LLM'e hiç gitmez; cevap `Şu an saat HH:MM` olarak yerel saatten
  üretilir ve doğrudan TTS'e verilir. "Saat kaç" gibi bilinen kalıplar
  sınıflandırıcıya bile sorulmaz; farklı sorulan saat soruları
  sınıflandırıcıdan `saat` etiketiyle dönüp yine aynı yerel yola girer.
- Diğer dört sınıf kendi ajan promptuna yönlendirilir. Her prompt yalnız kendi
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
curl --fail http://127.0.0.1:8000/api/health
```

Host üzerinde, SSH tüneli açıkken:

```bash
curl --fail http://localhost:8000/api/health
```

Windows PowerShell'de `curl` alias farklarından kaçınmak için:

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

## Sık hatalar

### `127.0.0.1:8090 Connection refused`

Terminal 1 kapalıdır veya Gemma henüz yüklenmemiştir. Terminal 1 komutunu
yeniden çalıştırın ve `listening` satırını bekleyin.

### `localhost:8000` açılmıyor

Terminal 2 backend'ini ve Terminal 3 SSH tünelini kontrol edin. Mac/Windows
host üzerinde:

```bash
ssh -v -N -L 8000:127.0.0.1:8000 odine@192.168.1.29
```

### `Bu kaynaktan erişime izin verilmiyor`

Orin'de:

```bash
cd /home/odine/pipeline/uysm-odine-demo-external/backend
sed -i 's|^CORS_ORIGINS=.*|CORS_ORIGINS=http://localhost:8000|' .env
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
  /Users/hasan/Desktop/uysm-odine-demo/ \
  odine@192.168.1.29:/home/odine/pipeline/uysm-odine-demo-external/
```

### Windows

Günlük çalıştırma için rsync gerekmez. Kod güncellemesi taşınacaksa en güvenli
eşdeğer WSL içinden aynı `rsync` komutunu çalıştırmaktır. Git Bash/WSL yoksa
`scp -r` kullanılabilir ancak `.env`, `.venv` ve `node_modules` exclude
edilemediği için deployment dizinini gereksiz veya yanlış dosyalarla
ezmemeye dikkat edilmelidir.
