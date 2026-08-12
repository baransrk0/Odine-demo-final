# Odine dört sınıflı intent fine-tune veri seti

## Kısa özet

Bu paket, önceki 700 soruluk paketin dört sınıfa eşlenen kayıtları ile kullanıcı tarafından sağlanan 400 yeni sorunun birleşimidir. `saat` sınıfındaki 72 kayıt, repoda deterministik kural tarafından işlendiği için sınıflandırıcı eğitiminden çıkarıldı.

Toplam 1028 benzersiz soru, seed 17 ile sınıf bazında deterministik olarak yeniden train/validation/test splitlerine ayrıldı.

## Etiket eşlemesi

- `ilk yardım` → `medikal`
- Telsiz, nöbet, harita, mevzi, KBRN ve angajman alt sınıfları → `savaş yönergeleri`
- `matematik` → `matematik`
- `sohbet` → `sohbet`
- `saat` → eğitim dışında, kural tabanlı yol

## Dağılım

| Sınıf | Toplam | Train | Validation | Test |
|---|---:|---:|---:|---:|
| medikal | 167 | 117 | 25 | 25 |
| savaş yönergeleri | 528 | 370 | 79 | 79 |
| matematik | 182 | 127 | 27 | 28 |
| sohbet | 151 | 106 | 23 | 22 |
| **Toplam** | **1028** | **720** | **154** | **154** |

## Kaynak ve kalite notları

- Eski paketten 628 kayıt, yeni JSONL dosyasından 400 kayıt alındı.
- Yeni 400 kayıtta parse hatası, boş soru, normalize tekrar veya eski paketle normalize-birebir çakışma bulunmadı.
- Eski audit kararları korundu: 529 `keep`, 99 `revise`. Yeni kayıtlar bağımsız insan/LLM incelemesinden geçirilmedi ve `provided` olarak işaretlendi.
- Public CSV dosyaları yalnızca `soru,sinif` kolonlarını içerir. Alt sınıf, kaynak ve karar bilgileri `audit/review_audit.csv` içindedir.
- Splitler yeniden üretildi; eski test splitinin üyeliği korunmadı.
- Test seti eğitim veya model seçimi için kullanılmamalıdır.
