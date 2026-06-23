# Sermaye Piyasalarında Dinamik Fiyat Tahmini

Bu proje, “Sermaye Piyasalarında Dinamik Fiyat Tahmini İçin Yeni Nesil Yapay Zeka Yaklaşımları” adlı bitirme tezinin interaktif Streamlit web uygulamasıdır.

## Proje özeti

Uygulama, Apple Inc. (`AAPL`) hissesine ait geçmiş piyasa verilerini ve tez çalışması sırasında kaydedilmiş model sonuçlarını kullanıcı dostu bir arayüzde gösterir. Production uygulaması model eğitimi veya canlı çıkarım yapmaz; yalnızca doğrulanmış yerel CSV, JSON ve PNG sonuç dosyalarını okur.

## Akademik amaç

Bu çalışma eğitim ve araştırma amacıyla hazırlanmıştır. Ekranda gösterilen tahminler ve performans metrikleri yatırım tavsiyesi değildir; finansal karar verme amacıyla kullanılmamalıdır.

## Kullanılan modeller

- LSTM: Direct Multi-Output yaklaşımıyla beş günlük tahmin.
- TFT Tabanlı Model: TFT tabanlı sadeleştirilmiş mimari.
- Chronos: `amazon/chronos-t5-small` ile zero-shot çıkarım ve 0.10–0.90 tahmin aralığı.

Naive Baseline tez kapsamına dahil olmadığı için uygulama arayüzüne ve aktif sonuç dosyalarına aktarılmamıştır.

## Uygulama özellikleri

- AAPL piyasa verisi analizi.
- Beş günlük sabit tez tahminlerinin gösterimi.
- LSTM, TFT Tabanlı Model ve Chronos karşılaştırması.
- Chronos için medyan tahmin ve tahmin aralığı görselleştirmesi.
- Metodoloji ve proje açıklama sayfaları.
- Sunum Modu ile geniş ekran sunuma uygun görünüm.
- Yerel CSV fallback desteği.

## Proje klasör yapısı

```text
capital_market_ai_app/
├── app.py
├── pages/
├── utils/
├── data/
├── results/
├── assets/
├── tests/
├── scripts/
├── requirements.txt
├── PROJECT_PLAN.md
├── AGENTS.md
└── DEPLOYMENT_CHECKLIST.md
```

`imports/` klasörü tez çıktılarının projeye aktarımı için kullanılan kaynak dosyaları içerir. Uygulama standart `data/`, `results/`, `assets/`, `pages/` ve `utils/` dosyalarından çalıştığı için production arayüzünün açılması açısından `imports/` klasörü zorunlu değildir.

## Yerel kurulum

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uygulamayı çalıştırma

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Bütünlük testlerini çalıştırma

Tek komutluk proje kontrolü:

```powershell
.\.venv\Scripts\python.exe scripts\validate_project.py
```

Unittest testleri:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Model sonuçlarının kaynağı

Uygulama, tez çalışması sırasında üretilmiş ve standart klasörlere aktarılmış sonuçları okur:

- `data/model_results.json`
- `results/predictions/`
- `results/metrics/`
- `results/plots/`

Sonuçlar uygulama açılırken yeniden hesaplanmaz.

## Test dönemlerindeki farklılık

LSTM ve TFT Tabanlı Model aynı rolling test döneminde değerlendirilmiştir. Chronos zero-shot deneyi daha geniş bir test aralığını kapsar ve bu dönem farkı uygulamada akademik notlarla belirtilir. Chronos rolling test verisi LSTM/TFT dönemine göre kırpılmamıştır.

## Sunum modu

Her sayfanın kenar çubuğunda “Sunum Modu” anahtarı bulunur. Bu mod başlıkları, kartları ve grafikleri geniş ekran sunuma daha uygun hâle getirir. Akademik uyarılar ve deney dönemi açıklamaları gizlenmez.

## Deployment hazırlığı

Production bağımlılıkları `requirements.txt` içinde doğrudan kullanılan hafif paketlerle sınırlandırılmıştır. TensorFlow, PyTorch, Chronos, Transformers veya benzeri ağır model paketleri production uygulamasına eklenmemiştir.

Deployment öncesinde:

- `scripts\validate_project.py` başarıyla çalıştırılmalı.
- `DEPLOYMENT_CHECKLIST.md` gözden geçirilmeli.
- Secrets veya API anahtarı bulunmadığı doğrulanmalı.
- Deployment sonrasında altı sayfa ve üç model grafiği manuel olarak kontrol edilmeli.

## Sorumluluk reddi

Bu uygulamadaki içerikler yalnızca akademik ve deneysel amaçlıdır. Gösterilen tahminler gelecekteki piyasa hareketlerini garanti etmez ve yatırım tavsiyesi niteliği taşımaz.
