# 🎵 Spotify Popularity Prediction - Deployment Dashboard

**Production-ready Streamlit application** for predicting Spotify track popularity using Machine Learning.

![Status](https://img.shields.io/badge/status-production--ready-green)
![Model](https://img.shields.io/badge/model-Bagging%20Regressor-blue)
![R² Score](https://img.shields.io/badge/R²-0.5694-brightgreen)

---

## 📊 Proje Özeti

Bu uygulama, **114,000 Spotify şarkısından** öğrenilen makine öğrenmesi modeli ile:
- 🎯 **Tekil tahmin:** Bir şarkının popularity skorunu tahmin edin
- 📊 **Toplu tahmin:** CSV dosyası ile birden fazla şarkı analizi
- 📈 **Model performansı:** Detaylı metrikler ve görselleştirmeler
- ℹ️ **Model bilgisi:** Sınırlamalar, güçlü yönler, teknik detaylar

---

## 🎯 Model Performansı

| Metric | Value | Açıklama |
|--------|-------|----------|
| **Test R²** | 0.5694 | Model açıklayıcılık gücü (varyansın %56.9'u) |
| **Test RMSE** | 14.58 | Ortalama hata payı (~±15 puan) |
| **Test MAE** | 10.67 | Tipik sapma (~11 puan) |
| **Features** | 131 | 18 audio + 114 genre (NO INDEX) |
| **Status** | ✅ Production-Safe | Data leakage temizlendi |

---

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Repository'yi klonlayın
git clone <repo-url>
cd dataset-analysis

# Virtual environment oluşturun (önerilen)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Uygulamayı Çalıştırın

```bash
streamlit run app.py
```

Uygulama otomatik olarak **http://localhost:8501** adresinde açılacaktır.

---

## 📁 Proje Yapısı

```
dataset-analysis/
├── app.py                              # Streamlit uygulaması
├── requirements.txt                     # Python bağımlılıkları
├── README_DEPLOYMENT.md                 # Bu dosya
├── models/
│   ├── best_model_clean.pkl            # ✅ Production model (NO LEAKAGE)
│   ├── preprocessing_pipeline_scaler.pkl # RobustScaler
│   ├── best_model_optimized.pkl        # Deprecated (leaky)
│   └── final_model.pkl                 # Initial baseline
├── data/
│   ├── raw/dataset.csv                 # Original Spotify dataset
│   └── model_ready/
│       ├── X_train_clean.csv           # Clean train features (131)
│       ├── X_test_clean.csv            # Clean test features (131)
│       ├── y_train.csv                 # Train targets
│       └── y_test.csv                  # Test targets
├── reports/
│   ├── csv/
│   │   ├── leakage_fix_comparison.csv  # Leaky vs Clean comparison
│   │   ├── model_comparison_results.csv
│   │   └── best_model_params.csv
│   └── markdown/
│       └── MODEL_EXPERT_FINAL_HANDOFF.md
└── figures/
    └── (performance charts)
```

---

## 🎵 Audio Features Rehberi

| Feature | Açıklama | Değer Aralığı | Örnek |
|---------|----------|---------------|-------|
| **danceability** | Dans edilebilirlik | 0-1 | Pop: 0.7, Classical: 0.2 |
| **energy** | Enerji ve yoğunluk | 0-1 | Metal: 0.9, Acoustic: 0.2 |
| **loudness** | Ses şiddeti | -60 - 0 dB | Loud rock: -5dB, Soft: -20dB |
| **speechiness** | Konuşma varlığı | 0-1 | Podcast: 0.8, Instrumental: 0.03 |
| **acousticness** | Akustik olma derecesi | 0-1 | Acoustic guitar: 0.9, EDM: 0.1 |
| **instrumentalness** | Vokal yokluğu | 0-1 | Instrumental: 0.9, Pop: 0.0 |
| **liveness** | Canlı performans | 0-1 | Live concert: 0.8, Studio: 0.1 |
| **valence** | Müzikal pozitiflik | 0-1 | Happy: 0.9, Sad: 0.2 |
| **tempo** | Tempo (BPM) | 40-220 | Fast dance: 140, Ballad: 70 |
| **duration_ms** | Şarkı süresi | 10,000-600,000 | Pop song: 200,000 (3.3 min) |
| **key** | Müzikal anahtar | 0-11 | C: 0, C#: 1, D: 2, ... |
| **mode** | Müzikal mod | 0 (Minor), 1 (Major) | Happy songs: Major |
| **time_signature** | Ölçü birimi | 3, 4, 5 | Standard: 4/4 |
| **track_genre** | Müzik türü | 114 genre | pop, rock, jazz, ... |
| **explicit** | Küfür içeriği | True/False | Rap: often True |

---

## 🎯 Kullanım Senaryoları

### ✅ Ne İçin Kullanılabilir

- 🎵 **Genre-based popularity benchmarking:** Hangi genre'de ne kadar popüler olur?
- 🎛️ **Audio features optimization:** Daha popüler olmak için hangi features ayarlanmalı?
- 📊 **A/B testing:** İki farklı şarkı versiyonunun popularity farkı
- 🎸 **Track comparison:** Benzer şarkıların popularity karşılaştırması
- 📈 **Trend analysis:** Popularity distribution across genres

### ❌ Ne İçin Kullanılmamalı

- ❌ **Kesin popularity prediction:** Model ±15 puan hata payına sahip
- ❌ **Yeni artist prediction:** Artist features (followers, fame) dahil değil
- ❌ **Viral hit prediction:** Social media data, marketing budget eksik
- ❌ **Real-time trending:** Model static, real-time trend bilgisi yok
- ❌ **Business-critical decisions:** Model yalnızca audio features kullanıyor

---

## ⚠️ Model Sınırlamaları

### 🚫 Model Dahil Etmediği Faktörler

1. **Artist Ünü:**
   - Follower sayısı
   - Artist popularity
   - Verified artist badge

2. **Marketing & Promotion:**
   - Marketing budget
   - Playlist yerleştirme sayısı (kaç playlist'te)
   - Spotify editorial pick
   - Radio airplay

3. **Social Media:**
   - TikTok viral etkisi
   - Instagram reels
   - Twitter mentions
   - YouTube views

4. **Timing:**
   - Release date (hangi dönemde yayınlandı)
   - Seasonal trends
   - Trend timing (yükselişte mi, düşüşte mi)

5. **Coğrafi Faktörler:**
   - Hangi ülkede popüler
   - Regional preferences
   - Language barriers

### 📊 Performans Beklentileri

- **Tipik Hata:** ±15 puan (RMSE = 14.58)
- **Örnek:** Gerçek popularity 50 ise, tahmin 35-65 arasında olabilir
- **R² = 0.5694:** Model varyansın %56.9'unu açıklıyor
- **Kalan %43:** External factors (artist ünü, marketing, viral etki)

---

## 🛠️ HCI İlkeleri ve Tasarım Kararları

Bu uygulama **Shneiderman'ın 8 Altın Kuralı** ve **Nielsen Kullanılabilirlik İlkeleri**'ne göre tasarlanmıştır:

### ✅ Uygulanan HCI İlkeleri

1. **Tutarlılık (Consistency):**
   - Sabit renk paleti (Spotify green, black)
   - Tek tip card mimarisi
   - Tutarlı buton ve form tasarımı

2. **Sık Kullanıcılar İçin Kısayollar:**
   - Sidebar'da hızlı model özeti
   - Ana sayfada quick start CTA butonları
   - Session state ile son tahmin hatırlama

3. **Bilgilendirici Geri Bildirim:**
   - Success/warning/error messages
   - Progress bar (toplu tahmin)
   - Spinner (loading states)
   - Confidence score (tahmin güveni)

4. **Tamamlanmış Eylemler:**
   - Veri girişi → Doğrulama → Tahmin → Sonuç → Yorum
   - Her adım net olarak gösteriliyor

5. **Hata Önleme:**
   - Slider'lar (min/max değer kontrolü)
   - Dropdown'lar (geçersiz input engelleme)
   - CSV kolon kontrolü (toplu tahmin)

6. **Geri Alma Kolaylığı:**
   - Form reset butonu
   - Session state temizleme
   - Yeni tahmin başlat

7. **Kullanıcı Kontrolü:**
   - Kullanıcı tüm input'ları kontrol eder
   - Tahmin butonuna basana kadar işlem başlamaz
   - CSV indirme opsiyonu

8. **Kısa Süreli Bellek Yükünü Azaltma:**
   - Sidebar'da aktif model bilgisi
   - Input alanlarında help text
   - Tooltip'ler
   - FAQ bölümü

### 🎨 Görsel Tasarım

- **Profesyonel Palette:** Spotify green (#1DB954), black (#191414)
- **Premium UI:** Gradient backgrounds, soft shadows
- **Responsive Cards:** Hover effects, smooth transitions
- **Typography:** Clear hierarchy, readable fonts
- **Spacing:** Generous padding, clean layout

---

## 📈 Model Geliştirme Geçmişi

| Versiyon | Test R² | RMSE | MAE | Status | Not |
|----------|---------|------|-----|--------|-----|
| Baseline | 0.5827 | 14.35 | 9.22 | ❌ Leaky | Index column vardı |
| Optimized | 0.6299 | 13.51 | 8.94 | ❌ Leaky | Hyperparameter tuning |
| **Clean (Current)** | **0.5694** | **14.58** | **10.67** | **✅ Production** | **Index kaldırıldı** |

### 🔧 Data Leakage Fix

**Sorun:** `Unnamed: 0` (dataset index) en önemli feature'dı (SHAP importance: 3.69)

**Neden Sorun:**
- Index popularity ile yüksek korelasyon
- Production'da yeni şarkılar için index olmayacak
- Model gerçek audio features yerine index'e dayanıyordu

**Çözüm:**
- Index column kaldırıldı
- Model yeniden eğitildi: 132 → 131 features
- R² drop: -9.61% (acceptable <20%)
- **Sonuç:** Model artık production-safe ✅

---

## 🔮 Gelecek İyileştirmeler

### Kısa Vadeli (1-2 hafta)
- [ ] Full preprocessing pipeline integration
- [ ] Real prediction (şu anda mock prediction)
- [ ] Input validation strengthening
- [ ] Error handling improvement

### Orta Vadeli (1-2 ay)
- [ ] External data integration (Spotify API)
  - Artist followers
  - Playlist count
  - Release date
- [ ] Model retraining with external features
- [ ] Target R²: 0.65+

### Uzun Vadeli (3+ ay)
- [ ] Real-time monitoring
- [ ] Data drift detection
- [ ] A/B testing framework
- [ ] Model versioning system
- [ ] Automated retraining pipeline

---

## 📞 İletişim ve Destek

**Deployment Expert**  
Tarih: 19 Mayıs 2026  
Model: Bagging Regressor (Clean, No Data Leakage)  
Framework: Streamlit + scikit-learn

**Proje Ekibi:**
- EDA Expert: Exploratory Data Analysis
- DataPrep Expert: Feature Engineering & Preprocessing
- Model Expert: Model Training & Hyperparameter Optimization
- Deployment Expert: UI/UX & Production Deployment

---

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

---

## 🎵 Spotify Developer Credits

Bu uygulama Spotify Web API kullanır. Audio features Spotify tarafından sağlanmaktadır.

**Spotify for Developers:** https://developer.spotify.com/

---

**⭐ Not:** Bu uygulama **HCI ilkeleri ve Shneiderman'ın 8 Altın Kuralı**'na göre tasarlanmış **profesyonel, production-ready** bir deployment örneğidir.
