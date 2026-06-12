# Spotify Popülerlik Sınıflandırma — Deployment Rehberi

**Durum:** Streamlit sınıflandırmaya uyarlandı (`app.py`)  
**Kaynak:** `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb`

![Model](https://img.shields.io/badge/model-Random%20Forest%20Classifier-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-0.85-brightgreen)
![F1](https://img.shields.io/badge/F1%20(weighted)-0.84-brightgreen)

---

## Proje Özeti

114.000 Spotify şarkısı ile eğitilmiş **ikili sınıflandırma** modeli:

| Sınıf | Popularity | Anlam |
|-------|------------|-------|
| 0 | 0 – 49 | Düşük Popüler |
| 1 | 50 – 100 | Yüksek Popüler |

**Streamlit özellikleri (`app.py`):**
- Tekil tahmin: Düşük / Yüksek + güven skoru
- Model bilgisi ve özellik önemi grafiği
- Rastgele doldur / varsayılan butonları

---

## Model Performansı (Test Seti)

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.853 |
| F1-Score (weighted) | 0.842 |
| ROC-AUC | 0.902 |
| Features | 131 |
| Model dosyası | `models/best_classifier_rf.pkl` |

---

## Kurulum

```bash
cd dataset-analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Notebook (güncel pipeline)

```bash
jupyter notebook notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb
```

### Streamlit

```bash
streamlit run app.py
```

Eski uygulama sürümleri ve hibrit demo: `archive/` klasörü.

---

## Proje Yapısı

```
dataset-analysis/
├── notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb  # Ana pipeline
├── app.py                         # ✅ Güncel RF sınıflandırma dashboard
├── README.md                      # Proje kaynak doğruluk belgesi
├── models/
│   ├── best_classifier_rf.pkl     # ✅ Güncel model
│   └── best_model_clean.pkl       # ⚠️ Arşiv (regresyon)
├── data/model_ready/
│   ├── X_train_clean.csv
│   ├── X_test_clean.csv
│   ├── y_train.csv                # Ham popularity; binarize notebook'ta
│   └── y_test.csv
└── reports/markdown/
```

---

## Audio Features Rehberi

| Feature | Açıklama | Aralık |
|---------|----------|--------|
| danceability | Dans edilebilirlik | 0–1 |
| energy | Enerji | 0–1 |
| loudness | Ses şiddeti (dB) | -60 – 0 |
| speechiness | Konuşma oranı | 0–1 |
| acousticness | Akustiklik | 0–1 |
| instrumentalness | Enstrümantal | 0–1 |
| liveness | Canlı performans | 0–1 |
| valence | Pozitiflik | 0–1 |
| tempo | BPM | 0–250+ |
| track_genre | Müzik türü (114 kategori) | one-hot |

---

## Eski Regresyon Dosyaları

| Dosya | Durum |
|-------|-------|
| `models/best_model_clean.pkl` | Arşiv |
| `app_old.py`, `app_new.py`, `app_streamlit.py` | Arşiv / referans |
| `scripts/model_training_*.py` | Arşiv — `scripts/ARCHIVED_REGRESSION_PIPELINE.md` |

Detaylı proje bilgisi: **`README.md`**
