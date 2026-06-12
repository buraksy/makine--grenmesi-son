# Spotify Popülerlik Sınıflandırma Projesi

**Kaynak doğruluk:** `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb`  
**Son güncelleme:** Haziran 2026  
**Durum:** Modelleme, raporlar ve Streamlit (`app.py`) sınıflandırmaya göre hizalandı

---

## Proje Özeti

114.000 Spotify şarkısı üzerinde **ikili sınıflandırma** yapılır: bir şarkının popülerliği **Düşük** mü **Yüksek** mi?

| Sınıf | Etiket | Popularity aralığı |
|-------|--------|-------------------|
| 0 | Düşük Popüler | 0 – 49 |
| 1 | Yüksek Popüler | 50 – 100 |

---

## Final Model

| Özellik | Değer |
|---------|-------|
| **Algoritma** | Random Forest Classifier |
| **Özellik sayısı** | 131 (18 sayısal + engineered + 114 genre one-hot) |
| **Eğitim örnekleri** | 91.200 |
| **Test örnekleri** | 22.800 |
| **Sınıf dengesizliği** | ~%74 Düşük / ~%26 Yüksek |
| **class_weight** | balanced |
| **Model dosyası** | `models/best_classifier_rf.pkl` |
| **Notebook** | `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb` |

### Test Seti Performansı (Random Forest, `n_estimators=200`)

| Metrik | Değer |
|--------|-------|
| Accuracy | 0.853 |
| F1-Score (weighted) | 0.842 |
| F1-Score (macro) | 0.780 |
| ROC-AUC | 0.902 |
| Düşük sınıf recall | 0.96 |
| Yüksek sınıf recall | 0.54 |

> Azınlık sınıf (Yüksek Popüler) recall'u düşüktür; bu dengesiz veri setinde beklenen bir durumdur. `class_weight='balanced'` ve StratifiedKFold ile hafifletilmiştir.

---

## Proje Yapısı

```
dataset-analysis/
├── notebooks/
│   └── Spotify_ML_Pipeline_Reorganized_v2.ipynb   # ← Ana pipeline (EDA + modelleme)
├── data/
│   ├── raw/dataset.csv
│   └── model_ready/
│       ├── X_train_clean.csv    # Özellikler (131 sütun)
│       ├── X_test_clean.csv
│       ├── y_train.csv            # Ham popularity (0-100); modellemede binarize edilir
│       └── y_test.csv
├── models/
│   ├── best_classifier_rf.pkl     # ✅ Güncel sınıflandırma modeli
│   ├── preprocessing_pipeline_scaler.pkl
│   └── best_model_clean.pkl       # ⚠️ ARŞİV — eski regresyon modeli
├── reports/markdown/              # Güncellenmiş raporlar
├── modules/                       # Streamlit yardımcı modülleri (opsiyonel)
├── scripts/                       # Pipeline scriptleri (regresyon scriptleri arşiv notlu)
├── archive/                       # Eski uygulamalar + app_hybrid.py (demo)
└── app.py                         # ✅ Güncel RF sınıflandırma dashboard
```

---

## Hızlı Başlangıç

```bash
cd dataset-analysis
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb
```

---

## Önemli Notlar

1. **EDA raporları** popularity'yi sürekli değişken olarak analiz eder; bu geçerlidir. Modelleme aşamasında ikili sınıfa dönüştürülür.
2. **`y_train.csv` / `y_test.csv`** ham popularity skorlarını içerir. Binarize işlemi notebook'taki `binarize_popularity()` ile yapılır.
3. **Eski regresyon dosyaları** (`best_model_clean.pkl`, `scripts/model_training_*.py`, `archive/app_old.py` vb.) arşivlenmiştir.
4. **Streamlit:** `streamlit run app.py` — Random Forest sınıflandırma dashboard'u.
5. **Hibrit demo (opsiyonel):** `streamlit run archive/app_hybrid.py` — LR + tür önceliği; ana model değildir.

---

## Raporlar

| Dosya | İçerik |
|-------|--------|
| `reports/markdown/EDA_FINAL_REPORT.md` | Keşifsel veri analizi |
| `reports/markdown/DATAPREP_MODEL_EXPERT_HANDOFF.md` | Veri hazırlama |
| `reports/markdown/MODEL_EXPERT_FINAL_HANDOFF.md` | Model seçimi ve metrikler |
| `reports/markdown/DEPLOYMENT_EXPERT_FINAL_HANDOFF.md` | Deployment planı |
| `reports/markdown/phase7_modeling_strategy.md` | Modelleme stratejisi |
| `README_DEPLOYMENT.md` | Streamlit deployment rehberi |
