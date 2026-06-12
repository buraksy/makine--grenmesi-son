# ⚠️ ARŞİV — Regresyon Pipeline Scriptleri

**Durum:** Kullanımdan kaldırıldı (Haziran 2026)

Bu klasördeki aşağıdaki scriptler **popularity skorunu 0–100 arası sürekli değer olarak tahmin eden** eski regresyon pipeline'ına aittir:

- `data_preparation.py` — regresyon hedefli split (stratify yok)
- `model_training_18plus.py`, `model_training_22plus_extended.py`
- `hyperparameter_optimization.py`, `fix_data_leakage_retrain.py`
- `explainability_analysis.py` — SHAP regresyon odaklı
- `phase1_data_overview.py` … `phase7_model_readiness.py` — EDA scriptleri (analiz geçerli, regresyon önerileri güncel değil)

## Güncel kaynak

| Ne | Nerede |
|----|--------|
| EDA + modelleme + değerlendirme | `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb` |
| Proje özeti ve metrikler | `README.md` |
| Production modeli | `models/best_classifier_rf.pkl` |

Scriptleri yeniden çalıştırmayın; notebook ile çelişen regresyon çıktıları üretirler.
