# Model Dosyaları

## Güncel (Sınıflandırma)

| Dosya | Açıklama |
|-------|----------|
| `best_classifier_rf.pkl` | Random Forest Classifier — production modeli (`app.py`) |
| `preprocessing_pipeline_scaler.pkl` | RobustScaler pipeline |

## Opsiyonel (Hibrit demo — `archive/app_hybrid.py`)

| Dosya | Açıklama |
|-------|----------|
| `audio_classifier_lr.pkl` | Logistic Regression (sadece audio özellikleri) |
| `genre_priors.json` | Tür başına yüksek popülerlik önceliği |
| `hybrid_metrics.json` | Hibrit model test metrikleri |

## Arşiv (Regresyon — kullanmayın)

| Dosya | Açıklama |
|-------|----------|
| `best_model_clean.pkl` | Bagging Regressor, R²=0.5694 |
| `best_model_optimized.pkl` | Leaky optimized regressor |
| `final_model.pkl` | İlk baseline regressor |

Eğitim ve metrikler: `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb`
