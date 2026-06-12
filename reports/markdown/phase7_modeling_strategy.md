# Önerilen Modelleme Stratejisi (Sınıflandırma)

**Güncelleme:** Haziran 2026  
**Kaynak:** `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb`

---

## 1. HEDEF

**İkili sınıflandırma:** Şarkının popülerliği Düşük mü, Yüksek mi?

| Popularity | Sınıf | Etiket |
|------------|-------|--------|
| 0 – 49 | 0 | Düşük Popüler |
| 50 – 100 | 1 | Yüksek Popüler |

---

## 2. KRİTİK UYARILAR

- Audio features ile popularity arasında zayıf ilişki (korelasyon < 0.1) — EDA bulgusu geçerli  
- **track_genre** en güçlü özellik — one-hot encoding zorunlu  
- Sınıf dengesizliği: ~%74 Düşük / ~%26 Yüksek → `class_weight='balanced'` kullan  
- Ham `y_train.csv` popularity skorları içerir; binarize notebook'ta yapılır  

---

## 3. ÖNERİLEN MODELLER

| Kategori | Modeller |
|----------|----------|
| Linear | Logistic Regression, Ridge Classifier, SGD Classifier, Linear SVC, LDA |
| Tree | Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost |
| Boosting | XGBoost, LightGBM |
| Diğer | KNN, Naive Bayes |

**Şampiyon model:** Random Forest Classifier (GridSearchCV + `class_weight='balanced'`)

---

## 4. VERİ HAZIRLIĞI

- ✅ `X_train_clean.csv` / `X_test_clean.csv` — 131 özellik, leakage temiz  
- ✅ RobustScaler uygulanmış  
- ✅ Genre one-hot (114 sütun)  
- ⚠️ Train-test split regresyon döneminden kaldı; CV için **StratifiedKFold** kullan  
- İleride: stratified yeniden split önerilir  

---

## 5. DEĞERLENDİRME METRİKLERİ

| Metrik | Açıklama |
|--------|----------|
| **Accuracy** | Genel doğruluk |
| **F1-Score (weighted)** | Dengesiz sınıflar için ana metrik |
| **F1-Score (macro)** | Sınıf başına eşit ağırlık |
| **Precision / Recall** | Sınıf bazlı (özellikle Yüksek sınıf recall) |
| **ROC-AUC** | Ayırt etme gücü |
| **Confusion Matrix** | FP / FN analizi |

**Gerçekleşen test metrikleri (Random Forest):** Accuracy 0.853 · F1 weighted 0.842 · ROC-AUC 0.902

---

## 6. CROSS-VALIDATION

- **StratifiedKFold** (k=5) — sınıf oranını korur  
- Scoring: `f1_weighted`, `accuracy`  
- Hiperparametre: GridSearchCV, 3-fold StratifiedKFold  

---

## 7. BEKLENTİ YÖNETİMİ

- Yüksek popüler sınıf recall'u (~0.54) düşük kalabilir — azınlık sınıf  
- Genre ve audio features ile %85 accuracy elde edildi; regresyon R² beklentileri artık geçerli değil  
- İyileştirme: external features, threshold tuning, SMOTE  

---

## 8. ARŞİV

Eski regresyon stratejisi (R², RMSE, Bagging Regressor) → `scripts/ARCHIVED_REGRESSION_PIPELINE.md`
