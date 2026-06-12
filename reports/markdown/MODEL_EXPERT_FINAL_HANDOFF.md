# MODEL EXPERT — FINAL HANDOFF REPORT (Sınıflandırma)

**Tarih:** Haziran 2026 (regresyon → sınıflandırma geçişi)  
**Veri Seti:** Spotify Music Dataset (114.000 şarkı)  
**Problem:** **İkili sınıflandırma** (Düşük / Yüksek popülerlik)  
**Kaynak:** `notebooks/Spotify_ML_Pipeline_Reorganized_v2.ipynb`  
**Eğitilen Model Sayısı:** 14 sınıflandırıcı karşılaştırıldı  

---

## YÖNETİCİ ÖZETİ

### Hedef Değişken Dönüşümü

| Ham popularity | Sınıf | Etiket |
|----------------|-------|--------|
| 0 – 49 | 0 | Düşük Popüler |
| 50 – 100 | 1 | Yüksek Popüler |

```python
y = np.where(popularity < 50, 0, 1)
```

### Veri Durumu

- **Train:** 91.200 × 131 feature  
- **Test:** 22.800 × 131 feature  
- **Özellikler:** 18 sayısal + engineered + 114 genre one-hot  
- **Leakage:** Yok (index sütunu çıkarıldı)  
- **Scaling:** RobustScaler (DataPrep pipeline)  
- **Sınıf dağılımı (train):** %74.2 Düşük / %25.8 Yüksek  

### Final Model

**Random Forest Classifier** — `class_weight='balanced'`, GridSearchCV ile optimize  

**Dosya:** `models/best_classifier_rf.pkl`

### Test Seti Performansı

| Metrik | Değer |
|--------|-------|
| **Accuracy** | 0.853 |
| **F1-Score (weighted)** | 0.842 |
| **F1-Score (macro)** | 0.780 |
| **ROC-AUC** | 0.902 |
| **Precision (Düşük / Yüksek)** | 0.86 / 0.82 |
| **Recall (Düşük / Yüksek)** | 0.96 / 0.54 |

---

## MODEL KARŞILAŞTIRMASI (14 Model)

Notebook'ta tüm modeller `class_weight='balanced'` (veya eşdeğeri) ile test edildi.

| Sıra | Model | Accuracy | F1 (weighted) |
|------|-------|----------|---------------|
| 1 | Ridge Classifier | ~0.779 | ~0.758 |
| 2 | Logistic Regression | ~0.778 | ~0.756 |
| … | … | … | … |
| **Şampiyon** | **Random Forest** | **0.853** | **0.842** |

Tam tablo notebook Hücre 35 çıktısında.

---

## HİPERPARAMETRE OPTİMİZASYONU

- **Yöntem:** GridSearchCV + StratifiedKFold (3-fold)  
- **Scoring:** `f1_weighted`  
- **Parametre ızgarası:** `n_estimators`, `max_depth`, `min_samples_split`, `max_features`  
- **Sonuç:** `grid_search.best_params_` → final modele aktarıldı  

---

## ÇAPRAZ DOĞRULAMA

- **Yöntem:** 5-fold StratifiedKFold  
- **Model:** Random Forest Classifier  
- **Metrikler:** accuracy, f1_weighted, precision_weighted, recall_weighted  

---

## BEKLENTİ YÖNETİMİ

### Neden Yüksek Sınıf Recall Düşük?

1. **Sınıf dengesizliği:** Yüksek popüler şarkılar veri setinin yalnızca ~%26'sı  
2. **Audio features zayıf sinyal:** EDA'da popularity ile korelasyon < 0.1  
3. **Genre en güçlü özellik:** Tür bilgisi sınıflandırmada belirleyici  
4. **External faktörler eksik:** Sanatçı ünü, playlist sayısı vb. yok  

### Eski Regresyon Pipeline (Arşiv)

Önceki çalışmada Bagging Regressor ile R² = 0.5694 elde edilmişti. Proje gereksinimi **sınıflandırma** olduğu için bu model arşivlendi (`models/best_model_clean.pkl`). Detay: `scripts/ARCHIVED_REGRESSION_PIPELINE.md`

---

## DEPLOYMENT NOTU

- Streamlit (`app.py`) henüz sınıflandırmaya uyarlanmadı  
- Deployment hedefi: sınıf tahmini + olasılık skoru + confusion matrix  
- Güncel deployment planı: `DEPLOYMENT_EXPERT_FINAL_HANDOFF.md`  

---

## SONRAKİ ADIMLAR

1. ✅ Notebook sınıflandırma pipeline'ı tamamlandı  
2. ✅ Raporlar hizalandı  
3. ⏳ Streamlit uyarlaması  
4. ⏳ SHAP / açıklanabilirlik (sınıflandırma için güncelleme)  
