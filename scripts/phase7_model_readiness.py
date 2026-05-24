"""
PHASE 7: MODEL READINESS ASSESSMENT
Amaç: Verinin modelleme aşamasına hazır olup olmadığını değerlendirmek
"""

import os
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# Klasörlerin varlığını garantile
Path('../reports/csv').mkdir(parents=True, exist_ok=True)
Path('../reports/markdown').mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 7: MODEL HAZIRLIK DEĞERLENDİRMESİ")
print("="*80)

# Analiz raporlarını yükle
phase1_summary = pd.read_csv('../reports/csv/phase1_data_overview_summary.csv')
phase5_missing = pd.read_csv('../reports/csv/phase5_missing_values_summary.csv') if os.path.exists('../reports/csv/phase5_missing_values_summary.csv') else pd.DataFrame()
phase5_outliers = pd.read_csv('../reports/csv/phase5_outlier_analysis.csv')
phase6_recommendations = pd.read_csv('../reports/csv/phase6_all_data_prep_recommendations.csv', encoding='utf-8-sig')

print("\n" + "="*80)
print("MODEL HAZIRLIK KONTROL LİSTESİ")
print("="*80)

readiness_checklist = []

# 1. Eksik Veri Yönetimi
print("\n1. EKSİK VERİ YÖNETİMİ")
if len(phase5_missing) == 0 or phase5_missing['Eksik Değer Oranı (%)'].max() < 1:
    eksik_status = "✓ HAZıR"
    eksik_aciklama = "Eksik veri oranı ihmal edilebilir seviyede (%0.0). Doğrudan modellemeye geçilebilir."
    eksik_oneri = "Minimal müdahale: Eksik 3 satır silinebilir veya forward fill uygulanabilir."
else:
    eksik_status = "⚠ KISMI HAZIR"
    eksik_aciklama = "Eksik veri mevcut, imputasyon stratejisi gerekli."
    eksik_oneri = "KNN imputasyon, MICE veya domain bazlı doldurma uygulanmalı."

readiness_checklist.append({
    "Kriter": "Eksik Veri Yönetimi",
    "Durum": eksik_status,
    "Açıklama": eksik_aciklama,
    "Öneri": eksik_oneri
})
print(f"  {eksik_status}")
print(f"  {eksik_aciklama}")

# 2. Encoding Gereksinimi
print("\n2. ENCODING GEREKSİNİMİ")
encoding_status = "⚠ İŞLEM GEREKLİ (KRİTİK)"
encoding_aciklama = "track_genre (EN GÜÇLÜ FEATURE) encoding gerektiriyor. explicit binary encoding."
encoding_oneri = "track_genre: One-Hot veya Target Encoding (KRİTİK - popularity'nin en güçlü açıklayıcısı). explicit: Binary Encoding. artists/album_name çıkarılmalı."

readiness_checklist.append({
    "Kriter": "Encoding Gereksinimi",
    "Durum": encoding_status,
    "Açıklama": encoding_aciklama,
    "Öneri": encoding_oneri
})
print(f"  {encoding_status}")
print(f"  {encoding_aciklama}")

# 3. Scaling Gereksinimi
print("\n3. SCALING GEREKSİNİMİ")
scaling_status = "⚠ İŞLEM GEREKLİ"
scaling_aciklama = "Sayısal değişkenler farklı ölçeklerde (duration_ms: 0-5M, danceability: 0-1). Scaling gerekli."
scaling_oneri = "StandardScaler (genel) veya RobustScaler (outlier'lar için) uygulanmalı. Tree-based modellerde opsiyonel."

readiness_checklist.append({
    "Kriter": "Scaling Gereksinimi",
    "Durum": scaling_status,
    "Açıklama": scaling_aciklama,
    "Öneri": scaling_oneri
})
print(f"  {scaling_status}")
print(f"  {scaling_aciklama}")

# 4. Outlier İşleme
print("\n4. OUTLIER İŞLEME")
high_outliers = phase5_outliers[phase5_outliers['Outlier Oranı (%)'] > 10]
if len(high_outliers) > 0:
    outlier_status = "⚠ İŞLEM ÖNERİLİR"
    outlier_aciklama = f"{len(high_outliers)} değişkende >%10 outlier (instrumentalness, speechiness, time_signature)."
    outlier_oneri = "Robust scaling veya winsorization önerilir. Tree-based modellerde outlier etkisi düşüktür."
else:
    outlier_status = "✓ HAZIR"
    outlier_aciklama = "Kritik seviyede outlier sorunu yok."
    outlier_oneri = "Outlier işleme opsiyonel."

readiness_checklist.append({
    "Kriter": "Outlier İşleme",
    "Durum": outlier_status,
    "Açıklama": outlier_aciklama,
    "Öneri": outlier_oneri
})
print(f"  {outlier_status}")
print(f"  {outlier_aciklama}")

# 5. Target Distribution (Regression)
print("\n5. TARGET DISTRIBUTION (Hedef Dağılımı - Regression)")
target_status = "✓ DENGELİ"
target_aciklama = "Popularity (0-100): Ortalaması 33.2, medyan 35.0, skewness 0.046 (neredeyse simetrik). Dengeli dağılım."
target_oneri = "Regresyon problemi - class imbalance sorunu yok. Ancak extreme yüksek değerler az (%0.84 > 80)."

readiness_checklist.append({
    "Kriter": "Target Distribution",
    "Durum": target_status,
    "Açıklama": target_aciklama,
    "Öneri": target_oneri
})
print(f"  {target_status}")
print(f"  {target_aciklama}")

# 6. Leakage Riski
print("\n6. LEAKAGE RİSKİ")
leakage_status = "✓ DÜŞüK RİSK"
leakage_aciklama = "track_id, artists, album_name, track_name - identifier sütunlar. Hedef değişkeni doğrudan temsil etmiyorlar."
leakage_oneri = "ID sütunları (track_id, artists, album_name, track_name) modelden çıkarılmalı."

readiness_checklist.append({
    "Kriter": "Leakage Riski",
    "Durum": leakage_status,
    "Açıklama": leakage_aciklama,
    "Öneri": leakage_oneri
})
print(f"  {leakage_status}")
print(f"  {leakage_aciklama}")

# 7. Train-Test Split Stratejisi
print("\n7. TRAIN-TEST SPLIT STRATEJİSİ")
split_status = "✓ RANDOM SPLIT UYGUN"
split_aciklama = "Regression problemi - popularity sürekli değişken. Stratified split gerekmez."
split_oneri = "Random train-test split (80-20 veya 70-30). Zaman serisi yok, chronological split gerekmez."

readiness_checklist.append({
    "Kriter": "Train-Test Split",
    "Durum": split_status,
    "Açıklama": split_aciklama,
    "Öneri": split_oneri
})
print(f"  {split_status}")
print(f"  {split_aciklama}")

# 8. Multicollinearity
print("\n8. MULTİCOLLİNEARİTY YÖNETİMİ")
multi_status = "⚠ ORTA RİSK"
multi_aciklama = "VIF > 10: tempo, energy, danceability. Korelasyon: energy-loudness (0.76), energy-acousticness (-0.73)."
multi_oneri = "Regularization (Ridge/Lasso) veya PCA. Tree-based modellerde tolerans yüksek."

readiness_checklist.append({
    "Kriter": "Multicollinearity",
    "Durum": multi_status,
    "Açıklama": multi_aciklama,
    "Öneri": multi_oneri
})
print(f"  {multi_status}")
print(f"  {multi_aciklama}")

# Kontrol listesini kaydet
readiness_df = pd.DataFrame(readiness_checklist)
readiness_df.to_csv('../reports/csv/phase7_model_readiness_checklist.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ Model hazırlık kontrol listesi kaydedildi:")
print(f"  ../reports/csv/phase7_model_readiness_checklist.csv")

print("\n" + "="*80)
print("FİNAL DEĞERLENDİRME")
print("="*80)

# Hazırlık skoru hesapla
hazir_count = sum([1 for item in readiness_checklist if "HAZIR" in item['Durum']])
kismi_hazir_count = sum([1 for item in readiness_checklist if "KISMI HAZIR" in item['Durum'] or "İŞLEM" in item['Durum']])
toplam_count = len(readiness_checklist)

hazir_oran = (hazir_count / toplam_count) * 100
kismi_hazir_oran = (kismi_hazir_count / toplam_count) * 100

print(f"\nHazırlık Durumu:")
print(f"  ✓ Hazır: {hazir_count}/{toplam_count} (%{hazir_oran:.1f})")
print(f"  ⚠ İşlem Gerekli/Kısmen Hazır: {kismi_hazir_count}/{toplam_count} (%{kismi_hazir_oran:.1f})")

if hazir_oran >= 70:
    final_verdict = "HAZIR"
    final_color = "YEŞİL"
elif hazir_oran >= 40:
    final_verdict = "KıSMEN HAZIR"
    final_color = "SARI"
else:
    final_verdict = "HAZIR DEĞİL"
    final_color = "KIRMIZI"

print(f"\n" + "="*80)
print(f"GENEL DEĞERLENDİRME: {final_verdict} ({final_color})")
print("="*80)

if final_verdict == "HAZIR":
    print("\n✓ Veri seti modelleme için hazırdır.")
    print("✓ Minimal veri hazırlama adımlarıyla modelleme başlatılabilir.")
    print("✓ Önerilen ilk model: Random Forest / XGBoost (multi-class classification)")
elif final_verdict == "KISMEN HAZIR":
    print("\n⚠ Veri seti modellemeye yakındır ancak ön işleme adımları gereklidir.")
    print("⚠ Encoding, scaling ve ID sütunlarının çıkarılması önceliklidir.")
    print("⚠ Data Prep Expert ile koordinasyon önerilir.")
else:
    print("\n⚠ Veri seti modelleme için yeterli değildir.")
    print("⚠ Ciddi veri hazırlama çalışması gereklidir.")

print("\n" + "="*80)
print("ÖNERİLEN MODELLEME STRATEJİSİ")
print("="*80)

modeling_strategy = """
1. HEDEF: Şarkı Popülaritesi Tahmini (Regression - 0-100 arası popularity skoru)

2. ⚠️ KRİTİK UYARI: Audio Features ile Popularity Arasında Çok Zayıf İlişki!
   - En yüksek korelasyon: loudness (0.05) - bu çok düşük
   - Mevcut feature'larla R² < 0.10 beklenmeli
   - Model başarısı için external features (sanatçı, playlist, yayın tarihi) KRİTİK

3. ÖNERİLEN MODELLER:
   a) Baseline: Linear Regression (R² baseline oluştur)
   b) Regularized: Ridge, Lasso, ElasticNet (düşük korelasyonlar için)
   c) Tree-based: Random Forest Regressor, XGBoost Regressor, LightGBM
   d) Neural Network: MLP Regressor (polynomial relationships test için)
   e) Ensemble: Stacking/Blending (çok küçük iyileştirme sağlayabilir)

4. ÖNCELİKLİ VERİ HAZIRLIĞI:
   - ⚠️ KRİTİK: track_genre ONE-HOT veya TARGET ENCODING (EN GÜÇLÜ FEATURE!)
   - ID sütunlarını çıkar (track_id, artists, album_name, track_name)
   - explicit: Binary encoding (0/1)
   - Sayısal değişkenler: StandardScaler (linear models için) veya RobustScaler
   - Train-test split: Random 80-20
   - ⚠️ Harici feature eklenmesi şiddetle önerilir (artist followers, playlist count, release date)

5. DEĞERLENDİRME METRİKLERİ:
   - R² (Coefficient of Determination) - Model başarısı
   - RMSE (Root Mean Squared Error) - Ortalama hata
   - MAE (Mean Absolute Error) - Mutlak hata
   - MAPE (Mean Absolute Percentage Error) - Yüzdelik hata
   - Residual Plot (hata dağılımı kontrol)
   - ⚠️ Beklenen R²: 0.05-0.15 (çok düşük - audio features yetersiz)

6. FEATURE ENGINEERING FıRSATLARı (Öncelik Sırasıyla):
   a) ⚠️ KRİTİK: track_genre encoding (EN GÜÇLÜ FEATURE - mutlaka ekle)
   b) ⚠️ YÜKSEK: External features (artist takipçi sayısı, playlist sayısı, yayın tarihi)
   c) Düşük: Audio polynomials (loudness², danceability²) - zaten zayıf
   d) Düşük: Mood score (valence x energy) - popularity ile ilişki yok
   e) Düşük: Feature interactions - lineer ilişki zaten yok

7. CROSS-VALİDATİON:
   - K-Fold (k=5 veya k=10) - Stratified gerekmez
   - Regression için random fold assignment yeterli

8. ⚠️ BEKLENTİ YÖNETİMİ:
   - Mevcut audio features ile yüksek accuracy BEKLENMEMELİ
   - Model başarısı düşük olabilir (R² < 0.15)
   - Popülerlik, müzikal özelliklerden çok pazarlama/platform faktörlerine bağlı
   - İyileştirme için artist_name, playlist_count, release_date gibi harici data gerekli
"""

print(modeling_strategy)

# Modelleme stratejisini kaydet
with open('../reports/markdown/phase7_modeling_strategy.md', 'w', encoding='utf-8') as f:
    f.write("# Önerilen Modelleme Stratejisi\n\n")
    f.write(modeling_strategy)

print(f"\n✓ Modelleme stratejisi kaydedildi:")
print(f"  ../reports/markdown/phase7_modeling_strategy.md")

print("\n" + "="*80)
print("PHASE 7 TAMAMLANDI")
print("="*80)
print("\n" + "="*80)
print("TÜM 7 PHASE EDA SÜRECİ BAŞARIYLA TAMAMLANDI!")
print("="*80)
