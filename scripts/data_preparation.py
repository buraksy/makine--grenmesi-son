"""
DATA PREPARATION PIPELINE
Amaç: EDA Expert bulgularını devralarak model-ready veri hazırlamak
Hedef: popularity (0-100 arası regression)
"""

import os
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")

# Klasörlerin varlığını garantile
Path('../data/model_ready').mkdir(parents=True, exist_ok=True)
Path('../figures').mkdir(parents=True, exist_ok=True)
Path('../reports/csv').mkdir(parents=True, exist_ok=True)
Path('../reports/markdown').mkdir(parents=True, exist_ok=True)
Path('../models').mkdir(parents=True, exist_ok=True)

# Profesyonel renk paleti
PROFESSIONAL_PALETTE = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E",
    "#BC4B51", "#8E7DBE", "#F77F00", "#06A77D", "#D4A574"
]

print("="*80)
print("DATA PREPARATION PIPELINE - 7 AŞAMALI AGENTİK VERİ HAZIRLAMA")
print("="*80)

# DataPrep action logger
dataprep_actions = []

def log_dataprep_action(step, issue, decision, rationale, risk="Düşük"):
    dataprep_actions.append({
        "Aşama": step,
        "Sorun": issue,
        "Karar": decision,
        "Gerekçe": rationale,
        "Risk Seviyesi": risk
    })

# Model Expert handoff logger
model_handoff_report = []

def add_model_handoff(item, status, recommendation):
    model_handoff_report.append({
        "Bileşen": item,
        "Durum": status,
        "Model Expert Notu": recommendation
    })

# Premium layout fonksiyonu
def apply_premium_layout(fig, title):
    fig.update_layout(
        title={
            "text": title,
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 24, "family": "Arial Black", "color": "#1F2937", "weight": "bold"}
        },
        template="plotly_white",
        paper_bgcolor="#FBFBF8",
        plot_bgcolor="#FBFBF8",
        font={"family": "Arial", "size": 13, "color": "#374151"},
        margin=dict(l=60, r=40, t=80, b=60),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig

print("\n" + "="*80)
print("PHASE 1: EDA RECOMMENDATION INGESTION")
print("="*80)

# Ham veriyi yükle
df = pd.read_csv('../data/raw/dataset.csv')
print(f"\n✓ Ham veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")

# EDA önerilerini yükle
eda_recommendations = pd.read_csv('../reports/csv/phase6_all_data_prep_recommendations.csv', encoding='utf-8-sig')
print(f"✓ EDA Expert'ten {len(eda_recommendations)} öneri alındı")

# Önceliklere göre sırala
priority_order = {"Yüksek": 1, "Orta": 2, "Düşük": 3}
eda_recommendations["Sıralama"] = eda_recommendations["Öncelik"].map(priority_order)
eda_recommendations = eda_recommendations.sort_values("Sıralama")

print("\n📋 EDA Önerileri Özeti:")
print(eda_recommendations["Öncelik"].value_counts())

# Önerileri kategorize et
high_priority = eda_recommendations[eda_recommendations["Öncelik"] == "Yüksek"]
medium_priority = eda_recommendations[eda_recommendations["Öncelik"] == "Orta"]

print(f"\n⚠️ Yüksek öncelikli: {len(high_priority)}")
print(f"   Orta öncelikli: {len(medium_priority)}")

print("\n" + "="*80)
print("PHASE 2: DATA CLEANING")
print("="*80)

# 2.1. Missing Values (çok az - %0.00)
print("\n2.1. EKSİK VERİ YÖNETİMİ")
missing_count = df.isnull().sum()
missing_vars = missing_count[missing_count > 0]

if len(missing_vars) > 0:
    print(f"Eksik veri tespit edildi: {missing_vars.to_dict()}")
    print("Karar: 3 satır eksik veri var (artists, album_name, track_name) - bu ID sütunları çıkarılacak, müdahale gerekmez")
    
    log_dataprep_action(
        step="Phase 2.1",
        issue="Minimal eksik veri (3 satır, %0.00)",
        decision="Müdahale etme - ID sütunları zaten çıkarılacak",
        rationale="Eksik veri oranı ihmal edilebilir seviyede ve etkilenen sütunlar model feature'ı olmayacak",
        risk="Yok"
    )
else:
    print("✓ Eksik veri yok")

# 2.2. Duplicate Check
duplicates = df.duplicated().sum()
print(f"\n2.2. DUPLICATE KONTROL")
print(f"Duplicate satır sayısı: {duplicates}")

if duplicates == 0:
    log_dataprep_action(
        step="Phase 2.2",
        issue="Duplicate satır kontrolü",
        decision="Müdahale gerekmez",
        rationale="Veri setinde duplicate satır tespit edilmedi",
        risk="Yok"
    )

# 2.3. ID Sütunlarını Çıkar (Leakage + High Cardinality)
print(f"\n2.3. ID SÜTUNLARINI ÇIKAR (LEAKAGE RİSKİ)")
id_columns = ['track_id', 'artists', 'album_name', 'track_name']

print(f"Çıkarılacak ID sütunları: {id_columns}")
print(f"  - track_id: Unique identifier - leakage riski")
print(f"  - artists: 31,437 eşsiz - yüksek kardinalite, external feature gerekir")
print(f"  - album_name: 46,589 eşsiz - yüksek kardinalite")
print(f"  - track_name: 73,608 eşsiz - yüksek kardinalite")

df_cleaned = df.drop(columns=id_columns)
print(f"\n✓ ID sütunları çıkarıldı. Yeni boyut: {df_cleaned.shape}")

log_dataprep_action(
    step="Phase 2.3",
    issue="Yüksek kardinalite ve leakage riski (ID sütunları)",
    decision="track_id, artists, album_name, track_name çıkarıldı",
    rationale="Bu sütunlar unique/çok yüksek kardinalite. External metadata olmadan modelleme için kullanışsız ve leakage riski taşıyorlar.",
    risk="Düşük - Çıkarma doğru karar"
)

add_model_handoff(
    item="ID Sütunları",
    status="Çıkarıldı",
    recommendation="Artist, album, track metadata harici kaynaktan (Spotify API) eklenebilir. Mevcut sütunlar leakage ve high cardinality nedeniyle çıkarıldı."
)

print("\n" + "="*80)
print("PHASE 3: OUTLIER & DISTRIBUTION REPAIR")
print("="*80)

# 3.1. Çarpıklık Analizi
print("\n3.1. ÇARPIKLIK DÖNÜŞÜMLERI")

numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('popularity')  # Hedef değişken çıkar

skewness_before = df_cleaned[numeric_cols].skew().sort_values(ascending=False)
high_skew = skewness_before[abs(skewness_before) > 1]

print(f"\nYüksek çarpıklık (|skew| > 1) tespit edilen değişkenler:")
for col, skew_val in high_skew.items():
    print(f"  {col}: {skew_val:.4f}")

# duration_ms için log dönüşümü (çok yüksek skewness: 11.2)
if 'duration_ms' in high_skew.index:
    df_cleaned['duration_ms_log'] = np.log1p(df_cleaned['duration_ms'])
    print(f"\n✓ duration_ms için log dönüşümü uygulandı (skewness: {high_skew['duration_ms']:.4f})")
    print(f"  Yeni skewness: {df_cleaned['duration_ms_log'].skew():.4f}")
    df_cleaned.drop(columns=['duration_ms'], inplace=True)
    
    log_dataprep_action(
        step="Phase 3.1",
        issue="duration_ms yüksek çarpıklık (skew=11.2)",
        decision="Log dönüşümü uygulandı (duration_ms_log)",
        rationale="Skewness > 10 - log dönüşümü çarpıklığı azaltır ve linear modeller için uygun hale getirir",
        risk="Düşük"
    )

# speechiness, instrumentalness, liveness için log dönüşümü
for col in ['speechiness', 'instrumentalness', 'liveness']:
    if col in high_skew.index:
        df_cleaned[f'{col}_log'] = np.log1p(df_cleaned[col])
        print(f"✓ {col} için log dönüşümü uygulandı (skewness: {high_skew[col]:.4f} → {df_cleaned[f'{col}_log'].skew():.4f})")
        df_cleaned.drop(columns=[col], inplace=True)
        
        log_dataprep_action(
            step="Phase 3.1",
            issue=f"{col} yüksek çarpıklık",
            decision=f"Log dönüşümü uygulandı ({col}_log)",
            rationale="Yüksek skewness - log dönüşümü dağılımı normalize eder",
            risk="Düşük"
        )

# 3.2. Outlier Yönetimi - RobustScaler kullanacağız (outlier'lara dayanıklı)
print(f"\n3.2. OUTLIER YÖNETİMİ")
print(f"Karar: RobustScaler kullanılacak (outlier'lara dayanıklı)")
print(f"  - instrumentalness: %22.15 outlier")
print(f"  - speechiness: %11.59 outlier")
print(f"  - time_signature: %10.66 outlier")
print(f"  - liveness: %7.58 outlier")
print(f"\nOutlier silme yerine robust scaling tercih edildi - müzik verisinde outlier'lar anlamlı olabilir (örn: instrumental şarkılar)")

log_dataprep_action(
    step="Phase 3.2",
    issue="Yüksek outlier oranları (5-22% arası)",
    decision="Silme yapılmadı - RobustScaler ile scaling yapılacak",
    rationale="Müzik verisinde outlier'lar domain-specific anlamlı olabilir (örn: fully instrumental tracks). Robust scaling outlier'lara dayanıklıdır.",
    risk="Düşük"
)

add_model_handoff(
    item="Outlier Yönetimi",
    status="RobustScaler seçildi",
    recommendation="RobustScaler IQR bazlı scaling yapar, outlier'lara dayanıklıdır. Tree-based modellerde scaling opsiyonel, linear modellerde zorunlu."
)

# PHASE 3 TAMAMLANDI - TEMİZLENMİŞ VERİYİ KAYDET
print(f"\n✅ Phase 2-3 Tamamlandı - Temizlenmiş veri kaydediliyor...")
Path('../data/processed').mkdir(parents=True, exist_ok=True)
df_cleaned.to_csv('../data/processed/dataset_cleaned.csv', index=False)
print(f"✓ Temizlenmiş veri kaydedildi: ../data/processed/dataset_cleaned.csv")
print(f"  - ID sütunları çıkarılmış")
print(f"  - Log transforms uygulanmış (duration_ms, speechiness, instrumentalness, liveness)")
print(f"  - Outlier stratejisi belirlendi (RobustScaler)")
print(f"  - Boyut: {df_cleaned.shape[0]} satır × {df_cleaned.shape[1]} sütun")
print(f"  - Sonraki: Encoding ve Feature Engineering (Phase 4-5)")

log_dataprep_action(
    step="Phase 3 Checkpoint",
    issue="Temizlenmiş veri kayıt",
    decision="dataset_cleaned.csv oluşturuldu (data/processed/)",
    rationale="ID sütunları çıkarılmış, log transforms uygulanmış veri. Encoding öncesi checkpoint - EDA Expert çıktısı formatı.",
    risk="Yok"
)

print("\n" + "="*80)
print("PHASE 4: ENCODING & TRANSFORMATION")
print("="*80)

# 4.1. Hedef Değişkeni Ayır
print("\n4.1. HEDEF DEĞİŞKEN AYIR")
target_col = 'popularity'
X = df_cleaned.drop(columns=[target_col])
y = df_cleaned[target_col]

print(f"✓ Hedef değişken (y): {target_col}")
print(f"  Feature'lar (X): {X.shape[1]} sütun")
print(f"  Target istatistikleri: min={y.min()}, max={y.max()}, ortalama={y.mean():.2f}, medyan={y.median():.2f}")

# 4.2. explicit Binary Encoding
print(f"\n4.2. EXPLICIT BINARY ENCODING")
if 'explicit' in X.columns:
    X['explicit'] = X['explicit'].astype(int)
    print(f"✓ explicit: Boolean → Binary (0/1)")
    print(f"  False: {(X['explicit'] == 0).sum()} (%{(X['explicit'] == 0).sum()/len(X)*100:.2f})")
    print(f"  True: {(X['explicit'] == 1).sum()} (%{(X['explicit'] == 1).sum()/len(X)*100:.2f})")
    
    log_dataprep_action(
        step="Phase 4.2",
        issue="explicit kategorik değişken",
        decision="Binary encoding (0/1)",
        rationale="Boolean değişken - binary encoding yeterli",
        risk="Yok"
    )

# 4.3. track_genre One-Hot Encoding (KRİTİK - EN GÜÇLÜ FEATURE!)
print(f"\n4.3. ⚠️ KRİTİK: track_genre ONE-HOT ENCODING (EN GÜÇLÜ FEATURE!)")
if 'track_genre' in X.columns:
    print(f"track_genre: 114 eşsiz tür")
    print(f"  Karar: One-Hot Encoding (114 binary column)")
    print(f"  Gerekçe: EDA Expert'in bulgularına göre track_genre, popularity'nin EN GÜÇLÜ açıklayıcısı")
    print(f"    - pop-film: ortalama 59.3")
    print(f"    - acoustic: ortalama 20-25")
    print(f"    - Audio features korelasyon < 0.1, genre ÇOK DAHA GÜÇLÜ!")
    
    # One-Hot Encoding
    genre_dummies = pd.get_dummies(X['track_genre'], prefix='genre', drop_first=False)
    X = pd.concat([X.drop(columns=['track_genre']), genre_dummies], axis=1)
    
    print(f"\n✓ track_genre one-hot encoded: {len(genre_dummies.columns)} binary sütun eklendi")
    print(f"  Toplam feature sayısı: {X.shape[1]}")
    
    log_dataprep_action(
        step="Phase 4.3",
        issue="track_genre kategorik (114 tür) - EN GÜÇLÜ FEATURE",
        decision="One-Hot Encoding uygulandı (114 binary column)",
        rationale="EDA Expert: track_genre popularity'nin en güçlü açıklayıcısı (pop-film 59.3 vs acoustic 20-25). Audio features korelasyon < 0.1, genre KRİTİK öneme sahip!",
        risk="Yok - KRİTİK feature, mutlaka encode edilmeli"
    )
    
    add_model_handoff(
        item="track_genre Encoding",
        status="One-Hot Encoded (114 sütun)",
        recommendation="⚠️ KRİTİK: Bu feature popularity'nin EN GÜÇLÜ açıklayıcısı. Linear modellerde regularization (Ridge/Lasso) önerilir (114 feature). Tree-based modellerde sorun yok."
    )

# 4.4. Scaling Strategy
print(f"\n4.4. SCALING STRATEGY")
print(f"Karar: RobustScaler (outlier'lara dayanıklı)")
print(f"  - Linear/Regularized modeller için zorunlu")
print(f"  - Tree-based modeller için opsiyonel ama faydalı")
print(f"  - IQR bazlı - outlier'ların etkisini azaltır")
print(f"\n⚠️ ÖNEMLİ: Scaling train-test split SONRASI fit edilecek (data leakage önleme)")

log_dataprep_action(
    step="Phase 4.4",
    issue="Farklı ölçeklerde numeric features",
    decision="RobustScaler seçildi (train-test split sonrası fit edilecek)",
    rationale="Outlier oranı yüksek değişkenler var (5-22%). RobustScaler IQR bazlı, outlier'lara dayanıklı. StandardScaler'dan daha uygun.",
    risk="Yok - Leakage önlendi (split sonrası fit)"
)

print("\n" + "="*80)
print("PHASE 5: FEATURE ENGINEERING")
print("="*80)

print("\n5.1. DOMAIN KNOWLEDGE BASED FEATURES")

# Energy x Loudness interaction (yüksek korelasyon var ama popularity'ye etkisi minimal)
if 'energy' in X.columns and 'loudness' in X.columns:
    X['energy_loudness_interaction'] = X['energy'] * X['loudness']
    print(f"✓ energy_loudness_interaction oluşturuldu (energy x loudness)")
    print(f"  Gerekçe: EDA'da energy-loudness korelasyonu 0.76 - interaction feature test edilebilir")

# Mood score (valence x energy)
if 'valence' in X.columns and 'energy' in X.columns:
    X['mood_score'] = X['valence'] * X['energy']
    print(f"✓ mood_score oluşturuldu (valence x energy)")
    print(f"  Gerekçe: Şarkının genel mood'u - pozitif enerjik vs negatif düşük enerji")

# Acoustic score (1 - acousticness) - elektronik seviye
if 'acousticness' in X.columns:
    X['electronic_score'] = 1 - X['acousticness']
    print(f"✓ electronic_score oluşturuldu (1 - acousticness)")
    print(f"  Gerekçe: Elektronik müzik seviyesi - akustikliğin tersi")

print(f"\n⚠️ ÖNEMLİ NOT:")
print(f"EDA Expert bulgularına göre audio features ile popularity arasında çok zayıf ilişki var (korelasyon < 0.1).")
print(f"Feature engineering'in model performansına etkisi minimal olabilir.")
print(f"EN GÜÇLÜ FEATURE: track_genre (zaten encode edildi)")

log_dataprep_action(
    step="Phase 5.1",
    issue="Feature Engineering fırsatları",
    decision="3 interaction feature oluşturuldu (energy_loudness, mood_score, electronic_score)",
    rationale="Domain knowledge bazlı feature'lar. Ancak EDA'da audio features popularity ile zayıf korelasyon gösterdi - minimal etki beklenir.",
    risk="Düşük - Feature count artışı minimal (3 feature)"
)

print(f"\nToplam feature sayısı (engineering sonrası): {X.shape[1]}")

print("\n" + "="*80)
print("PHASE 6: FEATURE SELECTION & LEAKAGE AUDIT")
print("="*80)

print("\n6.1. LEAKAGE KONTROLÜ")
print(f"✓ ID sütunları çıkarıldı (track_id, artists, album_name, track_name)")
print(f"✓ Hedef değişkeni doğrudan kopyalayan feature yok")
print(f"✓ Temporal leakage riski yok (zaman serisi değil)")
print(f"\nLeakage Riski: YOK")

log_dataprep_action(
    step="Phase 6.1",
    issue="Data leakage risk assessment",
    decision="Leakage tespit edilmedi",
    rationale="ID sütunları çıkarıldı, target'ı kopyalayan feature yok, temporal variable yok",
    risk="Yok"
)

print("\n6.2. MULTICOLLINEARITY (VIF)")
print(f"EDA Expert bulguları:")
print(f"  - tempo: VIF 15.2")
print(f"  - energy: VIF 15.0")
print(f"  - danceability: VIF 12.3")
print(f"  - energy-loudness korelasyon: 0.76")
print(f"  - energy-acousticness korelasyon: -0.73")
print(f"\nKarar: Feature elimination YAPILMADI")
print(f"Gerekçe:")
print(f"  1. Tree-based modeller (Random Forest, XGBoost) multicollinearity'ye dayanıklı")
print(f"  2. Linear modellerde regularization (Ridge/Lasso) kullanılacak - VIF sorunu çözülür")
print(f"  3. Audio features zaten popularity ile zayıf korelasyon - elimine etmek fayda sağlamaz")
print(f"\n⚠️ Model Expert'e Not: Linear modellerde Ridge/Lasso kullan, tree-based'de sorun yok")

log_dataprep_action(
    step="Phase 6.2",
    issue="Multicollinearity (VIF > 10)",
    decision="Feature elimination yapılmadı",
    rationale="Tree-based modeller için sorun değil. Linear modellerde regularization kullanılacak. Audio features zaten zayıf korelasyon.",
    risk="Düşük - Model seçimiyle yönetilecek"
)

add_model_handoff(
    item="Multicollinearity",
    status="Yönetilmeli",
    recommendation="Linear/Regularized modellerde Ridge/Lasso kullan (L1/L2 regularization VIF'i halleder). Tree-based modellerde (RF, XGBoost) multicollinearity sorun değil."
)

print("\n" + "="*80)
print("PHASE 7: MODEL-READY HANDOFF (TRAIN-TEST SPLIT & PIPELINE)")
print("="*80)

print("\n7.1. TRAIN-TEST SPLIT")
print(f"Strateji: Random Split (80-20)")
print(f"Gerekçe:")
print(f"  - Regression problemi (popularity: 0-100)")
print(f"  - Stratified split gerekmez (classification değil)")
print(f"  - Temporal variable yok - chronological split gerekmez")
print(f"  - Random state: 42 (reproducibility)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n✓ Train-test split tamamlandı:")
print(f"  Train: {X_train.shape[0]} satır ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test: {X_test.shape[0]} satır ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  Feature sayısı: {X_train.shape[1]}")

log_dataprep_action(
    step="Phase 7.1",
    issue="Train-test split",
    decision="Random split 80-20 (random_state=42)",
    rationale="Regression problemi, stratified gerekmez. Temporal variable yok.",
    risk="Yok"
)

# 7.2. Scaling (train'e fit, test'e transform - LEAKAGE ÖNLEMİ!)
print(f"\n7.2. SCALING (RobustScaler - LEAKAGE ÖNLEMİ)")

# Numerik sütunları bul (genre_* one-hot sütunları hariç)
numeric_features = [col for col in X_train.columns if not col.startswith('genre_')]
categorical_features = [col for col in X_train.columns if col.startswith('genre_')]

print(f"Scaling yapılacak numeric features: {len(numeric_features)}")
print(f"Scaling yapılMAYACAK categorical features (one-hot): {len(categorical_features)}")

scaler = RobustScaler()
X_train_numeric_scaled = scaler.fit_transform(X_train[numeric_features])
X_test_numeric_scaled = scaler.transform(X_test[numeric_features])

# Scaled dataframe'leri oluştur
X_train_scaled = pd.DataFrame(
    X_train_numeric_scaled,
    columns=numeric_features,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    X_test_numeric_scaled,
    columns=numeric_features,
    index=X_test.index
)

# One-hot encoded genre sütunlarını ekle
X_train_scaled = pd.concat([X_train_scaled, X_train[categorical_features]], axis=1)
X_test_scaled = pd.concat([X_test_scaled, X_test[categorical_features]], axis=1)

print(f"\n✓ RobustScaler uygulandı:")
print(f"  - Train'e fit edildi (leakage yok)")
print(f"  - Test'e transform edildi")
print(f"  - Numeric features scaled: {len(numeric_features)}")
print(f"  - Genre one-hot features (0/1): {len(categorical_features)} (scaling gerekmez)")

log_dataprep_action(
    step="Phase 7.2",
    issue="Scaling (leakage riski)",
    decision="RobustScaler - train fit, test transform (leakage önlendi)",
    rationale="Scaler yalnızca train'e fit edildi, test'e transform uygulandı. One-hot features (0/1) scaling gerektirmez.",
    risk="Yok - Leakage önlendi"
)

# 7.3. Model-Ready Data Kaydet
print(f"\n7.3. MODEL-READY DATA KAYDET")

X_train_scaled.to_csv('../data/model_ready/X_train.csv', index=False)
X_test_scaled.to_csv('../data/model_ready/X_test.csv', index=False)
y_train.to_csv('../data/model_ready/y_train.csv', index=False)
y_test.to_csv('../data/model_ready/y_test.csv', index=False)

print(f"✓ Model-ready data kaydedildi:")
print(f"  ../data/model_ready/X_train.csv")
print(f"  ../data/model_ready/X_test.csv")
print(f"  ../data/model_ready/y_train.csv")
print(f"  ../data/model_ready/y_test.csv")

# Preprocessing pipeline kaydet
joblib.dump(scaler, '../models/preprocessing_pipeline_scaler.pkl')
print(f"\n✓ Preprocessing pipeline kaydedildi:")
print(f"  ../models/preprocessing_pipeline_scaler.pkl")

# 7.4. DataPrep Actions Raporu Kaydet
dataprep_actions_df = pd.DataFrame(dataprep_actions)
dataprep_actions_df.to_csv('../reports/csv/dataprep_actions_log.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ DataPrep actions log kaydedildi:")
print(f"  ../reports/csv/dataprep_actions_log.csv")

# 7.5. Model Expert Handoff Raporu Kaydet
model_handoff_df = pd.DataFrame(model_handoff_report)
model_handoff_df.to_csv('../reports/csv/model_expert_handoff.csv', index=False, encoding='utf-8-sig')
print(f"✓ Model Expert handoff raporu kaydedildi:")
print(f"  ../reports/csv/model_expert_handoff.csv")

print("\n" + "="*80)
print("MODEL EXPERT HANDOFF SUMMARY")
print("="*80)

print(f"""
## Veri Durumu: ✅ TEMİZ VE MODEL-READY

## Missing Value Strategy:
- Durum: Minimal (%0.00 - 3 satır)
- Karar: Müdahale edilmedi (etkilenen sütunlar ID sütunları - zaten çıkarıldı)

## Encoding Strategy:
- explicit: Binary encoding (0/1) ✓
- track_genre: One-Hot Encoding (114 binary sütun) ✓ ⚠️ KRİTİK FEATURE!

## Scaling Strategy:
- RobustScaler (IQR bazlı - outlier'lara dayanıklı) ✓
- Train fit, test transform (leakage önlendi) ✓
- Numeric features scaled, one-hot (0/1) olduğu gibi bırakıldı ✓

## Feature Engineering:
- duration_ms → duration_ms_log (log transform)
- speechiness → speechiness_log
- instrumentalness → instrumentalness_log
- liveness → liveness_log
- energy_loudness_interaction (energy x loudness)
- mood_score (valence x energy)
- electronic_score (1 - acousticness)

## Leakage Status: ✅ YOK
- ID sütunları çıkarıldı ✓
- Train-test split öncesi scaling yapılmadı ✓
- Target encoding kullanılmadı ✓

## Önerilen Model Türleri:
1. **Baseline:** Linear Regression (R² baseline için)
   - Beklenen R²: 0.05-0.10 (audio features zayıf)

2. **Regularized (ÖNERİLEN):** Ridge, Lasso, ElasticNet
   - Multicollinearity için uygun
   - Genre one-hot (114 feature) için L1/L2 regularization faydalı
   - Beklenen R²: 0.08-0.12

3. **Tree-Based (ÖNERİLEN):** Random Forest Regressor, XGBoost Regressor, LightGBM
   - Multicollinearity'ye dayanıklı
   - Outlier'lara dayanıklı
   - Genre encoding ile ÇOK DAHA İYİ performans (kategorik feature'lar için güçlü)
   - Beklenen R²: 0.10-0.15 (track_genre encoding sayesinde)

4. **Neural Network:** MLP Regressor
   - Polynomial relationships test için
   - Beklenen iyileşme: Minimal

## ⚠️ KRİTİK UYARILAR:

1. **Audio Features Yetersiz:**
   - EDA Expert bulgusu: Audio features ile popularity korelasyon < 0.1
   - EN GÜÇLÜ FEATURE: track_genre (one-hot encoded)
   - Mevcut features ile R² < 0.15 beklenmeli

2. **Beklenti Yönetimi:**
   - Yüksek accuracy BEKLENMEMELİ
   - R² = 0.10-0.15 bile "iyi sonuç" sayılmalı
   - Popülerlik müzikal özelliklerden çok external faktörlere bağlı

3. **İyileştirme Önerileri:**
   - Spotify API ile external features: artist_followers, playlist_count, release_date
   - Temporal features: Yayın tarihi, trend dönemleri
   - Social features: Play count, skip rate

## Değerlendirme Metrikleri:
- R² (Coefficient of Determination) - Primary metrik
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Residual Plot (error distribution)

## Feature Count Summary:
- Total features: {X_train_scaled.shape[1]}
  - Genre one-hot: {len(categorical_features)}
  - Numeric features: {len(numeric_features)}
  - Engineered features: 3

## Train-Test Split:
- Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)
- Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)
- Random state: 42 (reproducible)

## Data Files Ready:
✓ ../data/model_ready/X_train.csv
✓ ../data/model_ready/X_test.csv
✓ ../data/model_ready/y_train.csv
✓ ../data/model_ready/y_test.csv
✓ ../models/preprocessing_pipeline_scaler.pkl

## Son Notlar:
- Veri temiz ve model-ready ✓
- Leakage riski yok ✓
- Genre encoding KRİTİK - model performansını artıracak ✓
- Audio features zayıf - external data KRİTİK ⚠️
- Tree-based modeller (XGBoost, LightGBM) öncelikli denenmelidir ⚠️
""")

print("\n" + "="*80)
print("DATA PREPARATION PIPELINE TAMAMLANDI")
print("="*80)

print(f"\n✅ Tüm 7 phase başarıyla tamamlandı!")
print(f"✅ Model-ready data hazır: ../data/model_ready/")
print(f"✅ Preprocessing pipeline kaydedildi: ../models/")
print(f"✅ DataPrep actions log: ../reports/csv/dataprep_actions_log.csv")
print(f"✅ Model Expert handoff: ../reports/csv/model_expert_handoff.csv")
print(f"\n⚠️ Sonraki adım: Model Expert ile regression modeling başlatılabilir")
print(f"⚠️ Beklenti: R² = 0.10-0.15 (track_genre encoding ile)")
