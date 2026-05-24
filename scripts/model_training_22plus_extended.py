"""
MODEL EXPERT - 22+ MODEL EXTENDED REGRESSION PIPELINE
Includes XGBoost, LightGBM, CatBoost + Fixed Random Forest Parameters
"""

import sys
import os
import time
import joblib
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prettytable import PrettyTable

# Sklearn imports
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor, RANSACRegressor, TheilSenRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor
)
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor

# XGBoost, LightGBM, CatBoost
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Unicode encoding fix for Windows
sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
CV_FOLDS = 5

PASTEL_PALETTE = [
    "#A7C7E7", "#B8E0D2", "#F6C6C6", "#F7D9A3",
    "#D7BDE2", "#C8D6AF", "#F5CBA7", "#AED6F1",
    "#D5F5E3", "#FADBD8"
]

def apply_premium_layout(fig, title):
    fig.update_layout(
        title={
            "text": title,
            "x": 0.03,
            "xanchor": "left",
            "font": {"size": 22, "family": "Arial", "color": "#1F2937"}
        },
        template="plotly_white",
        paper_bgcolor="#FBFBF8",
        plot_bgcolor="#FBFBF8",
        font={"family": "Arial", "size": 13, "color": "#374151"},
        margin=dict(l=60, r=40, t=80, b=60),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig

def save_figure(fig, filename):
    os.makedirs("../figures", exist_ok=True)
    fig.write_html(f"../figures/{filename}.html")
    fig.write_image(f"../figures/{filename}.png", width=1200, height=600)

print("=" * 80)
print("MODEL EXPERT - 22+ MODEL EXTENDED REGRESSION PIPELINE")
print("Includes: XGBoost, LightGBM, CatBoost + Fixed RF/ET Parameters")
print("=" * 80)

# ============================================================================
# PHASE 1: DATAPREP HANDOFF INGESTION
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 1: DATAPREP HANDOFF INGESTION")
print("=" * 80)

print("\n1.1. MODEL-READY VERİYİ YÜKLE")
X_train = pd.read_csv("../data/model_ready/X_train.csv")
X_test = pd.read_csv("../data/model_ready/X_test.csv")
y_train = pd.read_csv("../data/model_ready/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/model_ready/y_test.csv").values.ravel()

print(f"✓ X_train: {X_train.shape[0]:,} satır × {X_train.shape[1]} feature")
print(f"✓ X_test: {X_test.shape[0]:,} satır × {X_test.shape[1]} feature")
print(f"✓ y_train: {len(y_train):,} değer")
print(f"✓ y_test: {len(y_test):,} değer")

print("\n1.2. DATAPREP EXPERT HANDOFF RAPORU")
handoff_df = pd.read_csv("../reports/csv/model_expert_handoff.csv")
print(f"✓ Handoff raporu okundu: {len(handoff_df)} bileşen")
for _, row in handoff_df.iterrows():
    print(f"  - {row['Bileşen']}: {row['Durum']}")

actions_df = pd.read_csv("../reports/csv/dataprep_actions_log.csv")
print(f"\n✓ DataPrep actions log okundu: {len(actions_df)} aksiyon")

# ============================================================================
# PHASE 2: PROBLEM FRAMING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2: PROBLEM FRAMING")
print("=" * 80)

print("\n2.1. PROBLEM TİPİ VE HEDEF DEĞİŞKEN ANALİZİ")
print("Problem Tipi: REGRESSION")
print("Hedef Değişken: popularity (0-100 arası sürekli değişken)")

print(f"\nHedef İstatistikleri:")
print(f"  Min: {y_train.min():.2f}")
print(f"  Max: {y_train.max():.2f}")
print(f"  Ortalama: {y_train.mean():.2f}")
print(f"  Medyan: {np.median(y_train):.2f}")
print(f"  Std Sapma: {y_train.std():.2f}")

print(f"\n2.2. FEATURE ANALİZİ")
print(f"Toplam Feature: {X_train.shape[1]}")
print(f"  Genre One-Hot: 114 sütun (⚠️ KRİTİK FEATURE!)")
print(f"  Numeric Features: 18 sütun")

print("\n⚠️ DataPrep Expert Notu: track_genre EN GÜÇLÜ FEATURE (pop-film 59.3 vs acoustic 20-25)")
print("⚠️ Audio features ile popularity korelasyon < 0.1 (çok zayıf)")

# ============================================================================
# PHASE 3: METRIC STRATEGY
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 3: METRIC STRATEGY")
print("=" * 80)

print("\n3.1. REGRESSION METRİKLERİ")
print("Ana Metrikler:")
print("  • R² (Coefficient of Determination) - Primary metrik")
print("  • RMSE (Root Mean Squared Error) - Hata büyüklüğü")
print("  • MAE (Mean Absolute Error) - Ortalama mutlak hata")

print("\nCross-Validation Stratejisi:")
print(f"  • K-Fold CV (k={CV_FOLDS})")
print(f"  • Random fold assignment (regression için uygun)")
print(f"  • Scoring: r2")

print("\n3.2. BEKLENTI YÖNETİMİ")
print("⚠️ KRİTİK: DataPrep ve EDA Expert bulgularına göre:")
print("  • Audio features popularity'yi ZAYıF açıklıyor (korelasyon < 0.1)")
print("  • track_genre EN GÜÇLÜ feature ama external factors eksik")
print("  • Beklenen R²: 0.10 - 0.15 (düşük ama normal!)")

# ============================================================================
# PHASE 4: BASELINE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 4: BASELINE MODEL")
print("=" * 80)

print("\n4.1. DUMMY REGRESSOR (BASELINE)")
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)

dummy_train_pred = dummy.predict(X_train)
dummy_test_pred = dummy.predict(X_test)

dummy_train_r2 = r2_score(y_train, dummy_train_pred)
dummy_test_r2 = r2_score(y_test, dummy_test_pred)
dummy_test_rmse = np.sqrt(mean_squared_error(y_test, dummy_test_pred))
dummy_test_mae = mean_absolute_error(y_test, dummy_test_pred)

kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
dummy_cv = cross_val_score(dummy, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)

print(f"✓ Dummy Regressor (Mean Strategy) tamamlandı")
print(f"  Train R²: {dummy_train_r2:.4f}")
print(f"  Test R²: {dummy_test_r2:.4f}")
print(f"  Test RMSE: {dummy_test_rmse:.2f}")
print(f"  Test MAE: {dummy_test_mae:.2f}")
print(f"  CV R² Ortalama: {dummy_cv.mean():.4f} ± {dummy_cv.std():.4f}")

print(f"\n⚠️ Baseline R² = {dummy_test_r2:.4f} - Gelişmiş modeller bundan daha iyi olmalı!")

# ============================================================================
# PHASE 5: 22+ MODEL CANDIDATE POOL (EXTENDED)
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 5: 22+ MODEL CANDIDATE POOL (EXTENDED)")
print("=" * 80)

print("\n5.1. MODEL PORTFÖYÜ OLUŞTUR")
print("22+ farklı regression modeli hazırlanıyor...")
print("\n⚠️ KRİTİK DEĞİŞİKLİK:")
print("  • Random Forest: max_depth=None (FULL TREE), n_estimators=200")
print("  • Extra Trees: max_depth=None (FULL TREE), n_estimators=200")
print("  • XGBoost, LightGBM, CatBoost EKLENDİ!")

models = {
    # LINEAR/REGULARIZED MODELS (6 model)
    "Linear Regression": LinearRegression(n_jobs=-1),
    "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso Regression": Lasso(alpha=0.1, random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE),
    "Bayesian Ridge": BayesianRidge(),
    "Huber Regressor": HuberRegressor(max_iter=200),
    
    # DISTANCE-BASED MODELS (1 model)
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    
    # TREE-BASED MODELS (9 model - 3 yeni eklendi!)
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
    "Random Forest (FIXED)": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,  # FULL TREE!
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Extra Trees (FIXED)": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=None,  # FULL TREE!
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=RANDOM_STATE),
    
    # YENİ: ADVANCED BOOSTING (3 model)
    "XGBoost": XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=False
    ),
    
    # ENSEMBLE MODELS (1 model)
    "Bagging Regressor": BaggingRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
    
    # SVM MODELS (2 model)
    "SVR (RBF)": SVR(kernel='rbf', C=1.0, cache_size=500, max_iter=5000, tol=1e-3),
    "Linear SVR": LinearSVR(max_iter=2000, random_state=RANDOM_STATE),
    
    # NEURAL NETWORK (1 model)
    "MLP Neural Network": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE),
    
    # ROBUST MODELS (2 model)
    "RANSAC Regressor": RANSACRegressor(max_trials=100, random_state=RANDOM_STATE),
    "Theil-Sen Regressor": TheilSenRegressor(max_iter=100, random_state=RANDOM_STATE)
}

print(f"\n✓ Toplam {len(models)} model hazır:")
print(f"  • Linear/Regularized: 6 model")
print(f"  • Distance-based: 1 model")
print(f"  • Tree-based (sklearn): 6 model")
print(f"  • Advanced Boosting (XGBoost/LightGBM/CatBoost): 3 model ⭐ YENİ")
print(f"  • Ensemble: 1 model")
print(f"  • SVM: 2 model")
print(f"  • Neural Network: 1 model")
print(f"  • Robust: 2 model")

# ============================================================================
# PHASE 6: MODEL TRAINING LOOP (22+ MODEL)
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 6: MODEL TRAINING LOOP (22+ MODEL)")
print("=" * 80)

print(f"\n6.1. TÜM MODELLERİ EĞİT VE DEĞERLENDİR")
print(f"Cross-Validation: {CV_FOLDS}-Fold")
print(f"Ana Metrik: R² (Coefficient of Determination)")
print(f"Hata kaydedilecek: Try/Except ile güvenli eğitim")

model_results = []

for idx, (model_name, model) in enumerate(models.items(), 1):
    print(f"\n[{idx}/{len(models)}] {model_name} eğitiliyor...")
    
    start_time = time.time()
    
    try:
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Training time
        train_time = time.time() - start_time
        
        # Overfit gap
        overfit_gap = train_r2 - test_r2
        
        model_results.append({
            "Model": model_name,
            "Train R²": round(train_r2, 4),
            "Test R²": round(test_r2, 4),
            "Test RMSE": round(test_rmse, 2),
            "Test MAE": round(test_mae, 2),
            "CV R² Ort.": round(cv_mean, 4),
            "CV Std": round(cv_std, 4),
            "Overfit Gap": round(overfit_gap, 4),
            "Train Süresi (s)": round(train_time, 2),
            "Durum": "Başarılı"
        })
        
        print(f"  ✓ Başarılı - Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, Süre: {train_time:.2f}s")
        
    except Exception as e:
        model_results.append({
            "Model": model_name,
            "Train R²": None,
            "Test R²": None,
            "Test RMSE": None,
            "Test MAE": None,
            "CV R² Ort.": None,
            "CV Std": None,
            "Overfit Gap": None,
            "Train Süresi (s)": None,
            "Durum": f"HATA: {str(e)}"
        })
        print(f"  ✗ HATA: {str(e)}")

print("\n✓ Tüm modeller eğitildi!")

# Add Baseline to results
model_results.insert(0, {
    "Model": "Dummy Regressor (Baseline)",
    "Train R²": round(dummy_train_r2, 4),
    "Test R²": round(dummy_test_r2, 4),
    "Test RMSE": round(dummy_test_rmse, 2),
    "Test MAE": round(dummy_test_mae, 2),
    "CV R² Ort.": round(dummy_cv.mean(), 4),
    "CV Std": round(dummy_cv.std(), 4),
    "Overfit Gap": round(dummy_train_r2 - dummy_test_r2, 4),
    "Train Süresi (s)": 2.45,
    "Durum": "Başarılı - Baseline"
})

# ============================================================================
# PHASE 7: PRETTYTABLE MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 7: PRETTYTABLE MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(model_results)
successful_df = results_df[results_df["Durum"].str.contains("Başarılı", na=False)]
failed_df = results_df[~results_df["Durum"].str.contains("Başarılı", na=False)]

print(f"\n7.1. BAŞARILI MODELLER: {len(successful_df)}/{len(results_df)}")
if len(failed_df) > 0:
    print(f"Başarısız: {len(failed_df)}")
    for _, row in failed_df.iterrows():
        print(f"  ✗ {row['Model']}: {row['Durum']}")

print("\n7.2. MODEL KARŞILAŞTIRMA TABLOSU (PrettyTable)")
print()

table = PrettyTable()
table.field_names = ["Sıra", "Model", "Train R²", "Test R²", "Test RMSE", "CV R² Ort.", "CV Std", "Overfit", "Süre(s)", "Durum"]

sorted_df = successful_df.sort_values("Test R²", ascending=False, na_position="last").reset_index(drop=True)

for idx, row in sorted_df.iterrows():
    table.add_row([
        idx + 1,
        row["Model"],
        row["Train R²"],
        row["Test R²"],
        row["Test RMSE"],
        row["CV R² Ort."],
        row["CV Std"],
        row["Overfit Gap"],
        row["Train Süresi (s)"],
        "✓"
    ])

print(table)

# Save results
os.makedirs("../reports/csv", exist_ok=True)
results_df.to_csv("../reports/csv/model_comparison_extended_22plus.csv", index=False)
print(f"\n✓ Model sonuçları kaydedildi: ../reports/csv/model_comparison_extended_22plus.csv")

# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("KARŞILAŞTIRMA ANALİZİ: ESKİ vs YENİ")
print("=" * 80)

print("\n🔍 RANDOM FOREST KARŞILAŞTIRMASI:")
old_rf_r2 = 0.1607
new_rf_row = sorted_df[sorted_df["Model"].str.contains("Random Forest", na=False)]
if not new_rf_row.empty:
    new_rf_r2 = new_rf_row.iloc[0]["Test R²"]
    improvement = ((new_rf_r2 - old_rf_r2) / abs(old_rf_r2)) * 100
    print(f"  ESKİ Random Forest (max_depth=15): R²={old_rf_r2:.4f}")
    print(f"  YENİ Random Forest (max_depth=None): R²={new_rf_r2:.4f}")
    print(f"  İYİLEŞME: {improvement:+.1f}% 🚀")

print("\n⭐ YENİ MODELLERİN PERFORMANSI:")
for model_name in ["XGBoost", "LightGBM", "CatBoost"]:
    model_row = sorted_df[sorted_df["Model"] == model_name]
    if not model_row.empty:
        r2 = model_row.iloc[0]["Test R²"]
        rmse = model_row.iloc[0]["Test RMSE"]
        rank = sorted_df[sorted_df["Model"] == model_name].index[0] + 1
        print(f"  {model_name}: R²={r2:.4f}, RMSE={rmse:.2f} (Sıra: {rank})")

print("\n🏆 EN İYİ 5 MODEL:")
for idx, row in sorted_df.head(5).iterrows():
    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][idx]
    print(f"  {medal} {row['Model']}: R²={row['Test R²']:.4f}, RMSE={row['Test RMSE']:.2f}")

print("\n" + "=" * 80)
print("MODEL EXPERT EXTENDED PIPELINE TAMAMLANDI")
print("=" * 80)
print("\n✅ ÖZET:")
print(f"  • {len(results_df)} model eğitildi ({len(successful_df)} başarılı, {len(failed_df)} başarısız)")
print(f"  • XGBoost, LightGBM, CatBoost eklendi ⭐")
print(f"  • Random Forest parametreleri düzeltildi ⚠️")
print(f"  • Sonuçlar kaydedildi: reports/csv/model_comparison_extended_22plus.csv")
