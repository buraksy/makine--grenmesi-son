"""
================================================================================
EXPLAINABILITY ANALYSIS - MODEL INTERPRETABILITY
================================================================================
SHAP (SHapley Additive exPlanations) ile model açıklanabilirliği:
  • Feature Importance
  • SHAP Summary Plots
  • SHAP Dependence Plots
  • SHAP Waterfall Plots

Model: En iyi optimize edilmiş model (Bagging Regressor R²=0.6299)
"""

import sys
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings('ignore')

# Windows terminal encoding fix
sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("EXPLAINABILITY ANALYSIS - MODEL INTERPRETABILITY")
print("="*80)

# ============================================================================
# PHASE 1: LOAD MODEL AND DATA
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: MODEL VE VERİ YÜKLEME")
print("="*80)

# Load best optimized model
model = joblib.load("../models/best_model_optimized.pkl")
print(f"✓ Model yüklendi: {type(model).__name__}")

# Load data
X_train = pd.read_csv("../data/model_ready/X_train.csv")
X_test = pd.read_csv("../data/model_ready/X_test.csv")
y_train = pd.read_csv("../data/model_ready/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/model_ready/y_test.csv").values.ravel()

print(f"✓ X_train: {X_train.shape}")
print(f"✓ X_test: {X_test.shape}")

feature_names = X_train.columns.tolist()
print(f"✓ Total features: {len(feature_names)}")

# ============================================================================
# PHASE 2: FEATURE IMPORTANCE (TREE-BASED)
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: FEATURE IMPORTANCE ANALİZİ")
print("="*80)

# Get feature importance from base estimators (if tree-based)
if hasattr(model, 'estimators_'):
    # Average feature importance across all base estimators
    importances_list = []
    for estimator in model.estimators_:
        if hasattr(estimator, 'feature_importances_'):
            # Check if length matches
            if len(estimator.feature_importances_) == len(feature_names):
                importances_list.append(estimator.feature_importances_)
    
    if importances_list:
        feature_importance = np.mean(importances_list, axis=0)
        
        # Ensure same length
        if len(feature_importance) == len(feature_names):
            # Create DataFrame
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
        else:
            print(f"⚠️ Feature importance length mismatch: {len(feature_importance)} vs {len(feature_names)}")
            fi_df = None
    else:
        print("⚠️ No feature importances found")
        fi_df = None
else:
    print("⚠️ Model has no estimators_ attribute")
    fi_df = None

if fi_df is not None:
        print(f"\n📊 TOP 20 ÖNEMLİ FEATURE:")
        print(fi_df.head(20).to_string(index=False))
        
        # Save
        fi_df.to_csv("../reports/csv/feature_importance.csv", index=False, encoding='utf-8-sig')
        print(f"\n✓ Feature importance kaydedildi: ../reports/csv/feature_importance.csv")
        
        # Plot
        plt.figure(figsize=(12, 8))
        top_20 = fi_df.head(20)
        plt.barh(range(len(top_20)), top_20['Importance'])
        plt.yticks(range(len(top_20)), top_20['Feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance (Bagging Regressor)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("../figures/feature_importance_top20.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Grafik kaydedildi: ../figures/feature_importance_top20.png")

# ============================================================================
# PHASE 3: SHAP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: SHAP ANALİZİ")
print("="*80)
print("⚠️ SHAP hesaplama UZUN sürebilir (10-30 dakika)")

# Sample data for SHAP (full dataset too slow)
sample_size = 100
np.random.seed(42)
sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
X_train_sample = X_train.iloc[sample_indices]

print(f"\nSample size: {sample_size} (SHAP hesaplama için)")

# Use shap.sample for background data (faster)
background_size = 50
X_background = shap.sample(X_train, background_size, random_state=42)
print(f"Background size: {background_size}")

# Create SHAP explainer
print("\n3.1. SHAP Explainer oluşturuluyor...")
try:
    # For BaggingRegressor, use first base estimator with TreeExplainer
    if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        print("⚠️ BaggingRegressor için ilk base estimator kullanılıyor")
        base_model = model.estimators_[0]
        explainer = shap.TreeExplainer(base_model, X_background)
        print("✓ TreeExplainer kullanılıyor (base estimator)")
    else:
        explainer = shap.Explainer(model, X_background)
        print("✓ TreeExplainer kullanılıyor")
except Exception as e:
    # Fallback to KernelExplainer (slower but works for any model)
    print(f"⚠️ TreeExplainer çalışmadı: {e}")
    print("⚠️ KernelExplainer kullanılıyor (daha yavaş)")
    explainer = shap.KernelExplainer(model.predict, X_background)

# Calculate SHAP values
print("\n3.2. SHAP values hesaplanıyor...")
shap_values = explainer(X_train_sample)
print(f"✓ SHAP values hesaplandı: {shap_values.values.shape}")

# ============================================================================
# PHASE 4: SHAP SUMMARY PLOT
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: SHAP SUMMARY PLOT")
print("="*80)

# Summary plot (beeswarm)
print("\n4.1. SHAP Summary Plot (Beeswarm)...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_train_sample, show=False, max_display=20)
plt.tight_layout()
plt.savefig("../figures/shap_summary_plot.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Kaydedildi: ../figures/shap_summary_plot.png")

# Summary plot (bar)
print("\n4.2. SHAP Summary Plot (Bar)...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_sample, plot_type="bar", show=False, max_display=20)
plt.tight_layout()
plt.savefig("../figures/shap_summary_bar.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Kaydedildi: ../figures/shap_summary_bar.png")

# ============================================================================
# PHASE 5: SHAP DEPENDENCE PLOTS (TOP 5 FEATURES)
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: SHAP DEPENDENCE PLOTS (TOP 5 FEATURES)")
print("="*80)

# Get top 5 features by mean absolute SHAP value
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_5_indices = np.argsort(mean_abs_shap)[-5:][::-1]
top_5_features = [feature_names[i] for i in top_5_indices]

print(f"\nTop 5 features: {top_5_features}")

for feature in top_5_features:
    print(f"\n  Plotting: {feature}")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values.values,
        X_train_sample,
        show=False,
        ax=ax
    )
    plt.tight_layout()
    safe_filename = feature.replace('/', '_').replace('\\', '_').replace(':', '_')
    plt.savefig(f"../figures/shap_dependence_{safe_filename}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Kaydedildi: ../figures/shap_dependence_{safe_filename}.png")

# ============================================================================
# PHASE 6: SHAP WATERFALL PLOTS (3 EXAMPLES)
# ============================================================================
print("\n" + "="*80)
print("PHASE 6: SHAP WATERFALL PLOTS (3 ÖRN)")
print("="*80)

# Select 3 instances: low, medium, high popularity
y_train_sample = y_train[sample_indices]
low_idx = np.argmin(y_train_sample)
high_idx = np.argmax(y_train_sample)
median_idx = np.argsort(y_train_sample)[len(y_train_sample)//2]

examples = [
    (low_idx, "Low Popularity", y_train_sample[low_idx]),
    (median_idx, "Medium Popularity", y_train_sample[median_idx]),
    (high_idx, "High Popularity", y_train_sample[high_idx])
]

for idx, label, actual_value in examples:
    print(f"\n  {label}: Actual={actual_value:.2f}")
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap_values[idx], max_display=15, show=False)
    plt.tight_layout()
    safe_label = label.replace(' ', '_').lower()
    plt.savefig(f"../figures/shap_waterfall_{safe_label}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Kaydedildi: ../figures/shap_waterfall_{safe_label}.png")

# ============================================================================
# PHASE 7: SHAP FORCE PLOT (GLOBAL)
# ============================================================================
print("\n" + "="*80)
print("PHASE 7: SHAP FORCE PLOT (SINGLE EXAMPLE)")
print("="*80)

print("\nForce plot oluşturuluyor (tek örnek)...")
# Use single sample for matplotlib force plot
sample_idx = len(X_train_sample) // 2  # Middle sample
shap.force_plot(
    explainer.expected_value,
    shap_values.values[sample_idx],
    X_train_sample.iloc[sample_idx],
    show=False,
    matplotlib=True
)
plt.tight_layout()
plt.savefig("../figures/shap_force_plot_example.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Kaydedildi: ../figures/shap_force_plot_example.png")

# ============================================================================
# PHASE 8: TOP FEATURES SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 8: TOP FEATURES ÖZET RAPORU")
print("="*80)

# Calculate mean absolute SHAP for all features
mean_abs_shap_all = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap_all
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\n📊 TOP 20 FEATURE (SHAP IMPORTANCE):")
print(shap_importance_df.head(20).to_string(index=False))

# Save
shap_importance_df.to_csv("../reports/csv/shap_feature_importance.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ SHAP importance kaydedildi: ../reports/csv/shap_feature_importance.csv")

# Compare with tree-based feature importance
if 'fi_df' in locals() and fi_df is not None:
    comparison_df = fi_df.merge(
        shap_importance_df,
        on='Feature',
        how='inner'
    ).sort_values('Mean_Abs_SHAP', ascending=False).head(20)
    
    print("\n📊 KARŞILAŞTIRMA (Tree Importance vs SHAP):")
    print(comparison_df.to_string(index=False))
    
    comparison_df.to_csv("../reports/csv/feature_importance_comparison.csv", index=False, encoding='utf-8-sig')
    print(f"\n✓ Karşılaştırma kaydedildi: ../reports/csv/feature_importance_comparison.csv")

print("\n" + "="*80)
print("EXPLAINABILITY ANALYSIS TAMAMLANDI!")
print("="*80)
print("✅ SHAP analizi tamamlandı")
print("✅ Feature importance hesaplandı")
print("✅ Grafikler kaydedildi: figures/")
print("✅ Raporlar kaydedildi: reports/csv/")
