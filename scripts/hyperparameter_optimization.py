"""
================================================================================
HIPERPARAMETRE OPTİMİZASYONU - SPOTIFY POPULARITY REGRESSION
================================================================================
Top 4 modelin hiperparametre optimizasyonu:
  1. Random Forest (FIXED) - R²=0.5902
  2. Bagging Regressor - R²=0.5827
  3. Extra Trees (FIXED) - R²=0.5598
  4. XGBoost - R²=0.3698

Yöntem: RandomizedSearchCV (GridSearch çok yavaş)
CV: 5-Fold
Metrik: R²
n_iter: 50 (her model için 50 kombinasyon denenecek)
"""

import sys
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from prettytable import PrettyTable
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Windows terminal encoding fix
sys.stdout.reconfigure(encoding='utf-8')

# Random state
RANDOM_STATE = 42

print("="*80)
print("HIPERPARAMETRE OPTİMİZASYONU - TOP 4 MODEL")
print("="*80)

# ============================================================================
# PHASE 1: DATA LOADING
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: VERİ YÜKLEME")
print("="*80)

X_train = pd.read_csv("../data/model_ready/X_train.csv")
X_test = pd.read_csv("../data/model_ready/X_test.csv")
y_train = pd.read_csv("../data/model_ready/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/model_ready/y_test.csv").values.ravel()

print(f"✓ X_train: {X_train.shape}")
print(f"✓ X_test: {X_test.shape}")
print(f"✓ y_train: {len(y_train)}")
print(f"✓ y_test: {len(y_test)}")

# ============================================================================
# PHASE 2: MODEL VE PARAMETRE GRİDLERİ
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: MODEL VE PARAMETRE GRİDLERİ TANIMLAMA")
print("="*80)

# Random Forest
rf_param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [None, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True]
}

# Extra Trees
et_param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [None, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Bagging (DecisionTree base estimator)
bagging_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_samples': [0.5, 0.7, 0.9, 1.0],
    'max_features': [0.5, 0.7, 0.9, 1.0],
    'bootstrap': [True],
    'bootstrap_features': [False, True]
}

# XGBoost
xgb_param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

models_config = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": rf_param_grid
    },
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": et_param_grid
    },
    "Bagging Regressor": {
        "model": BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "param_grid": bagging_param_grid
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "param_grid": xgb_param_grid
    }
}

print(f"✓ 4 model tanımlandı:")
for model_name, config in models_config.items():
    grid_size = np.prod([len(v) for v in config['param_grid'].values()])
    print(f"  • {model_name}: {len(config['param_grid'])} parametre, ~{grid_size:,} kombinasyon")

# ============================================================================
# PHASE 3: RANDOMIZED SEARCH (50 ITERATIONS PER MODEL)
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: RANDOMIZED SEARCH CV (n_iter=50, cv=5)")
print("="*80)
print("⚠️ Bu işlem UZUN sürebilir (her model 10-30 dakika)")

results = []
best_models = {}

for model_name, config in models_config.items():
    print(f"\n[{model_name}] Hiperparametre optimizasyonu başlıyor...")
    start_time = time.time()
    
    try:
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['param_grid'],
            n_iter=50,  # 50 random combination
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )
        
        # Fit
        random_search.fit(X_train, y_train)
        
        # Best model predictions
        best_model = random_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        cv_r2 = random_search.best_score_
        
        duration = time.time() - start_time
        
        print(f"  ✓ Başarılı!")
        print(f"    Best CV R²: {cv_r2:.4f}")
        print(f"    Test R²: {test_r2:.4f}")
        print(f"    Test RMSE: {test_rmse:.2f}")
        print(f"    Süre: {duration:.1f}s ({duration/60:.1f} dakika)")
        print(f"    Best params: {random_search.best_params_}")
        
        results.append({
            'Model': model_name,
            'Train R²': round(train_r2, 4),
            'Test R²': round(test_r2, 4),
            'Test RMSE': round(test_rmse, 2),
            'Test MAE': round(test_mae, 2),
            'CV R² (Best)': round(cv_r2, 4),
            'Best Params': str(random_search.best_params_),
            'Duration (s)': round(duration, 2),
            'Status': 'SUCCESS'
        })
        
        best_models[model_name] = best_model
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"  ✗ HATA: {str(e)}")
        results.append({
            'Model': model_name,
            'Train R²': None,
            'Test R²': None,
            'Test RMSE': None,
            'Test MAE': None,
            'CV R² (Best)': None,
            'Best Params': None,
            'Duration (s)': round(duration, 2),
            'Status': f'ERROR: {str(e)}'
        })

# ============================================================================
# PHASE 4: SONUÇLARI KARŞILAŞTIR
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: OPTİMİZASYON SONUÇLARI")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test R²', ascending=False).reset_index(drop=True)

# PrettyTable
table = PrettyTable()
table.field_names = ['Sıra', 'Model', 'Test R²', 'RMSE', 'CV R² (Best)', 'Süre (dk)', 'Durum']

for idx, row in results_df.iterrows():
    if row['Status'] == 'SUCCESS':
        table.add_row([
            idx + 1,
            row['Model'],
            f"{row['Test R²']:.4f}",
            f"{row['Test RMSE']:.2f}",
            f"{row['CV R² (Best)']:.4f}",
            f"{row['Duration (s)']/60:.1f}",
            '✓'
        ])
    else:
        table.add_row([
            idx + 1,
            row['Model'],
            'HATA',
            'HATA',
            'HATA',
            f"{row['Duration (s)']/60:.1f}",
            '✗'
        ])

print(table)

# Save results
results_df.to_csv("../reports/csv/hyperparameter_optimization_results.csv", index=False, encoding='utf-8-sig')
print(f"\n✓ Sonuçlar kaydedildi: ../reports/csv/hyperparameter_optimization_results.csv")

# ============================================================================
# PHASE 5: EN İYİ MODELİ KAYDET
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: EN İYİ MODELİ KAYDETME")
print("="*80)

best_result = results_df[results_df['Status'] == 'SUCCESS'].iloc[0]
best_model_name = best_result['Model']
best_model_obj = best_models[best_model_name]

print(f"🏆 EN İYİ MODEL: {best_model_name}")
print(f"  Test R²: {best_result['Test R²']:.4f}")
print(f"  Test RMSE: {best_result['Test RMSE']:.2f}")
print(f"  CV R² (Best): {best_result['CV R² (Best)']:.4f}")
print(f"  Best Params: {best_result['Best Params']}")

# Save model
model_path = "../models/best_model_optimized.pkl"
joblib.dump(best_model_obj, model_path)
print(f"\n✓ Model kaydedildi: {model_path}")

# Save best params
params_df = pd.DataFrame([{
    'Model': best_model_name,
    'Test R²': best_result['Test R²'],
    'Test RMSE': best_result['Test RMSE'],
    'Best Params': best_result['Best Params']
}])
params_df.to_csv("../reports/csv/best_model_params.csv", index=False, encoding='utf-8-sig')
print(f"✓ Best params kaydedildi: ../reports/csv/best_model_params.csv")

# ============================================================================
# PHASE 6: BASELINE KARŞILAŞTIRMASI
# ============================================================================
print("\n" + "="*80)
print("PHASE 6: BASELINE KARŞILAŞTIRMA (Önceki vs Optimizasyon)")
print("="*80)

baseline_results = pd.read_csv("../reports/csv/model_comparison_extended_22plus.csv")

comparison_table = PrettyTable()
comparison_table.field_names = ['Model', 'Baseline R²', 'Optimized R²', 'İyileşme', 'Durum']

for idx, row in results_df.iterrows():
    if row['Status'] == 'SUCCESS':
        model_name = row['Model']
        # Find baseline
        if model_name == "Random Forest":
            baseline_row = baseline_results[baseline_results['Model'] == 'Random Forest (FIXED)']
        elif model_name == "Extra Trees":
            baseline_row = baseline_results[baseline_results['Model'] == 'Extra Trees (FIXED)']
        else:
            baseline_row = baseline_results[baseline_results['Model'] == model_name]
        
        if not baseline_row.empty:
            baseline_r2 = baseline_row['Test R²'].values[0]
            optimized_r2 = row['Test R²']
            improvement = ((optimized_r2 - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
            
            comparison_table.add_row([
                model_name,
                f"{baseline_r2:.4f}",
                f"{optimized_r2:.4f}",
                f"{improvement:+.2f}%",
                '🚀' if improvement > 0 else '📉'
            ])

print(comparison_table)

print("\n" + "="*80)
print("HİPERPARAMETRE OPTİMİZASYONU TAMAMLANDI!")
print("="*80)
print(f"✅ 4 model optimize edildi")
print(f"✅ En iyi model: {best_model_name} (R²={best_result['Test R²']:.4f})")
print(f"✅ Sonuçlar kaydedildi: reports/csv/")
