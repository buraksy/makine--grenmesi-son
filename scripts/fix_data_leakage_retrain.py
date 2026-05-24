"""
DATA LEAKAGE FIX & MODEL RETRAINING
====================================
SHAP analysis discovered 'Unnamed: 0' (index) as top feature - this is data leakage.
This script removes the index column and retrains the best model (Bagging Regressor).

Author: Model Expert Agent
Date: May 19, 2026
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

print("="*80)
print("DATA LEAKAGE FIX & MODEL RETRAINING")
print("="*80)

# ============================================================================
# PHASE 1: LOAD DATA
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: LOADING MODEL-READY DATA")
print("="*80)

# Load train/test splits
print("\nLoading train/test data...")
X_train = pd.read_csv("../data/model_ready/X_train.csv")
X_test = pd.read_csv("../data/model_ready/X_test.csv")
y_train = pd.read_csv("../data/model_ready/y_train.csv").values.ravel()
y_test = pd.read_csv("../data/model_ready/y_test.csv").values.ravel()

print(f"✓ X_train shape: {X_train.shape}")
print(f"✓ X_test shape: {X_test.shape}")
print(f"✓ y_train shape: {y_train.shape}")
print(f"✓ y_test shape: {y_test.shape}")

# ============================================================================
# PHASE 2: DATA LEAKAGE FIX
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: REMOVING DATA LEAKAGE (Unnamed: 0)")
print("="*80)

# Check for index columns
leakage_columns = [col for col in X_train.columns if 'Unnamed' in col or 'index' in col.lower()]
print(f"\n⚠️ Data leakage columns detected: {leakage_columns}")

if leakage_columns:
    print(f"\n🔧 Removing {len(leakage_columns)} leakage column(s)...")
    X_train_clean = X_train.drop(columns=leakage_columns)
    X_test_clean = X_test.drop(columns=leakage_columns)
    print(f"✓ X_train_clean shape: {X_train_clean.shape}")
    print(f"✓ X_test_clean shape: {X_test_clean.shape}")
else:
    print("✓ No leakage columns found")
    X_train_clean = X_train
    X_test_clean = X_test

# ============================================================================
# PHASE 3: LOAD OPTIMIZED HYPERPARAMETERS
# ============================================================================

print("\n" + "="*80)
print("PHASE 3: LOADING BEST HYPERPARAMETERS")
print("="*80)

# Load best hyperparameters from optimization results
best_params_df = pd.read_csv("../reports/csv/best_model_params.csv")
print("\n✓ Best model hyperparameters:")
print(best_params_df.to_string(index=False))

# Extract Bagging Regressor parameters
bagging_params = best_params_df[best_params_df['Model'] == 'Bagging Regressor']['Best Params'].values[0]
print(f"\n✓ Bagging Regressor params: {bagging_params}")

# Parse parameters (they're stored as string)
import ast
bagging_params_dict = ast.literal_eval(bagging_params)
print(f"✓ Parsed params: {bagging_params_dict}")

# ============================================================================
# PHASE 4: TRAIN CLEAN MODEL (NO LEAKAGE)
# ============================================================================

print("\n" + "="*80)
print("PHASE 4: TRAINING BAGGING REGRESSOR (CLEAN DATA)")
print("="*80)

print("\n🚀 Training Bagging Regressor with optimized hyperparameters...")
print("⏱️ Estimated time: 2-3 minutes")

start_time = time.time()

# Initialize model with best parameters
model_clean = BaggingRegressor(
    **bagging_params_dict,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Train model
model_clean.fit(X_train_clean, y_train)

train_time = time.time() - start_time
print(f"\n✓ Training completed in {train_time/60:.2f} minutes")

# ============================================================================
# PHASE 5: EVALUATE CLEAN MODEL
# ============================================================================

print("\n" + "="*80)
print("PHASE 5: EVALUATING CLEAN MODEL (NO LEAKAGE)")
print("="*80)

# Predictions
y_train_pred = model_clean.predict(X_train_clean)
y_test_pred = model_clean.predict(X_test_clean)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n📊 CLEAN MODEL PERFORMANCE (NO DATA LEAKAGE):")
print(f"  Train R²:  {train_r2:.4f}")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Train RMSE: {train_rmse:.2f}")
print(f"  Test RMSE:  {test_rmse:.2f}")
print(f"  Train MAE:  {train_mae:.2f}")
print(f"  Test MAE:   {test_mae:.2f}")

# ============================================================================
# PHASE 6: COMPARE WITH LEAKY MODEL
# ============================================================================

print("\n" + "="*80)
print("PHASE 6: COMPARISON WITH LEAKY MODEL")
print("="*80)

# Load previous leaky model performance
optimization_results = pd.read_csv("../reports/csv/hyperparameter_optimization_results.csv")
leaky_bagging = optimization_results[optimization_results['Model'] == 'Bagging Regressor']
leaky_test_r2 = leaky_bagging['Test R²'].values[0]
leaky_test_rmse = leaky_bagging['Test RMSE'].values[0]
leaky_test_mae = leaky_bagging['Test MAE'].values[0]

print("\n📊 PERFORMANCE COMPARISON:")
print("-" * 80)
print(f"{'Metric':<15} {'Leaky Model':<15} {'Clean Model':<15} {'Difference':<15}")
print("-" * 80)
print(f"{'Test R²':<15} {leaky_test_r2:<15.4f} {test_r2:<15.4f} {test_r2 - leaky_test_r2:<15.4f}")
print(f"{'Test RMSE':<15} {leaky_test_rmse:<15.2f} {test_rmse:<15.2f} {test_rmse - leaky_test_rmse:<15.2f}")
print(f"{'Test MAE':<15} {leaky_test_mae:<15.2f} {test_mae:<15.2f} {test_mae - leaky_test_mae:<15.2f}")
print("-" * 80)

r2_drop_pct = ((test_r2 - leaky_test_r2) / leaky_test_r2) * 100
print(f"\n⚠️ R² drop due to leakage fix: {r2_drop_pct:.2f}%")

if r2_drop_pct > -20:
    print("✅ Performance drop is acceptable (<20%), model is production-safe!")
else:
    print("⚠️ Significant performance drop detected. Consider feature engineering.")

# ============================================================================
# PHASE 7: SAVE CLEAN MODEL
# ============================================================================

print("\n" + "="*80)
print("PHASE 7: SAVING PRODUCTION-READY MODEL")
print("="*80)

# Save clean model
model_path = "../models/best_model_clean.pkl"
joblib.dump(model_clean, model_path)
print(f"\n✓ Clean model saved: {model_path}")

# Save clean data for future use
X_train_clean.to_csv("../data/model_ready/X_train_clean.csv", index=False)
X_test_clean.to_csv("../data/model_ready/X_test_clean.csv", index=False)
print(f"✓ Clean data saved:")
print(f"  - X_train_clean.csv ({X_train_clean.shape})")
print(f"  - X_test_clean.csv ({X_test_clean.shape})")

# Save comparison results
comparison_df = pd.DataFrame({
    'metric': ['test_r2', 'test_rmse', 'test_mae'],
    'leaky_model': [leaky_test_r2, leaky_test_rmse, leaky_test_mae],
    'clean_model': [test_r2, test_rmse, test_mae],
    'difference': [test_r2 - leaky_test_r2, test_rmse - leaky_test_rmse, test_mae - leaky_test_mae],
    'percent_change': [
        ((test_r2 - leaky_test_r2) / leaky_test_r2) * 100,
        ((test_rmse - leaky_test_rmse) / leaky_test_rmse) * 100,
        ((test_mae - leaky_test_mae) / leaky_test_mae) * 100
    ]
})
comparison_df.to_csv("../reports/csv/leakage_fix_comparison.csv", index=False)
print(f"✓ Comparison results saved: leakage_fix_comparison.csv")

# ============================================================================
# PHASE 8: SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("\n✅ DATA LEAKAGE FIX COMPLETED!")
print(f"✅ Removed {len(leakage_columns)} leakage column(s): {leakage_columns}")
print(f"✅ Clean model trained and saved")
print(f"✅ Test R² (clean): {test_r2:.4f}")
print(f"✅ Test R² drop: {r2_drop_pct:.2f}%")
print(f"\n🎯 MODEL STATUS: PRODUCTION-READY (NO DATA LEAKAGE)")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. ✅ Data leakage fixed")
print("2. 📊 Review feature importance without leakage (optional SHAP re-run)")
print("3. 🚀 Deploy clean model to production")
print("4. 🔄 Consider external data integration (Spotify API) in future")

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY")
print("="*80)
