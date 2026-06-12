"""
Model ve pipeline özelliklerini kontrol et
"""
import joblib
import pandas as pd

print("="*80)
print("MODEL FEATURE ANALİZİ")
print("="*80)

# Model'i yükle
model = joblib.load('models/best_model_clean.pkl')
print(f"\n1. MODEL DETAYLARI:")
print(f"   • Model tipi: {type(model).__name__}")
print(f"   • feature_names_in_ var mı: {hasattr(model, 'feature_names_in_')}")

if hasattr(model, 'feature_names_in_'):
    features = model.feature_names_in_
    print(f"   • Özellik sayısı: {len(features)}")
    print(f"\n2. İLK 20 ÖZELLİK:")
    for i, feat in enumerate(features[:20], 1):
        print(f"      {i}. {feat}")
    
    print(f"\n3. SON 10 ÖZELLİK:")
    for i, feat in enumerate(features[-10:], len(features)-9):
        print(f"      {i}. {feat}")
    
    print(f"\n4. ÖZEL KONTROLLER:")
    print(f"   • 'Unnamed: 0' feature'da mı: {'Unnamed: 0' in features}")
    print(f"   • 'genre_' ile başlayan: {sum(1 for f in features if f.startswith('genre_'))}")
    print(f"   • İlk genre feature: {[f for f in features if f.startswith('genre_')][:5]}")

# Pipeline'ı yükle
print(f"\n" + "="*80)
print("PREPROCESSING PIPELINE ANALİZİ")
print("="*80)
pipeline = joblib.load('models/preprocessing_pipeline_scaler.pkl')
print(f"\n5. PIPELINE DETAYLARI:")
print(f"   • Pipeline tipi: {type(pipeline).__name__}")
print(f"   • Pipeline yapısı:\n{pipeline}")

# X_train_clean.csv'nin kolonlarını oku
print(f"\n" + "="*80)
print("X_TRAIN_CLEAN.CSV KOLONLARI")
print("="*80)
X_train = pd.read_csv('data/model_ready/X_train_clean.csv', nrows=1)
print(f"\n6. X_TRAIN_CLEAN.CSV:")
print(f"   • Sütun sayısı: {len(X_train.columns)}")
print(f"   • İlk 20 sütun:")
for i, col in enumerate(X_train.columns[:20], 1):
    print(f"      {i}. {col}")

print(f"\n7. ÖZEL KONTROLLER:")
print(f"   • 'Unnamed: 0' var mı: {'Unnamed: 0' in X_train.columns}")
print(f"   • 'genre_' ile başlayan: {sum(1 for c in X_train.columns if c.startswith('genre_'))}")

print("\n" + "="*80)
print("SONUÇ")
print("="*80)
