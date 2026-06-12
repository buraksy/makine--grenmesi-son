import pandas as pd
import json

# Training data'nın feature names'ini al
df = pd.read_csv('data/model_ready/X_train_clean.csv')
feature_names = df.columns.tolist()

print("=" * 80)
print(f"TOPLAM FEATURE SAYISI: {len(feature_names)}")
print("=" * 80)
print("\nTÜM FEATURES:")
for i, col in enumerate(feature_names, 1):
    print(f"{i:3d}. {col}")

# JSON formatında kaydet
with open('feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
    
print("\n✅ feature_names.json dosyası oluşturuldu")
