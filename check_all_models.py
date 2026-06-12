"""
Tüm model dosyalarının feature'larını kontrol et
"""
import joblib
import os

models_dir = 'models'
model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and 'model' in f]

print("="*80)
print("TÜM MODEL DOSYALARI ANALİZİ")
print("="*80)

for model_file in model_files:
    print(f"\n📁 {model_file}")
    print("-" * 80)
    
    try:
        model_path = os.path.join(models_dir, model_file)
        model = joblib.load(model_path)
        
        print(f"   • Model tipi: {type(model).__name__}")
        print(f"   • Has feature_names_in_: {hasattr(model, 'feature_names_in_')}")
        
        if hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
            print(f"   • Feature sayısı: {len(features)}")
            print(f"   • İlk 5 feature: {list(features[:5])}")
            print(f"   • Son 3 feature: {list(features[-3:])}")
            print(f"   • 'Unnamed: 0' var mı: {'Unnamed: 0' in features}")
            print(f"   • 'genre_' ile başlayan: {sum(1 for f in features if f.startswith('genre_'))}")
        else:
            print("   ⚠️ feature_names_in_ yok!")
            
    except Exception as e:
        print(f"   ❌ HATA: {e}")

print("\n" + "="*80)
