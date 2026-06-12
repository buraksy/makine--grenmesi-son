"""Model dosyalarını kontrol et — sınıflandırma vs regresyon ayrımı."""
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

print("=" * 80)
print("MODEL KONTROL (Sınıflandırma odaklı)")
print("=" * 80)

models = {
    "best_classifier_rf.pkl (GÜNCEL)": MODELS_DIR / "best_classifier_rf.pkl",
    "best_model_clean.pkl (ARŞİV-regresyon)": MODELS_DIR / "best_model_clean.pkl",
    "best_model_optimized.pkl (ARŞİV-regresyon)": MODELS_DIR / "best_model_optimized.pkl",
    "final_model.pkl (ARŞİV)": MODELS_DIR / "final_model.pkl",
}

for name, path in models.items():
    try:
        model = joblib.load(path)
        print(f"\n📦 {name}")
        print(f"   Type: {type(model).__name__}")
        print(f"   Features: {getattr(model, 'n_features_in_', 'Unknown')}")
        if hasattr(model, "predict_proba"):
            print("   ✅ Sınıflandırma (predict_proba mevcut)")
        else:
            print("   ⚠️ Regresyon veya predict_proba yok")
        if hasattr(model, "classes_"):
            print(f"   Classes: {model.classes_}")
    except FileNotFoundError:
        print(f"\n⏳ {name}: dosya bulunamadı")
    except Exception as e:
        print(f"\n❌ {name}: {e}")

scaler_path = MODELS_DIR / "preprocessing_pipeline_scaler.pkl"
if scaler_path.exists():
    scaler = joblib.load(scaler_path)
    print(f"\n📦 preprocessing_pipeline_scaler.pkl")
    print(f"   Type: {type(scaler).__name__}")
    print(f"   Features: {getattr(scaler, 'n_features_in_', 'Unknown')}")

print("\n" + "=" * 80)
print("Güncel model: models/best_classifier_rf.pkl")
print("Detay: README.md")
print("=" * 80)
