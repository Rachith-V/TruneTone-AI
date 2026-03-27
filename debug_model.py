import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_model.pkl")

print(f"Checking model at: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")
print(f"File size: {os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'N/A'} bytes")

try:
    print("\nAttempting to load model...")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}")
    print(f"Details: {e}")
    import traceback
    traceback.print_exc()
