#!/usr/bin/env python
"""Replicate the exact Flask startup to diagnose the model loading issue"""
import os
import sys
import joblib

# Replicate the exact code from app.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"SCRIPT_DIR: {SCRIPT_DIR}")

MODEL_PATH = os.path.join(SCRIPT_DIR, "rf_model.pkl")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")

model = None
MODEL_LOADED = False

print("\nAttempting to load model (replicating app.py startup)...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    print(f"  File exists, loading with joblib...")
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
    print(f"  Model type: {type(model)}")
    print(f"  MODEL_LOADED flag: {MODEL_LOADED}")
    
except FileNotFoundError as e:
    print(f"❌ FileNotFoundError: {e}")
except Exception as e:
    print(f"❌ Error loading model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"\nFinal state:")
print(f"  MODEL_LOADED = {MODEL_LOADED}")
print(f"  model is None = {model is None}")
