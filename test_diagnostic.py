#!/usr/bin/env python
"""Test script to diagnose the API issue"""
import sys
sys.path.insert(0, '.')

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔍 DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Check model loading
print("\n1️⃣ Testing model loading...")
try:
    import joblib
    MODEL_PATH = "rf_model.pkl"
    if not os.path.exists(MODEL_PATH):
        print(f"   ❌ Model file not found: {MODEL_PATH}")
    else:
        print(f"   ✅ Model file exists: {os.path.getsize(MODEL_PATH)} bytes")
        model = joblib.load(MODEL_PATH)
        print(f"   ✅ Model loaded: {type(model)}")
except Exception as e:
    print(f"   ❌ Model load error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check ffmpeg
print("\n2️⃣ Testing ffmpeg...")
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"   ✅ ffmpeg found: {ffmpeg_exe}")
except Exception as e:
    print(f"   ❌ ffmpeg error: {e}")

# Test 3: Test Flask app directly
print("\n3️⃣ Testing Flask app startup...")
try:
    from app import app, MODEL_LOADED, model
    print(f"   ✅ Flask app imported")
    print(f"   ✅ MODEL_LOADED: {MODEL_LOADED}")
    print(f"   ✅ model object: {model is not None}")
    
    if not MODEL_LOADED:
        print("   ⚠️  Model NOT loaded in app.py!")
    
except Exception as e:
    print(f"   ❌ Flask error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("If model loading fails above, the prediction endpoint will fail!")
print("=" * 60)
