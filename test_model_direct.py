#!/usr/bin/env python
"""Direct model test"""
import os
import sys

# Make sure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print(f"Working directory: {os.getcwd()}")
print(f"rf_model.pkl exists: {os.path.exists('rf_model.pkl')}")
print(f"rf_model.pkl exists (absolute): {os.path.exists(os.path.join(script_dir, 'rf_model.pkl'))}")

import joblib
import numpy as np

try:
    print("\nLoading model...")
    model = joblib.load('rf_model.pkl')
    print(f"✅ Model loaded: {model}")
    print(f"   Type: {type(model)}")
    print(f"   Classes: {model.classes_ if hasattr(model, 'classes_') else 'N/A'}")
    
    # Test a prediction with dummy data
    dummy_features = np.random.randn(1, 13)  # MFCC has 13 features
    pred = model.predict(dummy_features)
    print(f"✅ Test prediction works: {pred}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
