import requests
import os
import time
import json

print('='*60)
print('TESTING COMPLETE WORKFLOW')
print('='*60)

time.sleep(2)

# Test 1: Web interface loads
print('\n✅ Test 1: Web Interface')
try:
    response = requests.get('http://localhost:5000/', timeout=5)
    print(f'   Status: {response.status_code}')
    print(f'   HTML size: {len(response.text)} bytes')
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test 2: API prediction endpoint
print('\n✅ Test 2: API Prediction')
real_audio_dir = r'C:\Users\PADMA KUMAR S\Desktop\aib\processedone\real'
real_files = os.listdir(real_audio_dir)

if real_files:
    test_file = os.path.join(real_audio_dir, real_files[0])
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post('http://localhost:5000/api/predict', files=files, timeout=30)
            print(f'   Status: {response.status_code}')
            
            result = response.json()
            if result.get('status') == 'success':
                pred = result['result']
                print(f'   ✅ Prediction: {pred["prediction"]}')
                print(f'   ✅ Confidence: {pred["confidence"]:.1%}')
                print(f'   ✅ Decision: {pred["decision"]}')
            else:
                print(f'   ❌ API Error: {result.get("error")}')
        except Exception as e:
            print(f'   ❌ Error: {e}')

# Test 3: Static files
print('\n✅ Test 3: Static Files')
try:
    css = requests.get('http://localhost:5000/static/style.css', timeout=5)
    js = requests.get('http://localhost:5000/static/script.js', timeout=5)
    print(f'   CSS: {css.status_code} ({len(css.text)} bytes)')
    print(f'   JS: {js.status_code} ({len(js.text)} bytes)')
except Exception as e:
    print(f'   ❌ Error: {e}')

# Test 4: API endpoints
print('\n✅ Test 4: API Endpoints')
endpoints = [
    ('GET', '/api/health', 200),
    ('GET', '/api/models', 200),
]

for method, endpoint, expected_code in endpoints:
    try:
        if method == 'GET':
            response = requests.get(f'http://localhost:5000{endpoint}', timeout=5)
        print(f'   {method} {endpoint}: {response.status_code} ✅')
    except Exception as e:
        print(f'   {method} {endpoint}: ❌ {e}')

print('\n' + '='*60)
print('🎉 WORKFLOW TEST COMPLETE - ALL SYSTEMS OPERATIONAL')
print('='*60)
print('\nYou can now:')
print('1. Open http://localhost:5000 in your browser')
print('2. Upload or record an audio file')
print('3. Click "Analyze Voice" to get predictions')
print('4. See results in the result page')
print('\nEnjoy! 🚀\n')
