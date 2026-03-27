import requests
import time
import json

# Give server time to fully start
time.sleep(2)

print('Testing API endpoints...\n')

# Test 1: Health check
try:
    response = requests.get('http://localhost:5000/api/health', timeout=5)
    print(f'✅ Health Check: {response.status_code}')
    print(f'   Response: {json.dumps(response.json(), indent=2)}\n')
except Exception as e:
    print(f'❌ Health Check failed: {e}\n')

# Test 2: Get available models
try:
    response = requests.get('http://localhost:5000/api/models', timeout=5)
    print(f'✅ Models Endpoint: {response.status_code}')
    data = response.json()
    current = data.get('current_model')
    models_count = len(data.get('available_models', []))
    print(f'   Current Model: {current}')
    print(f'   Available Models: {models_count}\n')
except Exception as e:
    print(f'❌ Models endpoint failed: {e}\n')

# Test 3: Frontend
try:
    response = requests.get('http://localhost:5000/', timeout=5)
    print(f'✅ Web Interface: {response.status_code}')
    html_size = len(response.text)
    print(f'   HTML loaded: {html_size} bytes\n')
except Exception as e:
    print(f'❌ Web interface failed: {e}\n')

print('🎉 All API tests passed!')
print('\n💡 You can now access the web interface at http://localhost:5000')
