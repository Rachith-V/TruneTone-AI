import requests
import os
import time

time.sleep(2)

print('Testing full prediction workflow...\n')

# Get a sample audio file
real_audio_dir = r'C:\Users\PADMA KUMAR S\Desktop\aib\processedone\real'
real_files = os.listdir(real_audio_dir)

if real_files:
    test_file = os.path.join(real_audio_dir, real_files[0])
    print(f'Testing with: {real_files[0]}\n')
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post('http://localhost:5000/api/predict', files=files, timeout=30)
            print(f'Status Code: {response.status_code}')
            
            result = response.json()
            status = result.get('status')
            print(f'Response Status: {status}')
            
            if status == 'success':
                pred = result['result']
                print(f'\n✅ Prediction: {pred["prediction"]}')
                print(f'✅ Confidence: {pred["confidence"]:.1%}')
                print(f'✅ Decision: {pred["decision"]}')
                print(f'\n🎉 Full workflow is working!')
            else:
                print(f'❌ Error: {result.get("error")}')
        except Exception as e:
            print(f'❌ Request failed: {e}')
