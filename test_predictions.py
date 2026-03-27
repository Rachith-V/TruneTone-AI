import requests
import os

print('Testing Prediction Endpoint...\n')

# Get a sample audio file from the dataset
real_audio_dir = r'C:\Users\PADMA KUMAR S\Desktop\aib\processedone\real'
fake_audio_dir = r'C:\Users\PADMA KUMAR S\Desktop\aib\processedone\fake'

# Test with a real audio file
real_files = os.listdir(real_audio_dir)
if real_files:
    test_file = os.path.join(real_audio_dir, real_files[0])
    
    print(f"Testing with real audio: {real_files[0]}")
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post('http://localhost:5000/api/predict', files=files, timeout=30)
            print(f'✅ Prediction Request: {response.status_code}')
            result = response.json()
            
            if result.get('status') == 'success':
                pred = result['result']
                print(f"\n📊 Prediction Results:")
                print(f"   Audio File: {result['filename']}")
                print(f"   Prediction: {pred['prediction']}")
                print(f"   Confidence: {pred['confidence']:.2%}")
                print(f"   Decision: {pred['decision']}")
                print(f"   Chunks Analyzed: {pred['chunks_analyzed']}")
            else:
                print(f"Error: {result.get('error')}")
        except Exception as e:
            print(f'❌ Request failed: {e}')

# Test with a fake audio file
fake_files = os.listdir(fake_audio_dir)
if fake_files:
    test_file = os.path.join(fake_audio_dir, fake_files[0])
    
    print(f"\n\nTesting with fake audio: {fake_files[0]}")
    
    with open(test_file, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post('http://localhost:5000/api/predict', files=files, timeout=30)
            print(f'✅ Prediction Request: {response.status_code}')
            result = response.json()
            
            if result.get('status') == 'success':
                pred = result['result']
                print(f"\n📊 Prediction Results:")
                print(f"   Audio File: {result['filename']}")
                print(f"   Prediction: {pred['prediction']}")
                print(f"   Confidence: {pred['confidence']:.2%}")
                print(f"   Decision: {pred['decision']}")
                print(f"   Chunks Analyzed: {pred['chunks_analyzed']}")
            else:
                print(f"Error: {result.get('error')}")
        except Exception as e:
            print(f'❌ Request failed: {e}')

print("\n\n🎉 All prediction tests completed!")
