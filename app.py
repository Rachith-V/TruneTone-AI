from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import joblib
import os
from werkzeug.utils import secure_filename
import traceback
from pydub import AudioSegment
import tempfile

# Configure pydub to use ffmpeg from imageio-ffmpeg if available
ffmpeg_path = None
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    # Add to PATH so pydub can find it
    os.environ['PATH'] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
    print(f"[FFMPEG] Configured pydub to use ffmpeg: {ffmpeg_path}")
except Exception as e:
    print(f"[FFMPEG] Could not configure imageio-ffmpeg: {e}")

app = Flask(__name__)

# ==============================
# SETTINGS
# ==============================
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(SCRIPT_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'webm', 'ogg', 'flac'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Use absolute path for model
MODEL_PATH = os.path.join(SCRIPT_DIR, "rf_model.pkl")
SR = 16000
DURATION = 3
SAMPLES_PER_TRACK = SR * DURATION

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODEL
# ==============================
print(f"\n[STARTUP] Starting model loading...")
print(f"[STARTUP] SCRIPT_DIR = {SCRIPT_DIR}")
print(f"[STARTUP] MODEL_PATH = {MODEL_PATH}")
print(f"[STARTUP] File exists: {os.path.exists(MODEL_PATH)}")

model = None
MODEL_LOADED = False

try:
    print(f"[STARTUP] Checking file existence...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    
    print(f"[STARTUP] File exists, attempting joblib.load()...")
    model = joblib.load(MODEL_PATH)
    print(f"[STARTUP] joblib.load() completed successfully")
    
    MODEL_LOADED = True
    print(f"[MODEL] Model loaded successfully: {MODEL_PATH}")
except FileNotFoundError as e:
    print(f"[MODEL] FileNotFoundError: {e}")
except Exception as e:
    print(f"[MODEL] Error loading model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print(f"[STARTUP] Model loading complete. MODEL_LOADED={MODEL_LOADED}, model is None={model is None}\n")

# ==============================
# HELPER FUNCTIONS
# ==============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features_from_array(audio):
    """Extract MFCC features from audio array"""
    try:
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0)
        
        return mfcc.reshape(1, -1)
    except Exception as e:
        raise Exception(f"Feature extraction failed: {str(e)}")

def split_audio(audio, sr, chunk_duration=3):
    """Split audio into fixed-length chunks"""
    chunk_length = sr * chunk_duration
    chunks = []
    
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i+chunk_length]
        
        if len(chunk) < chunk_length:
            break
        
        chunks.append(chunk)
    
    return chunks

def decision_logic(confidence):
    """Determine decision based on confidence score"""
    if confidence >= 0.85:
        return "High Confidence"
    elif confidence >= 0.65:
        return "Needs Review"
    else:
        return "Uncertain"

import subprocess

def convert_audio_to_wav(file_path):
    """Convert non-WAV audio files to WAV format using ffmpeg"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.wav':
            print(f"    [CONVERT] Already WAV format")
            return file_path

        print(f"    [CONVERT] Converting {file_ext} to WAV using ffmpeg...")
        
        wav_path = os.path.join(UPLOAD_FOLDER, f"temp_{os.getpid()}.wav")

        # Use full path to ffmpeg
        ffmpeg_exe = ffmpeg_path if ffmpeg_path else "ffmpeg"
        
        command = [
            ffmpeg_exe,
            "-y",  # overwrite
            "-i", file_path,
            "-ar", "16000",
            "-ac", "1",
            "-q:a", "9",
            wav_path
        ]

        print(f"    Running ffmpeg: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"    ffmpeg exit code: {result.returncode}")
        if result.stderr:
            print(f"    ffmpeg stderr: {result.stderr[:500]}")

        if result.returncode != 0:
            raise Exception(f"FFmpeg failed with exit code {result.returncode}")

        if not os.path.exists(wav_path):
            raise Exception(f"Output WAV file not created at {wav_path}")

        # Check file size
        file_size = os.path.getsize(wav_path)
        print(f"    [CONVERT] Converted successfully to {wav_path} ({file_size} bytes)")
        
        # Remove original file
        if os.path.exists(file_path):
            os.remove(file_path)

        return wav_path

    except subprocess.TimeoutExpired:
        print(f"  [CONVERT] FFmpeg conversion timed out")
        return file_path
    except Exception as e:
        print(f"  [CONVERT] FFmpeg conversion error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return file_path

def predict_from_file(file_path):
    """Predict voice type from audio file"""
    try:
        print(f"    📝 Starting prediction for: {file_path}")
        
        if not MODEL_LOADED:
            return None, "Model not loaded"
        
        if model is None:
            return None, "Model object is None"
        
        # Check file exists
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        
        file_size = os.path.getsize(file_path)
        print(f"    📄 File size: {file_size} bytes")
        
        # Convert non-WAV formats to WAV
        print(f"    Converting audio...")
        file_path = convert_audio_to_wav(file_path)
        
        if not os.path.exists(file_path):
            return None, f"Audio conversion failed - file not found: {file_path}"
        
        print(f"    Loading audio with librosa...")
        # Load audio
        try:
            audio, sr = librosa.load(file_path, sr=SR)
        except Exception as load_err:
            print(f"    [ERROR] Librosa load failed: {type(load_err).__name__}: {load_err}")
            return None, f"Failed to load audio: {str(load_err)}"
        
        print(f"    Audio loaded: {len(audio)} samples at {sr} Hz")
        
        # Ensure minimum length
        if len(audio) < SAMPLES_PER_TRACK:
            # Pad if too short
            pad_length = SAMPLES_PER_TRACK - len(audio)
            audio = np.pad(audio, (0, pad_length))
            print(f"    Padded audio to: {len(audio)} samples")
        
        # Split into chunks
        print(f"    Splitting into chunks...")
        chunks = split_audio(audio, sr)
        
        if len(chunks) == 0:
            return None, "Audio too short"
        
        print(f"    Analyzing {len(chunks)} chunks...")
        predictions = []
        confidences = []
        
        # Predict for each chunk
        for i, chunk in enumerate(chunks):
            features = extract_features_from_array(chunk)
            pred = model.predict(features)[0]
            prob = np.max(model.predict_proba(features))
            
            predictions.append(pred)
            confidences.append(prob)
            print(f"      Chunk {i+1}: pred={pred}, conf={prob:.4f}")
        
        # Final decision
        final_pred = max(set(predictions), key=predictions.count)
        final_conf = np.mean(confidences)
        
        label = "Human Voice" if final_pred == 0 else "Machine Voice"
        decision = decision_logic(final_conf)
        
        print(f"    [SUCCESS] Prediction complete: {label} ({final_conf:.4f})")
        
        return {
            'prediction': label,
            'class': int(final_pred),
            'confidence': float(final_conf),
            'decision': decision,
            'chunks_analyzed': len(chunks)
        }, None
        
    except Exception as e:
        print(f"    [EXCEPTION] Prediction exception: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Prediction error: {type(e).__name__}: {str(e)}"

# ==============================
# ROUTES
# ==============================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/result.html')
def result():
    """Result page"""
    return render_template('result.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    print(f"[HEALTH] Health check called")
    print(f"[HEALTH] MODEL_LOADED: {MODEL_LOADED}")
    print(f"[HEALTH] model is None: {model is None}")
    print(f"[HEALTH] model object: {model}")
    
    status = 'OK' if MODEL_LOADED else 'ERROR'
    http_code = 200 if MODEL_LOADED else 503
    
    response = {
        'status': status,
        'model_loaded': MODEL_LOADED,
        'model_path': MODEL_PATH,
        'model_is_none': model is None,
        'model_object_type': str(type(model)),
        'api_version': '1.0'
    }
    print(f"[HEALTH] Returning: {response}")
    return jsonify(response), http_code

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict voice type from uploaded file"""
    try:
        print(f"\n[PREDICT] /api/predict request received")
        print(f"[PREDICT] Files in request: {list(request.files.keys())}")
        print(f"[PREDICT] MODEL_LOADED: {MODEL_LOADED}")
        print(f"[PREDICT] model is None: {model is None}")
        print(f"[PREDICT] model object: {model}")
        
        # Check if model is loaded
        if not MODEL_LOADED or model is None:
            print("[PREDICT] Model not loaded - returning error")
            resp = {
                'error': 'Model not loaded',
                'status': 'error',
                'MODEL_LOADED': MODEL_LOADED,
                'model_is_none': model is None,
                'model_object': str(model)
            }
            print(f"[PREDICT] Returning: {resp}")
            return jsonify(resp), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            print("  ❌ No 'file' field in request")
            return jsonify({
                'error': 'No file provided',
                'status': 'error'
            }), 400
        
        file = request.files['file']
        print(f"  📄 File: {file.filename}")
        
        # Check if file is selected
        if file.filename == '':
            print("  ❌ Empty filename")
            return jsonify({
                'error': 'No file selected',
                'status': 'error'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            print(f"  ❌ File type not allowed: {file.filename}")
            return jsonify({
                'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                'status': 'error'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"  [FILE] Saving to: {filepath}")
        file.save(filepath)
        print(f"  [FILE] File saved, size: {os.path.getsize(filepath)} bytes")
        
        try:
            # Make prediction
            print(f"  [PREDICT] Making prediction...")
            result, error = predict_from_file(filepath)
            
            if error:
                print(f"  [ERROR] Prediction error: {error}")
                return jsonify({
                    'error': error,
                    'status': 'error'
                }), 400
            
            print(f"  [SUCCESS] Prediction success: {result}")
            return jsonify({
                'status': 'success',
                'filename': filename,
                'result': result
            }), 200
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"  [CLEANUP] Cleaned up: {filepath}")
    
    except Exception as e:
        print(f"  [EXCEPTION] Exception in predict(): {type(e).__name__}: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        print(f"  Full traceback:\n{tb}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error',
            'exception_type': type(e).__name__,
            'traceback': tb
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict voice type for multiple files"""
    try:
        if not MODEL_LOADED:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
        
        results = []
        errors = []
        
        # Check if files are in request
        if 'files' not in request.files:
            return jsonify({
                'error': 'No files provided',
                'status': 'error'
            }), 400
        
        files = request.files.getlist('files')
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                errors.append({
                    'filename': file.filename,
                    'error': 'File type not allowed'
                })
                continue
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                result, error = predict_from_file(filepath)
                
                if error:
                    errors.append({
                        'filename': filename,
                        'error': error
                    })
                else:
                    results.append({
                        'filename': filename,
                        'result': result
                    })
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'total_files': len(files),
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    models = {
        'rf_model.pkl': 'Random Forest',
        'gb_model.pkl': 'Gradient Boosting',
        'svm_model.pkl': 'Support Vector Machine',
        'bag_model.pkl': 'Bagging',
        'logistic_model.pkl': 'Logistic Regression'
    }
    
    available_models = []
    for model_file, model_name in models.items():
        if os.path.exists(model_file):
            available_models.append({
                'file': model_file,
                'name': model_name,
                'available': True
            })
    
    return jsonify({
        'status': 'success',
        'current_model': 'Random Forest (rf_model.pkl)',
        'available_models': available_models
    }), 200

# ==============================
# ERROR HANDLERS
# ==============================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large. Maximum size: 50MB',
        'status': 'error'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

# ==============================
# RUN APP
# ==============================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Voice Authentication API")
    print("="*60)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"🤖 Model loaded: {MODEL_LOADED}")
    print(f"📄 Model path: {MODEL_PATH}")
    if MODEL_LOADED:
        print(f"[MODEL] Model type: Random Forest Classifier")
        print(f"[MODEL] Accuracy: 93.79%")
    else:
        print(f"[MODEL] MODEL FAILED TO LOAD - Check model path above")
    print(f"🌐 Server running on http://0.0.0.0:5000")
    print(f"🌐 Open http://localhost:5000 in your browser")
    print("="*60 + "\n")
    
    if not MODEL_LOADED:
        print("[WARNING] Model not loaded. API will not work properly.")
        print(f"   Please check: {MODEL_PATH}")
    
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
