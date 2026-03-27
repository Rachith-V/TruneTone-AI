import numpy as np
import librosa
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import os
from pydub import AudioSegment


# SETTINGS

MODEL_PATH = "rf_model.pkl"   
SR = 16000
DURATION = 3
SAMPLES_PER_TRACK = SR * DURATION

def preprocess_audio(input_path):
    # Convert any format → WAV
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(SR).set_channels(1)
    audio.export("temp.wav", format="wav")

    # Load
    audio, sr = librosa.load("temp.wav", sr=SR)

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Fix length
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    else:
        repeat = int(np.ceil(SAMPLES_PER_TRACK / len(audio)))
        audio = np.tile(audio, repeat)[:SAMPLES_PER_TRACK]

    return audio


# LOAD MODEL

model = joblib.load(MODEL_PATH)


# FEATURE EXTRACTION

def extract_features_from_file(file_path):
    audio = preprocess_audio(file_path)

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1, -1)

# DECISION LAYER

def decision_logic(confidence):
    if confidence >= 0.85:
        return "✅ High Confidence"
    elif confidence >= 0.65:
        return "❓ Needs Review"
    else:
        return "⚠️ Uncertain"

# PREDICT FUNCTION

def split_audio(audio, sr, chunk_duration=3):
    chunk_length = sr * chunk_duration
    chunks = []

    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i+chunk_length]

        if len(chunk) < chunk_length:
            break

        chunks.append(chunk)

    return chunks
def predict_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SR)

    chunks = split_audio(audio, sr)

    predictions = []
    confidences = []

    for chunk in chunks:
        features = extract_features_from_file(file_path)

        pred = model.predict(features)[0]
        prob = np.max(model.predict_proba(features))

        predictions.append(pred)
        confidences.append(prob)

    # final decision
    final_pred = max(set(predictions), key=predictions.count)
    final_conf = np.mean(confidences)

    label = "Human Voice" if final_pred == 0 else "Machine Voice"
    decision = decision_logic(final_conf)

    print("\n===== RESULT =====")
    print(f"Prediction      : {label}")
    print(f"Confidence      : {final_conf:.2f}")
    print(f"Final Decision  : {decision}")
    print("==================")

def record_audio(filename="temp.wav", duration=5, fs=16000):
    print("\n🎤 Recording... Speak now")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    write(filename, fs, audio)
    print("✅ Recording complete")

    return filename

# RUN

if __name__ == "__main__":
    print("\nSelect option:")
    print("1 → Test using audio file")
    print("2 → Real-time mic detection")

    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        file_path = input("Enter path to audio file: ").strip().replace('"', '')
        predict_audio(file_path)

    elif choice == "2":
        temp_file = record_audio()
        predict_audio(temp_file)

        # Optional: delete temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

    else:
        print("Invalid choice")




