import os
import librosa
import numpy as np
import soundfile as sf

# SETTINGS
base_path = r"C:\Users\PADMA KUMAR S\Desktop\aib\dataset"
output_path = r"C:\Users\PADMA KUMAR S\Desktop\aib\processedone"

TARGET_DURATION = 3  # seconds
SR = 16000

# Create output folders
os.makedirs(output_path + "/real", exist_ok=True)
os.makedirs(output_path + "/fake", exist_ok=True)

def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SR)
    target_length = TARGET_DURATION * SR

    if len(audio) > target_length:
        audio = audio[:target_length]  # trim
    else:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding))  # pad

    return audio

# 🔹 Process REAL files
real_path = base_path + "/real"

for file in os.listdir(real_path):
    file_path = os.path.join(real_path, file)

    processed_audio = process_audio(file_path)

    save_path = os.path.join(output_path, "real", file)
    sf.write(save_path, processed_audio, SR)

# 🔹 Process FAKE files
fake_path = base_path + "/fake"

for file in os.listdir(fake_path):
    file_path = os.path.join(fake_path, file)

    processed_audio = process_audio(file_path)

    save_path = os.path.join(output_path, "fake", file)
    sf.write(save_path, processed_audio, SR)

print("✅ All files converted to 3 seconds")