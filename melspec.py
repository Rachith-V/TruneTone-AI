import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def load_and_process(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_db, sr


def compare_spectrograms(file1, file2):
    mel1, sr1 = load_and_process(file1)
    mel2, sr2 = load_and_process(file2)

    plt.figure(figsize=(12, 6))

    # 🔹 Human
    plt.subplot(2, 1, 1)
    librosa.display.specshow(mel1, sr=sr1, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Human Voice Spectrogram")

    # 🔹 Machine
    plt.subplot(2, 1, 2)
    librosa.display.specshow(mel2, sr=sr2, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Machine Voice Spectrogram")

    plt.tight_layout()
    plt.show()


# RUN

if __name__ == "__main__":
    file1 = input("Enter HUMAN audio path: ").strip().replace('"','')
    file2 = input("Enter MACHINE audio path: ").strip().replace('"','')

    compare_spectrograms(file1, file2)