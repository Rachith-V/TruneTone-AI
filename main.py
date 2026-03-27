import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_PATH = r"C:\Users\PADMA KUMAR S\Desktop\aib\processedone"
SR = 16000
DURATION = 3
SAMPLES_PER_TRACK = SR * DURATION


# FUNCTION: Load & Extract MFCC

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SR)
        audio = audio / np.max(np.abs(audio))
        # Ensure fixed length
        if len(audio) > SAMPLES_PER_TRACK:
            audio = audio[:SAMPLES_PER_TRACK]
        else:
            pad_length = SAMPLES_PER_TRACK - len(audio)
            audio = np.pad(audio, (0, pad_length))

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Take mean (important)
        mfcc = np.mean(mfcc.T, axis=0)
        zcr = librosa.feature.zero_crossing_rate(audio)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

        features = np.hstack([
            np.mean(mfcc.T, axis=0),
            np.mean(zcr),
            np.mean(spectral_centroid)
        ])
        return mfcc

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None



# LOAD DATA

X = []
y = []
   
real_path = os.path.join(r'C:\Users\PADMA KUMAR S\Desktop\aib\processedone\real')
fake_path = os.path.join(r'C:\Users\PADMA KUMAR S\Desktop\aib\processedone\fake')

print("Loading REAL files...")
for file in os.listdir(real_path):
    if file.endswith(".wav"):
        file_path = os.path.join(real_path, file)
        features = extract_features(file_path)
        if features is not None:    
            X.append(features)
            y.append(0)

print("Loading FAKE files...")
for file in os.listdir(fake_path):
    if file.endswith(".wav"):
        file_path = os.path.join(fake_path, file)
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(1)

X = np.array(X)
y = np.array(y)
print(f"Dataset loaded: {len(X)} samples")


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

df = pd.DataFrame(X)  # X = your feature matrix

corr = df.corr()

sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


df = pd.DataFrame(X)

print("\nDataset Head (Normalized Features):")
print(df.head())
results = {}


# MODEL 1: Logistic Regression

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Accuracy:", lr_acc)
results["Logistic"] = lr_acc

# MODEL 2: SVM

print("\nTraining SVM...")
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_acc)
results["SVM"] = svm_acc

#random forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)
results["Random Forest"] = rf_acc

#Gradient boosting
print("\nTraining Gradient Boosting...")
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

print("Gradient Boosting Accuracy:", gb_acc)
results["Gradient Boosting"] = gb_acc

#Bagging
print("\nTraining Bagging...")
bag_model = BaggingClassifier(n_estimators=50)
bag_model.fit(X_train, y_train)

bag_pred = bag_model.predict(X_test)
bag_acc = accuracy_score(y_test, bag_pred)

print("Bagging Accuracy:", bag_acc)
results["Bagging"] = bag_acc



models = list(results.keys())
accuracies = list(results.values())

plt.figure(figsize=(10,5))
plt.bar(models, accuracies, color=['blue','green','orange','red','purple'])
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.title("Model Comparison")
plt.ylim(0,1)
plt.show()

# SAVE MODELS 

import joblib

joblib.dump(lr_model, "logistic_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(gb_model, "gb_model.pkl")
joblib.dump(bag_model, "bag_model.pkl")

print("\nModels saved successfully ✅")
best_model_name = max(results, key=results.get)
print("\nBest Model:", best_model_name)