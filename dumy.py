#Feature Extraction of Audio Files Function 
#Extract features (mfcc, chroma, mel) from a sound file
import numpy as np
import librosa
import soundfile as sf
import pickle

def extract_features(file_path, max_features=180, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file_path, sr=None)
        result = np.array([])

        stft = np.abs(librosa.stft(X)) if chroma else None

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))

        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))

        # Pad or trim to fixed length
        if len(result) < max_features:
            result = np.pad(result, (0, max_features - len(result)), mode='constant')
        elif len(result) > max_features:
            result = result[:max_features]

        return result
    
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return np.zeros(max_features)


Pkl_Filename='model.pkl'
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    model = pickle.load(file)

file_path = "dumy.wav"
features = extract_features(file_path)  # Always 180 now

print("Feature shape:", features.shape)  # Should be (180,)

# Reshape for model
features = features.reshape(1, -1)  # Shape: (1, 180)

# Predict
prediction = model.predict(features)
print("Predicted Emotion:", prediction[0])

