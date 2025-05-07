# feature_extraction.py
import numpy as np
import librosa

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

        if len(result) < max_features:
            result = np.pad(result, (0, max_features - len(result)), mode='constant')
        elif len(result) > max_features:
            result = result[:max_features]

        return result
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return np.zeros(max_features)
