import joblib
from feature_extraction import extract_features  # make sure this is your feature extractor filename

# Load the trained model
model = joblib.load("model.pkl")  # or your actual model filename

def predict_emotion(file_path):
    features = extract_features(file_path)
    return model.predict([features])[0]

