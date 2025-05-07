import sys
import pickle
import numpy as np
from feature_extraction import extract_features  # Your function from the earlier code

Pkl_Filename = 'model.pkl'

# Load model
with open(Pkl_Filename, 'rb') as file:
    model = pickle.load(file)

# Get file path from command-line arguments
file_path = sys.argv[1]

# Extract features
features = extract_features(file_path)
features = features.reshape(1, -1)

# Predict
prediction = model.predict(features)

# Print the emotion (stdout will be read by Node.js)
print(prediction[0])
