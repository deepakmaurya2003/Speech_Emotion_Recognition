from flask import Flask, request, jsonify
from flask_cors import CORS
from model_utils import predict_emotion
from email_alert import send_alert_email
import os
import json
from datetime import datetime
import wave

app = Flask(__name__)
CORS(app)

LOG_FILE = "emotion_log.json"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    # We expect raw wav data in request.files['audio'] (from blob)
    if 'audio' not in request.files:
        return jsonify({"error": "No audio data provided"}), 400

    audio_file = request.files['audio']
    filename = os.path.join(UPLOAD_DIR, f"recorded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
    audio_file.save(filename)

    try:
        emotion = predict_emotion(filename)

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": emotion
        }

        logs = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)

        logs.append(log_entry)

        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)

        if emotion.lower() in ["disgust", "fearful", "sad"]:
            send_alert_email(emotion)

        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filename):
            os.remove(filename)

@app.route("/get_emotions", methods=["GET"])
def get_emotions():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
        return jsonify(logs)
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True, port=5000)


