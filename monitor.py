import time
from recorder import record_audio
from model_utils import predict_emotion

def monitor_loop():
    print("[Monitoring Started] Press Ctrl+C to stop.")
    try:
        while True:
            print("\n[Recording]...")
            filename = "realtime.wav"
            record_audio(filename, duration=3)

            print("[Predicting Emotion]...")
            emotion = predict_emotion(filename)

            print(f"[Result] Detected Emotion: {emotion}")
            time.sleep(1)  # Pause between recordings
    except KeyboardInterrupt:
        print("\n[Stopped Monitoring]")

if __name__ == "__main__":
    monitor_loop()
