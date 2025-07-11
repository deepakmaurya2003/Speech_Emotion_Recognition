import smtplib
from email.mime.text import MIMEText

def send_alert_email(emotion):
    sender = "selfdeepak2003@gmail.com"
    receiver = "cse21148@glbitm.ac.in"
    password = "wiqb zwmx unjo zpxg"  # Use App Password for Gmail

    subject = f"⚠️ Emotion Alert: {emotion.upper()}"
    body = f"Real-time system detected emotion: {emotion.upper()}."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"[EMAIL SENT] Alert for {emotion}")
    except Exception as e:
        print(f"[EMAIL ERROR]: {e}")
