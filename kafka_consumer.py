"""
kafka_consumer.py — Real-Time Streaming Consumer
Listens to a Kafka topic for incoming transactions, queries the FastAPI endpoint,
and optionally triggers alerts if high-risk fraud is detected.
"""

import os
import json
import requests
from dotenv import load_dotenv
from kafka import KafkaConsumer
from src.alerts import send_email_alert, build_fraud_email

# Load environment variables from .env file securely
load_dotenv()

KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "financial_transactions"
FASTAPI_ENDPOINT = "http://127.0.0.1:8000/predict"

# Alerting configurations (Read securely from .env)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PWD = os.getenv("SENDER_APP_PASSWORD", "").replace(" ", "")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "admin@bank.com")

def get_risk_level(score):
    if score >= 0.85: return "CRITICAL"
    if score >= 0.70: return "HIGH"
    if score >= 0.50: return "MEDIUM"
    return "LOW"

def run_consumer():
    print(f"Initializing Kafka Consumer listening on {KAFKA_BROKER}...")
    try:
        consumer = KafkaConsumer(
            TOPIC_NAME,
            bootstrap_servers=[KAFKA_BROKER],
            group_id='fraud_detection_group',
            auto_offset_reset='latest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
    except Exception as e:
        print(f"[FATAL] Could not connect to Kafka Broker: {e}")
        return

    print(f"Connected successfully! Listening to topic: {TOPIC_NAME}")
    print("Waiting for real-time transactions...\n" + "-"*50)

    try:
        for message in consumer:
            transaction = message.value
            transaction_id = str(message.offset) # Simulated ID for logs
            
            print(f"[CONSUMER] Received Tx ID: {transaction_id} - Size: {len(transaction)} features")
            
            # Map features expected by FastAPI
            # We must map raw "Amount" to "scaled_Amount" roughly or use the preprocessor in a real setup.
            # Here, the FastAPI takes default 0.0 if scaling isn't perfectly mapped.
            features = {}
            for k, v in transaction.items():
                if k == 'Amount':
                    # Rough mapping for our test
                    features['scaled_Amount'] = v / 100.0  
                elif k == 'Time':
                    features['scaled_Time'] = v / 100000.0
                else:
                    features[k] = v

            payload = {"features": features}

            # POST to our FastAPI Deployment
            try:
                response = requests.post(FASTAPI_ENDPOINT, json=payload, timeout=2.0)
                if response.status_code == 200:
                    result = response.json()
                    is_fraud = result.get('consensus_fraud', False)
                    risk_score = result.get('risk_score', 0.0)
                    
                    risk_level = get_risk_level(risk_score)
                    
                    if is_fraud or risk_score > 0.6:
                        color = "\033[91m" # Red
                        print(f" {color}>>> \u26A0\uFE0F  FRAUD DETECTED! Risk Score: {risk_score:.2f} [{risk_level}]\033[0m")
                        
                        # Trigger Alert System if configured securely
                        if SENDER_EMAIL and SENDER_PWD:
                            body = build_fraud_email(transaction, risk_score, risk_level)
                            res = send_email_alert(SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PWD, RECIPIENT_EMAIL, "Fraud Alert", body)
                            
                            if res.get('success'):
                                print(f"    [ALERT EMAIL] Sent successfully to {RECIPIENT_EMAIL}!")
                            else:
                                print(f"    [ALERT EMAIL FAILED] Error: {res.get('error')}")
                        else:
                            print("    [WARNING] Email not sent. Please set SENDER_EMAIL & SENDER_APP_PASSWORD in .env")
                    else:
                        color = "\033[92m" # Green
                        print(f" {color}>>> \u2705 CLEAN. Risk Score: {risk_score:.2f} [{risk_level}]\033[0m")
                else:
                    print(f" [API ERROR] Received code {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f" [API ERROR] Failed to reach prediction server: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] Consumer stopped.")
    finally:
        consumer.close()
        print("[INFO] Consumer connection closed.")

if __name__ == "__main__":
    run_consumer()
