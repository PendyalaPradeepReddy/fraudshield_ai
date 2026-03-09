"""
local_streamer.py — Pure Python Real-Time Streaming Simulation
Simulates the Kafka Producer/Consumer architecture using Python Threads and Queues.
Perfect for demonstrating the real-time pipeline when a standalone Kafka server is not available.
"""

import os
import time
import json
import random
import requests
import threading
import pandas as pd
from queue import Queue
from dotenv import load_dotenv

# We reuse the same alert logic from earlier
from src.alerts import send_email_alert, build_fraud_email

load_dotenv()

DATA_PATH = "creditcard.csv"
FASTAPI_ENDPOINT = "http://127.0.0.1:8000/predict"

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

def producer_thread(queue: Queue, stop_event: threading.Event):
    """Acts like the Kafka Producer, pushing row-by-row into the queue"""
    print("[PRODUCER] Starting data ingestion thread...")
    try:
        df = pd.read_csv(DATA_PATH, nrows=1000)
        df = df.sample(frac=1).reset_index(drop=True)
    except FileNotFoundError:
        print(f"[FATAL] Could not find {DATA_PATH}")
        stop_event.set()
        return

    for idx, row in df.iterrows():
        if stop_event.is_set():
            break

        transaction = row.drop('Class').to_dict() if 'Class' in row else row.to_dict()
        
        # In a real system, you push to Kafka. Here we push to a python Queue.
        queue.put((idx, transaction))
        
        # Simulate arrival rate (1-5 transactions per second)
        time.sleep(random.uniform(0.2, 1.0))
        
    print("[PRODUCER] Finished streaming all transactions.")

def consumer_thread(queue: Queue, stop_event: threading.Event):
    """Acts like the Kafka Consumer, pulling from the queue and hitting the API"""
    print("[CONSUMER] Starting listener thread...")
    while not stop_event.is_set():
        try:
            # Block until an item is available, timeout allows checking the stop_event
            idx, transaction = queue.get(timeout=1.0)
        except:
            continue
            
        # 1. Structure the features
        features = {}
        for k, v in transaction.items():
            if k == 'Amount':
                features['scaled_Amount'] = v / 100.0  
            elif k == 'Time':
                features['scaled_Time'] = v / 100000.0
            else:
                features[k] = v

        payload = {"features": features}

        # 2. Hit the prediction API (which is running completely isolated)
        try:
            response = requests.post(FASTAPI_ENDPOINT, json=payload, timeout=2.0)
            if response.status_code == 200:
                result = response.json()
                is_fraud = result.get('consensus_fraud', False)
                risk_score = result.get('risk_score', 0.0)
                risk_level = get_risk_level(risk_score)
                
                # 3. Handle Fraud Detections
                if is_fraud or risk_score > 0.6:
                    color = "\033[91m" # Red
                    print(f" {color}>>> \u26A0\uFE0F  [Tx {idx}] FRAUD DETECTED! Risk: {risk_score:.2f} [{risk_level}]\033[0m")
                    
                    if SENDER_EMAIL and SENDER_PWD:
                        body = build_fraud_email(transaction, risk_score, risk_level)
                        res = send_email_alert(SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PWD, RECIPIENT_EMAIL, "Fraud Alert", body)
                        if res.get('success'):
                            print(f"    \u2709\uFE0F [ALERT EMAIL] Sent successfully to {RECIPIENT_EMAIL}")
                else:
                    color = "\033[92m" # Green
                    print(f" {color}>>> \u2705 [Tx {idx}] CLEAN. Risk: {risk_score:.2f} [{risk_level}]\033[0m")
        except Exception as e:
            print(f" [API ERROR] FastApi might be down: {e}")

        queue.task_done()

if __name__ == "__main__":
    print("="*60)
    print("  SIMULATED REAL-TIME STREAMING ARCHITECTURE  ")
    print("="*60)
    
    # The message broker (like our Kafka Topic)
    message_queue = Queue()
    stop_event = threading.Event()
    
    t_producer = threading.Thread(target=producer_thread, args=(message_queue, stop_event))
    t_consumer = threading.Thread(target=consumer_thread, args=(message_queue, stop_event))
    
    t_consumer.start()
    t_producer.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping streams...")
        stop_event.set()
        t_producer.join()
        t_consumer.join()
        print("[INFO] Offline.")
