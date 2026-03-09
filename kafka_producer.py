"""
kafka_producer.py — Real-Time Streaming Producer (Simulated Data Ingestion)
Reads from the creditcard.csv dataset and streams transactions into a Kafka topic.
"""

import time
import json
import random
import pandas as pd
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "financial_transactions"
DATA_PATH = "creditcard.csv"

def delivery_report(err, msg):
    if err is not None:
        print(f"[ERROR] Message delivery failed: {err}")
    else:
        print(f"[PRODUCER] Streamed transaction to {msg.topic()} [{msg.partition()}]")

def run_producer():
    print(f"Initializing Kafka Producer connecting to {KAFKA_BROKER}...")
    try:
        # kafka-python producer
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"[FATAL] Could not connect to Kafka Broker at {KAFKA_BROKER}: {e}")
        print("💡 Make sure your Kafka server (or Docker container) is running!")
        return

    print("Loading transaction dataset...")
    # Load just a chunk for simulation purposes to keep it lightweight
    try:
        df = pd.read_csv(DATA_PATH, nrows=5000)
    except FileNotFoundError:
        print(f"[ERROR] Data file {DATA_PATH} not found.")
        return

    # To simulate new transactions arriving, we'll shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"Starting to stream transactions to topic: {TOPIC_NAME}")
    print("Press Ctrl+C to stop.\n" + "-"*50)

    try:
        for idx, row in df.iterrows():
            # Convert row to a dict, drop the 'Class' target since this is inference
            transaction = row.drop('Class').to_dict() if 'Class' in row else row.to_dict()
            
            # Send to Kafka
            producer.send(TOPIC_NAME, value=transaction)
            print(f"[PRODUCER] Streamed transaction {idx} - Amount: ${transaction.get('Amount', 0):.2f}")
            
            # Simulate real-time delay (e.g., streaming 1-5 transactions per second)
            time.sleep(random.uniform(0.2, 1.0))
            producer.flush()  # Ensure it gets sent
            
    except KeyboardInterrupt:
        print("\n[INFO] Streaming stopped by user.")
    finally:
        producer.close()
        print("[INFO] Producer closed.")

if __name__ == "__main__":
    run_producer()
