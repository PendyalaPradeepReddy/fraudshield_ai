import requests
import json

def test_prediction():
    print("Testing /predict endpoint...")
    url = "http://127.0.0.1:8000/predict"
    
    # Example mock transaction (based on credit card dataset features)
    payload = {
        "features": {
            "V1": -1.3598,
            "V2": -0.0727,
            "V3": 2.5363,
            "V4": 1.3781,
            "V5": -0.3383,
            "V6": 0.4623,
            "V7": 0.2395,
            "V8": 0.0986,
            "V9": 0.3637,
            "scaled_Time": 0.0,
            "scaled_Amount": 1.5
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Response:", json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"Prediction failed: {e}")

def test_retraining():
    print("\nTesting /retrain endpoint...")
    url = "http://127.0.0.1:8000/retrain"
    
    try:
        response = requests.post(url)
        response.raise_for_status()
        print("Response:", json.dumps(response.json(), indent=4))
    except Exception as e:
        print(f"Retrain failed: {e}")

if __name__ == "__main__":
    test_prediction()
    test_retraining()
