"""
api.py — FastAPI Deployment Script for Real-Time Fraud Detection
Provides endpoints for predicting transactions and retraining models on the fly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any

from src.preprocessing import preprocess, prepare_single_transaction
from src.models import train_all, predict_transaction

app = FastAPI(
    title="FraudShield AI API",
    description="Real-Time Financial Fraud Detection API endpoints for inference and model retraining.",
    version="1.0.0"
)

# Global app state for models and components
app.state.models = None
app.state.feature_names = None

@app.on_event("startup")
def load_artifacts():
    print("[INIT] Loading preprocessed data and models into memory...")
    # This loads cached data if it exists, otherwise computes it
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(force=False)
    # This loads cached models or trains them
    models, metrics_df, curves = train_all(X_train, X_test, y_train, y_test, feature_names, force=False)
    
    app.state.models = models
    app.state.feature_names = feature_names
    print("[INIT] Models loaded successfully and ready for inference.")

class TransactionInput(BaseModel):
    """
    Input expected for the transaction prediction.
    'features' should be a dictionary mapping feature names to their float values.
    For missing features, a default of 0.0 is used.
    Example:
        {
            "V1": -1.35, "V2": 1.40, ...,
            "scaled_Amount": 1.25, "scaled_Time": -0.5
        }
    """
    features: Dict[str, float]

class PredictionOutput(BaseModel):
    """
    Output model for predictions. Returns the probability score for each active model,
    the overall risk score, and a boolean classification.
    """
    predictions: Dict[str, float]
    consensus_fraud: bool
    risk_score: float

@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    """
    Endpoint to predict single transactions in real-time.
    """
    if not app.state.models or not app.state.feature_names:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    # Convert the transaction dictionary into a numpy array matching feature shapes
    input_array = prepare_single_transaction(transaction.features, app.state.feature_names)
    
    # Run through all loaded models
    predictions = predict_transaction(app.state.models, app.state.feature_names, input_array)
    
    # Calculate a composite risk score based on average probabilities
    risk_score = sum(predictions.values()) / len(predictions)
    consensus_fraud = risk_score > 0.5  # Simple ensemble threshold
    
    return PredictionOutput(
        predictions=predictions,
        consensus_fraud=consensus_fraud,
        risk_score=round(risk_score, 4)
    )

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    """
    Triggers an asynchronous retraining of the models. 
    Ideal for adaptive learning mechanisms built on a stream of new labeled transactions.
    """
    def _do_retrain():
        print("[RETRAIN] Starting model retraining based on latest data splits...")
        # Simulating loading new data via the preprocessor, followed by forceful retraining
        X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(force=False)
        models, metrics_df, curves = train_all(X_train, X_test, y_train, y_test, feature_names, force=True)
        
        # Update app memory with the newly registered models
        app.state.models = models
        print("[RETRAIN] Models successfully retrained and updated in disk cache and API memory.")

    background_tasks.add_task(_do_retrain)
    return {"message": "Retraining task initiated in the background successfully."}

# Example root endpoint
@app.get("/")
def read_root():
    return {
        "status": "online", 
        "app": "FraudShield AI API", 
        "endpoints": ["/predict", "/retrain"]
    }
