"""
preprocessing.py — Data loading, class-imbalance handling, feature scaling,
and train/test splitting for the credit-card fraud dataset.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "creditcard.csv")
CACHE_PATH   = os.path.join(os.path.dirname(__file__), "..", "cache")
SCALER_PATH  = os.path.join(CACHE_PATH, "scaler.pkl")
SPLITS_PATH  = os.path.join(CACHE_PATH, "splits.pkl")

os.makedirs(CACHE_PATH, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    """Load the raw credit-card dataset (or synthetic demo if not found)."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Fallback for cloud deployments where 150MB CSV is missing
        np.random.seed(42)
        n_samples = 10000
        data = {'Time': np.random.uniform(0, 172792, n_samples)}
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1.5, n_samples)
        data['Class'] = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        data['Amount'] = np.where(data['Class'] == 1, np.random.exponential(300, n_samples), np.random.exponential(60, n_samples))
        
        # Shift a few features so models can actually learn something from the synthetic data
        for i in range(1, 6):
            data[f'V{i}'] = np.where(data['Class'] == 1, data[f'V{i}'] - 3.0, data[f'V{i}'])
            
        df = pd.DataFrame(data)
    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return high-level stats about the dataset."""
    total      = len(df)
    fraud      = int(df["Class"].sum())
    normal     = total - fraud
    fraud_rate = fraud / total * 100
    amount_at_risk = df[df["Class"] == 1]["Amount"].sum()
    return {
        "total":        total,
        "fraud":        fraud,
        "normal":       normal,
        "fraud_rate":   round(fraud_rate, 4),
        "amount_at_risk": round(amount_at_risk, 2),
        "avg_fraud_amount": round(df[df["Class"] == 1]["Amount"].mean(), 2),
        "avg_normal_amount": round(df[df["Class"] == 0]["Amount"].mean(), 2),
    }


def preprocess(df: pd.DataFrame = None, force: bool = False):
    """
    Full preprocessing pipeline:
    - Scale 'Time' and 'Amount' with RobustScaler
    - SMOTE to balance classes
    - Stratified train/test split (80/20)

    Returns (X_train, X_test, y_train, y_test, feature_names, scaler)
    Results are cached to disk for speed.
    """
    if not force and os.path.exists(SPLITS_PATH) and os.path.exists(SCALER_PATH):
        data   = joblib.load(SPLITS_PATH)
        scaler = joblib.load(SCALER_PATH)
        return (*data, scaler)

    if df is None:
        df = load_raw_data()

    # Scale Time and Amount
    scaler = RobustScaler()
    df = df.copy()
    df["scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
    df["scaled_Time"]   = scaler.fit_transform(df[["Time"]])
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]
    feature_names = X.columns.tolist()

    # Stratified split BEFORE SMOTE (apply SMOTE only on train set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE on training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    data = (X_train_res, X_test, y_train_res, y_test, feature_names)
    joblib.dump(data,   SPLITS_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return (*data, scaler)


def prepare_single_transaction(input_dict: dict, feature_names: list) -> np.ndarray:
    """
    Prepare a single transaction dict for prediction.
    Returns a (1, n_features) numpy array ordered by feature_names.
    """
    row = [input_dict.get(f, 0.0) for f in feature_names]
    return np.array(row).reshape(1, -1)
