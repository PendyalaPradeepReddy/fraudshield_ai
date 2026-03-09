"""
models.py — Train, evaluate, and persist four fraud-detection models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, IsolationForest
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from xgboost import XGBClassifier

CACHE_PATH  = os.path.join(os.path.dirname(__file__), "..", "cache")
MODELS_PATH = os.path.join(CACHE_PATH, "models.pkl")
METRICS_PATH = os.path.join(CACHE_PATH, "metrics.pkl")

os.makedirs(CACHE_PATH, exist_ok=True)


# ── Model definitions ──────────────────────────────────────────────────────────

def _build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=577,  # len(non-fraud)/len(fraud)
            use_label_encoder=False, eval_metric="aucpr",
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=0.001727,  # fraud rate ~0.173%
            random_state=42, n_jobs=-1,
        ),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test, feature_names, force: bool = False):
    """
    Train all models and compute metrics.
    Returns (models_dict, metrics_df, curve_data_dict).
    Results are cached; pass force=True to retrain.
    """
    if not force and os.path.exists(MODELS_PATH) and os.path.exists(METRICS_PATH):
        models  = joblib.load(MODELS_PATH)
        package = joblib.load(METRICS_PATH)
        return models, package["metrics_df"], package["curves"]

    models  = _build_models()
    metrics = []
    curves  = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Isolation Forest: -1 → 1 (fraud), 1 → 0 (normal)
        if name == "Isolation Forest":
            raw_pred  = model.predict(X_test)
            y_pred    = np.where(raw_pred == -1, 1, 0)
            scores    = -model.score_samples(X_test)   # higher = more anomalous
            y_scores  = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            y_pred   = model.predict(X_test)
            y_scores = model.predict_proba(X_test)[:, 1]

        cm   = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        prec, rec, _ = precision_recall_curve(y_test, y_scores)

        metrics.append({
            "Model":       name,
            "Accuracy":    round(accuracy_score(y_test, y_pred)  * 100, 2),
            "Precision":   round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "Recall":      round(recall_score(y_test, y_pred,    zero_division=0) * 100, 2),
            "F1 Score":    round(f1_score(y_test, y_pred,        zero_division=0) * 100, 2),
            "ROC-AUC":     round(roc_auc_score(y_test, y_scores) * 100, 2),
            "PR-AUC":      round(average_precision_score(y_test, y_scores) * 100, 2),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        })
        curves[name] = {
            "fpr": fpr, "tpr": tpr,
            "prec": prec, "rec": rec,
            "cm": cm,
            "y_pred": y_pred,
            "y_scores": y_scores,
        }

    metrics_df = pd.DataFrame(metrics).set_index("Model")

    joblib.dump(models, MODELS_PATH)
    joblib.dump({"metrics_df": metrics_df, "curves": curves}, METRICS_PATH)

    return models, metrics_df, curves


def predict_transaction(models, feature_names, input_array):
    """
    Run a single transaction through all supervised models.
    Returns a dict → {"Model": prob_fraud}.
    """
    results = {}
    for name, model in models.items():
        if name == "Isolation Forest":
            score = -model.score_samples(input_array)[0]
            # Normalize to [0,1] using training contamination assumption
            results[name] = float(np.clip(score / 5.0, 0, 1))
        else:
            results[name] = float(model.predict_proba(input_array)[0, 1])
    return results
