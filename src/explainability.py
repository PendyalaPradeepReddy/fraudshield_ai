"""
explainability.py — SHAP-based model explanations.
"""

import os
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cache")
SHAP_PATH  = os.path.join(CACHE_PATH, "shap_values.pkl")

os.makedirs(CACHE_PATH, exist_ok=True)


def compute_shap_values(model, X_test, model_name: str, n_samples: int = 500, force: bool = False):
    """
    Compute SHAP values for a tree-based model (RF or XGBoost).
    Uses a background sample for speed.
    Results are cached.
    """
    cache_key = f"{SHAP_PATH}_{model_name.replace(' ','_')}.pkl"
    if not force and os.path.exists(cache_key):
        return joblib.load(cache_key)

    # Sample for speed
    idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For RF (binary → list of 2 arrays), take the fraud class (index 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        # Newer SHAP returns 3D ndarray (samples, features, classes) — take fraud class
        shap_values = shap_values[:, :, 1]

    result = {
        "shap_values": shap_values,
        "X_sample":    X_sample,
        "explainer":   explainer,
    }
    joblib.dump(result, cache_key)
    return result


def shap_summary_fig(shap_data: dict, feature_names: list, max_display: int = 15):
    """Return a matplotlib figure of the SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#12122a")

    X_df = shap_data["X_sample"]
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df, columns=feature_names)

    shap.summary_plot(
        shap_data["shap_values"],
        X_df,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
        color_bar=True,
    )
    plt.title("SHAP Feature Impact on Fraud Prediction", color="#f0f0f0", fontsize=14, pad=12)
    plt.xlabel("SHAP Value (Impact on Model Output)", color="#8888aa")
    plt.xticks(color="#8888aa")
    plt.yticks(color="#f0f0f0")
    plt.tight_layout()
    return fig


def shap_bar_fig(shap_data: dict, feature_names: list, max_display: int = 15):
    """Return a matplotlib figure of the SHAP mean absolute bar plot."""
    shap_vals = np.abs(shap_data["shap_values"]).mean(axis=0)
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[:, 1]  # defensive: take fraud class if still 2D
    feat_importance = pd.Series(shap_vals, index=feature_names).nlargest(max_display)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#12122a")

    bars = ax.barh(feat_importance.index[::-1], feat_importance.values[::-1],
                   color="#e63946", alpha=0.85, edgecolor="#2a2a4a", linewidth=0.5)

    # Gradient effect
    for i, bar in enumerate(bars):
        bar.set_alpha(0.6 + 0.4 * (i / len(bars)))

    ax.set_xlabel("Mean |SHAP Value|", color="#8888aa", fontsize=11)
    ax.set_title("Top Features Driving Fraud Detection", color="#f0f0f0", fontsize=14, pad=12)
    ax.tick_params(colors="#f0f0f0")
    ax.spines[:].set_color("#2a2a4a")
    ax.grid(axis="x", color="#2a2a4a", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def shap_single_transaction(explainer, input_array, feature_names: list):
    """
    Compute SHAP values for a single transaction and return top contributors.
    Returns a pd.DataFrame with Feature, SHAP Value, Direction.
    """
    sv = explainer.shap_values(input_array)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv.flatten()

    df = pd.DataFrame({
        "Feature":    feature_names,
        "SHAP Value": sv,
        "Direction":  np.where(sv > 0, "↑ Increases Risk", "↓ Decreases Risk"),
    })
    df["Abs"] = df["SHAP Value"].abs()
    return df.nlargest(10, "Abs").drop("Abs", axis=1)
