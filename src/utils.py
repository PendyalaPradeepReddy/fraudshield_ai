"""
utils.py — Shared helpers, color palette, and risk scoring.
"""

import numpy as np

# ── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    "bg_dark":     "#0a0a1a",
    "bg_card":     "#12122a",
    "bg_sidebar":  "#0d0d20",
    "accent_red":  "#e63946",
    "accent_gold": "#ffd700",
    "accent_blue": "#4cc9f0",
    "accent_green":"#06d6a0",
    "text_primary":"#f0f0f0",
    "text_muted":  "#8888aa",
    "border":      "#2a2a4a",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0a0a1a",
        "plot_bgcolor":  "#12122a",
        "font":     {"color": "#f0f0f0", "family": "Inter, sans-serif"},
        "colorway": ["#e63946", "#4cc9f0", "#ffd700", "#06d6a0", "#a78bfa"],
        "xaxis":  {"gridcolor": "#2a2a4a", "linecolor": "#2a2a4a"},
        "yaxis":  {"gridcolor": "#2a2a4a", "linecolor": "#2a2a4a"},
        "legend": {"bgcolor": "#12122a", "bordercolor": "#2a2a4a"},
    }
}


def risk_score(fraud_probability: float) -> dict:
    """Convert fraud probability to a risk level label and color."""
    if fraud_probability >= 0.75:
        return {"level": "CRITICAL", "color": "#e63946", "icon": "🔴"}
    elif fraud_probability >= 0.50:
        return {"level": "HIGH",     "color": "#ff6b35", "icon": "🟠"}
    elif fraud_probability >= 0.25:
        return {"level": "MEDIUM",   "color": "#ffd700", "icon": "🟡"}
    else:
        return {"level": "LOW",      "color": "#06d6a0", "icon": "🟢"}


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def compute_confusion_values(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


FEATURE_DESCRIPTIONS = {
    "V1":  "Anonymized PCA Feature 1",
    "V2":  "Anonymized PCA Feature 2",
    "V3":  "Anonymized PCA Feature 3",
    "V4":  "Anonymized PCA Feature 4",
    "V5":  "Anonymized PCA Feature 5",
    "V6":  "Anonymized PCA Feature 6",
    "V7":  "Anonymized PCA Feature 7",
    "V8":  "Anonymized PCA Feature 8",
    "V9":  "Anonymized PCA Feature 9",
    "V10": "Anonymized PCA Feature 10",
    "V11": "Anonymized PCA Feature 11",
    "V12": "Anonymized PCA Feature 12",
    "V14": "Anonymized PCA Feature 14",
    "V17": "Anonymized PCA Feature 17",
    "Amount": "Transaction Amount (USD)",
    "Time":   "Seconds Since First Transaction",
}

# The top fraud-correlated features (used in Live Simulator)
KEY_FEATURES = ["V14", "V17", "V12", "V10", "V4", "V11", "V3", "V7", "Amount"]
