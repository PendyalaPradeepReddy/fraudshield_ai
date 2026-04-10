# 🛡️ FraudShield AI — Financial Fraud Detection System

A **production-grade** fraud detection dashboard built with Machine Learning and Data Analytics. Designed to identify suspicious transactions and provide real-time insights through an interactive multi-page dashboard.

**🔴 Live Demo:** [Milo AI Chatbot][(https://fraudshieldai.streamlit.app/)

---

## 🚀 Features

| Feature | Details |
|---|---|
| **4 ML Models** | Logistic Regression, Random Forest, XGBoost, Isolation Forest |
| **6-Page Dashboard** | Executive Summary · Data Explorer · Model Arena · Live Simulator · AI Explainability · Fraud Alert Center |
| **SHAP Explainability** | SHAP beeswarm, bar, and single-transaction analysis |
| **Live Simulator** | Real-time fraud probability prediction with gauge chart |
| **Fraud Alert Center** | Filterable flagged transaction table with CSV download |
| **Premium Dark UI** | Crimson + gold on deep dark, animated cards, Plotly charts |

---

## 📁 Project Structure

```
DA 2/
├── app.py                  # Main Streamlit dashboard
├── creditcard.csv          # Kaggle dataset (284,807 transactions)
├── requirements.txt        # Dependencies
├── src/
│   ├── preprocessing.py    # Data cleaning, SMOTE, scaling
│   ├── models.py           # 4 ML models + evaluation
│   ├── explainability.py   # SHAP analysis
│   └── utils.py            # Helpers, colors, risk scoring
└── cache/                  # Auto-generated model cache (fast reruns)
```

---

## ⚙️ Installation & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## 📊 Dataset

- **Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud Rate**: 0.173% (492 fraudulent)
- **Features**: V1–V28 (PCA-transformed), Amount, Time

---

## 🤖 Models & Performance (Expected)

| Model | ROC-AUC | F1 Score |
|---|---|---|
| XGBoost | ~98% | ~87% |
| Random Forest | ~97% | ~85% |
| Logistic Regression | ~95% | ~78% |
| Isolation Forest | ~92% | ~60% |

---

*Built as a premium Data Analyst portfolio project.*
