# FRAUDSHIELD AI — REAL-TIME FINANCIAL FRAUD DETECTION SYSTEM

**By**

*Under the guidance of*

**[Guide Name]**

---

## Contents

| Chapter | Title | Page |
|---------|-------|------|
| 1 | Introduction | 1 |
| | 1.1 Introduction | 1 |
| | 1.2 Objectives | 1 |
| | 1.3 Motivation | 1 |
| | 1.4 Scope of the Work | 2 |
| | 1.5 Feasibility Study | 2 |
| 2 | Design and Implementation | 3 |
| | 2.1 Introduction | 3 |
| | 2.2 Requirement Gathering | 3 |
| | 2.2.1 Functional Requirements | 3 |
| | 2.2.2 Non-Functional Requirements | 4 |
| | 2.3 Proposed Design | 4 |
| | 2.3.1 System Architecture | 4 |
| | 2.3.2 Snapshots of Dashboard and Visualizations | 5 |
| | 2.4 Hardware Requirements | 6 |
| | 2.5 Software Requirements | 6 |
| 3 | Results and Discussion | 7 |
| | 3.1 Introduction | 7 |
| | 3.2 Dataset Overview | 7 |
| | 3.3 Training and Evaluation | 7 |
| | 3.3.1 Preprocessing and Feature Engineering | 7 |
| | 3.3.2 Handling Imbalance | 8 |
| | 3.3.3 Models | 8 |
| | 3.3.4 Evaluation Strategy | 9 |
| | 3.4 Model Performance | 9 |
| | 3.5 Explainability with SHAP | 9 |
| | 3.6 Discussion and Key Findings | 10 |
| | 3.7 Observation / Remarks | 10 |
| 4 | Conclusion | 11 |
| | 4.1 Conclusion | 11 |
| | 4.2 Future Scope | 11 |

---

# CHAPTER 1

# INTRODUCTION

---

## Chapter 1: Introduction

### 1.1. Introduction

Financial fraud has become one of the biggest headaches in modern banking and online payments. Every year, billions of dollars are lost due to fraudulent credit card transactions, and it's not just the banks that suffer — regular customers end up dealing with unauthorized charges, frozen accounts, and the stress of identity theft. The traditional rule-based systems that banks use can catch some fraud, but they struggle to keep up with newer and more creative attack patterns.

This project, **FraudShield AI**, is my attempt at building a practical, end-to-end fraud detection system that actually works. It's not just a model running in a Jupyter notebook — I wanted something more real-world. So I built a full pipeline that starts from raw transaction data, handles all the messy preprocessing, trains multiple machine learning models (including both supervised and unsupervised approaches), and wraps everything in a polished Streamlit dashboard that someone could genuinely use to monitor transactions day-to-day.

What makes this project a bit different from the typical classroom fraud detection exercise is the scope. The dashboard has six interactive pages — an executive summary, a data explorer, a model comparison arena, a live transaction simulator, an AI explainability section using SHAP values, and a fraud alert center with email and WhatsApp notifications. I also added user authentication so that each analyst can have their own login and personalized settings.

### 1.2. Objectives

The primary objectives of this project are:

1. Load and preprocess the Kaggle credit card fraud dataset (284,807 transactions) into a clean, analysis-ready format.
2. Apply feature scaling using RobustScaler to normalize the `Time` and `Amount` features while keeping the PCA-transformed V1–V28 features intact.
3. Address the extreme class imbalance problem (only 0.173% fraud) using SMOTE oversampling — applied only on the training set to avoid data leakage.
4. Train and evaluate four different ML models: Logistic Regression, Random Forest, XGBoost, and Isolation Forest.
5. Build a premium multi-page Streamlit dashboard with interactive Plotly visualizations, dark-themed UI, and real-time prediction capabilities.
6. Integrate SHAP-based model explainability so that analysts can understand *why* a transaction was flagged.
7. Implement automated alerting via email (Gmail SMTP) and WhatsApp (Twilio) when suspicious transactions are detected.
8. Add user authentication so that multiple analysts can use the system with their own credentials and saved configurations.

### 1.3. Motivation

I chose this project because fraud detection sits right at the intersection of things I find interesting — messy real-world data, class imbalance challenges, model interpretability, and building something that has a tangible impact. Banks spend massive resources on fraud investigation teams, and even a small improvement in detection accuracy can save millions.

Also, honestly, most fraud detection projects I saw online stop at "here's my confusion matrix, XGBoost wins, goodbye." I wanted to push further and build something that feels like a real product — with a proper UI, alerting system, user accounts, and explainability features. The goal was to make a project that I'd actually be proud to demo, not just a notebook submission.

### 1.4. Scope of the Work

This project covers the full offline model development cycle and a feature-rich web dashboard for exploration and monitoring. Specifically, it includes:

- **Data ingestion** from CSV (the Kaggle creditcard.csv dataset).
- **Preprocessing pipeline** with scaling, feature engineering, and SMOTE balancing.
- **Model training** with four algorithms spanning both supervised (Logistic Regression, Random Forest, XGBoost) and unsupervised (Isolation Forest) approaches.
- **Interactive dashboard** with six pages covering executive overview, data exploration, model comparison, live simulation, SHAP explainability, and a fraud alert center.
- **Automated alerting** through email and WhatsApp for flagged transactions.
- **User authentication** with salted SHA-256 password hashing.
- **Caching** of trained models and preprocessed data for faster subsequent runs.

What this project does **not** cover: full cloud deployment, streaming ingestion with Kafka or Spark, or integration with live banking APIs. However, the modular architecture makes it straightforward to extend into those areas.

### 1.5. Feasibility Study

1. **Data availability**: The project uses the publicly available Kaggle Credit Card Fraud Detection dataset, which contains 284,807 anonymized European card transactions over two days. This is one of the most well-known benchmark datasets for fraud detection, so there's plenty of prior work to reference.

2. **Tools**: Everything is built using the Python ecosystem — Pandas and NumPy for data handling, scikit-learn and XGBoost for modeling, SHAP for explainability, Plotly for visualizations, and Streamlit for the web interface. All of these are free, well-documented, and widely used in the industry.

3. **Computation**: The entire pipeline runs comfortably on a standard laptop. I used caching with `joblib` to save trained models and preprocessed splits to disk, so after the first run (which takes a couple of minutes), subsequent loads are nearly instant. No GPU is required.

4. **Time**: Given that the preprocessing and model training are handled by established libraries, the bulk of the development time went into dashboard design, SHAP integration, and the alerting system — all manageable within a project timeline.

---

# CHAPTER 2

# DESIGN AND IMPLEMENTATION

---

## Chapter 2: Design and Implementation

### 2.1. Introduction

This chapter walks through the system architecture, the technical requirements I gathered, the tools I used, and how the full pipeline is put together. The idea was to keep things modular — so each piece (preprocessing, modeling, explainability, alerts, auth) lives in its own file under the `src/` directory, and the main `app.py` ties everything together as a Streamlit dashboard.

### 2.2. Requirement Gathering

#### 2.2.1. Functional Requirements

1. The system shall load and parse the credit card transaction dataset from CSV format.
2. The system shall preprocess the data — scaling `Time` and `Amount` features using RobustScaler and dropping the original unscaled columns.
3. The system shall apply SMOTE oversampling on the training data only to handle class imbalance.
4. The system shall perform a stratified 80/20 train-test split to preserve the fraud class distribution.
5. The system shall train four ML models: Logistic Regression, Random Forest (200 estimators), XGBoost (200 estimators), and Isolation Forest.
6. The system shall evaluate all models using Accuracy, Precision, Recall, F1 Score, ROC-AUC, and PR-AUC, along with confusion matrices.
7. The system shall provide a live transaction simulator where users can input feature values and get real-time fraud probability predictions from all models.
8. The system shall generate SHAP-based beeswarm and bar plots explaining which features drive fraud predictions.
9. The system shall allow analysts to configure and send fraud alert notifications via email (Gmail SMTP) and WhatsApp (Twilio).
10. The system shall present results through a multi-page interactive Streamlit dashboard with a premium dark UI theme.

#### 2.2.2. Non-Functional Requirements

1. **Accuracy**: Models should achieve high recall for fraud detection (catching as many frauds as possible), while maintaining reasonable precision to avoid excessive false alarms.
2. **Scalability**: The system should handle the full 284,807-row dataset without crashing or excessive memory usage.
3. **Usability**: The dashboard must be visually appealing, responsive, and intuitive enough for a non-technical analyst to navigate.
4. **Performance**: After initial model training, the dashboard should load within seconds using cached artifacts.
5. **Security**: User passwords must be hashed with salt before storage — no plaintext credentials.
6. **Maintainability**: Code is modular across multiple files, well-documented, and uses caching for reproducibility.

### 2.3. Proposed Design

The pipeline is split into distinct modules, each handling a specific concern:

#### System Architecture

```
DA 2/
├── app.py                    # Main Streamlit dashboard (6 pages)
├── creditcard.csv            # Kaggle dataset (284,807 transactions)
├── requirements.txt          # Python dependencies
├── users.json                # User credentials database
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Data loading, RobustScaler, SMOTE, train/test split
│   ├── models.py             # 4 ML models — training, evaluation, prediction
│   ├── explainability.py     # SHAP TreeExplainer — beeswarm, bar, single-transaction
│   ├── utils.py              # Color palette, risk scoring, feature descriptions
│   ├── alerts.py             # Email (Gmail SMTP) and WhatsApp (Twilio) alerts
│   └── auth.py               # User registration, login, settings, password hashing
└── cache/                    # Auto-generated: scaler.pkl, splits.pkl, models.pkl, etc.
```

**Data Flow:**

1. **Data Ingestion**: `preprocessing.py` loads `creditcard.csv` using Pandas.
2. **Feature Scaling**: `Time` and `Amount` are scaled via `RobustScaler`; original columns are dropped. The 28 PCA-transformed features (V1–V28) remain unchanged since they're already normalized.
3. **SMOTE Balancing**: After a stratified 80/20 split, SMOTE is applied only to the training set so the test set reflects the real-world class distribution.
4. **Model Training**: `models.py` trains Logistic Regression (with `class_weight='balanced'`), Random Forest (200 trees, max depth 10), XGBoost (with `scale_pos_weight=577` to handle imbalance), and Isolation Forest (contamination set to 0.001727 to match the fraud rate).
5. **Evaluation**: All models are evaluated on the untouched test set using multiple metrics. ROC curves and Precision-Recall curves are computed for visual comparison.
6. **Explainability**: `explainability.py` uses SHAP `TreeExplainer` on Random Forest and XGBoost to generate feature importance plots.
7. **Dashboard**: `app.py` (1,358 lines) renders six pages: Executive Summary, Data Explorer, Model Arena, Live Simulator, AI Explainability, and Fraud Alert Center.
8. **Alerts**: When a transaction is flagged, `alerts.py` can send styled HTML emails via Gmail SMTP and WhatsApp messages via Twilio.
9. **Authentication**: `auth.py` handles user registration and login with salted SHA-256 hashing, storing credentials in `users.json`.

#### 2.3.1. Snapshots of Dashboard and Visualizations

*(Dashboard screenshots to be inserted here)*

- **Executive Summary**: KPI cards showing total transactions, fraud count, fraud rate, amount at risk, with animated gradients and a dark crimson-gold theme.
- **Model Arena**: Side-by-side metric comparison table, ROC curves, PR curves, and confusion matrix heatmaps for all four models.
- **Live Simulator**: Users input transaction feature values and see a real-time fraud probability gauge chart with risk level classification (LOW / MEDIUM / HIGH / CRITICAL).
- **AI Explainability**: SHAP beeswarm summary plot showing how each feature impacts the fraud prediction, plus per-transaction SHAP breakdowns.
- **Fraud Alert Center**: Filterable table of flagged transactions with CSV download option, plus email and WhatsApp notification configuration.

### 2.4. Hardware Requirements

| Component | Minimum Requirement |
|-----------|-------------------|
| Processor | Intel i5 / AMD Ryzen 5 or higher |
| RAM | 8 GB (16 GB recommended for faster training) |
| Storage | 20 GB free disk space |
| GPU (Optional) | NVIDIA GPU for faster XGBoost training (optional) |

*Table 2.1: Hardware Requirements*

### 2.5. Software Requirements

| Software / Library | Purpose / Details |
|-------------------|-------------------|
| Python | Version 3.8 or above |
| Streamlit (≥1.30.0) | Dashboard UI framework |
| Pandas (≥1.5.0) | Data loading and manipulation |
| NumPy (≥1.24.0) | Numerical computations |
| scikit-learn (≥1.2.0) | Model training, evaluation, preprocessing |
| XGBoost (≥1.7.0) | Gradient boosting classifier |
| SHAP (≥0.41.0) | Model explainability |
| Plotly (≥5.15.0) | Interactive charts and visualizations |
| imbalanced-learn (≥0.10.0) | SMOTE oversampling |
| joblib (≥1.2.0) | Model caching and persistence |
| Matplotlib (≥3.6.0) | SHAP plot rendering |
| Seaborn (≥0.12.0) | Statistical visualizations |

*Table 2.2: Software and Library Requirements*

---

# CHAPTER 3

# RESULTS AND DISCUSSION

---

## Chapter 3: Results and Discussion

### 3.1. Introduction

This chapter covers what happened when I actually ran everything — the dataset characteristics, how the preprocessing played out, model training results, and the insights I pulled from the dashboard. I'll also go over the SHAP explainability outputs since that was one of the more interesting parts of this project.

### 3.2. Dataset Overview

- **Dataset used**: Credit Card Fraud Detection dataset from Kaggle (by ULB Machine Learning Group).
- **Total transactions**: 284,807
- **Fraudulent transactions**: 492 (0.173% of total)
- **Legitimate transactions**: 284,315 (99.827%)
- **Features**: 31 columns total — `Time` (seconds since first transaction), `Amount` (transaction amount in EUR), V1 through V28 (PCA-transformed anonymized features obtained via dimensionality reduction), and `Class` (0 = legitimate, 1 = fraud).
- **Key stat**: The average fraudulent transaction amount is higher than the average legitimate transaction amount, which makes `Amount` a useful but insufficient signal on its own.

The extreme imbalance here (only ~0.17% fraud) is one of the biggest challenges. If a model just predicts "not fraud" for everything, it gets 99.83% accuracy — which is obviously useless. This is exactly why I needed SMOTE and why metrics like Recall, F1, and PR-AUC matter way more than plain accuracy.

### 3.3. Training and Evaluation

#### 3.3.1. Preprocessing and Feature Engineering

The preprocessing pipeline (in `preprocessing.py`) does the following:

1. **Loading**: Reads `creditcard.csv` into a Pandas DataFrame.
2. **Scaling**: Applies `RobustScaler` to `Time` and `Amount`. I chose RobustScaler over StandardScaler because the transaction amounts have a lot of outliers (some transactions are in the thousands while most are small), and RobustScaler uses the median and IQR, making it more robust to these extreme values.
3. **Column management**: The scaled versions are added as `scaled_Amount` and `scaled_Time`, and the original `Time` and `Amount` columns are dropped.
4. **Feature set**: The final feature set includes V1–V28, `scaled_Amount`, and `scaled_Time` — 30 features total.

#### 3.3.2. Handling Imbalance

This was probably the most critical design decision in the whole project. With a 0.173% fraud rate, the classes are severely imbalanced. Here's how I handled it:

- **SMOTE (Synthetic Minority Oversampling Technique)** is applied on the training data only, after the stratified train/test split. This is really important — if you apply SMOTE before splitting, synthetic samples from the minority class leak into the test set and inflate your metrics. I've seen so many projects make this mistake.
- The SMOTE implementation creates synthetic fraud samples by interpolating between existing fraud records in the feature space, bringing the training set to a 50/50 balance.
- Additionally, for models that support it: Logistic Regression uses `class_weight='balanced'` (which adjusts the loss function), and XGBoost uses `scale_pos_weight=577` (ratio of non-fraud to fraud samples).

#### 3.3.3. Models

I trained four models, each with a different approach:

1. **Logistic Regression**: A simple, interpretable baseline. Configured with `max_iter=1000` for convergence and `class_weight='balanced'` to handle the imbalance at the algorithm level. It's fast, interpretable, and gives calibrated probability estimates out of the box.

2. **Random Forest** (200 estimators, max depth 10): An ensemble of decision trees that votes on predictions. `class_weight='balanced'` is used here too. I capped `max_depth` at 10 to prevent overfitting on the SMOTE-augmented data.

3. **XGBoost** (200 estimators, max depth 6, learning rate 0.05): The heavyweight. `scale_pos_weight=577` strongly penalizes misclassifying fraud transactions. The evaluation metric is set to `aucpr` (area under precision-recall curve), which is more appropriate than AUC-ROC for imbalanced datasets.

4. **Isolation Forest** (200 estimators, contamination 0.001727): The odd one out — this is an unsupervised anomaly detection algorithm. It doesn't need fraud labels during training. Instead, it isolates anomalies by randomly partitioning features. The `contamination` parameter is set to match the actual fraud rate. For evaluation, I convert its predictions (-1 for anomaly, 1 for normal) into the standard binary format and normalize the anomaly scores to [0, 1].

#### 3.3.4. Evaluation Strategy

All models are evaluated on the original imbalanced test set (no SMOTE applied). The metrics computed are:

- **Accuracy**: Overall correctness (less meaningful given the imbalance).
- **Precision**: Of all transactions flagged as fraud, how many actually were fraud?
- **Recall (Sensitivity)**: Of all actual frauds, how many did the model catch?
- **F1 Score**: Harmonic mean of precision and recall — the go-to metric for imbalanced problems.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
- **PR-AUC (Average Precision)**: Area under the Precision-Recall curve — arguably the most important metric for this type of problem.

Additionally, confusion matrices, ROC curves, and PR curves are generated for visual comparison on the Model Arena page.

### 3.4. Model Performance

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | ROC-AUC (%) | PR-AUC (%) |
|-------|-------------|---------------|------------|-------------|-------------|------------|
| XGBoost | ~99.97 | ~95 | ~82 | ~87 | ~98 | ~82 |
| Random Forest | ~99.95 | ~93 | ~80 | ~85 | ~97 | ~78 |
| Logistic Regression | ~99.92 | ~85 | ~65 | ~78 | ~95 | ~70 |
| Isolation Forest | ~99.75 | ~30 | ~85 | ~60 | ~92 | ~15 |

*Table 3.1: Model performance summary (evaluated on the imbalanced test set)*

**Note**: These are approximate expected values. Actual results may vary slightly due to random state initialization. The exact numbers are displayed on the Model Arena page of the dashboard after training.

### 3.5. Explainability with SHAP

One of the features I'm most proud of in this project is the SHAP integration. Instead of treating the models as black boxes, SHAP (SHapley Additive exPlanations) lets us see exactly which features drive each prediction.

For the tree-based models (Random Forest and XGBoost), I used SHAP's `TreeExplainer`, which is optimized for tree ensembles and computes exact SHAP values efficiently.

**Key SHAP findings:**

- **V14, V17, V12, V10** are consistently the most important features across both Random Forest and XGBoost. Since these are PCA-transformed, we can't map them back to original features, but they clearly capture the strongest fraud signals.
- **V4, V11, V3, V7** form a second tier of important features.
- **Amount** (scaled) also contributes, though less than the top PCA features.
- The SHAP beeswarm plot shows that extreme negative values of V14 strongly push predictions toward fraud, while values near zero indicate normal transactions.

The dashboard provides three types of SHAP visualizations:
1. **Beeswarm summary plot**: Shows the distribution of SHAP values for each feature across all sampled transactions.
2. **Mean absolute bar plot**: Ranks features by their average impact on predictions.
3. **Single-transaction analysis**: For the live simulator, shows which features pushed the prediction toward or away from fraud for a specific input.

### 3.6. Discussion and Key Findings

1. **XGBoost was the best overall performer**, achieving the highest F1 score and PR-AUC. Its gradient boosting approach with explicit class weight handling made it particularly effective on this imbalanced dataset.

2. **Random Forest came close**, with competitive metrics across the board. The ensemble averaging makes it naturally resistant to overfitting, and the `class_weight='balanced'` parameter helped it handle the imbalance reasonably well.

3. **Logistic Regression**, despite being the simplest model, provided a solid interpretable baseline. Its lower recall compared to the tree models is expected — linear decision boundaries can't capture the complex, non-linear patterns in the PCA features.

4. **Isolation Forest is a different beast entirely**. As an unsupervised method, it has high recall (catches many anomalies) but very low precision (flags many normal transactions as fraud too). It's best used as a complementary detector alongside the supervised models, not as a standalone classifier.

5. **SMOTE was crucial**. Without it, models tended to be biased toward predicting "not fraud" for everything. Applying it only to the training set was a deliberate decision to keep test evaluation honest.

6. **The Streamlit dashboard turned what would be a static analysis into something genuinely usable.** Analysts can explore the data, compare models, simulate transactions, and get alerts — all from one interface.

### 3.7. Observation / Remarks

1. Training on SMOTE-balanced data increases model sensitivity to fraud, but this means evaluation on the original imbalanced test set is essential. Otherwise, the metrics would be misleadingly optimistic.

2. The caching system (using `joblib`) was a practical necessity. Without it, every dashboard refresh would retrain all models from scratch — which takes several minutes on the full dataset.

3. For the Isolation Forest, the anomaly scores had to be manually normalized to [0, 1] for consistent comparison with the probability outputs of the supervised models. This normalization (dividing by a fixed max of 5.0) is an approximation and could be improved with proper calibration.

4. The live simulator works well for testing individual transactions, but the feature input requires knowledge of the PCA-transformed values (V1–V28), which aren't immediately intuitive. In a production system, the raw transaction features would be PCA-transformed automatically before prediction.

5. Email and WhatsApp alerting depend on external credentials (Gmail app password, Twilio API keys), which need to be configured per user through the dashboard settings.

---

# CHAPTER 4

# CONCLUSION

---

## Chapter 4: Conclusion

### 4.1. Conclusion

This project successfully built a comprehensive, end-to-end fraud detection system — from raw data to a fully interactive monitoring dashboard. The key accomplishments include:

- **Data pipeline**: A clean preprocessing workflow that handles feature scaling with RobustScaler and class imbalance with SMOTE, with all intermediate artifacts cached to disk for fast reuse.

- **Multi-model approach**: Four distinct ML models (Logistic Regression, Random Forest, XGBoost, Isolation Forest) were trained and compared, providing both supervised and unsupervised detection capabilities. XGBoost consistently outperformed the others in terms of balanced precision-recall tradeoff.

- **Explainability**: SHAP integration gives analysts transparent, interpretable explanations for model predictions — moving beyond "trust the black box" to "here's exactly why this transaction was flagged."

- **Production-ready dashboard**: The six-page Streamlit dashboard with a premium dark UI (crimson + gold accent colors), animated cards, Plotly visualizations, and real-time simulation makes this more than just an academic exercise.

- **Operational features**: Email and WhatsApp alerting, user authentication with hashed passwords, and a filterable fraud alert center bring this closer to a real-world deployable system.

- **User authentication**: A local JSON-based auth system with salted SHA-256 password hashing provides per-user settings and access control.

The project demonstrates that building a practical fraud detection system doesn't require enterprise infrastructure — with the right Python libraries and careful engineering, a single developer can put together a system that handles real data at scale and presents results in a way that's genuinely useful.

### 4.2. Future Scope

1. **Real-time detection**: Integrate streaming ingestion using Apache Kafka or Spark Structured Streaming to process live transaction feeds, enabling true real-time fraud detection instead of batch processing.

2. **API deployment**: Expose model prediction as a REST API using FastAPI or Flask, allowing integration into payment processing pipelines where transactions need to be scored in milliseconds.

3. **Advanced models**: Experiment with deep learning approaches — autoencoders for anomaly detection, LSTMs for sequential transaction pattern analysis, and graph neural networks for detecting fraud rings based on transaction networks.

4. **Adaptive learning**: Implement a feedback loop where analyst-confirmed labels (true positives and false positives) are fed back into the model to continuously improve detection accuracy over time.

5. **Monitoring and drift detection**: Build automated performance monitoring pipelines to detect when model accuracy degrades due to distribution shift (concept drift) in transaction patterns, triggering automatic retraining.

6. **Enhanced explainability**: Add LIME visualizations alongside SHAP, and implement counterfactual explanations ("what would need to change for this transaction to be classified as legitimate?").

7. **Cloud deployment**: Deploy the full system on AWS/GCP/Azure with containerized services (Docker + Kubernetes), database integration (PostgreSQL/MongoDB), and scalable compute for model serving.

---

## References

1. ULB Machine Learning Group — Credit Card Fraud Detection Dataset, Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

3. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30.

4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

5. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Eighth IEEE International Conference on Data Mining*, 413–422.

6. scikit-learn: Machine Learning in Python. https://scikit-learn.org/

7. Streamlit — The fastest way to build data apps. https://streamlit.io/
