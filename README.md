# FRAUDGUARD: AI-Powered Credit Card Fraud Detection Web Platform

> **Intelligence Against Financial Fraud.**  
> Developed and Owned by **Isbandu Rahuman R.**

---

## 1. Project Overview

**FRAUDGUARD** is an end-to-end financial transaction monitoring and fraud detection platform. It is engineered to detect, classify, and audit fraudulent payment activity within highly imbalanced transaction streams. 

Trained on the European Credit Card Fraud Detection benchmark (284,807 transactions with an authentic 0.17% fraud rate), the system employs SMOTE-resampled Random Forest classification, RobustScaler feature transformations, cost-sensitive threshold optimization, and an interactive Streamlit cybersecurity dashboard backed by SQLite audit persistence.

---

## 2. Key Capabilities

- **Real-Time Single Transaction Scoring**: Evaluates 30 genuine transaction telemetry features, displaying risk levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), calibrated fraud probability, and anomalous latent factors.
- **High-Throughput Bulk CSV Scanner**: Validates, scales, and scores multi-transaction batches, providing risk distribution triage and downloadable annotated reports.
- **Operations Dashboard**: Live KPI tracking, transaction volume velocity, logarithmic amount distribution comparisons, and feature importance analysis.
- **SQLite Audit Trail**: Parameterized persistent storage capturing prediction outcomes, confidence scores, and analyst review notes.
- **Model Intelligence Hub**: Evaluation metrics calculated on a held-out test split (56,962 transactions / 98 confirmed frauds), including PR-AUC curves, ROC curves, and confusion matrices.
- **Zero Data Leakage Pipeline**: Stratified splitting executed before fitting scalers; resampling restricted strictly to training partitions.

---

## 3. Technology Stack

- **Core Programming**: Python 3.11+
- **Data Engineering**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, Imbalanced-Learn (SMOTE)
- **Web Frontend**: Streamlit, Custom Dark CSS, Plotly Interactive Visuals
- **Database**: SQLite3 (Thread-safe, parameterized queries)
- **Model Persistence**: Joblib
- **Testing & Quality Assurance**: Pytest (17 automated unit tests)

---

## 4. Architecture and File Hierarchy

```
FRAUDGUARD/
├── app.py                     # Main Streamlit application orchestrator
├── app/
│   ├── __init__.py
│   ├── dashboard.py           # Executive monitoring dashboard
│   ├── prediction.py          # Single transaction inference engine
│   ├── bulk_analysis.py       # Batch CSV screening module
│   ├── history.py             # SQLite audit trail & record management
│   └── model_insights.py      # Benchmark curves and diagnostic metrics
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV validation and dataset loader
│   ├── preprocessing.py       # RobustScaler pipeline & stratified splitter
│   ├── train_model.py         # Model training & threshold calibration
│   ├── predict.py             # Vectorized inference & risk triage
│   ├── evaluation.py          # PR-AUC, ROC, and metric calculations
│   └── database.py            # SQLite schema, queries, and migrations
├── notebooks/
│   └── fraud_analysis.ipynb   # Exploratory data analysis notebook
├── data/
│   └── .gitkeep               # Directory for creditcard.csv (gitignored)
├── models/
│   └── .gitkeep               # Directory for trained joblib artifacts
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py  # Tests for scaling, ordering & leakage
│   ├── test_model.py          # Tests for artifacts, metrics & thresholds
│   ├── test_prediction.py     # Tests for single/bulk inference & schema
│   └── test_database.py       # Tests for SQLite storage, filters & stats
├── assets/
│   └── logo.png               # Brand emblem
├── .streamlit/
│   └── config.toml            # Dark theme & server parameters
├── requirements.txt           # Verified dependencies
├── runtime.txt                # Target Python runtime
├── pytest.ini                 # Pytest configuration
├── .gitignore                 # Excludes datasets, DBs, and venvs
├── README.md                  # Comprehensive technical documentation
└── LICENSE                    # Project license
```

---

## 5. Machine Learning Methodology & Performance

### Training Configuration
- **Dataset**: European Cardholder Transactions (284,807 rows)
- **Partitioning**: Stratified 80/20 split (227,845 train / 56,962 test)
- **Imbalance Mitigation**: SMOTE applied exclusively to training vectors (sampling strategy: 0.1)
- **Scaling**: `RobustScaler` applied to `Time` and `Amount` (resilient to monetary outliers)

### Evaluation on Held-Out Test Set (56,962 transactions, 98 fraud cases)

| Metric | Random Forest (Selected) | Logistic Regression (Baseline) |
|---|---|---|
| **PR-AUC (Primary Metric)** | **0.8333** | 0.7213 |
| **Fraud Recall** | **85.71% (84 / 98 caught)** | 47.96% |
| **Precision** | **73.04%** | 81.03% |
| **F1-Score** | **0.7887** | 0.6026 |
| **ROC-AUC** | **0.9632** | 0.9680 |
| **Calibrated Threshold** | **0.6202** | 1.0000 |

> **Selection Rationale**: In extreme fraud imbalance (0.17%), **PR-AUC** is the decisive metric. Random Forest captures 85.7% of fraud while maintaining 73% precision, preventing false positive alarms from overwhelming fraud investigation teams.

---

## 6. Installation and Setup Instructions

### Prerequisites
- Python 3.11 installed
- Git installed

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/FRAUDGUARD.git
cd FRAUDGUARD
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Setup
1. Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the file inside the `data/` folder: `data/creditcard.csv`.
3. *(Alternative)* If starting without the dataset file, launch the web application and upload `creditcard.csv` directly via the **Dataset & System Status** panel.

### 5. Train the Machine Learning Pipeline
```bash
python -m src.train_model
```
This script validates data integrity, executes the stratified train/test split, trains both models, computes evaluation curves, and saves the trained artifacts into `models/`.

### 6. Run the Test Suite
```bash
pytest -v
```
All 17 tests verify data preprocessing, data leakage prevention, schema validation, single/batch prediction, threshold tuning, and SQLite persistence.

### 7. Launch the Web Application
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 7. Streamlit Community Cloud Deployment

1. Fork or push the repository to GitHub (ensure `data/creditcard.csv` and local `.db` files remain gitignored as configured in `.gitignore`).
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your repository, select branch `main`, and set the main file path to `app.py`.
4. Deploy the application. The system will start cleanly, load stored model metadata, and allow users to upload datasets and evaluate transactions immediately.

---

## 8. Limitations & Future Roadmap

- **PCA Feature Anonymity**: Original features are PCA transformed due to European cardholder confidentiality, limiting domain-specific feature engineering (e.g., merchant category code, geolocation).
- **Future Integration**: Implement streaming Kafka ingestion for high-throughput sub-millisecond fraud scoring, and explore Graph Neural Networks (GNNs) for detecting coordinated fraud syndicates.

---

## 9. Author and Developer

- **Developer & Owner**: Isbandu Rahuman R.
- **Project**: FRAUDGUARD
- **Portfolio Edition**: 2026
