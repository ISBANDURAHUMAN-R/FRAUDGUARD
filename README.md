<p align="center">
  <img src="assets/banner.png" alt="FRAUDGUARD Banner" width="100%">
</p>

# ◈ FRAUDGUARD — AI-Powered Credit Card Fraud Detection Platform

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/imbalanced--learn-0.12%2B-orange.svg)](https://imbalanced-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24%2B-3F4F75.svg)](https://plotly.com/)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57.svg)](https://www.sqlite.org/)
[![Tests: Pytest](https://img.shields.io/badge/pytest-17%20passed-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ◆ About the Project
**FRAUDGUARD** is an enterprise-grade financial cyber-intelligence and transaction surveillance platform designed to identify, intercept, and audit fraudulent payment activity in real-time. Operating on the European Credit Card benchmark comprising 284,807 transactions with an authentic 0.17% fraud prevalence rate, FRAUDGUARD combines SMOTE-resampled Random Forest classification, outlier-resistant `RobustScaler` transformations, cost-sensitive decision threshold optimization, and an interactive 8-page Streamlit security console backed by SQLite audit persistence.

```mermaid
flowchart TD
    subgraph Ingestion [" Ingestion & Validation "]
        CSV[" European Credit Card Dataset <br/> (284,807 Transactions) "] --> Val[" Schema Validation &amp; <br/> Data Integrity Check "]
        Val --> Split[" Stratified Train-Test Split <br/> (80% Train / 20% Test) "]
    end

    subgraph Pipeline [" Preprocessing & ML Training "]
        Split -->|" Train Set Only "| SMOTE[" SMOTE Balancing Engine <br/> (Sampling Ratio: 0.1) "]
        Split --> Scaler[" RobustScaler Transformation <br/> (Time &amp; Amount Outliers) "]
        SMOTE --> RF[" Random Forest Classifier <br/> (n_estimators=100) "]
        Scaler --> RF
        RF --> Calib[" Cost-Sensitive Threshold Calibration <br/> (Optimal Cutoff: 0.6202) "]
    end

    subgraph Serving [" Inference & Intelligence Serving "]
        Calib --> Engine[" Real-Time Scoring Engine <br/> (Vectorized Prediction Pipeline) "]
        Engine --> Single[" Single Transaction Scoring <br/> (30 Telemetry Features) "]
        Engine --> Bulk[" Bulk CSV Stream Scanner <br/> (Batch Risk Assessment) "]
        Engine --> Monitor[" Live Threat Surveillance <br/> &amp; Capital Exposure Dashboard "]
    end

    subgraph Storage [" Audit Trail & Persistence "]
        Single --> DB[(" SQLite Database <br/> (Thread-Safe Transaction Audit) ")]
        Bulk --> DB
        DB --> Hist[" Historical Incident Review <br/> &amp; Analyst Governance "]
    end
```

## ❖ Key Features
- **Zero Data Leakage Pipeline**: Stratified 80/20 train/test partition executed prior to fitting scalers; synthetic oversampling (SMOTE) is strictly confined to the training set.
- **Cost-Sensitive Threshold Tuning**: Precision-Recall AUC optimization balances financial recall ($85.7\%$) against false-positive customer friction ($73.0\%$ precision).
- **Sub-Second Real-Time Inference**: Vectorized single-transaction scoring evaluates 30 PCA features, classifying transactions into risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
- **High-Throughput Bulk CSV Scanner**: Parallel stream validation and scoring for uploaded transaction files with CSV report export.
- **Executive Security Dashboard**: Live KPI tracking, fraud velocity telemetry, logarithmic amount distribution comparisons, and feature importance rankings.
- **Thread-Safe SQLite Audit Persistence**: Parameterized transactional storage logging predictions, probabilities, and analyst investigation notes.
- **Model Intelligence & Diagnostic Curves**: Direct PR-AUC, ROC-AUC, and Confusion Matrix evaluations computed against genuine held-out test splits.
- **Clean White & Green Enterprise UI**: Designed with high-contrast accessibility, zero emojis, and fast in-memory singleton caching.

## ⬢ Technologies Used
- **Core Language**: Python 3.11+
- **Machine Learning**: Scikit-Learn, Imbalanced-Learn (SMOTE), Joblib
- **Data Engineering**: Pandas, NumPy, PyArrow (High-Speed Binary Parquet Caching)
- **Interactive UI & Visuals**: Streamlit, Plotly Express & Graph Objects
- **Database & Persistence**: SQLite3 (Thread-Safe, Parameterized Queries)
- **Quality Assurance**: Pytest (17 automated unit and integration tests)

## 📊 Machine Learning Evaluation & Benchmarks
Evaluated on an authentic held-out test split of **56,962 transactions** containing **98 confirmed fraud cases**:

| Metric | Random Forest (Selected Model) | Logistic Regression (Baseline) | Performance Delta |
|:---|:---:|:---:|:---:|
| **PR-AUC (Primary Metric)** | **0.8333** | 0.7213 | **+11.20%** |
| **Fraud Recall** | **85.71% (84 / 98 caught)** | 47.96% (47 / 98) | **+37.75%** |
| **Precision** | **73.04%** | 81.03% | -7.99% |
| **F1-Score** | **0.7887** | 0.6026 | **+18.61%** |
| **ROC-AUC** | **0.9575** | 0.9452 | **+1.23%** |
| **False Positives** | **31** | 11 | Cost-Tolerant |

## § Requirements
- Python 3.11 or higher
- Git
- 2 GB RAM minimum (for 284,807 record in-memory dataset handling)
- Modern web browser (Chrome, Edge, Firefox, Safari)

## ▲ Installation & Execution
```bash
# 1. Clone Repository & Navigate
git clone https://github.com/ISBANDURAHUMAN-R/FRAUDGUARD.git
cd FRAUDGUARD

# 2. Setup Virtual Environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run Automated Test Suite (17 Tests)
pytest tests

# 5. Launch FRAUDGUARD Intelligence Platform
streamlit run app.py
```

Access the security operations console in your browser at `http://localhost:8501`.

## ※ Developer
Developer: ISBANDU RAHUMAN R  
GitHub: https://github.com/ISBANDURAHUMAN-R
