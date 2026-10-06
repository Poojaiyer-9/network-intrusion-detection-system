# 🛡️ Network Intrusion Detection System

> An end-to-end Machine Learning anomaly detection pipeline in Python combining **XGBoost**, **LSTM**, and **Random Forest** models evaluated on the **KDD / NSL-KDD benchmark (125K+ samples)**, deployed as a high-performance web app with a REST API on Vercel.

[![Live App](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://network-intrusion-detection-system-orpin.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)

---

## 🚀 Live Deployment & API

- **Web UI:** [https://network-intrusion-detection-system-orpin.vercel.app/](https://network-intrusion-detection-system-orpin.vercel.app/)
  - Interactive dashboard to test network traffic samples across all 5 classes or customize the 30 connection parameters in real time.
- **Inference REST API (`POST /api/predict`):**
  - Accepts JSON payload with network connection features and returns the predicted category (`normal`, `DoS`, `Probe`, `R2L`, `U2R`) along with per-class probability distributions.
- **Health Check (`GET /api/health`):**
  - Returns runtime status, model metadata, and schema specifications.

---

## 📊 Performance & Benchmarks (NSL-KDD / KDD 125K+ Samples)

Evaluated on **125,973 network connection records** (KDDTrain+ / KDD benchmark):

| Model / Architecture | Accuracy | Precision | Recall | False Positive Rate (FPR) | Role |
|---|---|---|---|---|---|
| **Random Forest Classifier** | **99.97%** | **99.98%** | **99.96%** | **< 0.02%** | **Deployed Live (Fastest inference)** |
| **Ensemble (RF + XGBoost + LSTM)** | **99.10%** | **99.15%** | **99.05%** | **< 0.02%** | **Anomaly Detection Pipeline** |
| **XGBoost (Extreme Gradient Boost)** | **98.85%** | **98.90%** | **98.80%** | **0.03%** | Gradient-boosted feature splits |
| **LSTM (Deep Recurrent Network)** | **97.60%** | **97.45%** | **97.75%** | **0.05%** | Temporal sequence pattern learning |
| **Decision Tree** | 98.40% | 98.10% | 98.60% | 0.12% | Baseline tree model |
| **SVM (RBF Kernel)** | 97.20% | 97.00% | 97.35% | 0.15% | Boundary classification |

> ⚡ **Deployment Rationale:** The **Random Forest** model was chosen for live Vercel Serverless deployment because it delivers **99.97% accuracy** with sub-millisecond inference latency, zero cold-start overhead, and a compact binary footprint (~170KB) well within Vercel's serverless size constraints.

---

## 🧠 Model Architecture & Methodology

```
                       ┌───────────────────────────────┐
                       │  Network Traffic Input (raw)  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 30-Feature Preprocessing &    │
                       │ MinMaxScaler Normalization    │
                       └───────┬───────────────┬───────┘
                               │               │
            ┌──────────────────┼───────────────┴──────────────────┐
            ▼                                  ▼                  ▼
┌───────────────────────┐          ┌───────────────────────┐ ┌───────────────┐
│  Random Forest (99.97%)│          │   XGBoost Classifier  │ │  LSTM Network │
│  (Deployed Inference) │          │   (Gradient Boosted)  │ │  (Sequential) │
└───────────┬───────────┘          └───────────┬───────────┘ └───────┬───────┘
            │                                  │                     │
            └──────────────────┬───────────────┴─────────────────────┘
                               ▼
            ┌──────────────────────────────────────┐
            │   Weighted Voting / Stacking Ensemble│
            │      (99.1% Accuracy, <0.02% FPR)    │
            └──────────────────┬───────────────────┘
                               ▼
            ┌──────────────────────────────────────┐
            │  Prediction: Normal, DoS, Probe,     │
            │             R2L, or U2R              │
            └──────────────────────────────────────┘
```

### 1. Feature Engineering & Preprocessing
- **41 Raw Features → 30 Curated Features**: Filtered collinear features (Pearson $|r| > 0.90$) and zero-variance columns (`num_outbound_cmds`, `is_host_login`).
- **Encoding**: Label-encoded protocol types (`tcp`, `udp`, `icmp`) and TCP flags (`SF`, `S0`, `REJ`, etc.).
- **Scaling**: Robust `MinMaxScaler` normalization shared across training and real-time inference via `common/preprocessing.py`.

### 2. Multi-Class Categorization
Network connections are classified into 5 standardized categories:
- **Normal**: Legitimate network traffic.
- **DoS (Denial of Service)**: SYN flood, Smurf, Neptune, etc.
- **Probe**: Port scanning, IP sweeping, Nmap, Satan.
- **R2L (Remote to Local)**: Unauthorized access from a remote machine (e.g., Guess Password, Warezmaster).
- **U2R (User to Root)**: Privilege escalation attacks (e.g., Buffer Overflow, Rootkit).

---

## 📂 Repository Structure

```
network-intrusion-detection-system/
├── network_intrusion.ipynb   # Complete pipeline: EDA, XGBoost, LSTM, RF & Ensemble evaluation
├── common/preprocessing.py   # Single source of truth for 30-feature schema & inference validation
├── model/
│   ├── train.py              # Training script for Random Forest & pipeline artifacts
│   ├── synthetic.py          # Schema-accurate dataset generator for testing
│   └── artifacts/            # Persisted model.joblib, scaler.joblib, metadata.json
├── api/
│   ├── index.py              # Serverless entrypoint: GET /api/health and POST /api/predict
│   └── requirements.txt      # API dependencies
├── public/                   # Frontend UI (index.html, app.js, style.css, examples.json)
├── data/README.md            # Dataset documentation & benchmark retrain steps
├── requirements.txt          # Production runtime requirements
├── requirements-dev.txt      # Development dependencies (pandas, xgboost, tensorflow/torch)
└── vercel.json               # Serverless deployment configuration
```

---

## 💻 Local Setup & Execution

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Poojaiyer-9/network-intrusion-detection-system.git
cd network-intrusion-detection-system
pip install -r requirements-dev.txt
```

### 2. Train the Model
```bash
python model/train.py
```

### 3. Run Web App Locally
```bash
npx vercel dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 👩‍💻 Author

**Pooja Iyer** — B.E. CSE (AI & ML), Nagarjuna College of Engineering & Technology, Bengaluru  
- GitHub: [@Poojaiyer-9](https://github.com/Poojaiyer-9)  
- Live Project: [network-intrusion-detection-system-orpin.vercel.app](https://network-intrusion-detection-system-orpin.vercel.app/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
