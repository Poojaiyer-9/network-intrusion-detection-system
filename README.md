 🛡️ Network Intrusion Detection System
> A machine learning-based IDS that detects cyberattacks in network traffic using the KDD Cup 99 dataset — achieving up to 99.97% accuracy with Random Forest.




 📌 Overview

Traditional firewalls and rule-based systems struggle to detect novel or evolving cyberattacks. This project builds an intelligent, data-driven Intrusion Detection System (IDS) that uses supervised machine learning to classify network traffic as normal or one of four attack categories:

| Attack Type | Examples |
|---|---|
| DoS (Denial of Service) | smurf, neptune, teardrop |
| Probe | ipsweep, nmap, satan, portsweep |
| R2L (Remote to Local) | ftp_write, guess_passwd, imap |
| U2R (User to Root) | buffer_overflow, rootkit, perl |

---

 📂 Repository Structure

```
network-intrusion-detection-system/
│
├── intrusion_detection_system.ipynb    Main notebook (EDA + Preprocessing + Modelling)
├── README.md                           Project documentation
```

---

 📊 Dataset

KDD Cup 1999 Dataset — one of the most widely used benchmarks for IDS research.

| Property | Value |
|---|---|
| Source | [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) |
| File used | `kddcup.data_10_percent_corrected` |
| Total records | 494,021 |
| Features | 41 network traffic attributes |
| Classes | normal, dos, probe, r2l, u2r |
| Missing values | None |

Key features include: `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, and 35+ statistical traffic features.

---

 ⚙️ Methodology

 1. Data Preprocessing
- Loaded KDD Cup dataset with 41 features + target label
- Mapped 22 specific attack names → 4 attack categories + normal
- Label-encoded categorical features: `protocol_type`, `flag`
- Dropped `service` (high cardinality, low signal)
- Removed 8 highly correlated features identified via heatmap analysis:
  `num_root`, `srv_serror_rate`, `srv_rerror_rate`, `dst_host_srv_serror_rate`,
  `dst_host_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate`, `dst_host_same_srv_rate`
- Dropped 2 zero-variance features: `is_host_login`, `num_outbound_cmds`
- Applied MinMaxScaler normalization
- Final feature set: 30 features
- Train/Test split: 330,994 / 163,027 samples (67/33)

 2. Models Trained
Six classifiers were implemented and compared:
- Naive Bayes (GaussianNB)
- Decision Tree (entropy criterion, max_depth=4)
- Random Forest (n_estimators=30)
- Support Vector Machine (RBF kernel)
- Logistic Regression
- Gradient Boosting

 3. Evaluation
Models were evaluated on test accuracy and training/inference time.

---

 📈 Results

| Model | Train Accuracy | Test Accuracy | Train Time |
|---|---|---|---|
| 🥇 Random Forest | 100.00% | 99.97% | 5.2s |
| Gradient Boosting | 99.91% | 99.91% | 165.4s |
| SVM | 99.88% | 99.88% | 111.6s |
| Decision Tree | 99.39% | 99.38% | 0.7s |
| Logistic Regression | 99.36% | 99.36% | 9.2s |
| Naive Bayes | 87.95% | 87.90% | 0.6s |

Random Forest achieves the best accuracy with a practical training time, making it the best overall model for this task.

---

 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas & NumPy | Data loading, manipulation |
| Scikit-learn | ML models, preprocessing, evaluation |
| Matplotlib & Seaborn | EDA visualizations, correlation heatmap |
| Google Colab (T4 GPU) | Training environment |

---

 🚀 How to Run

 Option 1: Google Colab (Recommended)
1. Open the notebook in Colab
2. Upload the KDD Cup dataset files:
   - `kddcup.data_10_percent_corrected`
   - `kddcup.names.txt`
   - `training_attack_types.txt`
3. Run all cells sequentially

Option 2: Local Setup
```bash
Clone the repo
git clone https://github.com/Poojaiyer-9/network-intrusion-detection-system.git
cd network-intrusion-detection-system

 Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

Launch notebook
jupyter notebook intrusion_detection_system.ipynb
```

> Dataset download: [KDD Cup 99 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

🔍 Key Findings

- Feature engineering matters — removing 10 redundant/zero-variance features improved model efficiency without hurting accuracy
- Ensemble methods dominate — Random Forest and Gradient Boosting outperform single classifiers significantly
- Speed vs accuracy tradeoff — Gradient Boosting matches Random Forest accuracy but takes 32× longer to train
- Naive Bayes limitation — its independence assumption doesn't hold well for correlated network traffic features

---

 🔭 Future Work

- [ ] Add XGBoost and LSTM models for deeper comparison
- [ ] Evaluate on UNSW-NB15 and CICIDS2017 for generalizability
- [ ] Implement real-time traffic simulation and prediction
- [ ] Add SHAP-based feature importance explainability
- [ ] Deploy as a REST API using FastAPI or Flask


👩💻 Author

Pooja V — B.E. CSE (AI & ML), Nagarjuna College of Engineering & Technology, Bengaluru


📄 License

This project is licensed under the MIT License.

