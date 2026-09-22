 🛡️ Network Intrusion Detection System
> A machine learning-based IDS that classifies network traffic as normal or one of four
> attack categories (DoS, Probe, R2L, U2R), served as a web app + REST API deployable on
> Vercel, and originally developed against the KDD Cup 99 dataset (Random Forest, up to
> 99.97% accuracy on the real dataset — see "Model & data" below for what's bundled here).

---

 🚀 Live app

- `GET /` — a form-based UI: load a sample connection or fill in the 30 traffic features,
  click Predict, see the classification + per-class confidence.
- `POST /api/predict` — JSON API. Body = the 30 features (see
  `common/preprocessing.py::FINAL_FEATURE_ORDER`); returns `{"prediction": ..., "probabilities": {...}}`.
- `GET /api/health` — model/deployment status (data source, accuracy, feature schema).

---

 📂 Repository structure

```
network-intrusion-detection-system/
├── network_intrusion.ipynb   Original EDA + preprocessing + model-comparison notebook
├── common/preprocessing.py   Single source of truth for the 30-feature schema, shared
│                             by training and inference (keeps them from drifting apart)
├── model/
│   ├── train.py              Trains the Random Forest and saves deployable artifacts
│   ├── synthetic.py          Schema-accurate synthetic data generator (fallback, see below)
│   └── artifacts/            model.joblib, scaler.joblib, metadata.json (committed, ~170KB)
├── api/
│   ├── index.py              Single Vercel Python entrypoint (this account's Vercel
│   │                         Python runtime only auto-detects one); GET /api/health and
│   │                         POST /api/predict both route here via vercel.json rewrites,
│   │                         dispatched by HTTP method
│   └── requirements.txt      (mirrors root requirements.txt)
├── public/                   Static frontend (index.html, app.js, style.css, examples.json)
├── data/README.md            How to fetch the real dataset for a production retrain
├── requirements.txt          Runtime deps for the deployed API (numpy, scikit-learn, joblib)
├── requirements-dev.txt      + pandas, for local training
└── vercel.json
```

---

 🧠 Model & data — read this before trusting accuracy numbers

The original notebook trains on the real **KDD Cup 1999** 10%-subset (494,021 rows,
downloaded manually — see `data/README.md`). That file is ~70MB, isn't included in this
repo, and this build environment had no network access to fetch it automatically.

So that the app is fully deployable and testable out of the box, `model/train.py` falls
back to a **synthetic, schema-identical dataset** (`model/synthetic.py`) when the real file
isn't present at `data/kddcup.data_10_percent_corrected`. The bundled `model/artifacts/`
were built this way — `metadata.json`'s `"source": "synthetic"` and the health endpoint
both say so explicitly. It exercises the exact same preprocessing/training/inference code
path as real data, but its accuracy numbers are **not meaningful** — it's a demo, not a
validated classifier.

**To deploy a production-accuracy model:**
```bash
pip install -r requirements-dev.txt
# download the real dataset per data/README.md, then:
python model/train.py
git add model/artifacts && git commit -m "Retrain on real KDD Cup 99 data" && git push
```
`train.py` auto-detects the real file and uses it in preference to synthetic data.

 Model size & Vercel limits
`train.py` caps the Random Forest at `n_estimators=30, max_depth=14` (vs. unbounded depth
in the original notebook) specifically so the serialized model stays small — Vercel
serverless functions have a 250MB unzipped bundle limit, and scikit-learn/numpy/scipy
themselves already consume a large share of that. The synthetic model is ~170KB; a real
10%-KDD-trained model at these settings should still be a few MB. If you increase
`max_depth`/`n_estimators` for the real dataset, re-check the deployed function size
(`vercel inspect` after deploying, or the dashboard's function size panel) before shipping.

---

 🐛 Bugs fixed vs. the original notebook

- **Silent NaN propagation on unseen categorical values.** The notebook's
  `.map(pmap)`/`.map(fmap)` produce `NaN` for any `protocol_type`/`flag` value outside the
  training set's vocabulary, which `RandomForestClassifier` doesn't accept — a malformed
  request would previously either crash with an opaque error or (depending on sklearn
  version) silently corrupt predictions. `common/preprocessing.py::encode_record` now
  validates every field up front and raises a clear, user-facing 400 error naming exactly
  which field/value is invalid.
- **No reproducible environment.** No `requirements.txt` existed; exact dependency
  versions weren't pinned anywhere, so notebook runs weren't reproducible. Added pinned
  `requirements.txt` / `requirements-dev.txt`.
- **No deployable artifact.** The notebook trained 6 models in-memory and never persisted
  anything — there was nothing to deploy. Added `model/train.py`, which saves versioned,
  reloadable `model.joblib` + `scaler.joblib` + `metadata.json`.
- **Unhandled exceptions would leak internals.** `api/index.py` catches
  input-validation errors (400), missing-artifact errors (503), and unexpected errors
  (500, generic message — set `IDS_DEBUG=1` in Vercel's environment variables to include
  exception details while debugging, and unset it in production).
- **No `.gitignore`.** Local venvs, `__pycache__`, and the large raw dataset file had no
  guard against being accidentally committed.

---

 🧩 Methodology (unchanged from the original notebook)

 1. Preprocessing
- 41 raw KDD features + label; 22 attack names mapped → 4 categories (DoS, Probe, R2L, U2R) + normal
- Label-encoded `protocol_type`, `flag`; dropped `service` (high cardinality, low signal)
- Removed 8 highly-correlated features (heatmap analysis) and 2 zero-variance features
- MinMaxScaler normalization → **30 final features**
- Train/test split: 67% / 33%

 2. Models compared (see notebook for full results)
Naive Bayes, Decision Tree, Random Forest, SVM, Logistic Regression, Gradient Boosting —
**Random Forest** was selected for deployment: best accuracy (99.97% on the real dataset
per the original notebook run) at a practical training/inference cost.

---

 🔧 Tech stack

| Layer | Tool |
|---|---|
| Model | scikit-learn (RandomForestClassifier), MinMaxScaler |
| API | Python stdlib `http.server` handlers, deployed as Vercel Python serverless functions |
| Frontend | Static HTML/CSS/vanilla JS (no build step) |
| Hosting | Vercel |

---

 🖥️ Local development

```bash
# 1. Install dev deps (includes pandas, needed only for training)
pip install -r requirements-dev.txt

# 2. Train (uses synthetic data unless the real dataset is present, see data/README.md)
python model/train.py

# 3. Serve locally with the Vercel CLI (installs on first run)
npx vercel dev
```
Then open http://localhost:3000.

Run the original EDA notebook (`network_intrusion.ipynb`) separately for exploratory
analysis, correlation heatmaps, and the full 6-model comparison; it isn't part of the
deployed app.

---

 ☁️ Deploying to Vercel

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. In the [Vercel dashboard](https://vercel.com/new), import the repository. No build
   command or framework preset is needed — Vercel auto-detects the static `public/`
   output (via `vercel.json`'s `outputDirectory`) and the `api/*.py` Python functions.
3. Deploy. Verify `GET /api/health` returns `"status": "ok"` and check whether
   `model.source` is `"synthetic"` or `"real"` before treating predictions as trustworthy.

Or via CLI: `npx vercel --prod` from the repo root.

---

 🔭 Future work

- [ ] Retrain and ship on the full real KDD Cup 99 dataset (see `data/README.md`)
- [ ] Add XGBoost / LSTM models for deeper comparison
- [ ] Evaluate on UNSW-NB15 and CICIDS2017 for generalizability
- [ ] SHAP-based feature importance explainability in the UI

---

 👩‍💻 Author

Pooja V — B.E. CSE (AI & ML), Nagarjuna College of Engineering & Technology, Bengaluru

 📄 License

MIT License.
