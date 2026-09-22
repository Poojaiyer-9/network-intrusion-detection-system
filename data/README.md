# Dataset

This directory intentionally does not ship the KDD Cup 99 dataset (it's
~70MB, and redistribution isn't clearly licensed).

To train on the real data instead of the bundled synthetic demo model:

1. Download `kddcup.data_10_percent.gz` from
   http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
2. Decompress it and place it here as:
   `data/kddcup.data_10_percent_corrected`
3. Run:
   ```bash
   pip install -r requirements-dev.txt
   python model/train.py
   ```
4. Commit the regenerated `model/artifacts/*.joblib` files (or wire them
   into your CI/CD pipeline) before deploying.

`model/train.py` automatically detects the real file here and uses it;
otherwise it falls back to a synthetic, schema-identical dataset so the app
still runs end-to-end without the download.
