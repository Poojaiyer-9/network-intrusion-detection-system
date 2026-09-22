"""
Vercel Python serverless function — single entrypoint.

This project's Vercel Python runtime requires exactly one entrypoint file to
auto-detect (multiple api/*.py handler files triggered a "No python
entrypoint found in default locations" build error). GET /api/health and
POST /api/predict both resolve here via vercel.json rewrites; dispatch is by
HTTP method (not path), so it works regardless of which URL reached this
function:
  - GET  -> health/status info (model metadata + accepted field schema)
  - POST -> run a prediction on the JSON body

Body for POST: JSON object with the 30 feature fields described in
common.preprocessing.FINAL_FEATURE_ORDER (protocol_type/flag as strings,
e.g. "tcp"/"SF"; everything else numeric). See public/examples.json for
sample payloads per attack category.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402

from common.preprocessing import FINAL_FEATURE_ORDER, InvalidInputError, encode_record  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "model" / "artifacts"

_model = None
_scaler = None


def _load_artifacts():
    global _model, _scaler
    if _model is None or _scaler is None:
        _model = joblib.load(ARTIFACTS_DIR / "model.joblib")
        _scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    return _model, _scaler


def _predict(payload: dict) -> dict:
    model, scaler = _load_artifacts()
    vector = encode_record(payload)
    scaled = scaler.transform([vector])
    prediction = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    probabilities = {cls: round(float(p), 4) for cls, p in zip(model.classes_, proba)}
    return {"prediction": prediction, "probabilities": probabilities}


def _health() -> dict:
    metadata_path = ARTIFACTS_DIR / "metadata.json"
    if not metadata_path.exists():
        return {"status": "error", "detail": "Model artifacts missing"}
    return {
        "status": "ok",
        "model": json.loads(metadata_path.read_text()),
        "fields": FINAL_FEATURE_ORDER,
    }


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        data = _health()
        self._send_json(200 if data["status"] == "ok" else 503, data)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 1_000_000:
                raise InvalidInputError("Request body missing or too large")
            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise InvalidInputError(f"Body is not valid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise InvalidInputError("Body must be a JSON object")

            result = _predict(payload)
            self._send_json(200, result)
        except InvalidInputError as exc:
            self._send_json(400, {"error": str(exc)})
        except FileNotFoundError:
            self._send_json(
                503,
                {"error": "Model artifacts not found. Run model/train.py and redeploy."},
            )
        except Exception as exc:  # noqa: BLE001
            debug = os.environ.get("IDS_DEBUG") == "1"
            self._send_json(500, {"error": "Internal error" + (f": {exc}" if debug else "")})
