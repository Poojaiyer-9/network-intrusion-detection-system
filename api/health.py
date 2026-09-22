"""Vercel Python serverless function: GET /api/health — deployment/model status."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "model" / "artifacts"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        metadata_path = ARTIFACTS_DIR / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            body = {"status": "ok", "model": metadata}
            status = 200
        else:
            body = {"status": "error", "detail": "Model artifacts missing"}
            status = 503

        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
