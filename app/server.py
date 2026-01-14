import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .engine import NaiveBayesIntent

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "intents.json"
WEB_DIR = BASE_DIR / "web"

MODEL = NaiveBayesIntent()


def load_samples():
    if not DATA_PATH.exists():
        return []
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def save_samples(samples):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(samples, indent=2), encoding="utf-8")


def seed_samples():
    samples = [
        {"text": "reset my password", "intent": "account_support"},
        {"text": "I cannot log in", "intent": "account_support"},
        {"text": "upgrade to pro plan", "intent": "billing"},
        {"text": "how much does enterprise cost", "intent": "billing"},
        {"text": "schedule a demo", "intent": "sales"},
        {"text": "can we book a call", "intent": "sales"},
        {"text": "report a bug", "intent": "product_feedback"},
        {"text": "feature request for analytics", "intent": "product_feedback"},
    ]
    save_samples(samples)
    return samples


def train_model():
    samples = load_samples()
    if not samples:
        samples = seed_samples()
    MODEL.train(samples)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        return

    def _send_json(self, payload, status=HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            payload = self._read_json()
            text = payload.get("text", "")
            ranked = MODEL.predict(text)
            self._send_json({"results": [{"intent": k, "score": v} for k, v in ranked]})
            return
        if parsed.path == "/api/seed":
            samples = seed_samples()
            MODEL.train(samples)
            self._send_json({"count": len(samples)})
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        super().do_GET()


def run(host="127.0.0.1", port=5173):
    train_model()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"IntentForge running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run IntentForge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()

    run(host=args.host, port=args.port)
