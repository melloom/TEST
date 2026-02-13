from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict

from confidence_layer import ConfidenceRiskEngine
from config import CONFIG_VERSION
from evaluation import EvalExample, EvaluationHarness
from schema import SCHEMA_VERSION, validate_profile


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("confidence-layer-api")

engine = ConfidenceRiskEngine()
evaluator = EvaluationHarness(engine)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/evaluate":
            self._handle_evaluate()
            return

        if self.path != "/predict-confidence-risk":
            self._send(404, {"error": "not found"})
            return

        started = time.time()
        try:
            payload = self._read_json_payload()
            text = str(payload.get("text", ""))
            profile = validate_profile(str(payload.get("profile", "balanced")))
            prediction = engine.predict(text, profile=profile).to_dict()
            self._send(200, prediction)
            LOGGER.info(
                "predict ok path=%s profile=%s risk_label=%s is_ood=%s duration_ms=%d",
                self.path,
                profile,
                prediction["risk_label"],
                prediction["is_ood"],
                int((time.time() - started) * 1000),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            self._send(400, {"error": "invalid json payload"})
            LOGGER.warning("predict bad_request path=%s", self.path)

    def _handle_evaluate(self) -> None:
        started = time.time()
        try:
            payload = self._read_json_payload()
            profile = validate_profile(str(payload.get("profile", "balanced")))
            raw_examples = payload.get("examples", [])
            examples = [
                EvalExample(
                    text=str(ex.get("text", "")),
                    risk_label=str(ex.get("risk_label", "safe")),
                    is_ood=bool(ex.get("is_ood", False)),
                )
                for ex in raw_examples
                if isinstance(ex, dict)
            ]
            report = evaluator.evaluate(examples, profile=profile)
            self._send(200, report)
            LOGGER.info(
                "evaluate ok path=%s profile=%s samples=%d duration_ms=%d",
                self.path,
                profile,
                report["count"],
                int((time.time() - started) * 1000),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            self._send(400, {"error": "invalid json payload"})
            LOGGER.warning("evaluate bad_request path=%s", self.path)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"ok": True})
            return
        if self.path == "/schema":
            self._send(
                200,
                {
                    "schema_version": SCHEMA_VERSION,
                    "config_version": CONFIG_VERSION,
                    "risk_labels": ["safe", "caution", "high_risk"],
                    "actions": ["allow", "warn", "block", "escalate"],
                    "profiles": ["strict", "balanced", "lenient"],
                    "ood_module": {"calibration": True, "threshold_fitting": True},
                    "evaluation": {"endpoint": "/evaluate", "dashboard": True, "error_analysis": True},
                    "logging": {"enabled": True, "fields": ["path", "profile", "risk_label", "is_ood", "duration_ms"]},
                },
            )
            return
        if self.path == "/config":
            self._send(200, engine.get_runtime_config())
            return
        self._send(404, {"error": "not found"})

    def _read_json_payload(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), Handler)
    LOGGER.info("Serving confidence layer API on http://0.0.0.0:%d", port)
    server.serve_forever()


if __name__ == "__main__":
    run()
