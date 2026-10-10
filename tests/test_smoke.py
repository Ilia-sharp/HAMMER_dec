"""Интеграционный smoke-тест для уже запущенного API."""
import os
from pathlib import Path

import httpx
import pytest

BASE_URL = os.getenv("SMOKE_BASE_URL")
DEMO_PATH = Path(__file__).resolve().parents[1] / "static" / "img" / "demo.jpg"
pytestmark = pytest.mark.skipif(not BASE_URL, reason="Set SMOKE_BASE_URL to run the external smoke test")


def test_running_service_end_to_end():
    with httpx.Client(base_url=BASE_URL.rstrip("/"), timeout=180.0) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        ready = client.get("/ready")
        assert ready.status_code == 200, f"Model not ready: {ready.text}"
        assert ready.json()["status"] == "ready"

        info = client.get("/info")
        assert info.status_code == 200
        assert info.json()["variant"] == 16

        assert DEMO_PATH.is_file(), f"Demo image is missing: {DEMO_PATH}"
        with DEMO_PATH.open("rb") as image:
            result = client.post("/detect", files={"file": ("demo.jpg", image, "image/jpeg")})
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["request_id"] and payload["model_version"]
        assert isinstance(payload["detections"], list)
        assert isinstance(payload["inference_ms"], (int, float))
        assert payload["count"] == len(payload["detections"])
