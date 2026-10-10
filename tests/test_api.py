from fastapi.testclient import TestClient

from src import api

client = TestClient(api.app)


def test_health_is_liveness_even_while_model_loads(monkeypatch):
    monkeypatch.setitem(api._state, "stage", "loading_model")
    monkeypatch.setitem(api._state, "model_loaded", False)
    monkeypatch.setitem(api._state, "warmed_up", False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["warmed_up"] is False
    assert response.headers.get("x-request-id")


def test_ready_is_503_until_warmup_finishes(monkeypatch):
    monkeypatch.setitem(api._state, "stage", "warming_up")
    monkeypatch.setitem(api._state, "model_loaded", True)
    monkeypatch.setitem(api._state, "warmed_up", False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["stage"] == "warming_up"


def test_ready_is_success_only_when_loaded_and_warmed(monkeypatch):
    monkeypatch.setitem(api._state, "stage", "ready")
    monkeypatch.setitem(api._state, "model_loaded", True)
    monkeypatch.setitem(api._state, "warmed_up", True)
    monkeypatch.setitem(api._state, "backend", "onnxruntime-cpu")
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model_loaded"] is True
    assert response.json()["warmed_up"] is True


def test_info_documents_limits_and_variant():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["variant"] == 16
    assert data["class"] == "hammer"
    assert data["max_upload_mb"] == 4
    assert data["max_image_side_px"] == 1280


def test_unsupported_media_type_is_rejected():
    response = client.post("/detect", files={"file": ("note.txt", b"not an image", "text/plain")})
    assert response.status_code == 415
    assert response.json()["request_id"]


def test_oversized_upload_is_rejected_before_decode():
    response = client.post("/detect", files={"file": ("large.jpg", b"0" * (4 * 1024 * 1024 + 1), "image/jpeg")})
    assert response.status_code == 413
    assert response.json()["request_id"]


def test_invalid_image_is_rejected_without_crashing():
    response = client.post("/detect", files={"file": ("broken.jpg", b"not a jpeg", "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["request_id"]


def test_static_ui_is_served_from_single_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Nazdratenko" in response.text
    assert "Найди" in response.text
