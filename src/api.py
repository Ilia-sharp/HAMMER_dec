"""ЛР8, вариант 16: одностраничный сервис детекции молотка на YOLO26n ONNX."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
CONFIG_PATH = ROOT / "configs" / "detect.yaml"
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_IMAGE_SIDE = 1280
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
MODEL_VERSION = os.getenv("MODEL_VERSION", "yolo26n-hammer-v1.0")
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "best.onnx")))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("nazdratenko.hammer")

try:
    _cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    _CONFIG_ERROR = None
except Exception as exc:
    _cfg, _CONFIG_ERROR = {}, type(exc).__name__

INFER_SIZE = int(_cfg.get("imgsz", 448))
CONF_THRESHOLD = float(_cfg.get("conf", 0.30))
IOU_THRESHOLD = float(_cfg.get("iou", 0.45))
MAX_DETECTIONS = int(_cfg.get("max_det", 10))

_state: dict[str, Any] = {
    "stage": "starting", "backend": None, "session": None, "input_name": None,
    "imgsz": INFER_SIZE, "model_loaded": False, "warmed_up": False, "error_code": None,
}
_state_lock = threading.RLock()
_inference_lock = threading.Lock()


def _set_state(**updates: Any) -> None:
    with _state_lock:
        _state.update(updates)


def _snapshot() -> dict[str, Any]:
    with _state_lock:
        return dict(_state)


def _json_log(event: str, **fields: Any) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event, "model_version": MODEL_VERSION, **fields,
    }
    logger.info(json.dumps(record, ensure_ascii=False, separators=(",", ":")))


def _letterbox(image: np.ndarray, target: int) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    scale = min(target / height, target / width)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (target - resized_width) / 2.0, (target - resized_height) / 2.0
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    boxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return boxed, scale, pad_x, pad_y


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, limit: int) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size and len(keep) < limit:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
            0.0, boxes[current, 3] - boxes[current, 1]
        )
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        union = area_current + area_rest - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def _predict_onnx(image: np.ndarray) -> list[dict[str, Any]]:
    state = _snapshot()
    session = state["session"]
    if session is None:
        raise RuntimeError("Модель ещё не загружена")

    target = int(state["imgsz"])
    height, width = image.shape[:2]
    boxed, scale, pad_x, pad_y = _letterbox(image, target)
    tensor = cv2.dnn.blobFromImage(boxed, 1.0 / 255.0, (target, target), swapRB=True, crop=False)
    raw = session.run(None, {state["input_name"]: tensor})[0]
    prediction = np.asarray(raw)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise RuntimeError("Неожиданная форма выхода ONNX-модели")
    # ONNX YOLO output usually contains (4 + class_count, anchors); transpose if needed.
    if prediction.shape[0] <= 128 and prediction.shape[1] > prediction.shape[0]:
        prediction = prediction.T
    if prediction.shape[1] < 5:
        raise RuntimeError("В выходе модели отсутствуют оценки объектов")

    boxes_xywh = prediction[:, :4].astype(np.float32, copy=False)
    if prediction.shape[1] == 5:
        scores = prediction[:, 4].astype(np.float32, copy=False)
        class_ids = np.zeros(scores.shape, dtype=np.int32)
    else:
        class_scores = prediction[:, 4:].astype(np.float32, copy=False)
        class_ids = class_scores.argmax(axis=1).astype(np.int32)
        scores = class_scores[np.arange(len(class_scores)), class_ids]
    selected = scores >= CONF_THRESHOLD
    if not np.any(selected):
        return []
    boxes_xywh, scores, class_ids = boxes_xywh[selected], scores[selected], class_ids[selected]
    cx, cy, bw, bh = boxes_xywh.T
    boxes = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), axis=1)

    detections: list[dict[str, Any]] = []
    for index in _nms_xyxy(boxes, scores, IOU_THRESHOLD, MAX_DETECTIONS):
        x1 = min(max((float(boxes[index, 0]) - pad_x) / scale, 0.0), float(width))
        y1 = min(max((float(boxes[index, 1]) - pad_y) / scale, 0.0), float(height))
        x2 = min(max((float(boxes[index, 2]) - pad_x) / scale, 0.0), float(width))
        y2 = min(max((float(boxes[index, 3]) - pad_y) / scale, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(class_ids[index])
        detections.append({
            "class": "hammer" if class_id == 0 else "class_" + str(class_id),
            "class_id": class_id,
            "conf": round(float(scores[index]), 4),
            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    detections.sort(key=lambda item: item["conf"], reverse=True)
    return detections


def _inference_serialized(image: np.ndarray) -> list[dict[str, Any]]:
    # On Render's 0.1 CPU, concurrent ONNX calls increase latency and RAM pressure.
    with _inference_lock:
        return _predict_onnx(image)


def _bootstrap_model() -> None:
    _set_state(stage="loading_model", error_code=None)
    try:
        if _CONFIG_ERROR:
            raise RuntimeError("Не удалось прочитать configs/detect.yaml: " + _CONFIG_ERROR)
        if not MODEL_PATH.is_file():
            raise FileNotFoundError("Файл models/best.onnx отсутствует в контейнере")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = session.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX-модель не содержит входов")
        shape = inputs[0].shape
        if len(shape) != 4 or (isinstance(shape[1], int) and shape[1] != 3):
            raise RuntimeError("Ожидался вход модели NCHW с тремя каналами")
        model_size = shape[2] if isinstance(shape[2], int) else INFER_SIZE
        if isinstance(shape[3], int) and shape[3] != model_size:
            raise RuntimeError("Ожидался квадратный вход модели")

        _set_state(
            session=session, input_name=inputs[0].name, imgsz=int(model_size),
            backend="onnxruntime-cpu", model_loaded=True, stage="warming_up", warmed_up=False,
        )
        _json_log("model_loaded", backend="onnxruntime-cpu", input_size=int(model_size))
        # A true forward pass validates model execution and postprocessing before /ready.
        _predict_onnx(np.zeros((320, 320, 3), dtype=np.uint8))
        _set_state(stage="ready", warmed_up=True, error_code=None)
        _json_log("model_ready", backend="onnxruntime-cpu")
    except Exception as exc:
        _set_state(
            stage="error", model_loaded=False, warmed_up=False, backend=None,
            session=None, input_name=None, error_code=type(exc).__name__,
        )
        _json_log("model_startup_failed", error_type=type(exc).__name__)
        logger.exception("Model initialization failed")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_bootstrap_model, name="model-bootstrap", daemon=True).start()
    yield


app = FastAPI(title="Nazdratenko — Hammer Detector", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def request_logging(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "")
    valid = 1 <= len(incoming) <= 64 and all(ch.isalnum() or ch in "-_." for ch in incoming)
    request_id = incoming if valid else str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        _json_log(
            "http_request", request_id=request_id, method=request.method,
            route=request.url.path, status_code=status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def _not_ready_payload() -> dict[str, Any]:
    state = _snapshot()
    payload: dict[str, Any] = {
        "status": "not_ready", "stage": state["stage"],
        "model_loaded": bool(state["model_loaded"]), "warmed_up": bool(state["warmed_up"]),
        "model_version": MODEL_VERSION,
    }
    if state["stage"] == "error":
        payload["message"] = "Модель не подготовлена. Проверьте лог контейнера и файл best.onnx."
    return payload


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    state = _snapshot()
    return {
        "status": "ok", "service": "nazdratenko-hammer-lr8",
        "stage": state["stage"], "model_loaded": bool(state["model_loaded"]),
        "warmed_up": bool(state["warmed_up"]), "model_version": MODEL_VERSION,
    }


@app.get("/ready")
def ready():
    state = _snapshot()
    if state["stage"] != "ready" or not state["model_loaded"] or not state["warmed_up"]:
        return JSONResponse(status_code=503, content=_not_ready_payload())
    return {
        "status": "ready", "stage": "ready", "model_loaded": True, "warmed_up": True,
        "backend": state["backend"], "model_version": MODEL_VERSION,
    }


@app.get("/info")
def info():
    state = _snapshot()
    return {
        "service": "Nazdratenko Hammer Detector", "variant": 16, "class": "hammer",
        "model": "YOLO26n fine-tuned, ONNX Runtime CPU", "model_version": MODEL_VERSION,
        "stage": state["stage"], "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "max_image_side_px": MAX_IMAGE_SIDE,
        "endpoints": {"GET /health": "liveness", "GET /ready": "model readiness",
                      "POST /detect": "multipart image detection"},
    }


@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):
    request_id = request.state.request_id
    if file.content_type not in ALLOWED_MEDIA_TYPES:
        return JSONResponse(status_code=415, content={
            "error": "Поддерживаются только JPEG, PNG и WebP.", "request_id": request_id,
        })
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={
            "error": "Размер файла превышает лимит 4 MB.", "request_id": request_id,
        })
    if not content:
        return JSONResponse(status_code=400, content={"error": "Файл пустой.", "request_id": request_id})
    encoded = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    del encoded, content
    if image is None:
        return JSONResponse(status_code=400, content={
            "error": "Не удалось декодировать изображение. Выберите корректный файл.",
            "request_id": request_id,
        })

    original_h, original_w = image.shape[:2]
    if original_h < 1 or original_w < 1:
        return JSONResponse(status_code=400, content={
            "error": "Некорректный размер изображения.", "request_id": request_id,
        })
    resized = False
    if max(original_h, original_w) > MAX_IMAGE_SIDE:
        factor = MAX_IMAGE_SIDE / max(original_h, original_w)
        image = cv2.resize(
            image, (max(1, round(original_w * factor)), max(1, round(original_h * factor))),
            interpolation=cv2.INTER_AREA,
        )
        resized = True

    state = _snapshot()
    if state["stage"] != "ready" or not state["warmed_up"]:
        return JSONResponse(status_code=503, content={**_not_ready_payload(), "request_id": request_id})

    started = time.perf_counter()
    try:
        detections = await run_in_threadpool(_inference_serialized, image)
    except Exception as exc:
        _json_log("inference_failed", request_id=request_id, error_type=type(exc).__name__)
        return JSONResponse(status_code=500, content={
            "error": "Ошибка анализа. Повторите запрос и сохраните request_id для диагностики.",
            "request_id": request_id,
        })
    inference_ms = round((time.perf_counter() - started) * 1000, 2)
    inference_h, inference_w = image.shape[:2]
    if resized:
        scale_x, scale_y = original_w / inference_w, original_h / inference_h
        for item in detections:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            item["bbox_xyxy"] = [
                round(x1 * scale_x, 1), round(y1 * scale_y, 1),
                round(x2 * scale_x, 1), round(y2 * scale_y, 1),
            ]

    _json_log("inference_complete", request_id=request_id, inference_ms=inference_ms,
              detection_count=len(detections), image_width=original_w, image_height=original_h)
    return {
        "request_id": request_id, "model_version": MODEL_VERSION,
        "model": "yolo26n-hammer", "backend": state["backend"],
        "image": {"width": original_w, "height": original_h,
                  "inference_width": inference_w, "inference_height": inference_h},
        "conf_threshold": CONF_THRESHOLD, "detections": detections, "count": len(detections),
        "inference_ms": inference_ms, "inference_sec": round(inference_ms / 1000, 4),
    }
