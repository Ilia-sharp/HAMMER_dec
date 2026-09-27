"""
ЛР2, вариант 16 (hammer): REST API + веб-интерфейс детекции молотков
(FastAPI + YOLO26n ONNX).

v1.4.0 — инференс переведён с torch/ultralytics на onnxruntime:
  - models/best.onnx (9 MB): letterbox 448 + декод выхода + NMS средствами
    numpy/cv2 — результаты идентичны ultralytics (сверено на 9 кадрах, IoU>=0.99);
  - torch больше не импортируется: RSS ~150 MB вместо ~450 (лимит 512 MB),
    сборка Render ~2 мин вместо ~10, холодный старт заметно короче;
  - скорость: 448 вместо 640 -> ~25 мс/кадр локально, ~0.5 с на 0.1 CPU;
  - если onnxruntime недоступен — резервный бэкенд ultralytics (best.pt);
  - /detect — синхронный обработчик (threadpool): event loop свободен, /health
    отвечает даже во время инференса (иначе Render перезапускает сервис -> 502);
  - прогрев сессии фиктивным кадром при старте; ограничение запроса 4 MB и
    стороны 1280 px; рамки рисует клиент (canvas) — ноль нагрузки на сервер.

Запуск локально: uvicorn src.api:app --host 0.0.0.0 --port 8000
Эндпоинты:
  GET  /         — веб-интерфейс (static/index.html: drag&drop + bbox)
  GET  /info     — JSON-описание сервиса
  GET  /health   — состояние (используется Render как healthCheckPath)
  POST /detect   — multipart file=изображение -> JSON с детекциями
"""
import gc
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MAX_UPLOAD_MB = 4
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIDE_PX = 1280  # ограничение стороны изображения — экономия RAM/CPU на free-инстансе

_cfg = yaml.safe_load((ROOT / "configs" / "detect.yaml").read_text(encoding="utf-8"))
PT_PATH = ROOT / _cfg["weights"]
ONNX_PATH = ROOT / _cfg.get("onnx_weights", "models/best.onnx")
INFER_IMGSZ = int(_cfg["imgsz"])
INFER_CONF = float(_cfg["conf"])
INFER_IOU = float(_cfg["iou"])
INFER_MAXDET = int(_cfg["max_det"])
CLASS_NAME = "hammer"  # датасет однослассовый (вариант 16)

_state = {"backend": None, "sess": None, "inp": None, "imgsz": INFER_IMGSZ,
          "pt": None, "error": None, "warm": False}


# ── Инициализация бэкенда ────────────────────────────────────────────
def _init_backend():
    """ONNX Runtime — основной бэкенд; ultralytics (torch) — резервный.
    Ошибка не валит сервис: её покажет /health, а /detect вернёт 503."""
    if _state["backend"] is not None:
        return
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1   # 0.1 CPU: многопоточность только вредит
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
        _state["sess"] = sess
        _state["inp"] = sess.get_inputs()[0].name
        shp = sess.get_inputs()[0].shape          # [1, 3, 448, 448]
        _state["imgsz"] = int(shp[2]) if isinstance(shp[2], int) else INFER_IMGSZ
        _state["backend"] = "onnxruntime-cpu"
        return
    except Exception as e:  # noqa: BLE001
        _state["error"] = f"onnxruntime: {e}"
    try:  # резервный путь: ultralytics + PyTorch
        import torch

        torch.set_num_threads(1)
        from ultralytics import YOLO

        _state["pt"] = YOLO(str(PT_PATH))
        _state["backend"] = "ultralytics-torch-cpu"
    except Exception as e:  # noqa: BLE001
        _state["error"] = f"{_state['error']}; ultralytics: {e}"


def _warmup():
    """Прогрев: первый прогон аллоцирует внутренние буферы — делаем это на старте."""
    try:
        detect_objects(np.zeros((320, 320, 3), np.uint8))
        _state["warm"] = True
    except Exception as e:  # noqa: BLE001
        _state["error"] = str(e)


@asynccontextmanager
async def lifespan(_app):
    _init_backend()
    if _state["backend"] == "onnxruntime-cpu":
        _warmup()  # сессия ORT греется за миллисекунды — синхронно
    else:
        threading.Thread(target=_warmup, daemon=True, name="warmup").start()
    yield


# ── Инференс: чистый onnxruntime (letterbox + декод + NMS) ───────────
def _letterbox(img, new):
    h0, w0 = img.shape[:2]
    r = min(new / h0, new / w0)
    nh, nw = round(h0 * r), round(w0 * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) / 2.0, (new - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, dw, dh


def _nms_xyxy(boxes, scores, iou_thr, max_det):
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        a1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a2 = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a1 + a2 - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def _predict_ort(img):
    n = _state["imgsz"]
    h0, w0 = img.shape[:2]
    lb, r, dw, dh = _letterbox(img, n)
    x = cv2.dnn.blobFromImage(lb, 1 / 255.0, (n, n), swapRB=True, crop=False)
    pred = _state["sess"].run(None, {_state["inp"]: x})[0][0]   # (4+nc, anchors)
    cx, cy, w, h, sc = pred[0], pred[1], pred[2], pred[3], pred[4]
    m = sc >= INFER_CONF
    if not m.any():
        return []
    cx, cy, w, h, sc = cx[m], cy[m], w[m], h[m], sc[m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    dets = []
    for i in _nms_xyxy(boxes, sc, INFER_IOU, INFER_MAXDET):
        ux1 = min(max((boxes[i, 0] - dw) / r, 0), w0)
        uy1 = min(max((boxes[i, 1] - dh) / r, 0), h0)
        ux2 = min(max((boxes[i, 2] - dw) / r, 0), w0)
        uy2 = min(max((boxes[i, 3] - dh) / r, 0), h0)
        dets.append({"class": CLASS_NAME, "conf": round(float(sc[i]), 3),
                     "bbox_xyxy": [round(float(ux1), 1), round(float(uy1), 1),
                                   round(float(ux2), 1), round(float(uy2), 1)]})
    dets.sort(key=lambda d: -d["conf"])
    return dets


def _predict_pt(img):
    r = _state["pt"].predict(img, imgsz=INFER_IMGSZ, conf=INFER_CONF,
                             iou=INFER_IOU, max_det=INFER_MAXDET, verbose=False)[0]
    return [{"class": r.names[int(b.cls)], "conf": round(float(b.conf), 3),
             "bbox_xyxy": [round(v, 1) for v in b.xyxy[0].tolist()]} for b in r.boxes]


def detect_objects(img):
    if _state["backend"] == "onnxruntime-cpu":
        return _predict_ort(img)
    if _state["backend"] == "ultralytics-torch-cpu":
        return _predict_pt(img)
    raise RuntimeError(_state["error"] or "backend not initialised")


# ── Приложение ───────────────────────────────────────────────────────
app = FastAPI(title="SII LR02 — hammer detection API", version="1.4.0", lifespan=lifespan)

_STATIC = ROOT / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Веб-интерфейс: загрузка фото, bbox на canvas, статистика."""
    return FileResponse(_STATIC / "index.html")


@app.get("/info")
def info():
    return {
        "service": "sii-lr02-hammer-api",
        "variant": 16,
        "class": "hammer",
        "model": "YOLO26n (fine-tuned, ONNX)",
        "ui": "/ (веб-интерфейс), /docs (Swagger UI)",
        "endpoints": {"POST /detect": "multipart file=image", "GET /health": "статус сервиса"},
        "usage": 'curl -F "file=@photo.jpg" <host>/detect',
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["backend"] is not None else "degraded",
        "model_loaded": _state["backend"] is not None,
        "warmed": _state["warm"],
        "backend": _state["backend"],
        "model_file": ONNX_PATH.name,
        "error": _state["error"],
    }


@app.post("/detect")
def detect(file: UploadFile):
    # Синхронный обработчик: FastAPI выполняет его в threadpool, event loop
    # остаётся свободным — /health отвечает даже во время инференса на 0.1 CPU.
    if file.content_type not in ALLOWED_TYPES:
        return JSONResponse(status_code=415, content={"error": f"unsupported content type {file.content_type}"})
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse(status_code=413, content={"error": f"file > {MAX_UPLOAD_MB} MB"})
    if not data:
        return JSONResponse(status_code=400, content={"error": "empty file"})

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    del data
    if img is None:
        return JSONResponse(status_code=400, content={"error": "cannot decode image"})
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE_PX:
        s = MAX_SIDE_PX / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))

    if _state["backend"] is None:
        return JSONResponse(status_code=503, content={"error": "model unavailable", "detail": _state["error"]})

    t0 = time.perf_counter()
    try:
        dets = detect_objects(img)
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": f"inference failed: {e}"})
    inference_sec = round(time.perf_counter() - t0, 3)
    _inf_w, _inf_h = img.shape[1], img.shape[0]
    del img
    gc.collect()  # вернуть память между запросами — на 512 MB каждый МБ на счету
    return {
        # width/height — исходный кадр; inference_* — после серверного ресайза
        # (bbox в координатах inference_*; UI масштабирует их на клиенте)
        "image": {
            "width": w,
            "height": h,
            "inference_width": _inf_w,
            "inference_height": _inf_h,
        },
        "model": "yolo26n-hammer",
        "backend": _state["backend"],
        "conf_threshold": INFER_CONF,
        "detections": dets,
        "count": len(dets),
        "inference_sec": inference_sec,
    }
