"""
ЛР2, вариант 16: сравнение предобученной YOLO26n (COCO) и дообученной (hammer).
Результат: outputs/baseline_vs_finetuned/ — изображения «до/после» + comparison.json.
Запуск: python src/compare_baseline.py
"""
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test_frames"
OUT = ROOT / "outputs" / "baseline_vs_finetuned"
OUT.mkdir(parents=True, exist_ok=True)
CONF = 0.25


def draw(img, boxes, label_fn, color):
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label_fn(b), (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    from ultralytics import YOLO

    base = YOLO("yolo26n.pt")          # COCO, 80 классов (без hammer)
    fine = YOLO(str(ROOT / "models" / "best.pt"))  # дообученная, 1 класс
    files = sorted(p for p in TEST.iterdir() if p.suffix.lower() in (".jpg", ".png"))

    report = []
    for f in files:
        img = cv2.imread(str(f))
        rb = base.predict(str(f), conf=CONF, imgsz=640, verbose=False)[0]
        rf = fine.predict(str(f), conf=CONF, imgsz=640, verbose=False)[0]

        base_classes = sorted({rb.names[int(b.cls)] for b in rb.boxes})
        hammer_ok = any(int(b.cls) == 0 for b in rf.boxes)

        img_b = draw(img.copy(), rb.boxes, lambda b: f"{rb.names[int(b.cls)]} {float(b.conf):.2f}", (0, 0, 255))
        img_f = draw(img.copy(), rf.boxes, lambda b: f"hammer {float(b.conf):.2f}", (0, 220, 0))

        h = max(img_b.shape[0], img_f.shape[0])
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        pair = cv2.hconcat([pad(img_b), pad(img_f)])
        # подписи сверху
        top = np_full(36, pair.shape[1], 30)
        cv2.putText(top, f"BASELINE COCO: {base_classes or '-'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
        w2 = pair.shape[1] // 2
        cv2.putText(top, "FINETUNED (hammer)", (w2 + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.imwrite(str(OUT / f"compare_{f.stem}.jpg"), cv2.vconcat([top, pair]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        report.append({
            "image": f.name,
            "baseline_detections": [
                {"class": rb.names[int(b.cls)], "conf": round(float(b.conf), 3)} for b in rb.boxes
            ],
            "finetuned_detections": [
                {"class": "hammer", "conf": round(float(b.conf), 3)} for b in rf.boxes
            ],
            "hammer_found": hammer_ok,
        })
        print(f"{f.name}: baseline={base_classes or '-'} | finetuned hammer={hammer_ok}")

    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    n_ok = sum(r["hammer_found"] for r in report)
    print(f"\nhammer найден на {n_ok}/{len(report)} тестовых кадрах -> {OUT}")


def np_full(h, w, v=0):
    import numpy as np
    return np.full((h, w, 3), v, np.uint8)


if __name__ == "__main__":
    main()
