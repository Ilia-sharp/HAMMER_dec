"""
ЛР2, вариант 16 (hammer): инференс обученной модели.
Запуск: python src/detect.py [--source data/test_frames]
Результат: outputs/detections.json + размеченные изображения в outputs/detections/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="каталог с изображениями")
    ap.add_argument("--conf", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "detect.yaml").read_text(encoding="utf-8"))
    source = args.source or str(ROOT / cfg["source_dir"])
    conf = args.conf or float(cfg["conf"])

    from ultralytics import YOLO

    model = YOLO(str(ROOT / cfg["weights"]))
    files = sorted(
        p for p in Path(source).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    results_out = []
    out_dir = ROOT / "outputs" / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        t0 = time.time()
        r = model.predict(str(f), conf=conf, imgsz=int(cfg["imgsz"]),
                          iou=float(cfg["iou"]), max_det=int(cfg["max_det"]), verbose=False)[0]
        dt = time.time() - t0
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = [round(v, 1) for v in b.xyxy[0].tolist()]
            dets.append({
                "class": r.names[int(b.cls)],
                "conf": round(float(b.conf), 3),
                "bbox_xyxy": [x1, y1, x2, y2],
            })
        results_out.append({"image": f.name, "detections": dets, "inference_sec": round(dt, 2)})
        print(f"{f.name}: {len(dets)} детекций ({dt:.2f}s) " +
              ", ".join(f"{d['conf']:.2f}" for d in dets))
        if cfg.get("save_annotated"):
            r.save(filename=str(out_dir / f"pred_{f.stem}.jpg"))

    out_json = ROOT / "outputs" / "detections.json"
    out_json.write_text(json.dumps(results_out, ensure_ascii=False, indent=1), encoding="utf-8")
    n_det = sum(len(r["detections"]) for r in results_out)
    print(f"\nИтого: {n_det} детекций на {len(results_out)} кадрах -> {out_json}")


if __name__ == "__main__":
    main()
