"""
ЛР2, вариант 16 (hammer): авторазметка через YOLOE visual prompting, версия 2.
Стратегия: 3 опорных кадра размечены вручную -> 3 отдельных PE -> 3 прохода по всем кадрам
-> объединение детекций через NMS -> YOLO-метки.
Текстовые промпты не работают для 'hammer' (0 детекций), поэтому используется визуальный промпт.
Запуск: python scripts/annotate_yoloe.py
"""
import os

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.expanduser("~/.ultralytics"))
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "data" / "auto_labels"
OUT.mkdir(exist_ok=True)
CONF = 0.08          # низкий порог, фильтрация через NMS
IMGSZ = 960
NMS_IOU = 0.45
MIN_AREA_FRAC = 0.01  # боксы < 1% площади кадра отбрасываем

# Опорные кадры, размеченные вручную (pixel XYXY, класс 0 = hammer)
REFS = {
    "wiki_002.jpg": [[195.0, 45.0, 730.0, 1390.0]],
    "wiki_037.jpg": [[145.0, 85.0, 715.0, 1545.0]],
    "web_004.jpg": [[45.0, 40.0, 1035.0, 870.0]],
}


def main():
    model = YOLOE("yoloe-11s-seg.pt")
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    all_dets = {f.name: [] for f in files}

    # 1) PE по каждому опорному кадру
    pes = []
    for name, boxes in REFS.items():
        prompts = dict(bboxes=np.array(boxes, dtype=np.float32), cls=np.array([0]))
        predictor = YOLOEVPDetectPredictor(overrides=dict(imgsz=IMGSZ, mode="predict", conf=0.01))
        predictor.setup_model(model=model.model)
        predictor.set_prompts(prompts)
        pe = torch.as_tensor(predictor.get_vpe(source=str(RAW / name))).float().cpu()
        pes.append(pe[0])  # (1, dim)
        print(f"PE {name}: shape={tuple(pe.shape)}")

    # 2) усреднённый PE -> один проход по всем кадрам
    pe_avg = torch.stack(pes).mean(dim=0, keepdim=True)  # (1, 1, dim)
    print("PE averaged:", tuple(pe_avg.shape))
    model.set_classes(["hammer"], pe_avg)
    model.predictor = None
    model.save_prompt_embeddings(PROJECT / "data" / "hammer_pe.npz")
    for f in files:
        r = model.predict(str(f), conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        h, w = r.orig_shape
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            all_dets[f.name].append([x1, y1, x2, y2, float(b.conf)])
    print("pass done")

    # 3) NMS-объединение и сохранение меток
    stats = []
    for f in files:
        dets = all_dets[f.name]
        h, w = cv2_size(RAW / f.name)
        stats_line = []
        if dets:
            boxes = torch.tensor([d[:4] for d in dets])
            scores = torch.tensor([d[4] for d in dets])
            keep = torchvision.ops.nms(boxes, scores, NMS_IOU)
            lines = []
            for k in keep:
                x1, y1, x2, y2 = boxes[k].tolist()
                if (x2 - x1) * (y2 - y1) < MIN_AREA_FRAC * w * h:
                    continue
                lines.append(
                    f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                )
                stats_line.append(round(float(scores[k]), 2))
            (OUT / (f.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        else:
            (OUT / (f.stem + ".txt")).write_text("")
        stats.append((f.name, len(stats_line), stats_line))

    zero = [s for s in stats if s[1] == 0]
    print(f"\nAnnotated: {len(files)} frames, empty labels: {len(zero)}")
    if zero:
        print("  zero:", [z[0] for z in zero])
    print("Mean boxes/frame:", round(np.mean([s[1] for s in stats]), 2))
    with open(OUT / "_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)


def cv2_size(p: Path):
    import cv2
    img = cv2.imread(str(p))
    return img.shape[0], img.shape[1]


if __name__ == "__main__":
    main()
