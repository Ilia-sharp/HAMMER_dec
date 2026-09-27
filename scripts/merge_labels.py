"""Пост-обработка меток: слияние частичных боксов (голова+рукоять) в один полный бокс.
Правило: если пересечение > 40% меньшего бокса -> объединяем в union. Повторяем до стабилизации."""
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
LBL = PROJECT / "data" / "auto_labels"
MERGE_THR = 0.40

for lb in sorted(LBL.glob("*.txt")):
    lines = [l for l in lb.read_text().splitlines() if l.strip()]
    if not lines:
        continue
    boxes = []
    for l in lines:
        _, cx, cy, bw, bh = map(float, l.split())
        boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                min_area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if min_area > 0 and inter / min_area > MERGE_THR:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    out = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        out.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    lb.write_text("\n".join(out) + "\n")
    if len(out) != len(lines):
        print(f"{lb.stem}: {len(lines)} -> {len(out)}")
print("merge done")
