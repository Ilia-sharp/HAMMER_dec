"""QA-коллаж авторазметки: рисуем боксы из data/auto_labels поверх изображений."""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
LBL = PROJECT / "data" / "auto_labels"
OUT = PROJECT / "data" / "qa_collages"
OUT.mkdir(exist_ok=True)
COLS, ROWS, TILE = 6, 4, 260

pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
files = sorted(p for p in RAW.glob(pattern) if p.suffix.lower() in (".jpg", ".png"))

page = 1
for start in range(0, len(files), COLS * ROWS):
    tiles = []
    for f in files[start:start + COLS * ROWS]:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        lb = LBL / (f.stem + ".txt")
        if lb.exists():
            for line in lb.read_text().splitlines():
                _, cx, cy, bw, bh = map(float, line.split())
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), max(2, w // 300))
        s = TILE / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.full((TILE, TILE, 3), 255, np.uint8)
        y = (TILE - img.shape[0]) // 2
        x = (TILE - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.rectangle(canvas, (0, 0), (TILE - 1, TILE - 1), (200, 200, 200), 1)
        n_boxes = len(lb.read_text().splitlines()) if lb.exists() else -1
        cv2.putText(canvas, f"{f.stem.replace('wiki_', '').replace('web_', 'W')}:{n_boxes}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(canvas)
    while len(tiles) % COLS:
        tiles.append(np.full((TILE, TILE, 3), 255, np.uint8))
    rows = [np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)]
    cv2.imwrite(str(OUT / f"qa_ann_{page:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"qa_ann_{page:02d}.jpg")
    page += 1
