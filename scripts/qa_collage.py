"""QA-коллажи: сетки изображений с подписями для ручного отбора."""
import sys
from pathlib import Path

import cv2
import numpy as np

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
OUT = RAW.parent / "qa_collages"
OUT.mkdir(exist_ok=True)
COLS, ROWS, TILE = 6, 4, 220  # 24 изображения на лист


def make_collage(files, page):
    tiles = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        s = TILE / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        canvas = np.full((TILE, TILE, 3), 255, np.uint8)
        y = (TILE - img.shape[0]) // 2
        x = (TILE - img.shape[1]) // 2
        canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
        cv2.rectangle(canvas, (0, 0), (TILE - 1, TILE - 1), (200, 200, 200), 1)
        label = f.stem.replace("wiki_", "")
        cv2.putText(canvas, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        tiles.append(canvas)
    while len(tiles) % COLS:
        tiles.append(np.full((TILE, TILE, 3), 255, np.uint8))
    rows = []
    for r in range(0, len(tiles), COLS):
        rows.append(np.hstack(tiles[r:r + COLS]))
    grid = np.vstack(rows)
    cv2.imwrite(str(OUT / f"qa_{page:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return len(tiles)


files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".jpg", ".png"))
print(f"{len(files)} files, collage page size {COLS * ROWS}")
page = 1
for i in range(0, len(files), COLS * ROWS):
    k = make_collage(files[i:i + COLS * ROWS], page)
    print(f"qa_{page:02d}.jpg: {k} tiles")
    page += 1
