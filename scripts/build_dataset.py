"""Сборка YOLO-датасета: перемешивание, сплит 44/11 (80/20), копирование изображений и меток,
отдельная папка test_frames (без меток). Генерирует data/dataset/dataset_stats.json."""
import json
import random
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
LBL = PROJECT / "data" / "auto_labels"
DS = PROJECT / "data" / "dataset"
TEST = PROJECT / "data" / "test_frames"

random.seed(16)  # вариант 16 — воспроизводимый сплит

files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in (".jpg", ".png"))
random.shuffle(files)
n_val = 11
val_files = set(files[:n_val])
train_files = [f for f in files if f not in val_files]

for split in ("train", "val", "test"):
    (DS / "images" / split).mkdir(parents=True, exist_ok=True)
    (DS / "labels" / split).mkdir(parents=True, exist_ok=True)

for f in train_files:
    shutil.copy2(f, DS / "images" / "train" / f.name)
    shutil.copy2(LBL / (f.stem + ".txt"), DS / "labels" / "train" / (f.stem + ".txt"))
for f in val_files:
    shutil.copy2(f, DS / "images" / "val" / f.name)
    shutil.copy2(LBL / (f.stem + ".txt"), DS / "labels" / "val" / (f.stem + ".txt"))
for f in sorted(TEST.glob("*")):
    if f.suffix.lower() in (".jpg", ".png"):
        shutil.copy2(f, DS / "images" / "test" / f.name)

stats = {
    "train": len(train_files),
    "val": n_val,
    "test": len(list((DS / "images" / "test").iterdir())),
    "val_files": sorted(f.name for f in val_files),
}
(DS / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps({k: v for k, v in stats.items() if k != "val_files"}))
print("val:", ", ".join(sorted(f.name for f in val_files)))
