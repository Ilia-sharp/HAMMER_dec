"""Генерация cvat_hammer_yolo11.zip (формат CVAT for images 1.1) из YOLO-меток датасета."""
import datetime
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "dataset"
OUT_ZIP = ROOT / "cvat_hammer_yolo11.zip"

pairs = []
for split in ("train", "val"):
    for img in sorted((DS / "images" / split).iterdir()):
        lb = DS / "labels" / split / (img.stem + ".txt")
        pairs.append((img, lb))

root = ET.Element("annotations")
ET.SubElement(root, "version").text = "1.1"
meta = ET.SubElement(root, "meta")
task = ET.SubElement(meta, "task")
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.000000+00:00")
for tag, val in [("id", "1"), ("name", "hammer_yolo11"), ("size", str(len(pairs))),
                 ("mode", "annotation"), ("overlap", "0"), ("bugtracker", ""),
                 ("created", now), ("updated", now), ("subset", "default"),
                 ("start_frame", "0"), ("stop_frame", str(len(pairs) - 1)), ("frame_filter", "")]:
    ET.SubElement(task, tag).text = val
segs = ET.SubElement(task, "segments")
seg = ET.SubElement(segs, "segment")
ET.SubElement(seg, "id").text = "1"
ET.SubElement(seg, "start_frame").text = "0"
ET.SubElement(seg, "stop_frame").text = str(len(pairs) - 1)
owner = ET.SubElement(task, "owner")
ET.SubElement(owner, "username").text = "ivanov"
ET.SubElement(owner, "email").text = ""
labels = ET.SubElement(task, "labels")
label = ET.SubElement(labels, "label")
ET.SubElement(label, "name").text = "hammer"
ET.SubElement(label, "color").text = "#ff4d00"
ET.SubElement(label, "type").text = "rectangle"
ET.SubElement(label, "attributes").text = ""
ET.SubElement(task, "dumped_checksums").text = ""
ET.SubElement(meta, "dump_format").text = "CVAT for images 1.1"

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for i, (img_path, lb_path) in enumerate(pairs):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        img_el = ET.SubElement(root, "image", id=str(i), name=img_path.name,
                               width=str(w), height=str(h),
                               subtype="train" if "train" in str(img_path) else "val")
        if lb_path.exists():
            for line in lb_path.read_text().splitlines():
                _, cx, cy, bw, bh = map(float, line.split())
                box = ET.SubElement(img_el, "box", label="hammer", source="manual",
                                    xc=f"{cx * w:.1f}", yc=f"{cy * h:.1f}",
                                    w=f"{bw * w:.1f}", h=f"{bh * h:.1f}", z_order="0")
                ET.SubElement(box, "attributes").text = ""
        z.write(img_path, img_path.name)
    xml_bytes = ET.tostring(root, encoding="unicode")
    z.writestr("annotations.xml", '<?xml version="1.0" encoding="utf-8"?>\n' + xml_bytes)

print(f"{OUT_ZIP.name}: {OUT_ZIP.stat().st_size / 1e6:.1f} MB, {len(pairs)} images")
