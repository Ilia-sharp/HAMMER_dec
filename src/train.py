"""
ЛР2, вариант 16 (hammer): обучение YOLO26n на кастомном датасете.
Использование: python src/train.py [--epochs N]
Особенность: ultralytics резолвит относительный path от datasets_dir, поэтому
здесь генерируется .data_runtime.yaml с абсолютным путём (см. README, п.6).
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="переопределить число эпох из configs/train.yaml")
    ap.add_argument("--resume", action="store_true", help="продолжить с outputs/train/weights/last.pt")
    args = ap.parse_args()

    last = ROOT / "outputs" / "train" / "weights" / "last.pt"
    if args.resume and last.exists():
        from ultralytics import YOLO  # noqa: E402

        model = YOLO(str(last))
        model.train(resume=True)
        _finalize(model, runtime=None)
        return

    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs

    # генерируем runtime-версию data.yaml с абсолютным путём
    data_cfg = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
    data_cfg["path"] = str(ROOT / "data" / "dataset")
    runtime = ROOT / ".data_runtime.yaml"
    runtime.write_text(yaml.safe_dump(data_cfg, allow_unicode=True), encoding="utf-8")

    from ultralytics import YOLO  # noqa: E402

    model_name = cfg.pop("model", "yolo26n.pt")
    model = YOLO(model_name)

    # на машинах с ограниченной памятью: 1 поток загрузки данных
    cfg.pop("data", None)  # путь к датасету задаём явно (.data_runtime.yaml)
    cfg.setdefault("workers", 0)
    model.train(data=str(runtime), project=str(ROOT / "outputs"), name="train", exist_ok=True, **cfg)
    _finalize(model, runtime=runtime)


def _finalize(model, runtime) -> None:
    """Копирование весов и финальные метрики."""
    import shutil

    ROOT = Path(__file__).resolve().parents[1]

    # копируем лучшие веса в models/best.pt
    best = ROOT / "outputs" / "train" / "weights" / "best.pt"
    if best.exists():
        (ROOT / "models").mkdir(exist_ok=True)
        shutil.copy2(best, ROOT / "models" / "best.pt")
        print(f"\nbest.pt -> models/best.pt ({(ROOT / 'models' / 'best.pt').stat().st_size / 1e6:.1f} MB)")

    if runtime is None:
        return  # при resume финальный val уже выполнен циклом обучения

    # итоговые метрики
    metrics = model.val(data=str(runtime), split="val", verbose=False)
    print("\n=== Итоговые метрики (val) ===")
    print(f"mAP50    = {metrics.box.map50:.3f}")
    print(f"mAP50-95 = {metrics.box.map:.3f}")
    print(f"Precision= {metrics.box.mp:.3f}")
    print(f"Recall   = {metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
