import os
import yaml
from pathlib import Path
from ultralytics import YOLO

DATASET_ROOT = r"\my_obb_dataset"
MODEL_START = "yolov8n-obb.pt"
EPOCHS = 250
IMGSZ = 640
BATCH = 16

CLASS_NAMES = [
    "puljersey"
]

def create_data_yaml():
    yaml_path = Path(DATASET_ROOT) / "data.yaml"
    data = {
        "path": str(Path(DATASET_ROOT).resolve().as_posix()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images" if (Path(DATASET_ROOT)/"test"/"images").exists() else None,
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    if data["test"] is None:
        del data["test"]
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"Created data.yaml → {yaml_path}")
    return str(yaml_path)

def main():
    data_yaml = create_data_yaml()

    required_folders = [
        Path(DATASET_ROOT)/"train"/"images",
        Path(DATASET_ROOT)/"train"/"labels",
        Path(DATASET_ROOT)/"valid"/"images",
        Path(DATASET_ROOT)/"valid"/"labels",
    ]


    print(f"Classes: {CLASS_NAMES}")
    print(f"Using model: {MODEL_START}")
    print(f"Epochs: {EPOCHS}   Image size: {IMGSZ}   Batch: {BATCH}\n")
    model = YOLO(MODEL_START)
    results = model.train(
        data      = data_yaml,
        epochs    = EPOCHS,
        imgsz     = IMGSZ,
        batch     = BATCH,
        workers   = 8,
        cache     = True,
        patience  = 50,
        cos_lr    = True,
        amp       = True,
        optimizer = "auto",
        lr0       = 0.01,
        device    = 0,
        project   = "runs/obb_training",
        name      = "puljersey_obb",
        exist_ok  = True,
        degrees   = 12.0,
        mosaic    = 1.0,
        mixup     = 0.15,
        copy_paste= 0.1,
    )
    print("\n" + "="*60)
    print("Training finished!")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(f"Last weights:  {results.save_dir}/weights/last.pt")
    print(f"Training results saved: {results.save_dir}")
    print("="*60 + "\n")

