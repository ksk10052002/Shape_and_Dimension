import os
import sys
import glob

import torch
import yaml
from ultralytics import YOLO

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 20


def find_dataset():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    candidates = glob.glob("*/data.yaml")
    if not candidates:
        print("[ERROR] No dataset folder with data.yaml found in this directory.")
        sys.exit(1)
    newest = max(candidates, key=os.path.getmtime)
    return os.path.abspath(os.path.dirname(newest))


def detect_task(dataset_dir):
    box_lines = 0
    poly_lines = 0
    label_dir = os.path.join(dataset_dir, "train", "labels")
    for path in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(path, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) == 5:
                    box_lines += 1
                else:
                    poly_lines += 1
    if box_lines and poly_lines:
        print("[ERROR] Mixed box and polygon labels. Fix the dataset in Roboflow first.")
        sys.exit(1)
    if poly_lines:
        return "segment"
    if box_lines:
        return "detect"
    print("[ERROR] No labels found in train/labels.")
    sys.exit(1)


def main():
    dataset_dir = find_dataset()
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found in {dataset_dir}")
        sys.exit(1)

    task = detect_task(dataset_dir)
    weights = "yolov8n-seg.pt" if task == "segment" else "yolov8n.pt"
    run_name = os.path.basename(dataset_dir.rstrip("/\\")).lower() + "_v1"
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Dataset : {dataset_dir}")
    print(f"[INFO] Task    : {task} -> starting from {weights}")
    print(f"[INFO] Device  : {'GPU (cuda:0)' if device == 0 else 'CPU'}")
    print(f"[INFO] Run name: {run_name}")

    model = YOLO(weights)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project="runs/train",
        name=run_name,
        exist_ok=True,
        plots=True,
    )

    best = os.path.join("runs", "train", run_name, "weights", "best.pt")
    print(f"\n[INFO] Training complete. Best weights: {best}")

    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    if "test" in cfg:
        print("[INFO] Evaluating on the held-out test split...")
        YOLO(best).val(
            data=data_yaml,
            split="test",
            device=device,
            project="runs/val",
            name=run_name + "_test",
            exist_ok=True,
        )
    else:
        print("[WARN] No test split in data.yaml, skipping test evaluation.")


if __name__ == "__main__":
    main()
