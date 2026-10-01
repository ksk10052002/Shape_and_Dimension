from ultralytics import YOLO
import torch

DATA_YAML = "Pla_Objects-2/data.yaml"
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 20


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Training on device: {'GPU (cuda:0)' if device == 0 else 'CPU'}")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project="runs/train",
        name="pla_shapes_v1",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\n[INFO] Training complete.")
    print(f"[INFO] Best weights saved to: runs/train/pla_shapes_v1/weights/best.pt")


if __name__ == "__main__":
    main()
