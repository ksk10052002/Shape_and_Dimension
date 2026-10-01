import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import time
from ultralytics import YOLO

MODEL_PATH = "runs/train/pla_shapes_v1/weights/best.pt"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CONF_THRESHOLD = 0.35
WINDOW_NAME = "Shape Detector - PLA Shapes"


def main():
    print(f"[INFO] Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Classes: {model.names}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting live inference with tracking. Press 'q' to quit.")

    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        t1 = time.time()

        results = model.track(frame, conf=CONF_THRESHOLD, persist=True,
                               tracker="bytetrack.yaml", verbose=False)
        annotated = results[0].plot()
        t2 = time.time()

        now = time.time()
        instant_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1

        cv2.putText(annotated, f"FPS: {fps_smooth:.1f} | capture: {(t1-t0)*1000:.0f}ms | inference: {(t2-t1)*1000:.0f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
