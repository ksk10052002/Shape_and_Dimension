import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import json
import time
import math
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/train/pla_shapes_v1/weights/best.pt"
CALIBRATION_FILE = "calibration_pc.json"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CONF_THRESHOLD = 0.35
CROP_MARGIN = 10


def load_calibration():
    with open(CALIBRATION_FILE, "r") as f:
        data = json.load(f)
    return data["mm_per_px"]


def extract_shape_contour(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    x1c = max(0, x1 - CROP_MARGIN)
    y1c = max(0, y1 - CROP_MARGIN)
    x2c = min(w, x2 + CROP_MARGIN)
    y2c = min(h, y2 + CROP_MARGIN)

    crop = frame[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return None, (x1c, y1c)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, (x1c, y1c)

    largest = max(contours, key=cv2.contourArea)
    return largest, (x1c, y1c)


def classify_and_measure(contour, mm_per_px):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return "unknown", ""

    circularity = 4 * math.pi * area / (perimeter ** 2)

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = len(approx)

    if circularity > 0.85:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        diameter_px = radius * 2
        diameter_mm = diameter_px * mm_per_px
        return "circle", f"D: {diameter_mm:.1f}mm"

    elif corners == 4:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        w_mm = w * mm_per_px
        h_mm = h * mm_per_px
        return "square", f"W: {w_mm:.1f}mm H: {h_mm:.1f}mm"

    elif corners >= 5:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        w_mm = w * mm_per_px
        h_mm = h * mm_per_px
        return "hexagon", f"W: {w_mm:.1f}mm H: {h_mm:.1f}mm ({corners} corners)"

    else:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        w_mm = w * mm_per_px
        h_mm = h * mm_per_px
        return f"{corners}-sided", f"W: {w_mm:.1f}mm H: {h_mm:.1f}mm"


def main():
    mm_per_px = load_calibration()
    print(f"[INFO] Loaded calibration: {mm_per_px:.4f} mm/px")

    model = YOLO(MODEL_PATH)
    print(f"[INFO] Classes: {model.names}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    print("[INFO] Starting. Press 'q' to quit.")
    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.track(frame, conf=CONF_THRESHOLD, persist=True,
                               tracker="bytetrack.yaml", verbose=False)

        annotated = frame.copy()

        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_class = model.names[int(box.cls[0])]
                conf = float(box.conf[0])

                contour, (ox, oy) = extract_shape_contour(frame, (x1, y1, x2, y2))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

                if contour is not None:
                    shifted_contour = contour + np.array([ox, oy])
                    cv2.drawContours(annotated, [shifted_contour], -1, (0, 200, 255), 2)

                    geo_label, dims_text = classify_and_measure(contour, mm_per_px)

                    label = f"{yolo_class} ({geo_label}) {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(annotated, dims_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
                else:
                    cv2.putText(annotated, f"{yolo_class} {conf:.2f} (no contour)",
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        now = time.time()
        instant_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1
        cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Shape Detector - Dimensions", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
