import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import json
import math
import sys

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CALIBRATION_FILE = "calibration_pc.json"


class ClickCollector:
    def __init__(self):
        self.points = []

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))
            print(f"[INFO] Point {len(self.points)}: ({x}, {y})")

    def reset(self):
        self.points = []


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("[INFO] Place a ruler flat where your shapes normally sit.")
    print("[INFO] Click 2 points a known distance apart (e.g. 0cm and 10cm marks).")
    print("[INFO] Press 'r' to reset, 'q' to quit without saving.")

    collector = ClickCollector()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", collector.callback)

    mm_per_px = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        for i, pt in enumerate(collector.points):
            cv2.circle(display, pt, 6, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(collector.points) == 2:
            cv2.line(display, collector.points[0], collector.points[1], (0, 255, 0), 2)

        cv2.putText(display, "Click 2 points on known-length object | 'r' reset | 'q' quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            collector.reset()
        if key == ord('q'):
            break

        if len(collector.points) == 2 and mm_per_px is None:
            (x1, y1), (x2, y2) = collector.points
            pixel_dist = math.hypot(x2 - x1, y2 - y1)
            try:
                known_mm = float(input(f"\n[INPUT] Pixel distance: {pixel_dist:.2f}px. "
                                        f"Enter real-world distance in mm: "))
            except ValueError:
                print("[WARN] Invalid number, resetting.")
                collector.reset()
                continue
            mm_per_px = known_mm / pixel_dist
            print(f"\n[RESULT] mm_per_px = {mm_per_px:.4f}")
            break

    cap.release()
    cv2.destroyAllWindows()

    if mm_per_px is None:
        print("[INFO] No calibration performed.")
        return

    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"mm_per_px": mm_per_px, "frame_width": FRAME_WIDTH, "frame_height": FRAME_HEIGHT}, f, indent=2)
    print(f"[INFO] Saved to {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
