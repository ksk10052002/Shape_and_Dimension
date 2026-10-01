import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import time

OUTPUT_DIR = "dataset_raw"
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    existing = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
    count = len(existing)

    print(f"[INFO] Starting at image count: {count}")
    print("[INFO] Press 'c' to capture a photo, 'q' to quit.")
    print("[INFO] Vary: position, rotation, single shapes, overlapping pairs, lighting angle.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, f"Captured: {count} | 'c' capture | 'q' quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Dataset Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            filename = os.path.join(OUTPUT_DIR, f"shape_{count:04d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"[SAVED] {filename}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Total images: {count}")


if __name__ == "__main__":
    main()
