import cv2

print("[INFO] Scanning camera indices 0-4...")
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"Index {i}: OK, resolution {w}x{h}")
        else:
            print(f"Index {i}: opened but no frame")
        cap.release()
    else:
        print(f"Index {i}: not available")
