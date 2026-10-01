import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"[INFO] isOpened: {cap.isOpened()}")
print("[INFO] Attempting single frame read with timing...")

start = time.time()
ret, frame = cap.read()
elapsed = time.time() - start

print(f"[INFO] read() returned after {elapsed:.2f}s, ret={ret}")
if ret:
    print(f"[INFO] Frame shape: {frame.shape}")
cap.release()
