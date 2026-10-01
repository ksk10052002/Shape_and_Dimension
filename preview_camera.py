import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
print(f"[INFO] FOURCC negotiated: {fourcc_str}")
print("[INFO] Warming up (first frame can take ~1s)...")

frame_count = 0
print("[INFO] Window should appear now. Press 'q' in it to close.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] frame read failed")
        break
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"[INFO] {frame_count} frames read so far, still running")
    cv2.imshow("Brio Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Closed. Total frames: {frame_count}")
