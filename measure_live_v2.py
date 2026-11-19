import os
import sys
import glob
import json
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

CALIBRATION_FILE = "calibration_pc.json"
CATALOG_FILE = "shape_catalog.json"
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CONF = 0.5
HISTORY = 10

ROUND_NAMES = {"circle", "disc", "ring", "cylinder"}
SIZE_MATCH_NAMES = {"circle", "disc", "cylinder"}
QUAD_NAMES = {"square", "rectangle"}

DEFAULT_CATALOG = {
    "square_max_aspect": 1.12,
    "size_tolerance_mm": 2.0,
    "round_sizes_mm": {"circle": None, "disc": None, "cylinder": None},
}


def normalize(name):
    n = name.lower().strip()
    return "cylinder" if n == "clyinder" else n


def newest_weights():
    paths = glob.glob(os.path.join("runs", "train", "*", "weights", "best.pt"))
    if not paths:
        print("[ERROR] No best.pt found under runs/train")
        sys.exit(1)
    return max(paths, key=os.path.getmtime)


def load_scale(actual_w, actual_h):
    if not os.path.exists(CALIBRATION_FILE):
        print(f"[ERROR] {CALIBRATION_FILE} not found. Run calibrate_pc.py first.")
        sys.exit(1)
    with open(CALIBRATION_FILE, "r") as f:
        cal = json.load(f)
    mm_per_px = cal["mm_per_px"]
    cal_w = cal.get("frame_width", actual_w)
    cal_h = cal.get("frame_height", actual_h)
    if (cal_w, cal_h) != (actual_w, actual_h):
        same_aspect = abs(cal_w / cal_h - actual_w / actual_h) < 0.01
        if same_aspect:
            mm_per_px *= cal_w / actual_w
            print(f"[WARN] Camera is {actual_w}x{actual_h}, calibration was {cal_w}x{cal_h}. Rescaled to {mm_per_px:.4f} mm/px.")
        else:
            print(f"[WARN] Camera is {actual_w}x{actual_h}, calibration was {cal_w}x{cal_h} (different shape). Sizes will be WRONG until you recalibrate.")
    return mm_per_px


def load_catalog():
    if not os.path.exists(CATALOG_FILE):
        with open(CATALOG_FILE, "w") as f:
            json.dump(DEFAULT_CATALOG, f, indent=2)
        print(f"[INFO] Created {CATALOG_FILE}. Fill in the round shape sizes (mm), then press r in the window.")
    with open(CATALOG_FILE, "r") as f:
        cat = json.load(f)
    for key, value in DEFAULT_CATALOG.items():
        cat.setdefault(key, value)
    filled = {k: v for k, v in cat["round_sizes_mm"].items() if v is not None}
    names = sorted(filled)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            gap = abs(filled[names[i]] - filled[names[j]])
            if gap <= 2 * cat["size_tolerance_mm"]:
                print(f"[WARN] {names[i]} and {names[j]} sizes are only {gap:.1f} mm apart. They cannot be told apart reliably by size.")
    return cat


def measure(poly, name, mm_per_px):
    pts = poly.astype(np.float32).reshape(-1, 1, 2)
    if name in ROUND_NAMES:
        (cx, cy), radius = cv2.minEnclosingCircle(pts)
        return "circle", (2 * radius * mm_per_px,), ((cx, cy), radius)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    return "rect", (max(w, h) * mm_per_px, min(w, h) * mm_per_px), rect


def decide_label(name, kind, vals, catalog):
    if name in QUAD_NAMES and kind == "rect":
        aspect = vals[0] / max(vals[1], 1e-6)
        final = "square" if aspect <= catalog["square_max_aspect"] else "rectangle"
        tag = "" if final == name else " [by aspect]"
        return final, tag, aspect
    if name in SIZE_MATCH_NAMES and kind == "circle":
        d = vals[0]
        refs = {k: v for k, v in catalog["round_sizes_mm"].items() if v is not None}
        if refs:
            best = min(refs, key=lambda k: abs(d - refs[k]))
            if abs(d - refs[best]) <= catalog["size_tolerance_mm"]:
                tag = "" if best == name else " [by size]"
                return best, tag, None
            return name, " [size?]", None
    return name, "", None


def format_size(kind, vals):
    if kind == "circle":
        return f"D {vals[0]:.1f} mm"
    return f"{vals[0]:.1f} x {vals[1]:.1f} mm"


def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, 0.55, 1)
    x = int(x)
    y = int(max(y, th + 8))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 4), font, 0.55, (0, 255, 0), 1)


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else newest_weights()
    print(f"[INFO] Using weights: {weights}")
    model = YOLO(weights)
    catalog = load_catalog()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    frame = None
    for _ in range(60):
        ok, frame = cap.read()
        if ok:
            break
    if frame is None:
        print("[ERROR] Camera gave no frames.")
        sys.exit(1)

    actual_h, actual_w = frame.shape[:2]
    mm_per_px = load_scale(actual_w, actual_h)
    print(f"[INFO] Frame {actual_w}x{actual_h}, scale {mm_per_px:.4f} mm/px. q = quit, r = reload {CATALOG_FILE}")

    history = {}
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        r = model.track(frame, conf=CONF, agnostic_nms=True, persist=True,
                        tracker="bytetrack.yaml", verbose=False)[0]
        out = frame.copy()
        counts = {}

        if r.masks is not None and r.boxes is not None and len(r.boxes) > 0:
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            classes = r.boxes.cls.int().tolist()
            for poly, cid, tid in zip(r.masks.xy, classes, ids):
                if len(poly) < 3:
                    continue
                name = normalize(model.names[cid])
                kind, vals, geom = measure(poly, name, mm_per_px)

                if tid is not None:
                    hist = history.setdefault(tid, deque(maxlen=HISTORY))
                    hist.append((kind, vals))
                    same = [v for k, v in hist if k == kind]
                    vals = tuple(np.median(np.array(same), axis=0))

                final, tag, aspect = decide_label(name, kind, vals, catalog)
                counts[final] = counts.get(final, 0) + 1

                cv2.polylines(out, [poly.astype(np.int32)], True, (0, 255, 0), 2)
                if kind == "circle":
                    (cx, cy), radius = geom
                    cv2.circle(out, (int(cx), int(cy)), int(radius), (0, 200, 255), 1)
                else:
                    box = cv2.boxPoints(geom).astype(np.int32)
                    cv2.polylines(out, [box], True, (0, 200, 255), 1)

                text = f"{final} {format_size(kind, vals)}"
                if aspect is not None:
                    text += f" ratio {aspect:.2f}"
                text += tag
                draw_label(out, text, poly[:, 0].min(), poly[:, 1].min() - 2)

        now = time.time()
        fps = fps * 0.9 + (1.0 / max(now - prev, 1e-6)) * 0.1
        prev = now
        summary = f"Total: {sum(counts.values())}  " + "  ".join(f"{k}: {v}" for k, v in counts.items())
        cv2.putText(out, summary, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Shape measurement", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            catalog = load_catalog()
            print("[INFO] Catalog reloaded.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
