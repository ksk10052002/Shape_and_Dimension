import os
import sys
import glob
from collections import Counter

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

DEFAULT_WORKSPACE = "kamal-kumar-8dloj"
IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_images(img_dir):
    if not os.path.isdir(img_dir):
        return []
    return [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]


def source_ids(img_dir):
    # Roboflow names files like IMG_3492_jpg.rf.<hash>.jpg; the part before
    # ".rf." identifies the original photo.
    return {name.split(".rf.")[0] for name in list_images(img_dir)}


def inspect_labels(label_dir):
    box_lines = 0
    poly_lines = 0
    empty_files = 0
    class_counts = Counter()
    for path in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(path, "r") as f:
            lines = [ln.split() for ln in f if ln.strip()]
        if not lines:
            empty_files += 1
        for parts in lines:
            class_counts[int(parts[0])] += 1
            if len(parts) == 5:
                box_lines += 1
            else:
                poly_lines += 1
    return box_lines, poly_lines, empty_files, class_counts


def main():
    if len(sys.argv) < 3:
        print("Usage: python download_dataset.py <project_slug> <version> [workspace]")
        sys.exit(1)

    project_slug = sys.argv[1]
    version_num = int(sys.argv[2])
    workspace = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_WORKSPACE

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY missing in .env")
        sys.exit(1)

    print("[INFO] Downloading from Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    dataset = project.version(version_num).download("yolov8")
    root = os.path.abspath(dataset.location)
    print(f"[INFO] Downloaded to: {root}")

    # ---- Rewrite data.yaml with absolute paths ----
    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    names = cfg["names"]
    new_cfg = {"names": names, "nc": cfg.get("nc", len(names))}
    for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        img_dir = os.path.join(root, folder, "images")
        if os.path.isdir(img_dir):
            new_cfg[key] = img_dir.replace(os.sep, "/")
        else:
            print(f"[WARN] Missing split folder: {folder}")

    with open(yaml_path, "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)
    print(f"[OK] Rewrote {yaml_path} with absolute paths")

    # ---- Split sizes ----
    print("\n--- Image counts ---")
    for folder in ("train", "valid", "test"):
        n = len(list_images(os.path.join(root, folder, "images")))
        print(f"{folder:6s}: {n}")

    # ---- Label format and class balance ----
    print("\n--- Label check (train) ---")
    box, poly, empty, counts = inspect_labels(os.path.join(root, "train", "labels"))
    print(f"box lines: {box} | polygon lines: {poly} | images with no labels: {empty}")
    if poly > 0 and box == 0:
        print("[RESULT] POLYGON labels -> use segmentation model: yolov8n-seg.pt")
    elif box > 0 and poly == 0:
        print("[RESULT] BOX labels -> use detection model: yolov8n.pt")
    else:
        print("[WARN] Mixed box and polygon labels. Fix the dataset before training.")

    print("\nInstances per class (train):")
    name_list = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    for cid, cnt in sorted(counts.items()):
        label = name_list[cid] if cid < len(name_list) else str(cid)
        print(f"  {label}: {cnt}")

    # ---- Leakage check ----
    train_ids = source_ids(os.path.join(root, "train", "images"))
    val_ids = source_ids(os.path.join(root, "valid", "images"))
    overlap = train_ids & val_ids
    print("\n--- Leakage check ---")
    print(f"unique source photos: train={len(train_ids)}, valid={len(val_ids)}")
    if overlap:
        print(f"[WARN] {len(overlap)} source photos appear in BOTH train and valid. Scores will be inflated.")
    else:
        print("[OK] No source photo is shared between train and valid.")

    print("\n[DONE] Dataset folder:", root)


if __name__ == "__main__":
    main()
