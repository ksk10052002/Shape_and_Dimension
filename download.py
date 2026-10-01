import os
import sys
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    print("[ERROR] ROBOFLOW_API_KEY not found. Make sure .env exists with ROBOFLOW_API_KEY=your_key_here in it.")
    sys.exit(1)


def main():
    print("[INFO] Connecting to Roboflow...")
    rf = Roboflow(api_key=API_KEY)

    print("[INFO] Fetching project...")
    project = rf.workspace("kamal-kumar-8dloj").project("pla_objects-ils3d")
    version = project.version(2)

    print("[INFO] Downloading dataset in YOLOv8 format...")
    dataset = version.download("yolov8")

    print(f"[INFO] Dataset downloaded to: {dataset.location}")

    yaml_path = os.path.join(dataset.location, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"[OK] Found data.yaml at: {yaml_path}")
        with open(yaml_path, "r") as f:
            print("\n--- data.yaml contents ---")
            print(f.read())
    else:
        print(f"[WARN] data.yaml not found at expected path: {yaml_path}")


if __name__ == "__main__":
    main()
