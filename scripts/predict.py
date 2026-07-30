from pathlib import Path
from ultralytics import YOLO
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "runs"
    / "train"
    / "weights"
    / "best.pt"
)


def predict(image_path):

    print("Loading model...")
    print(f"Model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    results = model(image_path)

    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            print("No fracture detected.")
            return

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = result.names[cls_id]

            print(
                f"Class: {label} | "
                f"Confidence: {confidence:.2%}"
            )


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/predict.py <image_path>"
        )
        sys.exit(1)

    predict(sys.argv[1])