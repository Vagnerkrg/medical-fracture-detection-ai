from pathlib import Path
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_CONFIG = (
    ROOT_DIR
    / "data"
    / "extracted"
    / "Human Bone Fractures Multi-modal Image Dataset (HBFMID)"
    / "Bone Fractures Detection"
    / "data.yaml"
)

MODEL_NAME = "yolo11n.pt"


def train_model():
    print("Starting YOLO training...")
    print(f"Dataset: {DATASET_CONFIG}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATASET_CONFIG),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(ROOT_DIR / "models" / "runs"),
        name="fracture_detection_baseline",
        exist_ok=True,
    )

    print("Training finished.")


if __name__ == "__main__":
    train_model()