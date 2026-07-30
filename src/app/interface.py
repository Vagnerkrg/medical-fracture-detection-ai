from pathlib import Path

import gradio as gr
from ultralytics import YOLO


MODEL_PATH = Path(
    "models/runs/train/weights/best.pt"
)


def load_model():
    """
    Load trained YOLO model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return YOLO(
        str(MODEL_PATH)
    )


model = load_model()


def predict(image):
    """
    Execute fracture prediction.
    """

    results = model.predict(
        source=image,
        save=False
    )

    result = results[0]

    plotted_image = result.plot()

    predictions = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        predictions.append(
            f"{class_name}: {confidence:.2%}"
        )

    if not predictions:
        predictions.append(
            "No fracture detected"
        )

    return (
        plotted_image,
        "\n".join(predictions)
    )


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="numpy",
        label="Upload X-Ray image"
    ),
    outputs=[
        gr.Image(
            label="Prediction"
        ),
        gr.Textbox(
            label="Results"
        )
    ],
    title="Medical Fracture Detection AI",
    description=(
        "Upload an X-Ray image "
        "to detect possible fractures."
    )
)


if __name__ == "__main__":

    interface.launch()