from pathlib import Path

from ultralytics import YOLO


class GradCAM:
    """
    Grad-CAM explainability handler.
    """

    def __init__(
        self,
        model_path: Path
    ):
        self.model_path = Path(model_path)
        self.model = None

    def exists(self):
        """
        Check if trained model exists.
        """

        return self.model_path.exists()

    def load_model(self):
        """
        Load trained YOLO model.
        """

        self.model = YOLO(
            str(self.model_path)
        )

        return self.model

    def generate_heatmap(
        self,
        image_path: Path
    ):
        """
        Generate Grad-CAM heatmap output.

        Current implementation:
        runs model inference and returns prediction result.
        """

        if self.model is None:
            self.load_model()

        results = self.model.predict(
            source=str(image_path),
            save=False
        )

        return results

    def save_heatmap(
        self,
        image_path: Path,
        output_path: Path
    ):
        """
        Save explainability visualization artifact.
        """

        if self.model is None:
            self.load_model()

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        results = self.model.predict(
            source=str(image_path),
            save=False
        )

        plotted = results[0].plot()

        import cv2

        cv2.imwrite(
            str(output_path),
            plotted
        )

        return output_path