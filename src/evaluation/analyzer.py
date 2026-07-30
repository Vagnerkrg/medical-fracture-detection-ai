from pathlib import Path


class TrainingAnalyzer:
    """
    Analyze YOLO training experiment artifacts.
    """

    def __init__(
        self,
        experiment_path="models/runs/train"
    ):
        self.experiment_path = Path(
            experiment_path
        )

        self.weights_path = (
            self.experiment_path / "weights"
        )


    def exists(self):
        """
        Check if training experiment exists.
        """
        return self.experiment_path.exists()


    def has_weights(self):
        """
        Check if model weights exist.
        """
        return (
            self.weights_path.exists()
            and
            (self.weights_path / "best.pt").exists()
        )


    def get_artifacts(self):
        """
        List generated training artifacts.
        """

        if not self.experiment_path.exists():
            return []

        return [
            str(path)
            for path in self.experiment_path.rglob("*")
            if path.is_file()
        ]