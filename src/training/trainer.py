from pathlib import Path

from ultralytics import YOLO

from src.training.config import TrainingConfig


class YOLOTrainer:
    """
    Responsible for training
    YOLO fracture detection models.
    """

    def __init__(
        self,
        config: TrainingConfig
    ):
        self.config = config

        self._validate_environment()

        self.model = YOLO(
            self.config.model_name
        )


    def _validate_environment(self):
        """
        Validate training requirements.
        """

        dataset_yaml = (
            self.config.get_dataset_yaml()
        )

        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset yaml not found: {dataset_yaml}"
            )


        output_dir = (
            self.config.get_output_dir()
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )


    def train(self):
        """
        Execute YOLO training.
        """

        print(
            "Starting YOLO training..."
        )

        print(
            f"Model: {self.config.model_name}"
        )

        print(
            f"Device: {self.config.device}"
        )


        results = self.model.train(

            data=str(
                self.config.get_dataset_yaml()
            ),

            epochs=self.config.epochs,

            imgsz=self.config.image_size,

            batch=self.config.batch_size,

            device=self.config.device,

            project=str(
                self.config.project_dir
            ),

            name=self.config.experiment_name,

            exist_ok=True

        )


        print(
            "Training completed."
        )


        return results