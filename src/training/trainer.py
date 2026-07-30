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

        self.model = YOLO(
            self.config.model_name
        )


    def train(self):
        """
        Execute YOLO training.
        """

        results = self.model.train(

            data=str(
                self.config.get_dataset_yaml()
            ),

            epochs=self.config.epochs,

            imgsz=self.config.image_size,

            batch=self.config.batch_size,

            device=self.config.device,

            project=str(
                self.config.project_dir.resolve()
            ),

            name="",

            exist_ok=True

        )

        return results