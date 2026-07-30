from pathlib import Path


class TrainingConfig:
    """
    Central training configuration
    for YOLO fracture detection model.
    """

    def __init__(self):

        self.dataset_path = Path(
            "data/extracted/Human Bone Fractures Multi-modal Image Dataset (HBFMID)/Bone Fractures Detection"
        )

        self.model_name = (
            "yolo11n.pt"
        )

        self.image_size = (
            640
        )

        self.epochs = (
            50
        )

        self.batch_size = (
            16
        )

        self.device = (
            "cuda"
        )

        self.project_dir = Path(
            "models/runs"
        )

        self.experiment_name = (
            "fracture_detection_baseline"
        )


    def get_dataset_yaml(self):
        """
        Return YOLO dataset configuration file.
        """

        return (
            self.dataset_path / "data.yaml"
        )


    def get_output_dir(self):
        """
        Return training output directory.
        """

        return (
            self.project_dir / self.experiment_name
        )