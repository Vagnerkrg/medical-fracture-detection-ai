from pathlib import Path
import yaml


class DatasetLoader:
    """
    Responsible for loading and organizing
    the medical fracture detection dataset metadata.
    """

    def __init__(
        self,
        dataset_path: str
    ):
        self.dataset_path = Path(
            dataset_path
        )

        self.config_path = (
            self.dataset_path / "data.yaml"
        )


    def load_config(self):
        """
        Load YOLO dataset configuration.
        """

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Dataset config not found: {self.config_path}"
            )


        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:

            config = yaml.safe_load(
                file
            )


        return config



    def get_classes(self):
        """
        Return dataset classes.
        """

        config = self.load_config()

        return config.get(
            "names",
            []
        )



    def get_dataset_info(self):
        """
        Return basic dataset metadata.
        """

        config = self.load_config()

        return {

            "path": str(
                self.dataset_path
            ),

            "classes": config.get(
                "names",
                []
            ),

            "num_classes": config.get(
                "nc",
                0
            )

        }