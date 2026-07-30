from pathlib import Path
import yaml


class DatasetStatistics:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)

    def count_images(self, split: str) -> int:
        images_path = (
            self.dataset_path
            / split
            / "images"
        )

        if not images_path.exists():
            return 0

        return len(list(images_path.glob("*.jpg")))

    def load_classes(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return data["names"]