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

    def count_class_distribution(self, labels_path: str):
        labels_dir = Path(labels_path)

        distribution = {}

        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, "r", encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        class_id = int(line.split()[0])

                        distribution[class_id] = (
                            distribution.get(class_id, 0) + 1
                        )

        return distribution

    def get_named_class_distribution(
        self,
        labels_path: str,
        classes: list
    ):
        class_distribution = self.count_class_distribution(
            labels_path
        )

        named_distribution = {}

        for class_id, count in class_distribution.items():
            class_name = classes[class_id]

            named_distribution[class_name] = count

        return named_distribution

    def generate_summary(
        self,
        split: str,
        labels_path: str,
        classes: list
    ):
        total_images = self.count_images(split)

        class_distribution = (
            self.get_named_class_distribution(
                labels_path,
                classes
            )
        )

        return {
            "total_images": total_images,
            "classes": class_distribution
        }