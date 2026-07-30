from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


class DatasetVisualization:
    """
    Responsible for visual analysis
    of fracture detection dataset.
    """

    def __init__(
        self,
        dataset_path: str
    ):
        self.dataset_path = Path(
            dataset_path
        )


    def get_images(
        self,
        split: str
    ):
        """
        Return dataset images from split.
        """

        images_path = (
            self.dataset_path
            / split
            / "images"
        )

        if not images_path.exists():
            return []

        return list(
            images_path.glob("*.jpg")
        )


    def get_label_path(
        self,
        image_path: Path
    ):
        """
        Return label file associated
        with an image.
        """

        labels_path = (
            image_path.parent.parent
            / "labels"
        )

        return (
            labels_path
            / f"{image_path.stem}.txt"
        )


    def read_annotations(
        self,
        label_path: Path
    ):
        """
        Read YOLO annotations.
        """

        annotations = []

        if not label_path.exists():
            return annotations

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                if not line.strip():
                    continue

                values = line.split()

                annotations.append(
                    {
                        "class_id": int(values[0]),
                        "x_center": float(values[1]),
                        "y_center": float(values[2]),
                        "width": float(values[3]),
                        "height": float(values[4])
                    }
                )

        return annotations


    def get_images_by_class(
        self,
        split: str,
        class_id: int
    ):
        """
        Return images containing
        a specific class.
        """

        images = self.get_images(
            split
        )

        class_images = []

        for image in images:

            annotations = self.read_annotations(
                self.get_label_path(image)
            )

            for annotation in annotations:

                if annotation["class_id"] == class_id:
                    class_images.append(image)
                    break

        return class_images


    def create_image_grid(
        self,
        images,
        columns=2
    ):
        """
        Create image grid visualization.
        """

        if not images:
            return None

        rows = (
            len(images) + columns - 1
        ) // columns

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(8, 8)
        )

        if rows == 1:
            axes = [axes]

        else:
            axes = axes.flatten()


        for ax in axes:
            ax.axis("off")


        for index, image_path in enumerate(images):

            image = Image.open(
                image_path
            )

            axes[index].imshow(
                image,
                cmap="gray"
            )

            axes[index].set_title(
                image_path.name
            )

            axes[index].axis("off")


        return fig