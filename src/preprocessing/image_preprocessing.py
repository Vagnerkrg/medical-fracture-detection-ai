from pathlib import Path

import numpy as np
from PIL import Image


class ImagePreprocessing:
    """
    Responsible for basic preprocessing
    of medical X-ray images.
    """

    def __init__(self):
        pass


    def validate_image_dimensions(
        self,
        image_path: Path
    ):
        """
        Return image dimensions.
        """

        image = Image.open(
            image_path
        )

        width, height = image.size

        return {
            "width": width,
            "height": height
        }


    def normalize_image(
        self,
        image_path: Path
    ):
        """
        Normalize image pixels
        to range 0-1.
        """

        image = Image.open(
            image_path
        ).convert(
            "L"
        )

        image_array = np.array(
            image
        )

        return (
            image_array / 255.0
        )


    def prepare_image(
        self,
        image_path: Path,
        size=(224, 224)
    ):
        """
        Prepare image for model input.
        """

        image = Image.open(
            image_path
        ).convert(
            "L"
        )

        image = image.resize(
            size
        )

        return np.array(
            image
        )


    def validate_yolo_annotation(
        self,
        label_path: Path
    ):
        """
        Validate YOLO annotation format.

        Expected format:

        class_id x_center y_center width height
        """

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()


        for line in lines:

            values = line.strip().split()

            if len(values) != 5:
                return False

            try:
                int(values[0])

                coordinates = [
                    float(value)
                    for value in values[1:]
                ]

            except ValueError:
                return False


            if not all(
                0 <= value <= 1
                for value in coordinates
            ):
                return False


        return True