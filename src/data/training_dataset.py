from pathlib import Path


class TrainingDatasetBuilder:
    """
    Build classification dataset
    from YOLO formatted annotations.
    """

    def __init__(
        self,
        dataset_path: str
    ):
        self.dataset_path = Path(dataset_path)


    def get_split_path(
        self,
        split: str
    ):
        return (
            self.dataset_path
            / split
        )


    def read_label(
        self,
        label_path: Path
    ):
        """
        Extract first class id
        from YOLO annotation.
        """

        if not label_path.exists():
            return None


        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()


        if not lines:
            return None


        first_line = lines[0].strip()

        class_id = int(
            first_line.split()[0]
        )

        return class_id



    def build_dataset(
        self,
        split="train"
    ):
        """
        Create image-label pairs.
        """

        images_path = (
            self.get_split_path(split)
            / "images"
        )

        labels_path = (
            self.get_split_path(split)
            / "labels"
        )


        dataset = []


        for image_path in images_path.iterdir():

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                continue


            label_path = (
                labels_path
                /
                f"{image_path.stem}.txt"
            )


            class_id = self.read_label(
                label_path
            )


            if class_id is not None:

                dataset.append(
                    {
                        "image": image_path,
                        "label": class_id
                    }
                )


        return dataset