from pathlib import Path

from src.data.dataset_visualization import DatasetVisualization


DATASET_PATH = (
    "data/extracted/"
    "Human Bone Fractures Multi-modal Image Dataset (HBFMID)/"
    "Bone Fractures Detection"
)


def test_dataset_visualization_initialization():
    visualization = DatasetVisualization(
        DATASET_PATH
    )

    assert visualization.dataset_path.exists()


def test_get_images_from_split():

    visualization = DatasetVisualization(
        DATASET_PATH
    )

    images = visualization.get_images(
        "train"
    )

    assert len(images) > 0


def test_get_label_path():

    visualization = DatasetVisualization(
        DATASET_PATH
    )

    image_path = Path(
        DATASET_PATH
        + "/train/images/"
        + "101_jpg.rf.275531ad788db79438f0f14a4cf4cd9e.jpg"
    )

    label_path = visualization.get_label_path(
        image_path
    )

    assert label_path.exists()
    assert label_path.suffix == ".txt"


def test_read_annotations():

    visualization = DatasetVisualization(
        DATASET_PATH
    )

    label_path = Path(
        DATASET_PATH
        + "/train/labels/"
        + "101_jpg.rf.275531ad788db79438f0f14a4cf4cd9e.txt"
    )

    annotations = visualization.read_annotations(
        label_path
    )

    assert len(annotations) > 0
    assert annotations[0]["class_id"] == 8


def test_get_images_by_class():

    visualization = DatasetVisualization(
        DATASET_PATH
    )

    images = visualization.get_images_by_class(
        "train",
        8
    )

    assert len(images) > 0


def test_create_image_grid():

    visualization = DatasetVisualization(
        DATASET_PATH
    )

    images = visualization.get_images_by_class(
        "train",
        8
    )

    grid = visualization.create_image_grid(
        images[:4]
    )

    assert grid is not None