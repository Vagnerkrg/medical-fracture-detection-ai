from src.data.dataset_statistics import DatasetStatistics


DATASET_PATH = (
    "data/extracted/"
    "Human Bone Fractures Multi-modal Image Dataset (HBFMID)/"
    "Bone Fractures Detection"
)


def test_count_train_images():
    statistics = DatasetStatistics(DATASET_PATH)

    count = statistics.count_images("train")

    assert count > 0


def test_count_validation_images():
    statistics = DatasetStatistics(DATASET_PATH)

    count = statistics.count_images("valid")

    assert count > 0


def test_load_dataset_classes():
    yaml_path = (
        f"{DATASET_PATH}/data.yaml"
    )

    statistics = DatasetStatistics(DATASET_PATH)

    classes = statistics.load_classes(yaml_path)

    assert len(classes) == 10
    assert "Healthy" in classes