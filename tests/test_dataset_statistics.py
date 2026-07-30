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


def test_count_class_distribution():
    labels_path = (
        f"{DATASET_PATH}/train/labels"
    )

    statistics = DatasetStatistics(DATASET_PATH)

    distribution = statistics.count_class_distribution(
        labels_path
    )

    assert len(distribution) > 0
    assert sum(distribution.values()) > 0


def test_get_named_class_distribution():
    yaml_path = (
        f"{DATASET_PATH}/data.yaml"
    )

    labels_path = (
        f"{DATASET_PATH}/train/labels"
    )

    statistics = DatasetStatistics(DATASET_PATH)

    classes = statistics.load_classes(yaml_path)

    distribution = statistics.get_named_class_distribution(
        labels_path,
        classes
    )

    assert len(distribution) > 0
    assert "Healthy" in distribution
    assert distribution["Healthy"] > 0


def test_generate_summary():
    yaml_path = (
        f"{DATASET_PATH}/data.yaml"
    )

    labels_path = (
        f"{DATASET_PATH}/train/labels"
    )

    statistics = DatasetStatistics(DATASET_PATH)

    classes = statistics.load_classes(yaml_path)

    summary = statistics.generate_summary(
        "train",
        labels_path,
        classes
    )

    assert summary["total_images"] > 0
    assert len(summary["classes"]) > 0
    assert "Healthy" in summary["classes"]