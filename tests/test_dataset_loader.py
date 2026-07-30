from pathlib import Path

from src.data.dataset_loader import DatasetLoader



DATASET_PATH = (
    "data/extracted/"
    "Human Bone Fractures Multi-modal Image Dataset (HBFMID)/"
    "Bone Fractures Detection"
)



def test_dataset_loader_loads_config():

    loader = DatasetLoader(
        DATASET_PATH
    )

    config = loader.load_config()


    assert config is not None

    assert "names" in config

    assert "nc" in config



def test_dataset_loader_returns_classes():

    loader = DatasetLoader(
        DATASET_PATH
    )

    classes = loader.get_classes()


    assert len(classes) == 10

    assert "Healthy" in classes

    assert "Spiral" in classes



def test_dataset_loader_returns_dataset_info():

    loader = DatasetLoader(
        DATASET_PATH
    )

    info = loader.get_dataset_info()


    assert info["num_classes"] == 10

    assert len(
        info["classes"]
    ) == 10

    assert Path(
        info["path"]
    ).exists()