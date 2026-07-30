from src.training.config import TrainingConfig
from src.training.trainer import YOLOTrainer


def test_training_config():

    config = TrainingConfig()

    assert config.device == "cuda"

    assert (
        config.get_dataset_yaml().exists()
    )


def test_yolo_trainer_initialization():

    config = TrainingConfig()

    trainer = YOLOTrainer(
        config
    )

    assert trainer.model is not None