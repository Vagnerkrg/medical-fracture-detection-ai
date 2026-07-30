from src.training.config import TrainingConfig
from src.training.trainer import YOLOTrainer


def test_training_execution(monkeypatch):

    config = TrainingConfig()

    trainer = YOLOTrainer(
        config
    )


    class MockResult:
        pass


    def mock_train(**kwargs):
        return MockResult()


    monkeypatch.setattr(
        trainer.model,
        "train",
        mock_train
    )


    result = trainer.train()


    assert result is not None