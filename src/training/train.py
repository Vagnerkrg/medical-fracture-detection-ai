from src.training.config import TrainingConfig
from src.training.trainer import YOLOTrainer


def main():
    """
    Execute YOLO baseline training experiment.
    """

    config = TrainingConfig()

    trainer = YOLOTrainer(
        config
    )

    results = trainer.train()

    return results


if __name__ == "__main__":
    main()