from pathlib import Path

from src.training.config import TrainingConfig


def test_training_artifacts_exist():

    config = TrainingConfig()

    output_dir = config.get_output_dir()

    weights_dir = (
        output_dir / "weights"
    )

    assert output_dir.exists()

    assert weights_dir.exists()

    assert (
        weights_dir / "best.pt"
    ).exists()

    assert (
        weights_dir / "last.pt"
    ).exists()