from src.evaluation.analyzer import TrainingAnalyzer


def test_training_experiment_exists():

    analyzer = TrainingAnalyzer()

    assert analyzer.exists()


def test_training_weights_exist():

    analyzer = TrainingAnalyzer()

    assert analyzer.has_weights()


def test_training_artifacts_listing():

    analyzer = TrainingAnalyzer()

    artifacts = analyzer.get_artifacts()

    assert len(artifacts) > 0
    assert any(
        "best.pt" in artifact
        for artifact in artifacts
    )