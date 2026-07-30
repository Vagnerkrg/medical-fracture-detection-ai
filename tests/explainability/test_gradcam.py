from pathlib import Path

from src.explainability.gradcam import GradCAM


def test_gradcam_initialization():

    model_path = (
        Path("models")
        / "runs"
        / "fracture_detection_baseline"
        / "weights"
        / "best.pt"
    )

    gradcam = GradCAM(
        model_path
    )

    assert gradcam.exists()


def test_gradcam_load_model():

    model_path = (
        Path("models")
        / "runs"
        / "fracture_detection_baseline"
        / "weights"
        / "best.pt"
    )

    gradcam = GradCAM(
        model_path
    )

    model = gradcam.load_model()

    assert model is not None


def test_gradcam_generate_heatmap():

    model_path = (
        Path("models")
        / "runs"
        / "fracture_detection_baseline"
        / "weights"
        / "best.pt"
    )

    image_path = next(
        Path("data")
        .rglob("*.jpg")
    )

    gradcam = GradCAM(
        model_path
    )

    output = gradcam.generate_heatmap(
        image_path
    )

    assert output is not None


def test_gradcam_save_heatmap():

    model_path = (
        Path("models")
        / "runs"
        / "fracture_detection_baseline"
        / "weights"
        / "best.pt"
    )

    image_path = next(
        Path("data")
        .rglob("*.jpg")
    )

    output_path = (
        Path("models")
        / "explainability"
        / "heatmap_test.jpg"
    )

    gradcam = GradCAM(
        model_path
    )

    result = gradcam.save_heatmap(
        image_path,
        output_path
    )

    assert result.exists()