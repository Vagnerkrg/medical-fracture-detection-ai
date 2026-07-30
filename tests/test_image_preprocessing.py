from pathlib import Path

from PIL import Image

from src.preprocessing.image_preprocessing import ImagePreprocessing


def test_image_preprocessing_initialization():

    preprocessing = ImagePreprocessing()

    assert preprocessing is not None


def test_validate_image_dimensions(tmp_path):

    image_path = tmp_path / "test_image.jpg"

    image = Image.new(
        "L",
        (640, 480)
    )

    image.save(
        image_path
    )

    preprocessing = ImagePreprocessing()

    dimensions = preprocessing.validate_image_dimensions(
        image_path
    )

    assert dimensions["width"] == 640
    assert dimensions["height"] == 480


def test_normalize_image(tmp_path):

    image_path = tmp_path / "test_image.jpg"

    image = Image.new(
        "L",
        (2, 2),
        color=255
    )

    image.save(
        image_path
    )

    preprocessing = ImagePreprocessing()

    normalized = preprocessing.normalize_image(
        image_path
    )

    assert normalized.max() == 1.0
    assert normalized.min() == 1.0


def test_prepare_image_for_model(tmp_path):

    image_path = tmp_path / "test_image.jpg"

    image = Image.new(
        "L",
        (640, 480)
    )

    image.save(
        image_path
    )

    preprocessing = ImagePreprocessing()

    prepared = preprocessing.prepare_image(
        image_path,
        size=(224, 224)
    )

    assert prepared.shape == (224, 224)


def test_validate_yolo_annotation(tmp_path):

    label_path = tmp_path / "sample.txt"

    label_path.write_text(
        "8 0.5 0.5 0.2 0.2"
    )

    preprocessing = ImagePreprocessing()

    result = preprocessing.validate_yolo_annotation(
        label_path
    )

    assert result is True