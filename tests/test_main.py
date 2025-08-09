"""Unit tests for `sketch_image`.

Covers image-to-sketch conversion, file processing helpers, and the CLI entrypoint.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sketch_image.main import (
    convert,
    convert_image_to_sketch,
    process_image_file,
)


def create_dummy_bgr_image(width: int = 32, height: int = 32) -> np.ndarray:
    """Return a small synthetic BGR gradient image for use in tests."""
    # Create a simple gradient BGR image
    x = np.linspace(0, 255, num=width, dtype=np.uint8)
    y = np.linspace(0, 255, num=height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    b = xv
    g = yv
    r = (xv // 2 + yv // 2).astype(np.uint8)
    return np.dstack([b, g, r])


def test_convert_image_to_sketch_valid_image():
    """Produces a 2D uint8 sketch with same height/width for a valid BGR image."""
    img = create_dummy_bgr_image()
    sketch = convert_image_to_sketch(img)
    assert sketch.ndim == 2
    assert sketch.dtype == np.uint8
    assert sketch.shape == img.shape[:2]


@pytest.mark.parametrize("bad_image", [None, "not-an-array", np.array([], dtype=np.uint8)])
def test_convert_image_to_sketch_invalid_inputs(bad_image):
    """Raises TypeError/ValueError for invalid image inputs."""
    with pytest.raises((TypeError, ValueError)):
        convert_image_to_sketch(bad_image)  # type: ignore[arg-type]


def test_process_image_file_roundtrip(tmp_path: Path):
    """Writes a dummy image, processes it, and validates sketch properties."""
    # Save a dummy image using OpenCV and then process it
    img = create_dummy_bgr_image(16, 16)
    input_path = tmp_path / "input.jpg"

    # Lazy import cv2 here to keep test dependency surface small for pure unit parts
    import cv2 as cv

    assert cv.imwrite(str(input_path), img)
    sketch = process_image_file(str(input_path))
    assert sketch.ndim == 2
    assert sketch.shape == (16, 16)


def test_process_image_file_not_found(tmp_path: Path):
    """Raises FileNotFoundError when the input image path does not exist."""
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError):
        process_image_file(str(missing))


def test_cli_happy_path(tmp_path: Path):
    """CLI succeeds and writes output file when given a valid input image."""
    img = create_dummy_bgr_image(16, 16)
    input_path = tmp_path / "in.jpg"
    output_path = tmp_path / "out.jpg"

    import cv2 as cv

    assert cv.imwrite(str(input_path), img)

    runner = CliRunner()
    result = runner.invoke(convert, ["--image", str(input_path), "--output", str(output_path)])
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_cli_errors_on_missing_input(tmp_path: Path):
    """CLI exits non-zero and reports error when the input image is missing."""
    runner = CliRunner()
    output_path = tmp_path / "out.jpg"
    result = runner.invoke(convert, ["--image", str(tmp_path / "no.jpg"), "--output", str(output_path)])
    assert result.exit_code != 0
    assert "Image could not be read" in result.output
