"""
Sketch Image

Convert images into sketches
"""

import os
from datetime import datetime

import cv2 as cv
import click
import numpy as np


DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.expanduser("~"), f"Pictures/sketch-{datetime.now().timestamp()}.jpg"
)


def convert_image_to_sketch(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to its pencil sketch representation.

    Parameters
    ----------
    image_bgr: np.ndarray
        An image array in BGR color space (as returned by OpenCV's imread).

    Returns
    -------
    np.ndarray
        A single-channel uint8 image containing the pencil sketch.
    """
    if image_bgr is None:
        raise ValueError("image_bgr must not be None")
    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy ndarray")
    if image_bgr.size == 0:
        raise ValueError("image_bgr must not be empty")

    gray_image = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

    inverted_image = 255 - gray_image
    blurred_image = cv.GaussianBlur(inverted_image, (21, 21), 0)
    inverted_blurred_img = 255 - blurred_image
    pencil_sketch = cv.divide(gray_image, inverted_blurred_img, scale=256.0)

    return pencil_sketch


def process_image_file(image_path: str) -> np.ndarray:
    """Read an image from disk and return its pencil sketch.

    Raises a FileNotFoundError or ValueError if the image cannot be read.
    """
    if not image_path or not isinstance(image_path, str):
        raise ValueError("image_path must be a non-empty string")

    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    return convert_image_to_sketch(img)


@click.command()
@click.option("--image", prompt="Image", help="Path of the image to sketch.")
@click.option("--output", default=DEFAULT_OUTPUT_PATH, help="Path of the output file.")
def convert(image: str, output: str | None) -> None:
    """CLI entrypoint to convert an image file path to a sketch and save it."""
    try:
        pencil_sketch = process_image_file(image)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise click.ClickException(f"[Error]: {exc}") from exc

    # Ensure output directory exists if provided
    output_path = output or DEFAULT_OUTPUT_PATH
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            raise click.ClickException(f"[Error]: Could not create output directory: {exc}") from exc

    write_ok = cv.imwrite(output_path, pencil_sketch)
    if not write_ok:
        raise click.ClickException("[Error]: File could not be written!")

    click.secho(f"File written to {output_path}", fg="green")
    click.secho("THANK YOU!", fg="green")
