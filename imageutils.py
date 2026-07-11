"""
Shared image loading helpers for the visualization scripts
"""
import cv2

MAX_PIXELS = 1.2e6


def load_grayscale(path, max_pixels=MAX_PIXELS):
    """Load an image as grayscale, downscaling it if it exceeds max_pixels."""
    image = cv2.imread(path, 0)
    assert image is not None, "Failed to read image: {}".format(path)

    if image.shape[0] * image.shape[1] > max_pixels:
        aspect_ratio = image.shape[1] / image.shape[0]
        image = cv2.resize(
            image, (int(1024 * aspect_ratio), 1024), interpolation=cv2.INTER_CUBIC
        )

    return image
