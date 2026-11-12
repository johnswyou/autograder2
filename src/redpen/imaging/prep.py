"""Quarter turns, contrast correction and crops of a page image.

A box is [x0, y0, x1, y1] in fractions of an image, from its top-left corner. The
box functions here map a box between a turned or cropped image and the page it came
from. Rotation is applied here but never detected: the map stage reports it.
"""

from collections.abc import Sequence
from math import ceil, floor

from PIL import Image, ImageOps

_CLOCKWISE = {
    1: Image.Transpose.ROTATE_270,
    2: Image.Transpose.ROTATE_180,
    3: Image.Transpose.ROTATE_90,
}


def rotate(image: Image.Image, turns: int) -> Image.Image:
    """`image` turned clockwise by `turns` quarter turns (negative turns go back)."""
    turns %= 4
    return image.transpose(_CLOCKWISE[turns]) if turns else image.copy()


def box_to_rotated(box: Sequence[float], turns: int) -> list[float]:
    """Where `box` on an image lands once the image is turned by `rotate(_, turns)`."""
    x0, y0, x1, y1 = box
    for _ in range(turns % 4):
        x0, y0, x1, y1 = 1 - y1, x0, 1 - y0, x1
    return [x0, y0, x1, y1]


def box_from_rotated(box: Sequence[float], turns: int) -> list[float]:
    """Where `box` on the turned image lies on the image before `rotate(_, turns)`."""
    return box_to_rotated(box, -turns)


def correct_contrast(image: Image.Image) -> Image.Image:
    """A photographed page stretched to the full tonal range, its hues kept."""
    return ImageOps.autocontrast(image, cutoff=1, preserve_tone=True)


def crop(
    image: Image.Image, box: Sequence[float], margin: float = 0.0
) -> tuple[Image.Image, list[float]]:
    """The part of `image` in `box` grown by `margin` (a fraction of the image) on
    every side, and the box actually used: clamped to the image and widened to
    whole pixels."""
    width, height = image.size
    x0, y0, x1, y1 = box
    # Rounded first, so that float noise such as 120.00000000000001 adds no pixel.
    left = max(floor(round((x0 - margin) * width, 6)), 0)
    top = max(floor(round((y0 - margin) * height, 6)), 0)
    right = min(ceil(round((x1 + margin) * width, 6)), width)
    bottom = min(ceil(round((y1 + margin) * height, 6)), height)
    used = [left / width, top / height, right / width, bottom / height]
    return image.crop((left, top, right, bottom)), used
