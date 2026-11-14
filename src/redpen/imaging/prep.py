"""Quarter turns, contrast correction and crops of a page image, and the image as
it is sent to a model.

A box is [x0, y0, x1, y1] in fractions of an image, from its top-left corner. The
box functions here map a box between a turned or cropped image and the page it came
from. Rotation is applied here but never detected: the map stage reports it.
"""

import io
from collections.abc import Sequence
from math import ceil, floor

from PIL import Image, ImageOps

# The flat colour of a blanked region: a reader of the image sees it was covered.
BLANKED = "black"

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


def box_within(box: Sequence[float], frame: Sequence[float]) -> list[float] | None:
    """The part of `box` inside `frame`, both boxes on one image, as fractions of
    `frame`; None when they share no area."""
    x0, y0, x1, y1 = box
    f0, g0, f1, g1 = frame
    left, top, right, bottom = max(x0, f0), max(y0, g0), min(x1, f1), min(y1, g1)
    if left >= right or top >= bottom:
        return None
    width, height = f1 - f0, g1 - g0
    return [
        (left - f0) / width,
        (top - g0) / height,
        (right - f0) / width,
        (bottom - g0) / height,
    ]


def correct_contrast(image: Image.Image) -> Image.Image:
    """A photographed page stretched to the full tonal range, its hues kept."""
    return ImageOps.autocontrast(image, cutoff=1, preserve_tone=True)


def white_share(image: Image.Image, level: int) -> float:
    """The share of `image`'s pixels whose grey level is `level` or more."""
    counts = image.convert("L").histogram()
    return sum(counts[level:]) / sum(counts)


def crop(
    image: Image.Image, box: Sequence[float], margin: float = 0.0
) -> tuple[Image.Image, list[float]]:
    """The part of `image` in `box` grown by `margin` (a fraction of the image) on
    every side, and the box actually used: clamped to the image and widened to
    whole pixels."""
    width, height = image.size
    left, top, right, bottom = _pixels(image.size, box, margin)
    used = [left / width, top / height, right / width, bottom / height]
    return image.crop((left, top, right, bottom)), used


def reduced_png(
    image: Image.Image, budget: int, blanked: Sequence[Sequence[float]] = ()
) -> bytes:
    """`image` as PNG, reduced to fit `budget` pixels on its longest edge (the image
    budget of design 10.2), then each box of `blanked` filled with `BLANKED` on
    every pixel it touches."""
    small = image.copy()
    small.thumbnail((budget, budget))
    for box in blanked:
        small.paste(BLANKED, _pixels(small.size, box))
    buffer = io.BytesIO()
    small.save(buffer, "PNG")
    return buffer.getvalue()


def _pixels(
    size: tuple[int, int], box: Sequence[float], margin: float = 0.0
) -> tuple[int, int, int, int]:
    """The whole pixels `box` grown by `margin` touches, clamped to an image of
    `size`: left, top, right and bottom, the last two exclusive."""
    width, height = size
    x0, y0, x1, y1 = box
    # Rounded first, so that float noise such as 120.00000000000001 adds no pixel.
    left = max(floor(round((x0 - margin) * width, 6)), 0)
    top = max(floor(round((y0 - margin) * height, 6)), 0)
    right = min(ceil(round((x1 + margin) * width, 6)), width)
    bottom = min(ceil(round((y1 + margin) * height, 6)), height)
    return left, top, right, bottom
