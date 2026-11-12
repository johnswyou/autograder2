"""Quarter turns, contrast correction and crops (src/redpen/imaging/prep.py)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from redpen.imaging.prep import (
    box_from_rotated,
    box_to_rotated,
    correct_contrast,
    crop,
    rotate,
)


@st.composite
def boxes(draw: st.DrawFn) -> list[float]:
    xs = sorted(draw(st.lists(st.floats(0, 1), min_size=2, max_size=2, unique=True)))
    ys = sorted(draw(st.lists(st.floats(0, 1), min_size=2, max_size=2, unique=True)))
    return [xs[0], ys[0], xs[1], ys[1]]


@given(boxes(), st.integers(0, 3))
def test_a_box_turned_and_back_is_unchanged(box: list[float], turns: int) -> None:
    assert box_from_rotated(box_to_rotated(box, turns), turns) == pytest.approx(box)


def test_a_turn_is_clockwise() -> None:
    image = Image.new("L", (4, 8))
    image.putpixel((0, 0), 255)

    assert rotate(image, 1).size == (8, 4)
    assert rotate(image, 1).getpixel((7, 0)) == 255
    assert rotate(image, -1).getpixel((0, 3)) == 255


@pytest.mark.parametrize("turns", range(4))
def test_a_turned_box_holds_what_the_box_held(turns: int) -> None:
    image = Image.effect_noise((64, 128), 64)
    box = [0.25, 0.125, 0.75, 0.5]  # whole pixels in every orientation

    held, _ = crop(image, box)
    turned, _ = crop(rotate(image, turns), box_to_rotated(box, turns))

    assert turned.tobytes() == rotate(held, turns).tobytes()


def test_contrast_correction_stretches_a_flat_photo() -> None:
    tones = [100 + v // 5 for v in range(256)]
    flat = Image.linear_gradient("L").point(tones).convert("RGB")

    low, high = correct_contrast(flat).convert("L").getextrema()

    assert (low, high) == (0, 255)


def test_crop_clamps_its_margin_to_the_image_and_returns_the_box_used() -> None:
    image = Image.new("RGB", (200, 100))

    part, used = crop(image, [0.05, 0.5, 0.5, 0.95], margin=0.1)

    assert used == [0.0, 0.4, 0.6, 1.0]
    assert part.size == (120, 60)


def test_crop_widens_a_box_to_whole_pixels() -> None:
    part, used = crop(Image.new("RGB", (10, 10)), [0.12, 0.12, 0.48, 0.48])

    assert used == [0.1, 0.1, 0.5, 0.5]
    assert part.size == (4, 4)
