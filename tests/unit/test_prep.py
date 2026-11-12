"""Quarter turns, contrast correction and crops (src/redpen/imaging/prep.py)."""

import pypdfium2 as pdfium
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image, ImageOps, ImageStat

from redpen.imaging.prep import (
    box_from_rotated,
    box_to_rotated,
    correct_contrast,
    crop,
    rotate,
)
from redpen.imaging.render import render_page
from redpen.schemas.key import Key
from redpen.schemas.submission import MapEntry
from tests.fixture_truth import FIXTURES, SUBMISSIONS, truth_of


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


# The truth file has no rotation field. It implies a page's turn through its boxes:
# on an upright page, written work is wider than tall and the parts run from top to
# bottom in key order. Drawn responses have no such shape.
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
ORDER = [part.id for part in KEY.parts]
WRITTEN = {
    p.id for p in KEY.parts if p.response_type not in ("free_body_diagram", "graph")
}


def sits_upright(entries: list[MapEntry], size: tuple[int, int], turns: int) -> bool:
    width, height = (size[1], size[0]) if turns % 2 else size
    boxes = {e.part_id: box_to_rotated(e.box, turns) for e in entries}
    wide = all(
        (x1 - x0) * width > (y1 - y0) * height
        for part, (x0, y0, x1, y1) in boxes.items()
        if part in WRITTEN
    )
    tops = [boxes[part][1] for part in sorted(boxes, key=ORDER.index)]
    return wide and tops == sorted(tops)


def ink_variance(image: Image.Image) -> tuple[float, float]:
    """How much the ink varies from row to row and from column to column. Lines of
    text that read left to right vary more from row to row."""
    gray = ImageOps.invert(image.convert("L"))
    rows = gray.resize((1, gray.height), Image.Resampling.BOX)
    columns = gray.resize((gray.width, 1), Image.Resampling.BOX)
    return ImageStat.Stat(rows).var[0], ImageStat.Stat(columns).var[0]


def test_the_rotated_fixture_reads_upright_after_the_turn_its_truth_implies() -> None:
    [path] = [p for p in SUBMISSIONS if truth_of(p).layout_class == "rotated"]
    truth = truth_of(path)
    pdf = pdfium.PdfDocument(path)
    turned = 0
    for page in truth.pages:
        entries = [e for e in truth.map.entries if e.page == page.number]
        size = (page.width_px, page.height_px)
        [turns] = [t for t in range(4) if sits_upright(entries, size, t)]
        if turns == 0:
            continue
        turned += 1
        upright = rotate(render_page(pdf[page.number - 1], "classkick").image, turns)
        assert upright.size == (page.height_px, page.width_px)
        rows, columns = 0.0, 0.0
        for entry in entries:
            if entry.part_id in WRITTEN:
                work, _ = crop(upright, box_to_rotated(entry.box, turns))
                down, across = ink_variance(work)
                rows, columns = rows + down, columns + across
        assert rows > 1.5 * columns
    assert turned == 1  # the class has one photo turned sideways
