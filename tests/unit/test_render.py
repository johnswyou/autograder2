"""Page images, sidebars and regions (src/redpen/imaging/render.py).

The fixture submissions are checked against their truth files in test_ingest.py.
"""

from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from PIL import Image, ImageChops, ImageFilter, ImageStat
from reportlab.pdfgen.canvas import Canvas

from redpen.imaging.prep import crop
from redpen.imaging.render import (
    DPI,
    MAX_EDGE,
    classkick_sidebars,
    open_pdf,
    render_page,
    render_region,
    sidebar,
)
from tests.fixture_truth import FIXTURES

BLANK = FIXTURES / "blank.pdf"
BLANK_CLASSKICK = FIXTURES / "blank_classkick.pdf"


def edges(image: Image.Image) -> float:
    """Variance of an edge filter: higher on a sharper image of the same content."""
    return ImageStat.Stat(image.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]


def test_the_vector_blank_renders_at_200_dpi_with_its_text_layer() -> None:
    pdf = pdfium.PdfDocument(BLANK)

    assert classkick_sidebars(pdf) is None
    for page in pdf:
        rendered = render_page(page, "other")
        width, height = (round(side * DPI / 72) for side in page.get_size())
        assert rendered.image.size == (width, height)
        assert rendered.dpi == DPI
        assert rendered.native_ppi is None
        assert rendered.sidebar is None
        assert rendered.text
        page_text = page.get_textpage().get_text_bounded()
        spans = "".join(span.text for span in rendered.text)
        assert "".join(spans.split()) == "".join(page_text.split())


def test_the_classkick_blank_renders_its_slides_and_sidebars() -> None:
    pdf = pdfium.PdfDocument(BLANK_CLASSKICK)

    sidebars = classkick_sidebars(pdf)

    assert sidebars is not None
    assert [s.slide_number for s in sidebars] == list(range(1, len(pdf) + 1))
    assert {s.slide_count for s in sidebars} == {len(pdf)}
    assert all(s.student_name is None and s.assignment_title for s in sidebars)
    for page in pdf:
        rendered = render_page(page, "classkick")
        [slide] = page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])
        assert isinstance(slide, pdfium.PdfImage)
        width, height = slide.get_px_size()
        left, _, right, _ = slide.get_bounds()
        assert rendered.image.size == (width, height)
        assert rendered.dpi is None
        assert rendered.native_ppi == pytest.approx(width * 72 / (right - left))
        assert rendered.text == []


@pytest.mark.parametrize("rotation", [0, 90])
def test_an_other_region_and_text_span_land_where_the_page_shows_them(
    rotation: int,
) -> None:
    page = pdfium.PdfDocument(BLANK)[0]
    page.set_rotation(rotation)
    working = render_page(page, "other")
    box = [0.1, 0.2, 0.6, 0.45]
    part, _ = crop(working.image, box)

    region = render_region(page, "other", box, max(part.size))

    assert abs(region.image.width - part.width) <= 1
    assert abs(region.image.height - part.height) <= 1
    difference = ImageChops.difference(region.image.resize(part.size), part)
    assert max(ImageStat.Stat(difference).mean) < 2
    for span in working.text:
        ink, _ = crop(working.image, span.box)
        assert ImageStat.Stat(ink.convert("L")).extrema[0][0] < 128, span.text


def test_a_region_rendered_at_three_times_is_sharper_than_the_page_enlarged() -> None:
    page = pdfium.PdfDocument(BLANK)[0]
    working = render_page(page, "other")
    line = max(working.text, key=lambda span: len(span.text))
    part, box = crop(working.image, line.box, margin=0.01)
    enlarged = part.resize((part.width * 3, part.height * 3), Image.Resampling.LANCZOS)

    region = render_region(page, "other", box, 3 * max(part.size))

    assert region.scale is None  # vector content: no ceiling
    assert abs(region.image.width - enlarged.width) <= 3
    assert edges(region.image) > 1.5 * edges(enlarged)


def test_a_classkick_region_is_cropped_from_the_slide_and_enlarged_at_most_twice() -> (
    None
):
    page = pdfium.PdfDocument(BLANK_CLASSKICK)[0]
    slide = render_page(page, "classkick").image
    box = [0.1, 0.1, 0.4, 0.2]
    part, used = crop(slide, box)

    native = render_region(page, "classkick", box, max(part.size))
    capped = render_region(page, "classkick", box, 5 * max(part.size))

    assert native.scale == 1.0
    assert native.image.tobytes() == part.tobytes()
    assert native.box == used
    assert capped.scale == 2.0
    assert capped.image.size == (2 * part.width, 2 * part.height)


def test_a_classkick_export_with_anything_added_to_a_page_is_other() -> None:
    pdf = pdfium.PdfDocument(BLANK_CLASSKICK)
    pasted, annotated, untouched = pdf[0], pdf[1], pdf[2]
    photo = pdfium.PdfImage.new(pdf)
    photo.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (50, 50), "gray")))
    photo.set_matrix(pdfium.PdfMatrix().scale(72, 72).translate(30, 30))  # 50 ppi
    pasted.insert_obj(photo)
    pasted.gen_content()
    annot = pdfium_c.FPDFPage_CreateAnnot(annotated.raw, pdfium_c.FPDF_ANNOT_INK)
    pdfium_c.FPDFPage_CloseAnnot(annot)

    assert sidebar(pasted) is None
    assert sidebar(annotated) is None
    assert sidebar(untouched) is not None
    assert classkick_sidebars(pdf) is None
    rendered = render_page(pasted, "other")
    assert rendered.image.size == tuple(round(s * DPI / 72) for s in pasted.get_size())
    slide = render_page(untouched, "classkick").native_ppi  # the larger image's
    assert rendered.native_ppi == pytest.approx(slide)


def place(
    pdf: pdfium.PdfDocument, page: pdfium.PdfPage, px: int, inches: float
) -> None:
    """Put a square image of `px` pixels a side, `inches` wide, on `page`."""
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(pdfium.PdfBitmap.from_pil(Image.new("RGB", (px, px), "gray")))
    image.set_matrix(pdfium.PdfMatrix().scale(inches * 72, inches * 72))
    page.insert_obj(image)
    page.gen_content()


def test_a_huge_page_renders_with_its_longest_edge_at_the_cap() -> None:
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(30 * 72, 40 * 72)  # 30 x 40 inches

    rendered = render_page(page, "other")

    assert rendered.image.size == (3072, MAX_EDGE)
    assert rendered.dpi == pytest.approx(MAX_EDGE / 40)
    assert rendered.image.width / 30 == pytest.approx(rendered.dpi, abs=0.05)


def test_native_resolution_is_the_largest_image_s_not_a_small_sharp_one() -> None:
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(612, 792)
    place(pdf, page, 800, 8)  # 100 ppi over most of the page
    place(pdf, page, 300, 0.5)  # 600 ppi, a small logo

    assert render_page(page, "other").native_ppi == pytest.approx(100)


def test_an_answer_typed_into_a_form_field_shows_in_page_and_region(
    tmp_path: Path,
) -> None:
    path = tmp_path / "form.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792))
    canvas.drawString(72, 750, "Q1. What is 2 + 2?")
    canvas.acroForm.textfield(  # pyright: ignore[reportUnknownMemberType]
        name="q1",
        value="FOUR",
        x=72,
        y=680,
        width=300,
        height=40,
        fontSize=20,
        borderWidth=0,
        fillColor=None,
    )
    canvas.save()
    field = [72 / 612, (792 - 720) / 792, 372 / 612, (792 - 680) / 792]

    def darkest(image: Image.Image) -> int:
        return ImageStat.Stat(image.convert("L")).extrema[0][0]

    pdf = open_pdf(path)
    page = pdf[0]

    assert darkest(crop(render_page(page, "other").image, field)[0]) < 128
    assert darkest(render_region(page, "other", field, 600).image) < 128
    # Without the form environment the field renders empty.
    bare = render_page(pdfium.PdfDocument(path)[0], "other")
    assert darkest(crop(bare.image, field)[0]) > 200


def test_a_pdf_whose_form_environment_fails_is_closed_before_the_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[pdfium.PdfDocument] = []

    def fails(pdf: pdfium.PdfDocument) -> None:
        opened.append(pdf)
        raise pdfium.PdfiumError("no form environment")

    monkeypatch.setattr(pdfium.PdfDocument, "init_forms", fails)
    with pytest.raises(pdfium.PdfiumError):
        open_pdf(BLANK)

    assert opened[0].raw is None
