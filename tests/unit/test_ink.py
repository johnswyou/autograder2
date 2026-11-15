"""Ink drawn as vector paths has no resolution ceiling; a scan keeps its own
(src/autograder2/imaging/render.py, through the read stage and the flags it leads to).

The three pages of tests/fixtures/ink answer Q4(b) of the synthetic assignment in
small pen writing: as vector paths on a white page, the same paths over a worksheet
image of 72 pixels per inch, and the same answer inside a 72 ppi scan that carries a
vector frame and page number.
"""

import asyncio
import json
import shutil
from math import floor
from pathlib import Path

import httpx
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageFilter, ImageStat
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from autograder2.grade.ingest import ingest
from autograder2.grade.read import Readings, read_student, unit_images, units
from autograder2.imaging import prep, render
from autograder2.llm.client import Gateway
from autograder2.llm.models import ModelInfo
from autograder2.report.report import flags, load
from autograder2.schemas.common import Provenance
from autograder2.schemas.flags import FlagCode
from autograder2.schemas.key import Key
from autograder2.schemas.reading import Reading
from autograder2.schemas.submission import (
    IngestRecord,
    Map,
    MapEntry,
    MapPage,
    MapRecord,
)
from autograder2.workspace import Workspace, write_model
from tests.fixture_truth import FIXTURES, SPEC

INK = Path(__file__).parents[1] / "fixtures" / "ink"
TRUTH = json.loads((INK / "truth.json").read_text())
BOX: list[float] = TRUTH["box"]
PAGES = ["pen-on-white", "pen-on-raster", "scan-with-frame"]
LOW_PPI = 72  # the worksheet's and the scan's resolution
BUDGET = 1536
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
MODEL = ModelInfo.model_validate(
    {
        "slug": "vendor/vision-1",
        "supported_parameters": ["structured_outputs", "max_tokens"],
        "accepts_images": True,
        "prompt_price": "0",
        "completion_price": "0",
        "providers": ["zero"],
    }
)
MAPPED = MapRecord(
    provenance=Provenance(key_part_hash=None, prompt_version=None, model_slug=None),
    pages=[MapPage(number=1, rotation=0, photo=False)],
    map=Map(
        entries=[
            MapEntry(
                part_id="Q4.b",
                page=1,
                box=BOX,
                basis="position",
                role="work",
                confidence="high",
            )
        ],
        unanswered=[p.id for p in SPEC.parts if p.id != "Q4.b"],
        set_aside=[],
    ),
)


def edges(image: Image.Image) -> float:
    """Variance of an edge filter: higher on a sharper image of the same content."""
    return ImageStat.Stat(image.convert("L").filter(ImageFilter.FIND_EDGES)).var[0]


@pytest.fixture(scope="module")
def ingested(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Workspace, dict[str, IngestRecord]]:
    """The three pages ingested, one student each, named as their files."""
    root = tmp_path_factory.mktemp("ink")
    (root / "submissions").mkdir()
    for name in PAGES:
        shutil.copyfile(INK / f"{name}.pdf", root / "submissions" / f"{name}.pdf")
    workspace = Workspace(root / "ws")
    summary = ingest(root / "submissions", workspace)
    return workspace, {r.student_name: r for r in summary.students}


def copied(
    ingested: tuple[Workspace, dict[str, IngestRecord]], name: str, root: Path
) -> tuple[Workspace, IngestRecord]:
    """A workspace holding the page `name`, ingested and mapped, to write into."""
    shared, records = ingested
    record = records[name]
    workspace = Workspace(root)
    shutil.copytree(
        shared.student_dir(record.student_key),
        workspace.student_dir(record.student_key),
    )
    write_model(workspace.map_json(record.student_key), MAPPED)
    return workspace, record


def saved_crop(workspace: Workspace, record: IngestRecord) -> Image.Image:
    """Q4.b's region as the read stage saved it for the grade stage."""
    with Image.open(workspace.crop_png(record.student_key, "Q4.b", 1)) as image:
        image.load()
    return image


@pytest.mark.parametrize("name", PAGES)
def test_pen_strokes_are_rendered_to_the_budget_and_the_scan_stops_at_its_ceiling(
    ingested: tuple[Workspace, dict[str, IngestRecord]], tmp_path: Path, name: str
) -> None:
    workspace, record = copied(ingested, name, tmp_path / "ws")
    [unit] = units(SPEC, MAPPED.map)

    images = unit_images(workspace, record.student_key, unit, MAPPED.pages, BUDGET)

    [crop] = images.crops["Q4.b"]
    region = saved_crop(workspace, record)
    if name == "scan-with-frame":  # as before: twice the scan's own pixels
        x0, _, x1, _ = crop.box
        assert crop.scale is not None and 1.99 < crop.scale <= 2.0
        assert region.width == pytest.approx(2 * (x1 - x0) * 8.5 * LOW_PPI, abs=2)
    else:
        assert crop.scale is None
        assert max(region.size) == BUDGET


def test_pen_strokes_over_a_raster_hold_more_detail_than_its_render_at_the_ceiling(
    ingested: tuple[Workspace, dict[str, IngestRecord]], tmp_path: Path
) -> None:
    workspace, record = copied(ingested, "pen-on-raster", tmp_path / "ws")
    [unit] = units(SPEC, MAPPED.map)
    images = unit_images(workspace, record.student_key, unit, MAPPED.pages, BUDGET)
    [crop] = images.crops["Q4.b"]
    sent = saved_crop(workspace, record)
    # What the read stage sent before: the region at twice the worksheet's pixels.
    x0, y0, x1, y1 = crop.box
    longest = max((x1 - x0) * 612, (y1 - y0) * 792)
    with render.open_pdf(workspace.source_pdf(record.student_key)) as pdf:
        edge = floor(2 * LOW_PPI * longest / 72)
        before = render.render_region(pdf[0], "other", crop.box, edge).image

    enlarged = before.resize(sent.size, Image.Resampling.LANCZOS)
    assert max(before.size) < BUDGET // 4
    assert edges(sent) > 2 * edges(enlarged)


@pytest.mark.parametrize("name", PAGES)
def test_a_hard_to_read_part_is_low_resolution_on_the_scan_alone(
    ingested: tuple[Workspace, dict[str, IngestRecord]], tmp_path: Path, name: str
) -> None:
    workspace, record = copied(ingested, name, tmp_path / "ws")
    poor = Reading.model_validate(TRUTH["reading"] | {"legibility": "poor"})

    def answer(request: httpx.Request) -> httpx.Response:
        value = Readings(readings=[poor]).model_dump_json()
        return httpx.Response(
            200,
            json={
                "model": MODEL.slug,
                "choices": [{"message": {"content": value}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )

    async def main() -> None:
        async with Gateway(
            tmp_path / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(answer),
        ) as gateway:
            await read_student(
                workspace, record.student_key, SPEC, gateway, MODEL, BUDGET
            )

    asyncio.run(main())

    raised = {
        f.code
        for f in flags(KEY, [load(workspace, KEY, record)])
        if f.part_id == "Q4.b"
    }
    assert FlagCode.HARD_TO_READ in raised
    assert (FlagCode.LOW_RESOLUTION in raised) == (name == "scan-with-frame")


def marked_scan(path: Path, inside: int, kind: str) -> list[float]:
    """A page that is one 72 ppi image, with `inside` marks of `kind` (a filled
    path, a visible or an invisible text object) wholly inside the returned box,
    and twice as many that are not: one row across the box's top edge, one apart."""
    canvas = Canvas(str(path), pagesize=(612, 792))
    gray = Image.new("RGB", (612, 792), (235, 235, 235))
    canvas.drawImage(ImageReader(gray), 0, 0, 612, 792)
    canvas.setFont("Helvetica", 8)
    for n in range(inside):
        for x, y in ((100 + 10 * n, 600), (100 + 10 * n, 608), (100 + 10 * n, 300)):
            if kind == "path":
                canvas.rect(x, y, 4, 4, stroke=0, fill=1)
            else:
                text = canvas.beginText(x, y)
                text.setTextRenderMode(3 if kind == "invisible" else 0)
                text.textOut("x")
                canvas.drawText(text)
    canvas.save()
    # 90 to 100 + 10 * inside points across, 590 to 610 up: the first row of marks.
    right = 100 + 10 * inside
    return [90 / 612, (792 - 610) / 792, right / 612, (792 - 590) / 792]


@pytest.mark.parametrize("rotation", [0, 90])
@pytest.mark.parametrize(
    ("kind", "inside", "ceiling"),
    [
        ("path", render.VECTOR_MARKS - 1, True),
        ("path", render.VECTOR_MARKS, False),
        ("text", render.VECTOR_MARKS, False),
        ("invisible", render.VECTOR_MARKS, True),
    ],
)
def test_a_scan_keeps_its_ceiling_until_its_region_holds_enough_vector_marks(
    tmp_path: Path, rotation: int, kind: str, inside: int, ceiling: bool
) -> None:
    box = marked_scan(tmp_path / "scan.pdf", inside, kind)
    page = pdfium.PdfDocument(tmp_path / "scan.pdf")[0]
    page.set_rotation(rotation)
    shown = prep.box_to_rotated(box, rotation // 90)

    region = render.render_region(page, "other", shown, BUDGET)

    width, height = page.get_size()
    longest = max((shown[2] - shown[0]) * width, (shown[3] - shown[1]) * height)
    if ceiling:  # image pixels per pixel of the 72 ppi image
        assert region.scale == pytest.approx(BUDGET / longest)
    else:
        assert region.scale is None
    assert max(region.image.size) == pytest.approx(BUDGET, abs=1)


def test_marks_inside_a_form_xobject_count_where_the_page_shows_them(
    tmp_path: Path,
) -> None:
    canvas = Canvas(str(tmp_path / "form.pdf"), pagesize=(612, 792))
    gray = Image.new("RGB", (612, 792), (235, 235, 235))
    canvas.drawImage(ImageReader(gray), 0, 0, 612, 792)
    canvas.beginForm("ink")  # pyright: ignore[reportUnknownMemberType]
    for n in range(render.VECTOR_MARKS):
        canvas.rect(10 * n, 0, 4, 4, stroke=0, fill=1)
    canvas.endForm()  # pyright: ignore[reportUnknownMemberType]
    canvas.translate(100, 600)  # the marks' own space is not the page's
    canvas.doForm("ink")  # pyright: ignore[reportUnknownMemberType]
    canvas.save()
    right = 100 + 10 * render.VECTOR_MARKS
    box = [90 / 612, (792 - 610) / 792, right / 612, (792 - 590) / 792]
    page = pdfium.PdfDocument(tmp_path / "form.pdf")[0]

    assert render.render_region(page, "other", box, BUDGET).scale is None


def test_the_scan_s_own_frame_and_page_number_leave_its_whole_page_the_ceiling() -> (
    None
):
    page = pdfium.PdfDocument(INK / "scan-with-frame.pdf")[0]

    region = render.render_region(page, "other", [0, 0, 1, 1], BUDGET)

    assert region.scale == pytest.approx(BUDGET / 792 * 72 / LOW_PPI)


def worksheet(path: Path, size: tuple[int, int] = (612, 792)) -> Canvas:
    """A canvas whose page is one 72 ppi image, to draw marks on."""
    canvas = Canvas(str(path), pagesize=size)
    gray = Image.new("RGB", size, (235, 235, 235))
    canvas.drawImage(ImageReader(gray), 0, 0, *size)
    return canvas


# The box of `marked_scan`'s first row for VECTOR_MARKS marks, and of `worksheet`'s.
ROW = [90 / 612, (792 - 610) / 792, (100 + 10 * render.VECTOR_MARKS) / 612, 202 / 792]


def test_a_vector_region_rendered_below_twice_native_still_has_no_ceiling(
    tmp_path: Path,
) -> None:
    box = marked_scan(tmp_path / "scan.pdf", render.VECTOR_MARKS, "path")
    page = pdfium.PdfDocument(tmp_path / "scan.pdf")[0]

    assert render.render_region(page, "other", box, 200).scale is None


def test_a_page_with_no_image_is_never_searched_for_vector_marks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canvas = Canvas(str(tmp_path / "ink.pdf"), pagesize=(612, 792))
    for n in range(render.VECTOR_MARKS):
        canvas.rect(100 + 10 * n, 600, 4, 4, stroke=0, fill=1)
    canvas.save()
    page = pdfium.PdfDocument(tmp_path / "ink.pdf")[0]

    def counted(*_: object) -> bool:
        raise AssertionError("the marks of a page with no image were counted")

    monkeypatch.setattr(render, "_drawn", counted)

    assert render.render_region(page, "other", ROW, BUDGET).scale is None


def test_marks_inside_a_form_inside_a_form_count_where_the_page_shows_them(
    tmp_path: Path,
) -> None:
    canvas = worksheet(tmp_path / "forms.pdf")
    canvas.beginForm("inner")  # pyright: ignore[reportUnknownMemberType]
    for n in range(render.VECTOR_MARKS):
        canvas.rect(10 * n, 0, 4, 4, stroke=0, fill=1)
    canvas.endForm()  # pyright: ignore[reportUnknownMemberType]
    canvas.beginForm("outer")  # pyright: ignore[reportUnknownMemberType]
    canvas.translate(30, 20)  # each form's space is its own
    canvas.doForm("inner")  # pyright: ignore[reportUnknownMemberType]
    canvas.endForm()  # pyright: ignore[reportUnknownMemberType]
    canvas.translate(70, 580)
    canvas.doForm("outer")  # pyright: ignore[reportUnknownMemberType]
    canvas.save()
    page = pdfium.PdfDocument(tmp_path / "forms.pdf")[0]

    assert render.render_region(page, "other", ROW, BUDGET).scale is None


def test_images_inside_a_region_are_not_vector_marks(tmp_path: Path) -> None:
    canvas = worksheet(tmp_path / "stamps.pdf")
    stamp = ImageReader(Image.new("RGB", (8, 8), (0, 0, 0)))
    for n in range(render.VECTOR_MARKS):
        canvas.drawImage(stamp, 100 + 10 * n, 600, 4, 4)
    canvas.save()
    page = pdfium.PdfDocument(tmp_path / "stamps.pdf")[0]

    assert render.render_region(page, "other", ROW, BUDGET).scale is not None


def test_a_mark_counts_only_with_all_four_corners_inside(tmp_path: Path) -> None:
    # Each mark a square turned 45 degrees by its form: a diamond whose left and
    # right corners are inside a thin box, its top and bottom corners outside.
    canvas = worksheet(tmp_path / "diamonds.pdf")
    canvas.beginForm("square")  # pyright: ignore[reportUnknownMemberType]
    canvas.rect(0, 0, 4, 4, stroke=0, fill=1)
    canvas.endForm()  # pyright: ignore[reportUnknownMemberType]
    for n in range(render.VECTOR_MARKS):
        canvas.saveState()
        canvas.translate(100 + 10 * n, 600)
        canvas.rotate(-45)
        canvas.doForm("square")  # pyright: ignore[reportUnknownMemberType]
        canvas.restoreState()
    canvas.save()
    page = pdfium.PdfDocument(tmp_path / "diamonds.pdf")[0]
    thin = [95 / 612, (792 - 601) / 792, 270 / 612, (792 - 599) / 792]
    whole = [95 / 612, (792 - 604) / 792, 270 / 612, (792 - 596) / 792]

    assert render.render_region(page, "other", thin, BUDGET).scale is not None
    assert render.render_region(page, "other", whole, BUDGET).scale is None


def test_a_mark_on_the_edge_of_the_region_is_inside_it(tmp_path: Path) -> None:
    # Sizes in powers of two, so that the box's edges are the marks' exactly.
    side, n = 1024, render.VECTOR_MARKS
    canvas = worksheet(tmp_path / "edges.pdf", (side, side))
    for i in range(n):
        canvas.rect(128 + 8 * i, 512, 4, 4, stroke=0, fill=1)
    canvas.save()
    page = pdfium.PdfDocument(tmp_path / "edges.pdf")[0]
    right = 128 + 8 * (n - 1) + 4
    box = [128 / side, (side - 516) / side, right / side, (side - 512) / side]

    assert render.render_region(page, "other", box, BUDGET).scale is None
