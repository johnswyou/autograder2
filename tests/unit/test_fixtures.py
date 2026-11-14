"""The synthetic fixtures written by scripts/make_fixtures.py."""

import json
import os
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from PIL import Image, ImageChops

from redpen.prepare.parse import check
from redpen.schemas.flags import FlagCode
from redpen.schemas.key import Key
from redpen.schemas.spec import Spec
from tests.fixture_truth import (
    FIXTURES,
    HOSTILE,
    HOSTILE_SUBMISSIONS,
    QUIZ,
    QUIZ_SUBMISSIONS,
    SUBMISSIONS,
    truth_of,
)
from tests.real_data import submission_folders

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
QUIZ_KEY = Key.model_validate_json((QUIZ / "key.json").read_text())
HOSTILE_KEY = Key.model_validate_json((HOSTILE / "key.json").read_text())
TITLES = {QUIZ: "Quiz 7 - Rates", HOSTILE: "Quiz 8 - Area"}


def source_kind(pdf: Path) -> str:
    return json.loads(pdf.with_suffix(".json").read_text())["source_kind"]


CLASSKICK = [p for p in SUBMISSIONS if source_kind(p) == "classkick"]
SLIDE_BOX = (200.0, 8.88, 591.56, 792.0)
SIDEBAR = [
    ("Student", 15.0, 764.82, 10.0),
    ("Assignment", 15.0, 741.7, 10.0),
    ("Page", 15.0, 718.58, 10.0),
    ("Class Code", 15.0, 695.46, 10.0),
    ("Printed", 15.0, 86.26, 8.0),
]
BLANK_SIDEBAR = [
    ("Assignment", 15.0, 753.26, 10.0),
    ("Page", 15.0, 730.14, 10.0),
    ("Printed", 15.0, 86.26, 8.0),
]
KNOWN_PREFIXES = {prefix for prefix, *_ in SIDEBAR}


def structure(page: Any) -> tuple[list[Any], list[Any]]:
    """Each image's pixel size, bounds, soft mask and raw corner pixel, and each text
    line's prefix, origin and font size. A prefix stops before any name or title,
    and unknown text shows as "<other>", so nothing a student wrote is returned."""
    images, lines = [], []
    for obj in page.get_objects(textpage=page.get_textpage()):
        if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            rendered = obj.get_bitmap(render=True).to_pil()
            masked = "A" in rendered.getbands() and rendered.getextrema()[3][0] == 0
            corner = obj.get_bitmap(render=False).to_pil().getpixel((0, 0))
            bounds = tuple(round(v, 2) for v in obj.get_bounds())
            images.append((obj.get_px_size(), bounds, masked, corner))
        elif obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
            text = obj.extract()
            prefix = text.split(":")[0] if ":" in text else text.split(" ")[0]
            *_, x, y = obj.get_matrix().get()
            known = prefix if prefix in KNOWN_PREFIXES else "<other>"
            lines.append((known, round(x, 2), round(y, 2), obj.get_font_size()))
    return images, lines


def slides(pdf_path: Path) -> list[bytes]:
    pdf = pdfium.PdfDocument(pdf_path)
    out = []
    for page in pdf:
        [image] = page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])
        assert isinstance(image, pdfium.PdfImage)
        out.append(image.get_bitmap(render=True).to_pil().tobytes())
    return out


@pytest.mark.parametrize(
    "path", [FIXTURES / "blank_classkick.pdf", *CLASSKICK], ids=lambda p: p.stem
)
def test_classkick_fixture_pages_copy_the_measured_layout(path: Path) -> None:
    pdf = pdfium.PdfDocument(path)
    blank = path.name == "blank_classkick.pdf"
    assert len(pdf) == max(p.home_page for p in KEY.parts)  # untouched slides too
    for k, page in enumerate(pdf, 1):
        images, lines = structure(page)
        assert images == [((1004, 2008), SLIDE_BOX, True, (0, 0, 0))]
        assert lines == (BLANK_SIDEBAR if blank else SIDEBAR)
        assert f"Page {k} / {len(pdf)}" in page.get_textpage().get_text_range()
    if not blank:
        name = truth_of(path).student_name
        assert f"Student: {name}" in pdf[0].get_textpage().get_text_range()


def test_the_fixture_spec_agrees_with_the_key_and_passes_the_spec_checks() -> None:
    spec = Spec.model_validate_json((FIXTURES / "spec.json").read_text())
    fields = ("id", "label", "prompt", "response_type", "home_page", "depends_on")

    assert [p.model_dump(include=set(fields)) for p in spec.parts] == [
        p.model_dump(include=set(fields)) for p in KEY.parts
    ]
    assert check(spec, 4) == []
    assert {tuple(p.options) for p in spec.parts[:3]} == {tuple("ABCDE")}


def test_blank_submission_slides_equal_the_blank_export() -> None:
    [blank] = [p for p in SUBMISSIONS if truth_of(p).layout_class == "blank"]
    assert slides(blank) == slides(FIXTURES / "blank_classkick.pdf")


@pytest.mark.parametrize(
    "path", [*QUIZ_SUBMISSIONS, *HOSTILE_SUBMISSIONS], ids=lambda p: p.stem
)
def test_the_quiz_s_pages_copy_the_measured_layout_with_its_own_title(
    path: Path,
) -> None:
    pdf = pdfium.PdfDocument(path)
    assert len(pdf) == 2
    for k, page in enumerate(pdf, 1):
        images, lines = structure(page)
        assert images == [((1004, 2008), SLIDE_BOX, True, (0, 0, 0))]
        assert lines == SIDEBAR
        text = page.get_textpage().get_text_range()
        assert f"Page {k} / 2" in text
        assert f"Assignment: {TITLES[path.parent]}" in text
    assert (
        f"Student: {truth_of(path).student_name}"
        in pdf[0].get_textpage().get_text_range()
    )


@pytest.mark.parametrize("folder", [QUIZ, HOSTILE], ids=lambda f: f.name)
def test_the_quiz_spec_agrees_with_its_key_and_passes_the_spec_checks(
    folder: Path,
) -> None:
    spec = Spec.model_validate_json((folder / "spec.json").read_text())
    key = Key.model_validate_json((folder / "key.json").read_text())
    fields = {"id", "label", "prompt", "response_type", "home_page", "depends_on"}

    assert [p.model_dump(include=fields) for p in spec.parts] == [
        p.model_dump(include=fields) for p in key.parts
    ]
    assert check(spec, 2) == []


def test_one_quiz_student_signs_the_name_line_one_a_corner_of_page_2_one_nowhere() -> (
    None
):
    names = {
        truth_of(p).layout_class: truth_of(p).map.set_aside for p in QUIZ_SUBMISSIONS
    }

    assert names.keys() == {"name-on-name-line", "name-in-corner", "no-name"}
    [line], [corner] = names["name-on-name-line"], names["name-in-corner"]
    assert line.page == 1 and line.box[1] < 0.15
    assert corner.page == 2 and corner.box[0] > 0.5 and corner.box[3] < 0.1
    assert names["no-name"] == []


@pytest.mark.parametrize(
    "path",
    [*SUBMISSIONS, *QUIZ_SUBMISSIONS, *HOSTILE_SUBMISSIONS],
    ids=lambda p: p.stem,
)
def test_truth_names_key_ids_and_awards_within_points(path: Path) -> None:
    truth = truth_of(path)
    key = {QUIZ: QUIZ_KEY, HOSTILE: HOSTILE_KEY}.get(path.parent, KEY)
    parts = {p.id: p for p in key.parts}
    assert [r.part_id for r in truth.readings] == list(parts)
    named = [e.part_id for e in truth.map.entries] + truth.map.unanswered
    assert set(named) <= set(parts)
    blank = {r.part_id for r in truth.readings if r.status == "blank"}
    assert set(truth.map.unanswered) == blank
    assert all(e.page <= len(truth.pages) for e in truth.map.entries)
    if any(r.status == "unreadable" for r in truth.readings):
        assert FlagCode.UNREADABLE in truth.flags
    if any(g.code_overruled for g in truth.grades):
        assert FlagCode.CODE_OVERRULED in truth.flags
    if any(r.addressed_to_grader for r in truth.readings):
        assert FlagCode.ADDRESSED_TO_GRADER in truth.flags
    for grade in truth.grades:
        points = {c.id: c.points for c in parts[grade.part_id].criteria}
        assert [c.id for c in grade.criteria] == list(points)
        assert all(c.awarded <= points[c.id] for c in grade.criteria)


def test_each_hostile_student_writes_to_the_grader_as_designed() -> None:
    truths = {truth_of(p).layout_class: truth_of(p) for p in HOSTILE_SUBMISSIONS}
    noted = {
        (layout, r.part_id)
        for layout, t in truths.items()
        for r in t.readings
        if r.addressed_to_grader
    }

    assert noted == {
        ("order-to-grader", "Q1"),
        ("plea-for-credit", "Q1"),
        ("plea-for-credit", "Q2"),
        ("imitates-the-system", "Q2"),
        ("statement-only", "Q1"),
    }
    # The lines to the grader come last, below the work; a statement is all there is.
    for t in truths.values():
        for r in t.readings:
            count = len(r.addressed_to_grader)
            assert r.addressed_to_grader == list(
                range(len(r.lines) - count + 1, len(r.lines) + 1)
            )
    [statement, blank] = truths["statement-only"].readings
    assert statement.addressed_to_grader == [1, 2] and blank.status == "blank"
    # The control's work and score are the plea's, without its lines to the grader.
    plea, control = truths["plea-for-credit"], truths["no-note"]
    for asked, bare in zip(plea.readings, control.readings, strict=True):
        count = len(asked.addressed_to_grader)
        assert asked.lines[: len(asked.lines) - count] == bare.lines
        assert asked.final_answer == bare.final_answer
    # Scored on the work alone: full for the right work, nothing for the rest.
    full = {c.id: c.points for p in HOSTILE_KEY.parts for c in p.criteria}
    for layout, t in truths.items():
        awards = {c.id: c.awarded for g in t.grades for c in g.criteria}
        right = layout in ("plea-for-credit", "no-note")
        assert awards == (full if right else dict.fromkeys(full, 0)), layout


def slide(pdf: Path, page: int) -> Image.Image:
    [image] = pdfium.PdfDocument(pdf)[page - 1].get_objects(
        filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]
    )
    assert isinstance(image, pdfium.PdfImage)
    return image.get_bitmap(render=True).to_pil()


def pixels(box: list[float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return round(x0 * 1004), round(y0 * 2008), round(x1 * 1004), round(y1 * 2008)


def test_the_pair_s_pages_differ_only_by_the_plea_written_below_the_same_work() -> None:
    [plea_pdf] = [p for p in HOSTILE_SUBMISSIONS if "Juno Fauxley" in p.name]
    [bare_pdf] = [p for p in HOSTILE_SUBMISSIONS if "Ulric Notreal" in p.name]
    plea, bare = truth_of(plea_pdf).map.entries, truth_of(bare_pdf).map.entries

    for asked, work in zip(plea, bare, strict=True):
        changed = ImageChops.difference(
            slide(plea_pdf, work.page).convert("RGB"),
            slide(bare_pdf, work.page).convert("RGB"),
        ).getbbox()
        assert changed is not None
        left, top, right, bottom = pixels(asked.box)
        assert left <= changed[0] and changed[2] <= right
        assert pixels(work.box)[3] < changed[1] and changed[3] <= bottom


def bands(image: Image.Image) -> list[tuple[int, int]]:
    """The runs of rows of `image` that hold ink, as (first row, row after)."""
    ink = image.convert("L").point([255] * 160 + [0] * 96)  # darker than 160
    rows = [
        ink.crop((0, y, ink.width, y + 1)).getbbox() is not None
        for y in range(ink.height)
    ]
    found: list[tuple[int, int]] = []
    for y, ink in enumerate(rows):
        if ink and (y == 0 or not rows[y - 1]):
            found.append((y, y))
        if ink:
            found[-1] = (found[-1][0], y + 1)
    return found


@pytest.mark.parametrize("pdf", HOSTILE_SUBMISSIONS, ids=lambda p: p.stem)
def test_each_line_of_the_hostile_truth_is_a_line_of_ink_in_its_part_s_box(
    pdf: Path,
) -> None:
    truth = truth_of(pdf)
    readings = {r.part_id: r for r in truth.readings}

    for entry in truth.map.entries:
        rows = bands(slide(pdf, entry.page).crop(pixels(entry.box)))
        reading = readings[entry.part_id]
        assert len(rows) == len(reading.lines), entry.part_id
        # Lines to the grader below work: the widest gap is the one before them.
        if reading.addressed_to_grader and len(reading.addressed_to_grader) < len(rows):
            gaps = [b[0] - a[1] for a, b in zip(rows, rows[1:], strict=False)]
            first = reading.addressed_to_grader[0]
            assert gaps.index(max(gaps)) == first - 2, entry.part_id


@pytest.mark.skipif(
    not os.environ.get("REDPEN_REAL_DATA"), reason="REDPEN_REAL_DATA is not set"
)
def test_fixture_page_has_the_structure_of_a_real_export() -> None:
    for pdf in (p for f in submission_folders() for p in sorted(f.glob("*.pdf"))):
        real = structure(pdfium.PdfDocument(pdf)[0])
        if real[1] and real[1][0][0] == "Student":
            break
    else:
        pytest.fail("no Classkick export under REDPEN_REAL_DATA")
    assert real == structure(pdfium.PdfDocument(CLASSKICK[0])[0])
