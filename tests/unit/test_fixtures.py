"""The synthetic fixtures written by scripts/make_fixtures.py."""

import json
import os
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest

from redpen.prepare.parse import check
from redpen.schemas.flags import FlagCode
from redpen.schemas.key import Key
from redpen.schemas.spec import Spec
from tests.fixture_truth import (
    FIXTURES,
    QUIZ,
    QUIZ_SUBMISSIONS,
    SUBMISSIONS,
    truth_of,
)
from tests.real_data import submission_folders

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
QUIZ_KEY = Key.model_validate_json((QUIZ / "key.json").read_text())


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


@pytest.mark.parametrize("path", QUIZ_SUBMISSIONS, ids=lambda p: p.stem)
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
        assert f"Page {k} / 2" in text and "Assignment: Quiz 7 - Rates" in text
    assert (
        f"Student: {truth_of(path).student_name}"
        in pdf[0].get_textpage().get_text_range()
    )


def test_the_quiz_spec_agrees_with_its_key_and_passes_the_spec_checks() -> None:
    spec = Spec.model_validate_json((QUIZ / "spec.json").read_text())
    fields = {"id", "label", "prompt", "response_type", "home_page", "depends_on"}

    assert [p.model_dump(include=fields) for p in spec.parts] == [
        p.model_dump(include=fields) for p in QUIZ_KEY.parts
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
    "path", [*SUBMISSIONS, *QUIZ_SUBMISSIONS], ids=lambda p: p.stem
)
def test_truth_names_key_ids_and_awards_within_points(path: Path) -> None:
    truth = truth_of(path)
    key = QUIZ_KEY if path.parent == QUIZ else KEY
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
    for grade in truth.grades:
        points = {c.id: c.points for c in parts[grade.part_id].criteria}
        assert [c.id for c in grade.criteria] == list(points)
        assert all(c.awarded <= points[c.id] for c in grade.criteria)


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
