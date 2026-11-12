"""The ground truth of the synthetic fixtures written by scripts/make_fixtures.py."""

from pathlib import Path
from typing import Any, Literal

from redpen.schemas.common import Schema
from redpen.schemas.flags import FlagCode
from redpen.schemas.grade import Grade
from redpen.schemas.key import Answer
from redpen.schemas.reading import Reading
from redpen.schemas.submission import IngestPage, Map

FIXTURES = Path(__file__).parent / "fixtures" / "synthetic"
SUBMISSIONS = sorted(FIXTURES.glob("* - *.pdf"))


# The ground truth of one synthetic submission, in the JSON file beside its PDF.
# Reading lines leave out the student's own part labels such as "4b)". A part whose
# response is unreadable has no grade, because the design withholds that score.
class SubmissionTruth(Schema):
    layout_class: Literal[
        "typed-in-place",
        "ink-in-place",
        "photo-per-page",
        "all-on-foreign-pages",
        "rotated",
        "partly-blank",
        "blank",
        "letters-listed",
        "small-writing",
        "low-resolution",
        "not-classkick",
    ]
    student_name: str
    source_kind: Literal["classkick", "other"]
    pages: list[IngestPage]
    # Per page, the clockwise degrees that turn the student's work upright.
    rotations: list[Literal[0, 90, 180, 270]]
    map: Map
    readings: list[Reading]
    grades: list[Grade]
    flags: list[FlagCode]


def truth_of(pdf: Path) -> SubmissionTruth:
    return SubmissionTruth.model_validate_json(pdf.with_suffix(".json").read_text())


def answer(**fields: Any) -> Answer:
    """An `Answer` with `fields` filled and every other field null or empty."""
    lists = {"choices", "accept_also", "elements", "must_include"}
    return Answer.model_validate(
        {name: [] if name in lists else None for name in Answer.model_fields} | fields
    )
