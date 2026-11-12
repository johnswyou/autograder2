"""The ground truth of the synthetic fixtures written by scripts/make_fixtures.py."""

from pathlib import Path
from typing import Literal

from redpen.schemas.common import Schema
from redpen.schemas.flags import FlagCode
from redpen.schemas.grade import Grade
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
    map: Map
    readings: list[Reading]
    grades: list[Grade]
    flags: list[FlagCode]


def truth_of(pdf: Path) -> SubmissionTruth:
    return SubmissionTruth.model_validate_json(pdf.with_suffix(".json").read_text())
