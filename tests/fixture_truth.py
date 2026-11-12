"""The ground truth of the synthetic fixtures written by scripts/make_fixtures.py."""

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from redpen.config import Config
from redpen.grade.read import problem_of, unit_images, units
from redpen.schemas.common import Provenance, Schema
from redpen.schemas.flags import FlagCode
from redpen.schemas.grade import Grade
from redpen.schemas.key import Answer
from redpen.schemas.reading import Crop, ProblemReadings, Reading, ReadingRecord
from redpen.schemas.spec import Spec
from redpen.schemas.submission import IngestPage, Map, MapPage, MapRecord
from redpen.workspace import Workspace, read_model, write_model

FIXTURES = Path(__file__).parent / "fixtures" / "synthetic"
SUBMISSIONS = sorted(FIXTURES.glob("* - *.pdf"))
SPEC = read_model(FIXTURES / "spec.json", Spec)
NOBODY = Provenance(key_part_hash=None, prompt_version=None, model_slug=None)


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
    # Per page, whether the student's work on it is a photograph or scan of paper.
    photos: list[bool]
    map: Map
    readings: list[Reading]
    grades: list[Grade]
    flags: list[FlagCode]


def truth_of(pdf: Path) -> SubmissionTruth:
    return SubmissionTruth.model_validate_json(pdf.with_suffix(".json").read_text())


def map_of(truth: SubmissionTruth) -> MapRecord:
    """The map record a correct map stage would write for the truth's submission:
    its map, and each page's rotation and photo flag, as written by code."""
    pages = [
        MapPage(number=n, rotation=rotation, photo=photo)
        for n, (rotation, photo) in enumerate(
            zip(truth.rotations, truth.photos, strict=True), 1
        )
    ]
    assert len(pages) == len(truth.pages)
    return MapRecord(provenance=NOBODY, pages=pages, map=truth.map)


def store_truth_readings(
    workspace: Workspace, student: str, truth: SubmissionTruth
) -> None:
    """Write the truth's readings as the read stage would have, with the crops it
    would have read them from (built from the truth's map, no model call)."""
    mapped = map_of(truth)
    crops: dict[str, list[Crop]] = {}
    for unit in units(SPEC, mapped.map):
        images = unit_images(
            workspace, student, unit, mapped.pages, Config().image_budget
        )
        crops |= images.crops
    files: defaultdict[str, list[ReadingRecord]] = defaultdict(list)
    for r in truth.readings:
        record = ReadingRecord(
            provenance=NOBODY, reading=r, crops=crops.get(r.part_id, [])
        )
        files[problem_of(r.part_id)].append(record)
    for problem, records in files.items():
        write_model(
            workspace.reading_json(student, problem), ProblemReadings(records=records)
        )


def answer(**fields: Any) -> Answer:
    """An `Answer` with `fields` filled and every other field null or empty."""
    lists = {"choices", "accept_also", "elements", "must_include"}
    return Answer.model_validate(
        {name: [] if name in lists else None for name in Answer.model_fields} | fields
    )
