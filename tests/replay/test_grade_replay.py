"""Grading each fixture submission with recorded calls to the real grader model.

The inputs are the fixtures' own, never another stage's recording: `key.json` as
the key, the truth file's readings as the stored readings, and their crops built
from the truth's map with the read stage's image function, on the session's fresh
ingest. Each layout class has its own recording,
`tests/recordings/grade_<class>.jsonl`. Re-record (paid) with
`REDPEN_RECORD=1 uv run pytest tests/replay/test_grade_replay.py`.
"""

import asyncio
import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pytest

from redpen.config import Config
from redpen.grade.grade import grade_student
from redpen.grade.ingest import IngestSummary
from redpen.grade.read import problem_of, unit_images, units
from redpen.llm.client import Gateway
from redpen.schemas.common import Provenance
from redpen.schemas.grade import GradeRecord
from redpen.schemas.key import Key
from redpen.schemas.reading import Crop, ProblemReadings, ReadingRecord
from redpen.schemas.spec import Spec
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, SUBMISSIONS, SubmissionTruth, map_of, truth_of
from tests.pinned import GRADER

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
SPEC = read_model(FIXTURES / "spec.json", Spec)
NOBODY = Provenance(key_part_hash=None, prompt_version=None, model_slug=None)


def store_truth_readings(
    workspace: Workspace, student: str, truth: SubmissionTruth
) -> None:
    """Write the truth's readings as the read stage would have, with the crops it
    would have read them from."""
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


@pytest.mark.parametrize("pdf", SUBMISSIONS, ids=lambda p: truth_of(p).layout_class)
def test_each_fixture_grades_as_its_truth_says(
    pdf: Path,
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    shared, summary = fixture_workspace
    truth = truth_of(pdf)
    [record] = [r for r in summary.students if r.student_name == truth.student_name]
    student = record.student_key
    workspace = Workspace(tmp_path / "ws")
    shutil.copytree(shared.student_dir(student), workspace.student_dir(student))
    store_truth_readings(workspace, student, truth)

    async def main() -> dict[str, GradeRecord]:
        async with replay(f"grade_{truth.layout_class}") as gateway:
            return await grade_student(workspace, student, KEY, gateway, GRADER)

    records = asyncio.run(main())

    unreadable = {r.part_id for r in truth.readings if r.status == "unreadable"}
    assert not unreadable & set(records)
    assert set(records) == {g.part_id for g in truth.grades}
    for want in truth.grades:
        got = {c.id: c.awarded for c in records[want.part_id].grade.criteria}
        assert got == {c.id: c.awarded for c in want.criteria}, want.part_id
