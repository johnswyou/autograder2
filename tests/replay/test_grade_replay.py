"""Grading each fixture submission with recorded calls to the real grader model.

The inputs are the fixtures' own, never another stage's recording: `key.json` as
the key, the truth file's readings as the stored readings, and their crops built
from the truth's map with the read stage's image function, on the session's fresh
ingest. Each layout class has its own recording,
`tests/recordings/grade_<class>.jsonl`; the case of a later part worked afresh has
`grade_fresh-route.jsonl`. Re-record (paid) with
`REDPEN_RECORD=1 uv run pytest tests/replay/test_grade_replay.py`.
"""

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from redpen.grade.grade import grade_student
from redpen.grade.ingest import IngestSummary
from redpen.grade.read import problem_of
from redpen.llm.client import Gateway
from redpen.schemas.grade import GradeRecord
from redpen.schemas.key import Key
from redpen.schemas.reading import FinalAnswer, ProblemReadings, Reading, ReadingRecord
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import (
    FIXTURES,
    NOBODY,
    SUBMISSIONS,
    SubmissionTruth,
    store_truth_readings,
    truth_of,
)
from tests.pinned import GRADER

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())

# 5(b) worked afresh from the question, reaching the key's value without the wrong
# 5(a) result it could have carried.
AFRESH = Reading(
    part_id="Q5.b",
    status="answered",
    medium="handwritten",
    lines=["x = -b/(2a) = 6/2 = 3", "f(3) = 3^2 - 6(3) + 5 = -4", "min = -4"],
    final_answer=FinalAnswer(choices=[], value=-4.0, unit=None, expression=None),
    elements=[],
    uncertain=[],
    addressed_to_grader=[],
    legibility="clear",
)


def student_of(
    fixture_workspace: tuple[Workspace, IngestSummary],
    truth: SubmissionTruth,
    root: Path,
) -> tuple[Workspace, str]:
    """A workspace holding the truth's ingested student, to write into."""
    shared, summary = fixture_workspace
    [record] = [r for r in summary.students if r.student_name == truth.student_name]
    workspace = Workspace(root)
    shutil.copytree(
        shared.student_dir(record.student_key),
        workspace.student_dir(record.student_key),
    )
    return workspace, record.student_key


def store_afresh(workspace: Workspace, student: str, truth: SubmissionTruth) -> None:
    """The truth's readings with 5(b) replaced by `AFRESH`, stored with no crops, so
    that no image contradicts it."""
    store_truth_readings(workspace, student, truth)
    path = workspace.reading_json(student, "Q5")
    fresh = ReadingRecord(provenance=NOBODY, reading=AFRESH, crops=[])
    records = [
        fresh if r.reading.part_id == "Q5.b" else r
        for r in read_model(path, ProblemReadings).records
    ]
    write_model(path, ProblemReadings(records=records))


@pytest.mark.parametrize("pdf", SUBMISSIONS, ids=lambda p: truth_of(p).layout_class)
def test_each_fixture_grades_as_its_truth_says(
    pdf: Path,
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    truth = truth_of(pdf)
    workspace, student = student_of(fixture_workspace, truth, tmp_path / "ws")
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


def test_a_marked_criterion_is_met_by_the_key_s_value_reached_afresh(
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    # The other side of the carry-forward case: the same wrong 5(a), and a 5(b)
    # that reaches the key's value by its own route instead of carrying it.
    [truth] = [
        t
        for t in map(truth_of, SUBMISSIONS)
        if t.layout_class == "all-on-foreign-pages"
    ]
    workspace, student = student_of(fixture_workspace, truth, tmp_path / "ws")
    store_afresh(workspace, student, truth)
    q5 = Key(
        policies=KEY.policies,
        parts=[p for p in KEY.parts if problem_of(p.id) == "Q5"],
    )

    async def main() -> dict[str, GradeRecord]:
        async with replay("grade_fresh-route") as gateway:
            return await grade_student(workspace, student, q5, gateway, GRADER)

    records = asyncio.run(main())

    [q5b] = [p for p in KEY.parts if p.id == "Q5.b"]
    assert all(c.carry_forward for c in q5b.criteria)
    got = {c.id: c.awarded for c in records["Q5.b"].grade.criteria}
    assert got == {c.id: c.points for c in q5b.criteria}
