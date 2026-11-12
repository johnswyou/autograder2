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
from collections.abc import Callable
from pathlib import Path

import pytest

from redpen.grade.grade import grade_student
from redpen.grade.ingest import IngestSummary
from redpen.llm.client import Gateway
from redpen.schemas.grade import GradeRecord
from redpen.schemas.key import Key
from redpen.workspace import Workspace
from tests.fixture_truth import FIXTURES, SUBMISSIONS, store_truth_readings, truth_of
from tests.pinned import GRADER

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())


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
