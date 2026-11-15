"""The second grade of each fixture submission, with recorded calls to the real
grader (`tests.pinned.GRADER`).

The inputs are the fixtures' own, as the grade replay's are: `key.json` as the
key, the truth file's readings as the stored readings with their crops built from
the truth's map, on the session's fresh ingest; and as the first grade, the
truth's grades, stored as the grade stage stores a grade made by the pinned grader
under the grade prompt now. So each second grade's request is the grade stage's
own request for those readings, at the second draw. Its answers are
`tests/recordings/second_grade_class.jsonl`, one call for each problem a model
grades. It was made by running this seed and `grade_again` live into a persistent
scratch cache and keeping exactly the calls made. Under
`AUTOGRADER2_RECORD=1` the `replay` fixture would call the grader again (paid) and
rewrite it once the test passes."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from autograder2.grade.grade import by_problem, graded_from, request
from autograder2.grade.ingest import IngestSummary
from autograder2.grade.second_grade import grade_again
from autograder2.llm.client import Gateway
from autograder2.prepare.key import part_hash
from autograder2.schemas.common import Provenance
from autograder2.schemas.grade import (
    AnswerCheck,
    GradeRecord,
    ProblemGrades,
    SecondGrade,
)
from autograder2.workspace import Workspace, write_model
from tests.fixture_truth import (
    SUBMISSIONS,
    SubmissionTruth,
    store_truth_readings,
    truth_of,
)
from tests.pinned import GRADER
from tests.replay.test_grade_replay import KEY, student_of


def store_truth_grades(
    workspace: Workspace, student: str, truth: SubmissionTruth
) -> dict[str, GradeRecord]:
    """The truth's grades, stored as the grade stage stores them from the stored
    readings: a part code scores as code scores it, and every other one as the
    pinned grader's grade under the grade prompt now, with the answer check of its
    reading. Returns every record, by part id."""
    grades = {g.part_id: g for g in truth.grades}
    stored: dict[str, GradeRecord] = {}
    for problem, parts in by_problem(KEY):
        if (found := request(workspace, student, KEY, parts)) is None:
            continue
        records = dict(found.by_code)
        for p in found.asked:
            verdict = found.checks[p.id]
            records[p.id] = GradeRecord(
                provenance=Provenance(
                    key_part_hash=part_hash(p, KEY.policies),
                    prompt_version=found.version,
                    model_slug=GRADER.slug,
                ),
                grade=grades[p.id],
                check=AnswerCheck(**verdict._asdict()) if verdict else None,
                reading_digest=graded_from(p, found.shown),
            )
        mine = [records[p.id] for p in parts if p.id in records]
        write_model(workspace.grade_json(student, problem), ProblemGrades(records=mine))
        stored |= records
    return stored


@pytest.mark.parametrize("pdf", SUBMISSIONS, ids=lambda p: truth_of(p).layout_class)
def test_each_fixture_is_graded_again_and_the_grader_agrees_with_its_truth(
    pdf: Path,
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    truth = truth_of(pdf)
    workspace, student = student_of(fixture_workspace, truth, tmp_path / "ws")
    store_truth_readings(workspace, student, truth)
    first = store_truth_grades(workspace, student, truth)

    async def main() -> list[SecondGrade]:
        async with replay("second_grade_class") as gateway:
            return await grade_again(workspace, student, KEY, gateway, GRADER)

    found = asyncio.run(main())

    # Every part a model grades, and none code scores, each answered from the
    # recording and the same as the truth's grade.
    by_model = [i for i, r in first.items() if r.provenance.model_slug]
    assert [r.part_id for r in found] == [p.id for p in KEY.parts if p.id in by_model]
    assert {r.result for r in found} <= {"same"}
    for r in found:
        assert r.second is not None
        got = {c.id: c.awarded for c in r.second.grade.criteria}
        assert got == {c.id: c.awarded for c in first[r.part_id].grade.criteria}
