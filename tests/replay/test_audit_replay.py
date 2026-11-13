"""The class audit over the fixture class, with recorded calls to the real grader.

The inputs are the fixtures' own: `key.json` as the key, every truth file's
readings, and each student's grades as the grade stage gives them from its own
recordings (`grade_<class>.jsonl`, whose awards are the truth's). One award is then
altered by hand: small-writing's 4(c) c1, from 1 to 0, so that seven identical
readings of 4(c) disagree. The audit's calls are `tests/recordings/audit_class.jsonl`:
two groups, the seeded one and 4(a) c2, where partly-blank's drawing rightly earns
less for the same cited arrow. It was made by running this seed live into a
persistent scratch cache and dumping exactly the calls made (task 4.7 report);
`REDPEN_RECORD=1` does not rewrite it."""

import asyncio
import shutil
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from redpen.grade.audit import audit_class
from redpen.grade.grade import grade_student
from redpen.grade.ingest import IngestSummary
from redpen.llm.client import Gateway
from redpen.report.report import load, score
from redpen.schemas.audit import PartAudit
from redpen.schemas.grade import ProblemGrades
from redpen.schemas.key import Key
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, SUBMISSIONS, store_truth_readings, truth_of
from tests.pinned import GRADER

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
SEEDED = ("small-writing", "Q4.c.c1")


def test_the_audit_puts_the_altered_award_back_and_changes_nothing_else(
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    shared, summary = fixture_workspace
    workspace = Workspace(tmp_path / "ws")
    truths = {truth_of(p).student_name: truth_of(p) for p in SUBMISSIONS}
    layout = {
        s.student_key: truths[s.student_name].layout_class for s in summary.students
    }
    keys = [s.student_key for s in summary.students]

    async def main() -> list[PartAudit]:
        for record in summary.students:
            student, truth = record.student_key, truths[record.student_name]
            shutil.copytree(shared.student_dir(student), workspace.student_dir(student))
            store_truth_readings(workspace, student, truth)
            async with replay(f"grade_{truth.layout_class}") as gateway:
                await grade_student(workspace, student, KEY, gateway, GRADER)
        [seeded] = [s for s in keys if layout[s] == SEEDED[0]]
        path = workspace.grade_json(seeded, "Q4")
        stored = read_model(path, ProblemGrades)
        [c1] = [
            c for r in stored.records for c in r.grade.criteria if c.id == SEEDED[1]
        ]
        assert c1.awarded == 1
        c1.awarded = Decimal(0)
        write_model(path, stored)
        async with replay("audit_class") as gateway:
            return await audit_class(workspace, KEY, keys, gateway, GRADER)

    audits = {a.part_id: a for a in asyncio.run(main())}

    assert {
        p: (a.groups, a.changed, a.undecided) for p, a in audits.items() if a.groups
    } == {
        "Q4.a": (1, 0, 0),
        "Q4.c": (1, 1, 0),
    }
    [change] = audits["Q4.c"].changes
    assert (layout[change.student_key], change.criterion) == SEEDED
    assert (change.before, change.after) == (0, 1)
    assert not any(a.key_suspect for a in audits.values())
    for student in summary.students:
        want = {
            c.id: c.awarded
            for g in truths[student.student_name].grades
            for c in g.criteria
        }
        for found in score(KEY, load(workspace, KEY, student)):
            for c in found.record.grade.criteria if found.record else []:
                assert c.awarded == want[c.id], (layout[student.student_key], c.id)
