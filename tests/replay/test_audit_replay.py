"""The class audit over the fixture class, with recorded calls to the real grader.

The inputs are the fixture truth files': `key.json` as the key, each truth file's
readings, and its grades stored as the grade stage stores them. The truth's grades
cite no evidence, so each criterion cites the line or element `CITED` names (when
the reading has it), and each part a model would grade carries the answer check
the grade stage would have stored. One award is then altered by hand:
small-writing's 4(c) c1, from 1 to 0, so that seven identical readings of 4(c)
disagree. The audit's calls are `tests/recordings/audit_class.jsonl`: the seeded
group, and 4(a) c2, where partly-blank's drawing rightly earns less for the same
cited arrow. It was made by running this seed live into a persistent scratch cache
and dumping exactly the calls made (task 4.7 report, fix round 1);
`REDPEN_RECORD=1` does not rewrite it."""

import asyncio
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

from redpen.grade.audit import audit_class
from redpen.grade.check import check_answer
from redpen.grade.ingest import IngestSummary
from redpen.grade.read import problem_of
from redpen.llm.client import Gateway
from redpen.prepare.key import part_hash
from redpen.report.report import load, score
from redpen.schemas.audit import PartAudit
from redpen.schemas.common import Provenance
from redpen.schemas.grade import AnswerCheck, GradeRecord, ProblemGrades
from redpen.schemas.key import Key
from redpen.schemas.reading import ProblemReadings, ReadingRecord
from redpen.schemas.spec import CHOICE_TYPES
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, NOBODY, SUBMISSIONS, truth_of
from tests.pinned import GRADER

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
PARTS = {p.id: p for p in KEY.parts}
# The line or element, counting from 1, that each model-decided criterion cites.
CITED: dict[str, tuple[str, int]] = {
    "Q4.a.c1": ("element", 2),
    "Q4.a.c2": ("element", 3),
    "Q4.b.c1": ("line", 2),
    "Q4.b.c2": ("line", 3),
    "Q4.c.c1": ("line", 1),
    "Q4.c.c2": ("line", 4),
    "Q5.a.c1": ("line", 2),
    "Q5.b.c1": ("line", 1),
    "Q5.b.c2": ("line", 2),
    "Q5.c.c1": ("element", 2),
    "Q5.c.c2": ("element", 3),
}
SEEDED = ("small-writing", "Q4.c.c1")


def seed(workspace: Workspace, summary: IngestSummary) -> None:
    """Store each student's truth readings and grades as the read and grade stages
    store them: a choice part or a blank reading scored by code, every other part
    by the grader, citing `CITED`, with the answer check of its reading."""
    truths = {truth_of(p).student_name: truth_of(p) for p in SUBMISSIONS}
    for student in summary.students:
        truth, who = truths[student.student_name], student.student_key
        readings = {r.part_id: r for r in truth.readings}
        for problem in dict.fromkeys(problem_of(p.id) for p in KEY.parts):
            mine = [r for r in truth.readings if problem_of(r.part_id) == problem]
            records = [
                ReadingRecord(provenance=NOBODY, reading=r, crops=[]) for r in mine
            ]
            write_model(
                workspace.reading_json(who, problem), ProblemReadings(records=records)
            )
            graded: list[GradeRecord] = []
            for g in truth.grades:
                part, reading = PARTS[g.part_id], readings[g.part_id]
                if problem_of(part.id) != problem:
                    continue
                by_code = (
                    part.response_type in CHOICE_TYPES or reading.status == "blank"
                )
                cited = [
                    {}
                    if by_code
                    else _evidence(CITED[c.id], reading.lines, reading.elements)
                    for c in g.criteria
                ]
                criteria = [
                    c.model_copy(
                        update={"evidence_line": None, "evidence_element": None} | e
                    )
                    for c, e in zip(g.criteria, cited, strict=True)
                ]
                check = None if by_code else check_answer(part, reading, KEY.policies)
                graded.append(
                    GradeRecord(
                        provenance=Provenance(
                            key_part_hash=part_hash(part, KEY.policies),
                            prompt_version=None if by_code else "truth",
                            model_slug=None if by_code else GRADER.slug,
                        ),
                        grade=g.model_copy(update={"criteria": criteria}),
                        check=AnswerCheck(**check._asdict()) if check else None,
                    )
                )
            write_model(
                workspace.grade_json(who, problem), ProblemGrades(records=graded)
            )


def _evidence(
    cited: tuple[str, int], lines: Sequence[str], elements: Sequence[object]
) -> dict[str, int]:
    what, n = cited
    have = len(lines) if what == "line" else len(elements)
    return {f"evidence_{what}": n} if n <= have else {}


def layouts(summary: IngestSummary) -> dict[str, str]:
    """Each student's layout class, by student key."""
    truths = {truth_of(p).student_name: truth_of(p).layout_class for p in SUBMISSIONS}
    return {s.student_key: truths[s.student_name] for s in summary.students}


def alter(
    workspace: Workspace, summary: IngestSummary, layout: str, criterion: str, to: str
) -> Decimal:
    """Set `criterion`'s award of the student of `layout` to `to` in their stored
    grade file, as a hand edit would; return the award it had."""
    [who] = [k for k, found in layouts(summary).items() if found == layout]
    path = workspace.grade_json(who, problem_of(criterion))
    stored = read_model(path, ProblemGrades)
    [c] = [c for r in stored.records for c in r.grade.criteria if c.id == criterion]
    before, c.awarded = c.awarded, Decimal(to)
    write_model(path, stored)
    return before


def test_the_audit_puts_the_altered_award_back_and_changes_nothing_else(
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    _, summary = fixture_workspace
    workspace = Workspace(tmp_path / "ws")
    truths = {truth_of(p).student_name: truth_of(p) for p in SUBMISSIONS}
    layout = layouts(summary)
    seed(workspace, summary)
    assert alter(workspace, summary, *SEEDED, "0") == 1

    async def main() -> list[PartAudit]:
        async with replay("audit_class") as gateway:
            return await audit_class(workspace, KEY, summary.students, gateway, GRADER)

    audits = {a.part_id: a for a in asyncio.run(main())}

    found = {
        p: (a.groups, a.changed, a.undecided) for p, a in audits.items() if a.groups
    }
    assert found == {"Q4.a": (1, 0, 0), "Q4.c": (1, 1, 0)}
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
        for found_ in score(KEY, load(workspace, KEY, student)):
            for c in found_.record.grade.criteria if found_.record else []:
                assert c.awarded == want[c.id], (layout[student.student_key], c.id)
