"""The report stage: score files, feedback, flags and the run report, computed from
stored files with no model call.

The fixture students' stored files are the truth's: `key.json` as the key, each truth
file's readings (with crops built from the truth's map by the read stage's image
function) and grades, on the session's fresh ingest. They are stored and reported
once for the module; a test that changes something works on a copy."""

import csv
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from redpen.grade.ingest import IngestSummary
from redpen.grade.read import problem_of
from redpen.llm.ledger import Ledger, Usage
from redpen.prepare.key import number, page_points, part_hash
from redpen.report.report import (
    NO_OPTION,
    OTHER_KEY,
    UNGRADED,
    UNREADABLE,
    flags,
    load,
    status,
    write_report,
)
from redpen.schemas.common import Provenance
from redpen.schemas.flags import FlagCode
from redpen.schemas.grade import CriterionGrade, Grade, GradeRecord, ProblemGrades
from redpen.schemas.key import Criterion, Key, KeyPart, Policies
from redpen.schemas.reading import (
    FinalAnswer,
    ProblemReadings,
    Reading,
    ReadingRecord,
)
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import (
    FIXTURES,
    NOBODY,
    SUBMISSIONS,
    SubmissionTruth,
    answer,
    store_truth_readings,
    truth_of,
)

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
NO_ANSWER = FinalAnswer(choices=[], value=None, unit=None, expression=None)
CRITERIA = {c.id: c for p in KEY.parts for c in p.criteria}
PROBLEMS = list(dict.fromkeys(problem_of(p.id) for p in KEY.parts))
LEDGER = [
    ("reader", Usage(1200, 300, Decimal("0.0213"))),
    ("grader", Usage(900, 250, Decimal("0.0175"))),
    ("grader", Usage(800, 200, Decimal("0.0161"))),
]
LOCK = """profile = "balanced"

[models.grader]
slug = "example/grader-model"
supported_parameters = []
accepts_images = true
prompt_price = "0.000002"
completion_price = "0.00001"
providers = []
"""


def graded(grade: Grade, key: Key = KEY) -> GradeRecord:
    """`grade` as the grade stage stores it, under `key`'s version of its part."""
    [part] = [p for p in key.parts if p.id == grade.part_id]
    provenance = Provenance(
        key_part_hash=part_hash(part, key.policies),
        prompt_version=None,
        model_slug=None,
    )
    return GradeRecord(provenance=provenance, grade=grade)


def feedback_on(criterion: str) -> str:
    return f"Feedback on {criterion}."


def with_feedback(grade: Grade) -> Grade:
    """`grade` with a feedback sentence on every criterion. The truth files hold
    none, so without this the feedback test would check nothing."""
    criteria = [
        c.model_copy(update={"feedback": feedback_on(c.id)}) for c in grade.criteria
    ]
    return grade.model_copy(update={"criteria": criteria})


def store_grades(
    workspace: Workspace, student: str, records: Sequence[GradeRecord], key: Key = KEY
) -> None:
    """One grade file per top-level problem, as the grade stage writes them: a
    problem with no graded part has a file with no records."""
    for problem in dict.fromkeys(problem_of(p.id) for p in key.parts):
        mine = [r for r in records if problem_of(r.grade.part_id) == problem]
        write_model(workspace.grade_json(student, problem), ProblemGrades(records=mine))


def new_workspace(root: Path, folder: str = "ws") -> Workspace:
    workspace = Workspace(root / folder)
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    return workspace


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def raw_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.reader(file))


def cell(text: str) -> Decimal:
    """A score cell's points, an empty cell counting as none."""
    return Decimal(text) if text else Decimal(0)


@dataclass(frozen=True)
class Built:
    workspace: Workspace
    students: list[IngestRecord]
    truths: dict[str, SubmissionTruth]  # by student key


@pytest.fixture(scope="module")
def built(
    fixture_workspace: tuple[Workspace, IngestSummary],
    tmp_path_factory: pytest.TempPathFactory,
) -> Built:
    shared, summary = fixture_workspace
    workspace = new_workspace(tmp_path_factory.mktemp("report"))
    truths: dict[str, SubmissionTruth] = {}
    for pdf in SUBMISSIONS:
        truth = truth_of(pdf)
        [record] = [r for r in summary.students if r.student_name == truth.student_name]
        student = record.student_key
        shutil.copytree(shared.student_dir(student), workspace.student_dir(student))
        store_truth_readings(workspace, student, truth)
        store_grades(
            workspace, student, [graded(with_feedback(g)) for g in truth.grades]
        )
        truths[student] = truth
    with closing(sqlite3.connect(workspace.cache_sqlite, autocommit=True)) as db:
        ledger = Ledger(db)
        for role, usage in LEDGER:
            ledger.record(role, "example/model", usage)
    workspace.config_lock_toml.write_text(LOCK)
    write_report(workspace, KEY, summary.students)
    return Built(workspace, summary.students, truths)


def student_of(built: Built, layout_class: str) -> IngestRecord:
    [student] = [
        s
        for s in built.students
        if built.truths[s.student_key].layout_class == layout_class
    ]
    return student


def row_of(path: Path, student: IngestRecord) -> dict[str, str]:
    [row] = [r for r in rows(path) if r["student"] == student.student_name]
    return row


def page_columns(row: dict[str, str]) -> list[str]:
    return [name for name in row if name != "student"]


LAYOUTS = [truth_of(pdf).layout_class for pdf in SUBMISSIONS]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_page_totals_equal_problem_totals_equal_the_total(
    built: Built, layout: str
) -> None:
    student = student_of(built, layout)
    truth = built.truths[student.student_key]
    by_problem = row_of(built.workspace.scores_by_problem_csv, student)
    by_page = row_of(built.workspace.scores_by_page_csv, student)

    pages = sum((cell(by_page[n]) for n in page_columns(by_page)), Decimal(0))
    problems = sum((cell(by_problem[p]) for p in PROBLEMS), Decimal(0))
    total = Decimal(by_problem["total"])
    assert pages == problems == total
    assert total == sum(
        (c.awarded for g in truth.grades for c in g.criteria), Decimal(0)
    )
    assert by_problem["maximum"] == number(sum(page_points(KEY).values(), Decimal(0)))
    assert page_columns(by_page) == ["page 1", "page 2", "page 3", "page 4"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_cell_is_the_truth_s_sum_or_empty_where_the_truth_withholds(
    built: Built, layout: str
) -> None:
    student = student_of(built, layout)
    truth = built.truths[student.student_key]
    awarded = {
        g.part_id: sum((c.awarded for c in g.criteria), Decimal(0))
        for g in truth.grades
    }

    def want(parts: list[KeyPart]) -> str:
        if any(p.id not in awarded for p in parts):
            return ""
        return number(sum((awarded[p.id] for p in parts), Decimal(0)))

    by_problem = row_of(built.workspace.scores_by_problem_csv, student)
    by_page = row_of(built.workspace.scores_by_page_csv, student)
    for problem in PROBLEMS:
        parts = [p for p in KEY.parts if problem_of(p.id) == problem]
        assert by_problem[problem] == want(parts), problem
    for page in range(1, 5):
        parts = [p for p in KEY.parts if p.home_page == page]
        assert by_page[f"page {page}"] == want(parts), page


def test_a_part_written_on_a_foreign_page_scores_under_its_printed_page(
    built: Built,
) -> None:
    student = student_of(built, "all-on-foreign-pages")
    truth = built.truths[student.student_key]
    # Q5.b is printed on page 4; this student wrote it on a notebook photo on page 3.
    [part] = [p for p in KEY.parts if p.id == "Q5.b"]
    assert part.home_page == 4
    assert {e.page for e in truth.map.entries if e.part_id == "Q5.b"} == {3}
    assert not [e for e in truth.map.entries if e.page == 4]

    by_page = row_of(built.workspace.scores_by_page_csv, student)
    q5b, q5c = (
        sum((c.awarded for c in g.criteria), Decimal(0))
        for g in truth.grades
        if g.part_id in ("Q5.b", "Q5.c")
    )
    assert q5b > 0
    assert by_page["page 4"] == number(q5b + q5c)


def test_no_cell_exceeds_its_maximum(built: Built) -> None:
    pages = page_points(KEY)
    problems = {
        q: sum(
            (c.points for p in KEY.parts if problem_of(p.id) == q for c in p.criteria),
            Decimal(0),
        )
        for q in PROBLEMS
    }
    for row in rows(built.workspace.scores_by_page_csv):
        for page, points in pages.items():
            assert cell(row[f"page {page}"]) <= points
    for row in rows(built.workspace.scores_by_problem_csv):
        for problem, points in problems.items():
            assert cell(row[problem]) <= points


def test_unreadable_work_leaves_empty_cells_never_zero_and_the_student_incomplete(
    built: Built,
) -> None:
    student = student_of(built, "low-resolution")
    by_problem = row_of(built.workspace.scores_by_problem_csv, student)
    by_page = row_of(built.workspace.scores_by_page_csv, student)

    assert [by_problem[p] for p in PROBLEMS] == [""] * len(PROBLEMS)
    assert [by_page[n] for n in page_columns(by_page)] == [""] * 4
    assert by_problem["total"] == "0"
    assert by_problem["status"] == "incomplete"
    feedback = built.workspace.feedback_md(student.student_key).read_text()
    assert feedback.count(UNREADABLE) == len(KEY.parts)
    detail = [
        r
        for r in rows(built.workspace.scores_detail_csv)
        if r["student"] == student.student_name
    ]
    assert [(r["awarded"], r["rationale"]) for r in detail] == [("", UNREADABLE)] * len(
        CRITERIA
    )


def test_a_blank_submission_scores_zero_everywhere_and_is_not_withheld(
    built: Built,
) -> None:
    student = student_of(built, "blank")
    by_problem = row_of(built.workspace.scores_by_problem_csv, student)
    by_page = row_of(built.workspace.scores_by_page_csv, student)

    assert [by_problem[p] for p in PROBLEMS] == ["0"] * len(PROBLEMS)
    assert [by_page[n] for n in page_columns(by_page)] == ["0"] * 4
    assert (by_problem["total"], by_problem["status"]) == ("0", "final")
    assert by_problem["open_flags"] == "0"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_derived_flags_are_the_truth_s(built: Built, layout: str) -> None:
    student = student_of(built, layout)
    truth = built.truths[student.student_key]
    raised = flags(KEY, [load(built.workspace, KEY, student)])

    assert {f.code for f in raised} == set(truth.flags)
    assert all(f.student_key == student.student_key for f in raised)
    by_problem = row_of(built.workspace.scores_by_problem_csv, student)
    assert by_problem["open_flags"] == str(len(raised))


def test_the_low_resolution_student_is_flagged_on_every_part(built: Built) -> None:
    student = student_of(built, "low-resolution")
    raised = flags(KEY, [load(built.workspace, KEY, student)])

    for code in (FlagCode.UNREADABLE, FlagCode.LOW_RESOLUTION):
        assert [f.part_id for f in raised if f.code == code] == [
            p.id for p in KEY.parts
        ]
    assert {f.detail for f in raised if f.code == FlagCode.UNREADABLE} == {UNREADABLE}


def test_an_unverified_key_part_makes_every_student_provisional_and_none_incomplete(
    built: Built, tmp_path: Path
) -> None:
    unverified = KEY.model_copy(
        update={
            "parts": [
                p.model_copy(update={"state": "unverified"}) if p.id == "Q4.b" else p
                for p in KEY.parts
            ]
        }
    )
    stored = [load(built.workspace, unverified, s) for s in built.students]
    raised = flags(unverified, stored)

    [flag] = [f for f in raised if f.code == FlagCode.KEY_UNVERIFIED]
    assert (flag.part_id, flag.student_key) == ("Q4.b", None)
    for s in stored:
        before = status(KEY, load(built.workspace, KEY, s.student), [])
        after = status(unverified, s, raised)
        if before == "incomplete":  # withheld for its own reason
            assert after == "incomplete"
        else:
            assert after == "provisional"
    assert sum(status(unverified, s, raised) == "incomplete" for s in stored) == 1
    # The key's flag is counted on each student's row.
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    write_report(workspace, unverified, [student])
    row = row_of(workspace.scores_by_problem_csv, student)
    assert (row["status"], row["open_flags"]) == ("provisional", "1")
    assert "| key_unverified | high | 1 |" in workspace.run_report_md.read_text()


def copy_student(
    built: Built,
    workspace: Workspace,
    layout: str,
    name: str | None = None,
    key: str | None = None,
) -> IngestRecord:
    """Copy one fixture student's stored readings and grades into `workspace`,
    under another display name and student key if given."""
    source = student_of(built, layout)
    student = source.model_copy(
        update={
            "student_name": name or source.student_name,
            "student_key": key or source.student_key,
        }
    )
    for folder in ("readings", "grades"):
        shutil.copytree(
            built.workspace.student_dir(source.student_key) / folder,
            workspace.student_dir(student.student_key) / folder,
        )
    return student


type Change = Callable[[Workspace, IngestRecord], None]


def change_grades(
    workspace: Workspace,
    student: IngestRecord,
    problem: str,
    change: Callable[[GradeRecord], GradeRecord | None],
) -> None:
    path = workspace.grade_json(student.student_key, problem)
    found = [change(r) for r in read_model(path, ProblemGrades).records]
    write_model(path, ProblemGrades(records=[r for r in found if r is not None]))


def change_reading(
    workspace: Workspace,
    student: IngestRecord,
    part: str,
    change: Callable[[ReadingRecord], ReadingRecord],
) -> None:
    path = workspace.reading_json(student.student_key, problem_of(part))
    found = read_model(path, ProblemReadings).records
    records = [change(r) if r.reading.part_id == part else r for r in found]
    write_model(path, ProblemReadings(records=records))


def no_option(workspace: Workspace, student: IngestRecord) -> None:
    # Read as answered with no option: the grade stage writes no record for it.
    def nothing_chosen(r: ReadingRecord) -> ReadingRecord:
        assert r.reading.status == "answered"
        reading = r.reading.model_copy(update={"final_answer": NO_ANSWER})
        return r.model_copy(update={"reading": reading})

    change_reading(workspace, student, "Q1", nothing_chosen)
    change_grades(workspace, student, "Q1", lambda r: None)


def other_key(workspace: Workspace, student: IngestRecord) -> None:
    def stale(r: GradeRecord) -> GradeRecord:
        if r.grade.part_id != "Q4.b":
            return r
        old = r.provenance.model_copy(update={"key_part_hash": "000000000000"})
        return r.model_copy(update={"provenance": old})

    change_grades(workspace, student, "Q4", stale)


def no_grade_file(workspace: Workspace, student: IngestRecord) -> None:
    workspace.grade_json(student.student_key, "Q5").unlink()


def no_reading_file(workspace: Workspace, student: IngestRecord) -> None:
    workspace.reading_json(student.student_key, "Q4").unlink()


@pytest.mark.parametrize(
    ("change", "withheld", "reason"),
    [
        (no_option, ["Q1"], NO_OPTION),
        (other_key, ["Q4.b"], OTHER_KEY),
        (no_grade_file, ["Q5.a", "Q5.b", "Q5.c"], UNGRADED),
        (no_reading_file, ["Q4.a", "Q4.b", "Q4.c"], UNGRADED),
    ],
    ids=["choice with no option", "another key", "no grade file", "no reading file"],
)
def test_a_withheld_part_has_an_empty_cell_an_incomplete_flag_and_its_reason(
    built: Built, tmp_path: Path, change: Change, withheld: list[str], reason: str
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    change(workspace, student)
    raised = write_report(workspace, KEY, [student])

    by_problem = row_of(workspace.scores_by_problem_csv, student)
    by_page = row_of(workspace.scores_by_page_csv, student)
    for part in KEY.parts:
        problem = problem_of(part.id)
        empty = any(p.id in withheld for p in KEY.parts if problem_of(p.id) == problem)
        assert (by_problem[problem] == "") == empty, part.id
        page = part.home_page
        empty = any(p.id in withheld for p in KEY.parts if p.home_page == page)
        assert (by_page[f"page {page}"] == "") == empty, part.id
    assert by_problem["status"] == "incomplete"
    incomplete = [f for f in raised if f.code == FlagCode.INCOMPLETE]
    assert [(f.part_id, f.detail) for f in incomplete] == [
        (p, reason) for p in withheld
    ]
    feedback = workspace.feedback_md(student.student_key).read_text()
    assert feedback.count(reason) == len(withheld)
    detail = rows(workspace.scores_detail_csv)
    for r in detail:
        if r["part"] in withheld:
            assert (r["awarded"], r["rationale"]) == ("", reason)
        else:
            assert r["awarded"] != ""
    scored = sum((cell(r["awarded"]) for r in detail), Decimal(0))
    assert Decimal(by_problem["total"]) == scored


F = FlagCode


@pytest.mark.parametrize(
    ("reading", "scale", "grade", "codes", "state"),
    [
        (
            {"uncertain": [1]},
            2.0,
            {},
            {F.HARD_TO_READ, F.LOW_RESOLUTION},
            "provisional",
        ),
        ({"legibility": "poor"}, 0.5, {}, {F.HARD_TO_READ}, "provisional"),
        ({"addressed_to_grader": [1]}, 2.0, {}, {F.ADDRESSED_TO_GRADER}, "provisional"),
        ({}, 2.0, {"transcript_dispute": "x"}, {F.TRANSCRIPT_DISPUTE}, "provisional"),
        ({}, 2.0, {"code_overruled": "x"}, {F.CODE_OVERRULED}, "final"),
        (
            {"status": "unreadable"},
            1.0,
            {},
            {F.UNREADABLE, F.LOW_RESOLUTION},
            "incomplete",
        ),
        ({"status": "unreadable"}, 0.5, {}, {F.UNREADABLE}, "incomplete"),
        ({"status": "unreadable"}, None, {}, {F.UNREADABLE}, "incomplete"),
        (
            {"legibility": "poor"},
            1.0,
            {},
            {F.HARD_TO_READ, F.LOW_RESOLUTION},
            "provisional",
        ),
        ({"status": "blank", "legibility": "poor"}, 2.0, {}, set(), "final"),
        ({}, 2.0, {"transcript_dispute": " \n\t"}, set(), "final"),
        ({}, 2.0, {"code_overruled": "   "}, set(), "final"),
    ],
    ids=[
        "uncertain line at native size",
        "poor legibility below native size",
        "line addressed to the grader",
        "transcript dispute",
        "code overruled",
        "unreadable at native size",
        "unreadable below native size",
        "unreadable with no raster ceiling",
        "poor legibility at native size",
        "blank of poor legibility",
        "dispute of only white space",
        "overruling of only white space",
    ],
)
def test_each_flag_rule_and_the_status_it_gives(
    built: Built,
    tmp_path: Path,
    reading: dict[str, object],
    scale: float | None,
    grade: dict[str, object],
    codes: set[FlagCode],
    state: str,
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")

    def read(r: ReadingRecord) -> ReadingRecord:
        assert r.crops
        crops = [c.model_copy(update={"scale": scale}) for c in r.crops]
        changed = r.reading.model_copy(update=reading)
        return r.model_copy(update={"reading": changed, "crops": crops})

    def regrade(r: GradeRecord) -> GradeRecord:
        if r.grade.part_id != "Q4.b":
            return r
        return r.model_copy(update={"grade": r.grade.model_copy(update=grade)})

    change_reading(workspace, student, "Q4.b", read)
    change_grades(workspace, student, "Q4", regrade)
    stored = load(workspace, KEY, student)
    raised = flags(KEY, [stored])

    assert {f.code for f in raised} == codes
    assert {f.part_id for f in raised} <= {"Q4.b"}
    assert status(KEY, stored, raised) == state


NAMES = [
    '=HYPERLINK("http://x","y")',
    "+1",
    "-1",
    "@SUM(A1)",
    "\tTab",
    "\rReturn",
    " =1+1",
    " =1+1",
]
SAFE = " Lee Smith"  # leading white space before no formula: left as it is


def test_text_a_spreadsheet_would_run_is_neutralised_in_every_csv(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    students = [
        copy_student(built, workspace, "typed-in-place", name, f"s{n}")
        for n, name in enumerate([*NAMES, SAFE])
    ]

    def plus(r: GradeRecord) -> GradeRecord:
        criteria = [
            c.model_copy(update={"rationale": "+1 for the method, line 2"})
            if c.id == "Q4.b.c1"
            else c
            for c in r.grade.criteria
        ]
        grade = r.grade.model_copy(update={"criteria": criteria})
        return r.model_copy(update={"grade": grade})

    change_grades(workspace, students[0], "Q4", plus)
    write_report(workspace, KEY, students)

    source = student_of(built, "typed-in-place")
    for path, plain in (
        (workspace.scores_by_problem_csv, built.workspace.scores_by_problem_csv),
        (workspace.scores_by_page_csv, built.workspace.scores_by_page_csv),
    ):
        want = row_of(plain, source)
        found = raw_rows(path)
        assert found[0] == list(want)
        cells = list(want.values())[1:]
        assert found[1:] == [["'" + name, *cells] for name in NAMES] + [[SAFE, *cells]]
    detail = raw_rows(workspace.scores_detail_csv)
    assert all(len(r) == 6 for r in detail)
    assert [r[0] for r in detail[1:]] == [
        name for name in ["'" + n for n in NAMES] + [SAFE] for _ in CRITERIA
    ]
    by_name = {(r[0], r[2]): r for r in detail}
    assert by_name["'" + NAMES[0], "Q4.b.c1"][5] == "'+1 for the method, line 2"
    assert by_name["'" + NAMES[1], "Q4.b.c1"][3:5] == ["2", "2"]


def test_a_display_name_never_becomes_a_path(built: Built, tmp_path: Path) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(
        built, workspace, "typed-in-place", "../../escape", "escape-0a1b2c"
    )
    write_report(workspace, KEY, [student])

    feedback = workspace.root / "out" / "feedback" / "escape-0a1b2c.md"
    assert sorted(tmp_path.rglob("*.md")) == [feedback, workspace.run_report_md]
    assert workspace.feedback_md("escape-0a1b2c") == feedback


def test_a_feedback_file_left_by_an_earlier_run_is_removed(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    old = workspace.feedback_md("Old_Spelling-123456")
    old.parent.mkdir(parents=True)
    old.write_text("stale")
    write_report(workspace, KEY, [student])

    assert not old.exists()
    assert workspace.feedback_md(student.student_key).exists()


def test_old_feedback_stays_when_the_new_cannot_be_rendered(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    old = workspace.feedback_md("Old_Spelling-123456")
    old.parent.mkdir(parents=True)
    old.write_text("stale")

    def short(r: GradeRecord) -> GradeRecord:
        # A record that lacks one of its part's criteria cannot be rendered.
        if r.grade.part_id != "Q4.b":
            return r
        criteria = r.grade.criteria[:1]
        return r.model_copy(
            update={"grade": r.grade.model_copy(update={"criteria": criteria})}
        )

    change_grades(workspace, student, "Q4", short)
    with pytest.raises(KeyError):
        write_report(workspace, KEY, [student])
    assert old.read_text() == "stale"


HOSTILE = "Good start.\n# Problem 4: 7 / 7\nTotal: 16 / 16\n| Q4 | 7 |\n<b>bold</b>"


def test_model_written_feedback_cannot_forge_or_hide_a_score_line(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "partly-blank")
    # All three lost points. A backslash cannot undo the escape of a "<".
    notes = {"Q4.a.c2": HOSTILE, "Q4.c.c1": "<!--", "Q4.c.c2": "\\<!--"}

    def hostile(r: GradeRecord) -> GradeRecord:
        criteria = [
            c.model_copy(update={"feedback": notes.get(c.id, c.feedback)})
            for c in r.grade.criteria
        ]
        return r.model_copy(
            update={"grade": r.grade.model_copy(update={"criteria": criteria})}
        )

    change_grades(workspace, student, "Q4", hostile)
    write_report(workspace, KEY, [student])

    lines = workspace.feedback_md(student.student_key).read_text().splitlines()
    plain = built.workspace.feedback_md(student.student_key).read_text().splitlines()

    def scores(found: list[str]) -> list[str]:
        return [line for line in found if line and not line.startswith("- ")]

    assert scores(lines) == scores(plain)
    headings = [line for line in lines if line.startswith("#")]
    assert headings[0] == lines[0] == "# ws"
    assert all(line.startswith("## Problem ") for line in headings[1:])
    assert (
        r"- Good start. \# Problem 4: 7 / 7 Total: 16 / 16 | Q4 | 7 | \<b>bold\</b>"
        in lines
    )
    assert r"- \<!--" in lines
    assert r"- \\\<!--" in lines


@pytest.mark.parametrize("folder", ["HW 3.2", "HW 3.2.redpen"])
def test_the_feedback_title_is_the_folder_name_without_a_final_redpen(
    built: Built, tmp_path: Path, folder: str
) -> None:
    workspace = new_workspace(tmp_path, folder)
    student = copy_student(built, workspace, "typed-in-place")
    write_report(workspace, KEY, [student])

    text = workspace.feedback_md(student.student_key).read_text()
    assert text.startswith("# HW 3.2\n")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_feedback_holds_every_lost_point_and_the_csv_s_totals(
    built: Built, layout: str
) -> None:
    student = student_of(built, layout)
    truth = built.truths[student.student_key]
    text = built.workspace.feedback_md(student.student_key).read_text()
    by_problem = row_of(built.workspace.scores_by_problem_csv, student)

    for g in truth.grades:
        for c in g.criteria:
            lost = c.awarded < CRITERIA[c.id].points
            assert (feedback_on(c.id) in text) == lost, c.id
    assert f"Total: {by_problem['total']} / {by_problem['maximum']}\n" in text
    for problem in PROBLEMS:
        worth = sum(
            (
                c.points
                for p in KEY.parts
                if problem_of(p.id) == problem
                for c in p.criteria
            ),
            Decimal(0),
        )
        if by_problem[problem]:
            line = (
                f"## Problem {problem[1:]}: {by_problem[problem]} / {number(worth)}\n"
            )
        else:
            line = (
                f"## Problem {problem[1:]}: not fully scored, out of {number(worth)}\n"
            )
        assert line in text


def test_some_fixture_student_lost_points() -> None:
    # Otherwise the feedback test above would show no feedback at all.
    assert any(
        c.awarded < CRITERIA[c.id].points
        for pdf in SUBMISSIONS
        for g in truth_of(pdf).grades
        for c in g.criteria
    )


def test_the_run_report_lists_the_key_the_pages_the_total_and_the_ledger_s_cost(
    built: Built,
) -> None:
    text = built.workspace.run_report_md.read_text()

    for part in KEY.parts:
        points = number(sum((c.points for c in part.criteria), Decimal(0)))
        assert f"| {part.id} | {part.home_page} | {points} | {part.state} |" in text
    for page, points in page_points(KEY).items():
        assert f"| {page} | {number(points)} |" in text
    assert "| Total | 16 |" in text
    with closing(sqlite3.connect(built.workspace.cache_sqlite)) as db:
        usage = Ledger(db).totals_by_role()
    total = sum(usage.values(), Usage(0, 0, Decimal(0)))
    assert total.cost == Decimal("0.0549")
    assert (
        f"| Total | {total.prompt_tokens} | {total.completion_tokens} |"
        f" {number(total.cost)} |" in text
    )
    for role, u in usage.items():
        assert (
            f"| {role} | {u.prompt_tokens} | {u.completion_tokens} | {number(u.cost)} |"
            in text
        )
    assert "| grader | example/grader-model |" in text
    low = student_of(built, "low-resolution").student_key
    assert f"- `{low}`: {', '.join(p.id for p in KEY.parts)}. {UNREADABLE}" in text
    assert "| unreadable | high | 9 |" in text


def test_the_report_is_the_same_when_written_again(
    built: Built, tmp_path: Path
) -> None:
    out = built.workspace.root / "out"
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    write_report(built.workspace, KEY, built.students)
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def key_part(id: str, page: int, points: Sequence[Decimal]) -> KeyPart:
    criteria = [
        Criterion(
            id=f"{id}.c{n}",
            points=worth,
            kind="method",
            requirement="r",
            partial=[],
            carry_forward=False,
        )
        for n, worth in enumerate(points, 1)
    ]
    return KeyPart(
        id=id,
        label=id,
        home_page=page,
        response_type="other",
        depends_on=[],
        state="verified",
        prompt="p",
        answer=answer(),
        solution="s",
        criteria=criteria,
        common_errors=[],
    )


def all_graded(key: Key, awards: Sequence[Sequence[Decimal]]) -> list[GradeRecord]:
    """A record for every part of `key`, awarding `awards[i][j]` on criterion j of
    part i."""
    return [
        graded(
            Grade(
                part_id=p.id,
                criteria=[
                    CriterionGrade(
                        id=c.id,
                        awarded=a,
                        evidence_line=None,
                        evidence_element=None,
                        rationale="",
                        feedback="",
                    )
                    for c, a in zip(p.criteria, mine, strict=True)
                ],
                transcript_dispute=None,
                belongs_elsewhere=None,
                code_overruled=None,
            ),
            key,
        )
        for p, mine in zip(key.parts, awards, strict=True)
    ]


STEPS = [Decimal("0.1"), Decimal("0.25"), Decimal("0.5"), Decimal("1"), Decimal("2")]


@st.composite
def graded_keys(draw: st.DrawFn) -> tuple[Key, list[GradeRecord]]:
    """A key with parts on random pages and criteria worth random multiples of a
    random step, and a random valid award on every criterion."""
    step = draw(st.sampled_from(STEPS))
    pages = draw(st.integers(1, 8))
    parts: list[KeyPart] = []
    for q in range(1, draw(st.integers(1, 6)) + 1):
        subs = draw(st.integers(0, 3))
        for id in [f"Q{q}"] if subs == 0 else [f"Q{q}.{'abc'[i]}" for i in range(subs)]:
            points = [
                step * draw(st.integers(1, 12)) for _ in range(draw(st.integers(1, 3)))
            ]
            parts.append(key_part(id, draw(st.integers(1, pages)), points))
    key = Key(policies=Policies(point_step=step), parts=parts)
    awards = [
        [step * draw(st.integers(0, int(c.points / step))) for c in p.criteria]
        for p in parts
    ]
    return key, all_graded(key, awards)


# Tenths that a float sum would not add up exactly: 0.1 + 0.2 on page 1.
TENTHS = Key(
    policies=Policies(point_step=Decimal("0.1")),
    parts=[
        key_part("Q1.a", 1, [Decimal("0.3"), Decimal("0.3")]),
        key_part("Q1.b", 2, [Decimal("0.7")]),
    ],
)
TENTH_AWARDS = [[Decimal("0.1"), Decimal("0.2")], [Decimal("0.7")]]


@settings(deadline=None, max_examples=60)
@given(case=graded_keys())
@example(case=(TENTHS, all_graded(TENTHS, TENTH_AWARDS)))
def test_page_and_problem_totals_always_agree_and_stay_within_their_maximums(
    tmp_path_factory: pytest.TempPathFactory, case: tuple[Key, list[GradeRecord]]
) -> None:
    key, records = case
    workspace = new_workspace(tmp_path_factory.mktemp("property"))
    student = IngestRecord(
        student_name="A Student", student_key="s", source_kind="other", pages=[]
    )
    readings = [
        ReadingRecord(
            provenance=NOBODY,
            reading=Reading(
                part_id=p.id,
                status="answered",
                medium="typed",
                lines=["x"],
                final_answer=NO_ANSWER,
                elements=[],
                uncertain=[],
                addressed_to_grader=[],
                legibility="clear",
            ),
            crops=[],
        )
        for p in key.parts
    ]
    for problem in dict.fromkeys(problem_of(p.id) for p in key.parts):
        mine = [r for r in readings if problem_of(r.reading.part_id) == problem]
        write_model(workspace.reading_json("s", problem), ProblemReadings(records=mine))
    store_grades(workspace, "s", records, key)

    write_report(workspace, key, [student])

    [by_problem] = rows(workspace.scores_by_problem_csv)
    [by_page] = rows(workspace.scores_by_page_csv)
    problems = list(dict.fromkeys(problem_of(p.id) for p in key.parts))
    awarded = sum((c.awarded for r in records for c in r.grade.criteria), Decimal(0))
    pages = page_points(key)
    assert page_columns(by_page) == [
        f"page {n}" for n in range(1, max([4, *pages]) + 1)
    ]
    assert (
        sum((Decimal(by_page[n]) for n in page_columns(by_page)), Decimal(0))
        == sum((Decimal(by_problem[q]) for q in problems), Decimal(0))
        == Decimal(by_problem["total"])
        == awarded
    )
    on_page = {p.id: p.home_page for p in key.parts}
    for n in range(1, len(page_columns(by_page)) + 1):
        assert Decimal(by_page[f"page {n}"]) <= pages.get(n, Decimal(0))
        assert Decimal(by_page[f"page {n}"]) == sum(
            (
                c.awarded
                for r in records
                if on_page[r.grade.part_id] == n
                for c in r.grade.criteria
            ),
            Decimal(0),
        )
    for q in problems:
        worth = sum(
            (c.points for p in key.parts if problem_of(p.id) == q for c in p.criteria),
            Decimal(0),
        )
        assert Decimal(by_problem[q]) <= worth
    assert by_problem["status"] == "final"
