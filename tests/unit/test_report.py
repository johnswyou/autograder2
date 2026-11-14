"""The report stage: score files, feedback, flags and the run report, computed from
stored files with no model call.

The fixture students' stored files are the truth's: `key.json` as the key, each truth
file's readings (with crops built from the truth's map by the read stage's image
function) and grades, on the session's fresh ingest. They are stored and reported
once for the module; a test that changes something works on a copy."""

import csv
import io
import json
import re
import shutil
import socket
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner, Result

from redpen import cli
from redpen.cli import app
from redpen.grade import grade as stage
from redpen.grade.audit import groups
from redpen.grade.ingest import IngestSummary
from redpen.grade.read import digest, problem_of
from redpen.llm.ledger import Ledger, Usage
from redpen.prepare.key import (
    load_key,
    number,
    page_points,
    part_hash,
    save_key,
    yaml_reader,
    yaml_writer,
)
from redpen.report import report as reporting
from redpen.report.overrides import record_review
from redpen.report.report import (
    CLOSED,
    CORRECTED,
    DETAIL,
    NO_OPTION,
    NOT_INGESTED,
    NOTHING_TO_REVIEW,
    OTHER_KEY,
    OTHER_READING,
    OUTDATED,
    STALE,
    UNGRADED,
    UNREADABLE,
    Entry,
    Stored,
    code_span,
    flags,
    load,
    one_line,
    review,
    review_entries,
    review_line,
    score,
    status,
    write_report,
)
from redpen.schemas.audit import Change as AuditChange
from redpen.schemas.audit import PartAudit, Suspect
from redpen.schemas.common import DiagramElement, Provenance
from redpen.schemas.flags import Flag, FlagCode
from redpen.schemas.grade import (
    AnswerCheck,
    CriterionGrade,
    Grade,
    GradeRecord,
    Override,
    ProblemGrades,
    SecondGrade,
    SecondGrades,
)
from redpen.schemas.key import Criterion, Key, KeyPart, PointConstraint, Policies
from redpen.schemas.reading import (
    FinalAnswer,
    ProblemReadings,
    Reading,
    ReadingRecord,
    SecondReading,
    SecondReadings,
)
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, keep_out_of_git, read_model, write_model
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
    return GradeRecord(
        provenance=provenance, grade=grade, check=None, reading_digest=None
    )


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


def graded_from(
    workspace: Workspace,
    student: IngestRecord,
    reading: Callable[[ReadingRecord], str | None],
    **provenance: str,
) -> None:
    """Q4.b's grade record made from `reading` of the stored reading record (its
    digest, or None as before digests were kept), with `provenance` changed."""
    path = workspace.reading_json(student.student_key, "Q4")
    [stored] = [
        r
        for r in read_model(path, ProblemReadings).records
        if r.reading.part_id == "Q4.b"
    ]

    def made(r: GradeRecord) -> GradeRecord:
        if r.grade.part_id != "Q4.b":
            return r
        old = r.provenance.model_copy(update=provenance)
        return r.model_copy(
            update={"provenance": old, "reading_digest": reading(stored)}
        )

    change_grades(workspace, student, "Q4", made)


def earlier(r: ReadingRecord) -> str:
    return digest(r.model_copy(update={"crops": []}))


def other_reading(workspace: Workspace, student: IngestRecord) -> None:
    graded_from(workspace, student, earlier)


def other_key_and_reading(workspace: Workspace, student: IngestRecord) -> None:
    graded_from(workspace, student, earlier, key_part_hash="000000000000")


def no_grade_file(workspace: Workspace, student: IngestRecord) -> None:
    workspace.grade_json(student.student_key, "Q5").unlink()


def no_reading_file(workspace: Workspace, student: IngestRecord) -> None:
    workspace.reading_json(student.student_key, "Q4").unlink()


@pytest.mark.parametrize(
    ("change", "withheld", "reason"),
    [
        (no_option, ["Q1"], NO_OPTION),
        (other_key, ["Q4.b"], OTHER_KEY),
        (other_reading, ["Q4.b"], OTHER_READING),
        (other_key_and_reading, ["Q4.b"], OTHER_KEY),
        (no_grade_file, ["Q5.a", "Q5.b", "Q5.c"], UNGRADED),
        (no_reading_file, ["Q4.a", "Q4.b", "Q4.c"], UNGRADED),
    ],
    ids=[
        "choice with no option",
        "another key",
        "another reading",
        "another key and another reading",
        "no grade file",
        "no reading file",
    ],
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
    # Each entry links the readings, never a grade withheld.
    for head, lines in listed(workspace):
        assert lines[-1].startswith("- Files:") and "[grade]" not in lines[-1], head


@pytest.mark.parametrize(
    ("reading", "provenance"),
    [
        (digest, {}),
        (lambda r: None, {}),
        (digest, {"prompt_version": "older", "model_slug": "example/older"}),
    ],
    ids=["the stored reading", "kept before digests", "an earlier prompt"],
)
def test_a_grade_made_from_the_stored_reading_is_shown_under_any_prompt(
    built: Built,
    tmp_path: Path,
    reading: Callable[[ReadingRecord], str | None],
    provenance: dict[str, str],
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    graded_from(workspace, student, reading, **provenance)

    [q4b] = [
        s for s in score(KEY, load(workspace, KEY, student)) if s.part.id == "Q4.b"
    ]

    assert q4b.withheld is None and q4b.awarded is not None


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


# What the stored second reading of a part read with doubt does to its flags.
DOUBT = {F.HARD_TO_READ, F.LOW_RESOLUTION}
UNSURE = {"medium": "handwritten", "uncertain": [1]}  # read again
TYPED = {"medium": "typed", "uncertain": [1]}  # in doubt, but never read again
SURE = {"medium": "handwritten"}


@dataclass(frozen=True)
class Case:
    reading: Mapping[str, object]  # the first reading of Q4.b, changed
    doubted: bool
    result: Literal["same", "differ", "not decided", "failed"]
    codes: set[FlagCode]
    state: str
    counts: tuple[tuple[int, ...], tuple[int, ...]]  # in doubt, the others
    stale: str = ""  # how the stored result was made for something else


CASES = {
    "same": Case(UNSURE, True, "same", set(), "final", ((1, 1, 0, 0, 0), (0, 0, 0, 0))),
    "differ": Case(
        UNSURE,
        True,
        "differ",
        DOUBT | {F.READERS_DISAGREE},
        "provisional",
        ((1, 0, 1, 0, 0), (0, 0, 0, 0)),
    ),
    "not decided": Case(
        UNSURE, True, "not decided", DOUBT, "provisional", ((1, 0, 0, 1, 0), (0,) * 4)
    ),
    "failed": Case(
        UNSURE, True, "failed", DOUBT, "provisional", ((1, 0, 0, 0, 1), (0,) * 4)
    ),
    "the first reading made again": Case(
        UNSURE, True, "same", DOUBT, "provisional", ((0,) * 5, (0,) * 4), "edited"
    ),
    "another version of the part": Case(
        UNSURE, True, "same", DOUBT, "provisional", ((0,) * 5, (0,) * 4), "hash"
    ),
    "a file of an earlier shape": Case(
        UNSURE, True, "same", DOUBT, "provisional", ((0,) * 5, (0,) * 4), "shape"
    ),
    "the same on a part not in doubt": Case(
        TYPED, False, "same", DOUBT, "provisional", ((0,) * 5, (1, 1, 0, 0))
    ),
    "differ on a part not in doubt": Case(
        SURE,
        False,
        "differ",
        {F.READERS_DISAGREE},
        "provisional",
        ((0,) * 5, (1, 0, 1, 0)),
    ),
    "not decided on a part not in doubt": Case(
        SURE, False, "not decided", set(), "final", ((0,) * 5, (1, 0, 0, 1))
    ),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_a_second_reading_changes_the_part_s_flags_and_never_its_score(
    built: Built, tmp_path: Path, case: Case
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")

    def read(r: ReadingRecord) -> ReadingRecord:
        crops = [c.model_copy(update={"scale": 2.0}) for c in r.crops]
        reading = r.reading.model_copy(update=case.reading)
        return r.model_copy(update={"reading": reading, "crops": crops})

    change_reading(workspace, student, "Q4.b", read)
    before = load(workspace, KEY, student)
    first = before.readings["Q4.b"]
    [part] = [p for p in KEY.parts if p.id == "Q4.b"]
    second = first.model_copy(update={"reading": first.reading.model_copy()})
    record = SecondReading(
        part_id="Q4.b",
        part_hash="000000000000"
        if case.stale == "hash"
        else part_hash(part, KEY.policies),
        first=digest(first),
        doubted=case.doubted,
        result=case.result,
        second=None if case.result == "failed" else second,
    )
    path = workspace.second_json(student.student_key)
    write_model(path, SecondReadings(records=[record]))
    if case.stale == "edited":  # the first reading's final answer, read anew

        def again(r: ReadingRecord) -> ReadingRecord:
            stated = r.reading.final_answer.model_copy(update={"value": 2.6})
            reading = r.reading.model_copy(update={"final_answer": stated})
            return r.model_copy(update={"reading": reading})

        change_reading(workspace, student, "Q4.b", again)
    if case.stale == "shape":  # as this task's first version wrote it
        old = record.model_dump(mode="json", exclude={"part_hash", "doubted"})
        path.write_text(json.dumps({"records": [old]}))

    stored = load(workspace, KEY, student)
    raised = flags(KEY, [stored])

    assert {f.code for f in raised} == case.codes
    assert {f.part_id for f in raised} <= {"Q4.b"}
    assert all(f.severity == "high" for f in raised if f.code == F.READERS_DISAGREE)
    assert status(KEY, stored, raised) == case.state
    assert score(KEY, stored) == score(KEY, before)
    write_report(workspace, KEY, [student])
    lines = workspace.run_report_md.read_text().splitlines()
    (n, same, differ, open_, failed), (m, same2, differ2, open2) = case.counts
    assert (
        "Parts the reader was unsure of, read a second time by another vendor's"
        f" model: {n}. The same: {same}. Different: {differ}. Not decided: {open_}."
        f" Failed: {failed}."
    ) in lines
    assert (
        f"The other parts of those units: {m}. The same: {same2}. Different:"
        f" {differ2}. Not decided: {open2}."
    ) in lines


# What a stored second grade of Q4.b does to the flags, and never to a score.
@dataclass(frozen=True)
class Regraded:
    result: Literal["same", "differ", "failed"]
    codes: set[FlagCode]
    state: str
    counts: tuple[int, int, int, int]  # awards compared, different, flagged, failed
    stale: str = ""  # how the stored result was made for another first grade


NONE_COUNTED = (0, 0, 0, 0)
REGRADED = {
    "same": Regraded("same", set(), "final", (2, 0, 0, 0)),
    "differ": Regraded("differ", {F.GRADERS_DISAGREE}, "provisional", (2, 1, 1, 0)),
    "failed": Regraded("failed", set(), "final", (0, 0, 0, 1)),
    "made from another reading": Regraded(
        "differ", set(), "final", NONE_COUNTED, "reading"
    ),
    "made for another version of the part": Regraded(
        "differ", set(), "final", NONE_COUNTED, "hash"
    ),
    "made for another first award": Regraded(
        "differ", set(), "final", NONE_COUNTED, "award"
    ),
    "a file of another shape": Regraded(
        "differ", set(), "final", NONE_COUNTED, "shape"
    ),
    "a second grade without a criterion": Regraded(
        "differ", set(), "final", NONE_COUNTED, "criterion"
    ),
}
SECOND_RATIONALE = "The second grader's reason."


def second_grade_of_q4b(
    stored: Stored,
    result: Literal["same", "differ", "failed"],
    rationale: str = SECOND_RATIONALE,
) -> SecondGrade:
    """A second grade of the stored first grade of Q4.b: the same, or 1 in place
    of 2 on its first criterion, with `rationale`. Either way the second grader
    disputes the reading and overrules the check, which only the first may do."""
    [part] = [p for p in KEY.parts if p.id == "Q4.b"]
    first = stored.grades["Q4.b"]
    c1, c2 = first.grade.criteria
    if result == "differ":
        c1 = c1.model_copy(update={"awarded": Decimal(1), "rationale": rationale})
    grade = first.grade.model_copy(
        update={
            "criteria": [c1, c2],
            "transcript_dispute": "The reading is wrong.",
            "code_overruled": "The check is wrong.",
        }
    )
    return SecondGrade(
        part_id="Q4.b",
        part_hash=part_hash(part, KEY.policies),
        reading_digest=stage.graded_from(part, stored.readings),
        first=digest(first),
        result=result,
        second=None
        if result == "failed"
        else first.model_copy(update={"grade": grade}),
    )


@pytest.mark.parametrize("case", REGRADED.values(), ids=REGRADED.keys())
def test_a_second_grade_flags_a_differing_award_and_changes_no_score(
    built: Built, tmp_path: Path, case: Regraded
) -> None:
    workspace = new_workspace(tmp_path)
    workspace.config_lock_toml.write_text(LOCK)
    student = copy_student(built, workspace, "typed-in-place")
    record = second_grade_of_q4b(load(workspace, KEY, student), case.result)
    if case.stale == "hash":
        record = record.model_copy(update={"part_hash": "000000000000"})
    if case.stale == "criterion" and record.second:  # by hand: the stage asks for all
        grade = record.second.grade
        lacking = grade.model_copy(update={"criteria": grade.criteria[:1]})
        second = record.second.model_copy(update={"grade": lacking})
        record = record.model_copy(update={"second": second})
    if case.stale == "reading":  # Q4.b read anew since

        def anew(r: ReadingRecord) -> ReadingRecord:
            lines = [*r.reading.lines, "Read anew."]
            return r.model_copy(
                update={"reading": r.reading.model_copy(update={"lines": lines})}
            )

        change_reading(workspace, student, "Q4.b", anew)
    if case.stale == "award":  # Q4.b graded anew since, with another award

        def lower_c2(r: GradeRecord) -> GradeRecord:
            if r.grade.part_id != "Q4.b":
                return r
            c1, c2 = r.grade.criteria
            c2 = c2.model_copy(update={"awarded": Decimal(0)})
            grade = r.grade.model_copy(update={"criteria": [c1, c2]})
            return r.model_copy(update={"grade": grade})

        change_grades(workspace, student, "Q4", lower_c2)
    alone = load(workspace, KEY, student)
    write_report(workspace, KEY, [student])
    unflagged = {
        p: p.read_bytes()
        for p in (
            workspace.feedback_md(student.student_key),
            workspace.scores_by_page_csv,
            workspace.scores_detail_csv,
        )
    }
    path = workspace.second_grade_json(student.student_key)
    write_model(path, SecondGrades(records=[record]))
    if case.stale == "shape":
        path.write_text(json.dumps({"records": [{"part_id": "Q4.b"}]}))

    stored = load(workspace, KEY, student)
    raised = flags(KEY, [stored])

    assert {f.code for f in raised} == case.codes
    assert {f.part_id for f in raised} <= {"Q4.b"}
    assert all(f.severity == "high" for f in raised if f.code == F.GRADERS_DISAGREE)
    assert status(KEY, stored, raised) == case.state
    # The first grade is the grade, and the class audit sees it alone.
    assert score(KEY, stored) == score(KEY, alone)
    [part] = [p for p in KEY.parts if p.id == "Q4.b"]
    hashed = part_hash(part, KEY.policies)
    assert groups(part, hashed, [stored]) == groups(part, hashed, [alone])
    write_report(workspace, KEY, [student])
    assert {p: p.read_bytes() for p in unflagged} == unflagged
    lines = workspace.run_report_md.read_text().splitlines()
    awards, different, flagged, failed = case.counts
    assert (
        "Awards graded a second time, by the same grader with the same request, and"
        f" compared: {awards}. Different: {different}. Parts flagged"
        f" (`graders_disagree`): {flagged}. Problems whose second grade failed:"
        f" {failed}."
    ) in lines
    rate = f" ({(awards - different) / awards:.1%})" if awards else ""
    assert f"Agreement: {awards - different} of {awards} awards the same{rate}." in (
        lines
    )


def test_an_award_the_instructor_corrected_stays_theirs_whatever_the_second_grade(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    record = second_grade_of_q4b(load(workspace, KEY, student), "differ")
    write_model(
        workspace.second_grade_json(student.student_key),
        SecondGrades(records=[record]),
    )
    [part] = [p for p in KEY.parts if p.id == "Q4.b"]
    correction = Override(
        student=student.student_key,
        criterion="Q4.b.c1",
        points=Decimal("0.5"),
        part_hash=part_hash(part, KEY.policies),
    )

    stored = replace(load(workspace, KEY, student), overrides={"Q4.b.c1": correction})

    [q4b] = [s for s in score(KEY, stored) if s.part.id == "Q4.b"]
    assert q4b.awarded == Decimal("1.5")  # the instructor's 0.5, and c2's 1
    assert F.GRADERS_DISAGREE in {f.code for f in flags(KEY, [stored])}


def test_the_review_list_shows_both_awards_and_rationales_as_untrusted_text(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    hostile = "Fine.\n### Someone else, Q1\n- Proposed score: 99 / 3.\n`x` <!-- y"

    def disputed(r: GradeRecord) -> GradeRecord:  # an entry with another flag
        if r.grade.part_id != "Q4.a":
            return r
        grade = r.grade.model_copy(update={"transcript_dispute": "Not what I read."})
        return r.model_copy(update={"grade": grade})

    change_grades(workspace, student, "Q4", disputed)
    first = load(workspace, KEY, student)
    record = second_grade_of_q4b(first, "differ", hostile)
    write_model(
        workspace.second_grade_json(student.student_key),
        SecondGrades(records=[record]),
    )

    write_report(workspace, KEY, [student])

    text = workspace.review_md.read_text()
    lines = text.splitlines()
    [c1] = [c for c in first.grades["Q4.b"].grade.criteria if c.id == "Q4.b.c1"]
    [at] = [n for n, line in enumerate(lines) if "`graders_disagree`" in line]
    assert lines[at] == f"- `graders_disagree`, high: {DETAIL[F.GRADERS_DISAGREE]}"
    assert lines[at + 1 : at + 3] == [
        f"  - Q4.b.c1, the first grade: 2. The grader wrote: {code_span(c1.rationale)}",
        f"  - Q4.b.c1, the second grade: 1. The grader wrote: {code_span(hostile)}",
    ]
    # Only the criterion whose awards differ.
    assert not any("Q4.b.c2, the" in line for line in lines)
    # Every piece of the second grader's text stays inside its code span.
    for piece in hostile.splitlines():
        assert [line for line in lines if piece in line] == [lines[at + 2]]
    who = student.student_key
    # On the entry with the flag, and on no other.
    entries = [e.split("\n")[0] for e in text.split("\n### ")[1:]]
    assert sorted(e.rsplit(" ", 1)[1] for e in entries) == ["Q4.a", "Q4.b"]
    assert text.count("[second grade](") == 1
    assert f"[second grade](../students/{who}/grades/second.json)" in text
    for path in (workspace.root / "out").rglob("*"):
        if path.is_file() and path != workspace.review_md:
            assert "Someone else" not in path.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize(
    "profile",
    ["economy", "balanced", "thorough", "no lock", "economy, a result stored"],
)
def test_the_run_report_says_when_the_profile_does_no_second_grade(
    built: Built, tmp_path: Path, profile: str
) -> None:
    workspace = new_workspace(tmp_path)
    if profile != "no lock":
        workspace.config_lock_toml.write_text(
            LOCK.replace("balanced", profile.split(",")[0])
        )
    student = copy_student(built, workspace, "typed-in-place")
    if profile.endswith("stored"):  # graded a second time; the lock edited since
        record = second_grade_of_q4b(load(workspace, KEY, student), "differ")
        write_model(
            workspace.second_grade_json(student.student_key),
            SecondGrades(records=[record]),
        )

    write_report(workspace, KEY, [student])

    lines = workspace.run_report_md.read_text().splitlines()
    at = lines.index("## Second grade")
    said = lines[at + 2]
    if profile == "economy":
        assert said == "The second grade is not done under the profile economy."
    else:
        assert said.startswith("Awards graded a second time")
    # A flag on a first grade known to depend on the draw stays raised.
    if profile.endswith("stored"):
        assert "| graders_disagree | high | 1 |" in lines


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
    assert sorted(tmp_path.rglob("*.md")) == [
        feedback,
        workspace.review_md,
        workspace.run_report_md,
    ]
    assert workspace.feedback_md("escape-0a1b2c") == feedback


def test_files_not_ingested_are_listed_one_per_line_in_code_and_never_opened(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    names = [
        "HW - Lee.pdf",
        "a`b ``c.pdf",
        "x\n# Total: 16 / 16‮.pdf",
        "../../etc/passwd",
    ]
    write_report(workspace, KEY, [student], names)

    text = workspace.run_report_md.read_text()
    listed = text.split("## Files not ingested\n")[1].split("\n## ")[0]
    assert [line for line in listed.splitlines() if line.startswith("- ")] == [
        f"- `HW - Lee.pdf`: {NOT_INGESTED}",
        f"- ```a`b ``c.pdf```: {NOT_INGESTED}",
        f"- `x?# Total: 16 / 16?.pdf`: {NOT_INGESTED}",
        f"- `../../etc/passwd`: {NOT_INGESTED}",
    ]
    assert not any(line.startswith("# Total") for line in text.splitlines())
    assert [p for p in tmp_path.iterdir()] == [workspace.root]


def test_the_run_report_has_no_list_of_files_not_ingested_when_there_are_none(
    built: Built,
) -> None:
    assert "Files not ingested" not in built.workspace.run_report_md.read_text()


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
    assert built.workspace.review_md in before
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


# --- the instructor's points (task 4.5a) ---------------------------------------------

# Page maximums unlike the criteria's 3, 5, 4, 4. Allocated by hand (0.25 steps):
# page 1, 24 steps for Q1-Q3 (1 each): 2, 2, 2. Page 2, 16 steps for Q4.a 2 and
# Q4.b 3: 6.4 and 9.6, floors 6 and 9, the step left to Q4.b: 1.5 and 2.5. Page 3,
# 12 steps for Q4.c 2 and Q5.a 2: 1.5 and 1.5. Page 4, 8 steps: Q5.b 1, Q5.c 1.
PAGES_KEY = KEY.model_copy(
    update={"points": PointConstraint(pages=[Decimal(n) for n in (6, 4, 3, 2)])}
)
SCORE_FILES = ["scores_by_problem.csv", "scores_by_page.csv", "scores_detail.csv"]


def test_page_maximums_scale_each_part_and_every_cell_sums_its_scores(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    blank = copy_student(built, workspace, "partly-blank")
    photo = copy_student(built, workspace, "photo-per-page")
    stale = copy_student(
        built, workspace, "photo-per-page", "Stale Copy", "stale-1a2b3c"
    )
    other_key(workspace, stale)  # Q4.b graded under another key: withheld
    low = copy_student(built, workspace, "low-resolution")

    write_report(workspace, PAGES_KEY, [blank, photo, stale, low])

    def cells(student: IngestRecord) -> tuple[list[str], list[str]]:
        by_problem = row_of(workspace.scores_by_problem_csv, student)
        by_page = row_of(workspace.scores_by_page_csv, student)
        problems = [by_problem[q] for q in [*PROBLEMS, "total", "maximum"]]
        return problems, [by_page[f"page {n}"] for n in range(1, 5)]

    # Partly blank earned 1, 0, 1 on Q1-Q3; 1.5 of 2, 3 of 3, 0 of 2 on Q4; 2 of 2,
    # 0 of 2, 0 of 2 on Q5. Q4.a: 1.5 x 1.5 / 2 = 1.125, 4.5 steps, a tie: 1.25.
    # Q4 = 1.25 + 2.5 + 0; Q5 = 1.5; pages 2 + 0 + 2, 1.25 + 2.5, 0 + 1.5, 0 + 0.
    assert cells(blank) == (
        ["2", "0", "2", "3.75", "1.5", "9.25", "15"],
        ["4", "3.75", "1.5", "0"],
    )
    # Photo per page earned everything but 1 of Q4.b's 3: 2.5 / 3 = 0.83, 3.33
    # steps: 0.75. Q4 = 1.5 + 0.75 + 1.5; Q5 = 1.5 + 1 + 1; page 2 = 1.5 + 0.75.
    assert cells(photo) == (
        ["2", "2", "2", "3.75", "3.5", "13.25", "15"],
        ["6", "2.25", "3", "2"],
    )
    # The same with Q4.b withheld: its problem and page are empty, never 0.
    assert cells(stale) == (
        ["2", "2", "2", "", "3.5", "12.5", "15"],
        ["6", "", "3", "2"],
    )
    assert cells(low) == (["", "", "", "", "", "0", "15"], ["", "", "", ""])
    feedback = workspace.feedback_md(blank.student_key).read_text()
    for line in [
        "Total: 9.25 / 15",
        "## Problem 4: 3.75 / 5.5",
        "**4(a): 1.25 / 1.5**",
    ]:
        assert f"\n{line}\n" in feedback
    assert "**4(b): not scored, out of 2.5**" in (
        workspace.feedback_md(stale.student_key).read_text()
    )
    # The detail stays in the criteria's own points.
    plain = new_workspace(tmp_path, "plain")
    for student in (blank, photo):
        copy_student(built, plain, built.truths[student.student_key].layout_class)
    write_report(plain, KEY, [blank, photo])
    named = (blank.student_name, photo.student_name)
    assert rows(plain.scores_detail_csv) == [
        r for r in rows(workspace.scores_detail_csv) if r["student"] in named
    ]
    text = workspace.run_report_md.read_text()
    assert (
        "\n\nPoints follow page maximums of 6, 4, 3, 2. A part's score is its points"
        " times the share of its criteria's points the student earned, rounded to the"
        " nearest 0.25, a tie rounding up. scores_detail.csv stays in the criteria's"
        " own points, the grader's units.\n\n| Part |" in text
    )
    assert "| Q4.a | 2 | 1.5 | verified |\n| Q4.b | 2 | 2.5 | verified |" in text
    assert "| 1 | 6 |\n| 2 | 4 |\n| 3 | 3 |\n| 4 | 2 |\n| Total | 15 |" in text


def test_page_maximums_equal_to_the_criteria_change_no_score(
    built: Built, tmp_path: Path
) -> None:
    shutil.copytree(built.workspace.root, tmp_path / "ws")
    workspace = Workspace(tmp_path / "ws")
    same = KEY.model_copy(
        update={"points": PointConstraint(pages=[Decimal(n) for n in (3, 5, 4, 4)])}
    )

    write_report(workspace, same, built.students)

    for name in SCORE_FILES:
        assert (workspace.root / "out" / name).read_bytes() == (
            built.workspace.root / "out" / name
        ).read_bytes()
    for path in built.workspace.feedback_dir.glob("*.md"):
        assert (workspace.feedback_dir / path.name).read_bytes() == path.read_bytes()


def test_setting_changing_and_removing_the_points_regrades_nobody(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(built.workspace.root, tmp_path / "ws")
    workspace = Workspace(tmp_path / "ws")
    summary = IngestSummary(
        students=built.students,
        duplicates=[],
        not_on_roster=[],
        missing_from_roster=[],
        ignored_files=0,
        unsafe_members=[],
        failed=[],
    )
    write_model(workspace.ingest_summary_json, summary)
    # No API key and no network: `report` needs neither.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled for this test")

    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    out = workspace.root / "out"

    def report(key: Key) -> dict[Path, bytes]:
        save_key(workspace.key_yaml, key)
        result = CliRunner().invoke(app, ["report", "--workspace", str(workspace.root)])
        assert result.exit_code == 0, result.output
        loaded = load_key(workspace.key_yaml)
        assert [part_hash(p, loaded.policies) for p in loaded.parts] == [
            part_hash(p, KEY.policies) for p in KEY.parts
        ]
        return {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}

    before = report(KEY)
    assert before == {
        out / p.relative_to(built.workspace.root / "out"): p.read_bytes()
        for p in (built.workspace.root / "out").rglob("*")
        if p.is_file()
    }
    total = KEY.model_copy(update={"points": PointConstraint(total=Decimal(20))})
    for key, maximum in [(PAGES_KEY, "15"), (total, "20")]:
        files = report(key)
        assert files != before
        # Every stored grade still counts: only the unreadable student is withheld.
        for row in rows(workspace.scores_by_problem_csv):
            assert row["maximum"] == maximum
            unread = row["student"] == student_of(built, "low-resolution").student_name
            assert (row["status"] == "incomplete") == unread
        assert not any(OTHER_KEY.encode() in text for text in files.values())
    assert report(KEY) == before


# --- the review list (task 4.8) ------------------------------------------------------


def entries(text: str) -> list[tuple[str, list[str]]]:
    """The review list's entries in order: each one's heading and its lines up to the
    next heading, blank lines left out."""
    found: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("### "):
            found.append((line.removeprefix("### "), []))
        elif found and line and not line.startswith("#"):
            found[-1][1].append(line)
    return found


def listed(workspace: Workspace) -> list[tuple[str, list[str]]]:
    return entries(workspace.review_md.read_text())


def students_of(built: Built, workspace: Workspace, count: int) -> list[IngestRecord]:
    """`count` copies of the typed-in-place student: `Student 1` (`s1`) and on."""
    return [
        copy_student(built, workspace, "typed-in-place", f"Student {n}", f"s{n}")
        for n in range(1, count + 1)
    ]


def poorly_read(
    workspace: Workspace, student: IngestRecord, part: str, scale: float = 0.5
) -> None:
    """`part` read with poor legibility: `hard_to_read`, and `low_resolution` too
    when every crop is at native size or above."""

    def poor(r: ReadingRecord) -> ReadingRecord:
        crops = [c.model_copy(update={"scale": scale}) for c in r.crops]
        reading = r.reading.model_copy(update={"legibility": "poor"})
        return r.model_copy(update={"reading": reading, "crops": crops})

    change_reading(workspace, student, part, poor)


def graded_with(
    workspace: Workspace, student: IngestRecord, part: str, **reasons: str
) -> None:
    """The grade of `part` with `reasons` (a transcript dispute, an overruling)."""

    def change(r: GradeRecord) -> GradeRecord:
        if r.grade.part_id != part:
            return r
        return r.model_copy(update={"grade": r.grade.model_copy(update=reasons)})

    change_grades(workspace, student, problem_of(part), change)


def entries_line(count: int) -> str:
    return f"Entries in the review list, out/review.md: {count}."


def test_with_no_flag_the_review_list_says_so_in_one_fixed_line(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")

    assert write_report(workspace, KEY, [student]) == []
    assert workspace.review_md.read_text() == f"# Review list\n\n{NOTHING_TO_REVIEW}\n"
    assert entries_line(0) in workspace.run_report_md.read_text().splitlines()


def test_a_part_with_two_flags_is_one_entry_and_two_students_are_two(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    students = students_of(built, workspace, 2)
    for student in students:
        poorly_read(workspace, student, "Q4.b", scale=2.0)

    raised = write_report(workspace, KEY, students)

    assert len(raised) == 4
    found = listed(workspace)
    assert [heading for heading, _ in found] == [
        "`Student 1` (`s1`), Q4.b",
        "`Student 2` (`s2`), Q4.b",
    ]
    for _, lines in found:
        assert [line for line in lines if line.startswith("- `")] == [
            f"- `hard_to_read`, medium: {DETAIL[F.HARD_TO_READ]}",
            f"- `low_resolution`, medium: {DETAIL[F.LOW_RESOLUTION]}",
        ]
    assert entries_line(2) in workspace.run_report_md.read_text().splitlines()


def unreadable(r: ReadingRecord) -> ReadingRecord:
    reading = r.reading.model_copy(update={"status": "unreadable"})
    crops = [c.model_copy(update={"scale": 0.5}) for c in r.crops]
    return r.model_copy(update={"reading": reading, "crops": crops})


def test_entries_come_by_severity_then_points_then_student_key_then_part_order(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    # Display names that sort the other way from the student keys.
    s1, s2, s3 = (
        copy_student(built, workspace, "typed-in-place", name, key)
        for name, key in [("Zed", "s1"), ("Yan", "s2"), ("Xia", "s3")]
    )
    # The key's parts reversed, so that its order is not the ids'. Q1 to Q3 are
    # worth 1, Q4.a and Q5.a 2, Q4.b 3.
    key = KEY.model_copy(
        update={
            "parts": [
                p.model_copy(update={"state": "unverified"})
                if p.id in ("Q2", "Q4.a")
                else p
                for p in reversed(KEY.parts)
            ]
        }
    )
    change_reading(workspace, s3, "Q2", unreadable)  # high
    # Medium and low on one part: it ranks as medium.
    graded_with(workspace, s1, "Q4.b", transcript_dispute="x", code_overruled="y")
    for student, part in [(s2, "Q4.a"), (s2, "Q5.a"), (s1, "Q4.a"), (s1, "Q1")]:
        poorly_read(workspace, student, part)  # medium
    graded_with(workspace, s3, "Q4.b", code_overruled="x")  # low

    write_report(workspace, key, [s1, s2, s3])

    found = listed(workspace)
    # At 2 points, s1's Q4.a before s2's Q5.a, which comes first in the key.
    assert [heading for heading, _ in found] == [
        "Q4.a, worth 2",
        "Q2, worth 1",
        "`Xia` (`s3`), Q2",
        "`Zed` (`s1`), Q4.b",
        "`Zed` (`s1`), Q4.a",
        "`Yan` (`s2`), Q5.a",
        "`Yan` (`s2`), Q4.a",
        "`Zed` (`s1`), Q1",
        "`Xia` (`s3`), Q4.b",
    ]
    # The key's own section, first: no transcript, and s3 is not scored on Q2.
    assert found[1][1] == [
        f"- `key_unverified`, high: {DETAIL[F.KEY_UNVERIFIED]}",
        "- Students scored on it: 2.",
        "- Files: none.",
    ]
    text = workspace.review_md.read_text()
    assert text.index("## The key") < text.index("## Students")


def test_under_page_maximums_the_score_and_the_order_use_the_part_s_points(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    # Q1 is worth 2 of page 1's 6 and its criteria 1; Q4.a 1.5 and its criteria 2.
    for part in ("Q1", "Q4.a"):
        poorly_read(workspace, student, part)

    write_report(workspace, PAGES_KEY, [student])

    found = listed(workspace)
    assert [heading.rsplit(", ", 1)[1] for heading, _ in found] == ["Q1", "Q4.a"]
    assert [
        [line for line in lines if line.startswith("- Proposed score:")]
        for _, lines in found
    ] == [["- Proposed score: 2 / 2."], ["- Proposed score: 1.5 / 1.5."]]


def test_an_entry_links_every_crop_and_only_the_files_stored(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")

    def twice(r: ReadingRecord) -> ReadingRecord:
        reading = r.reading.model_copy(update={"legibility": "poor"})
        crops = [c.model_copy(update={"scale": 0.5}) for c in r.crops * 2]
        return r.model_copy(update={"reading": reading, "crops": crops})

    change_reading(workspace, student, "Q4.b", twice)
    no_grade_file(workspace, student)  # Q5's parts: withheld, flagged incomplete

    write_report(workspace, KEY, [student])

    files = {
        heading.rsplit(", ", 1)[1]: [
            line for line in lines if line.startswith("- Files")
        ]
        for heading, lines in listed(workspace)
    }
    at = "../students/s1"
    assert files["Q4.b"] == [
        f"- Files: [crop 1]({at}/crops/Q4.b-1.png), [crop 2]({at}/crops/Q4.b-2.png),"
        f" [reading]({at}/readings/Q4.json), [grade]({at}/grades/Q4.json)."
    ]
    assert files["Q5.a"] == [
        f"- Files: [crop 1]({at}/crops/Q5.a-1.png), [reading]({at}/readings/Q5.json)."
    ]


def test_the_transcript_gives_what_the_reading_states_and_no_more(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    # Flagged by a dispute, so that each reading stays as stored: clear, no mark.
    for part in ("Q1", "Q4.a", "Q4.c", "Q5.a"):
        graded_with(workspace, student, part, transcript_dispute="x")

    write_report(workspace, KEY, [student])

    transcripts = {
        heading.rsplit(", ", 1)[1]: [line for line in lines if line.startswith("  - ")][
            1:
        ]
        for heading, lines in listed(workspace)
    }
    assert transcripts == {
        "Q1": ["  - Line 1: `C`", "  - Final answer: options `C`."],
        "Q4.a": [
            "  - Element 1: label `N`, kind `force`, direction `up`, position"
            " `starts on the dot`.",
            "  - Element 2: label `mg`, kind `force`, direction `down`, position"
            " `starts on the dot`.",
            "  - Element 3: label `T`, kind `force`, direction `right`, position"
            " `starts on the dot`.",
            "  - Element 4: label `f`, kind `force`, direction `left`, position"
            " `starts on the dot`.",
            "  - Final answer: none stated.",
        ],
        "Q4.c": [
            "  - Line 1: `It slows down and stops.`",
            "  - Line 2: `With the rope cut, friction is`",
            "  - Line 3: `the only horizontal force, and`",
            "  - Line 4: `it points against the motion.`",
            "  - Final answer: none stated.",
        ],
        "Q5.a": [
            "  - Line 1: `x² − 6x + 9 − 9 + 5`",
            "  - Line 2: `= (x − 3)² − 4`",
            "  - Final answer: expression `(x - 3)^2 - 4`.",
        ],
    }


def test_the_proposed_score_is_the_feedback_s_awarded_withheld_or_after_the_audit(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    shown, stale, changed = students_of(built, workspace, 3)
    graded_with(workspace, shown, "Q4.b", code_overruled="x")  # with no check
    other_key(workspace, stale)
    [part] = [p for p in KEY.parts if p.id == "Q4.b"]
    hashed = part_hash(part, KEY.policies)

    def change(student: str, before: int, after: int, c: str = "c1") -> AuditChange:
        return AuditChange(
            student_key=student,
            criterion=f"Q4.b.{c}",
            part_hash=hashed,
            before=Decimal(before),
            after=Decimal(after),
            reason="r",
        )

    # The class audit lowers s3's method on Q4.b from 2 to its partial level 1.
    # Its changes to s3's answer and to s1's method were decided on awards (0
    # and 1) their grades no longer hold, so they do not hold either.
    audit = PartAudit(
        part_id="Q4.b",
        part_hash=hashed,
        groups=1,
        changed=1,
        undecided=0,
        changes=[change("s1", 1, 0), change("s3", 2, 1), change("s3", 0, 1, "c2")],
        key_suspect=None,
    )
    write_model(workspace.audit_json("Q4.b"), audit)

    write_report(workspace, KEY, [shown, stale, changed])

    found = dict(listed(workspace))
    proposed = {
        heading: [line for line in lines if line.startswith("- Proposed score:")]
        for heading, lines in found.items()
    }
    assert proposed == {
        "`Student 1` (`s1`), Q4.b": ["- Proposed score: 3 / 3."],
        "`Student 2` (`s2`), Q4.b": [
            f"- Proposed score: not scored, out of 3. {OTHER_KEY}"
        ],
        "`Student 3` (`s3`), Q4.b": ["- Proposed score: 2 / 3."],
    }
    for student, line in [
        (shown, "**4(b): 3 / 3**"),
        (stale, "**4(b): not scored, out of 3**"),
        (changed, "**4(b): 2 / 3**"),
    ]:
        assert line in workspace.feedback_md(student.student_key).read_text()
    # What is stored behind each flag, and the audit file only where it holds.
    s1, s3 = found["`Student 1` (`s1`), Q4.b"], found["`Student 3` (`s3`), Q4.b"]
    assert s1[: s1.index("- Proposed score: 3 / 3.")] == [
        f"- `code_overruled`, low: {DETAIL[F.CODE_OVERRULED]}",
        "  - The grader wrote: `x`",
    ]
    assert s3[: s3.index("- Proposed score: 2 / 3.")] == [
        f"- `audit_changed`, low: {DETAIL[F.AUDIT_CHANGED]}",
        "  - Q4.b.c1: 2 before the audit, 1 after.",
    ]
    assert "audit" not in s1[-1] and "[audit](../audit/Q4.b.json)" in s3[-1]
    assert "audit" not in "".join(found["`Student 2` (`s2`), Q4.b"])


@pytest.mark.parametrize("stored", [True, False], ids=["stored", "not stored"])
def test_the_key_s_entry_links_its_verification_and_gives_the_audit_s_counts(
    built: Built, tmp_path: Path, stored: bool
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    key = KEY.model_copy(
        update={
            "parts": [
                p.model_copy(update={"state": "unverified"}) if p.id == "Q4.b" else p
                for p in KEY.parts
            ]
        }
    )
    if stored:
        workspace.verification_md.write_text("v")
        [part] = [p for p in key.parts if p.id == "Q4.b"]
        audit = PartAudit(
            part_id="Q4.b",
            part_hash=part_hash(part, key.policies),
            groups=0,
            changed=0,
            undecided=0,
            changes=[],
            key_suspect=Suspect(shared=8, answered=9),
        )
        write_model(workspace.audit_json("Q4.b"), audit)

    write_report(workspace, key, [student])

    # The key's flags alone: the key's entry, and no word that nothing needs review.
    text = workspace.review_md.read_text()
    assert NOTHING_TO_REVIEW not in text and "## Students" not in text
    unverified = f"- `key_unverified`, high: {DETAIL[F.KEY_UNVERIFIED]}"
    if stored:
        lines = [
            unverified,
            f"- `key_suspect`, high: {DETAIL[F.KEY_SUSPECT]}",
            "  - Students who answered: 9. Sharing one answer that is not the"
            " key's: 8.",
            "- Students scored on it: 1.",
            "- Files: [verification](../key/verification.md),"
            " [audit](../audit/Q4.b.json).",
        ]
    else:
        lines = [unverified, "- Students scored on it: 1.", "- Files: none."]
    assert listed(workspace) == [("Q4.b, worth 3", lines)]


FIRST = [
    "  - Line 1: `f = µmg = 0.25(4.0)(10) = 10 N`",
    "  - Line 2: `Fnet = T − f = 20 − 10 = 10 N`",
    "  - Line 3: `a = Fnet/m = 10/4.0 = 2.5 m/s²`",
    "  - Final answer: value `2.5 m/s^2`.",
]
SECOND = [
    "- Second reading, by another vendor's model:",
    "  - Line 1: `a = 25 m/s^2`",
    "  - Final answer: value `25.0 m/s^2`.",
]
POOR = ["  - The reader marked the legibility as poor."]


@pytest.mark.parametrize(
    ("result", "doubted", "shown"),
    [
        ("differ", True, POOR + FIRST + SECOND),
        ("differ", False, FIRST + SECOND),
        ("not decided", True, POOR + FIRST),
    ],
    ids=["differ", "differ on a part not in doubt", "not decided"],
)
def test_the_second_reader_s_text_shows_only_where_the_readers_disagree(
    built: Built,
    tmp_path: Path,
    result: Literal["differ", "not decided"],
    doubted: bool,
    shown: list[str],
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    if doubted:
        poorly_read(workspace, student, "Q4.b")
    first = load(workspace, KEY, student).readings["Q4.b"]
    stated = first.reading.final_answer.model_copy(update={"value": 25.0})
    reading = first.reading.model_copy(
        update={"lines": ["a = 25 m/s^2"], "final_answer": stated, "legibility": None}
    )
    [part] = [p for p in KEY.parts if p.id == "Q4.b"]
    record = SecondReading(
        part_id="Q4.b",
        part_hash=part_hash(part, KEY.policies),
        first=digest(first),
        doubted=doubted,
        result=result,
        second=first.model_copy(update={"reading": reading}),
    )
    write_model(
        workspace.second_json(student.student_key), SecondReadings(records=[record])
    )

    write_report(workspace, KEY, [student])

    [(_, lines)] = listed(workspace)
    transcript = lines[lines.index("- First reading:") + 1 :]
    assert transcript[:-1] == shown
    assert transcript[-1].startswith("- Files: ")


def test_a_flag_on_a_whole_submission_is_one_entry_with_no_transcript(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    s1, s2 = students_of(built, workspace, 2)
    poorly_read(workspace, s1, "Q4.b")
    stored = [load(workspace, KEY, s) for s in (s1, s2)]
    # No code raises one yet: by hand, of medium severity like s1's.
    whole = Flag(
        code=F.UNCLAIMED_WORK,
        part_id=None,
        student_key="s2",
        detail="Some work is on a page that no part claims.",
    )
    raised = [*flags(KEY, stored), whole]

    text = review(workspace, KEY, [(s, score(KEY, s)) for s in stored], raised)

    # Both medium: a whole submission is at stake for the key's 16 points.
    found = entries(text)
    assert found[0] == (
        "`Student 2` (`s2`), the whole submission",
        [f"- `unclaimed_work`, medium: {whole.detail}"],
    )
    assert [heading for heading, _ in found[1:]] == ["`Student 1` (`s1`), Q4.b"]
    assert review_line(workspace, raised) == entries_line(2)


HOSTILE_NAME = "# Total: 16 / 16\n[x](http://x.test) | a | b |\n```\n<b>bold</b>"
HOSTILE_LINES = [
    "# Total: 16 / 16",
    "[x](http://x.test)",
    "| a | b |",
    "|---|---|",
    "```",
    "<b>bold</b>",
    "two\nlines",
]
HOSTILE_ELEMENT = DiagramElement(
    label="# label", kind="| a | b |", direction="<!--", position="``` up\nthen"
)


def test_text_a_student_or_a_model_wrote_cannot_leave_its_place_in_the_entry(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", HOSTILE_NAME, "s1")

    def hostile(r: ReadingRecord) -> ReadingRecord:
        stated = FinalAnswer(
            choices=["<b>B</b>"],
            value=2.5,
            unit="```\n# m",
            expression="[x](http://x.test)",
        )
        reading = r.reading.model_copy(
            update={
                "lines": HOSTILE_LINES,
                "final_answer": stated,
                "elements": [HOSTILE_ELEMENT],
                "uncertain": [1],
                "addressed_to_grader": [2],
                "legibility": "poor",
            }
        )
        crops = [c.model_copy(update={"scale": 0.5}) for c in r.crops]
        return r.model_copy(update={"reading": reading, "crops": crops})

    def disputed(r: GradeRecord) -> GradeRecord:
        if r.grade.part_id != "Q4.b":
            return r
        grade = r.grade.model_copy(
            update={
                "transcript_dispute": "# Total: 16 / 16\n<b>x</b>",
                "code_overruled": "[click](http://x.test)",
            }
        )
        check = AnswerCheck(verdict="does not match", reason="| a | b |")
        return r.model_copy(update={"grade": grade, "check": check})

    change_reading(workspace, student, "Q4.b", hostile)
    change_grades(workspace, student, "Q4", disputed)
    write_report(workspace, KEY, [student])

    lines = workspace.review_md.read_text().splitlines()
    starts = ("# Review list", "## ", "### ", "- ", "  - ")
    assert all(line.startswith(starts) for line in lines if line)
    assert [line for line in lines if line.startswith("#")] == [
        "# Review list",
        "## Students",
        "### ````# Total: 16 / 16?[x](http://x.test) | a | b |?```?<b>bold</b>````"
        " (`s1`), Q4.b",
    ]
    # High first: addressed_to_grader is raised after hard_to_read.
    assert [line for line in lines if line.startswith("- `")] == [
        f"- `addressed_to_grader`, high: {DETAIL[F.ADDRESSED_TO_GRADER]}",
        f"- `hard_to_read`, medium: {DETAIL[F.HARD_TO_READ]}",
        f"- `transcript_dispute`, medium: {DETAIL[F.TRANSCRIPT_DISPUTE]}",
        f"- `code_overruled`, low: {DETAIL[F.CODE_OVERRULED]}",
    ]
    assert [line for line in lines if line.startswith("  - ")] == [
        "  - The grader wrote: `# Total: 16 / 16?<b>x</b>`",
        "  - The answer check: does not match, `| a | b |`",
        "  - The grader wrote: `[click](http://x.test)`",
        "  - The reader marked the legibility as poor.",
        "  - Line 1: `# Total: 16 / 16` (the reader was unsure of this line)",
        "  - Line 2: `[x](http://x.test)` (addressed to the grader)",
        "  - Line 3: `| a | b |`",
        "  - Line 4: `|---|---|`",
        "  - Line 5: ```` ``` ````",
        "  - Line 6: `<b>bold</b>`",
        "  - Line 7: `two?lines`",
        "  - Element 1: label `# label`, kind `| a | b |`, direction `<!--`,"
        " position ```` ``` up?then ````.",
        "  - Final answer: options `<b>B</b>`; value ````2.5 ```?# m````; expression"
        " `[x](http://x.test)`.",
    ]
    assert code_span(HOSTILE_NAME) in lines[4]


def test_every_link_opens_a_stored_file_from_out(built: Built) -> None:
    text = built.workspace.review_md.read_text()
    out = built.workspace.review_md.parent
    targets = re.findall(r"\]\(([^)]*)\)", text)

    assert all(t.startswith("../students/") and (out / t).is_file() for t in targets)
    assert {Path(t).parent.name for t in targets} == {"crops", "readings", "grades"}
    assert "![" not in text  # links, not images


MARK = "MARKER-48 written by a student"


def test_report_gives_the_review_list_s_size_and_no_student_text_on_the_terminal(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", MARK, "s1")

    def marked(r: ReadingRecord) -> ReadingRecord:
        reading = r.reading.model_copy(update={"lines": [MARK], "legibility": "poor"})
        return r.model_copy(update={"reading": reading})

    change_reading(workspace, student, "Q4.b", marked)
    save_key(workspace.key_yaml, KEY)
    summary = IngestSummary(
        students=[student],
        duplicates=[],
        not_on_roster=[],
        missing_from_roster=[],
        ignored_files=0,
        unsafe_members=[],
        failed=[],
    )
    write_model(workspace.ingest_summary_json, summary)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = CliRunner().invoke(app, ["report", "--workspace", str(workspace.root)])

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"Wrote {workspace.root / 'out'}",
        entries_line(1),
    ]
    assert result.stderr == ""
    assert entries_line(1) in workspace.run_report_md.read_text().splitlines()
    assert MARK in workspace.review_md.read_text()
    assert not workspace.run_log.exists()


# --- the instructor's corrections ----------------------------------------------------


def correction(
    student: IngestRecord,
    criterion: str,
    points: str,
    note: str | None = None,
    part: str | None = None,
) -> Override:
    """A correction made under `KEY`'s version of the criterion's part, or of
    `part` when given."""
    made_for = part or criterion.rsplit(".", 1)[0]
    [found] = [p for p in KEY.parts if p.id == made_for]
    return Override(
        student=student.student_key,
        criterion=criterion,
        points=Decimal(points),
        part_hash=part_hash(found, KEY.policies),
        note=note,
    )


def corrections(workspace: Workspace, *entries: Override) -> None:
    """Write `entries` to overrides.yaml, as a person might."""
    text = io.StringIO()
    yaml_writer().dump([o.model_dump(exclude_none=True) for o in entries], text)
    workspace.overrides_yaml.write_text(text.getvalue())


def disputed(r: GradeRecord) -> GradeRecord:
    """Q4.b's grade disputing its reading: a flag, so the part has a review entry."""
    if r.grade.part_id != "Q4.b":
        return r
    return r.model_copy(
        update={"grade": r.grade.model_copy(update={"transcript_dispute": "d"})}
    )


def detail_of(workspace: Workspace, criterion: str) -> tuple[str, str]:
    [row] = [
        r for r in rows(workspace.scores_detail_csv) if r["criterion"] == criterion
    ]
    return row["awarded"], row["rationale"]


def entry_of(workspace: Workspace, part: str) -> list[str]:
    [found] = [lines for head, lines in listed(workspace) if head.endswith(f", {part}")]
    return found


def test_a_correction_is_the_award_in_every_file_and_the_totals_follow(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    change_grades(workspace, student, "Q4", disputed)
    write_report(workspace, KEY, [student])
    problem = row_of(workspace.scores_by_problem_csv, student)
    page = row_of(workspace.scores_by_page_csv, student)
    assert detail_of(workspace, "Q4.b.c1")[0] == "2"
    corrections(workspace, correction(student, "Q4.b.c1", "0.5", "See line 2."))

    write_report(workspace, KEY, [student])

    after = row_of(workspace.scores_by_problem_csv, student)
    lost = Decimal("1.5")
    for column in ("Q4", "total"):
        assert Decimal(after[column]) == Decimal(problem[column]) - lost
    by_page = row_of(workspace.scores_by_page_csv, student)
    assert Decimal(by_page["page 2"]) == Decimal(page["page 2"]) - lost
    assert detail_of(workspace, "Q4.b.c1") == ("0.5", CORRECTED)
    feedback = workspace.feedback_md(student.student_key).read_text().splitlines()
    assert "- Marked by your instructor: 0.5 / 2. See line 2." in feedback
    assert "- Feedback on Q4.b.c1." not in feedback
    assert "**4(b): 1.5 / 3**" in feedback
    entry = entry_of(workspace, "Q4.b")
    assert "- Proposed score: 1.5 / 3." in entry
    assert (
        "- Corrected: Q4.b.c1, the grader's award 2, the instructor's 0.5."
        " Note: `See line 2.`"
    ) in entry


def test_removing_a_correction_gives_back_the_grader_s_award(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    write_report(workspace, KEY, [student])
    before = {p: p.read_bytes() for p in (workspace.root / "out").rglob("*.*")}
    corrections(workspace, correction(student, "Q4.b.c1", "1", "A note."))
    write_report(workspace, KEY, [student])
    assert detail_of(workspace, "Q4.b.c1") == ("1", CORRECTED)

    corrections(workspace)
    write_report(workspace, KEY, [student])

    assert {p: p.read_bytes() for p in (workspace.root / "out").rglob("*.*")} == before


def test_a_correction_wins_over_the_class_audit_and_closes_no_flag(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    hashed = part_hash(next(p for p in KEY.parts if p.id == "Q4.b"), KEY.policies)
    change = AuditChange(
        student_key=student.student_key,
        criterion="Q4.b.c1",
        part_hash=hashed,
        before=Decimal(2),
        after=Decimal(1),
        reason="r",
    )
    write_model(
        workspace.audit_json("Q4.b"),
        PartAudit(
            part_id="Q4.b",
            part_hash=hashed,
            groups=1,
            changed=1,
            undecided=0,
            changes=[change],
            key_suspect=None,
        ),
    )
    # The instructor gives back the award the audit lowered: applied before the
    # audit's change, it would meet the change's "before" and be lowered again.
    corrections(workspace, correction(student, "Q4.b.c1", "2"))

    raised = write_report(workspace, KEY, [student])

    assert detail_of(workspace, "Q4.b.c1") == ("2", CORRECTED)
    assert [(f.code, f.part_id) for f in raised] == [(F.AUDIT_CHANGED, "Q4.b")]
    entry = entry_of(workspace, "Q4.b")
    assert "  - Q4.b.c1: 2 before the audit, 1 after." in entry
    assert "- Corrected: Q4.b.c1, the grader's award 1, the instructor's 2." in entry
    assert "- Proposed score: 3 / 3." in entry


@pytest.mark.parametrize("corrected", [["Q4.b.c1", "Q4.b.c2"], ["Q4.b.c2"]])
def test_a_withheld_part_is_scored_once_every_criterion_is_corrected(
    built: Built, tmp_path: Path, corrected: list[str]
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")

    def unreadable_q4b(r: ReadingRecord) -> ReadingRecord:
        read = r.reading.model_copy(update={"status": "unreadable", "lines": []})
        return r.model_copy(update={"reading": read})

    change_reading(workspace, student, "Q4.b", unreadable_q4b)
    change_grades(
        workspace, student, "Q4", lambda r: None if r.grade.part_id == "Q4.b" else r
    )
    corrections(workspace, *(correction(student, c, "1") for c in corrected))

    raised = write_report(workspace, KEY, [student])

    row = row_of(workspace.scores_by_problem_csv, student)
    assert F.UNREADABLE in {f.code for f in raised if f.part_id == "Q4.b"}
    feedback = workspace.feedback_md(student.student_key).read_text().splitlines()
    if len(corrected) == 2:
        assert (row["Q4"], row["status"]) == ("6", "provisional")
        assert detail_of(workspace, "Q4.b.c1") == ("1", CORRECTED)
        assert detail_of(workspace, "Q4.b.c2") == ("1", CORRECTED)
        assert "- Marked by your instructor: 1 / 2." in feedback
        assert "- Marked by your instructor: 1 / 1." in feedback
        assert "- Proposed score: 2 / 3." in entry_of(workspace, "Q4.b")
        return
    assert (row["Q4"], row["status"]) == ("", "incomplete")
    # How many criteria are open is for the instructor's two files only.
    why = f"{UNREADABLE} Criteria with no correction from the instructor: 1 of 2."
    assert detail_of(workspace, "Q4.b.c1") == ("", UNREADABLE)
    assert detail_of(workspace, "Q4.b.c2") == ("", UNREADABLE)
    assert f"- {UNREADABLE}" in feedback
    assert "instructor" not in "\n".join(feedback)
    entry = entry_of(workspace, "Q4.b")
    assert f"- Proposed score: not scored, out of 3. {why}" in entry
    assert "- Corrected: Q4.b.c2, the grader's award none, the instructor's 1." in entry
    assert f"- `{student.student_key}`: Q4.b. {why}" in (
        workspace.run_report_md.read_text().splitlines()
    )


def test_a_correction_made_under_another_version_of_its_part_is_listed_not_applied(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    edited = KEY.model_copy(
        update={
            "parts": [
                p.model_copy(
                    update={
                        "criteria": [
                            p.criteria[0].model_copy(update={"requirement": "Edited."}),
                            *p.criteria[1:],
                        ]
                    }
                )
                if p.id == "Q4.b"
                else p
                for p in KEY.parts
            ]
        }
    )
    # Graded again under the edited key, as regrade would.
    truth = built.truths[student.student_key]
    store_grades(
        workspace,
        student.student_key,
        [graded(with_feedback(g), edited) for g in truth.grades],
        edited,
    )
    corrections(
        workspace,
        correction(student, "Q4.b.c1", "0.5"),  # made before the edit
        correction(student, "Q5.a.c1", "1", part="Q4.a"),  # made for another part
        correction(student, "Q4.a.c1", "5", part="Q4.b"),  # above full, but stale
    )

    write_report(workspace, edited, [student])

    assert detail_of(workspace, "Q4.b.c1")[0] == "2"
    assert detail_of(workspace, "Q5.a.c1")[0] == "2"
    section = [
        "## Corrections not applied",
        "",
        STALE,
        "",
        f"- `{student.student_key}`, Q4.b.c1: 0.5.",
        f"- `{student.student_key}`, Q5.a.c1: 1.",
        f"- `{student.student_key}`, Q4.a.c1: 5.",
    ]
    for path in (workspace.run_report_md, workspace.review_md):
        text = path.read_text().splitlines()
        start = text.index(section[0])
        assert text[start : start + len(section)] == section, path


HOSTILE_NOTE = (
    "Fine.\n# Total: 16 / 16\n<!-- x --> `` ``` [x](http://x.test)\n| a |"
    " \u202eRTL\x00end\u200b"
)


def test_a_note_cannot_leave_its_place_in_the_feedback_file_or_the_review_list(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    change_grades(workspace, student, "Q4", disputed)
    corrections(workspace, correction(student, "Q4.b.c1", "1", HOSTILE_NOTE))

    write_report(workspace, KEY, [student])

    feedback = workspace.feedback_md(student.student_key).read_text().splitlines()
    [line] = [x for x in feedback if "Marked by your instructor" in x]
    assert line == "- Marked by your instructor: 1 / 2. " + one_line(HOSTILE_NOTE)
    assert line.endswith("| a | RTLend")
    heads = [x.split(":")[0] for x in feedback if x.startswith("#")]
    assert heads == ["# ws", *(f"## Problem {q[1:]}" for q in PROBLEMS)]
    assert all(
        x.startswith(("# ws", "## Problem ", "**", "- ", "Total: "))
        for x in feedback
        if x
    )
    entry = entry_of(workspace, "Q4.b")
    [noted] = [x for x in entry if x.startswith("- Corrected:")]
    assert noted == (
        "- Corrected: Q4.b.c1, the grader's award 2, the instructor's 1. Note: "
        + code_span(HOSTILE_NOTE)
    )
    review = workspace.review_md.read_text().splitlines()
    starts = ("# Review list", "## ", "### ", "- ", "  - ")
    assert all(x.startswith(starts) for x in review if x)
    assert [x for x in review if x.startswith("#")] == [
        "# Review list",
        "## Students",
        f"### `{student.student_name}` (`{student.student_key}`), Q4.b",
    ]


def test_under_a_point_constraint_a_correction_s_line_gives_no_points(
    built: Built, tmp_path: Path
) -> None:
    constrained = KEY.model_copy(update={"points": PointConstraint(total=Decimal(8))})
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place")
    corrections(workspace, correction(student, "Q4.b.c1", "1", "A note."))

    write_report(workspace, constrained, [student])

    feedback = workspace.feedback_md(student.student_key).read_text()
    part = feedback.split("**4(b)")[1].split("**4(c)")[0]
    assert part.splitlines()[1:] == ["- Marked by your instructor. A note.", ""]
    assert detail_of(workspace, "Q4.b.c1") == ("1", CORRECTED)  # its own points


# --- the instructor's reviews (`redpen review`) -----------------------------------


def review_of(
    workspace: Workspace,
    students: Sequence[IngestRecord],
    place: tuple[str | None, str | None],
    key: Key = KEY,
) -> None:
    """Record the review of the entry at `place` as it stands, as `redpen review`'s
    accept does."""
    [entry] = [
        e
        for e in review_entries(workspace, key, students)
        if (e.review.student, e.review.part) == place
    ]
    keys = [s.student_key for s in students]
    record_review(workspace.reviews_yaml, key, keys, entry.review)


def section(path: Path, heading: str) -> list[str]:
    """The lines of `path`'s section `heading`, up to the next, blank lines left
    out; none when it has no such section."""
    lines = path.read_text().splitlines()
    if heading not in lines:
        return []
    at = lines.index(heading) + 1
    end = next((n for n in range(at, len(lines)) if lines[n].startswith("## ")), None)
    return [line for line in lines[at:end] if line]


def open_entries(workspace: Workspace) -> list[str]:
    """The headings of the review list's open entries."""
    return [heading for heading, _ in listed(workspace)]


def test_a_review_closes_its_entry_s_flags_and_every_file_follows(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    change_grades(workspace, student, "Q4", disputed)  # medium
    graded_with(workspace, student, "Q5.a", code_overruled="y")  # low
    write_report(workspace, KEY, [student])
    row = row_of(workspace.scores_by_problem_csv, student)
    assert (row["status"], row["open_flags"]) == ("provisional", "2")
    kept = {
        p: p.read_bytes()
        for p in (
            workspace.feedback_md("s1"),
            workspace.scores_by_page_csv,
            workspace.scores_detail_csv,
        )
    }

    review_of(workspace, [student], ("s1", "Q4.b"))
    raised = write_report(workspace, KEY, [student])

    assert [(f.code, f.part_id) for f in raised] == [(F.CODE_OVERRULED, "Q5.a")]
    row = row_of(workspace.scores_by_problem_csv, student)
    assert (row["status"], row["open_flags"]) == ("final", "1")
    assert {p: p.read_bytes() for p in kept} == kept  # the score stands
    assert open_entries(workspace) == ["`A Student` (`s1`), Q5.a"]
    # A closed flag is not deleted: listed last, in one line.
    lines = workspace.review_md.read_text().splitlines()
    assert lines[lines.index("## Closed by review") :] == [
        "## Closed by review",
        "",
        CLOSED,
        "",
        "- `s1`, Q4.b: `transcript_dispute`. Proposed score: 3 / 3.",
    ]
    report = workspace.run_report_md.read_text().splitlines()
    assert "| transcript_dispute | medium | 1 |" in report  # raised, and counted
    assert "Flags raised: 2. Closed by review: 1. Open: 1." in report
    assert entries_line(1) in report
    assert section(workspace.review_md, "## Reviews out of date") == []


def test_with_every_entry_reviewed_the_list_says_nothing_needs_review(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    change_grades(workspace, student, "Q4", disputed)
    review_of(workspace, [student], ("s1", "Q4.b"))

    assert write_report(workspace, KEY, [student]) == []

    lines = workspace.review_md.read_text().splitlines()
    assert lines[:3] == ["# Review list", "", NOTHING_TO_REVIEW]
    assert section(workspace.review_md, "## Closed by review") == [
        CLOSED,
        "- `s1`, Q4.b: `transcript_dispute`. Proposed score: 3 / 3.",
    ]
    assert entries_line(0) in workspace.run_report_md.read_text().splitlines()


# What happens to the stored files before the review and after it: each `Step`
# returns the key to report with, or None for the one before.
type Step = Callable[[Built, Workspace, IngestRecord], Key | None]


def unchanged(built: Built, workspace: Workspace, student: IngestRecord) -> None:
    pass


def lowered(criterion: str) -> Step:
    """Q4.b graded again, with 0 on `criterion`."""

    def step(built: Built, workspace: Workspace, student: IngestRecord) -> None:
        def change(r: GradeRecord) -> GradeRecord:
            criteria = [
                c.model_copy(update={"awarded": Decimal(0)}) if c.id == criterion else c
                for c in r.grade.criteria
            ]
            grade = r.grade.model_copy(update={"criteria": criteria})
            return r.model_copy(update={"grade": grade})

        change_grades(workspace, student, "Q4", change)

    return step


def corrected(criterion: str) -> Step:
    def step(built: Built, workspace: Workspace, student: IngestRecord) -> None:
        corrections(workspace, correction(student, criterion, "1"))

    return step


def uncorrected(built: Built, workspace: Workspace, student: IngestRecord) -> None:
    corrections(workspace)


def overruled(built: Built, workspace: Workspace, student: IngestRecord) -> None:
    graded_with(workspace, student, "Q4.b", code_overruled="y")  # a flag more


def edited(key: Key) -> Key:
    """`key` with the requirement of Q4.b's first criterion edited."""
    return key.model_copy(
        update={
            "parts": [
                p.model_copy(
                    update={
                        "criteria": [
                            p.criteria[0].model_copy(update={"requirement": "Edited."}),
                            *p.criteria[1:],
                        ]
                    }
                )
                if p.id == "Q4.b"
                else p
                for p in key.parts
            ]
        }
    )


def regraded(
    built: Built, workspace: Workspace, student: IngestRecord, key: Key
) -> None:
    """The copied typed-in-place `student` graded again under `key`, as regrade
    would."""
    truth = built.truths[student_of(built, "typed-in-place").student_key]
    grades = [graded(with_feedback(g), key) for g in truth.grades]
    store_grades(workspace, student.student_key, grades, key)


def key_edited(built: Built, workspace: Workspace, student: IngestRecord) -> Key:
    """Q4.b's first criterion edited, and the student graded again under it."""
    key = edited(KEY)
    regraded(built, workspace, student, key)
    change_grades(workspace, student, "Q4", disputed)
    return key


def unread_q4b(built: Built, workspace: Workspace, student: IngestRecord) -> None:
    """Q4.b read as unreadable, so withheld, with no grade."""
    change_reading(workspace, student, "Q4.b", unreadable)
    change_grades(
        workspace, student, "Q4", lambda r: None if r.grade.part_id == "Q4.b" else r
    )


def read_again(line: str) -> Step:
    """Q4.b read a second time as `line`, not decided against the first: no flag."""

    def step(built: Built, workspace: Workspace, student: IngestRecord) -> None:
        first = load(workspace, KEY, student).readings["Q4.b"]
        reading = first.reading.model_copy(update={"lines": [line]})
        [part] = [p for p in KEY.parts if p.id == "Q4.b"]
        record = SecondReading(
            part_id="Q4.b",
            part_hash=part_hash(part, KEY.policies),
            first=digest(first),
            doubted=False,
            result="not decided",
            second=first.model_copy(update={"reading": reading}),
        )
        path = workspace.second_json(student.student_key)
        write_model(path, SecondReadings(records=[record]))

    return step


def graded_again(dispute: str) -> Step:
    """Q4.b graded a second time with the same awards, the second grader disputing
    the reading in `dispute`: no flag."""

    def step(built: Built, workspace: Workspace, student: IngestRecord) -> None:
        record = second_grade_of_q4b(load(workspace, KEY, student), "same")
        assert record.second
        grade = record.second.grade.model_copy(update={"transcript_dispute": dispute})
        second = record.second.model_copy(update={"grade": grade})
        path = workspace.second_grade_json(student.student_key)
        write_model(
            path, SecondGrades(records=[record.model_copy(update={"second": second})])
        )

    return step


def reread(built: Built, workspace: Workspace, student: IngestRecord) -> None:
    """Q4.b's reading stored anew with another line, its grade kept."""

    def anew(r: ReadingRecord) -> ReadingRecord:
        lines = [*r.reading.lines, "Read anew."]
        return r.model_copy(
            update={"reading": r.reading.model_copy(update={"lines": lines})}
        )

    change_reading(workspace, student, "Q4.b", anew)


def made_from_another_reading(
    built: Built, workspace: Workspace, student: IngestRecord
) -> None:
    """Q4.b's grade under the key now, made from another reading: withheld for
    another reason than before."""
    [part] = [p for p in KEY.parts if p.id == "Q4.b"]
    graded_from(
        workspace, student, earlier, key_part_hash=part_hash(part, KEY.policies)
    )


def crop(data: bytes) -> Step:
    """Q4.b's crop stored as `data`, its reading and grade as they are: another
    image read as before."""

    def step(built: Built, workspace: Workspace, student: IngestRecord) -> None:
        path = workspace.crop_png(student.student_key, "Q4.b", 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    return step


def audit_reason(reason: str) -> Step:
    """The class audit of Q4.b lowering the student's Q4.b.c1 from 2 to 1 for
    `reason`."""

    def step(built: Built, workspace: Workspace, student: IngestRecord) -> None:
        hashed = part_hash(next(p for p in KEY.parts if p.id == "Q4.b"), KEY.policies)
        change = AuditChange(
            student_key=student.student_key,
            criterion="Q4.b.c1",
            part_hash=hashed,
            before=Decimal(2),
            after=Decimal(1),
            reason=reason,
        )
        audit = PartAudit(
            part_id="Q4.b",
            part_hash=hashed,
            groups=1,
            changed=1,
            undecided=0,
            changes=[change],
            key_suspect=None,
        )
        write_model(workspace.audit_json("Q4.b"), audit)

    return step


OUT_OF_DATE: dict[str, tuple[Step, Step]] = {
    "a crop of it replaced": (crop(b"one image"), crop(b"another image")),
    "the audit's reason only": (audit_reason("One."), audit_reason("Two.")),
    "the part read again otherwise": (unchanged, reread),
    "a re-grade that changes an award": (unchanged, lowered("Q4.b.c2")),
    "a re-grade under a correction": (corrected("Q4.b.c1"), lowered("Q4.b.c1")),
    "a new flag on the part": (unchanged, overruled),
    "an edit to the key part": (unchanged, key_edited),
    "a correction added": (unchanged, corrected("Q4.b.c1")),
    "a correction removed": (corrected("Q4.b.c1"), uncorrected),
    "a correction added to a part withheld": (unread_q4b, corrected("Q4.b.c2")),
    "a changed second reading": (read_again("a = 2.5"), read_again("a = 25")),
    "a changed second grade": (graded_again("One."), graded_again("Two.")),
    "another reason the part is withheld": (
        lambda built, workspace, student: other_key(workspace, student),
        made_from_another_reading,
    ),
}


@pytest.mark.parametrize(
    ("before", "after"), OUT_OF_DATE.values(), ids=OUT_OF_DATE.keys()
)
def test_a_review_stops_closing_its_flags_once_its_entry_shows_something_else(
    built: Built, tmp_path: Path, before: Step, after: Step
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    change_grades(workspace, student, "Q4", disputed)
    key = before(built, workspace, student) or KEY
    review_of(workspace, [student], ("s1", "Q4.b"), key)
    assert write_report(workspace, key, [student]) == []

    key = after(built, workspace, student) or key
    raised = write_report(workspace, key, [student])

    assert {f.part_id for f in raised} == {"Q4.b"}
    assert open_entries(workspace) == ["`A Student` (`s1`), Q4.b"]
    for path in (workspace.review_md, workspace.run_report_md):
        assert section(path, "## Reviews out of date") == [OUTDATED, "- `s1`, Q4.b."]
    assert section(workspace.review_md, "## Closed by review") == []
    row = row_of(workspace.scores_by_problem_csv, student)
    assert row["status"] in ("provisional", "incomplete")
    assert row["open_flags"] == str(len(raised))


def test_a_review_stops_closing_once_a_later_version_raises_another_flag_on_it(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    change_grades(workspace, student, "Q4", disputed)
    review_of(workspace, [student], ("s1", "Q4.b"))
    flagged = reporting._part_flags

    def more(key: Key, part: KeyPart, stored: Stored) -> list[tuple[FlagCode, str]]:
        found = flagged(key, part, stored)
        extra = [(F.HARD_TO_READ, DETAIL[F.HARD_TO_READ])] if part.id == "Q4.b" else []
        return found + extra

    monkeypatch.setattr(reporting, "_part_flags", more)

    raised = write_report(workspace, KEY, [student])

    assert {f.code for f in raised} == {F.TRANSCRIPT_DISPUTE, F.HARD_TO_READ}
    assert section(workspace.review_md, "## Reviews out of date") == [
        OUTDATED,
        "- `s1`, Q4.b.",
    ]


def test_a_review_of_an_entry_no_longer_flagged_is_not_listed(
    built: Built, tmp_path: Path
) -> None:
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    change_grades(workspace, student, "Q4", disputed)
    review_of(workspace, [student], ("s1", "Q4.b"))
    change_grades(  # graded again with no dispute: no flag left
        workspace,
        student,
        "Q4",
        lambda r: r.model_copy(
            update={"grade": r.grade.model_copy(update={"transcript_dispute": None})}
        ),
    )

    assert write_report(workspace, KEY, [student]) == []

    for path in (workspace.review_md, workspace.run_report_md):
        assert "## Reviews out of date" not in path.read_text()
    assert "## Closed by review" not in workspace.review_md.read_text()


@pytest.mark.parametrize("change", ["the part edited", "another finding"])
def test_a_key_part_s_review_closes_its_flags_for_every_student_while_it_holds(
    built: Built, tmp_path: Path, change: str
) -> None:
    workspace = new_workspace(tmp_path)
    students = students_of(built, workspace, 2)
    key = KEY.model_copy(
        update={
            "parts": [
                p.model_copy(update={"state": "unverified"}) if p.id == "Q4.b" else p
                for p in KEY.parts
            ]
        }
    )

    def suspect(key: Key, shared: int) -> None:
        [part] = [p for p in key.parts if p.id == "Q4.b"]
        audit = PartAudit(
            part_id="Q4.b",
            part_hash=part_hash(part, key.policies),
            groups=0,
            changed=0,
            undecided=0,
            changes=[],
            key_suspect=Suspect(shared=shared, answered=9),
        )
        write_model(workspace.audit_json("Q4.b"), audit)

    suspect(key, 8)
    write_report(workspace, key, students)
    assert {r["status"] for r in rows(workspace.scores_by_problem_csv)} == {
        "provisional"
    }
    review_of(workspace, students, (None, "Q4.b"), key)

    assert write_report(workspace, key, students) == []

    found = rows(workspace.scores_by_problem_csv)
    assert {(r["status"], r["open_flags"]) for r in found} == {("final", "0")}
    assert section(workspace.review_md, "## Closed by review") == [
        CLOSED,
        "- The key, Q4.b: `key_unverified`, `key_suspect`. Worth 3.",
    ]
    if change == "the part edited":  # graded again under it, the finding the same
        key = edited(key)
        for student in students:
            regraded(built, workspace, student, key)
        suspect(key, 8)
    else:
        suspect(key, 7)

    raised = write_report(workspace, key, students)

    assert {(f.code, f.student_key) for f in raised} == {
        (F.KEY_UNVERIFIED, None),
        (F.KEY_SUSPECT, None),
    }
    assert {r["status"] for r in rows(workspace.scores_by_problem_csv)} == {
        "provisional"
    }
    for path in (workspace.review_md, workspace.run_report_md):
        assert section(path, "## Reviews out of date") == [OUTDATED, "- The key, Q4.b."]


def ready(
    workspace: Workspace, students: Sequence[IngestRecord], key: Key = KEY
) -> None:
    """`workspace` as a command takes it: the key saved, `students` the latest
    ingest, and the workspace kept out of git."""
    save_key(workspace.key_yaml, key)
    summary = IngestSummary(
        students=list(students),
        duplicates=[],
        not_on_roster=[],
        missing_from_roster=[],
        ignored_files=0,
        unsafe_members=[],
        failed=[],
    )
    write_model(workspace.ingest_summary_json, summary)
    keep_out_of_git(workspace)


def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No API key and no network."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled for this test")

    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)


def session(workspace: Workspace, *answers: str, listing: bool = False) -> Result:
    """`redpen review` on `workspace`, given `answers` one line each and then the
    end of input."""
    args = ["review", "--workspace", str(workspace.root)]
    return CliRunner().invoke(
        app,
        [*args, "--list"] if listing else args,
        input="".join(f"{a}\n" for a in answers),
    )


def files_of(workspace: Workspace) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in workspace.root.rglob("*") if p.is_file()}


def counts(reviewed: int, corrected: int, skipped: int, left: int) -> str:
    return (
        f"Reviewed: {reviewed}, {corrected} of them with corrections. Skipped:"
        f" {skipped}. Still open: {left}."
    )


def four_entries(built: Built, workspace: Workspace) -> IngestRecord:
    """One student, `s1`, with four medium entries, in this order: Q4.b disputed
    (3 points), then Q4.a and Q5.a (2 each, in key order) and Q1 (1), each poorly
    read."""
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    change_grades(workspace, student, "Q4", disputed)
    for part in ("Q4.a", "Q5.a", "Q1"):
        poorly_read(workspace, student, part)
    ready(workspace, [student])
    return student


ASK = "Accept, correct, skip or quit: "
AGAIN = "Type accept, correct, skip or quit."
POINTS_AGAIN = "Type a number from 0 to 1, or press return to keep the award."


def test_a_session_stores_each_action_as_it_is_taken_and_rebuilds_out(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    four_entries(built, workspace)
    # What `redpen override` writes, from a copy of the workspace as it is now.
    shutil.copytree(workspace.root, tmp_path / "copy")
    by_hand = CliRunner().invoke(
        app,
        ["override", "s1", "Q4.a.c2", "0.5", "--note", "See the arrow."]
        + ["--workspace", str(tmp_path / "copy")],
    )
    assert by_hand.exit_code == 0, by_hand.output

    result = session(
        workspace,
        *("maybe", "accept"),  # Q4.b: asked again, then accepted
        *("correct", "", "two", "-1", "1.5", "0.5", "See the arrow."),  # Q4.a
        "skip",  # Q5.a
        "quit",  # Q1
    )

    assert result.exit_code == 0, result.output
    out = result.stdout.splitlines()
    assert [line for line in out if line.startswith("### ")] == [
        "### `A Student` (`s1`), Q4.b",
        "### `A Student` (`s1`), Q4.a",
        "### `A Student` (`s1`), Q5.a",
        "### `A Student` (`s1`), Q1",
    ]
    # Each bad answer asked again in one fixed line, and nothing stored for it.
    assert out.count(AGAIN) == 1
    assert out.count(POINTS_AGAIN) == 3
    assert out[-1] == counts(2, 1, 1, 2)
    assert f"Wrote {workspace.root / 'out'}" in out
    assert result.stderr == ""
    # Exactly the two actions stored, the correction as `redpen override` writes it.
    reviews = yaml_reader().load(workspace.reviews_yaml)
    assert [(r["student"], r["part"]) for r in reviews] == [
        ("s1", "Q4.b"),
        ("s1", "Q4.a"),
    ]
    copied = Workspace(tmp_path / "copy")
    assert workspace.overrides_yaml.read_text() == copied.overrides_yaml.read_text()
    # out/ rebuilt: the two reviewed are closed, Q4.a with the instructor's award.
    assert open_entries(workspace) == [
        "`A Student` (`s1`), Q5.a",
        "`A Student` (`s1`), Q1",
    ]
    assert section(workspace.review_md, "## Closed by review") == [
        CLOSED,
        "- `s1`, Q4.b: `transcript_dispute`. Proposed score: 3 / 3.",
        "- `s1`, Q4.a: `hard_to_read`. Proposed score: 1.5 / 2.",
    ]
    assert detail_of(workspace, "Q4.a.c2") == ("0.5", CORRECTED)
    assert not workspace.run_log.exists()

    again = session(workspace, "quit")

    assert again.exit_code == 0, again.output
    [first, *_] = [
        line for line in again.stdout.splitlines() if line.startswith("### ")
    ]
    assert first == "### `A Student` (`s1`), Q5.a"
    assert again.stdout.splitlines()[-1] == counts(0, 0, 0, 2)


def test_the_terminal_shows_the_entry_its_criteria_and_where_each_file_is(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    four_entries(built, workspace)
    write_report(workspace, KEY, review_students(workspace))
    [(_, entry), *_] = listed(workspace)
    monkeypatch.chdir(tmp_path)  # the workspace given relative, as it may be typed

    result = session(Workspace(Path("ws")), "quit")

    out = result.stdout.splitlines()
    at = out.index("### `A Student` (`s1`), Q4.b")
    end = next(n for n in range(at, len(out)) if out[n].startswith(ASK))
    shown = [line for line in out[at + 1 : end] if line]
    criteria = shown.index("- Criteria:")
    # What review.md shows, up to its files; then the criteria; then the files.
    assert shown[:criteria] == entry[:-1]
    c1, c2 = (
        load(workspace, KEY, review_students(workspace)[0])
        .grades["Q4.b"]
        .grade.criteria
    )
    root = workspace.root.absolute() / "students" / "s1"
    assert shown[criteria:] == [
        "- Criteria:",
        f"  - Q4.b.c1: 2 / 2. {code_span(c1.rationale)}",
        f"  - Q4.b.c2: 1 / 1. {code_span(c2.rationale)}",
        "- Files:",
        f"  - crop 1: {root / 'crops' / 'Q4.b-1.png'}",
        f"  - reading: {root / 'readings' / 'Q4.json'}",
        f"  - grade: {root / 'grades' / 'Q4.json'}",
    ]


def review_students(workspace: Workspace) -> list[IngestRecord]:
    return read_model(workspace.ingest_summary_json, IngestSummary).students


def test_a_withheld_part_corrected_in_full_is_scored_and_reviewed(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    unread_q4b(built, workspace, student)
    ready(workspace, [student])

    # Q4.b.c1 left blank: no award is shown, so it is asked again.
    result = session(workspace, "correct", "", "1.5", "1", "")

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines().count("Type a number from 0 to 2.") == 1
    assert detail_of(workspace, "Q4.b.c1") == ("1.5", CORRECTED)
    assert detail_of(workspace, "Q4.b.c2") == ("1", CORRECTED)
    row = row_of(workspace.scores_by_problem_csv, student)
    assert row["status"] == "final" and row["Q4"] != ""
    assert section(workspace.review_md, "## Closed by review") == [
        CLOSED,
        "- `s1`, Q4.b: `unreadable`. Proposed score: 2.5 / 3.",
    ]
    assert result.stdout.splitlines()[-1] == counts(1, 1, 0, 0)
    for entry in yaml_reader().load(workspace.overrides_yaml):
        assert "note" not in entry


def test_a_key_part_offers_no_correction(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    key = KEY.model_copy(
        update={
            "parts": [
                p.model_copy(update={"state": "unverified"}) if p.id == "Q2" else p
                for p in KEY.parts
            ]
        }
    )
    ready(workspace, [student], key)

    result = session(workspace, "correct", "accept")

    assert result.exit_code == 0, result.output
    out = result.stdout.splitlines()
    assert "### Q2, worth 1" in out
    assert out.count("Type accept, skip or quit.") == 1
    assert not workspace.overrides_yaml.exists()
    assert [
        (r.get("student"), r["part"])
        for r in yaml_reader().load(workspace.reviews_yaml)
    ] == [(None, "Q2")]
    assert row_of(workspace.scores_by_problem_csv, student)["status"] == "final"


def test_the_list_asks_nothing_and_changes_no_file(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    four_entries(built, workspace)
    before = files_of(workspace)

    result = session(workspace, listing=True)

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "`s1`, Q4.b: `transcript_dispute`. Proposed score: 3 / 3.",
        "`s1`, Q4.a: `hard_to_read`. Proposed score: 2 / 2.",
        "`s1`, Q5.a: `hard_to_read`. Proposed score: 2 / 2.",
        "`s1`, Q1: `hard_to_read`. Proposed score: 1 / 1.",
    ]
    assert result.stderr == ""
    assert files_of(workspace) == before


def test_with_no_open_entry_both_forms_say_so_in_one_line(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", "A Student", "s1")
    ready(workspace, [student])
    before = files_of(workspace)

    for listing in (False, True):
        result = session(workspace, listing=listing)

        assert result.exit_code == 0, result.output
        assert result.stdout == f"{NOTHING_TO_REVIEW}\n"
        assert files_of(workspace) == before


def test_an_interrupted_session_keeps_what_was_stored_and_rebuilds_out(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    four_entries(built, workspace)

    # The input ends at Q4.a.c2's prompt: its correction is not taken.
    result = session(workspace, "accept", "correct", "")

    assert result.exit_code == 0, result.output
    reviews = yaml_reader().load(workspace.reviews_yaml)
    assert [(r["student"], r["part"]) for r in reviews] == [("s1", "Q4.b")]
    assert not workspace.overrides_yaml.exists()
    assert open_entries(workspace) == [
        "`A Student` (`s1`), Q4.a",
        "`A Student` (`s1`), Q5.a",
        "`A Student` (`s1`), Q1",
    ]
    assert result.stdout.splitlines()[-1] == counts(1, 0, 0, 3)


def interrupted_once_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C as soon as the first write of overrides.yaml lands."""
    replace = Path.replace

    def then_interrupted(self: Path, target: Path) -> Path:
        done = replace(self, target)
        if Path(target).name == "overrides.yaml":
            raise KeyboardInterrupt
        return done

    monkeypatch.setattr(Path, "replace", then_interrupted)


def interrupted_deriving_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C while the class is derived again, after the corrections."""
    entries = cli.review_entries
    calls: list[object] = []

    def second_interrupted(*args: Any) -> list[Entry]:
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return entries(*args)

    monkeypatch.setattr(cli, "review_entries", second_interrupted)


@pytest.mark.parametrize(
    "interrupt",
    [interrupted_once_written, interrupted_deriving_again],
    ids=["between two corrections", "while the class is derived again"],
)
def test_an_interrupt_outside_a_prompt_ends_the_session_as_quit_does(
    built: Built,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: Callable[[pytest.MonkeyPatch], None],
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    four_entries(built, workspace)
    interrupt(monkeypatch)

    result = session(workspace, "correct", "1", "0.5", "A note.")

    assert result.exit_code == 0, result.output
    # The action's corrections whole, in one write; its review never taken.
    assert [
        (o["criterion"], str(o["points"]), o.get("note"))
        for o in yaml_reader().load(workspace.overrides_yaml)
    ] == [("Q4.b.c1", "1", "A note."), ("Q4.b.c2", "0.5", None)]
    assert not workspace.reviews_yaml.exists()
    # out/ rebuilt with them, and the counts printed.
    assert detail_of(workspace, "Q4.b.c2") == ("0.5", CORRECTED)
    assert open_entries(workspace)[0] == "`A Student` (`s1`), Q4.b"
    assert result.stdout.splitlines()[-1] == counts(0, 0, 0, 4)


def test_one_session_clears_a_list_of_every_kind_of_entry(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    s1, s2, s3, s4, s5, s6 = students = students_of(built, workspace, 6)
    key = KEY.model_copy(  # a key part
        update={
            "parts": [
                p.model_copy(update={"state": "unverified"}) if p.id == "Q2" else p
                for p in KEY.parts
            ]
        }
    )
    # Two readers: the second reads s1's Q4.b otherwise.
    first = load(workspace, KEY, s1).readings["Q4.b"]
    stated = first.reading.final_answer.model_copy(update={"value": 25.0})
    [q4b] = [p for p in KEY.parts if p.id == "Q4.b"]
    write_model(
        workspace.second_json("s1"),
        SecondReadings(
            records=[
                SecondReading(
                    part_id="Q4.b",
                    part_hash=part_hash(q4b, KEY.policies),
                    first=digest(first),
                    doubted=False,
                    result="differ",
                    second=first.model_copy(
                        update={
                            "reading": first.reading.model_copy(
                                update={"final_answer": stated}
                            )
                        }
                    ),
                )
            ]
        ),
    )
    graded_with(workspace, s2, "Q4.b", transcript_dispute="d")  # a disputed reading
    graded_with(workspace, s3, "Q5.b", code_overruled="y")  # an overruled check
    [q4c] = [p for p in KEY.parts if p.id == "Q4.c"]
    write_model(  # an audit change
        workspace.audit_json("Q4.c"),
        PartAudit(
            part_id="Q4.c",
            part_hash=part_hash(q4c, KEY.policies),
            groups=1,
            changed=1,
            undecided=0,
            changes=[
                AuditChange(
                    student_key="s4",
                    criterion="Q4.c.c1",
                    part_hash=part_hash(q4c, KEY.policies),
                    before=Decimal(1),
                    after=Decimal(0),
                    reason="r",
                )
            ],
            key_suspect=None,
        ),
    )
    unread_q4b(built, workspace, s5)  # a part withheld
    write_model(  # graders that disagree
        workspace.second_grade_json("s6"),
        SecondGrades(records=[second_grade_of_q4b(load(workspace, KEY, s6), "differ")]),
    )
    ready(workspace, students, key)
    listing = session(workspace, listing=True).stdout.splitlines()
    assert [line.split(":")[0] for line in listing] == [
        "The key, Q2",
        "`s1`, Q4.b",
        "`s5`, Q4.b",
        "`s6`, Q4.b",
        "`s2`, Q4.b",
        "`s3`, Q5.b",
        "`s4`, Q4.c",
    ]
    assert {code for line in listing for code in re.findall(r"`([a-z_]+)`", line)} == {
        "key_unverified",
        "readers_disagree",
        "unreadable",
        "graders_disagree",
        "transcript_dispute",
        "code_overruled",
        "audit_changed",
    }

    result = session(
        workspace,
        *("accept", "accept", "accept", "accept"),  # the key's Q2, s1, s5, s6
        *("correct", "1", "0.5", "Check line 2."),  # s2's Q4.b: both criteria
        *("correct", "", ""),  # s3's Q5.b: both kept, so accepted
        "accept",  # s4
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines()[-1] == counts(7, 1, 0, 0)
    # The note on the first criterion typed alone.
    assert [
        (o["student"], o["criterion"], str(o["points"]), o.get("note"))
        for o in yaml_reader().load(workspace.overrides_yaml)
    ] == [("s2", "Q4.b.c1", "1", "Check line 2."), ("s2", "Q4.b.c2", "0.5", None)]
    assert session(workspace, listing=True).stdout == f"{NOTHING_TO_REVIEW}\n"
    states = {r["student"]: r["status"] for r in rows(workspace.scores_by_problem_csv)}
    assert states == {
        **{s.student_name: "final" for s in students},
        s5.student_name: "incomplete",
    }
    assert section(workspace.review_md, "## Reviews out of date") == []
    assert len(section(workspace.review_md, "## Closed by review")) == 1 + 7


TERMINAL = "\x1b[2J\x1b[H\rgone\x08‮RTL\x00"  # clears, moves, hides, reorders


def test_text_a_student_wrote_shows_stripped_on_stdout_in_review_alone(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline(monkeypatch)
    workspace = new_workspace(tmp_path)
    student = copy_student(built, workspace, "typed-in-place", MARK + TERMINAL, "s1")
    long = "x" * 100_000

    def hostile(r: ReadingRecord) -> ReadingRecord:
        lines = [f"{MARK} {TERMINAL}", long]
        reading = r.reading.model_copy(update={"lines": lines})
        return r.model_copy(update={"reading": reading})

    def rationale(r: GradeRecord) -> GradeRecord:
        if r.grade.part_id != "Q4.b":
            return r
        c1, c2 = r.grade.criteria
        c1 = c1.model_copy(update={"rationale": f"{MARK}{TERMINAL}"})
        grade = r.grade.model_copy(
            update={"criteria": [c1, c2], "transcript_dispute": f"{MARK}{TERMINAL}"}
        )
        return r.model_copy(update={"grade": grade})

    change_reading(workspace, student, "Q4.b", hostile)
    change_grades(workspace, student, "Q4", rationale)
    ready(workspace, [student])

    listing = session(workspace, listing=True)
    result = session(workspace, "accept")
    report = CliRunner().invoke(app, ["report", "--workspace", str(workspace.root)])

    assert result.exit_code == 0, result.output
    out = result.stdout.splitlines()
    assert f"  - Line 1: `{MARK} ?[2J?[H?gone??RTL?`" in out
    assert f"  - Line 2: `{long}`" in out
    assert f"  - The grader wrote: `{MARK}?[2J?[H?gone??RTL?`" in out
    assert f"  - Q4.b.c1: 2 / 2. `{MARK}?[2J?[H?gone??RTL?`" in out
    assert not set("\x1b\r\x08‮\x00") & set(result.stdout)
    assert result.stderr == ""
    # The prompt took its answer.
    assert out[-1] == counts(1, 0, 0, 0)
    assert section(workspace.review_md, "## Closed by review") != []
    for other in (listing, report):
        assert other.exit_code == 0, other.output
        assert "MARKER" not in other.stdout + other.stderr
    assert not workspace.run_log.exists()
