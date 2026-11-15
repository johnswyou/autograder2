"""`autograder2 eval`: the tool's scores against a person's, from stored files only.

The workspace holds the fixture class as the report stage's tests store it: the
truth's readings and grades under `key.json`, on the session's ingest. One part,
Q4.c, is marked unverified, so that a flag on the key itself is there to tell the
two counts of large disagreements not flagged apart. Each reading has one crop,
whose path the listing gives (no image is opened). A gold file is the tool's own
score file, with cells changed by hand."""

import csv
import json
import shutil
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner, Result

from autograder2.cli import app
from autograder2.eval.eval import Evaluation, GoldError, evaluate, markdown, table
from autograder2.grade.ingest import IngestSummary
from autograder2.grade.read import problem_of
from autograder2.prepare.key import page_points, part_hash, save_key
from autograder2.report.report import write_report
from autograder2.schemas.common import Provenance
from autograder2.schemas.grade import GradeRecord, ProblemGrades
from autograder2.schemas.key import Key, PointConstraint
from autograder2.schemas.reading import Crop, ProblemReadings, ReadingRecord
from autograder2.schemas.submission import IngestRecord
from autograder2.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, NOBODY, SUBMISSIONS, truth_of

D = Decimal
TRUTH_KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
KEY = TRUTH_KEY.model_copy(
    update={
        "parts": [
            p.model_copy(update={"state": "unverified"}) if p.id == "Q4.c" else p
            for p in TRUTH_KEY.parts
        ]
    }
)
PARTS = {p.id: p for p in KEY.parts}
PROBLEMS = list(dict.fromkeys(problem_of(p.id) for p in KEY.parts))
MAXIMA = {
    "page": {f"page {n}": points for n, points in page_points(KEY).items()},
    "problem": {
        q: sum(
            (c.points for p in KEY.parts if problem_of(p.id) == q for c in p.criteria),
            D(0),
        )
        for q in PROBLEMS
    },
}
CROP = Crop(page=1, box=[0.0, 0.0, 1.0, 1.0], scale=None)
runner = CliRunner()

type Form = Literal["page", "problem"]


@dataclass(frozen=True)
class Built:
    workspace: Workspace
    students: list[IngestRecord]


def store_class(workspace: Workspace, key: Key, summary: IngestSummary) -> None:
    """The truth's readings and grades of every fixture student, the key, the
    summary, and the reports written from them."""
    workspace.blank_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    save_key(workspace.key_yaml, key)
    write_model(workspace.ingest_summary_json, summary)
    for pdf in SUBMISSIONS:
        truth = truth_of(pdf)
        [student] = [
            s for s in summary.students if s.student_name == truth.student_name
        ]
        who = student.student_key
        for q in PROBLEMS:
            readings = [
                ReadingRecord(provenance=NOBODY, reading=r, crops=[CROP])
                for r in truth.readings
                if problem_of(r.part_id) == q
            ]
            write_model(
                workspace.reading_json(who, q), ProblemReadings(records=readings)
            )
            grades = [
                GradeRecord(
                    provenance=Provenance(
                        key_part_hash=part_hash(PARTS[g.part_id], key.policies),
                        prompt_version=None,
                        model_slug=None,
                    ),
                    grade=g,
                    check=None,
                    reading_digest=None,
                )
                for g in truth.grades
                if problem_of(g.part_id) == q
            ]
            write_model(workspace.grade_json(who, q), ProblemGrades(records=grades))
    write_report(workspace, key, summary.students)


@pytest.fixture(scope="module")
def built(
    fixture_workspace: tuple[Workspace, IngestSummary],
    tmp_path_factory: pytest.TempPathFactory,
) -> Built:
    _, summary = fixture_workspace
    workspace = Workspace(tmp_path_factory.mktemp("eval") / "ws")
    store_class(workspace, KEY, summary)
    return Built(workspace, summary.students)


def copy_of(built: Built, tmp_path: Path) -> Workspace:
    """A copy of the built workspace, for a test that writes in it."""
    shutil.copytree(built.workspace.root, tmp_path / "ws")
    return Workspace(tmp_path / "ws")


def tool_file(workspace: Workspace, by: Form) -> Path:
    if by == "page":
        return workspace.scores_by_page_csv
    return workspace.scores_by_problem_csv


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.reader(file))


def write_rows(path: Path, rows: list[list[str]], bom: bool = True) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as file:
        csv.writer(file).writerows(rows)
    return path


def gold(
    built: Built,
    path: Path,
    by: Form,
    changes: Mapping[tuple[str, str], str] | None = None,
    bom: bool = True,
) -> Path:
    """The tool's own score file with the cells `changes` names, by student and
    column, set to the text given."""
    rows = read_rows(tool_file(built.workspace, by))
    header = rows[0]
    for (name, column), text in (changes or {}).items():
        [row] = [r for r in rows if r[0] == name]
        row[header.index(column)] = text
    return write_rows(path, rows, bom)


def accounted(e: Evaluation, columns: int) -> None:
    """Each cell of the matched students' grid of `columns` is in one count."""
    counts = (e.cells, e.withheld, e.tool_only, e.neither, e.no_points)
    assert sum(counts) == len(e.totals) * columns


def run(gold: Path, workspace: Workspace) -> Result:
    return runner.invoke(
        app, ["eval", "--gold", str(gold), "--workspace", str(workspace.root)]
    )


# --- the tool's own file as gold -----------------------------------------------------


@pytest.mark.parametrize("by", ["page", "problem"])
def test_the_tool_s_own_file_as_gold_agrees_in_every_measure(
    built: Built, tmp_path: Path, by: Form
) -> None:
    e = evaluate(
        built.workspace, KEY, built.students, gold(built, tmp_path / "g.csv", by)
    )

    columns = list(MAXIMA[by])
    assert e.by == by
    assert [c.column for c in e.columns] == columns
    # Ten students scored in every cell; the eleventh is withheld throughout.
    assert e.cells == 10 * len(columns)
    assert all((c.cells, c.mean, c.share) == (10, 0, 0) for c in e.columns)
    assert (e.mean, e.share, e.bias, e.bias_share) == (0, 0, 0, 0)
    assert (e.total_bias, e.total_bias_share) == (0, 0)
    for share in (e.within_step, e.totals_within, e.choice):
        assert share.share == 1
    assert (e.unflagged.count, e.unflagged_named.count) == (0, 0)
    assert all(t.difference == 0 for t in e.totals)
    assert e.largest == []
    assert (e.unknown, e.no_gold, e.doubled) == ([], [], [])
    accounted(e, len(columns))
    # Low-resolution's cells are empty in the tool's file, so in both files.
    assert (e.withheld, e.tool_only, e.neither, e.no_points) == (0, 0, len(columns), 0)
    # The low-resolution student has no compared cell, so no complete total.
    assert e.incomplete_totals == 1
    assert "None: every compared cell agrees." in markdown(e)
    [choice] = [line for line in table(e) if line.startswith("Multiple-choice")]
    cancel = " It compares page sums, so two errors on one page can cancel."
    assert choice.endswith(cancel) == (by == "page")


# --- a gold file changed by hand -----------------------------------------------------


@dataclass(frozen=True)
class Hand:
    changes: dict[tuple[str, str], str]
    columns: dict[str, tuple[Decimal, Decimal]]  # mean, share
    cells: int
    mean: Decimal
    share: Decimal
    within_step: tuple[int, int]
    bias: Decimal
    bias_share: Decimal
    total_bias: Decimal
    total_bias_share: Decimal
    totals_within: tuple[int, int]
    choice: tuple[int, int]
    unflagged: tuple[int, int]
    unflagged_named: tuple[int, int]
    differences: dict[str, Decimal]  # the others' are 0
    largest: list[tuple[str, str]]
    flagged: dict[str, list[str]]  # the flags of the listed cells that have any


# Pages: maxima 3, 5, 4, 4; the step 0.25; 10 students compared on 4 pages.
BY_PAGE = Hand(
    changes={
        ("Avery Example", "page 2"): "4.75",  # tool 5: +0.25, one step
        ("Blake Sample", "page 2"): "2.5",  # tool 4: +1.5 > 1.25, no flag
        # tool 4: +1.25 > 1; code_overruled on Q5.b names the student
        ("Devon Stub", "page 4"): "2.75",
        ("Emery Dummy", "page 3"): "2.5",  # tool 4: +1.5 > 1; the key's flag, Q4.c
        ("Kit Specimen", "page 1"): "3",  # tool 2: -1 > 0.75, choice parts, no flag
        ("Finley Fakename", "page 4"): "1",  # tool 0: -1, exactly 25% of 4
    },
    # |d| sums 1, 1.75, 1.5, 2.25 over 10 cells each
    columns={
        "page 1": (D("0.1"), D(1) / 30),
        "page 2": (D("0.175"), D("0.035")),
        "page 3": (D("0.15"), D("0.0375")),
        "page 4": (D("0.225"), D("0.05625")),
    },
    cells=40,
    mean=D("6.5") / 40,  # sum |d| 6.5
    share=D("6.5") / 160,  # over 10 x 16 points
    within_step=(35, 40),
    bias=D("2.5") / 40,  # sum d 2.5
    bias_share=D("2.5") / 160,
    total_bias=D("0.25"),  # 2.5 over 10 students
    total_bias_share=D("2.5") / 160,
    totals_within=(5, 10),  # |total| <= 0.8: Avery and the four unchanged
    choice=(9, 10),  # page 1: Kit's
    unflagged=(2, 40),  # Blake, Kit
    unflagged_named=(3, 40),  # and Emery, flagged only on the key
    differences={
        "Avery Example": D("0.25"),
        "Blake Sample": D("1.5"),
        "Devon Stub": D("1.25"),
        "Emery Dummy": D("1.5"),
        "Kit Specimen": D(-1),
        "Finley Fakename": D(-1),
    },
    largest=[
        ("Blake Sample", "page 2"),
        ("Emery Dummy", "page 3"),
        ("Devon Stub", "page 4"),
        ("Finley Fakename", "page 4"),
        ("Kit Specimen", "page 1"),
        ("Avery Example", "page 2"),
    ],
    flagged={
        "Devon Stub": ["code_overruled on Q5.b"],
        "Emery Dummy": ["key_unverified on Q4.c"],
    },
)

# Problems: maxima 1, 1, 1, 7, 6; 10 students compared on 5 problems.
BY_PROBLEM = Hand(
    changes={
        ("Avery Example", "Q4"): "6.75",  # tool 7: +0.25, one step
        ("Blake Sample", "Q5"): "4.25",  # tool 6: +1.75 > 1.5, no flag
        # tool 5: +1.75 > 1.5; code_overruled on Q5.b names the student
        ("Devon Stub", "Q5"): "3.25",
        ("Emery Dummy", "Q4"): "5",  # tool 7: +2 > 1.75; the key's flag, Q4.c
        ("Kit Specimen", "Q1"): "1",  # tool 0: -1 > 0.25, a choice part, no flag
        ("Finley Fakename", "Q4"): "6.25",  # tool 4.5: -1.75, exactly 25% of 7
    },
    # |d| sums 1, 0, 0, 4, 3.5 over 10 cells each
    columns={
        "Q1": (D("0.1"), D("0.1")),
        "Q2": (D(0), D(0)),
        "Q3": (D(0), D(0)),
        "Q4": (D("0.4"), D(4) / 70),
        "Q5": (D("0.35"), D("3.5") / 60),
    },
    cells=50,
    mean=D("8.5") / 50,
    share=D("8.5") / 160,
    within_step=(45, 50),
    bias=D(3) / 50,
    bias_share=D(3) / 160,
    total_bias=D("0.3"),
    total_bias_share=D(3) / 160,
    totals_within=(5, 10),
    choice=(29, 30),
    unflagged=(2, 50),
    unflagged_named=(3, 50),
    differences={
        "Avery Example": D("0.25"),
        "Blake Sample": D("1.75"),
        "Devon Stub": D("1.75"),
        "Emery Dummy": D(2),
        "Kit Specimen": D(-1),
        "Finley Fakename": D("-1.75"),
    },
    largest=[
        ("Emery Dummy", "Q4"),
        ("Blake Sample", "Q5"),
        ("Devon Stub", "Q5"),
        ("Finley Fakename", "Q4"),
        ("Kit Specimen", "Q1"),
        ("Avery Example", "Q4"),
    ],
    # Q4 holds Q4.c, whose flag is on the key and so on every student's Q4.
    flagged={
        "Emery Dummy": ["key_unverified on Q4.c"],
        "Devon Stub": ["code_overruled on Q5.b"],
        "Finley Fakename": ["key_unverified on Q4.c"],
        "Avery Example": ["key_unverified on Q4.c"],
    },
)


@pytest.mark.parametrize(("by", "hand"), [("page", BY_PAGE), ("problem", BY_PROBLEM)])
def test_each_measure_equals_the_value_worked_out_by_hand(
    built: Built, tmp_path: Path, by: Form, hand: Hand
) -> None:
    path = gold(built, tmp_path / "g.csv", by, hand.changes)
    e = evaluate(built.workspace, KEY, built.students, path)

    assert {c.column: (c.mean, c.share) for c in e.columns} == hand.columns
    accounted(e, len(hand.columns))
    assert (e.cells, e.mean, e.share) == (hand.cells, hand.mean, hand.share)
    assert (e.bias, e.bias_share) == (hand.bias, hand.bias_share)
    assert (e.total_bias, e.total_bias_share) == (
        hand.total_bias,
        hand.total_bias_share,
    )
    for name in ("within_step", "totals_within", "choice"):
        share = getattr(e, name)
        count, of = getattr(hand, name)
        assert (share.count, share.of, share.share) == (count, of, D(count) / of)
    assert (e.unflagged.count, e.unflagged.of) == hand.unflagged
    assert (e.unflagged_named.count, e.unflagged_named.of) == hand.unflagged_named
    assert e.incomplete_totals == 1
    assert {t.student: t.difference for t in e.totals} == {
        s.student_name: hand.differences.get(s.student_name, D(0))
        for s in built.students
    }
    assert [(d.student, d.column) for d in e.largest] == hand.largest
    assert {d.student: d.flags for d in e.largest if d.flags} == hand.flagged


def test_a_disagreement_lists_both_scores_and_the_paths_of_its_files(
    built: Built, tmp_path: Path
) -> None:
    path = gold(built, tmp_path / "g.csv", "page", BY_PAGE.changes)
    e = evaluate(built.workspace, KEY, built.students, path, top=2)

    [student] = [s for s in built.students if s.student_name == "Emery Dummy"]
    folder = f"students/{student.student_key}"
    assert len(e.largest) == 2
    d = e.largest[1]  # page 3 holds Q4.c and Q5.a
    assert (d.student_key, d.tool, d.person, d.difference) == (
        student.student_key,
        4,
        D("2.5"),
        D("1.5"),
    )
    assert d.readings == [f"{folder}/readings/Q4.json", f"{folder}/readings/Q5.json"]
    assert d.grades == [f"{folder}/grades/Q4.json", f"{folder}/grades/Q5.json"]
    assert d.crops == [f"{folder}/crops/Q4.c-1.png", f"{folder}/crops/Q5.a-1.png"]
    for listed in (*d.readings, *d.grades):
        assert (built.workspace.root / listed).is_file()
    assert (
        f"2. `Emery Dummy`, page 3: tool 4, person 2.5, difference 1.5.\n"
        "   - Flags: key_unverified on Q4.c.\n"
        f"   - Readings: `{folder}/readings/Q4.json`, `{folder}/readings/Q5.json`.\n"
    ) in markdown(e)


def test_a_gold_score_is_read_as_written_never_through_float(
    built: Built, tmp_path: Path
) -> None:
    # 4.9 has no exact binary form: through float it would be 4.9000000000000003...
    path = gold(built, tmp_path / "g.csv", "page", {("Avery Example", "page 2"): "4.9"})
    e = evaluate(built.workspace, KEY, built.students, path)

    [d] = e.largest
    assert (d.person, d.difference) == (D("4.9"), D("0.1"))
    accounted(e, 4)


def test_a_total_exactly_5_percent_of_the_maximum_away_counts_as_within(
    built: Built, tmp_path: Path
) -> None:
    # 5% of 16 is 0.8. Casey's page 2: tool 3, person 2.2. Harper's: 5 and 4.15.
    changes = {("Casey Mockup", "page 2"): "2.2", ("Harper Testcase", "page 2"): "4.15"}
    path = gold(built, tmp_path / "g.csv", "page", changes)
    e = evaluate(built.workspace, KEY, built.students, path)

    differences = {t.student: t.difference for t in e.totals if t.difference}
    assert differences == {"Casey Mockup": D("0.8"), "Harper Testcase": D("0.85")}
    assert (e.totals_within.count, e.totals_within.of) == (9, 10)


def test_a_percent_is_rounded_half_up_where_half_even_would_differ(
    built: Built, tmp_path: Path
) -> None:
    # Page 2: |d| 0.125 over 10 cells of 5 points is 0.25%: 0.3%, not half-even's 0.2%.
    changes = {("Avery Example", "page 2"): "4.875"}
    path = gold(built, tmp_path / "g.csv", "page", changes)
    e = evaluate(built.workspace, KEY, built.students, path)

    assert [c.share for c in e.columns if c.column == "page 2"] == [D("0.0025")]
    assert ["page", "2", "5", "10", "0.01", "0.3%"] in [r.split() for r in table(e)]


# --- cells left out ------------------------------------------------------------------


@pytest.mark.parametrize("by", ["page", "problem"])
def test_a_cell_the_tool_withheld_is_counted_and_left_out_of_the_measures(
    built: Built, tmp_path: Path, by: Form
) -> None:
    # Low-resolution's every cell is withheld by the tool; the person scored them.
    changes = {("Jules Synthetic", c): "0" for c in MAXIMA[by]}
    base = evaluate(
        built.workspace, KEY, built.students, gold(built, tmp_path / "a.csv", by)
    )
    e = evaluate(
        built.workspace,
        KEY,
        built.students,
        gold(built, tmp_path / "b.csv", by, changes),
    )

    n = len(MAXIMA[by])
    assert (base.withheld, base.neither, e.withheld, e.neither) == (0, n, n, 0)
    accounted(base, n)
    accounted(e, n)
    assert e.model_copy(update={"withheld": 0, "neither": n}) == base


@pytest.mark.parametrize("by", ["page", "problem"])
def test_an_empty_gold_cell_is_left_out_of_the_measures(
    built: Built, tmp_path: Path, by: Form
) -> None:
    column = list(MAXIMA[by])[-1]
    changes = {("Kit Specimen", column): "", ("Avery Example", column): "  "}
    e = evaluate(
        built.workspace,
        KEY,
        built.students,
        gold(built, tmp_path / "g.csv", by, changes),
    )

    assert e.cells == 10 * len(MAXIMA[by]) - 2
    assert [c.cells for c in e.columns if c.column == column] == [8]
    assert (e.withheld, e.mean) == (0, 0)
    assert (e.tool_only, e.neither) == (2, len(MAXIMA[by]))
    accounted(e, len(MAXIMA[by]))
    # Kit and Avery now have one cell fewer, and low-resolution none: no complete
    # total, so out of the totals measures. The other eight count.
    assert (e.incomplete_totals, e.totals_within.of) == (3, 8)


def test_a_total_on_fewer_than_every_scorable_cell_is_left_out_of_the_totals(
    built: Built, tmp_path: Path
) -> None:
    # One cell per student, as in the review: no total rests on every column.
    header, *body = read_rows(built.workspace.scores_by_page_csv)
    rows = [header, *(r[:2] + [""] * 3 for r in body)]
    path = write_rows(tmp_path / "g.csv", rows)
    e = evaluate(built.workspace, KEY, built.students, path)

    assert e.cells == 10
    # Pages 2 to 4 of the ten scored students, and low-resolution's four.
    assert (e.tool_only, e.neither) == (30, 4)
    accounted(e, 4)
    assert (e.totals_within.count, e.totals_within.of) == (0, 0)
    assert (e.total_bias, e.total_bias_share, e.totals_within.share) == (None,) * 3
    assert e.incomplete_totals == len(built.students)
    assert (
        "Students left out of the two totals measures, their totals resting on fewer"
        f" than all 4 scorable cells: {len(built.students)}." in table(e)
    )
    assert "Totals within 5% of the assignment's maximum: 0 of 0 students (-)." in (
        table(e)
    )


# --- students ------------------------------------------------------------------------


def test_students_are_matched_with_case_and_spacing_ignored_and_nothing_guessed(
    built: Built, tmp_path: Path
) -> None:
    header, *body = read_rows(built.workspace.scores_by_page_csv)
    cells = {r[0]: r[1:] for r in body}
    rows = [
        header,
        *(
            [n, *c]
            for n, c in cells.items()
            if n not in ("Avery Example", "Blake Sample")
        ),
        ["  avery   EXAMPLE ", *cells["Avery Example"]],  # in place of Avery's row
        ["casey mockup", "0", "0", "0", "0"],  # a second row for one student
        ["Nobody Known", "1", "1", "1", "1"],
        ["Someone Else", "0", "0", "0", "0"],
    ]
    e = evaluate(
        built.workspace, KEY, built.students, write_rows(tmp_path / "g.csv", rows)
    )

    assert e.unknown == ["Nobody Known", "Someone Else"]
    assert e.no_gold == ["Blake Sample"]
    assert e.doubled == ["Casey Mockup"]
    compared = {t.student for t in e.totals}
    assert "Avery Example" in compared
    assert compared.isdisjoint({"Blake Sample", "Casey Mockup"})
    # Eight students scored, on four pages, and nothing differs.
    assert (e.cells, e.mean) == (8 * 4, 0)
    # Nine students matched: the eight, and low-resolution, empty on both sides.
    assert (len(e.totals), e.neither) == (9, 4)
    accounted(e, 4)
    text = markdown(e)
    for title, listed in [
        ("Gold rows naming no student", "- `Nobody Known`\n- `Someone Else`"),
        ("Students with no gold row", "- `Blake Sample`"),
        ("Students with more than one gold row, left out", "- `Casey Mockup`"),
    ]:
        assert f"## {title}\n\n{listed}\n" in text


def test_a_gold_name_in_decomposed_form_matches_the_composed_student_name(
    built: Built, tmp_path: Path
) -> None:
    # Ingest writes every name composed (NFC); a gold typed on macOS may not be.
    students = [
        s.model_copy(update={"student_name": "Zo\u00eb Example"})
        if s.student_name == "Avery Example"
        else s
        for s in built.students
    ]
    rows = read_rows(built.workspace.scores_by_page_csv)
    [avery] = [r for r in rows if r[0] == "Avery Example"]
    avery[0] = "Zoe\u0308 Example"
    e = evaluate(built.workspace, KEY, students, write_rows(tmp_path / "g.csv", rows))

    assert (e.unknown, e.no_gold) == ([], [])
    assert [t.cells for t in e.totals if t.student == "Zo\u00eb Example"] == [4]


@pytest.mark.parametrize(
    ("names", "why"),
    [
        (["Nobody"], "names a student of the ingest summary."),
        (
            ["Avery Example", "avery example", "Nobody"],
            "matches exactly one student: each student it names has more than one row.",
        ),
    ],
    ids=["no name known", "every match doubled"],
)
def test_a_gold_file_with_no_row_for_exactly_one_student_is_refused(
    built: Built, tmp_path: Path, names: list[str], why: str
) -> None:
    header = ["student", *MAXIMA["problem"]]
    path = write_rows(tmp_path / "g.csv", [header, *([n, *"11111"] for n in names)])

    with pytest.raises(GoldError):
        evaluate(built.workspace, KEY, built.students, path)
    result = run(path, copy_of(built, tmp_path))
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [f"No row of {path} {why}"]


def test_a_name_never_reaches_the_markdown_unescaped_or_becomes_a_path(
    built: Built, tmp_path: Path
) -> None:
    workspace = copy_of(built, tmp_path)
    before = sorted(p for p in (workspace.root / "out").rglob("*"))
    rows = read_rows(workspace.scores_by_page_csv)
    hostile = "x\n# Total: 16 / 16 ``` ../../escape"
    rows.append([hostile, "1", "1", "1", "1"])
    result = run(write_rows(tmp_path / "g.csv", rows), workspace)

    assert result.exit_code == 0
    text = workspace.eval_md.read_text()
    assert "- ````x?# Total: 16 / 16 ``` ../../escape````\n" in text
    assert not any(line.startswith("# Total") for line in text.splitlines())
    after = sorted(p for p in (workspace.root / "out").rglob("*"))
    assert after == sorted([*before, workspace.eval_md, workspace.eval_json])
    assert sorted(tmp_path.iterdir()) == [tmp_path / "g.csv", workspace.root]


# --- refusals ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("two", "is not a number of points."),
        ("nan", "is not a number of points."),
        ("inf", "is not a number of points."),
        ("-0.5", "is not a number of points."),
        ("1_000", "is not a number of points."),
        (
            "5.25",
            "is above the key's maximum there, 5. Is the gold on another point scale?",
        ),
    ],
)
def test_a_cell_that_is_not_a_score_is_refused_with_its_row_and_column(
    built: Built, tmp_path: Path, text: str, why: str
) -> None:
    # Avery is on row 2, under the header.
    path = gold(built, tmp_path / "g.csv", "page", {("Avery Example", "page 2"): text})
    workspace = copy_of(built, tmp_path)

    result = run(path, workspace)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stderr.splitlines() == [f"Row 2, column page 2 of {path} {why}"]
    assert not workspace.eval_json.exists()


@pytest.mark.parametrize(
    ("text", "points"),
    [
        ("4", D(4)),
        ("4.5", D("4.5")),
        ("0.25", D("0.25")),
        ("0", D(0)),
        ("5", D(5)),  # page 2's maximum
        ("04", D(4)),
        ("4.000001", D("4.000001")),  # six digits after the point
        (" 4 ", D(4)),
        ("\t4.5\u00a0", D("4.5")),  # a tab and a no-break space around it
    ],
)
def test_a_gold_score_of_plain_digits_is_accepted(
    built: Built, tmp_path: Path, text: str, points: Decimal
) -> None:
    path = gold(built, tmp_path / "g.csv", "page", {("Avery Example", "page 2"): text})
    e = evaluate(built.workspace, KEY, built.students, path)

    # Avery's other pages hold 2, 4 and 4.
    [total] = [t for t in e.totals if t.student == "Avery Example"]
    assert total.person == 10 + points


FORTY = "0." + "1" * 39  # a 40-digit decimal


@pytest.mark.parametrize(
    "text",
    [
        "0_5",
        "1_000",
        "+4",
        "1e0",
        "-0",
        "-0.5",
        "\uff14",  # full-width 4
        "\u0664",  # Arabic-Indic 4
        "\u00bd",  # one half
        FORTY,
        "4.1234567",  # seven digits after the point
        ".5",
        "4.",
        "4,5",
        "4 5",
        "0x10",
        "nan",
        "inf",
        "Infinity",
        "two",
    ],
)
def test_a_gold_score_that_is_not_plain_digits_is_not_a_number_of_points(
    built: Built, tmp_path: Path, text: str
) -> None:
    path = gold(built, tmp_path / "g.csv", "page", {("Avery Example", "page 2"): text})

    with pytest.raises(GoldError) as refused:
        evaluate(built.workspace, KEY, built.students, path)
    assert str(refused.value) == (
        f"Row 2, column page 2 of {path} is not a number of points."
    )


PAGE_COLUMNS = "student, page 1, page 2, page 3, page 4"
PROBLEM_COLUMNS = "student, Q1, Q2, Q3, Q4, Q5"
IGNORED_TOO = "and any of total, maximum, status, open_flags, which are ignored."


@pytest.mark.parametrize(
    ("header", "nearest", "wrong"),
    [
        (
            ["student", "page 1", "page 2", "page 3", "page 5"],
            f"by page: {PAGE_COLUMNS}",
            "Missing: `page 4`. Not expected: `page 5`.",
        ),
        (
            ["Student", "q1", "Q2", "Q3", "Q4", "Q5", "total"],
            f"by problem: {PROBLEM_COLUMNS}",
            "Missing: `student`, `Q1`. Not expected: `Student`, `q1`.",
        ),
        (
            ["student", "page 1", "page 1", "page 2", "page 3", "page 4"],
            f"by page: {PAGE_COLUMNS}",
            "Missing: none. Not expected: none. Repeated: `page 1`.",
        ),
        (
            ["student", "page 1", "page 2", "page 3", "page\x1b[2J4"],
            f"by page: {PAGE_COLUMNS}",
            "Missing: `page 4`. Not expected: `page?[2J4`.",
        ),
    ],
    ids=["a page column", "case", "a repeated column", "a control character"],
)
def test_a_header_that_fits_neither_score_file_is_refused_naming_the_columns(
    built: Built, tmp_path: Path, header: list[str], nearest: str, wrong: str
) -> None:
    path = write_rows(tmp_path / "g.csv", [header])

    result = run(path, copy_of(built, tmp_path))

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        f"{path} has the columns of neither score file. Nearest is {nearest},"
        f" {IGNORED_TOO} {wrong}"
    ]


@pytest.mark.parametrize(
    ("content", "why"),
    [
        (b"student,page 1\n\xff\xfe,1\n", "cannot be read as a UTF-8 CSV file"),
        (b"student,page 1,page 2,page 3,page 4\nA,1\n", "has 2 cells and its header 5"),
    ],
    ids=["not UTF-8", "a short row"],
)
def test_a_gold_file_that_cannot_be_read_gives_one_line(
    built: Built, tmp_path: Path, content: bytes, why: str
) -> None:
    path = tmp_path / "g.csv"
    path.write_bytes(content)

    result = run(path, copy_of(built, tmp_path))

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    [line] = result.stderr.splitlines()
    assert why in line


@pytest.mark.parametrize("by", ["page", "problem"])
def test_the_result_is_the_same_with_or_without_a_byte_order_mark(
    built: Built, tmp_path: Path, by: Form
) -> None:
    changes = BY_PAGE.changes if by == "page" else BY_PROBLEM.changes
    with_bom = gold(built, tmp_path / "a.csv", by, changes, bom=True)
    without = gold(built, tmp_path / "b.csv", by, changes, bom=False)
    assert with_bom.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not without.read_bytes().startswith(b"\xef\xbb\xbf")

    assert evaluate(built.workspace, KEY, built.students, with_bom) == evaluate(
        built.workspace, KEY, built.students, without
    )


# --- the command ---------------------------------------------------------------------


def test_the_terminal_shows_the_table_and_no_name_and_the_json_holds_its_numbers(
    built: Built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = copy_of(built, tmp_path)
    path = gold(built, tmp_path / "g.csv", "page", BY_PAGE.changes)
    # No API key and no network: the command needs neither.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("network disabled for this test")

    for name in ("create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)

    result = run(path, workspace)

    assert result.exit_code == 0, result.output
    terminal = result.output + result.stderr
    for s in built.students:
        assert s.student_name not in terminal
        assert s.student_key not in terminal
    lines = result.stdout.splitlines()
    assert lines[-1] == f"Wrote {workspace.eval_md} and eval.json"
    stored = read_model(workspace.eval_json, Evaluation)
    assert table(stored) == lines[:-1]
    assert stored == evaluate(workspace, KEY, built.students, path)
    accounted(stored, 4)
    assert lines[2:8] == [
        "Their 44 cells:",
        "  compared, scored on both sides: 40",
        "  withheld by the tool, scored by the person: 0",
        "  scored by the tool, empty in the gold: 0",
        "  empty on both sides: 4",
        "  in columns worth no points: 0",
    ]
    split = [line.split() for line in lines]
    assert ["overall", "40", "0.16", "4.1%"] in split
    # Halves round up when printed: 0.175 and 0.225 points, 3.75%.
    assert ["page", "2", "5", "10", "0.18", "3.5%"] in split
    assert ["page", "3", "4", "10", "0.15", "3.8%"] in split
    assert ["page", "4", "4", "10", "0.23", "5.6%"] in split
    assert "  counting only flags that name the student: 3 of 40 cells (7.5%)." in lines
    assert (
        "Signed bias, tool minus person, on cells: 0.06 points, 1.6% of the compared"
        " cells' maximum." in lines
    )
    assert (
        "Multiple-choice agreement: 9 of 10 cells of only choice parts (90%). It"
        " compares page sums, so two errors on one page can cancel." in lines
    )
    assert workspace.eval_md.read_text().startswith("# Evaluation\n")


@pytest.mark.parametrize("stored", ["readings", "grades"])
def test_a_malformed_stored_file_gives_one_line_and_no_stored_text(
    built: Built, tmp_path: Path, stored: str
) -> None:
    workspace = copy_of(built, tmp_path)
    student = built.students[0].student_key
    marker = "MARKER-7f3a written by a student"
    (workspace.student_dir(student) / stored / "Q4.json").write_text(
        f'{{"records": [{{"reading": "{marker}"}}]}}'
    )
    path = gold(built, tmp_path / "g.csv", "page")

    result = run(path, workspace)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stderr.splitlines() == [
        f"The comparison could not be made. The traceback is in {workspace.run_log}."
    ]
    assert "MARKER" not in result.output + result.stderr
    assert "ValidationError" in workspace.run_log.read_text()


@pytest.mark.parametrize("missing", ["key/key.yaml", "ingest_summary.json"])
def test_a_workspace_with_no_key_or_no_ingest_gives_one_line(
    built: Built, tmp_path: Path, missing: str
) -> None:
    workspace = copy_of(built, tmp_path)
    (workspace.root / missing).unlink()

    result = run(gold(built, tmp_path / "g.csv", "page"), workspace)

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        f"{workspace.root} holds no key or no ingest to compare with"
        " (FileNotFoundError)."
    ]


def test_a_column_with_no_points_is_not_compared_nor_a_mixed_one_a_choice_cell(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    # Moved by hand: Q5.c to page 6 of a 4-page blank, so page 5 holds nothing;
    # and the choice part Q3 to page 2, beside the parts Q4.a and Q4.b.
    moved = {"Q5.c": 6, "Q3": 2}
    key = KEY.model_copy(
        update={
            "parts": [
                p.model_copy(update={"home_page": moved[p.id]}) if p.id in moved else p
                for p in KEY.parts
            ]
        }
    )
    workspace = Workspace(tmp_path / "ws")
    store_class(workspace, key, fixture_workspace[1])
    path = tmp_path / "g.csv"
    shutil.copyfile(workspace.scores_by_page_csv, path)

    e = evaluate(workspace, key, fixture_workspace[1].students, path)

    columns = ["page 1", "page 2", "page 3", "page 4", "page 6"]
    assert [c.column for c in e.columns] == columns
    assert e.cells == 10 * 5
    # Page 5 of every student; low-resolution's five other pages.
    assert (e.no_points, e.neither) == (11, 5)
    accounted(e, 6)
    # Only page 1, of Q1 and Q2, holds choice parts alone.
    assert (e.choice.count, e.choice.of) == (10, 10)


# --- a property ----------------------------------------------------------------------


@settings(deadline=None, max_examples=40)
@given(data=st.data(), by=st.sampled_from(["page", "problem"]))
def test_the_measures_hold_together_for_any_scores_within_the_maximums(
    built: Built,
    tmp_path_factory: pytest.TempPathFactory,
    data: st.DataObject,
    by: Form,
) -> None:
    tool = {r[0]: r for r in read_rows(tool_file(built.workspace, by))}
    rows = read_rows(tool_file(built.workspace, by))
    header, step = rows[0], KEY.policies.point_step
    differences: dict[str, list[Decimal]] = {}
    for row in rows[1:]:
        mine = differences.setdefault(row[0], [])
        for column, maximum in MAXIMA[by].items():
            n = header.index(column)
            scored = st.integers(0, int(maximum / step)).map(lambda k: str(k * step))
            same = [st.just(row[n])] if row[n] else []
            row[n] = data.draw(st.one_of([st.just(""), scored, *same]))
            if row[n] and tool[row[0]][n]:
                mine.append(D(tool[row[0]][n]) - D(row[n]))
    path = write_rows(tmp_path_factory.mktemp("gold") / "g.csv", rows)

    e = evaluate(built.workspace, KEY, built.students, path)

    every = [d for mine in differences.values() for d in mine]
    assert e.cells == len(every)
    accounted(e, len(MAXIMA[by]))
    if every:
        assert e.mean is not None and e.bias is not None
        assert (e.mean == 0) == all(d == 0 for d in every)
        assert -e.mean <= e.bias <= e.mean
    else:
        assert (e.mean, e.bias) == (None, None)
    for t in e.totals:
        assert t.difference == sum(differences[t.student], D(0))
        assert t.cells == len(differences[t.student])
    complete = sum(len(d) == len(MAXIMA[by]) for d in differences.values())
    assert (e.totals_within.of, e.incomplete_totals) == (
        complete,
        len(differences) - complete,
    )


# --- the instructor's points (task 4.5a) ---------------------------------------------


def test_with_points_set_the_maxima_are_the_allocation_s_and_a_gold_above_is_refused(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    _, summary = fixture_workspace
    workspace = Workspace(tmp_path / "ws")
    # Allocated by hand in tests/unit/test_report.py: Q1-Q3 2 each, Q4.a 1.5, Q4.b
    # 2.5, Q4.c 1.5, Q5.a 1.5, Q5.b 1, Q5.c 1.
    pages = PointConstraint(pages=[D(6), D(4), D(3), D(2)])
    key = KEY.model_copy(update={"points": pages})
    store_class(workspace, key, summary)
    by_page = write_rows(tmp_path / "p.csv", read_rows(workspace.scores_by_page_csv))
    by_problem = write_rows(
        tmp_path / "q.csv", read_rows(workspace.scores_by_problem_csv)
    )

    for path, maxima in [
        (by_page, [("page 1", 6), ("page 2", 4), ("page 3", 3), ("page 4", 2)]),
        (
            by_problem,
            [("Q1", 2), ("Q2", 2), ("Q3", 2), ("Q4", D("5.5")), ("Q5", D("3.5"))],
        ),
    ]:
        e = evaluate(workspace, key, summary.students, path)
        assert [(c.column, c.maximum) for c in e.columns] == maxima
        assert (e.maximum, e.mean, e.largest) == (15, 0, [])
        # As eval.json writes them: in their shortest form, never 2.00 or 4.0.
        written = json.loads(e.model_dump_json())
        assert [c["maximum"] for c in written["columns"]] == [str(m) for _, m in maxima]
        assert written["maximum"] == "15"
        accounted(e, len(maxima))

    rows = read_rows(workspace.scores_by_page_csv)
    rows[1][rows[0].index("page 4")] = "2.25"  # on row 2, under the header
    path = write_rows(tmp_path / "g.csv", rows)
    result = run(path, workspace)
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        f"Row 2, column page 4 of {path} is above the key's maximum there, 2."
        " Is the gold on another point scale?"
    ]


def test_with_points_set_every_measure_s_denominator_is_the_allocated_maximum(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    # Page maximums 6, 4, 3, 2 (total 15), where the criteria give 3, 5, 4, 4 (16).
    _, summary = fixture_workspace
    workspace = Workspace(tmp_path / "ws")
    key = KEY.model_copy(
        update={"points": PointConstraint(pages=[D(6), D(4), D(3), D(2)])}
    )
    store_class(workspace, key, summary)
    rows = read_rows(workspace.scores_by_page_csv)
    by_name = {r[0]: r for r in rows[1:]}
    # Blake's page 1: tool 6, person 5.2, d +0.8. Under 6 not large (1.5) and not
    # within 5% of 15 (0.75); under the criteria's 3 and 16 it would be both.
    # Avery's page 4: tool 2, person 1.25, d +0.75. Under 2 large (0.5) and within
    # 5% of 15, exactly; under the criteria's 4 it would not be large.
    assert by_name["Blake Sample"][1] == "6" and by_name["Avery Example"][4] == "2"
    by_name["Blake Sample"][1], by_name["Avery Example"][4] = "5.2", "1.25"
    e = evaluate(workspace, key, summary.students, write_rows(tmp_path / "g.csv", rows))

    assert [(c.column, c.maximum, c.mean, c.share) for c in e.columns] == [
        ("page 1", 6, D("0.08"), D("0.8") / 60),
        ("page 2", 4, 0, 0),
        ("page 3", 3, 0, 0),
        ("page 4", 2, D("0.075"), D("0.0375")),
    ]
    accounted(e, len(MAXIMA["page"]))
    assert (e.maximum, e.cells) == (15, 40)
    assert (e.mean, e.share) == (D("1.55") / 40, D("1.55") / 150)
    assert (e.bias, e.bias_share) == (D("1.55") / 40, D("1.55") / 150)
    assert (e.total_bias, e.total_bias_share) == (D("0.155"), D("1.55") / 150)
    assert (e.totals_within.count, e.totals_within.of) == (9, 10)
    assert (e.unflagged.count, e.unflagged_named.count) == (1, 1)
    assert [(d.student, d.column) for d in e.largest] == [
        ("Blake Sample", "page 1"),
        ("Avery Example", "page 4"),
    ]

    # By problem, Q1 is worth 2 here where its criterion is worth 1.
    rows = read_rows(workspace.scores_by_problem_csv)
    q1 = rows[0].index("Q1")
    rows[1][q1] = "2"
    e = evaluate(workspace, key, summary.students, write_rows(tmp_path / "q.csv", rows))
    assert [c.maximum for c in e.columns if c.column == "Q1"] == [2]
    accounted(e, len(MAXIMA["problem"]))
    rows[1][q1] = "2.25"
    path = write_rows(tmp_path / "q.csv", rows)
    with pytest.raises(GoldError) as refused:
        evaluate(workspace, key, summary.students, path)
    assert str(refused.value) == (
        f"Row 2, column Q1 of {path} is above the key's maximum there, 2."
        " Is the gold on another point scale?"
    )
