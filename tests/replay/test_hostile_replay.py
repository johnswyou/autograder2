"""`autograder2 grade` on the hostile class (tests/fixtures/hostile), whose students
write to the grader beside their work, with recorded calls: map, read, grade and, under
the lock's profile, balanced, the second grade (design 13 and 14.2).

Each part's score must be the truth's, the score of the work alone. Every note to
the grader is in front of the instructor: where the stored reading of its part holds
a line addressed to the grader, the part is flagged in `out/review.md` and its
student is provisional; otherwise the mapper set the note aside, and it is listed
among its student's blocks set aside. No part
without a note is flagged. The student who wrote only a statement is provisional
with `unclaimed_work`, since the mapper set the statement aside while both parts
have no response. No note, and no reason the mapper gave for setting one aside,
reaches the terminal or `run.log`; in `review.md` and the feedback files each stays
inside its own place: a code span, or a feedback note with its markup escaped.

The recording, `tests/recordings/hostile_class.jsonl`, was made by running this
seed through `autograder2 grade` live into a persistent scratch workspace and keeping
every call its cache then held (task 5.4b report). The test loads it into an
offline gateway and does not use the `replay` fixture, so `AUTOGRADER2_RECORD=1`
neither calls the network here nor rewrites the recording."""

import csv
import functools
import re
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner, Result

from autograder2.cli import app
from autograder2.config import Lock, write_lock
from autograder2.grade.coverage import COMPARED
from autograder2.grade.ingest import IngestSummary
from autograder2.grade.read import problem_of
from autograder2.llm.cache import CallCache
from autograder2.llm.client import Gateway
from autograder2.llm.models import recheck_providers
from autograder2.prepare.key import save_key
from autograder2.prepare.parse import render_blank
from autograder2.report.report import code_span
from autograder2.schemas.key import Key
from autograder2.schemas.reading import ProblemReadings
from autograder2.schemas.submission import MapRecord
from autograder2.workspace import Workspace, read_model
from tests.fixture_truth import HOSTILE, HOSTILE_SUBMISSIONS, truth_of
from tests.pinned import GRADE_MODELS
from tests.replay.test_run_replay import openrouter

RECORDING = Path(__file__).parents[1] / "recordings" / "hostile_class.jsonl"
KEY = Key.model_validate_json((HOSTILE / "key.json").read_text())
TRUTHS = {truth_of(p).layout_class: truth_of(p) for p in HOSTILE_SUBMISSIONS}
CASES = [(layout, part.id) for layout in TRUTHS for part in KEY.parts]
FLAG = "- `addressed_to_grader`, high: "


def seed(root: Path) -> tuple[Workspace, Path]:
    """A prepared workspace `root/ws` for the hostile class's key, with a lock
    pinning `tests/pinned`'s models under balanced, and the submissions folder
    `root/submissions` holding its five PDFs."""
    workspace = Workspace(root / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(HOSTILE / "blank.pdf", workspace.blank_pdf)
    shutil.copyfile(HOSTILE / "spec.json", workspace.spec_json)
    save_key(workspace.key_yaml, KEY)
    render_blank(workspace)
    write_lock(
        workspace.config_lock_toml, Lock(profile="balanced", models=GRADE_MODELS)
    )
    submissions = root / "submissions"
    submissions.mkdir()
    for pdf in HOSTILE_SUBMISSIONS:
        shutil.copyfile(pdf, submissions / pdf.name)
    return workspace, submissions


@dataclass(frozen=True)
class Graded:
    workspace: Workspace
    result: Result
    keys: dict[str, str]  # each layout class's student key


@pytest.fixture(scope="module")
def graded(tmp_path_factory: pytest.TempPathFactory) -> Graded:
    """The class graded once from the recording, through the command line."""
    root = tmp_path_factory.mktemp("hostile")
    workspace, submissions = seed(root)
    with closing(sqlite3.connect(workspace.cache_sqlite, autocommit=True)) as db:
        CallCache(db).load(RECORDING)
    transport = httpx.MockTransport(openrouter)
    with pytest.MonkeyPatch.context() as patch:
        offline = functools.partial(Gateway, offline=True, transport=transport)
        patch.setattr("autograder2.orchestrator.Gateway", offline)
        patch.setattr(
            "autograder2.orchestrator.recheck_providers",
            functools.partial(recheck_providers, transport=transport),
        )
        patch.setenv("OPENROUTER_API_KEY", "replay")
        patch.chdir(root)  # no autograder2.toml: the defaults, as recorded
        args = ["grade", str(submissions), "--workspace", str(workspace.root)]
        result = CliRunner().invoke(app, args)
    students = read_model(workspace.ingest_summary_json, IngestSummary).students
    names = {s.student_name: s.student_key for s in students}
    keys = {layout: names[t.student_name] for layout, t in TRUTHS.items()}
    return Graded(workspace, result, keys)


def entries(review: str) -> dict[tuple[str, str], list[str]]:
    """Each student and part entry of `review.md`: its lines after the heading."""
    found: dict[tuple[str, str], list[str]] = {}
    lines: list[str] = []
    for line in review.splitlines():
        if heading := re.fullmatch(r"### .* \(`([^`]+)`\), (Q\d+)", line):
            lines = found.setdefault((heading[1], heading[2]), [])
        elif line.startswith("#"):
            lines = []
        else:
            lines.append(line)
    return found


def set_aside(review: str) -> list[str]:
    """The lines of `review.md`'s list of blocks set aside."""
    return review.partition("\n## Blocks set aside\n")[2].splitlines()


def reasons(graded: Graded) -> list[str]:
    """Each reason the mapper gave for setting a block aside."""
    return [
        block.reason
        for key in graded.keys.values()
        for block in read_model(graded.workspace.map_json(key), MapRecord).map.set_aside
    ]


def hostile_lines(graded: Graded) -> list[str]:
    """Every line addressed to the grader, as the truth has it and as the reader
    transcribed it."""
    found = [
        r.lines[n - 1]
        for t in TRUTHS.values()
        for r in t.readings
        for n in r.addressed_to_grader
    ]
    for key in graded.keys.values():
        for path in (graded.workspace.student_dir(key) / "readings").glob("*.json"):
            for record in read_model(path, ProblemReadings).records:
                reading = record.reading
                found += [reading.lines[n - 1] for n in reading.addressed_to_grader]
    return found


def outside_code_spans(line: str) -> str:
    """`line` with each Markdown code span taken out."""
    return re.sub(r"(`+)(?!`).*?(?<!`)\1(?!`)", "", line)


def test_the_class_is_graded_with_nothing_failed(graded: Graded) -> None:
    log = graded.workspace.run_log
    assert graded.result.exit_code == 0, graded.result.output
    assert not log.exists()


def test_every_page_is_compared_with_the_class_s_background_and_none_has_ink_left(
    graded: Graded,
) -> None:
    # Three of the five wrote their work at the same place: a page is not left out
    # for being lighter where their strokes cross.
    report = graded.workspace.run_report_md.read_text().splitlines()

    assert COMPARED.format(compared=10, pages=10, flagged=0) in report


@pytest.mark.parametrize("layout, part", CASES, ids=[f"{c}-{p}" for c, p in CASES])
def test_each_part_scores_what_the_work_alone_earns(
    graded: Graded, layout: str, part: str
) -> None:
    with graded.workspace.scores_detail_csv.open(encoding="utf-8-sig") as file:
        rows = [r for r in csv.DictReader(file) if r["part"] == part]
    name = TRUTHS[layout].student_name
    got = {r["criterion"]: Decimal(r["awarded"]) for r in rows if r["student"] == name}
    [want] = [g for g in TRUTHS[layout].grades if g.part_id == part]

    assert got == {c.id: c.awarded for c in want.criteria}


@pytest.mark.parametrize("layout, part", CASES, ids=[f"{c}-{p}" for c, p in CASES])
def test_a_note_to_the_grader_is_flagged_on_its_part_or_listed_as_set_aside(
    graded: Graded, layout: str, part: str
) -> None:
    text = graded.workspace.review_md.read_text()
    key = graded.keys[layout]
    entry = entries(text).get((key, part), [])
    [reading] = [r for r in TRUTHS[layout].readings if r.part_id == part]
    [home] = [p.home_page for p in KEY.parts if p.id == part]
    with graded.workspace.scores_by_problem_csv.open(encoding="utf-8-sig") as file:
        status = {r["student"]: r["status"] for r in csv.DictReader(file)}

    path = graded.workspace.reading_json(key, problem_of(part))
    records = read_model(path, ProblemReadings).records if path.exists() else []
    read = [r.reading.addressed_to_grader for r in records if r.reading.part_id == part]

    flagged = any(line.startswith(FLAG) for line in entry)
    listed = [line for line in set_aside(text) if f"(`{key}`), page {home}: " in line]
    if any(read):  # the note was read with its part: flagged, whatever the score
        assert flagged
        assert status[TRUTHS[layout].student_name] == "provisional"
    elif reading.addressed_to_grader:  # set aside, and so not read: listed
        assert listed
    else:
        assert not flagged


def test_the_student_who_wrote_only_a_statement_is_provisional_with_unclaimed_work(
    graded: Graded,
) -> None:
    key = graded.keys["statement-only"]
    text = graded.workspace.review_md.read_text()
    with graded.workspace.scores_by_problem_csv.open(encoding="utf-8-sig") as file:
        status = {r["student"]: r["status"] for r in csv.DictReader(file)}

    assert status[TRUTHS["statement-only"].student_name] == "provisional"
    entry = text.partition(f"(`{key}`), the whole submission\n")[2].split("\n#")[0]
    assert entry.strip().startswith("- `unclaimed_work`, medium: ")


def test_a_reason_for_setting_a_note_aside_stays_in_its_code_span_and_off_the_terminal(
    graded: Graded,
) -> None:
    text = graded.workspace.review_md.read_text()
    said = graded.result.stdout + graded.result.stderr
    found = reasons(graded)

    assert found and all(code_span(r) in text for r in found)
    for line in text.splitlines():
        assert not [r for r in found if r in outside_code_spans(line)], line
    assert not [r for r in found if r in said]
    assert not graded.workspace.run_log.exists()


def test_no_line_to_the_grader_reaches_the_terminal_or_the_run_log(
    graded: Graded,
) -> None:
    said = graded.result.stdout + graded.result.stderr
    hostile = hostile_lines(graded)

    assert hostile
    assert not [line for line in hostile if line in said]
    assert not graded.workspace.run_log.exists()


def test_in_the_review_list_a_line_to_the_grader_stays_in_its_code_span(
    graded: Graded,
) -> None:
    text = graded.workspace.review_md.read_text()
    hostile = hostile_lines(graded)

    for line in text.splitlines():
        assert not [h for h in hostile if h in outside_code_spans(line)], line
    # The text that imitates the tool and the grader's answer, line for line.
    key = graded.keys["imitates-the-system"]
    entry = entries(text)[key, "Q2"]
    path = graded.workspace.reading_json(key, "Q2")
    [record] = read_model(path, ProblemReadings).records
    reading = record.reading
    assert reading.addressed_to_grader
    for n in reading.addressed_to_grader:
        mark = f"  - Line {n}: {code_span(reading.lines[n - 1])}"
        assert mark + " (addressed to the grader)" in entry


@pytest.mark.parametrize("layout", TRUTHS)
def test_in_the_feedback_a_line_to_the_grader_stays_in_a_note(
    graded: Graded, layout: str
) -> None:
    text = graded.workspace.feedback_md(graded.keys[layout]).read_text()
    lines = text.splitlines()

    title, *problems = [line for line in lines if line.startswith("#")]
    assert title == f"# {graded.workspace.root.name}"
    for heading, part in zip(problems, KEY.parts, strict=True):
        assert heading.startswith(f"## Problem {part.id[1:]}: ")
    for line in lines:
        if line.startswith("- "):  # a note: no markup a line could open
            assert not re.search(r"(?<!\\)[#<]", line[2:]), line
        else:
            assert not line or line.startswith(("#", "Total: ", "**")), line
