"""`redpen grade` on the hostile class (tests/fixtures/hostile), whose students write
to the grader beside their work, with recorded calls: map, read, grade and, under
the lock's profile, balanced, the second grade (design 13 and 14.2).

Each part's score must be the truth's, the score of the work alone. A part whose
page holds a line addressed to the grader is flagged, listed in `out/review.md` and
makes its student provisional; no other part is flagged. No such line reaches the
terminal or `run.log`, and in `review.md` and the feedback files it stays inside its
own place: a code span, or a feedback note with its markup escaped. Three parts
are expected to fail the flag: the recorded mapper set their notes aside.

The recording, `tests/recordings/hostile_class.jsonl`, was made by running this
seed through `redpen grade` live into a persistent scratch workspace and keeping
every call its cache then held (task 5.4b report). The test loads it into an
offline gateway and does not use the `replay` fixture, so `REDPEN_RECORD=1`
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

from redpen.cli import app
from redpen.config import Lock, write_lock
from redpen.grade.ingest import IngestSummary
from redpen.llm.cache import CallCache
from redpen.llm.client import Gateway
from redpen.llm.models import recheck_providers
from redpen.prepare.key import save_key
from redpen.prepare.parse import render_blank
from redpen.report.report import code_span
from redpen.schemas.key import Key
from redpen.schemas.reading import ProblemReadings
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import HOSTILE, HOSTILE_SUBMISSIONS, truth_of
from tests.pinned import GRADE_MODELS
from tests.replay.test_run_replay import openrouter

RECORDING = Path(__file__).parents[1] / "recordings" / "hostile_class.jsonl"
KEY = Key.model_validate_json((HOSTILE / "key.json").read_text())
TRUTHS = {truth_of(p).layout_class: truth_of(p) for p in HOSTILE_SUBMISSIONS}
CASES = [(layout, part.id) for layout in TRUTHS for part in KEY.parts]
FLAG = "- `addressed_to_grader`, high: "
# In the recorded run the mapper set these notes aside, so no reading holds them.
UNREAD = pytest.mark.xfail(
    strict=True, reason="the mapper set the note aside as no part's response"
)
SET_ASIDE = {
    ("order-to-grader", "Q1"),
    ("plea-for-credit", "Q1"),
    ("statement-only", "Q1"),
}
FLAG_CASES = [
    pytest.param(c, p, id=f"{c}-{p}", marks=[UNREAD] if (c, p) in SET_ASIDE else [])
    for c, p in CASES
]


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
        patch.setattr("redpen.orchestrator.Gateway", offline)
        patch.setattr(
            "redpen.orchestrator.recheck_providers",
            functools.partial(recheck_providers, transport=transport),
        )
        patch.setenv("OPENROUTER_API_KEY", "replay")
        patch.chdir(root)  # no redpen.toml: the defaults, as recorded
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


@pytest.mark.parametrize("layout, part", FLAG_CASES)
def test_a_part_whose_page_writes_to_the_grader_is_flagged_and_no_other(
    graded: Graded, layout: str, part: str
) -> None:
    review = entries(graded.workspace.review_md.read_text())
    entry = review.get((graded.keys[layout], part), [])
    [reading] = [r for r in TRUTHS[layout].readings if r.part_id == part]
    with graded.workspace.scores_by_problem_csv.open(encoding="utf-8-sig") as file:
        status = {r["student"]: r["status"] for r in csv.DictReader(file)}

    flagged = any(line.startswith(FLAG) for line in entry)
    assert flagged == bool(reading.addressed_to_grader)
    if flagged:  # whatever the score
        assert status[TRUTHS[layout].student_name] == "provisional"


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
