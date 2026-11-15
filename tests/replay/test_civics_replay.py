"""`autograder2 grade` on the civics class (tests/fixtures/civics), questions on a
printed passage with no calculation, against its hand-written key, with recorded calls:
map, read, grade and, under the lock's profile, balanced, the second grade.

Each part's score must be the truth's. A score the recording gets wrong is an
expected failure, with its reason (task G1 report).

The recording, `tests/recordings/civics_class.jsonl`, was made by running this
seed through `autograder2 grade` live into a persistent scratch workspace and keeping
every call its cache then held. The test loads it into an offline gateway and does
not use the `replay` fixture, so `AUTOGRADER2_RECORD=1` neither calls the network here
nor rewrites the recording."""

import csv
import functools
import shutil
import sqlite3
from contextlib import closing
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from autograder2.cli import app
from autograder2.config import Lock, write_lock
from autograder2.grade.ingest import IngestSummary
from autograder2.llm.cache import CallCache
from autograder2.llm.client import Gateway
from autograder2.llm.models import recheck_providers
from autograder2.prepare.key import save_key
from autograder2.prepare.parse import render_blank
from autograder2.report.report import flags, load
from autograder2.schemas.flags import FlagCode
from autograder2.schemas.key import Key
from autograder2.schemas.submission import MapRecord
from autograder2.workspace import Workspace, read_model
from tests.fixture_truth import CIVICS, CIVICS_SUBMISSIONS, truth_of
from tests.pinned import GRADE_MODELS
from tests.replay.test_run_replay import openrouter

RECORDING = Path(__file__).parents[1] / "recordings" / "civics_class.jsonl"
KEY = Key.model_validate_json((CIVICS / "key.json").read_text())
TRUTHS = {truth_of(p).layout_class: truth_of(p) for p in CIVICS_SUBMISSIONS}
# A score the recording gets wrong: (layout class, part) and why, in one line.
WRONG = {
    ("swapped-answers", "Q4"): "the map gave Q4 the opinion written in its space; "
    "Q4's reasons, under question 6, were graded as Q6",
    ("swapped-answers", "Q6"): "the map gave Q6 the reasons written in its space, "
    "which state no opinion; the opinion, under question 4, was graded as Q4",
}
CASES = [
    pytest.param(
        layout,
        part.id,
        id=f"{layout}-{part.id}",
        marks=[pytest.mark.xfail(strict=True, reason=WRONG[layout, part.id])]
        if (layout, part.id) in WRONG
        else [],
    )
    for layout in TRUTHS
    for part in KEY.parts
]


def seed(root: Path) -> tuple[Workspace, Path]:
    """A prepared workspace `root/ws` for the civics class's hand-written key, with
    a lock pinning `tests/pinned`'s models under balanced, and the submissions
    folder `root/submissions` holding its five PDFs."""
    workspace = Workspace(root / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(CIVICS / "blank.pdf", workspace.blank_pdf)
    shutil.copyfile(CIVICS / "spec.json", workspace.spec_json)
    save_key(workspace.key_yaml, KEY)
    render_blank(workspace)
    write_lock(
        workspace.config_lock_toml, Lock(profile="balanced", models=GRADE_MODELS)
    )
    submissions = root / "submissions"
    submissions.mkdir()
    for pdf in CIVICS_SUBMISSIONS:
        shutil.copyfile(pdf, submissions / pdf.name)
    return workspace, submissions


@pytest.fixture(scope="module")
def graded(tmp_path_factory: pytest.TempPathFactory) -> tuple[Workspace, int]:
    """The class graded once from the recording, through the command line: the
    workspace and the exit code."""
    root = tmp_path_factory.mktemp("civics")
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
    return workspace, result.exit_code


def test_the_class_is_graded_with_nothing_failed(
    graded: tuple[Workspace, int],
) -> None:
    workspace, code = graded
    assert code == 0
    assert not workspace.run_log.exists()


@pytest.mark.parametrize("layout, part", CASES)
def test_each_part_scores_what_the_truth_gives(
    graded: tuple[Workspace, int], layout: str, part: str
) -> None:
    workspace, _ = graded
    with workspace.scores_detail_csv.open(encoding="utf-8-sig") as file:
        rows = [r for r in csv.DictReader(file) if r["part"] == part]
    name = TRUTHS[layout].student_name
    got = {r["criterion"]: Decimal(r["awarded"]) for r in rows if r["student"] == name}
    [want] = [g for g in TRUTHS[layout].grades if g.part_id == part]

    assert got == {c.id: c.awarded for c in want.criteria}


def test_the_answer_placed_on_its_content_is_flagged_and_its_student_provisional(
    graded: tuple[Workspace, int],
) -> None:
    workspace, _ = graded
    students = read_model(workspace.ingest_summary_json, IngestSummary).students
    names = {s.student_key: s.student_name for s in students}
    swapped = TRUTHS["swapped-answers"].student_name
    on_content = {
        (s.student_name, e.part_id, e.confidence)
        for s in students
        for e in read_model(workspace.map_json(s.student_key), MapRecord).map.entries
        if e.basis == "content"
    }
    raised = flags(KEY, [load(workspace, KEY, s) for s in students])
    with workspace.scores_by_problem_csv.open(encoding="utf-8-sig") as file:
        status = {r["student"]: r["status"] for r in csv.DictReader(file)}

    assert on_content == {(swapped, "Q4", "medium")}
    weak = [f for f in raised if f.code == FlagCode.WEAK_ATTRIBUTION]
    assert [(names[f.student_key or ""], f.part_id) for f in weak] == [(swapped, "Q4")]
    assert status[swapped] == "provisional"
