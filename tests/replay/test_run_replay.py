"""`redpen grade` end to end on three fixture submissions, with recorded calls.

This is the one replay test that chains stages: the reader is given the mapper's
map and the grader the reader's readings, not the truth's, and under the lock's
profile, balanced, each problem the grader graded is graded a second time. So a
change to any map, read or grade prompt orphans its recording,
`tests/recordings/run_chain.jsonl`. That was made, as the stages' recordings were,
by running this seed live into a scratch workspace with a persistent call cache and
keeping exactly the calls the run made (task 2.7 report; the second grade's four
calls were added so in task 4.6b). This test loads the recording itself into an
offline gateway and does not use the `replay` fixture, so `REDPEN_RECORD=1`
neither calls the network here nor rewrites the recording.

The workspace holds the fixtures' key (`spec.json`, and `key.yaml` written from
`key.json`), its blank's page images and a lock pinning `tests/pinned`'s models.
"""

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

from redpen.cli import app
from redpen.config import Lock, write_lock
from redpen.grade.ingest import IngestSummary
from redpen.llm.cache import CallCache
from redpen.llm.client import Gateway
from redpen.llm.models import recheck_providers
from redpen.prepare.key import save_key
from redpen.prepare.parse import render_blank
from redpen.schemas.key import Key
from redpen.schemas.reading import ProblemReadings
from redpen.schemas.submission import MapRecord
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES, SUBMISSIONS, truth_of
from tests.pinned import GRADE_MODELS, ZDR

CLASSES = ("typed-in-place", "all-on-foreign-pages", "blank")
RECORDING = Path(__file__).parents[1] / "recordings" / "run_chain.jsonl"
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())


def seed(root: Path) -> tuple[Workspace, Path]:
    """A prepared workspace `root/ws` and the submissions folder `root/submissions`
    holding the three classes' PDFs."""
    workspace = Workspace(root / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    shutil.copyfile(FIXTURES / "spec.json", workspace.spec_json)
    save_key(workspace.key_yaml, KEY)
    render_blank(workspace)
    write_lock(
        workspace.config_lock_toml, Lock(profile="balanced", models=GRADE_MODELS)
    )
    submissions = root / "submissions"
    submissions.mkdir()
    for pdf in SUBMISSIONS:
        if truth_of(pdf).layout_class in CLASSES:
            shutil.copyfile(pdf, submissions / pdf.name)
    return workspace, submissions


def openrouter(request: httpx.Request) -> httpx.Response:
    """The ZDR listing as recorded; any other request fails the test."""
    if request.url.path.endswith("/endpoints/zdr"):
        return httpx.Response(200, json={"data": ZDR})
    raise AssertionError(f"a replayed test reached the network: {request.url}")


def cells(row: dict[str, str], names: list[str]) -> Decimal:
    return sum((Decimal(row[n]) for n in names), Decimal(0))


def test_three_students_are_graded_end_to_end_and_again_for_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, submissions = seed(tmp_path)
    with closing(sqlite3.connect(workspace.cache_sqlite, autocommit=True)) as db:
        CallCache(db).load(RECORDING)
    transport = httpx.MockTransport(openrouter)
    gateway = functools.partial(Gateway, offline=True, transport=transport)
    monkeypatch.setattr("redpen.orchestrator.Gateway", gateway)
    monkeypatch.setattr(
        "redpen.orchestrator.recheck_providers",
        functools.partial(recheck_providers, transport=transport),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "replay")
    monkeypatch.chdir(tmp_path)  # no redpen.toml: the defaults, as recorded
    args = ["grade", str(submissions), "--workspace", str(workspace.root)]

    first = CliRunner().invoke(app, args)

    log = workspace.run_log.read_text() if workspace.run_log.exists() else ""
    assert first.exit_code == 0, first.output + log
    # Nothing failed, the second grade of each problem a model graded among them.
    assert not log
    assert (
        "Awards graded a second time, by the same grader with the same request, and"
        " compared: 22. Different: 0. Parts flagged (`graders_disagree`): 0."
        " Problems whose second grade failed: 0."
    ) in workspace.run_report_md.read_text().splitlines()
    students = read_model(workspace.ingest_summary_json, IngestSummary).students
    assert len(students) == len(CLASSES)
    out = workspace.root / "out"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.*")) == sorted(
        [
            "review.md",
            "run_report.md",
            "scores_by_page.csv",
            "scores_by_problem.csv",
            "scores_detail.csv",
            *(f"feedback/{s.student_key}.md" for s in students),
        ]
    )
    with workspace.scores_by_problem_csv.open(encoding="utf-8-sig") as file:
        by_problem = {r["student"]: r for r in csv.DictReader(file)}
    with workspace.scores_by_page_csv.open(encoding="utf-8-sig") as file:
        pages = list(csv.DictReader(file))
    for row in pages:
        problems = by_problem[row["student"]]
        page_names = [n for n in row if n.startswith("page ")]
        problem_names = list(problems)[1 : list(problems).index("total")]
        assert cells(row, page_names) == cells(problems, problem_names)
        assert cells(row, page_names) == Decimal(problems["total"])
    classes = {truth_of(p).student_name: truth_of(p).layout_class for p in SUBMISSIONS}
    [blank] = [s for s in students if classes[s.student_name] == "blank"]
    assert by_problem[blank.student_name]["total"] == "0"
    found = read_model(workspace.map_json(blank.student_key), MapRecord).map
    assert found.entries == [] and found.set_aside == []
    readings = [
        r.reading
        for path in (workspace.student_dir(blank.student_key) / "readings").iterdir()
        for r in read_model(path, ProblemReadings).records
    ]
    assert len(readings) == len(KEY.parts)
    assert all(r.status == "blank" for r in readings)

    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    paid = ledger_rows(workspace)
    again = CliRunner().invoke(app, args)

    assert again.exit_code == 0, again.output
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before
    assert ledger_rows(workspace) == paid == 0  # a cache hit writes no ledger row


def ledger_rows(workspace: Workspace) -> int:
    with closing(sqlite3.connect(workspace.cache_sqlite)) as db:
        [(rows,)] = db.execute("SELECT COUNT(*) FROM ledger").fetchall()
    return rows
