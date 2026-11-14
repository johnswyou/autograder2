"""Hardening, from the recordings (task 5.4a): a provider that fails, a failure no
retry cures, kills and resumes, and damaged stored files, each a `redpen` command
run as the instructor runs it over the three fixture students whose calls
`run_chain.jsonl` chains (see test_run_replay), with a cold call cache and a mock
OpenRouter that answers from the recording (`tests/provider.py`)."""

import asyncio
import csv
import ctypes
import itertools
import json
import os
import random
import re
import resource
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
import pypdfium2.raw as pdfium_c  # pyright: ignore[reportMissingTypeStubs]
import pytest
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Lock, write_lock
from redpen.grade.audit import MAX_CASES, audit_class
from redpen.grade.ingest import IngestSummary
from redpen.llm.cache import CallCache
from redpen.llm.client import MAX_ATTEMPTS, STOPS, Gateway
from redpen.prepare.key import save_key
from redpen.prepare.parse import render_blank
from redpen.report.report import write_report
from redpen.schemas.audit import PartAudit
from redpen.schemas.grade import ProblemGrades
from redpen.schemas.key import Key
from redpen.schemas.reading import ProblemReadings, ReadingRecord
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, NOBODY, SUBMISSIONS, truth_of
from tests.pinned import GRADE_MODELS, GRADER, KEY_MODELS
from tests.provider import Provider, recorded
from tests.replay.test_audit_replay import SEEDED, layouts
from tests.replay.test_audit_replay import seed as audit_seed

CLASSES = ("typed-in-place", "all-on-foreign-pages", "blank")
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
CHAIN = recorded("run_chain")
PREPARE = recorded("prepare_blank")
REPO = Path(__file__).parents[2]
runner = CliRunner()
# Every N of a sweep, and the class of 150, run under REDPEN_SLOW=1 only (`pytest -m
# slow` selects them); a few N run in every suite.
SLOW = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("REDPEN_SLOW") != "1", reason="slow; set REDPEN_SLOW=1"
    ),
]


def sweep(last: int, quick: set[int]) -> list[int | object]:
    """1 to `last`, those not in `quick` marked SLOW."""
    return [
        n if n in quick else pytest.param(n, marks=SLOW) for n in range(1, last + 1)
    ]


def seed(root: Path) -> tuple[Workspace, Path]:
    """A prepared workspace `root/ws` and `root/submissions` holding the three
    chained students' PDFs."""
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


def grade(workspace: Workspace, submissions: Path) -> list[str]:
    return ["grade", str(submissions), "--workspace", str(workspace.root)]


def stored_files(workspace: Workspace) -> dict[str, bytes]:
    """Every file of the workspace but the call cache and the run log, by path."""
    return {
        p.relative_to(workspace.root).as_posix(): p.read_bytes()
        for p in sorted(workspace.root.rglob("*"))
        if p.is_file() and p.name not in ("cache.sqlite", "run.log")
    }


def cached(workspace: Workspace) -> set[str]:
    with closing(sqlite3.connect(workspace.cache_sqlite)) as db:
        return {key for (key,) in db.execute("SELECT key FROM calls")}


@pytest.fixture
def chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Workspace, Path]]:
    """A seeded workspace, an API key made up, and the folder as cwd (no
    redpen.toml: the defaults, as recorded)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
    monkeypatch.chdir(tmp_path)
    yield seed(tmp_path)


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """The stored files of one `grade` run that nothing interrupted."""
    root = tmp_path_factory.mktemp("uninterrupted")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
        monkeypatch.chdir(root)
        workspace, submissions = seed(root)
        provider = Provider(CHAIN)
        provider.install(monkeypatch)
        result = runner.invoke(app, grade(workspace, submissions))
        assert result.exit_code == 0, result.output
        assert sorted(provider.asked) == sorted(CHAIN)  # each answer asked once
        return stored_files(workspace)


def lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip()]


# --- rule 2: a provider that fails every attempt for some items -----------------


def test_items_the_provider_always_fails_end_incomplete_and_are_all_a_rerun_asks(
    chained: tuple[Workspace, Path],
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted: dict[str, bytes],
) -> None:
    workspace, submissions = chained
    failing: set[str] = set()

    def fails(key: str, _: int) -> httpx.Response | None:
        """A 502 for the first two grade requests asked, whoever's they are: first
        grades, since the second grade comes after the grade stage."""
        if len(failing) < 2 and '"grades' in str(CHAIN[key]["choices"]):
            failing.add(key)
        return httpx.Response(502) if key in failing else None

    provider = Provider(CHAIN)
    provider.fault = fails
    provider.install(monkeypatch)

    first = runner.invoke(app, grade(workspace, submissions))

    assert first.exit_code == 1
    assert all(provider.asked.count(k) == MAX_ATTEMPTS for k in failing)
    said = lines(first.output)
    assert sum(" students failed." in line for line in said) == 1  # one line a kind
    assert [line for line in said if line.startswith("grade: ")] == [
        line for line in said if " students failed." in line
    ]
    report = workspace.run_report_md.read_text()
    assert "With parts not scored: 0." not in report
    answered = set(provider.asked) - failing

    healthy = Provider(CHAIN)
    healthy.install(monkeypatch)
    again = runner.invoke(app, grade(workspace, submissions))

    assert again.exit_code == 0, again.output
    assert not set(healthy.asked) & answered  # only what failed, and what needed it
    assert failing <= set(healthy.asked)
    assert len(healthy.asked) == len(set(healthy.asked))
    assert answered | set(healthy.asked) == set(CHAIN)
    assert stored_files(workspace) == uninterrupted


# --- rule 3: a failure no retry cures stops the run at once ---------------------


@pytest.mark.parametrize("code", [402])  # each line: test_gateway
def test_a_failure_no_retry_cures_stops_the_run_in_one_line_and_the_rerun_resumes(
    chained: tuple[Workspace, Path],
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted: dict[str, bytes],
    code: int,
) -> None:
    workspace, submissions = chained
    provider = Provider(CHAIN)
    at = 6  # the sixth chat request, among the map's
    provider.fault = lambda _, n: (
        httpx.Response(code, json={"error": {"code": code}}) if n >= at else None
    )
    provider.install(monkeypatch)

    stopped = runner.invoke(app, grade(workspace, submissions))

    assert stopped.exit_code == 1
    assert lines(stopped.output)[-1] == STOPS[code]
    assert sum(STOPS[code] in line for line in lines(stopped.output)) == 1
    assert len(provider.asked) == at  # nothing sent after it
    kept = cached(workspace)
    assert len(kept) == at - 1
    # At once: no later stage ran (the read stage writes its crops before it asks),
    # and no student's failure was logged for it.
    assert list(workspace.students_dir.glob("*/map.json"))
    assert not list(workspace.students_dir.glob("*/crops"))
    assert not list(workspace.students_dir.glob("*/readings"))
    assert not workspace.run_log.exists()

    healthy = Provider(CHAIN)
    healthy.install(monkeypatch)
    again = runner.invoke(app, grade(workspace, submissions))

    assert again.exit_code == 0, again.output
    assert not set(healthy.asked) & kept
    assert len(healthy.asked) == len(CHAIN) - len(kept)
    assert stored_files(workspace) == uninterrupted


# --- rules 4, 5 and 6: killed at any point, and stored files damaged --------------


class Killed(BaseException):
    """The process dies at a request: no handler of the code under test runs."""


def killing(at: int) -> Callable[[str, int], httpx.Response | None]:
    """A fault that kills the run at the `at`th chat request and at every later one
    (a dead process sends nothing)."""

    def fault(_: str, n: int) -> httpx.Response | None:
        if n >= at:
            raise Killed
        return None

    return fault


def assert_whole_json(workspace: Workspace) -> None:
    """No JSON file of the workspace is half written; a `.partial` may be left."""
    for path in workspace.root.rglob("*.json"):
        json.loads(path.read_bytes())


def run(workspace: Workspace, submissions: Path) -> list[str]:
    blank = str(FIXTURES / "blank.pdf")
    return ["run", blank, str(submissions), "--workspace", str(workspace.root)]


# The first and the last request, and the first of the read stage.
@pytest.mark.parametrize("at", sweep(len(CHAIN), {1, 14, 27}))
def test_run_killed_at_the_nth_request_resumes_asking_nothing_twice_to_the_same_files(
    chained: tuple[Workspace, Path],
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted: dict[str, bytes],
    at: int,
) -> None:
    workspace, submissions = chained
    provider = Provider(CHAIN)
    provider.fault = killing(at)
    provider.install(monkeypatch)

    with pytest.raises(Killed):
        runner.invoke(app, run(workspace, submissions))

    assert_whole_json(workspace)
    kept = cached(workspace)
    assert len(kept) == at - 1
    healthy = Provider(CHAIN)
    healthy.install(monkeypatch)
    resumed = runner.invoke(app, run(workspace, submissions))
    assert resumed.exit_code == 0, resumed.output
    assert not set(healthy.asked) & kept
    assert sorted([*kept, *healthy.asked]) == sorted(CHAIN)
    assert stored_files(workspace) == uninterrupted


def regrade(workspace: Workspace) -> list[str]:
    return ["regrade", "--workspace", str(workspace.root)]


def graded_cold(root: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A workspace graded, its call cache then removed: regrade asks for every grade
    and second grade again (8 requests)."""
    workspace, submissions = seed(root)
    Provider(CHAIN).install(monkeypatch)
    assert runner.invoke(app, grade(workspace, submissions)).exit_code == 0
    workspace.cache_sqlite.unlink()
    return workspace


@pytest.fixture(scope="module")
def regraded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """The stored files of such a regrade that nothing interrupted."""
    root = tmp_path_factory.mktemp("regraded")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
        monkeypatch.chdir(root)
        workspace = graded_cold(root, monkeypatch)
        provider = Provider(CHAIN)
        provider.install(monkeypatch)
        assert runner.invoke(app, regrade(workspace)).exit_code == 0
        assert len(provider.asked) == len(set(provider.asked)) == 8
        return stored_files(workspace)


@pytest.mark.parametrize("at", [1, 8])
def test_regrade_killed_at_the_nth_request_resumes_asking_nothing_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    regraded: dict[str, bytes],
    at: int,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
    monkeypatch.chdir(tmp_path)
    workspace = graded_cold(tmp_path, monkeypatch)
    provider = Provider(CHAIN)
    provider.fault = killing(at)
    provider.install(monkeypatch)

    with pytest.raises(Killed):
        runner.invoke(app, regrade(workspace))

    assert_whole_json(workspace)
    kept = cached(workspace)
    healthy = Provider(CHAIN)
    healthy.install(monkeypatch)
    resumed = runner.invoke(app, regrade(workspace))
    assert resumed.exit_code == 0, resumed.output
    assert not set(healthy.asked) & kept
    assert len(kept) + len(healthy.asked) == 8
    assert stored_files(workspace) == regraded


def prepared(root: Path) -> Workspace:
    """A workspace holding the fixture blank's spec and a lock, as the prepare
    replay seeds it."""
    workspace = Workspace(root / "ws")
    workspace.spec_json.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "spec.json", workspace.spec_json)
    write_lock(workspace.config_lock_toml, Lock(profile="balanced", models=KEY_MODELS))
    return workspace


def prepare(workspace: Workspace) -> list[str]:
    blank = str(FIXTURES / "blank.pdf")
    return ["prepare", blank, "--workspace", str(workspace.root)]


@pytest.fixture(scope="module")
def prepared_once(tmp_path_factory: pytest.TempPathFactory) -> dict[str, bytes]:
    """The files of one `prepare` that nothing interrupted."""
    root = tmp_path_factory.mktemp("prepared")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
        monkeypatch.chdir(root)
        workspace = prepared(root)
        provider = Provider(PREPARE)
        provider.install(monkeypatch)
        result = runner.invoke(app, prepare(workspace))
        assert result.exit_code == 0, result.output
        assert sorted(provider.asked) == sorted(PREPARE)
        return stored_files(workspace)


@pytest.mark.parametrize("at", sweep(len(PREPARE), {1, 16}))
def test_prepare_killed_at_the_nth_request_resumes_asking_nothing_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_once: dict[str, bytes],
    at: int,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
    monkeypatch.chdir(tmp_path)
    workspace = prepared(tmp_path)
    provider = Provider(PREPARE)
    provider.fault = killing(at)
    provider.install(monkeypatch)

    with pytest.raises(Killed):
        runner.invoke(app, prepare(workspace))

    assert_whole_json(workspace)
    kept = cached(workspace)
    healthy = Provider(PREPARE)
    healthy.install(monkeypatch)
    resumed = runner.invoke(app, prepare(workspace))
    assert resumed.exit_code == 0, resumed.output
    assert not set(healthy.asked) & kept
    assert sorted([*kept, *healthy.asked]) == sorted(PREPARE)
    assert stored_files(workspace) == prepared_once


# Run in a child process, which blocks at its `at`th chat request until killed.
CHILD = """
import sys, time
import pytest
from typer.testing import CliRunner
from redpen.cli import app
from tests.provider import Provider, recorded

marker, at, *args = sys.argv[1:]


def fault(key, n):
    if n == int(at):
        open(marker, "w").close()
        time.sleep(600)


provider = Provider(recorded("run_chain"))
provider.fault = fault
with pytest.MonkeyPatch.context() as monkeypatch:
    provider.install(monkeypatch)
    CliRunner().invoke(app, args)
"""


def test_grade_killed_for_real_mid_stage_resumes_asking_nothing_twice(
    chained: tuple[Workspace, Path],
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted: dict[str, bytes],
) -> None:
    workspace, submissions = chained
    marker = workspace.root.parent / "blocked"
    env = {**os.environ, "PYTHONPATH": str(REPO), "OPENROUTER_API_KEY": "made-up"}
    command = [sys.executable, "-c", CHILD, str(marker), "17"]  # in the read stage
    child = subprocess.Popen(
        [*command, *grade(workspace, submissions)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 600
    while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert marker.exists(), "the child never reached its 17th request"
    child.send_signal(signal.SIGKILL)
    assert child.wait() == -signal.SIGKILL

    assert_whole_json(workspace)
    kept = cached(workspace)
    assert len(kept) == 16
    healthy = Provider(CHAIN)
    healthy.install(monkeypatch)
    resumed = runner.invoke(app, grade(workspace, submissions))
    assert resumed.exit_code == 0, resumed.output
    assert not set(healthy.asked) & kept
    assert len(healthy.asked) == len(CHAIN) - len(kept)
    assert stored_files(workspace) == uninterrupted


def test_damaged_files_are_each_named_in_one_line_and_grade_makes_them_again(
    chained: tuple[Workspace, Path],
    monkeypatch: pytest.MonkeyPatch,
    uninterrupted: dict[str, bytes],
) -> None:
    workspace, submissions = chained
    Provider(CHAIN).install(monkeypatch)
    assert runner.invoke(app, grade(workspace, submissions)).exit_code == 0
    [first, second, *_] = sorted(workspace.students_dir.iterdir())
    damaged = [
        first / "grades" / "Q4.json",
        first / "readings" / "Q5.json",
        second / "map.json",
    ]
    for path in damaged[:2]:
        path.write_bytes(path.read_bytes()[:40])  # cut short, as by a full disk
    damaged[2].write_bytes(b"")  # emptied
    # A write a kill cut short before its rename, and one left beside a whole file.
    (second / "grades" / "Q4.json.partial").write_text('{"records": [')
    (workspace.root / "ingest_summary.json.partial").write_text("")
    report = ["report", "--workspace", str(workspace.root)]
    review = ["review", "--list", "--workspace", str(workspace.root)]
    # Regrade comes last: its grade stage makes the grade file again.
    commands = {"report": report, "review": review, "regrade": regrade(workspace)}
    for name, command in commands.items():
        if name == "regrade":
            damaged = damaged[1:]
        result = runner.invoke(app, command)
        said = lines(result.output)
        assert result.exit_code == 1, name
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert "Traceback" not in result.output, name
        named = [
            f"{path} is damaged (not JSON). Delete it and run `redpen grade` again"
            " to have it made again."
            for path in sorted(damaged)
        ]
        # Before them, only regrade's progress: a line a stage.
        assert said[-len(named) :] == named, name
        assert all(re.search(r" \d+/\d+ \$", line) for line in said[: -len(named)])

    healthy = Provider(CHAIN)
    healthy.install(monkeypatch)
    again = runner.invoke(app, grade(workspace, submissions))

    assert again.exit_code == 0, again.output
    assert healthy.asked == []  # every answer is in the cache
    # Each `.partial` is the name the file's next write takes, and is renamed.
    assert stored_files(workspace) == uninterrupted
    assert runner.invoke(app, report).exit_code == 0


# Run in a child process: answers written to a call cache, one after another, until
# the child is killed.
WRITER = """
import json, sqlite3, sys
from redpen.llm.cache import CallCache

cache = CallCache(sqlite3.connect(sys.argv[1], autocommit=True))
n = 0
while True:  # five answers written over and over, as `forget` and a new answer do
    n += 1
    cache.put(f"key {n % 5}", json.dumps({"n": n, "text": str(n) * 100_000}))
"""


def test_the_call_cache_survives_a_kill_during_a_write(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    # A journal that rolls a killed write back on the next open (sqlite's atomic
    # commit); `off` and `memory` leave a torn page behind.
    with closing(sqlite3.connect(db, autocommit=True)) as connection:
        CallCache(connection)
        [(mode,)] = connection.execute("PRAGMA journal_mode")
    assert mode in ("delete", "truncate", "persist", "wal")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for _ in range(20):  # each kill lands somewhere in a stream of writes
        child = subprocess.Popen([sys.executable, "-c", WRITER, str(db)], env=env)
        rows = 0
        deadline = time.monotonic() + 120
        while rows < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
            with closing(sqlite3.connect(db)) as connection:
                try:
                    [(rows,)] = connection.execute("SELECT COUNT(*) FROM calls")
                except sqlite3.OperationalError:  # no table yet, or locked
                    continue
        time.sleep(random.uniform(0.1, 0.4))  # not while it waits on this read's lock
        child.send_signal(signal.SIGKILL)
        assert child.wait() == -signal.SIGKILL

        with closing(sqlite3.connect(db)) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            answers = connection.execute("SELECT key, response FROM calls").fetchall()
        assert answers
        for key, response in answers:  # each answer is there whole
            found = json.loads(response)
            assert found["n"] % 5 == int(key.split()[1])
            assert found["text"] == str(found["n"]) * 100_000


# --- rule 8: a class audit request has an upper size -----------------------------


def decide_as_awarded(requests: list[list[dict[str, Any]]]) -> httpx.MockTransport:
    """A grader that decides each case of an audit request as it was awarded, and
    keeps each request's cases in `requests`."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        cases = json.loads(body["messages"][1]["content"][2]["text"])["cases"]
        requests.append(cases)
        found = [
            {"case": c["case"], "reason": "r", "awarded": c["awarded"]} for c in cases
        ]
        content = json.dumps({"decisions": found})
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )

    return httpx.MockTransport(handler)


def test_an_audit_group_too_large_for_one_request_is_split_recorded_and_reported(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    _, eleven = fixture_workspace
    # 150 students: the eleven fixture students' truth, each stored under new keys.
    copies = [
        s.model_copy(update={"student_key": f"Copy_{n:03d}-{n:06x}"})
        for n, s in zip(range(150), itertools.cycle(eleven.students), strict=False)
    ]
    summary = eleven.model_copy(update={"students": copies})
    workspace = Workspace(tmp_path / "ws")
    audit_seed(workspace, summary)
    for copy in copies:  # the seeded disagreement, in every copy of its student
        if layouts(summary)[copy.student_key] == SEEDED[0]:
            path = workspace.grade_json(copy.student_key, "Q4")
            stored = read_model(path, ProblemGrades)
            for record in stored.records:
                for c in record.grade.criteria:
                    if c.id == SEEDED[1]:
                        c.awarded = Decimal(0)
            write_model(path, stored)
    requests: list[list[dict[str, Any]]] = []

    async def main() -> list[PartAudit]:
        db = tmp_path / "cache.sqlite"
        transport = decide_as_awarded(requests)
        async with Gateway(db, "made-up", transport=transport) as gateway:
            return await audit_class(workspace, KEY, copies, gateway, GRADER)

    audits = {a.part_id: a for a in asyncio.run(main())}

    largest = max(map(len, requests))
    assert largest <= MAX_CASES
    split = {p: a.split for p, a in audits.items() if a.split}
    assert split  # the seeded group of Q4.c, past MAX_CASES at 150
    sizes = sorted(len(c) for c in requests)
    assert len(requests) > sum(a.groups for a in audits.values())
    assert sizes[0] * 2 >= sizes[-1]  # a group's parts are about the same size
    workspace.blank_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    write_report(workspace, KEY, copies, [])
    assert (
        f"Groups too large for one request, decided in parts that may differ:"
        f" {sum(split.values())}."
    ) in workspace.run_report_md.read_text()


# --- rules 7 and 9: a large class --------------------------------------------------


def renamed(pdf: Path, name: str, to: Path) -> None:
    """A copy of the Classkick export `pdf` at `to` whose sidebar names `name`: its
    pages, and so every request made from them, are the same as the original's."""
    document = pdfium.PdfDocument(pdf)
    for page in document:
        for found in page.get_objects(max_depth=0, textpage=page.get_textpage()):
            if isinstance(found, pdfium.PdfTextObj) and found.extract().startswith(
                "Student:"
            ):
                given = ctypes.create_string_buffer(
                    f"Student: {name}\0".encode("utf-16-le")
                )
                assert pdfium_c.FPDFText_SetText(
                    found.raw, ctypes.cast(given, pdfium_c.FPDF_WIDESTRING)
                )
        page.gen_content()
    document.save(to)
    document.close()


def a_class(root: Path, size: int) -> tuple[Workspace, Path]:
    """A seeded workspace and `size` submissions: the three chained students in
    turn, each under a name made up for it."""
    workspace, chained = seed(root)
    folder = root / f"class of {size}"
    folder.mkdir()
    originals = sorted(chained.iterdir())
    for n in range(size):
        name = f"Made Up{n:03d}"
        renamed(originals[n % 3], name, folder / f"Practice Set 3 - {name}.pdf")
    return workspace, folder


class Peak:
    """The largest resident memory and number of open files of this process while
    the `with` block runs, sampled every 20 ms, above what they were at its start."""

    def __enter__(self) -> "Peak":
        self.rss = self.files = 0
        self._start = (self._rss(), self._files())
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._done.set()
        self._thread.join()

    def _sample(self) -> None:
        while not self._done.wait(0.02):
            self.rss = max(self.rss, self._rss() - self._start[0])
            self.files = max(self.files, self._files() - self._start[1])

    @staticmethod
    def _rss() -> int:
        found = re.search(r"VmRSS:\s+(\d+) kB", Path("/proc/self/status").read_text())
        assert found
        return int(found[1]) * 1024

    @staticmethod
    def _files() -> int:
        return len(os.listdir("/proc/self/fd"))


@dataclass
class Graded:
    seconds: float
    requests: int
    rss: int
    files: int
    said: list[str]
    report: list[str]


FILE_LIMIT = 256  # macOS's default soft limit on open files


def graded_class(root: Path, monkeypatch: pytest.MonkeyPatch, size: int) -> Graded:
    """`redpen grade` of a class of `size` from the recordings, with the network
    disabled and at most FILE_LIMIT files open; what it took, said and wrote."""
    workspace, folder = a_class(root, size)
    provider = Provider(CHAIN)
    provider.install(monkeypatch)
    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (FILE_LIMIT, limits[1]))
    try:
        started = time.monotonic()
        with Peak() as peak:
            result = runner.invoke(app, grade(workspace, folder))
        seconds = time.monotonic() - started
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
    assert result.exit_code == 0, result.output
    assert provider.unrecorded == []
    # Identical pages are answered once: each request is asked once, by whoever
    # comes first, and the class asks nothing the three students did not.
    assert sorted(provider.asked) == sorted(CHAIN)
    for csv_file in (workspace.scores_by_problem_csv, workspace.scores_by_page_csv):
        with csv_file.open(encoding="utf-8-sig") as rows:
            assert len(list(csv.DictReader(rows))) == size
    with workspace.scores_detail_csv.open(encoding="utf-8-sig") as rows:
        assert len({r["student"] for r in csv.DictReader(rows)}) == size
    assert len(list(workspace.feedback_dir.glob("*.md"))) == size
    assert workspace.review_md.exists()
    report = workspace.run_report_md.read_text().splitlines()
    assert f"Students: {size}. With every part scored: {size}. With parts not" in (
        "\n".join(report)
    )
    found = Graded(
        seconds, len(provider.asked), peak.rss, peak.files, lines(result.output), report
    )
    shutil.rmtree(workspace.root.parent / f"class of {size}")
    shutil.rmtree(workspace.students_dir)  # the disk holds little to spare
    return found


def assert_readable(small: Graded, large: Graded) -> None:
    """Rule 9: nothing said or reported grows with the class but its counts."""
    assert len(large.said) == len(small.said)
    assert len(large.report) == len(small.report)


def test_a_class_of_11_is_graded_with_each_request_asked_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
    monkeypatch.chdir(tmp_path)
    three = graded_class(tmp_path / "3", monkeypatch, 3)
    eleven = graded_class(tmp_path / "11", monkeypatch, 11)
    assert_readable(three, eleven)


@SLOW[0]
@SLOW[1]
def test_a_class_of_150_is_graded_from_the_recordings_as_one_of_11_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
    monkeypatch.chdir(tmp_path)
    eleven = graded_class(tmp_path / "11", monkeypatch, 11)
    large = graded_class(tmp_path / "150", monkeypatch, 150)
    print(f"\n11: {eleven}\n150: {large}".replace(", said=", "\n  said="))
    assert_readable(eleven, large)
    assert large.requests == eleven.requests == len(CHAIN)
    assert large.files <= eleven.files + 8  # no file is kept open per student
    # No page image of every student is held at once: 150 students' 600 pages are
    # some 3.6 GB decoded.
    assert large.rss <= eleven.rss + 512 * 2**20


# --- rule 11: the estimate -------------------------------------------------------


def test_every_stage_that_sends_requests_has_a_row_in_the_estimate(
    chained: tuple[Workspace, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, submissions = chained
    dry = runner.invoke(app, [*grade(workspace, submissions), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    table, assumed = dry.stdout.split("Assumptions:")
    rows = {line[:15].strip(): line[15:].split() for line in table.splitlines()[2:]}
    estimated = {stage: float(row[0]) for stage, row in rows.items()}
    provider = Provider(CHAIN)
    provider.install(monkeypatch)

    assert runner.invoke(app, grade(workspace, submissions)).exit_code == 0

    stages = ["map", "read", "second", "grade", "second grade", "audit"]
    assert [s for s in rows if s not in ("prepare", "total")] == stages
    assert "the second reader is assumed to read again 25% of what" in " ".join(
        assumed.split()
    )
    assert "class audit to form 0.2 groups for each part a model grades" in " ".join(
        assumed.split()
    )
    # The calls made, by stage: the ledger books each by role; the class audit's are
    # the grader's Decisions.
    with closing(sqlite3.connect(workspace.cache_sqlite)) as db:
        roles = dict(db.execute("SELECT role, COUNT(*) FROM ledger GROUP BY role"))
    audit = sum('"decisions' in str(CHAIN[k]["choices"]) for k in provider.asked)
    made = {
        "map": roles.get("mapper", 0),
        "read": roles.get("reader", 0),
        "second": roles.get("second", 0),
        "grade": roles.get("grader", 0) - audit,
        "second grade": roles.get("second grade", 0),
        "audit": audit,
    }
    print("\nstage, estimated calls, calls made")
    for stage in stages:
        print(f"{stage}, {rows[stage][0]}, {made[stage]}")
    assert sum(made.values()) == len(CHAIN)
    assert all(estimated[s] > 0 for s in stages)  # each stage is priced


def test_a_stored_file_whose_write_is_cut_short_stays_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "Q4.json"
    readings = truth_of(SUBMISSIONS[0]).readings
    old, new = (
        ProblemReadings(
            records=[ReadingRecord(provenance=NOBODY, reading=r, crops=[]) for r in rs]
        )
        for rs in (readings[:1], readings)
    )
    write_model(path, old)
    write_text = Path.write_text

    def killed_halfway(self: Path, text: str, *args: Any, **kwargs: Any) -> int:
        write_text(self, text[: len(text) // 2], *args, **kwargs)
        raise Killed

    monkeypatch.setattr(Path, "write_text", killed_halfway)
    with pytest.raises(Killed):
        write_model(path, new)
    monkeypatch.undo()

    assert read_model(path, ProblemReadings) == old
    write_model(path, new)  # the next write takes the `.partial` left behind
    assert read_model(path, ProblemReadings) == new
