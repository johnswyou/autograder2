"""Hardening, from the recordings (task 5.4a): a provider that fails, a failure no
retry cures, kills and resumes, and damaged stored files, each a `redpen` command
run as the instructor runs it over the three fixture students whose calls
`run_chain.jsonl` chains (see test_run_replay), with a cold call cache and a mock
OpenRouter that answers from the recording (`tests/provider.py`)."""

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Lock, write_lock
from redpen.llm.client import MAX_ATTEMPTS, STOPS
from redpen.prepare.key import save_key
from redpen.prepare.parse import render_blank
from redpen.schemas.key import Key
from redpen.workspace import Workspace
from tests.fixture_truth import FIXTURES, SUBMISSIONS, truth_of
from tests.pinned import GRADE_MODELS, KEY_MODELS
from tests.provider import Provider, recorded

CLASSES = ("typed-in-place", "all-on-foreign-pages", "blank")
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
CHAIN = recorded("run_chain")
PREPARE = recorded("prepare_blank")
REPO = Path(__file__).parents[2]
runner = CliRunner()
# Every N of a sweep runs under REDPEN_SLOW=1 only; a few run in every suite.
SLOW = pytest.mark.skipif(
    os.environ.get("REDPEN_SLOW") != "1", reason="slow; set REDPEN_SLOW=1"
)


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
    # At once: no later stage ran, and no student's failure was logged for it.
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


# The first and last requests, and the first of each later stage: map, read, the
# second grade.
@pytest.mark.parametrize("at", sweep(len(CHAIN), {1, 13, 14, 20, 27}))
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


@pytest.mark.parametrize("at", [1, 4, 8])
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


@pytest.mark.parametrize("at", sweep(len(PREPARE), {1, 9, 16}))
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
while True:
    n += 1
    cache.put(f"key {n}", json.dumps({"n": n, "text": "x" * 500_000}))
"""


def test_the_call_cache_survives_a_kill_during_a_write(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    for _ in range(5):  # each kill lands somewhere in a stream of writes
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
        child.send_signal(signal.SIGKILL)
        assert child.wait() == -signal.SIGKILL

        with closing(sqlite3.connect(db)) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            answers = connection.execute("SELECT key, response FROM calls").fetchall()
        assert answers
        for key, response in answers:  # each answer is there whole
            assert json.loads(response)["n"] == int(key.split()[1])
