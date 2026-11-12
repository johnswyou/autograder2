import os
import shutil
import sqlite3
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from redpen.config import openrouter_api_key
from redpen.grade.ingest import IngestSummary, ingest
from redpen.llm.cache import CallCache
from redpen.llm.client import Gateway
from redpen.prepare.parse import render_blank
from redpen.workspace import Workspace
from tests.fixture_truth import FIXTURES, SUBMISSIONS

pytest_plugins = ["pytester"]

RECORDINGS = Path(__file__).parent / "recordings"
CALL_REPORT = pytest.StashKey[pytest.TestReport]()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Keep each test's call-phase report, for `replay` to read at teardown."""
    report = yield
    if report.when == "call":
        item.stash[CALL_REPORT] = report
    return report


@pytest.fixture(scope="session")
def fixture_workspace(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Workspace, IngestSummary]:
    """A workspace with the fixture blank (`blank.pdf`), its page images and
    `spec.json`, and every fixture submission ingested. Made once per session and
    shared: a test copies what it would change."""
    root = tmp_path_factory.mktemp("fixtures")
    source = root / "submissions"
    source.mkdir()
    for pdf in SUBMISSIONS:
        shutil.copyfile(pdf, source / pdf.name)
    workspace = Workspace(root / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    shutil.copyfile(FIXTURES / "spec.json", workspace.spec_json)
    render_blank(workspace)
    return workspace, ingest(source, workspace)


def _no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a replayed test reached the network: {request.url}")


@pytest.fixture
def replay(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[Callable[[str], Gateway]]:
    """`replay(name)`: an offline gateway answering from tests/recordings/<name>.jsonl.

    With REDPEN_RECORD=1 the gateway calls OpenRouter instead (paid), and the
    recording is rewritten when the test passes. A test that fails or skips leaves
    the recording as it was.
    """
    record = os.environ.get("REDPEN_RECORD") == "1"
    recorded: dict[Path, Path] = {}

    def gateway(name: str) -> Gateway:
        db, recording = tmp_path / f"{name}.sqlite", RECORDINGS / f"{name}.jsonl"
        if record:
            recorded[recording] = db
            return Gateway(db, openrouter_api_key())
        with closing(sqlite3.connect(db, autocommit=True)) as connection:
            CallCache(connection).load(recording)
        transport = httpx.MockTransport(_no_network)
        return Gateway(db, "no key", offline=True, transport=transport)

    yield gateway
    report = request.node.stash.get(CALL_REPORT, None)
    if report is None or not report.passed:
        return
    for recording, db in recorded.items():
        RECORDINGS.mkdir(exist_ok=True)
        with closing(sqlite3.connect(db)) as connection:
            CallCache(connection).dump(recording)
