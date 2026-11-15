"""The API key never shows in a formatted traceback, with locals or without, nor on
the terminal or in `run.log` when a command fails. Every key here is made up, and
no test frame holds one in a local of its own: only `autograder2`'s frames are tried."""

import asyncio
import subprocess
import sys
import traceback
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from autograder2.cli import _key, _parse
from autograder2.config import Config, Lock, openrouter_api_key, pin
from autograder2.llm.client import Gateway
from autograder2.llm.models import fetch_catalog, recheck_providers
from autograder2.orchestrator import grade_class, privacy_problems
from autograder2.schemas.key import Key
from autograder2.workspace import Workspace
from tests.fixture_truth import FIXTURES, SPEC
from tests.pinned import GRADE_MODELS, GRADER, KEY_MODELS

# Made-up keys by name. "not ascii" is a key pasted with a character no header may
# hold: the HTTP client refuses it while the key is in every frame from the caller.
KEYS = {
    "ascii": "made-up-key-0f3c9a7e51d24b68a1",
    "not ascii": "made-up-key-é-7d02b6c4e19f",
}


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


def refused(request: httpx.Request) -> httpx.Response:
    raise RuntimeError("the transport refused the request")


TRANSPORT = httpx.MockTransport(refused)
LOCK = Lock(profile="balanced", models=KEY_MODELS)


def missing(tmp_path: Path) -> Workspace:
    """A workspace whose folder does not exist, so its cache cannot be opened."""
    return Workspace(tmp_path / "not-made")


async def gateway_opened(tmp_path: Path, key: str) -> object:
    async with Gateway(missing(tmp_path).cache_sqlite, key):
        return None


async def gateway_made(tmp_path: Path, key: str) -> object:
    async with Gateway(tmp_path / "cache.sqlite", key):
        return None


async def gateway_called(tmp_path: Path, key: str) -> object:
    async with Gateway(tmp_path / "cache.sqlite", key, transport=TRANSPORT) as gw:
        return await gw.call(
            "grader", GRADER, "system", ["shared"], ["item"], Answer, "1"
        )


async def catalog(tmp_path: Path, key: str) -> object:
    return await fetch_catalog(key, transport=TRANSPORT)


async def pinned(tmp_path: Path, key: str) -> object:
    return await pin(Workspace(tmp_path), Config(), key, transport=TRANSPORT)


async def graded(tmp_path: Path, key: str) -> object:
    assignment = Key.model_validate_json((FIXTURES / "key.json").read_text())
    ws = missing(tmp_path)
    return await grade_class(ws, SPEC, assignment, GRADE_MODELS, Config(), key, [])


async def parsed(tmp_path: Path, key: str) -> object:
    return await _parse(missing(tmp_path), Config(), LOCK, key)


async def keyed(tmp_path: Path, key: str) -> object:
    return await _key(missing(tmp_path), Config(), LOCK, key, SPEC)


type Case = Callable[[Path, str], Coroutine[Any, Any, object]]
CASES: list[tuple[str, str, Case]] = [
    ("the gateway's constructor", "ascii", gateway_opened),
    ("the gateway's HTTP client", "not ascii", gateway_made),
    ("a request", "ascii", gateway_called),
    ("the catalog", "ascii", catalog),
    ("the catalog's HTTP client", "not ascii", catalog),
    ("the pin", "ascii", pinned),
    ("the class's grading", "ascii", graded),
    ("the parse", "ascii", parsed),
    ("the key", "ascii", keyed),
]


def formatted(error: BaseException) -> list[str]:
    """`error` as the standard library prints it with every frame's locals, and as
    pytest prints it in long style with locals and arguments, causes included."""
    plain = traceback.TracebackException.from_exception(error, capture_locals=True)
    info = pytest.ExceptionInfo.from_exception(error)
    long = info.getrepr(style="long", showlocals=True, funcargs=True, chain=True)
    return ["".join(plain.format()), str(long)]


@pytest.mark.parametrize(
    ("which", "case"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_a_failure_holding_the_key_never_prints_it(
    which: str, case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEYS[which])

    with pytest.raises(Exception) as caught:
        asyncio.run(case(tmp_path, openrouter_api_key()))

    texts = formatted(caught.value)
    assert all("Traceback" in t or "def " in t for t in texts)  # frames were printed
    assert not [t for t in texts if KEYS[which] in t]


def test_the_privacy_check_that_fails_never_names_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", KEYS["ascii"])
    lock = Lock(profile="balanced", models=GRADE_MODELS)

    found = asyncio.run(
        recheck_providers(
            lock.models, openrouter_api_key(), strict_privacy=True, transport=TRANSPORT
        )
    )

    async def refusing(*args: object, **kwargs: object) -> object:
        raise RuntimeError("the check could not be completed")

    monkeypatch.setattr("autograder2.orchestrator.recheck_providers", refusing)
    gated = asyncio.run(privacy_problems(lock, openrouter_api_key(), True))
    assert found and gated
    assert KEYS["ascii"] not in repr(found) + repr(gated)


# Run as `python -c`: a `autograder2 grade` whose step raises with a plain copy of the
# key in a local, so that any printed local would show it.
DRIVER = """
import os, sys
from pathlib import Path
from tests.replay.test_run_replay import seed
from autograder2 import cli

workspace, submissions = seed(Path(sys.argv[1]))

def failing(*args, **kwargs):
    plain = "".join(os.environ["OPENROUTER_API_KEY"])  # noqa: F841, a plain copy
    raise RuntimeError("a step failed")

async def failing_async(*args, **kwargs):
    failing()

async def no_problems(*args, **kwargs):
    return []

if sys.argv[2] == "unhandled":
    cli._ingest = failing
else:
    cli.privacy_problems = no_problems
    cli.grade_class = failing_async
args = ["grade", str(submissions), "--workspace", str(workspace.root)]
cli.app(args, prog_name="autograder2")
"""


@pytest.mark.parametrize("how", ["unhandled", "logged"])
def test_a_failed_command_prints_and_logs_no_key(how: str, tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    env = {
        "OPENROUTER_API_KEY": KEYS["ascii"],
        "PYTHONPATH": f"{root / 'src'}:{root}",
        "PATH": "/usr/bin:/bin",
        "COLUMNS": "200",
    }
    done = subprocess.run(
        [sys.executable, "-c", DRIVER, str(tmp_path), how],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    log = tmp_path / "ws" / "run.log"
    logged = log.read_text() if log.exists() else ""
    assert done.returncode == 1, done.stderr[-2000:]
    assert "a step failed" in done.stderr + logged
    assert KEYS["ascii"] not in done.stdout + done.stderr + logged
