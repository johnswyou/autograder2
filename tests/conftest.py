import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from redpen.config import openrouter_api_key
from redpen.llm.cache import CallCache
from redpen.llm.client import Gateway

RECORDINGS = Path(__file__).parent / "recordings"


def _no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a replayed test reached the network: {request.url}")


@pytest.fixture
def replay(tmp_path: Path) -> Iterator[Callable[[str], Gateway]]:
    """`replay(name)`: an offline gateway answering from tests/recordings/<name>.jsonl.

    With REDPEN_RECORD=1 the gateway calls OpenRouter instead (paid), and the
    recording is rewritten when the test ends.
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
    for recording, db in recorded.items():
        RECORDINGS.mkdir(exist_ok=True)
        with closing(sqlite3.connect(db)) as connection:
            CallCache(connection).dump(recording)
