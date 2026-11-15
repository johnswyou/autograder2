"""A mock OpenRouter that answers each chat request with the response recorded under
its cache key, so that a stage runs as it would live, with a cold call cache, while
nothing leaves the machine.

A request carries no cache key (its prompt version is never sent), so the key the
gateway makes for each call is kept in a context variable that the transport reads:
the transport runs in the task that made the call. `install` patches
`autograder2.llm.client.cache_key` to keep it, and gives the orchestrator's and the
CLI's gateways and the privacy gate this transport and an injected sleep."""

import asyncio
import functools
import json
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import httpx
import pytest

from autograder2.llm.cache import cache_key
from autograder2.llm.client import Gateway
from autograder2.llm.models import recheck_providers
from tests.pinned import ZDR

RECORDINGS = Path(__file__).parent / "recordings"
_KEY: ContextVar[str] = ContextVar("key")


def recorded(*names: str) -> dict[str, dict[str, Any]]:
    """Each response of the recordings `names`, by cache key."""
    return {
        row["key"]: row["response"]
        for name in names
        for row in map(
            json.loads, (RECORDINGS / f"{name}.jsonl").read_text().splitlines()
        )
    }


def _keyed(*args: Any, **kwargs: Any) -> str:
    key = cache_key(*args, **kwargs)
    _KEY.set(key)
    return key


async def no_wait(seconds: float) -> None:
    """The gateway's sleep in a test: it returns at once, once the other tasks have
    had their turn, as they would while it waited."""
    await asyncio.sleep(0)


class Provider:
    """Answers from `responses` (by cache key); keeps the key of every chat request
    in `asked`, in order. `fault(key, n)`, given, is asked first about the `n`th
    chat request and may answer it instead, or raise. A key with no response is
    refused with HTTP 400, so that a test sees it in `unrecorded`."""

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.asked: list[str] = []
        self.fault: Callable[[str, int], httpx.Response | None] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/endpoints/zdr"):
            return httpx.Response(200, json={"data": ZDR})
        key = _KEY.get()
        self.asked.append(key)
        if self.fault is not None and (reply := self.fault(key, len(self.asked))):
            return reply
        if key not in self.responses:
            return httpx.Response(400, json={"error": {"message": "not recorded"}})
        return httpx.Response(200, json=self.responses[key])

    @property
    def unrecorded(self) -> list[str]:
        return [k for k in self.asked if k not in self.responses]

    def install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sleep: Callable[[float], Any] = no_wait,
    ) -> None:
        monkeypatch.setattr("autograder2.llm.client.cache_key", _keyed)
        transport = httpx.MockTransport(self)
        gateway = functools.partial(Gateway, transport=transport, sleep=sleep)
        monkeypatch.setattr("autograder2.orchestrator.Gateway", gateway)
        monkeypatch.setattr("autograder2.cli.Gateway", gateway)
        monkeypatch.setattr(
            "autograder2.orchestrator.recheck_providers",
            functools.partial(recheck_providers, transport=transport),
        )
