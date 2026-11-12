import asyncio
import base64
import json
import math
import sqlite3
import traceback
from collections.abc import Awaitable, Callable, Sequence
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from redpen.llm.client import (
    BACKOFF_SECONDS,
    MAX_ATTEMPTS,
    FatalError,
    Gateway,
    Part,
    ResponseValidationError,
    Result,
    RetryableError,
    without_validation_keywords,
)
from redpen.llm.ledger import Ledger, Usage
from redpen.llm.models import ModelInfo

MODEL = ModelInfo(
    slug="vendor/model-1",
    supported_parameters=frozenset({"temperature", "structured_outputs"}),
    accepts_images=True,
    prompt_price=Decimal("0.000001"),
    completion_price=Decimal("0.000002"),
    providers=frozenset({"zero/us", "zero"}),
)
PNG = b"\x89PNG\r\n\x1a\nnot really a picture"


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    letter: str = Field(pattern="^[A-Z]$")


# Differs from Answer only in a validation keyword, which is not sent: the cache
# key must still tell them apart.
class LooseAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Answer")

    letter: str


def completion(
    content: str, model: str = MODEL.slug, usage: dict[str, Any] | None = None
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": usage
            or {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001},
        },
    )


class Server:
    """Answers each request with the next reply, the last one repeating.

    A string reply becomes a completion served by the model that was asked for.
    """

    def __init__(self, *replies: str | httpx.Response) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return (
            reply
            if isinstance(reply, httpx.Response)
            else completion(reply, body["model"])
        )


def run[T](
    db: Path,
    handler: Callable[[httpx.Request], Any],
    scenario: Callable[[Gateway], Awaitable[T]],
    **options: Any,
) -> T:
    async def main() -> T:
        options.setdefault("requests_per_second", math.inf)
        transport = httpx.MockTransport(handler)
        async with Gateway(db, "test-key", transport=transport, **options) as gateway:
            return await scenario(gateway)

    return asyncio.run(main())


def ask(
    shared: Sequence[Part] = ("shared text",),
    items: Sequence[Part] = ("item text", PNG),
    model: ModelInfo = MODEL,
    response_model: type[BaseModel] = Answer,
    prompt_version: str = "v1",
    params: dict[str, Any] | None = None,
) -> Callable[[Gateway], Awaitable[Result[BaseModel]]]:
    def scenario(gateway: Gateway) -> Awaitable[Result[BaseModel]]:
        return gateway.call(
            "reader",
            model,
            "system text",
            shared,
            items,
            response_model,
            prompt_version,
            params={"temperature": 0} if params is None else params,
            session_id="assignment:Q1",
        )

    return scenario


def totals(db: Path) -> dict[str, Usage]:
    with closing(sqlite3.connect(db)) as connection:
        return Ledger(connection).totals_by_role()


def test_request_has_the_designed_shape(tmp_path: Path) -> None:
    server = Server('{"letter": "B"}')

    result = run(
        tmp_path / "cache.sqlite",
        server,
        ask(params={"temperature": 0, "top_k": 5}),
    )

    assert result.value == Answer(letter="B")
    assert (result.model_slug, result.from_cache) == (MODEL.slug, False)
    assert result.usage == Usage(10, 2, Decimal("0.0001"))
    [body] = server.requests
    assert body["model"] == MODEL.slug
    assert body["messages"] == [
        {"role": "system", "content": "system text"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "shared text",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "item text"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(PNG).decode()
                    },
                },
            ],
        },
    ]
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "strict": True,
            "schema": without_validation_keywords(Answer.model_json_schema()),
        },
    }
    assert body["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "only": ["zero", "zero/us"],
    }
    assert body["session_id"] == "assignment:Q1"
    assert body["temperature"] == 0
    assert "top_k" not in body  # not supported by MODEL
    assert "models" not in body


def test_relaxed_privacy_keeps_only_require_parameters(tmp_path: Path) -> None:
    server = Server('{"letter": "B"}')

    run(tmp_path / "cache.sqlite", server, ask(), strict_privacy=False)

    assert server.requests[0]["provider"] == {"require_parameters": True}


def test_validation_keywords_are_removed_but_property_names_stay() -> None:
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "pattern": "^a$", "maxLength": 3},
            "n": {
                "anyOf": [{"type": "integer", "exclusiveMinimum": 0}, {"type": "null"}]
            },
            "xs": {"type": "array", "items": {"$ref": "#/$defs/X"}, "minItems": 1},
        },
        "$defs": {"X": {"type": "number", "minimum": 0, "maximum": 1}},
    }

    assert without_validation_keywords(schema) == {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "n": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "xs": {"type": "array", "items": {"$ref": "#/$defs/X"}},
        },
        "$defs": {"X": {"type": "number"}},
    }


def test_a_schema_failing_the_strict_lint_is_refused_before_any_request(
    tmp_path: Path,
) -> None:
    class Optional(BaseModel):
        letter: str = "A"

    server = Server('{"letter": "B"}')

    with pytest.raises(ValueError, match="is not a strict schema"):
        run(tmp_path / "cache.sqlite", server, ask(response_model=Optional))
    assert server.requests == []


def test_a_repeated_call_is_served_from_the_cache(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    server = Server('{"letter": "B"}')

    first = run(db, server, ask())
    again = run(db, server, ask())

    assert len(server.requests) == 1
    assert (again.value, again.model_slug, again.from_cache) == (
        first.value,
        MODEL.slug,
        True,
    )
    assert totals(db) == {"reader": Usage(10, 2, Decimal("0.0001"))}


@pytest.mark.parametrize(
    "changed",
    [
        ask(model=MODEL.model_copy(update={"slug": "vendor/model-2"})),
        ask(shared=("other shared text",)),
        ask(items=("item text", PNG + b"!")),
        ask(response_model=LooseAnswer),
        ask(params={"temperature": 0.5}),
        ask(prompt_version="v2"),
    ],
    ids=["slug", "text", "image", "schema", "parameter", "prompt version"],
)
def test_changing_one_input_misses_the_cache(
    tmp_path: Path, changed: Callable[[Gateway], Awaitable[Result[BaseModel]]]
) -> None:
    db = tmp_path / "cache.sqlite"
    server = Server('{"letter": "B"}')

    run(db, server, ask())
    result = run(db, server, changed)

    assert not result.from_cache
    assert len(server.requests) == 2


def test_an_invalid_response_is_repaired_once(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    server = Server('{"letter": "bb"}', '{"letter": "B"}')

    result = run(db, server, ask())

    assert result.value == Answer(letter="B")
    repair = server.requests[1]["messages"]
    assert repair[:2] == server.requests[0]["messages"]
    assert repair[2] == {"role": "assistant", "content": '{"letter": "bb"}'}
    assert repair[3]["role"] == "user" and "letter" in repair[3]["content"]
    assert server.requests[1]["provider"]["only"] == ["zero", "zero/us"]
    assert totals(db)["reader"].cost == Decimal("0.0002")  # both calls were paid for


def test_a_second_invalid_response_raises(tmp_path: Path) -> None:
    server = Server('{"letter": "bb"}')

    with pytest.raises(ResponseValidationError):
        run(tmp_path / "cache.sqlite", server, ask())
    assert len(server.requests) == 2


def sleeps_into(sleeps: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return sleep


@pytest.mark.parametrize(
    "failure",
    [
        httpx.Response(429, json={"error": {"code": 429, "message": "Rate limited"}}),
        httpx.Response(502, json={"error": {"code": 502, "message": "Provider down"}}),
        httpx.Response(200, json={"error": {"code": 502, "message": "Disconnected"}}),
    ],
    ids=["429", "502", "error inside a 200"],
)
def test_a_transient_failure_is_retried_after_a_backoff(
    tmp_path: Path, failure: httpx.Response
) -> None:
    server = Server(failure, '{"letter": "B"}')
    sleeps: list[float] = []

    result = run(tmp_path / "cache.sqlite", server, ask(), sleep=sleeps_into(sleeps))

    assert result.value == Answer(letter="B")
    assert len(server.requests) == 2
    [delay] = sleeps
    assert BACKOFF_SECONDS <= delay <= 2 * BACKOFF_SECONDS


def test_retries_end_with_the_retryable_error(tmp_path: Path) -> None:
    server = Server(httpx.Response(429, json={"error": {"code": 429}}))
    sleeps: list[float] = []

    with pytest.raises(RetryableError):
        run(tmp_path / "cache.sqlite", server, ask(), sleep=sleeps_into(sleeps))
    assert len(server.requests) == MAX_ATTEMPTS
    assert sleeps == sorted(sleeps)


def test_a_client_error_is_fatal_at_once(tmp_path: Path) -> None:
    server = Server(httpx.Response(400, json={"error": {"code": 400}}))

    with pytest.raises(FatalError, match="HTTP 400"):
        run(tmp_path / "cache.sqlite", server, ask())
    assert len(server.requests) == 1


def test_a_response_from_another_model_raises(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    server = Server(completion('{"letter": "B"}', model="vendor/other-model"))

    with pytest.raises(FatalError, match="served by vendor/other-model"):
        run(db, server, ask())
    assert totals(db)["reader"].cost == Decimal("0.0001")  # it was paid for


def test_a_missing_cost_is_priced_from_the_model(tmp_path: Path) -> None:
    db = tmp_path / "cache.sqlite"
    usage = {"prompt_tokens": 10, "completion_tokens": 2}
    server = Server(completion('{"letter": "B"}', usage=usage))

    run(db, server, ask())

    assert totals(db)["reader"].cost == Decimal("0.000014")


def test_no_more_requests_in_flight_than_the_concurrency_limit(tmp_path: Path) -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return completion('{"letter": "B"}')

    async def scenario(gateway: Gateway) -> None:
        calls = [ask(items=(f"item {n}",))(gateway) for n in range(6)]
        await asyncio.gather(*calls)

    run(tmp_path / "cache.sqlite", handler, scenario, max_concurrency=2)

    assert peak == 2


def test_requests_to_one_model_are_spaced_by_the_rate_limit(tmp_path: Path) -> None:
    sleeps: list[float] = []

    async def scenario(gateway: Gateway) -> None:
        await asyncio.gather(*[ask(items=(f"item {n}",))(gateway) for n in range(3)])

    run(
        tmp_path / "cache.sqlite",
        Server('{"letter": "B"}'),
        scenario,
        requests_per_second=2,
        sleep=sleeps_into(sleeps),
    )

    assert sleeps == pytest.approx([0.5, 1.0], abs=0.05)


def test_strict_privacy_refuses_a_model_with_no_pinned_provider(
    tmp_path: Path,
) -> None:
    server = Server('{"letter": "B"}')
    unpinned = MODEL.model_copy(update={"providers": frozenset()})

    with pytest.raises(ValueError, match="no pinned endpoint that retains nothing"):
        run(tmp_path / "cache.sqlite", server, ask(model=unpinned))
    assert server.requests == []


def test_a_stored_response_that_no_longer_validates_raises_without_a_request(
    tmp_path: Path,
) -> None:
    db = tmp_path / "cache.sqlite"
    server = Server('{"letter": "B"}')
    run(db, server, ask())
    stale = completion('{"letter": "bb"}').json()
    with closing(sqlite3.connect(db, autocommit=True)) as connection:
        connection.execute("UPDATE calls SET response = ?", (json.dumps(stale),))

    with pytest.raises(
        ResponseValidationError, match="Bump prompt version v1"
    ) as error:
        run(db, server, ask())
    assert len(server.requests) == 1
    assert "bb" not in str(error.value)


SECRET = "the student wrote that the answer is 42"


@pytest.mark.parametrize(
    ("reply", "raised"),
    [
        (json.dumps({"letter": SECRET}), ResponseValidationError),
        (
            httpx.Response(
                400,
                json={"error": {"code": 400, "message": "Bad", "metadata": SECRET}},
            ),
            FatalError,
        ),
        (
            httpx.Response(
                200,
                json={
                    "model": MODEL.slug,
                    "choices": [
                        {
                            "message": {"content": SECRET},
                            "finish_reason": "error",
                            "error": {"code": 400, "metadata": {"raw": SECRET}},
                        }
                    ],
                },
            ),
            FatalError,
        ),
    ],
    ids=["invalid twice", "HTTP error", "error inside a 200"],
)
def test_an_error_does_not_quote_the_response(
    tmp_path: Path, reply: str | httpx.Response, raised: type[Exception]
) -> None:
    with pytest.raises(raised) as error:
        run(tmp_path / "cache.sqlite", Server(reply), ask())

    assert SECRET not in "".join(traceback.format_exception(error.value))
