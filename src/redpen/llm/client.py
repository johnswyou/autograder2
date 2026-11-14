"""The gateway: every model call goes through `Gateway.call` (design 10.2 and 10.3)."""

import asyncio
import base64
import json
import random
import sqlite3
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn, Self, cast

import httpx
from pydantic import BaseModel, ValidationError

from redpen.llm.cache import CallCache, cache_key, image_digest
from redpen.llm.ledger import Ledger, Usage
from redpen.llm.models import API_URL, ModelInfo, authorization
from redpen.schemas.strict import strict_problems

MAX_ATTEMPTS = 6
BACKOFF_SECONDS = 1.0

# Keywords that only narrow the valid values. Provider strict modes differ on which
# they accept, and the full Pydantic model validates every response anyway.
VALIDATION_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)

type Part = str | bytes
"""Text, or a PNG or JPEG image."""


class RetryableError(Exception):
    """A rate limit or server error on every attempt."""


class FatalError(Exception):
    """A failure retrying cannot fix: a bad request, no credits, no provider that
    meets the routing requirements, a response served by another model, or a cache
    miss while offline."""


class ResponseValidationError(Exception):
    """The response failed validation, and so did the one repair attempt. Or a
    stored response no longer validates."""


class StageError(Exception):
    """A model's answer still fails its stage's checks after one retry. The message
    lists every failure."""


@dataclass(frozen=True)
class Result[T: BaseModel]:
    value: T
    model_slug: str
    """The slug that served the call, from the response."""
    usage: Usage
    """Of the response `value` came from. On a cache hit nothing was spent now."""
    from_cache: bool


def without_validation_keywords(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` without the `VALIDATION_KEYWORDS`. Property names stay."""
    node = {
        key: value for key, value in schema.items() if key not in VALIDATION_KEYWORDS
    }
    for keyword in ("properties", "$defs"):
        if keyword in node:
            node[keyword] = {
                name: without_validation_keywords(child)
                for name, child in node[keyword].items()
            }
    for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if keyword in node:
            node[keyword] = [without_validation_keywords(c) for c in node[keyword]]
    for keyword in ("items", "additionalProperties"):
        if isinstance(node.get(keyword), dict):
            node[keyword] = without_validation_keywords(node[keyword])
    return node


class Gateway:
    """Owns all traffic to OpenRouter. Use as `async with Gateway(...) as gateway`.

    `db_path` holds the call cache and the usage ledger. `max_concurrency` limits
    the requests in flight, `requests_per_second` the requests started per model.
    With `strict_privacy` every request goes only to the model's `providers`, the
    endpoints that retain nothing; `False` drops that and the zero-retention and
    no-data-collection requirements. With `offline` a cache miss raises and no
    request is ever made.
    """

    def __init__(
        self,
        db_path: Path,
        api_key: str,
        *,
        strict_privacy: bool = True,
        offline: bool = False,
        max_concurrency: int = 8,
        requests_per_second: float = 4.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._db = sqlite3.connect(db_path, autocommit=True)
        self._cache = CallCache(self._db)
        self._ledger = Ledger(self._db)
        self._http = httpx.AsyncClient(
            base_url=API_URL,
            headers=authorization(api_key),
            timeout=httpx.Timeout(600, connect=10),
            transport=transport,
        )
        self._strict_privacy = strict_privacy
        self._offline = offline
        self.max_concurrency = max_concurrency
        """The most requests in flight at once."""
        self._slots = asyncio.Semaphore(max_concurrency)
        self._interval = 1 / requests_per_second
        self._next_start: dict[str, float] = {}
        self._sleep = sleep

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()
        self._db.close()

    async def call[T: BaseModel](
        self,
        role: str,
        model: ModelInfo,
        system: str,
        shared: Sequence[Part],
        items: Sequence[Part],
        response_model: type[T],
        prompt_version: str,
        *,
        params: Mapping[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Result[T]:
        """A validated `response_model`, from the cache or from `model`.

        The cache breakpoint sits between `shared` and `items`. A parameter in
        `params` is sent only if `model` supports it. `role` labels the ledger rows.
        """
        name = response_model.__name__
        schema = response_model.model_json_schema()
        if problems := strict_problems(schema):
            raise ValueError(f"{name} is not a strict schema: {'; '.join(problems)}")
        provider: dict[str, Any] = {"require_parameters": True}
        if self._strict_privacy:
            if not model.providers:
                raise ValueError(
                    f"{model.slug} has no pinned endpoint that retains nothing,"
                    " and strict privacy sends it nothing"
                )
            provider |= {
                "data_collection": "deny",
                "zdr": True,
                "only": sorted(model.providers),
            }
        sent = {
            k: v for k, v in (params or {}).items() if k in model.supported_parameters
        }
        keyed = _messages(system, shared, items, digest=True)
        key = cache_key(model.slug, keyed, schema, sent, prompt_version)
        if (asked := _asked.get()) is not None:
            asked.append((self, key))
        if (raw := self._cache.get(key)) is not None:
            response = json.loads(raw)
            try:
                value = response_model.model_validate_json(_content(response))
            except ValidationError as error:
                # Fetching it again could change a grade (design 10.3).
                raise ResponseValidationError(
                    f"the stored {name} for role {role} no longer validates"
                    f" ({_locations(error, schema)})."
                    f" Bump prompt version {prompt_version}."
                ) from None
            return Result(value, response["model"], _usage(response, model), True)
        if self._offline:
            raise FatalError(
                f"offline, and nothing is recorded for role {role}"
                f" at prompt version {prompt_version}"
            )

        body: dict[str, Any] = {
            **sent,
            "model": model.slug,
            "messages": _messages(system, shared, items, digest=False),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": without_validation_keywords(schema),
                },
            },
            "provider": provider,
        }
        if session_id is not None:
            body["session_id"] = session_id
        response = await self._post(role, model, body)
        output = _content(response)
        try:
            value = response_model.model_validate_json(output)
        except ValidationError as error:
            invalid = [{"role": "assistant", "content": output}] if output else []
            body["messages"] = [
                *body["messages"],
                *invalid,
                {"role": "user", "content": str(error)},
            ]
            response = await self._post(role, model, body)
            try:
                value = response_model.model_validate_json(_content(response))
            except ValidationError as again:
                cut = _first_choice(response).get("finish_reason") == "length"
                # `from None`: Pydantic's own text quotes the output, which comes
                # from student work.
                raise ResponseValidationError(
                    f"{model.slug} returned an invalid {name} twice"
                    f" ({_locations(again, schema)})"
                    + (", the response cut off at the token limit" if cut else "")
                ) from None
        self._cache.put(key, json.dumps(response))
        return Result(value, response["model"], _usage(response, model), False)

    def forget(self, key: str) -> None:
        """Drop the cached response under `key`, so that it is asked for again. The
        ledger keeps its row: it was paid for."""
        self._cache.forget(key)

    async def _post(
        self, role: str, model: ModelInfo, body: dict[str, Any]
    ) -> dict[str, Any]:
        """The response to `body`, retried with backoff on a transient failure."""
        for attempt in range(MAX_ATTEMPTS - 1):
            try:
                return await self._attempt(role, model, body)
            except RetryableError:
                await self._sleep(BACKOFF_SECONDS * 2**attempt * random.uniform(1, 2))
        return await self._attempt(role, model, body)

    async def _attempt(
        self, role: str, model: ModelInfo, body: dict[str, Any]
    ) -> dict[str, Any]:
        async with self._slots:
            # Waiting inside the slot stops queued requests bursting past the rate.
            await self._wait_turn(model.slug)
            try:
                reply = await self._http.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                raise RetryableError(f"{model.slug}: {error!r}") from error
        try:
            response: dict[str, Any] = reply.json()
        except ValueError as error:
            if reply.status_code == 200:
                raise RetryableError(
                    f"{model.slug}: the response is not JSON"
                ) from error
            response = {}
        if (code := reply.status_code) != 200:
            _raise_for(
                code, f"{model.slug}: HTTP {code}{_message(response.get('error'))}"
            )
        self._ledger.record(role, model.slug, _usage(response, model))
        # A provider failure can arrive inside a 200 response.
        choice = _first_choice(response)
        failure: dict[str, Any] = response.get("error") or choice.get("error") or {}
        if failure or choice.get("finish_reason") == "error":
            code = failure.get("code", 502)
            _raise_for(code, f"{model.slug}: error {code}{_message(failure)}")
        if response.get("model") != model.slug:
            raise FatalError(
                f"asked for {model.slug}, served by {response.get('model')}"
            )
        return response

    async def _wait_turn(self, slug: str) -> None:
        """Start requests to `slug` at most `requests_per_second` a second."""
        now = time.monotonic()
        start = max(now, self._next_start.get(slug, now))
        self._next_start[slug] = start + self._interval
        if start > now:
            await self._sleep(start - now)


# The calls made while `ask_checked` waits for an answer, as (gateway, cache key). A
# context variable, so that the stage's own `ask` needs no change and concurrent
# checks, each in its own task, keep separate lists.
_asked: ContextVar[list[tuple[Gateway, str]] | None] = ContextVar("asked", default=None)


async def ask_checked[T](
    ask: Callable[[Sequence[Part]], Awaitable[T]],
    problems: Callable[[T], list[str]],
    what: str,
) -> T:
    """`ask([])`'s answer if `problems` finds nothing in it. Otherwise ask once more
    with the problems attached as the item `{"problems": [...]}`. If any remain,
    forget both cached responses, so that a later run asks again instead of failing
    on the cache, and raise `StageError`. `ask` makes its calls through a Gateway.
    `problems` lists failures in a fixed order: they become part of a cache key."""
    asked: list[tuple[Gateway, str]] = []
    token = _asked.set(asked)
    try:
        found = await ask([])
        if wrong := problems(found):
            found = await ask([json.dumps({"problems": wrong})])
            if wrong := problems(found):
                for gateway, key in asked:
                    gateway.forget(key)
                raise StageError(f"{what} fails its checks: {'; '.join(wrong)}")
        return found
    finally:
        _asked.reset(token)


async def finish_all[T](awaitables: Iterable[Awaitable[T]]) -> list[T]:
    """The results of `awaitables`, run concurrently, once every one has finished;
    then the first error, if any, with a note naming each later one. One failure
    does not cut the others' calls short, so none of them is paid for twice."""
    results = await asyncio.gather(*awaitables, return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        for later in errors[1:]:
            errors[0].add_note(f"also failed: {type(later).__name__}: {later}")
        raise errors[0]
    return cast(list[T], results)


def _raise_for(code: int, message: str) -> NoReturn:
    transient = code in (408, 429) or code >= 500
    raise (RetryableError if transient else FatalError)(message)


# Exception text never quotes a response: model output is derived from student work.


def _message(failure: dict[str, Any] | None) -> str:
    """`: ` and the start of an OpenRouter error's message, or "" if it has none."""
    message = (failure or {}).get("message")
    return f": {str(message)[:200]}" if message else ""


def _locations(error: ValidationError, schema: dict[str, Any]) -> str:
    """Each failure's type, and where it is as far as `schema`'s own field names and
    list indexes say. Other location parts, such as an unexpected key or a dict
    key, are text from the output."""
    fields = {
        name
        for node in [schema, *schema.get("$defs", {}).values()]
        for name in node.get("properties", {})
    }
    found: list[str] = []
    for e in error.errors():
        loc = [
            str(part)
            for part in e["loc"]
            if e["type"] != "extra_forbidden"
            and (isinstance(part, int) or part in fields)
        ]
        found.append(e["type"] + (f" at {'.'.join(loc)}" if loc else ""))
    return "; ".join(found)


def _messages(
    system: str, shared: Sequence[Part], items: Sequence[Part], *, digest: bool
) -> list[dict[str, Any]]:
    """The chat messages. With `digest`, images are SHA-256 digests, for the key."""
    content = [_part(part, digest) for part in [*shared, *items]]
    if shared:
        content[len(shared) - 1]["cache_control"] = {"type": "ephemeral"}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


def _part(part: Part, digest: bool) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    url = image_digest(part) if digest else _data_url(part)
    return {"type": "image_url", "image_url": {"url": url}}


def _data_url(image: bytes) -> str:
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        media_type = "image/png"
    elif image.startswith(b"\xff\xd8\xff"):
        media_type = "image/jpeg"
    else:
        raise ValueError("an image part must be PNG or JPEG bytes")
    return f"data:{media_type};base64,{base64.b64encode(image).decode()}"


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices: list[dict[str, Any]] = response.get("choices") or [{}]
    return choices[0]


def _content(response: dict[str, Any]) -> str:
    """The message text of `response`, or "" when it has none."""
    message: dict[str, Any] = _first_choice(response).get("message") or {}
    return message.get("content") or ""


def _usage(response: dict[str, Any], model: ModelInfo) -> Usage:
    """Tokens and cost from `usage`, priced from `model` if the cost is missing."""
    usage: dict[str, Any] = response.get("usage") or {}
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    cost = usage.get("cost")
    if cost is None:
        cost = prompt * model.prompt_price + completion * model.completion_price
    return Usage(prompt, completion, Decimal(str(cost)))
