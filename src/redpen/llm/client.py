"""The gateway: every model call goes through `Gateway.call` (design 10.2 and 10.3)."""

import asyncio
import base64
import json
import random
import re
import sqlite3
import time
from collections import defaultdict
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
# The longest `Retry-After` taken as asked, in seconds: OpenRouter's rate limits are
# per minute, and a longer wait would hold the whole run on one header.
MAX_WAIT = 60.0
# The failures no retry cures, each said in one line: every later request would
# meet them too, so the run stops at the first.
STOPS = {
    401: "OpenRouter refused the API key (HTTP 401). Set OPENROUTER_API_KEY to a"
    " valid key and run the same command again; what was answered is kept.",
    402: "The OpenRouter account has no credit left (HTTP 402). Add credit and run"
    " the same command again; what was answered is kept.",
    404: "OpenRouter no longer serves a pinned model or endpoint (HTTP 404)."
    " `redpen models check --workspace DIR` names it; run the same command again"
    " once it is served, or start a new workspace. What was answered is kept.",
}
NAME = "[name]"
"""What each name of the class is replaced by in the text posted to a model."""

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

    def __init__(self, message: str, wait: float | None = None) -> None:
        super().__init__(message)
        self.wait = wait
        """The seconds the provider asked to wait (`Retry-After`), if it did."""


class FatalError(Exception):
    """A failure retrying cannot fix: a bad request, no credits, no provider that
    meets the routing requirements, a response served by another model, or a cache
    miss while offline."""


class StopError(FatalError):
    """One of `STOPS`: once met, the gateway sends nothing more, and leaving its
    `async with` raises it."""


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
    request is ever made. Each of `names` (a class's, as ingest holds them) of two
    or more words is replaced by `NAME` in every text posted, after the cache key
    is made from the text as it was (design 14.1).
    """

    def __init__(
        self,
        db_path: Path,
        api_key: str,
        *,
        names: Iterable[str] = (),
        strict_privacy: bool = True,
        offline: bool = False,
        max_concurrency: int = 8,
        requests_per_second: float = 4.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._db = sqlite3.connect(db_path, autocommit=True)
        try:  # a raise here never reaches __aexit__, which closes the connection
            self._cache = CallCache(self._db)
            self._ledger = Ledger(self._db)
            self._names = _names_pattern(names)
            self._slots = asyncio.Semaphore(max_concurrency)
            self._interval = 1 / requests_per_second
            # Last, so that nothing after it can raise: it closes only when awaited.
            self._http = httpx.AsyncClient(
                base_url=API_URL,
                headers=authorization(api_key),
                timeout=httpx.Timeout(600, connect=10),
                transport=transport,
            )
        except BaseException:
            self._db.close()
            raise
        self._strict_privacy = strict_privacy
        self._offline = offline
        self.max_concurrency = max_concurrency
        """The most requests in flight at once."""
        self._next_start: dict[str, float] = {}
        self._held: dict[str, float] = {}  # by model: until when a Retry-After holds it
        self._sleep = sleep
        self._asking: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.stopped: StopError | None = None
        """The failure no retry cures that stopped the gateway, if one did."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._http.aclose()
        self._db.close()
        # Raised even where a stage took each failure as one unit's.
        if self.stopped is not None and exc_info[0] is None:
            raise self.stopped

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
        draw: int = 1,
    ) -> Result[T]:
        """A validated `response_model`, from the cache or from `model`.

        The cache breakpoint sits between `shared` and `items`. A parameter in
        `params` is sent only if `model` supports it. `role` labels the ledger rows.
        `draw` numbers independent answers to the same request: it is part of the
        cache key and is never sent.
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
        key = cache_key(model.slug, keyed, schema, sent, prompt_version, draw)
        if (asked := _asked.get()) is not None:
            asked.append((self, key))
        # One request at a time per key: an identical request already in flight is
        # awaited and then found in the cache, not sent again.
        async with self._asking[key]:
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
        """The response to `body`, its names replaced, retried with backoff on a
        transient failure."""
        if (names := self._names) is not None:
            body = {
                **body,
                "messages": [
                    {**m, "content": _unnamed(names, m["content"])}
                    for m in body["messages"]
                ],
            }
        for attempt in range(MAX_ATTEMPTS - 1):
            try:
                return await self._attempt(role, model, body)
            except RetryableError as error:
                backoff = BACKOFF_SECONDS * 2**attempt * random.uniform(1, 2)
                if error.wait is None:
                    await self._sleep(backoff)
                else:  # every request to the model waits it out, in `_wait_turn`
                    until = time.monotonic() + max(error.wait, backoff)
                    slug = model.slug
                    self._next_start[slug] = max(self._next_start.get(slug, 0), until)
                    self._held[slug] = max(self._held.get(slug, 0), until)
        return await self._attempt(role, model, body)

    async def _attempt(
        self, role: str, model: ModelInfo, body: dict[str, Any]
    ) -> dict[str, Any]:
        while True:
            self._go_on()
            # Its turn is waited for outside a slot, so that one model's wait holds
            # no request to another model.
            start = await self._wait_turn(model.slug)
            async with self._slots:
                self._go_on()  # only what is in flight is sent
                if self._held.get(model.slug, 0) > start:  # told to wait since
                    continue
                try:
                    reply = await self._http.post("/chat/completions", json=body)
                except httpx.TransportError as error:
                    raise RetryableError(f"{model.slug}: {error!r}") from error
            break
        try:
            response: dict[str, Any] = reply.json()
        except ValueError as error:
            if reply.status_code == 200:
                raise RetryableError(
                    f"{model.slug}: the response is not JSON"
                ) from error
            response = {}
        if (code := reply.status_code) != 200:
            if code in STOPS:
                self.stopped = StopError(STOPS[code])
                raise self.stopped
            message = f"{model.slug}: HTTP {code}{_message(response.get('error'))}"
            _raise_for(code, message, _retry_after(reply))
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

    def _go_on(self) -> None:
        """Raise the stop, once one is met: nothing more is sent."""
        if self.stopped is not None:
            raise StopError(*self.stopped.args)

    async def _wait_turn(self, slug: str) -> float:
        """Start requests to `slug` at most `requests_per_second` a second; the start
        taken."""
        now = time.monotonic()
        start = max(now, self._next_start.get(slug, now))
        self._next_start[slug] = start + self._interval
        if start > now:
            await self._sleep(start - now)
        return start


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
    errors.sort(key=lambda e: isinstance(e, StopError))  # a fault before the stop
    if errors:
        for later in errors[1:]:
            errors[0].add_note(f"also failed: {type(later).__name__}: {later}")
        raise errors[0]
    return cast(list[T], results)


def _raise_for(code: int, message: str, wait: float | None = None) -> NoReturn:
    if code in (408, 429) or code >= 500:
        raise RetryableError(message, wait)
    raise FatalError(message)


def _retry_after(reply: httpx.Response) -> float | None:
    """The seconds a `Retry-After` header asks for, at most `MAX_WAIT`; None without
    one in seconds (OpenRouter sends seconds)."""
    try:
        wait = float(reply.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return min(wait, MAX_WAIT) if wait >= 0 else None  # NaN fails the comparison


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


# Between two words of a name: white space, or its escape inside a JSON string.
_SPACE = r"(?:\s|\\[ntr])"


def _names_pattern(names: Iterable[str]) -> re.Pattern[str] | None:
    """Each of `names` of two or more words, in its own order or its last word first
    and a comma, in any letter case, never inside a longer word; `None` for none.
    A single word is too often an ordinary word or a unit."""
    forms: set[str] = set()
    for name in names:
        # Each word as written, or as `json.dumps` escapes it in a text part.
        words = [
            "(?:"
            + "|".join(dict.fromkeys(map(re.escape, (w, json.dumps(w)[1:-1]))))
            + ")"
            for w in name.split()
        ]
        if len(words) > 1:
            gap = _SPACE + "+"
            forms |= {
                gap.join(words),
                words[-1] + "," + _SPACE + "*" + gap.join(words[:-1]),
            }
    if not forms:
        return None
    # Longest first, so that "Ann Lee Ng" is tried before "Ann Lee".
    longest = "|".join(sorted(forms, key=lambda f: (-len(f), f)))
    return re.compile(rf"(?:(?<!\w)|(?<=\\[ntr]))(?:{longest})(?!\w)", re.IGNORECASE)


def _unnamed(names: re.Pattern[str], content: Any) -> Any:
    """A message's content, string or parts, with `NAME` for each of `names` in its
    text."""
    if isinstance(content, str):
        return names.sub(NAME, content)
    return [
        {**p, "text": names.sub(NAME, p["text"])} if p["type"] == "text" else p
        for p in content
    ]


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
