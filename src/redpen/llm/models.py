"""Model metadata from OpenRouter's models API, and the checks a role's model must
pass (design 10.1)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

API_URL = "https://openrouter.ai/api/v1"


class Secret(str):
    """The API key, or text holding it: unchanged as text, but its repr is a fixed
    mask, so that a traceback, a locals dump or a debugger never shows it."""

    def __repr__(self) -> str:
        return "'<secret>'"


def authorization(api_key: str) -> dict[str, str]:
    """The header that carries `api_key`, its value masked as the key is."""
    return {"Authorization": Secret(f"Bearer {api_key}")}


class ModelError(Exception):
    """A configured model cannot be used. The message is meant for the user."""


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str
    """Concrete OpenRouter slug, the models API `id`, such as `openai/gpt-6.1-sol`."""
    supported_parameters: frozenset[str]
    accepts_images: bool
    prompt_price: Decimal
    """USD per prompt token."""
    completion_price: Decimal
    """USD per completion token."""
    providers: frozenset[str] = frozenset()
    """Endpoint slugs that retain nothing: zero data retention and no Safety
    Retention. With strict privacy every request goes only to these. Empty when
    privacy is relaxed."""


@dataclass(frozen=True)
class Catalog:
    """What OpenRouter lists: its models by `id`, and its zero-data-retention
    endpoints, Safety Retention ones included."""

    models: Mapping[str, Mapping[str, Any]]
    zdr_endpoints: Sequence[Mapping[str, Any]]

    def resolve(self, name: str, *, strict_privacy: bool) -> ModelInfo:
        """The concrete model behind `name`, a slug or a `~author/family-latest`
        alias. With `strict_privacy`, `providers` is filled."""
        entry = self.models.get(name)
        if entry is not None and (target := entry.get("alias_target")):
            entry = self.models.get(target["slug"])
        if entry is None or entry["id"].startswith("~"):
            raise ModelError(f"OpenRouter lists no concrete model for {name}")
        slug: str = entry["id"]
        return ModelInfo(
            slug=slug,
            supported_parameters=frozenset(entry["supported_parameters"]),
            accepts_images="image" in entry["architecture"]["input_modalities"],
            prompt_price=Decimal(entry["pricing"]["prompt"]),
            completion_price=Decimal(entry["pricing"]["completion"]),
            providers=self._retaining_nothing(slug) if strict_privacy else frozenset(),
        )

    def _retaining_nothing(self, slug: str) -> frozenset[str]:
        """Endpoint slugs of `slug` that keep nothing and accept a JSON schema.

        An endpoint missing from the ZDR list retains data, and `zdr: true` keeps
        requests off it. A slug that would also match an endpoint that may retain
        is left out.
        """
        keeping_nothing, retaining = _split(self.zdr_endpoints, slug)
        return frozenset(
            tag
            for tag, endpoint in keeping_nothing.items()
            if "structured_outputs" in endpoint.get("supported_parameters", ())
            and not any(_matches(tag, other) for other in retaining)
        )


def _split(
    zdr_endpoints: Sequence[Mapping[str, Any]], slug: str
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    """`slug`'s endpoints in the ZDR list that keep nothing, by tag, and the tags of
    those that may retain: anything but an explicit `safety_retention: false`.
    An entry with no tag is skipped, since no slug in `only` can be told from it."""
    tagged = [e for e in zdr_endpoints if e.get("model_id") == slug and e.get("tag")]
    keeping_nothing = {
        e["tag"]: e for e in tagged if e.get("safety_retention") is False
    }
    retaining = [e["tag"] for e in tagged if e.get("safety_retention") is not False]
    return keeping_nothing, retaining


def _matches(slug: str, tag: str) -> bool:
    """Whether `slug` in `only` admits endpoint `tag`: it matches its own endpoint
    and every one below it (`azure` matches `azure/eu`)."""
    return f"{tag}/".startswith(f"{slug}/")


async def fetch_catalog(
    api_key: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> Catalog:
    """The models and the ZDR endpoints OpenRouter lists now. Listing is free. No
    answer, or one of another shape, is a ModelError, which every caller says."""
    try:
        async with _client(api_key, transport) as http:
            models = await _data(http, "/models")
            zdr_endpoints = await _data(http, "/endpoints/zdr")
        return Catalog({model["id"]: model for model in models}, zdr_endpoints)
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise ModelError(
            f"OpenRouter's list of models could not be fetched"
            f" ({type(error).__name__}). Check the network connection and run the"
            " same command again."
        ) from error


async def recheck_providers(
    models: Mapping[str, ModelInfo],
    api_key: str,
    *,
    strict_privacy: bool,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, list[str]]:
    """The problems of each role's pinned `providers` against OpenRouter's ZDR list
    now, `[]` for a role that has none. Run it before student work is sent.

    It only verifies: nothing is resolved again and the lock is not touched. A
    failed fetch is a problem for every role. Without strict privacy it does
    nothing.
    """
    if not strict_privacy:
        return {role: [] for role in models}
    try:
        async with _client(api_key, transport) as http:
            zdr_endpoints = await _data(http, "/endpoints/zdr")
    except Exception as error:  # fail closed: unverified is not safe
        reason = str(error) or type(error).__name__
        return {
            role: [f"{role}: cannot verify the endpoints of {model.slug} ({reason})"]
            for role, model in models.items()
        }
    problems: dict[str, list[str]] = {}
    for role, model in models.items():
        keeping_nothing, retaining = _split(zdr_endpoints, model.slug)
        about = f"{role}: {model.slug} pinned endpoint"
        found = [] if model.providers else [f"{about}s: none, so nothing is sent"]
        for slug in sorted(model.providers):
            if hit := next((tag for tag in retaining if _matches(slug, tag)), None):
                found.append(f"{about} {slug} now admits {hit}, which may retain data")
            elif slug not in keeping_nothing:
                found.append(f"{about} {slug} is no longer zero data retention")
        problems[role] = found
    return problems


def _client(
    api_key: str, transport: httpx.AsyncBaseTransport | None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=API_URL,
        headers=authorization(api_key),
        timeout=60,
        transport=transport,
    )


async def _data(http: httpx.AsyncClient, path: str) -> list[dict[str, Any]]:
    reply = await http.get(path)
    if reply.status_code != 200:
        raise ModelError(f"OpenRouter {path}: HTTP {reply.status_code}")
    return reply.json()["data"]


def check(
    catalog: Catalog, names: Mapping[str, str], *, strict_privacy: bool
) -> dict[str, list[str]]:
    """The problems of the model behind each role, `[]` for a role that has none.

    Every role in design 10.1 is sent images: the only text-only stage, the class
    audit, has no role of its own.
    """
    problems: dict[str, list[str]] = {}
    for role, name in names.items():
        try:
            model = catalog.resolve(name, strict_privacy=strict_privacy)
        except ModelError as error:
            problems[role] = [f"{role}: {error}"]
            continue
        about = f"{role}: {name}" + (f" ({model.slug})" if model.slug != name else "")
        failures = [
            (not model.accepts_images, "does not accept image input"),
            (
                "structured_outputs" not in model.supported_parameters,
                "does not support structured outputs",
            ),
            (
                strict_privacy and not model.providers,
                "has no endpoint that retains nothing (zero data retention without"
                " Safety Retention) and accepts a JSON schema, and strict_privacy"
                " is on",
            ),
        ]
        problems[role] = [f"{about} {reason}" for failed, reason in failures if failed]
    return problems
