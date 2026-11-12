import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from redpen.config import Config, Models, pin, read_lock
from redpen.llm.client import FatalError, Gateway
from redpen.llm.models import Catalog, ModelError, ModelInfo, check, fetch_catalog
from redpen.workspace import Workspace


def model(
    slug: str,
    inputs: tuple[str, ...] = ("text", "image"),
    params: tuple[str, ...] = ("max_tokens", "structured_outputs"),
    alias_of: str | None = None,
) -> dict[str, Any]:
    """An entry of the models API, as `GET /api/v1/models` lists it."""
    entry: dict[str, Any] = {
        "id": slug,
        "canonical_slug": f"{slug}-20261001",
        "architecture": {"input_modalities": list(inputs)},
        "supported_parameters": list(params),
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
    }
    if alias_of:
        entry["alias_target"] = {"name": "Vendor: Vision", "slug": alias_of}
    return entry


def endpoint(
    slug: str,
    tag: str,
    safety_retention: bool | None = False,
    params: tuple[str, ...] = ("max_tokens", "structured_outputs"),
) -> dict[str, Any]:
    """An entry of `GET /api/v1/endpoints/zdr`."""
    return {
        "model_id": slug,
        "tag": tag,
        "safety_retention": safety_retention,
        "supported_parameters": list(params),
    }


def without(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if k != key}


MODELS = [
    model("~vendor/vision-latest", alias_of="vendor/vision-2"),
    model("vendor/vision-2"),
    model("vendor/text-1", inputs=("text",)),
    model("vendor/loose-1", params=("max_tokens",)),
    model("vendor/kept-1"),
    model("vendor/null-1"),
    model("vendor/unflagged-1"),
]
# The provider "retainer" also serves vendor/vision-2. It keeps data, so it is not
# in the ZDR list at all.
ZDR_ENDPOINTS = [
    endpoint("vendor/vision-2", "zero"),
    endpoint("vendor/vision-2", "safety", safety_retention=True),
    endpoint("vendor/vision-2", "cloud"),  # in `only` it would also allow cloud/eu
    endpoint("vendor/vision-2", "cloud/eu", safety_retention=True),
    endpoint("vendor/vision-2", "nojson", params=("max_tokens",)),
    endpoint("vendor/text-1", "zero"),
    endpoint("vendor/loose-1", "zero"),
    endpoint("vendor/kept-1", "safety", safety_retention=True),
    # Anything but an explicit false may retain: a null flag, or none at all.
    endpoint("vendor/null-1", "zero", safety_retention=None),
    without("safety_retention", endpoint("vendor/unflagged-1", "zero")),
    # An entry with no tag cannot be named in `only`, so it is skipped.
    without("tag", endpoint("vendor/vision-2", "zero")),
]


class OpenRouter:
    """Lists MODELS and ZDR_ENDPOINTS, and answers a completion with each reply in
    turn, the last repeating, served by `served_by` or the model asked for."""

    def __init__(self, *replies: str, served_by: str | None = None) -> None:
        self.replies = list(replies) or ['{"letter": "B"}']
        self.served_by = served_by
        self.paths: list[str] = []
        self.completions: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": MODELS})
        if request.url.path == "/api/v1/endpoints/zdr":
            return httpx.Response(200, json={"data": ZDR_ENDPOINTS})
        body = json.loads(request.content)
        self.completions.append(body)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return httpx.Response(
            200,
            json={
                "model": self.served_by or body["model"],
                "choices": [{"message": {"content": reply}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001},
            },
        )


def catalog() -> Catalog:
    transport = httpx.MockTransport(OpenRouter())
    return asyncio.run(fetch_catalog("test-key", transport=transport))


def resolve(name: str, *, strict_privacy: bool = True) -> ModelInfo:
    return catalog().resolve(name, strict_privacy=strict_privacy)


def test_an_alias_resolves_to_its_concrete_model() -> None:
    assert resolve("~vendor/vision-latest", strict_privacy=False) == ModelInfo(
        slug="vendor/vision-2",  # the models API `id`, not the dated canonical_slug
        supported_parameters=frozenset({"max_tokens", "structured_outputs"}),
        accepts_images=True,
        prompt_price=Decimal("0.000001"),
        completion_price=Decimal("0.000002"),
    )


def test_only_endpoints_that_retain_nothing_are_pinned() -> None:
    assert resolve("~vendor/vision-latest").providers == {"zero"}


@pytest.mark.parametrize("slug", ["vendor/null-1", "vendor/unflagged-1"])
def test_a_null_or_missing_retention_flag_counts_as_retaining(slug: str) -> None:
    assert resolve(slug).providers == frozenset()


def test_relaxed_privacy_pins_no_providers_and_needs_none() -> None:
    assert resolve("vendor/kept-1", strict_privacy=False).providers == frozenset()
    assert check(catalog(), {"reader": "vendor/kept-1"}, strict_privacy=False) == {
        "reader": []
    }


@pytest.mark.parametrize(
    ("role", "name", "problem"),
    [
        ("solver", "~vendor/vision-latest", None),
        (
            "reader",
            "vendor/text-1",
            "reader: vendor/text-1 does not accept image input",
        ),
        (
            "grader",
            "vendor/loose-1",
            "grader: vendor/loose-1 does not support structured outputs",
        ),
        (
            "mapper",
            "vendor/kept-1",
            "mapper: vendor/kept-1 has no endpoint that retains nothing (zero data"
            " retention without Safety Retention) and accepts a JSON schema, and"
            " strict_privacy is on",
        ),
        (
            "parser",
            "~vendor/gone-latest",
            "parser: OpenRouter lists no concrete model for ~vendor/gone-latest",
        ),
    ],
    ids=["passes", "text only", "no structured outputs", "retains", "unknown"],
)
def test_check_names_the_role_the_model_and_the_reason(
    role: str, name: str, problem: str | None
) -> None:
    assert check(catalog(), {role: name}, strict_privacy=True) == {
        role: [problem] if problem else []
    }


class Letter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    letter: str


def ask(gateway: Gateway, model: ModelInfo) -> Any:
    return gateway.call("reader", model, "system", [], ["item"], Letter, "v1")


def test_every_request_goes_only_to_the_pinned_providers(tmp_path: Path) -> None:
    pinned = resolve("~vendor/vision-latest")
    server = OpenRouter("{}", '{"letter": "B"}')  # the first needs the repair

    async def main() -> None:
        transport = httpx.MockTransport(server)
        async with Gateway(tmp_path / "c.sqlite", "k", transport=transport) as gateway:
            await ask(gateway, pinned)

    asyncio.run(main())

    assert [body["provider"] for body in server.completions] == 2 * [
        {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "only": ["zero"],
        }
    ]


def test_a_response_from_another_slug_than_the_pinned_one_raises(
    tmp_path: Path,
) -> None:
    pinned = resolve("~vendor/vision-latest")
    server = OpenRouter(served_by="vendor/vision-3")

    async def main() -> None:
        transport = httpx.MockTransport(server)
        async with Gateway(tmp_path / "c.sqlite", "k", transport=transport) as gateway:
            await ask(gateway, pinned)

    with pytest.raises(FatalError, match="asked for vendor/vision-2, served by"):
        asyncio.run(main())


def config_with(**roles: str) -> Config:
    names = dict.fromkeys(Models.model_fields, "~vendor/vision-latest")
    return Config(models=Models(**names | roles))


def test_pin_writes_a_lock_that_is_then_read_back_as_it_is(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws")
    server = OpenRouter()

    lock = asyncio.run(
        pin(workspace, config_with(), "k", transport=httpx.MockTransport(server))
    )
    untouched = OpenRouter()
    again = asyncio.run(
        pin(
            workspace,
            config_with(reader="vendor/text-1"),
            "k",
            transport=httpx.MockTransport(untouched),
        )
    )

    assert lock.profile == "balanced"
    assert set(lock.models) == set(Models.model_fields)
    assert lock.models["reader"] == resolve("~vendor/vision-latest")
    assert read_lock(workspace.config_lock_toml) == lock
    assert again == lock
    assert untouched.paths == []


def test_pin_refuses_a_model_that_fails_the_check(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws")
    transport = httpx.MockTransport(OpenRouter())

    with pytest.raises(ModelError, match="reader: vendor/text-1 does not accept"):
        asyncio.run(
            pin(
                workspace, config_with(reader="vendor/text-1"), "k", transport=transport
            )
        )
    assert not workspace.config_lock_toml.exists()
