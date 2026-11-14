"""Global settings from `redpen.toml`, the OpenRouter API key, and the lock that
pins an assignment's settings in its workspace."""

import json
import os
import tomllib
from pathlib import Path
from typing import Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from redpen.llm.models import ModelError, ModelInfo, Secret, check, fetch_catalog
from redpen.workspace import Workspace

Profile = Literal["economy", "balanced", "thorough"]


class ConfigError(Exception):
    """A setting is missing or invalid. The message is meant for the user."""


class Models(BaseModel):
    """The model behind each role: an OpenRouter slug or `~author/family-latest`."""

    model_config = ConfigDict(extra="forbid")

    parser: str = "~anthropic/claude-sonnet-latest"
    solver: str = "~anthropic/claude-opus-latest"
    checker: str = "~openai/gpt-sol-latest"
    mapper: str = "~anthropic/claude-haiku-latest"
    reader: str = "~anthropic/claude-sonnet-latest"
    grader: str = "~anthropic/claude-sonnet-latest"
    second: str = "~openai/gpt-sol-latest"
    referee: str = "~anthropic/claude-opus-latest"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: Models = Field(default_factory=Models)
    profile: Profile = "balanced"
    image_budget: PositiveInt = 1536
    """Longest image edge, in pixels, sent to a model."""
    strict_privacy: bool = True
    """Route only to endpoints that retain nothing: zero data retention, no Safety
    Retention, no data collection."""
    mask_names: bool = False
    """Ask the mapper where a student's name is on each page, and blank it in every
    image of the page sent after the map (design 14.1)."""


def load_config(path: Path = Path("redpen.toml")) -> Config:
    """Load settings from `path`, or the defaults when the file does not exist."""
    if not path.exists():
        return Config()
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"{path}: {problems}") from exc


def openrouter_api_key() -> str:
    """Return the OpenRouter API key from the environment, as a `Secret`: its repr
    is a mask. No plain copy is bound to a name here."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Export your OpenRouter API key in it."
        )
    return Secret(os.environ["OPENROUTER_API_KEY"])


class Lock(BaseModel):
    """What an assignment is pinned to: `config.lock.toml` in its workspace."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: Profile
    models: dict[str, ModelInfo]
    """The concrete model behind each role."""
    prompts: dict[str, str] = Field(default_factory=dict[str, str])
    """The version of every prompt file at the latest run, by `<stage>/<name>`."""
    strict_privacy: bool | None = None
    """False once a grade run of this workspace sent pages with strict privacy off,
    and False from then on; True when every grade run had it on; None before any."""
    mask_names: bool | None = None
    """True when the latest grade run masked names; None otherwise."""


async def pin(
    workspace: Workspace,
    config: Config,
    api_key: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Lock:
    """The workspace's lock, as it is if it exists. Otherwise every role is resolved
    to a concrete model, which must pass `check`, and the lock is written."""
    path = workspace.config_lock_toml
    if path.exists():
        return read_lock(path)
    catalog = await fetch_catalog(api_key, transport=transport)
    names = config.models.model_dump()
    strict = config.strict_privacy
    problems = check(catalog, names, strict_privacy=strict)
    if any(problems.values()):
        raise ModelError("; ".join(p for found in problems.values() for p in found))
    lock = Lock(
        profile=config.profile,
        models={
            role: catalog.resolve(name, strict_privacy=strict)
            for role, name in names.items()
        },
    )
    write_lock(path, lock)
    return lock


def write_lock(path: Path, lock: Lock) -> None:
    # JSON's strings, booleans and arrays of strings are also TOML, and so is a
    # JSON string as a key.
    lines = [f"profile = {json.dumps(lock.profile)}"]
    if lock.strict_privacy is not None:
        lines.append(f"strict_privacy = {json.dumps(lock.strict_privacy)}")
    if lock.mask_names is not None:
        lines.append(f"mask_names = {json.dumps(lock.mask_names)}")
    for role, model in lock.models.items():
        lines += ["", f"[models.{role}]"]
        # The lists are sets: sorted, so that the same lock is the same file.
        for k, v in model.model_dump(mode="json").items():
            value = sorted(cast(list[str], v)) if isinstance(v, list) else v
            lines.append(f"{k} = {json.dumps(value)}")
    if lock.prompts:
        lines += ["", "[prompts]"]
        lines += [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in lock.prompts.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")  # whole or not at all
    partial.write_text("\n".join(lines) + "\n")
    partial.replace(path)


def read_lock(path: Path) -> Lock:
    with path.open("rb") as file:
        return Lock.model_validate(tomllib.load(file))
