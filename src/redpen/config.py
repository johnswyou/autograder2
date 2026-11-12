"""Global settings from `redpen.toml` and the OpenRouter API key."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

Profile = Literal["economy", "balanced", "thorough"]


class ConfigError(Exception):
    """A setting is missing or invalid. The message is meant for the user."""


class Models(BaseModel):
    """The model behind each role: an OpenRouter slug or `~author/family-latest`."""

    model_config = ConfigDict(extra="forbid")

    parser: str = "~anthropic/claude-sonnet-latest"
    solver: str = "~anthropic/claude-opus-latest"
    checker: str = "~google/gemini-pro-latest"
    mapper: str = "~google/gemini-flash-latest"
    reader: str = "~google/gemini-pro-latest"
    grader: str = "~anthropic/claude-sonnet-latest"
    second: str = "~openai/gpt-latest"
    referee: str = "~anthropic/claude-opus-latest"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: Models = Field(default_factory=Models)
    profile: Profile = "balanced"
    image_budget: PositiveInt = 1536
    """Longest image edge, in pixels, sent to a model."""
    strict_privacy: bool = True
    """Route only to endpoints with zero data retention and no data collection."""


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
    """Return the OpenRouter API key from the environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Export your OpenRouter API key in it."
        )
    return key
