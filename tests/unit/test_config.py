import re
import tomllib
from pathlib import Path

import pytest

from redpen.config import Config, ConfigError, load_config, openrouter_api_key


def test_defaults_load_with_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.models.model_dump() == {
        "parser": "~anthropic/claude-sonnet-latest",
        "solver": "~anthropic/claude-opus-latest",
        "checker": "~google/gemini-pro-latest",
        "mapper": "~google/gemini-flash-latest",
        "reader": "~google/gemini-pro-latest",
        "grader": "~anthropic/claude-sonnet-latest",
        "second": "~openai/gpt-latest",
        "referee": "~anthropic/claude-opus-latest",
    }
    assert config.profile == "balanced"
    assert config.image_budget == 1536
    assert config.strict_privacy is True


def test_file_overrides_a_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("redpen.toml").write_text('[models]\nreader = "vendor/model-1"\n')

    config = load_config()

    assert config.models.reader == "vendor/model-1"
    assert config.models.grader == "~anthropic/claude-sonnet-latest"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            '[models]\nreviewer = "x"\n',
            "models.reviewer: Extra inputs are not permitted",
        ),
        ("budget = 1\n", "budget: Extra inputs are not permitted"),
        (
            'profile = "fast"\n',
            "profile: Input should be 'economy', 'balanced' or 'thorough'",
        ),
        ("image_budget = 0\n", "image_budget: Input should be greater than 0"),
        ("profile = \n", "redpen.toml: Invalid value"),
    ],
)
def test_invalid_file_is_rejected_with_a_clear_message(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "redpen.toml"
    path.write_text(content)

    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(path)


def test_example_file_shows_every_setting_with_its_default() -> None:
    example = Path(__file__).parents[2] / "redpen.toml.example"

    assert tomllib.loads(example.read_text()) == Config().model_dump()


def test_api_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    assert openrouter_api_key() == "sk-test"


def test_missing_api_key_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY is not set"):
        openrouter_api_key()
