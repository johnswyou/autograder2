import re
import tomllib
from decimal import Decimal
from pathlib import Path

import pytest

from autograder2.config import (
    Config,
    ConfigError,
    Lock,
    load_config,
    openrouter_api_key,
    read_lock,
    write_lock,
)
from autograder2.llm.models import ModelInfo


def test_defaults_load_with_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.models.model_dump() == {
        "parser": "~anthropic/claude-sonnet-latest",
        "solver": "~anthropic/claude-opus-latest",
        "checker": "~openai/gpt-sol-latest",
        "mapper": "~anthropic/claude-haiku-latest",
        "reader": "~anthropic/claude-sonnet-latest",
        "grader": "~anthropic/claude-sonnet-latest",
        "second": "~openai/gpt-sol-latest",
        "referee": "~anthropic/claude-opus-latest",
    }
    assert config.profile == "balanced"
    assert config.image_budget == 1536
    assert config.strict_privacy is True
    assert config.mask_names is False


def test_masking_names_is_switched_on_in_the_file(tmp_path: Path) -> None:
    path = tmp_path / "autograder2.toml"
    path.write_text("mask_names = true\n")

    assert load_config(path).mask_names is True


def test_file_overrides_a_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("autograder2.toml").write_text('[models]\nreader = "vendor/model-1"\n')

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
        (
            "image_budget = 0\n",
            "image_budget: Input should be greater than or equal to 256",
        ),
        (
            "image_budget = 15360\n",
            "image_budget: Input should be less than or equal to 4096",
        ),
        ("image_budget = 2\n", "image_budget: Input should be greater than or equal"),
        ("image_budget = true\n", "image_budget: Input should be greater than or"),
        ("profile = \n", "autograder2.toml: Invalid value"),
    ],
)
def test_invalid_file_is_rejected_with_a_clear_message(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "autograder2.toml"
    path.write_text(content)

    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(path)


@pytest.mark.parametrize(
    ("kind", "error"),
    [("latin-1", "UnicodeDecodeError"), ("a folder", "IsADirectoryError")],
)
def test_a_file_that_is_not_utf_8_text_is_rejected_in_one_line(
    tmp_path: Path, kind: str, error: str
) -> None:
    path = tmp_path / "autograder2.toml"
    if kind == "a folder":
        path.mkdir()
    else:  # an accent in a comment, saved as latin-1
        path.write_bytes('# caf\u00e9\nprofile = "balanced"\n'.encode("latin-1"))

    with pytest.raises(ConfigError) as raised:
        load_config(path)

    assert str(raised.value) == (
        f"{path}: cannot be read as UTF-8 text ({error}). Save it as UTF-8 text, or"
        " move it away to use the defaults."
    )


@pytest.mark.parametrize("budget", [256, 1536, 4096])
def test_an_image_budget_in_its_bounds_is_taken(tmp_path: Path, budget: int) -> None:
    path = tmp_path / "autograder2.toml"
    path.write_text(f"image_budget = {budget}\n")

    assert load_config(path).image_budget == budget


def test_example_file_shows_every_setting_with_its_default() -> None:
    example = Path(__file__).parents[2] / "autograder2.toml.example"

    assert tomllib.loads(example.read_text()) == Config().model_dump()


def test_api_key_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    assert openrouter_api_key() == "sk-test"


def test_missing_api_key_raises_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY is not set"):
        openrouter_api_key()


def test_the_lock_is_written_with_its_lists_sorted_and_the_same_every_time(
    tmp_path: Path,
) -> None:
    model = ModelInfo(
        slug="vendor/model-1",
        supported_parameters=frozenset({"tools", "max_tokens", "response_format"}),
        accepts_images=True,
        prompt_price=Decimal("0.000002"),
        completion_price=Decimal("0.00001"),
        providers=frozenset({"zeta/us", "alpha", "mid/eu"}),
    )
    lock = Lock(profile="balanced", models={"reader": model}, prompts={"a/b": "1"})
    first, second = tmp_path / "first.toml", tmp_path / "second.toml"

    write_lock(first, lock)
    write_lock(second, Lock.model_validate(lock.model_dump()))

    assert first.read_bytes() == second.read_bytes()
    lines = first.read_text().splitlines()
    assert 'providers = ["alpha", "mid/eu", "zeta/us"]' in lines
    assert 'supported_parameters = ["max_tokens", "response_format", "tools"]' in lines
    assert read_lock(first) == lock


def test_the_lock_records_names_masked_and_otherwise_holds_no_such_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.lock.toml"
    model = ModelInfo(
        slug="vendor/model-1",
        supported_parameters=frozenset(),
        accepts_images=True,
        prompt_price=Decimal(0),
        completion_price=Decimal(0),
        providers=frozenset({"zero"}),
    )
    write_lock(path, Lock(profile="balanced", models={"mapper": model}))
    plain = path.read_text()

    write_lock(
        path, Lock(profile="balanced", models={"mapper": model}, mask_names=True)
    )

    assert "mask_names" not in plain and read_lock(path).mask_names is True
    assert path.read_text() == plain.replace("\n", "\nmask_names = true\n", 1)
