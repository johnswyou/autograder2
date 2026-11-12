import subprocess
import sys
from pathlib import Path

import pytest

from redpen import prompts

ROOT = Path(__file__).parents[2]


def test_the_version_changes_when_one_byte_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stage" / "name.md"
    path.parent.mkdir()
    monkeypatch.setattr(prompts, "files", lambda package: tmp_path)

    path.write_bytes(b"Read the page.\n")
    first = prompts.load("stage", "name")
    path.write_bytes(b"Read the page!\n")
    second = prompts.load("stage", "name")

    assert first[0] == "Read the page.\n" and second[0] == "Read the page!\n"
    assert first[1] != second[1]
    assert prompts.load("stage", "name") == second


def test_the_prompts_load_from_a_built_wheel(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )
    [wheel] = tmp_path.glob("*.whl")
    # -I and the wheel first on the path, so nothing is read from the source tree.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); from redpen import prompts;"
        "print(prompts.__file__);"
        "print(*(prompts.load('parse', name)[1] for name in ('page', 'stitch')))"
    )
    found = subprocess.run(
        [sys.executable, "-I", "-c", code, str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()

    assert found[0].startswith(str(wheel))
    assert found[1:] == [prompts.load("parse", name)[1] for name in ("page", "stitch")]
