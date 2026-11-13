"""Preparing the fixture blank's key with recorded calls to the real solver, checker,
referee and grader models.

The run starts from the fixtures' spec.json, not from a recorded parse, so that
recording the parse again never orphans these recordings. The recording is a dump
of the calls a live `redpen prepare` made in a scratch workspace (about $0.62).
"""

import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pint
import pytest
from sympy import Symbol
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Lock, write_lock
from redpen.llm.client import Gateway
from redpen.prepare.key import load_key, part_hash, save_key
from redpen.prepare.verify import Trial, Verification
from redpen.schemas.common import DiagramElement
from redpen.schemas.key import Answer, Key
from redpen.schemas.spec import CHOICE_TYPES
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES
from tests.fixture_truth import answer as filled
from tests.pinned import KEY_MODELS

TRUTH = Key.model_validate_json((FIXTURES / "key.json").read_text())
UNITS = pint.UnitRegistry()
X = Symbol("x")


def same_number(found: Answer, truth: Answer, rel_tol: float) -> bool:
    """Whether `found` is `truth`'s value within `rel_tol`, once in its unit."""
    assert found.value is not None and truth.value is not None
    value = UNITS.Quantity(found.value, found.unit or "")
    target = truth.value * UNITS(truth.unit or "")
    return abs(value.to(target.units).magnitude - target.magnitude) <= rel_tol * abs(
        target.magnitude
    )


def same_expression(found: str, truth: str) -> bool:
    """Whether two expressions in x agree at a few fixed points."""
    transformations = (
        *standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    a, b = (parse_expr(e, transformations=transformations) for e in (found, truth))
    return all(abs(float((a - b).subs(X, x))) < 1e-9 for x in (-2.5, 0, 1.5, 7))


def tokens(text: str | None) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", text or ""))


def stems(text: str | None) -> set[str]:
    return {word[:4] for word in re.findall(r"[a-z]+", (text or "").lower())}


def numbers(text: str | None) -> list[float]:
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text or "")]


def present(truth: DiagramElement, found: list[DiagramElement]) -> bool:
    """Loosely, whether `found` has `truth`'s element: one of the same kind whose
    direction carries the truth's words, and that is at the truth's coordinates
    (a point), or is a curve (named by its shape), or carries the truth's label."""
    for element in found:
        if element.kind.lower() != truth.kind.lower():
            continue
        if not stems(truth.direction) <= stems(element.direction):
            continue
        if numbers(truth.position):
            if numbers(element.position) == numbers(truth.position):
                return True
        elif truth.kind == "curve" or tokens(truth.label) <= tokens(element.label):
            return True
    return False


def assert_answers_match_the_truth(key: Key) -> None:
    assert [p.id for p in key.parts] == [t.id for t in TRUTH.parts]
    for found, truth in zip(key.parts, TRUTH.parts, strict=True):
        a, t = found.answer, truth.answer
        match truth.response_type:
            case "multiple_choice":
                assert a.choices == t.choices, found.id
            case "numeric":
                assert same_number(a, t, t.rel_tol or key.policies.rel_tol), found.id
            case "symbolic":
                assert a.expression and t.expression, found.id
                assert same_expression(a.expression, t.expression), found.id
            case "free_body_diagram" | "graph":
                missing = [e for e in t.elements if not present(e, a.elements)]
                assert not missing, (found.id, missing, a.elements)
            case "explanation":
                assert a.must_include, found.id
            case other:
                raise AssertionError(f"no comparison for {other}")


def test_the_loose_comparisons_tell_a_wrong_answer_from_a_right_one() -> None:
    def answer(**fields: Any) -> Answer:
        return filled(**fields)

    force = DiagramElement(label="F_N", kind="force", direction="up", position=None)
    assert same_number(
        answer(value=250, unit="cm/s^2"), answer(value=2.5, unit="m/s^2"), 0.02
    )
    assert not same_number(
        answer(value=2.6, unit="m/s^2"), answer(value=2.5, unit="m/s^2"), 0.02
    )
    with pytest.raises(pint.DimensionalityError):
        same_number(
            answer(value=2.5, unit="m/s"), answer(value=2.5, unit="m/s^2"), 0.02
        )
    assert same_expression("x^2 - 6x + 5", "(x - 3)^2 - 4")
    assert not same_expression("(x - 3)^2 + 4", "(x - 3)^2 - 4")
    assert present(
        DiagramElement(label="N", kind="force", direction="up", position=None), [force]
    )
    assert not present(
        DiagramElement(label="N", kind="force", direction="down", position=None),
        [force],
    )
    vertex = DiagramElement(
        label="vertex", kind="point", direction=None, position="(3, -4)"
    )
    wrong = vertex.model_copy(update={"position": "(3, 4)"})
    assert not present(vertex, [wrong])


runner = CliRunner()


def test_the_fixture_key_is_prepared_from_its_spec_and_matches_the_truth(
    replay: Callable[[str], Gateway],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    if not os.environ.get("OPENROUTER_API_KEY"):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("redpen.cli.Gateway", lambda *a, **k: replay("prepare_blank"))
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    workspace = Workspace(Path("ws"))
    workspace.spec_json.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "spec.json", workspace.spec_json)
    write_lock(workspace.config_lock_toml, Lock(profile="balanced", models=KEY_MODELS))

    prepared = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])
    checked = runner.invoke(app, ["key", "check", "--workspace", "ws"])

    assert prepared.exit_code == 0, prepared.output
    key = load_key(workspace.key_yaml)
    assert_answers_match_the_truth(key)
    for part in key.parts:
        assert part.criteria, part.id
        if part.response_type in CHOICE_TYPES:
            assert len(part.criteria) == 1
    assert checked.exit_code == 0, checked.output
    assert workspace.solutions_md.exists() and workspace.rubric_md.exists()
    # The two solvers agree on every part: in code on six, by the referee on the
    # drawing, the explanation and the graph.
    assert {p.state for p in key.parts} == {"verified"}
    assert "States: 9 verified, 0 refereed, 0 unverified" in prepared.output
    report = workspace.verification_md.read_text()
    assert report.count("Referee: the two agree in substance") == 3
    assert_every_rubric_passes(workspace, prepared.output)


def assert_every_rubric_passes(workspace: Workspace, output: str) -> None:
    """The key's own solution earned every criterion on the first rubric, and so
    did the second solution, of every part a model grades."""
    record = read_model(workspace.verification_json, Verification)
    passed = Trial(outcome="passed", lost=[])
    for v in record.parts:
        free = v.part_id.startswith(("Q4", "Q5"))
        assert v.tried == ([passed] if free else []), v.part_id
        assert v.second_tried == (passed if free else None), v.part_id
        assert v.key_check is None or v.key_check.verdict == "matches", v.part_id
    assert (
        "Rubric self-test: 6 passed, 0 failed, 0 not graded; 0 with the rubric"
        " written a second time." in output.splitlines()
    )


def test_key_verify_of_the_truth_key_moves_no_part_hash(
    replay: Callable[[str], Gateway],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    if not os.environ.get("OPENROUTER_API_KEY"):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("redpen.cli.Gateway", lambda *a, **k: replay("key_verify"))
    workspace = Workspace(Path("ws"))
    workspace.spec_json.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "blank.pdf", workspace.blank_pdf)
    shutil.copy(FIXTURES / "spec.json", workspace.spec_json)
    write_lock(workspace.config_lock_toml, Lock(profile="balanced", models=KEY_MODELS))
    # As a key solved once was written: every part unverified.
    once = [p.model_copy(update={"state": "unverified"}) for p in TRUTH.parts]
    save_key(workspace.key_yaml, TRUTH.model_copy(update={"parts": once}))

    result = runner.invoke(app, ["key", "verify", "--workspace", "ws"])

    assert result.exit_code == 0, result.output
    key = load_key(workspace.key_yaml)
    assert {p.id: part_hash(p, key.policies) for p in key.parts} == {
        p.id: part_hash(p, TRUTH.policies) for p in TRUTH.parts
    }
    # The checker agrees with the truth on every part, the referee on Q4 and Q5,
    # and only the states changed.
    assert {p.state for p in TRUTH.parts} == {"verified"} and key == TRUTH
    assert (
        result.output.splitlines()[0] == "States: 9 verified, 0 refereed, 0 unverified"
    )
    assert workspace.verification_md.read_text().count("Referee: not asked") == 6
    assert_every_rubric_passes(workspace, result.output)
