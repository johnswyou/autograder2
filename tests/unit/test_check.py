"""The answer check in code (src/redpen/grade/check.py): the table of pairs, hostile
input, and the property that an expression is equivalent to itself."""

import time
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from redpen.grade import check
from redpen.grade.check import MATCH, NO_MATCH, UNSURE, check_answer
from redpen.schemas.key import Key, KeyPart, Policies
from redpen.schemas.reading import FinalAnswer, Reading
from tests.fixture_truth import FIXTURES, answer

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
PARTS = {p.id: p for p in KEY.parts}
ON = Policies()  # 2% relative, units required, significant figures not penalised
OFF = Policies(require_units=False)
FIGURES = Policies(penalise_sig_figs=True)
FAST = 0.25  # seconds: "well under a second", with room for a slow machine


def numeric(**fields: Any) -> KeyPart:
    return PARTS["Q4.b"].model_copy(update={"answer": answer(**fields)})


def symbolic(expression: str) -> KeyPart:
    return PARTS["Q5.a"].model_copy(update={"answer": answer(expression=expression)})


def stated(**fields: Any) -> Reading:
    final = {"choices": [], "value": None, "unit": None, "expression": None} | fields
    return Reading(
        part_id="Q4.b",
        status="answered",
        medium="handwritten",
        lines=["a line"],
        final_answer=FinalAnswer.model_validate(final),
        elements=[],
        uncertain=[],
        addressed_to_grader=[],
        legibility="clear",
    )


ACCEL = {"value": 2.5, "unit": "m/s^2"}
# (case, the key's answer, the stated value, the stated unit, policies, verdict)
NUMBERS: list[tuple[str, dict[str, Any], float | None, str | None, Policies, str]] = [
    ("the key's value", ACCEL, 2.5, "m/s^2", ON, MATCH),
    ("within tolerance", ACCEL, 2.54, "m/s^2", ON, MATCH),
    ("just outside tolerance", ACCEL, 2.56, "m/s^2", ON, NO_MATCH),
    ("far outside", ACCEL, 5.0, "m/s^2", ON, NO_MATCH),
    ("cm for m", {"value": 2.5, "unit": "m"}, 250, "cm", ON, MATCH),
    ("m for cm", {"value": 250, "unit": "cm"}, 2.5, "m", ON, MATCH),
    ("cm for m, wrong value", {"value": 2.5, "unit": "m"}, 2.5, "cm", ON, NO_MATCH),
    ("km/h for m/s", {"value": 25, "unit": "m/s"}, 90, "km/h", ON, MATCH),
    ("m/s for km/h", {"value": 90, "unit": "km/h"}, 25, "m/s", ON, MATCH),
    ("g for kg", {"value": 0.5, "unit": "kg"}, 500, "g", ON, MATCH),
    ("kg*m/s^2 for N", {"value": 10, "unit": "N"}, 10, "kg*m/s^2", ON, MATCH),
    ("spaces in the unit", {"value": 10, "unit": "N"}, 10, "kg m / s^2", ON, MATCH),
    ("J/kg/K for J/(kg*K)", {"value": 4, "unit": "J/(kg*K)"}, 4, "J/kg/K", ON, MATCH),
    ("degC for K", {"value": 293.15, "unit": "K"}, 20, "degC", ON, MATCH),
    ("wrong dimension", ACCEL, 2.5, "m/s", ON, NO_MATCH),
    ("mass for force", {"value": 10, "unit": "N"}, 10, "kg", ON, NO_MATCH),
    ("a unit for a pure number", {"value": -4.0}, -4.0, "m", ON, NO_MATCH),
    ("no unit, units required", ACCEL, 2.5, None, ON, NO_MATCH),
    ("no unit, units not required", ACCEL, 2.5, None, OFF, MATCH),
    ("no unit, not required, wrong", ACCEL, 2.0, None, OFF, NO_MATCH),
    ("no unit for a pure number", {"value": -4.0}, -4.0, None, ON, MATCH),
    ("accepted alternative", {**ACCEL, "accept_also": [3.3]}, 3.3, "m/s^2", ON, MATCH),
    (
        "alternative in another unit",
        {**ACCEL, "accept_also": [3.3]},
        330,
        "cm/s^2",
        ON,
        MATCH,
    ),
    ("neither value", {**ACCEL, "accept_also": [3.3]}, 3.0, "m/s^2", ON, NO_MATCH),
    ("rounded to two figures", {"value": 14.7, "unit": "m"}, 15, "m", ON, MATCH),
    ("rounded, outside tolerance", {"value": 1.46, "unit": "m"}, 1.5, "m", ON, MATCH),
    ("rounded to one figure", {"value": 1.46, "unit": "m"}, 1, "m", ON, NO_MATCH),
    ("rounded in another unit", {"value": 1.46, "unit": "m"}, 150, "cm", ON, MATCH),
    (
        "rounded, figures penalised",
        {"value": 14.7, "unit": "m"},
        15,
        "m",
        FIGURES,
        NO_MATCH,
    ),
    ("rounded wrongly", {"value": 1.44, "unit": "m"}, 1.5, "m", ON, NO_MATCH),
    ("negative", {"value": -4.0}, -4.0, None, ON, MATCH),
    ("sign error", {"value": -4.0}, 4.0, None, ON, NO_MATCH),
    ("negative, rounded", {"value": -1.46, "unit": "m"}, -1.5, "m", ON, MATCH),
    ("zero key, small", {"value": 0.0, "unit": "m"}, 0.015, "m", ON, MATCH),
    ("zero key, too big", {"value": 0.0, "unit": "m"}, 0.05, "m", ON, NO_MATCH),
    ("zero key, another unit", {"value": 0.0, "unit": "m"}, 1, "cm", ON, MATCH),
    ("zero for zero", {"value": 0.0}, 0.0, None, ON, MATCH),
    ("scientific notation", {"value": 3.0e8, "unit": "m/s"}, 2.998e8, "m/s", ON, MATCH),
    ("small scientific", {"value": 6.67e-11}, 6.7e-11, None, ON, MATCH),
    ("wrong exponent", {"value": 6.67e-11}, 6.67e-10, None, ON, NO_MATCH),
    ("the part's own tolerance", {**ACCEL, "rel_tol": 0.1}, 2.7, "m/s^2", ON, MATCH),
    (
        "outside its own tolerance",
        {**ACCEL, "rel_tol": 0.1},
        2.8,
        "m/s^2",
        ON,
        NO_MATCH,
    ),
    ("no value stated", ACCEL, None, "m/s^2", ON, UNSURE),
    ("a unit Pint does not know", ACCEL, 2.5, "furlongz", ON, UNSURE),
    (
        "a unit grouped unclearly",
        {"value": 1, "unit": "kg/(m*s)"},
        1,
        "kg/m*s",
        ON,
        UNSURE,
    ),
    ("a unit written in symbols", ACCEL, 2.5, "m/s²", ON, UNSURE),
    ("a key unit it cannot read", {"value": 1, "unit": "m/s²"}, 1, "m/s^2", ON, UNSURE),
]

# (case, the key's expression, the stated expression, verdict)
EXPRESSIONS: list[tuple[str, str, str, str]] = [
    ("the key's expression", "(x - 3)^2 - 4", "(x - 3)^2 - 4", MATCH),
    ("terms reordered", "x^2 - 6*x + 5", "5 - 6*x + x^2", MATCH),
    ("expanded for factored", "(x - 3)^2 - 4", "x^2 - 6x + 5", MATCH),
    ("x*x for x^2", "x^2", "x*x", MATCH),
    ("a half written two ways", "1/2*m*v^2", "m*v^2/2", MATCH),
    ("a root as a power", "sqrt(x)", "x^(1/2)", MATCH),
    ("a double angle", "sin(2*theta)", "2*sin(theta)*cos(theta)", MATCH),
    ("a Pythagorean identity", "1 - cos(x)^2", "sin(x)^2", MATCH),
    ("a quotient of exponentials", "exp(-t)", "1/exp(t)", MATCH),
    ("side by side for times", "2*x*y", "2x y", MATCH),
    ("pi", "pi*r^2", "r*r*pi", MATCH),
    ("a leading name dropped", "sqrt(2*g*h)", "v = sqrt(2*g*h)", MATCH),
    ("inverse functions", "atan(x)", "arctan(x)", MATCH),
    ("a sign error", "(x - 3)^2 - 4", "(x - 3)^2 + 4", NO_MATCH),
    ("a missing factor", "1/2*m*v^2", "m*v^2", NO_MATCH),
    ("a wrong power", "m*v^2/2", "m*v^3/2", NO_MATCH),
    ("a different name", "m*g*h", "m*g*y", NO_MATCH),
    ("case alone", "m*g*h", "M*g*h", UNSURE),
    ("an underscore alone", "v0*t", "v_0*t", UNSURE),
    ("names side by side", "m*v", "mv", UNSURE),
    ("e for exp", "exp(x)", "e^x", UNSURE),
    ("unclear order after a division", "a/(2*b)", "a/2b", UNSURE),
    ("unclear order after a power", "x^(2*y)", "x^2y", UNSURE),
    ("log against ln", "log(x)", "ln(x)", UNSURE),
    ("log against log", "log(x)", "log(x)", MATCH),
    ("a function not on the list", "x", "f(x)", UNSURE),
    ("a key it cannot read", "x!", "x", UNSURE),
]


@pytest.fixture(autouse=True, scope="module")
def warm() -> None:
    check._registry()  # Pint's registry takes about half a second to build, once


@pytest.mark.parametrize(
    ("key", "value", "unit", "policies", "verdict"),
    [case[1:] for case in NUMBERS],
    ids=[case[0] for case in NUMBERS],
)
def test_a_stated_number_is_checked_against_the_key(
    key: dict[str, Any],
    value: float | None,
    unit: str | None,
    policies: Policies,
    verdict: str,
) -> None:
    found = check_answer(numeric(**key), stated(value=value, unit=unit), policies)

    assert found is not None and found.verdict == verdict


@pytest.mark.parametrize(
    ("key", "said", "verdict"),
    [case[1:] for case in EXPRESSIONS],
    ids=[case[0] for case in EXPRESSIONS],
)
def test_a_stated_expression_is_checked_against_the_key(
    key: str, said: str, verdict: str
) -> None:
    found = check_answer(symbolic(key), stated(expression=said), ON)

    assert found is not None and found.verdict == verdict


def test_the_table_has_forty_pairs_or_more() -> None:
    assert len(NUMBERS) + len(EXPRESSIONS) >= 40


@pytest.mark.parametrize("part", ["Q1", "Q4.a", "Q4.c", "Q5.c"])
def test_a_part_with_neither_value_nor_expression_has_no_check(part: str) -> None:
    assert check_answer(PARTS[part], stated(value=2.5, expression="x"), ON) is None


HOSTILE = [
    "__import__('os').system('x')",
    "().__class__",
    "a.b",
    "x[0]",
    "lambda: 1",
    "x__y + 1",
    "9^9^9^9",
    "10^(10^10)",
    "2^(3^(4^5))",
    "1^(10^10)",  # past the exponent limit, though it would evaluate to 1
    "(" * 10000 + "x" + ")" * 10000,
    "-" * 150 + "x",
    "x+" * 5000 + "x",
    "х",  # Cyrillic, looks like x
    "ｘ^2",  # full-width x
    "x + 1",  # a no-break space
    "",
    "   \n\t ",
    "'x'",
    "x if x else 1",
    "x < 1 < 2",
]


@pytest.mark.parametrize("said", HOSTILE, ids=range(len(HOSTILE)))
def test_hostile_expressions_are_cannot_tell_and_fast(said: str) -> None:
    start = time.perf_counter()
    found = check_answer(symbolic("x^2"), stated(expression=said), ON)
    took = time.perf_counter() - start

    assert found is not None and found.verdict == UNSURE
    assert took < FAST
    assert said.strip() == "" or said not in found.reason


class Spy:
    """Pint's registry, recording each unit name it is given."""

    def __init__(self) -> None:
        self.real = check._registry()
        self.names: list[str] = []

    def Unit(self, name: str) -> Any:  # noqa: N802 (Pint's name)
        self.names.append(name)
        return self.real.Unit(name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)


@pytest.mark.parametrize(
    "unit",
    [
        "m*(2+3)",
        "__import__('os').system('x')",
        "m.__class__",
        "m" * 5000,
        "m/" * 3000 + "s",
        "m^9^9^9",
        "s^99",
        "m s^-2; import os",
    ],
    ids=[
        "arithmetic",
        "a call",
        "an attribute",
        "long",
        "long",
        "powers",
        "big",
        "code",
    ],
)
def test_a_hostile_unit_never_reaches_pint(
    unit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = Spy()
    monkeypatch.setattr(check, "_registry", lambda: spy)
    start = time.perf_counter()

    found = check_answer(numeric(**ACCEL), stated(value=2.5, unit=unit), ON)

    assert time.perf_counter() - start < FAST
    assert found is not None and found.verdict == UNSURE
    assert spy.names == ["m", "s"]  # the key's unit only, one name at a time


# --- the property --------------------------------------------------------------------

LEAVES = st.sampled_from(["x", "y", "m", "1", "2", "3", "4"])


def grow(inner: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    return st.one_of(
        st.tuples(inner, st.sampled_from(["+", "-", "*"]), inner).map(
            lambda t: f"({t[0]} {t[1]} {t[2]})"
        ),
        st.tuples(inner, LEAVES).map(lambda t: f"({t[0]})/{t[1]}"),
        inner.map(lambda e: f"({e})^2"),
        st.tuples(st.sampled_from(["sin", "cos"]), inner).map(
            lambda t: f"{t[0]}({t[1]})"
        ),
    )


# Short, and squared at most twice, so that every value stays far from overflow and
# from where adding one is lost to rounding.
EXPRESSION = st.recursive(LEAVES, grow, max_leaves=4).filter(
    lambda e: len(e) <= 45 and e.count("^") <= 2
)


@given(st.lists(EXPRESSION, min_size=1, max_size=4))
def test_an_expression_is_equivalent_to_itself_reordered_and_not_plus_one(
    terms: list[str],
) -> None:
    whole = " + ".join(terms)
    key = symbolic(whole)

    def verdict(said: str) -> str:
        found = check_answer(key, stated(expression=said), ON)
        assert found is not None
        return found.verdict

    assert verdict(whole) == MATCH
    assert verdict(" + ".join(reversed(terms))) == MATCH
    assert verdict(f"{whole} + 1") == NO_MATCH
