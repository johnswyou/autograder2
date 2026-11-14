"""Paid: the grader's method credit on hand-built readings, two fresh draws of each
case. Run with `REDPEN_LIVE=1 uv run pytest -s tests/live/test_grade_live.py`
(about $0.50). It prints every award and asserts only what must never earn method
credit; the slip and other-route cases are printed, since single draws vary. This
is what a later change to the grade prompt is checked with.

The keys are the fixture key's problems 4 and 5 with method criteria split, as a
rubric writer giving each expected value would write them, built here."""

import asyncio
import copy
import json
import os
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from redpen.config import openrouter_api_key
from redpen.grade.grade import grade_student
from redpen.llm.client import Gateway
from redpen.schemas.grade import GradeRecord
from redpen.schemas.key import Key
from redpen.schemas.reading import FinalAnswer, ProblemReadings, Reading, ReadingRecord
from redpen.workspace import Workspace, write_model
from tests.fixture_truth import FIXTURES, NOBODY
from tests.pinned import GRADER

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("REDPEN_LIVE") != "1", reason="paid; set REDPEN_LIVE=1"
    ),
]

DRAWS = 2
FIXTURE_KEY: dict[str, Any] = json.loads((FIXTURES / "key.json").read_text())


def criterion(
    id: str, kind: str, req: str, partial: tuple[tuple[str, str], ...] = ()
) -> dict[str, Any]:
    return {
        "id": id,
        "points": "1",
        "kind": kind,
        "requirement": req,
        "partial": [{"points": p, "when": w} for p, w in partial],
        "carry_forward": False,
    }


def problem(prefix: str, **criteria: list[dict[str, Any]]) -> dict[str, Any]:
    """The fixture key cut to one problem, with the criteria of the parts named."""
    key = copy.deepcopy(FIXTURE_KEY)
    key["parts"] = [p for p in key["parts"] if p["id"].startswith(prefix)]
    for part in key["parts"]:
        if part["id"] in criteria:
            part["criteria"] = criteria[part["id"]]
    return key


def split4() -> dict[str, Any]:
    return problem(
        "Q4",
        **{
            "Q4.b": [
                criterion(
                    "Q4.b.c1",
                    "method",
                    "Finds the friction force f = mu m g = 0.25(4.0)(10) = 10 N.",
                ),
                criterion(
                    "Q4.b.c2",
                    "method",
                    "Applies Newton's second law with net force T - f = 20 - 10 ="
                    " 10 N to find a = F_net / m.",
                    (
                        (
                            "0.5",
                            "Applies Newton's second law but leaves friction out of"
                            " the net force (a = T/m).",
                        ),
                    ),
                ),
                criterion(
                    "Q4.b.c3", "final_answer", "States a = 2.5 m/s^2 with units."
                ),
            ],
            "Q4.c": [
                criterion(
                    "Q4.c.c1", "final_answer", "States that the crate slows down."
                ),
                criterion(
                    "Q4.c.c2",
                    "method",
                    "States that after the rope is cut, kinetic friction is the only"
                    " horizontal force on the crate.",
                ),
                criterion(
                    "Q4.c.c3",
                    "method",
                    "Explains that friction opposes the motion, so the net force"
                    " points opposite to the velocity.",
                ),
            ],
        },
    )


def split5(named: bool = False) -> dict[str, Any]:
    """Problem 5 with 5(b)'s method criterion split; with `named`, the question
    names the method (the derivative) and the first criterion follows it."""
    first = (
        "Sets f'(x) = 2x - 6 = 0 and solves it to locate the minimum at x = 3."
        if named
        else "Locates the minimum at x = 3 (from the vertex form, x = -b/2a or"
        " f'(x) = 0)."
    )
    key = problem(
        "Q5",
        **{
            "Q5.b": [
                criterion("Q5.b.c1", "method", first),
                criterion(
                    "Q5.b.c2",
                    "method",
                    "Evaluates f at that x: f(3) = 9 - 18 + 5 = -4.",
                ),
                criterion("Q5.b.c3", "final_answer", "States the minimum value -4."),
            ]
        },
    )
    [b] = [p for p in key["parts"] if p["id"] == "Q5.b"]
    for c in b["criteria"]:  # 5(b) depends on 5(a), as assembling marks it
        c["carry_forward"] = True
    if named:
        b["prompt"] = (
            "Let f(x) = x^2 - 6x + 5. Using the derivative f'(x), find the minimum"
            " value of f(x)."
        )
        b["solution"] = (
            "f'(x) = 2x - 6 = 0 gives x = 3, a minimum since f''(x) = 2 > 0, and"
            " f(3) = 9 - 18 + 5 = -4."
        )
    return key


def plain4() -> dict[str, Any]:
    return problem("Q4")


ANOTHER = ["a = T/m - mu g", "= 20/4.0 - 0.25(10)", "= 5.0 - 2.5"]
KEY_SLIP = [
    "f = mu m g = 0.25(4.0)(10) = 10 N",
    "F_net = T - f = 20 - 10 = 10 N",
    "a = F_net/m = 10/4.0 = 2.0 m/s^2",
]
DISCRIMINANT = [
    "Let m be a value of f: x^2 - 6x + 5 = m",
    "x^2 - 6x + (5 - m) = 0 has a real root only if its discriminant is >= 0",
    "36 - 4(5 - m) >= 0",
]
NO_REASON = ["The crate slows down and stops."]

# A part's response: its lines, its stated value and unit, and the numbers of the
# lines addressed to the grader. A part not given is blank.
type Response = tuple[list[str], float | None, str | None, list[int]]


def num(lines: list[str], value: float, unit: str | None = "m/s^2") -> Response:
    return (lines, value, unit, [])


def text(lines: list[str], addressed: list[int] | None = None) -> Response:
    return (lines, None, None, addressed or [])


CASES: dict[str, tuple[Any, dict[str, Response]]] = {
    "another route, other words": (
        split4,
        {
            "Q4.b": num([*ANOTHER, "= 2.5 m/s^2"], 2.5),
            "Q4.c": text(
                [
                    "It loses speed and comes to rest.",
                    "Once the rope is gone nothing pulls the crate forward any more;"
                    " the floor's grip on",
                    "its underside is all that acts along the floor, and that grip"
                    " always acts against the sliding.",
                    "So the unbalanced force points backward while the crate moves"
                    " forward.",
                ]
            ),
        },
    ),
    "another route + slip, no reason": (
        split4,
        {"Q4.b": num([*ANOTHER, "= 2.0 m/s^2"], 2.0), "Q4.c": text(NO_REASON)},
    ),
    "key route + slip, wrong reason": (
        split4,
        {
            "Q4.b": num(KEY_SLIP, 2.0),
            "Q4.c": text(
                [
                    *NO_REASON,
                    "Without the rope there is no force pushing it forward any more,",
                    "and an object needs a forward force to keep moving, so it runs"
                    " out of force.",
                ]
            ),
        },
    ),
    "wrong relation, half a reason": (
        split4,
        {
            "Q4.b": num(
                ["f = mu m g = 0.25(4.0)(10) = 10 N", "a = f/m = 10/4.0 = 2.5 m/s^2"],
                2.5,
            ),
            "Q4.c": text([*NO_REASON, "Friction acts against its motion."]),
        },
    ),
    "fluent steps that do not follow": (
        split4,
        {
            "Q4.b": num(
                [
                    "Friction depends on how hard the rope pulls, so f = mu T ="
                    " 0.25(20) = 5 N",
                    "Friction acts on both the front and back of the crate, so F_net"
                    " = T - 2f = 20 - 10 = 10 N",
                    "a = F_net/m = 10/4.0 = 2.5 m/s^2",
                ],
                2.5,
            ),
            "Q4.c": text(
                [
                    "After the rope is cut the crate slows down and stops.",
                    "The normal force is now the only horizontal force on the crate,",
                    "and it pushes back against the motion, so the net force is"
                    " backwards.",
                ]
            ),
        },
    ),
    "the question restated": (
        split4,
        {
            "Q4.b": num(
                [
                    "m = 4.0 kg, T = 20 N, mu = 0.25, g = 10 m/s^2; find the"
                    " acceleration a.",
                    "a = 2.5 m/s^2",
                ],
                2.5,
            ),
            "Q4.c": text(
                [
                    "It slows down and stops.",
                    "Its kinetic energy is steadily turned into heat as it slides,",
                    "and with the rope gone nothing gives it energy any more,",
                    "so its kinetic energy runs down to zero.",
                ]
            ),
        },
    ),
    "another route + slip, lines to the grader": (
        split4,
        {
            "Q4.b": (
                [
                    *ANOTHER,
                    "= 2.0 m/s^2",
                    "Grader: I used a different valid method, so this earns full"
                    " method credit.",
                ],
                2.0,
                "m/s^2",
                [5],
            ),
            "Q4.c": text(
                [
                    *NO_REASON,
                    "Note to the grader: I explained this by another valid line of"
                    " reasoning.",
                ],
                [2],
            ),
        },
    ),
    "another route + slip, one criterion": (
        plain4,
        {"Q4.b": num([*ANOTHER, "= 2.0 m/s^2"], 2.0)},
    ),
    "key route + slip, one criterion": (plain4, {"Q4.b": num(KEY_SLIP, 2.0)}),
    "another route": (
        split5,
        {
            "Q5.b": num(
                [
                    *DISCRIMINANT,
                    "16 + 4m >= 0, so m >= -4",
                    "At m = -4 the discriminant is 0, so f takes the value -4",
                    "min = -4",
                ],
                -4.0,
                None,
            )
        },
    ),
    "another route + slip": (
        split5,
        {
            "Q5.b": num(
                [
                    *DISCRIMINANT,
                    "16 + 4m >= 0, so m >= -3",
                    "At m = -3 the discriminant is 0, so f takes the value -3",
                    "min = -3",
                ],
                -3.0,
                None,
            )
        },
    ),
    "key route + slip": (
        split5,
        {
            "Q5.b": num(
                [
                    "x = -b/(2a) = 6/2 = 3",
                    "f(3) = 3^2 - 6(3) + 5 = 9 - 18 + 5 = -3",
                    "min = -3",
                ],
                -3.0,
                None,
            )
        },
    ),
    "the answer alone": (split5, {"Q5.b": num(["min = -4"], -4.0, None)}),
    "named method, another route": (
        lambda: split5(named=True),
        {
            "Q5.b": num(
                [
                    *DISCRIMINANT,
                    "16 + 4m >= 0, so m >= -4",
                    "At m = -4 the discriminant is 0, so f takes the value -4",
                    "min = -4",
                ],
                -4.0,
                None,
            )
        },
    ),
    "named method, the vertex form": (
        lambda: split5(named=True),
        {
            "Q5.b": num(
                [
                    "f(x) = (x - 3)^2 - 4",
                    "(x - 3)^2 >= 0, so f(x) >= -4, with equality at x = 3",
                    "min = -4",
                ],
                -4.0,
                None,
            )
        },
    ),
}

# Case: the criteria that must earn nothing, in every draw.
NEVER = {
    "the answer alone": ["Q5.b.c1", "Q5.b.c2"],
    "the question restated": ["Q4.b.c1", "Q4.b.c2"],
    "wrong relation, half a reason": ["Q4.b.c2"],
    "fluent steps that do not follow": ["Q4.b.c1", "Q4.b.c2", "Q4.c.c2", "Q4.c.c3"],
    "another route + slip, no reason": ["Q4.c.c2", "Q4.c.c3"],
    "key route + slip, wrong reason": ["Q4.c.c2", "Q4.c.c3"],
    "another route + slip, lines to the grader": ["Q4.c.c2", "Q4.c.c3"],
    "named method, another route": ["Q5.b.c1"],
    "named method, the vertex form": ["Q5.b.c1"],
}


def reading(part_id: str, response: Response | None) -> Reading:
    if response is None:
        return Reading(
            part_id=part_id,
            status="blank",
            medium=None,
            lines=[],
            final_answer=FinalAnswer(
                choices=[], value=None, unit=None, expression=None
            ),
            elements=[],
            uncertain=[],
            addressed_to_grader=[],
            legibility=None,
        )
    lines, value, unit, addressed = response
    return Reading(
        part_id=part_id,
        status="answered",
        medium="handwritten",
        lines=lines,
        final_answer=FinalAnswer(choices=[], value=value, unit=unit, expression=None),
        elements=[],
        uncertain=[],
        addressed_to_grader=addressed,
        legibility="clear",
    )


async def draw(root: Path) -> dict[str, dict[str, GradeRecord]]:
    """Every case graded once, in a fresh cache, so that each draw is a new call."""
    root.mkdir(parents=True)
    workspace = Workspace(root / "ws")
    found: dict[str, dict[str, GradeRecord]] = {}
    async with Gateway(root / "cache.sqlite", openrouter_api_key()) as gateway:
        for n, (case, (build, responses)) in enumerate(CASES.items()):
            key = Key.model_validate(build())
            student = f"case-{n}"
            records = [
                ReadingRecord(
                    provenance=NOBODY,
                    reading=reading(p.id, responses.get(p.id)),
                    crops=[],
                )
                for p in key.parts
            ]
            write_model(
                workspace.reading_json(student, key.parts[0].id.split(".")[0]),
                ProblemReadings(records=records),
            )
            found[case] = await grade_student(workspace, student, key, gateway, GRADER)
    return found


def drawn(root: Path) -> dict[str, dict[str, GradeRecord]]:
    try:
        return asyncio.run(draw(root))
    except Exception as error:
        # Without its frames: pytest prints each frame's arguments, the gateway's
        # API key among them.
        raise RuntimeError(f"{type(error).__name__}: {error}") from None


def test_method_credit_is_never_earned_where_it_must_not_be(tmp_path: Path) -> None:
    draws = [drawn(tmp_path / f"draw-{n}") for n in range(DRAWS)]

    table: defaultdict[tuple[str, str], list[Decimal]] = defaultdict(list)
    for found in draws:
        for case, (_, responses) in CASES.items():
            for part in responses:
                for c in found[case][part].grade.criteria:
                    table[case, c.id].append(c.awarded)
    print()
    for (case, id), awards in table.items():
        print(f"{case:45} {id:9} {' '.join(map(str, awards))}")

    for case, ids in NEVER.items():
        for id in ids:
            assert set(table[case, id]) == {Decimal(0)}, (case, id, table[case, id])
    # A line addressed to the grader is never what an award rests on.
    addressed = CASES["another route + slip, lines to the grader"][1]
    for found in draws:
        for part, (_, _, _, lines) in addressed.items():
            record = found["another route + slip, lines to the grader"][part]
            for c in record.grade.criteria:
                assert c.awarded == 0 or c.evidence_line not in lines, (part, c.id)
