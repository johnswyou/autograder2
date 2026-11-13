"""The point allocator (task 4.5a): points go to parts, in whole steps, so that they
sum exactly to the instructor's total or page maximums; a part's score is its
points times the share of its criteria's points earned, rounded to the step. No
model call anywhere."""

import re
import shutil
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from typer.testing import CliRunner

from redpen.cli import app
from redpen.grade.ingest import IngestSummary
from redpen.grade.read import problem_of
from redpen.prepare.key import (
    allocation,
    load_key,
    page_points,
    part_hash,
    point_problems,
    save_key,
    scaled,
)
from redpen.report.report import Stored, points_of, score, sum_of
from redpen.schemas.common import Provenance
from redpen.schemas.grade import CriterionGrade, Grade, GradeRecord
from redpen.schemas.key import Criterion, Key, KeyPart, PointConstraint, Policies
from redpen.schemas.reading import FinalAnswer, Reading, ReadingRecord
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, write_model
from tests.fixture_truth import FIXTURES, NOBODY, answer

D = Decimal
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
runner = CliRunner()


def part(id: str, page: int, *weights: Decimal | str) -> KeyPart:
    """A part on `page` with one criterion of each weight."""
    return KeyPart(
        id=id,
        label=id,
        home_page=page,
        response_type="other",
        depends_on=[],
        state="verified",
        prompt="p",
        answer=answer(),
        solution="s",
        criteria=[
            Criterion(
                id=f"{id}.c{n}",
                points=D(w),
                kind="method",
                requirement="r",
                partial=[],
                carry_forward=False,
            )
            for n, w in enumerate(weights, 1)
        ],
        common_errors=[],
    )


def keyed(parts: Sequence[KeyPart], step: str = "0.25", **points: object) -> Key:
    return Key(
        policies=Policies(point_step=D(step)),
        parts=list(parts),
        points=PointConstraint.model_validate(points) if points else None,
    )


def with_points(key: Key, **points: object) -> Key:
    return key.model_copy(update={"points": PointConstraint.model_validate(points)})


# --- by hand --------------------------------------------------------------------------


def test_the_worked_example_of_the_brief() -> None:
    # A 4-point page at 0.25 is 16 steps; the weights 9, 2, 2, 5 sum to 18.
    # 16 x (9, 2, 2, 5) / 18 = 8, 1.78, 1.78, 4.44: floors 8, 1, 1, 4 leave 2 steps,
    # which go to the two largest remainders (14/18 each): 8, 2, 2, 4 steps.
    key = keyed(
        [
            part("Q1", 1, "9"),
            part("Q2", 1, "2"),
            part("Q3", 1, "2"),
            part("Q4", 1, "5"),
        ],
        pages=["4"],
    )

    assert allocation(key) == {"Q1": 2, "Q2": D("0.5"), "Q3": D("0.5"), "Q4": 1}


def test_a_part_left_with_nothing_takes_a_step_from_the_later_of_the_largest() -> None:
    # A 1-point page is 4 steps for weights 1, 9, 9 (sum 19): 4 x (1, 9, 9) / 19 =
    # 0.21, 1.89, 1.89. Floors 0, 1, 1 leave 2 steps for Q2 and Q3 (remainders 17,
    # 17 against Q1's 4): 0, 2, 2. Q1 has nothing, so it takes one step from the part
    # holding the most; Q2 and Q3 tie at 2, and the later one, Q3, gives it up.
    key = keyed(
        [part("Q1", 1, "1"), part("Q2", 1, "9"), part("Q3", 1, "9")], pages=["1"]
    )

    assert allocation(key) == {"Q1": D("0.25"), "Q2": D("0.5"), "Q3": D("0.25")}


def test_a_step_left_over_between_equal_remainders_goes_to_the_earlier_part() -> None:
    # 3 steps for two equal weights: 1.5 each, floors 1 and 1, and the step left
    # goes to Q1.
    key = keyed([part("Q1", 1, "2"), part("Q2", 1, "2")], pages=["0.75"])

    assert allocation(key) == {"Q1": D("0.5"), "Q2": D("0.25")}


def test_a_total_is_shared_by_all_parts_and_a_page_s_maximum_is_its_parts_sum() -> None:
    # 20 points is 80 steps for weights 3, 5, 4, 4 (16): 15, 25, 20, 20 steps.
    key = with_points(KEY, total="20")

    # The fixture's parts by page: Q1-Q3 (1 each), Q4.a 2 and Q4.b 3, Q4.c 2 and
    # Q5.a 2, Q5.b 2 and Q5.c 2. 80 x 1/16 = 5 steps, x 2/16 = 10, x 3/16 = 15.
    assert list(allocation(key).values()) == [
        D("1.25"),
        D("1.25"),
        D("1.25"),
        D("2.5"),
        D("3.75"),
        D("2.5"),
        D("2.5"),
        D("2.5"),
        D("2.5"),
    ]
    assert page_points(key) == {1: D("3.75"), 2: D("6.25"), 3: 5, 4: 5}


def test_with_no_constraint_each_part_is_worth_its_criteria() -> None:
    assert list(allocation(KEY).values()) == [1, 1, 1, 2, 3, 2, 2, 2, 2]
    assert page_points(KEY) == {1: 3, 2: 5, 3: 4, 4: 4}


@pytest.mark.parametrize(
    ("points", "earned", "weight", "score"),
    [
        ("1", "1", "3", "0.25"),  # a third of 1: 1.33 steps, down to 1
        ("1", "2", "3", "0.75"),  # two thirds of 1: 2.67 steps, up to 3
        ("1", "1", "8", "0.25"),  # 0.125 is half a step: a tie goes up
        ("1.5", "1", "4", "0.5"),  # 0.375 is 1.5 steps: up to 2
        ("2", "0", "9", "0"),  # nothing earned
        ("2", "9", "9", "2"),  # everything earned
        ("0.25", "1", "18", "0"),  # 0.0139 is under half a step
    ],
    ids=["a third", "two thirds", "a tie", "another tie", "nothing", "all", "tiny"],
)
def test_a_score_is_rounded_to_the_nearest_step_a_tie_up(
    points: str, earned: str, weight: str, score: str
) -> None:
    assert scaled(D(points), D(earned), D(weight), D("0.25")) == D(score)


# --- the constraint in the key --------------------------------------------------------


def hashes(key: Key) -> dict[str, str]:
    return {p.id: part_hash(p, key.policies) for p in key.parts}


def test_no_part_hash_depends_on_the_constraint() -> None:
    pages, total = with_points(KEY, pages=[6, 4, 3, 2]), with_points(KEY, total=20)

    assert hashes(pages) == hashes(total) == hashes(KEY)


def test_the_constraint_is_written_at_the_end_of_key_yaml_and_left_out_when_unset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "key.yaml"
    save_key(path, KEY)
    plain = path.read_text()
    path.write_text("# Checked by hand.\n" + plain)

    for points, block in [
        (
            {"pages": [6, "4.5", 3, 2]},
            "  pages:\n    - 6\n    - 4.5\n    - 3\n    - 2\n",
        ),
        ({"total": "20"}, "  total: 20\n"),
    ]:
        key = with_points(KEY, **points)
        save_key(path, key)
        # Set, then replaced: only the field changes, and the comment stays.
        assert path.read_text() == "# Checked by hand.\n" + plain + "points:\n" + block
        assert load_key(path) == key
        assert hashes(load_key(path)) == hashes(KEY)

    save_key(path, KEY)

    assert "\npoints:" not in plain
    assert path.read_text() == "# Checked by hand.\n" + plain
    assert load_key(path).points is None


@pytest.mark.parametrize("points", [{}, {"total": 4, "pages": [4]}])
def test_a_constraint_is_a_total_or_page_maximums_never_both(
    points: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="either a total or page maximums"):
        PointConstraint.model_validate(points)


# --- refusals -------------------------------------------------------------------------


def moved_off_page_4(key: Key) -> Key:
    parts = [
        p.model_copy(update={"home_page": 3}) if p.home_page == 4 else p
        for p in key.parts
    ]
    return key.model_copy(update={"parts": parts})


REFUSED = [
    (with_points(KEY, pages=[3, -1, 4, 4]), "page 2's maximum is -1, below zero"),
    (with_points(KEY, total=-4), "the total is -4, below zero"),
    (
        with_points(KEY, pages=[3, "5.1", 4, 4]),
        "page 2's maximum is 5.1, not a multiple of the point step 0.25",
    ),
    (
        with_points(KEY, total="15.1"),
        "the total is 15.1, not a multiple of the point step 0.25",
    ),
    (
        with_points(KEY, pages=[3, 5, 4, 4, 0]),
        "the key gives 5 page maximums and the blank has 4 pages",
    ),
    (
        with_points(moved_off_page_4(KEY), pages=[3, 5, 8, 4]),
        "page 4's maximum is 4, but page 4 has no part",
    ),
    (
        with_points(KEY, pages=["0.5", 5, 4, 4]),
        "page 1's maximum is 0.5, 2 steps of 0.25 for the 3 parts of page 1",
    ),
    (
        with_points(KEY, pages=[0, 5, 4, 4]),
        "page 1's maximum is 0, 0 steps of 0.25 for the 3 parts of page 1",
    ),
    (
        with_points(KEY, total=2),
        "the total is 2, 8 steps of 0.25 for the 9 parts of the assignment",
    ),
]
REFUSED_IDS = [
    "page below zero",
    "total below zero",
    "page off the step",
    "total off the step",
    "wrong number of pages",
    "points but no part",
    "fewer steps than parts",
    "no step for parts",
    "total fewer steps than parts",
]


def refused_workspace(tmp_path: Path, key: Key) -> Workspace:
    """A workspace with the fixture blank (4 pages), spec and `key`, and an
    ingest of nobody: enough for `key check` and `report` to reach the key."""
    workspace = Workspace(tmp_path / "ws")
    workspace.spec_json.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "spec.json", workspace.spec_json)
    shutil.copy(FIXTURES / "blank.pdf", workspace.blank_pdf)
    save_key(workspace.key_yaml, key)
    write_model(
        workspace.ingest_summary_json,
        IngestSummary(
            students=[],
            duplicates=[],
            not_on_roster=[],
            missing_from_roster=[],
            ignored_files=0,
            unsafe_members=[],
            failed=[],
        ),
    )
    return workspace


@pytest.mark.parametrize(("key", "line"), REFUSED, ids=REFUSED_IDS)
def test_a_constraint_that_cannot_be_met_is_refused_in_one_line_naming_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: Key, line: str
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    workspace = refused_workspace(tmp_path, key)
    root = str(workspace.root)

    checked = runner.invoke(app, ["key", "check", "--workspace", root])
    reported = runner.invoke(app, ["report", "--workspace", root])

    assert checked.exit_code == 1
    assert checked.output.splitlines() == [line]
    assert reported.exit_code == 1
    assert isinstance(reported.exception, SystemExit)  # no traceback
    assert reported.stderr.splitlines() == [line]
    assert not (workspace.root / "out").exists()
    assert point_problems(key, 4) == [line]
    if "blank has" not in line:  # the page count is the blank's, not the key's
        with pytest.raises(ValueError, match=re.escape(line)):
            allocation(key)


# --- properties -----------------------------------------------------------------------

STEPS = [D("0.1"), D("0.25"), D("0.5"), D("1"), D("2")]


@st.composite
def constrained(draw: st.DrawFn) -> Key:
    """A key whose parts lie on random pages with random weights in steps, and a
    random constraint that can be met: page maximums or a total of at least one
    step per part."""
    step = draw(st.sampled_from(STEPS))
    pages = draw(st.integers(1, 6))
    parts: list[KeyPart] = []
    for q in range(1, draw(st.integers(1, 5)) + 1):
        subs = draw(st.integers(0, 3))
        for id in [f"Q{q}"] if subs == 0 else [f"Q{q}.{'abc'[i]}" for i in range(subs)]:
            weights = [
                step * draw(st.integers(1, 12)) for _ in range(draw(st.integers(1, 3)))
            ]
            parts.append(part(id, draw(st.integers(1, pages)), *weights))
    if draw(st.booleans()):
        total = step * (len(parts) + draw(st.integers(0, 60)))
        return keyed(parts, str(step), total=total)
    counts = [sum(p.home_page == n for p in parts) for n in range(1, pages + 1)]
    maxima = [step * (n + draw(st.integers(0, 30))) if n else D(0) for n in counts]
    return keyed(parts, str(step), pages=maxima)


@settings(max_examples=300)
@given(key=constrained())
def test_allocated_points_always_sum_exactly_to_the_constraint(key: Key) -> None:
    points, step = allocation(key), key.policies.point_step
    assert key.points is not None

    if key.points.total is not None:
        assert sum(points.values(), D(0)) == key.points.total
    else:
        for n, maximum in enumerate(key.points.pages or [], 1):
            on_page = [points[p.id] for p in key.parts if p.home_page == n]
            assert sum(on_page, D(0)) == maximum
            assert page_points(key).get(n, D(0)) == maximum
    assert list(points) == [p.id for p in key.parts]
    assert all(v > 0 and v % step == 0 for v in points.values())
    # Nothing but the key and the constraint: the same input, read back, gives the
    # same points.
    assert allocation(Key.model_validate_json(key.model_dump_json())) == points
    assert point_problems(key, len(key.points.pages or [])) == []


def stored(key: Key, earned: Mapping[str, Sequence[Decimal]]) -> Stored:
    """A student whose every part is read and graded under `key`, earning
    `earned[part]` on its criteria."""
    nothing = FinalAnswer(choices=[], value=None, unit=None, expression=None)
    readings = {
        p.id: ReadingRecord(
            provenance=NOBODY,
            reading=Reading(
                part_id=p.id,
                status="answered",
                medium="typed",
                lines=["x"],
                final_answer=nothing,
                elements=[],
                uncertain=[],
                addressed_to_grader=[],
                legibility="clear",
            ),
            crops=[],
        )
        for p in key.parts
    }
    grades = {
        p.id: GradeRecord(
            provenance=Provenance(
                key_part_hash=part_hash(p, key.policies),
                prompt_version=None,
                model_slug=None,
            ),
            grade=Grade(
                part_id=p.id,
                criteria=[
                    CriterionGrade(
                        id=c.id,
                        awarded=a,
                        evidence_line=None,
                        evidence_element=None,
                        rationale="",
                        feedback="",
                    )
                    for c, a in zip(p.criteria, earned[p.id], strict=True)
                ],
                transcript_dispute=None,
                belongs_elsewhere=None,
                code_overruled=None,
            ),
        )
        for p in key.parts
    }
    student = IngestRecord(
        student_name="A Student", student_key="s", source_kind="other", pages=[]
    )
    return Stored(student, readings, grades)


@settings(max_examples=200)
@given(key=constrained(), data=st.data())
def test_scores_stay_on_the_step_within_the_points_and_every_sum_agrees(
    key: Key, data: st.DataObject
) -> None:
    step = key.policies.point_step
    earned = {
        p.id: [
            step * data.draw(st.integers(0, int(c.points / step))) for c in p.criteria
        ]
        for p in key.parts
    }
    points = allocation(key)

    scores = score(key, stored(key, earned))

    for s in scores:
        assert s.awarded is not None
        assert 0 <= s.awarded <= points[s.part.id] and s.awarded % step == 0
        weight = sum((c.points for c in s.part.criteria), D(0))
        got = sum(earned[s.part.id], D(0))
        # Never less for more earned, up to everything.
        more = min(got + step, weight)
        assert scaled(points[s.part.id], more, weight, step) >= s.awarded
        assert scaled(points[s.part.id], weight, weight, step) == points[s.part.id]
    total = sum((s.awarded or D(0) for s in scores), D(0))
    pages = sorted({p.home_page for p in key.parts})
    by_page = [sum_of(s for s in scores if s.part.home_page == n) for n in pages]
    problems = dict.fromkeys(problem_of(p.id) for p in key.parts)
    by_problem = [
        sum_of(s for s in scores if problem_of(s.part.id) == q) for q in problems
    ]
    assert sum((c or D(0) for c in by_page), D(0)) == total
    assert sum((c or D(0) for c in by_problem), D(0)) == total
    maxima = page_points(key)
    for n, cell in zip(pages, by_page, strict=True):
        assert cell is not None and cell <= maxima[n]
    for q, cell in zip(problems, by_problem, strict=True):
        parts = [p for p in key.parts if problem_of(p.id) == q]
        assert cell is not None and cell <= points_of(key, parts)
