"""The second grade (src/redpen/grade/second_grade.py): which problems are graded
again, the request, the two answers kept apart, the comparison, a failure, and a
run again, with readings stored by hand and a mock transport. What a result does to
the flags, the scores and the report is in test_report.py; the commands are in
test_orchestrator.py."""

import asyncio
import contextlib
import json
import sqlite3
from collections.abc import Callable, Coroutine
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from redpen.grade.grade import Grades, grade_student
from redpen.grade.read import digest
from redpen.grade.second_grade import ROLE, differing, grade_again
from redpen.llm.client import Gateway, StageError
from redpen.llm.ledger import Ledger
from redpen.schemas.grade import (
    Grade,
    GradeRecord,
    ProblemGrades,
    SecondGrade,
    SecondGrades,
)
from redpen.schemas.key import Key
from redpen.schemas.reading import FinalAnswer, Reading
from redpen.workspace import Workspace, read_model
from tests.unit.test_grade import (
    ASKED,
    MODEL,
    PARTS,
    Q4,
    STUDENT,
    blank,
    full,
    key_of,
    reading,
    reply,
    store,
)

# Q1 a choice and Q5.a read blank, both scored by code; Q4 asked of the grader.
KEY = key_of(PARTS["Q1"], *Q4.parts, PARTS["Q5.a"])
CHOSEN = FinalAnswer(choices=["C"], value=None, unit=None, expression=None)
MARKER = "MARKER-the-grader-wrote-this"


class Grader:
    """Answers each request with `answer(the part ids it grades, how many requests
    with the same bytes came before)`, by default full points; None is HTTP 400.
    Keeps every request's bytes."""

    def __init__(
        self, answer: Callable[[list[str], int], Grades | None] | None = None
    ) -> None:
        self.answer = answer or (
            lambda ids, n: Grades(grades=[full(KEY, i) for i in ids])
        )
        self.sent: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        before = self.sent.count(request.content)
        self.sent.append(request.content)
        content = json.loads(request.content)["messages"][1]["content"]
        texts = [c["text"] for c in content if c["type"] == "text"]
        ids = [json.loads(t)["grade"]["part_id"] for t in texts if '"grade"' in t]
        value = self.answer(ids, before)
        if value is None:
            return httpx.Response(400, json={"error": {"message": "refused"}})
        return reply(json.dumps(value.model_dump(), default=float))


def stored(
    workspace: Workspace, q4a: Reading = ASKED["Q4.a"], q5a: bool = False
) -> None:
    """Q1 answered C, Q4 answered (each part with a crop), and Q5.a blank, or
    answered with `q5a`."""
    q5 = reading("Q5.a") if q5a else blank("Q5.a")
    store(
        workspace,
        [reading("Q1", final_answer=CHOSEN), q4a, ASKED["Q4.b"], ASKED["Q4.c"], q5],
        crops=1,
    )


def on_gateway[T](
    workspace: Workspace,
    grader: Grader,
    stage: Callable[[Gateway], Coroutine[Any, Any, T]],
) -> T:
    async def main() -> T:
        async with Gateway(
            workspace.root.parent / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(grader),
        ) as gateway:
            return await stage(gateway)

    return asyncio.run(main())


def first_grade(workspace: Workspace, grader: Grader, key: Key = KEY) -> None:
    async def stage(gateway: Gateway) -> None:
        with contextlib.suppress(StageError):
            await grade_student(workspace, STUDENT, key, gateway, MODEL)

    on_gateway(workspace, grader, stage)


def second_grade(
    workspace: Workspace, grader: Grader, key: Key = KEY
) -> list[SecondGrade]:
    return on_gateway(
        workspace,
        grader,
        lambda gateway: grade_again(workspace, STUDENT, key, gateway, MODEL),
    )


def grade_files(workspace: Workspace) -> dict[Path, bytes]:
    folder = workspace.grade_json(STUDENT, "Q1").parent
    return {p: p.read_bytes() for p in folder.glob("Q*.json")}


def firsts(workspace: Workspace, problem: str = "Q4") -> dict[str, GradeRecord]:
    path = workspace.grade_json(STUDENT, problem)
    return {r.grade.part_id: r for r in read_model(path, ProblemGrades).records}


# --- which problems, and the request ---------------------------------------------


def test_each_problem_a_model_graded_is_asked_once_more_byte_for_byte(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "ws")
    stored(workspace)
    grader = Grader()
    first_grade(workspace, grader)
    graded = grade_files(workspace)

    found = second_grade(workspace, grader)

    # Q4 alone went to the grader, so it alone is asked again, with the same bytes;
    # Q1 (a choice) and Q5 (read blank) are scored by code and never compared.
    [once, again] = grader.sent
    assert once == again
    assert [r.part_id for r in found] == ["Q4.a", "Q4.b", "Q4.c"]
    assert {r.result for r in found} == {"same"}
    first = firsts(workspace)
    for r in found:
        assert r.first == digest(first[r.part_id])
        assert r.part_hash == first[r.part_id].provenance.key_part_hash
        assert r.reading_digest == first[r.part_id].reading_digest
        assert r.second is not None
        assert r.second.provenance == first[r.part_id].provenance
    path = workspace.second_grade_json(STUDENT)
    assert read_model(path, SecondGrades).records == found
    assert grade_files(workspace) == graded  # no grade file is written
    with closing(sqlite3.connect(workspace.root.parent / "cache.sqlite")) as db:
        assert set(Ledger(db).totals_by_role()) == {"grader", ROLE}


def test_the_two_answers_are_kept_apart_and_a_second_run_asks_for_neither(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "ws")
    stored(workspace)

    def lower_the_second(ids: list[str], n: int) -> Grades:
        grades = [full(KEY, i) for i in ids]
        if n == 1:  # the second answer: a partial level against full points
            b = grades[1]
            lowered = b.criteria[0].model_copy(
                update={"awarded": Decimal(1), "rationale": MARKER}
            )
            grades[1] = b.model_copy(update={"criteria": [lowered, *b.criteria[1:]]})
        return Grades(grades=grades)

    grader = Grader(lower_the_second)
    first_grade(workspace, grader)
    found = second_grade(workspace, grader)

    assert len(grader.sent) == 2
    assert {r.part_id: r.result for r in found} == {
        "Q4.a": "same",
        "Q4.b": "differ",
        "Q4.c": "same",
    }
    [b] = [r for r in found if r.part_id == "Q4.b"]
    assert b.second is not None
    assert [c.awarded for c in b.second.grade.criteria] == [1, 1]
    # The first grade is the grade: its file holds the first answer.
    assert [c.awarded for c in firsts(workspace)["Q4.b"].grade.criteria] == [2, 1]
    second_file = workspace.second_grade_json(STUDENT).read_bytes()
    graded = grade_files(workspace)

    first_grade(workspace, grader)
    again = second_grade(workspace, grader)

    assert len(grader.sent) == 2  # the cache answers both draws
    assert again == found
    assert workspace.second_grade_json(STUDENT).read_bytes() == second_file
    assert grade_files(workspace) == graded


@pytest.mark.parametrize("earlier", ["no earlier file", "an earlier file"])
def test_no_problem_is_graded_again_where_its_first_grade_failed(
    tmp_path: Path, earlier: str
) -> None:
    workspace = Workspace(tmp_path / "ws")
    stored(workspace)
    if earlier == "an earlier file":  # graded once, then Q4.a read anew
        first_grade(workspace, Grader())
        second_grade(workspace, Grader())
        stored(workspace, ASKED["Q4.a"].model_copy(update={"lines": ["y = 5"]}))
    q4 = workspace.grade_json(STUDENT, "Q4")
    kept = q4.read_bytes() if q4.exists() else None
    # Every answer leaves Q4.c out, so the grade stage fails Q4 twice.
    grader = Grader(lambda ids, n: Grades(grades=[full(KEY, i) for i in ids[:2]]))
    first_grade(workspace, grader)
    assert len(grader.sent) == 2  # the first grade's call and its one retry
    assert (q4.read_bytes() if q4.exists() else None) == kept

    found = second_grade(workspace, grader)

    assert len(grader.sent) == 2
    assert found == []
    assert not workspace.second_grade_json(STUDENT).exists()


# --- a failure ---------------------------------------------------------------------


@pytest.mark.parametrize("how", ["refused", "fails its checks twice"])
def test_a_failed_second_grade_is_logged_and_the_other_problems_go_on(
    tmp_path: Path, how: str
) -> None:
    key = key_of(*Q4.parts, PARTS["Q5.a"])  # two problems asked
    workspace = Workspace(tmp_path / "ws")
    stored(workspace, q5a=True)
    asked_q4 = 0

    def later_q4_fails(ids: list[str], n: int) -> Grades | None:
        nonlocal asked_q4
        grades = [full(key, i) for i in ids]
        asked_q4 += "Q4.a" in ids
        if "Q4.a" not in ids or asked_q4 == 1:
            return Grades(grades=grades)
        if how == "refused":
            return None
        # Q4.c graded on one criterion of its two, the marker its rationale.
        c = grades[2].model_copy(update={"criteria": grades[2].criteria[:1]})
        c.criteria[0].rationale = MARKER
        return Grades(grades=[*grades[:2], c])

    grader = Grader(later_q4_fails)
    first_grade(workspace, grader, key)
    graded = grade_files(workspace)

    found = second_grade(workspace, grader, key)

    assert {r.part_id: r.result for r in found} == {
        "Q4.a": "failed",
        "Q4.b": "failed",
        "Q4.c": "failed",
        "Q5.a": "same",
    }
    assert all(r.second is None for r in found if r.result == "failed")
    # The first draw of each problem, the second of each, and the one retry.
    assert len(grader.sent) == (4 if how == "refused" else 5)
    log = workspace.run_log.read_text()
    [entry] = log.split("--- ")[1:]
    assert entry.split("\n")[0].split(" ", 1)[1] == (
        f"{ROLE} {STUDENT}: Q4 was not graded a second time"
    )
    assert MARKER not in log
    assert grade_files(workspace) == graded  # every first grade stands


# --- the comparison ----------------------------------------------------------------


def awarded(*awards: str) -> Grade:
    grade = full(KEY, "Q4.b")
    criteria = [
        c.model_copy(update={"awarded": Decimal(a)})
        for c, a in zip(grade.criteria, awards, strict=True)
    ]
    return grade.model_copy(update={"criteria": criteria})


@pytest.mark.parametrize(
    ("first", "second", "differ"),
    [
        (("2", "1"), ("2", "1"), []),
        (("2", "1"), ("2", "0"), ["Q4.b.c2"]),
        (("2", "1"), ("1", "1"), ["Q4.b.c1"]),
        (("2", "1"), ("0", "0"), ["Q4.b.c1", "Q4.b.c2"]),
        (("1", "1.0"), ("1.00", "1"), []),
    ],
    ids=["equal", "one criterion", "a partial level against full", "both", "1.00"],
)
def test_two_grades_are_compared_award_by_award(
    first: tuple[str, str], second: tuple[str, str], differ: list[str]
) -> None:
    assert differing(awarded(*first), awarded(*second)) == differ
