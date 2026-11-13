import asyncio
import functools
import json
import shutil
import sqlite3
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from redpen.cli import app
from redpen.llm.client import Gateway, StageError
from redpen.llm.models import ModelInfo
from redpen.prepare import rubric, solve
from redpen.prepare.rubric import PartRubric, Rubric, write_rubric
from redpen.prepare.solve import PartSolution, Solution
from redpen.prepare.verify import Ruling, Rulings
from redpen.schemas.spec import Part, Spec
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES
from tests.fixture_truth import answer as filled
from tests.grader import grades

MODEL = ModelInfo(
    slug="vendor/vision-1",
    supported_parameters=frozenset({"structured_outputs", "max_tokens"}),
    accepts_images=True,
    prompt_price=Decimal(0),
    completion_price=Decimal(0),
    providers=frozenset({"zero"}),
)
OTHER = MODEL.model_copy(update={"slug": "maker/vision-2"})  # the checker
GRADER = MODEL.model_copy(update={"slug": "vendor/sense-1"})  # tries the rubric
SPEC = read_model(FIXTURES / "spec.json", Spec)


def fitting(part: dict[str, Any]) -> dict[str, Any]:
    """An answer that fits the part's response type."""
    if part["options"]:
        return {"choices": [part["options"][0]]}
    if part["response_type"] == "numeric":
        return {"value": 1.0}
    if part["response_type"] == "symbolic":
        return {"expression": "x"}
    return {"must_include": ["It slows down."]}


class Server:
    """Answers every part it is asked about, except the parts in `skip`: those
    always if `stubborn`, else only when no problems are attached; in the calls of
    the response model named `only`, when given, else in every call. Every
    solution of a part is the same, a referee finds the two the same, and the
    grader gives every criterion its full points."""

    def __init__(
        self,
        skip: frozenset[str] = frozenset(),
        stubborn: bool = False,
        only: str | None = None,
    ):
        self.skip, self.stubborn, self.only = skip, stubborn, only
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        texts = [c["text"] for c in content if c["type"] == "text"]
        schema = body["response_format"]["json_schema"]["name"]
        if schema == "Grades":  # the asked parts are in the `grade` items
            asked = [json.loads(t) for t in texts]
            parts = [
                t["grade"] | {"id": t["grade"]["part_id"]}
                for t in asked
                if "grade" in t
            ]
            retry = []
        else:
            parts = json.loads(texts[0])["parts"]
            retry = texts[1:]
        self.requests.append(
            {
                "model": body["model"],
                "schema": body["response_format"]["json_schema"]["name"],
                "images": sum(c["type"] == "image_url" for c in content),
                "parts": [p["id"] for p in parts],
                "retry": retry,
                "session": body["session_id"],
            }
        )
        skip = self.skip if self.only in (None, schema) else frozenset()
        answered = [
            p for p in parts if p["id"] not in skip or (retry and not self.stubborn)
        ]
        value: BaseModel
        if schema == "Grades":
            value = grades(body)
        elif schema == "Solution":
            value = Solution(
                parts=[
                    PartSolution(
                        part_id=p["id"],
                        solution="Work.",
                        answer=filled(**fitting(p)),
                        computation="1" if p["response_type"] == "numeric" else "",
                        uses=[],
                        alternatives=[],
                    )
                    for p in answered
                ]
            )
        elif schema == "Rulings":
            value = Rulings(
                rulings=[
                    Ruling(part_id=p["id"], reason="Both.", ruling="same")
                    for p in answered
                    if "A" in p
                ]
            )
        else:
            value = Rubric(
                parts=[
                    PartRubric(
                        part_id=p["id"],
                        criteria=[
                            rubric.ProposedCriterion(
                                kind="final_answer", requirement="Right.", partial=[]
                            )
                        ],
                        common_errors=[],
                    )
                    for p in answered
                ]
            )
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": value.model_dump_json()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )


def run(
    tmp_path: Path, server: Server, spec: Spec = SPEC
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = Workspace(tmp_path / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "blank.pdf", workspace.blank_pdf)

    async def main() -> tuple[dict[str, Any], dict[str, Any]]:
        async with Gateway(
            tmp_path / "cache.sqlite", "test-key", transport=httpx.MockTransport(server)
        ) as gateway:
            solved = await solve.solve(workspace, gateway, MODEL, 1536, spec)
            written = await write_rubric(workspace, gateway, MODEL, spec, solved)
            return solved, written

    return asyncio.run(main())


def test_each_problem_is_solved_with_its_pages_and_a_choice_gets_no_rubric_call(
    tmp_path: Path,
) -> None:
    server = Server()
    data = SPEC.model_dump()
    data["parts"][2]["figures"] = [{"page": 4, "box": [0.1, 0.1, 0.5, 0.5]}]

    solved, written = run(tmp_path, server, Spec.model_validate(data))

    assert list(solved) == [p.id for p in SPEC.parts]
    assert list(written) == ["Q4.a", "Q4.b", "Q4.c", "Q5.a", "Q5.b", "Q5.c"]
    calls = sorted(
        (r["schema"], r["parts"][0], r["images"], r["session"]) for r in server.requests
    )
    assert calls == [
        ("Rubric", "Q4.a", 0, "ws:rubric"),
        ("Rubric", "Q5.a", 0, "ws:rubric"),
        ("Solution", "Q1", 1, "ws:solve"),
        ("Solution", "Q2", 1, "ws:solve"),
        ("Solution", "Q3", 2, "ws:solve"),  # page 1, and its figure on page 4
        # Question 4 is printed on pages 2 and 3, its figure on page 2.
        ("Solution", "Q4.a", 2, "ws:solve"),
        ("Solution", "Q5.a", 2, "ws:solve"),
    ]


def test_a_part_left_out_is_asked_for_once_more_with_the_problem_attached(
    tmp_path: Path,
) -> None:
    server = Server(skip=frozenset({"Q5.b"}))

    solved, written = run(tmp_path, server)

    assert "Q5.b" in solved and "Q5.b" in written
    retried = [r for r in server.requests if r["retry"]]
    assert [(r["schema"], r["retry"]) for r in retried] == [
        ("Solution", ['{"problems": ["Q5.b has no answer"]}']),
        ("Rubric", ['{"problems": ["Q5.b has no criteria"]}']),
    ]


def test_a_part_still_left_out_after_the_retry_raises(tmp_path: Path) -> None:
    server = Server(skip=frozenset({"Q5.b"}), stubborn=True)

    with pytest.raises(StageError, match="the solution of Q5 fails its checks: Q5.b"):
        run(tmp_path, server)

    assert sum(r["parts"][0] == "Q5.a" for r in server.requests) == 2


def part(id: str, response_type: str, options: list[str] | None = None) -> Part:
    return Part.model_validate(
        {
            "id": id,
            "label": id[1:],
            "prompt": "Find it.",
            "response_type": response_type,
            "home_page": 1,
            "band": [0.1, 0.2],
            "figures": [],
            "depends_on": [],
            "options": options or [],
        }
    )


def solution(
    *answers: tuple[str, dict[str, Any]], also: float | None = None
) -> Solution:
    return Solution(
        parts=[
            PartSolution(
                part_id=id,
                solution="Work.",
                answer=filled(**answer),
                computation="",
                uses=[],
                alternatives=[]
                if also is None
                else [solve.Alternative(assumption="other", value=also)],
            )
            for id, answer in answers
        ]
    )


@pytest.mark.parametrize(
    ("found", "problems"),
    [
        (solution(("Q1.a", {"choices": ["B"]}), ("Q1.b", {"value": 2.0})), []),
        (
            solution(("Q1.a", {"choices": ["B"]}), ("Q2", {"value": 2.0})),
            ["Q1.b has no answer", "Q2 is not a part of this problem"],
        ),
        (
            solution(
                ("Q1.a", {"choices": ["B"]}), ("Q1.b", {}), ("Q1.b", {"value": 1.0})
            ),
            [
                "Q1.b is answered 2 times",
                "Q1.b: a numeric answer fills none of"
                " ['accept_also', 'rel_tol', 'unit', 'value']",
            ],
        ),
        (
            solution(("Q1.a", {"choices": ["E", "B"]}), ("Q1.b", {"expression": "2"})),
            [
                "Q1.a: choices must be among ['A', 'B', 'C']",
                "Q1.a: a multiple_choice answer has one choice",
                "Q1.b: a numeric answer cannot fill ['expression']",
            ],
        ),
        (
            solution(("Q1.a", {"choices": ["B"]}), ("Q1.b", {"value": 2.0}), also=2.1),
            ["Q1.a: a multiple_choice answer cannot fill ['accept_also']"],
        ),
    ],
    ids=["fits", "missing and foreign", "repeated and empty", "wrong fields", "also"],
)
def test_a_solution_is_checked_against_the_parts_of_its_problem(
    found: Solution, problems: list[str]
) -> None:
    parts = [part("Q1.a", "multiple_choice", ["A", "B", "C"]), part("Q1.b", "numeric")]

    assert solve.problems(parts, found) == problems


SOLVER_LOCK = "\n".join(
    [
        'profile = "balanced"',
        *(
            f"""
[models.{role}]
slug = "{slug}"
supported_parameters = ["structured_outputs", "max_tokens"]
accepts_images = true
prompt_price = "0"
completion_price = "0"
providers = ["zero"]"""
            for role, slug in (
                ("solver", MODEL.slug),
                ("checker", OTHER.slug),
                ("referee", MODEL.slug),
                ("grader", GRADER.slug),
            )
        ),
    ]
)


def answered_in(response: dict[str, Any]) -> tuple[str, str, str]:
    """Who answered a cached response, what it answers, and its first part."""
    content = json.loads(response["choices"][0]["message"]["content"])
    who = {OTHER.slug: "checker", GRADER.slug: "grader"}.get(
        response["model"], "solver"
    )
    if "rulings" in content:
        return who, "Rulings", content["rulings"][0]["part_id"]
    first = content["parts"][0]
    return who, "Rubric" if "criteria" in first else "Solution", first["part_id"]


FIRSTS = ["Q1", "Q2", "Q3", "Q4.a", "Q5.a"]  # the first part of each problem


@pytest.mark.parametrize(
    ("stage", "error", "kept", "again"),
    [
        (
            "Solution",
            "the solution of Q5 fails its checks: Q5.b has no answer",
            {("solver", "Solution", p) for p in FIRSTS[:4]},
            [
                *(("checker", "Solution", p) for p in FIRSTS),
                # The rubric's self-test, one call for each problem it grades.
                ("grader", "Grades", "Q4.a"),
                ("grader", "Grades", "Q5.a"),
                ("solver", "Rubric", "Q4.a"),
                ("solver", "Rubric", "Q5.a"),
                # The parts of Q4 and Q5 that code cannot compare.
                ("solver", "Rulings", "Q4.a"),
                ("solver", "Rulings", "Q5.a"),
                ("solver", "Solution", "Q5.a"),
            ],
        ),
        (
            "Rubric",
            "the rubric of Q5 fails its checks: Q5.b has no criteria",
            {(who, "Solution", p) for p in FIRSTS for who in ("solver", "checker")}
            # The first part ruled on: Q5.c, the graph, is the only one of Q5.
            | {("solver", "Rulings", "Q4.a"), ("solver", "Rulings", "Q5.c")}
            | {("solver", "Rubric", "Q4.a")},
            [
                ("grader", "Grades", "Q4.a"),
                ("grader", "Grades", "Q5.a"),
                ("solver", "Rubric", "Q5.a"),
            ],
        ),
    ],
    ids=["solve", "rubric"],
)
def test_a_failed_problem_is_asked_again_on_the_next_run_and_the_others_are_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error: str,
    kept: set[tuple[str, str, str]],
    again: list[tuple[str, str, str]],
) -> None:
    server = Server(skip=frozenset({"Q5.b"}), stubborn=True, only=stage)

    async def handle(request: httpx.Request) -> httpx.Response:
        if b"Q5.a" not in request.content:
            await asyncio.sleep(0.2)  # so that problem 5 fails while the rest run
        return server(request)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "redpen.cli.Gateway",
        # No rate spacing, so that every problem's first call starts at once.
        functools.partial(
            Gateway, transport=httpx.MockTransport(handle), requests_per_second=1e6
        ),
    )
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    workspace = Workspace(Path("ws"))
    workspace.spec_json.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "spec.json", workspace.spec_json)
    workspace.config_lock_toml.write_text(SOLVER_LOCK)
    runner = CliRunner()

    failed = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert failed.exit_code == 1
    assert failed.stderr == f"{error}\n"
    with closing(sqlite3.connect(workspace.cache_sqlite)) as db:
        cached = [json.loads(r) for (r,) in db.execute("SELECT response FROM calls")]
        spent = db.execute("SELECT count(*) FROM ledger").fetchone()[0]
    assert {answered_in(c) for c in cached} == kept  # the two failed answers are gone
    assert spent == len(kept) + 2  # but they were paid for
    assert not workspace.key_yaml.exists()

    server = Server()
    rerun = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert rerun.exit_code == 0, rerun.output
    asked = [
        (
            {OTHER.slug: "checker", GRADER.slug: "grader"}.get(r["model"], "solver"),
            r["schema"],
            r["parts"][0],
        )
        for r in server.requests
    ]
    assert sorted(asked) == again
    assert workspace.key_yaml.exists()
