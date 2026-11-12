import asyncio
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from redpen.llm.client import Gateway
from redpen.llm.models import ModelInfo
from redpen.prepare import rubric, solve
from redpen.prepare.rubric import PartRubric, Rubric, write_rubric
from redpen.prepare.solve import PartSolution, Solution, StageError
from redpen.schemas.key import Answer
from redpen.schemas.spec import Part, Spec
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES

MODEL = ModelInfo(
    slug="vendor/vision-1",
    supported_parameters=frozenset({"structured_outputs", "max_tokens"}),
    accepts_images=True,
    prompt_price=Decimal(0),
    completion_price=Decimal(0),
    providers=frozenset({"zero"}),
)
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
    always if `stubborn`, else only when no problems are attached."""

    def __init__(self, skip: frozenset[str] = frozenset(), stubborn: bool = False):
        self.skip, self.stubborn = skip, stubborn
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        texts = [c["text"] for c in content if c["type"] == "text"]
        parts = json.loads(texts[0])["parts"]
        retry = texts[1:]
        self.requests.append(
            {
                "schema": body["response_format"]["json_schema"]["name"],
                "images": sum(c["type"] == "image_url" for c in content),
                "parts": [p["id"] for p in parts],
                "retry": retry,
                "session": body["session_id"],
            }
        )
        answered = [
            p
            for p in parts
            if p["id"] not in self.skip or (retry and not self.stubborn)
        ]
        value: BaseModel
        if body["response_format"]["json_schema"]["name"] == "Solution":
            value = Solution(
                parts=[
                    PartSolution(
                        part_id=p["id"],
                        solution="Work.",
                        answer=Answer.model_validate(fitting(p)),
                        uses=[],
                        alternatives=[],
                    )
                    for p in answered
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
                "model": MODEL.slug,
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
                answer=Answer.model_validate(answer),
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
