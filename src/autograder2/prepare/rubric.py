"""Write the rubric: one call per top-level problem, given its parts and
their solutions. A choice part gets no call: code writes its one criterion."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from autograder2 import prompts
from autograder2.llm.client import Gateway, ask_checked, finish_all
from autograder2.llm.client import Part as Content
from autograder2.llm.models import ModelInfo
from autograder2.prepare.solve import (
    PARAMS,
    ROLE,
    PartSolution,
    answer_of,
    by_problem,
    problem_of,
)
from autograder2.schemas.common import Schema
from autograder2.schemas.key import PartialCredit
from autograder2.schemas.spec import CHOICE_TYPES, Part, Spec
from autograder2.workspace import Workspace


# One criterion as the model proposes it. Code gives it its id and its one point.
class ProposedCriterion(Schema):
    kind: Literal["method", "final_answer"]
    requirement: str
    partial: list[PartialCredit]


class PartRubric(Schema):
    part_id: str
    criteria: list[ProposedCriterion]
    common_errors: list[str]


class Rubric(Schema):
    parts: list[PartRubric]


async def write_rubric(
    workspace: Workspace,
    gateway: Gateway,
    model: ModelInfo,
    spec: Spec,
    solutions: Mapping[str, PartSolution],
    not_earned: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
) -> dict[str, PartRubric]:
    """The proposed criteria of every part that is not a choice part, by part id.
    The problems are written concurrently. Raises `StageError` if a problem's rubric
    still fails its checks after one retry. With `not_earned`, by problem id, only
    those problems are written, each given, after what it is given the first time,
    the criteria of its earlier rubric that the key's own solution did not earn."""
    system, version = prompts.load("rubric", "problem")
    session = f"{workspace.root.name}:rubric"

    async def one(parts: list[Part]) -> list[PartRubric]:
        data = {
            "parts": [
                {
                    **p.model_dump(include={"id", "label", "response_type", "prompt"}),
                    "solution": solutions[p.id].solution,
                    "answer": answer_of(solutions[p.id]).model_dump(mode="json"),
                }
                for p in parts
            ]
        }

        given = (
            [json.dumps({"not_earned": not_earned[problem_of(parts)]})]
            if not_earned is not None
            else []
        )

        async def ask(extra: Sequence[Content]) -> Rubric:
            result = await gateway.call(
                ROLE,
                model,
                system,
                [],
                [json.dumps(data), *given, *extra],
                Rubric,
                version,
                params=PARAMS,
                session_id=session,
            )
            return result.value

        found = await ask_checked(
            ask, lambda r: problems(parts, r), f"the rubric of {problem_of(parts)}"
        )
        return found.parts

    free = [
        [p for p in parts if p.response_type not in CHOICE_TYPES]
        for parts in by_problem(spec)
    ]
    wanted = [
        p for p in free if p and (not_earned is None or problem_of(p) in not_earned)
    ]
    written = await finish_all(one(parts) for parts in wanted)
    return {r.part_id: r for found in written for r in found}


def problems(parts: Sequence[Part], found: Rubric) -> list[str]:
    """Every way `found` fails to give each of `parts`, and nothing else, at least
    one criterion."""
    wanted = [part.id for part in parts]  # a list: the message order is in the key
    seen = Counter(r.part_id for r in found.parts)
    wrong = [f"{id} has no criteria" for id in wanted if not seen[id]]
    wrong += [f"{id} is not a part of this problem" for id in seen if id not in wanted]
    wrong += [f"{id} is given {n} times" for id, n in seen.items() if n > 1]
    wrong += [f"{r.part_id} has no criteria" for r in found.parts if not r.criteria]
    return wrong
