"""Solve each problem of the spec (design 6.2, one solver): one call per top-level
problem, with the images of the pages it is printed on and the text of its parts."""

import json
from collections import Counter, defaultdict
from collections.abc import Sequence

from redpen import prompts
from redpen.imaging.prep import reduced_png
from redpen.llm.client import Gateway, ask_checked, finish_all
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.prepare.parse import render_blank
from redpen.schemas.common import Schema
from redpen.schemas.key import Answer, answer_problem
from redpen.schemas.spec import CHOICE_TYPES, Part, Spec
from redpen.workspace import Workspace

ROLE = "solver"
PARAMS = {"max_tokens": 32000}


# An assumption the question leaves open, such as a constant used with two values,
# and the final value that follows from the other defensible choice.
class Alternative(Schema):
    assumption: str
    value: float


# The solver's work on one part. `uses` names the earlier parts whose results the
# solution uses.
class PartSolution(Schema):
    part_id: str
    solution: str
    answer: Answer
    uses: list[str]
    alternatives: list[Alternative]


class Solution(Schema):
    parts: list[PartSolution]


async def solve(
    workspace: Workspace,
    gateway: Gateway,
    model: ModelInfo,
    image_budget: int,
    spec: Spec,
) -> dict[str, PartSolution]:
    """Every part's solution, by part id. The problems are solved concurrently.
    Raises `StageError` if a problem's solution still fails its checks after one
    retry."""
    images = [page.image for page in render_blank(workspace)]
    system, version = prompts.load("solve", "problem")
    session = f"{workspace.root.name}:solve"

    async def one(parts: list[Part]) -> list[PartSolution]:
        pages = sorted(
            {p.home_page for p in parts} | {f.page for p in parts for f in p.figures}
        )
        data = {
            "pages": pages,
            "parts": [
                p.model_dump(
                    include={"id", "label", "response_type", "prompt", "options"}
                )
                for p in parts
            ],
        }
        shown = [reduced_png(images[n - 1], image_budget) for n in pages]

        async def ask(extra: Sequence[Content]) -> Solution:
            items = [*shown, json.dumps(data), *extra]
            result = await gateway.call(
                ROLE,
                model,
                system,
                [],
                items,
                Solution,
                version,
                params=PARAMS,
                session_id=session,
            )
            return result.value

        found = await ask_checked(
            ask, lambda s: problems(parts, s), f"the solution of {problem_of(parts)}"
        )
        return found.parts

    solved = await finish_all(one(parts) for parts in by_problem(spec))
    return {s.part_id: s for found in solved for s in found}


def by_problem(spec: Spec) -> list[list[Part]]:
    """The parts of the spec grouped by top-level problem, in the spec's order."""
    groups: defaultdict[str, list[Part]] = defaultdict(list)
    for part in spec.parts:
        groups[part.id.split(".")[0]].append(part)
    return list(groups.values())


def problem_of(parts: Sequence[Part]) -> str:
    return parts[0].id.split(".")[0]


def answer_of(found: PartSolution) -> Answer:
    """The key's answer: the solver's, with the value that follows from each
    alternative assumption accepted as well."""
    also = [*found.answer.accept_also, *(a.value for a in found.alternatives)]
    return found.answer.model_copy(update={"accept_also": list(dict.fromkeys(also))})


def problems(parts: Sequence[Part], found: Solution) -> list[str]:
    """Every way `found` fails to answer exactly `parts`, each in a fitting form."""
    wanted = {part.id: part for part in parts}
    seen = Counter(s.part_id for s in found.parts)
    wrong = [f"{id} has no answer" for id in wanted if not seen[id]]
    wrong += [f"{id} is not a part of this problem" for id in seen if id not in wanted]
    wrong += [
        f"{id} is answered {n} times"
        for id, n in seen.items()
        if n > 1 and id in wanted
    ]
    for s in found.parts:
        part = wanted.get(s.part_id)
        if part is None:
            continue
        answer = answer_of(s)
        if problem := answer_problem(part.response_type, answer):
            wrong.append(f"{part.id}: {problem}")
        elif part.response_type in CHOICE_TYPES:
            if not set(answer.choices) <= set(part.options):
                wrong.append(f"{part.id}: choices must be among {part.options}")
            if part.response_type == "multiple_choice" and len(answer.choices) != 1:
                wrong.append(f"{part.id}: a multiple_choice answer has one choice")
    return wrong


def depends_on(spec: Spec, part: Part, found: PartSolution) -> list[str]:
    """The parts `part` depends on: those the spec or the solver names that come
    earlier in the spec, in the spec's order. A missed dependency would deny
    carry-forward credit, and an extra one costs nothing."""
    ids = [p.id for p in spec.parts]
    named = {*part.depends_on, *found.uses}
    return [id for id in ids[: ids.index(part.id)] if id in named]
