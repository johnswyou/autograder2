"""Verify the key (design 6.2). Each problem is solved a second time by the
`checker`, another vendor's model, with the solve prompt, schema and page images.
Code compares the two solutions where the response type allows it, and checks each
stated value against its final computation. What code cannot settle goes, in one
call per problem, to the `referee`, which sees the two solutions as A and B and is
told nothing of where they came from. Each part then gets one of the three states.

The stage serves a fresh key and one that already exists: `first` is the first
solver's work, or the key's own answers and solutions. A call, a check or a
comparison that fails leaves its parts unverified, never verified."""

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from redpen import prompts
from redpen.grade.check import MATCH, NO_MATCH, check_answer, evaluate
from redpen.llm.client import Gateway, ask_checked, finish_all
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.prepare.key import answer_lines
from redpen.prepare.parse import render_blank
from redpen.prepare.solve import (
    PARAMS,
    PartSolution,
    answer_of,
    by_problem,
    page_images,
    problem_of,
    solver,
)
from redpen.schemas.common import ResponseType, Schema
from redpen.schemas.key import Answer, KeyPart, Policies
from redpen.schemas.reading import FinalAnswer, Reading
from redpen.schemas.spec import CHOICE_TYPES, Part, Spec
from redpen.workspace import Workspace

CHECKER, REFEREE = "checker", "referee"
AGREE, DIFFER, UNSURE = "agree", "differ", "cannot tell"
# The response types whose answers code compares. Any other goes to the referee.
COMPARED = (*CHOICE_TYPES, "numeric", "symbolic")
PASSES = "passes"
# A failed arithmetic check, in the fixed words the referee is told.
UNREAD = "Its final computation could not be evaluated in code."
WRONG = "Its final computation, evaluated in code, does not give its stated value."
STATES = ("verified", "refereed", "unverified")

type State = Literal["verified", "refereed", "unverified"]


class Ruling(Schema):
    part_id: str
    reason: str
    ruling: Literal["same", "A", "B", "neither"]


class Rulings(Schema):
    rulings: list[Ruling]


@dataclass(frozen=True)
class Verdict:
    """What verification found for one part. `checks` holds the arithmetic check
    of each solution: None when none was made, `PASSES`, or the words of its
    failure. `ruling` is the referee's, as "same", "first", "second" or "neither";
    "failed" when a call the part needed failed; None when the referee was not
    asked. `reason` is the referee's own text, or the fixed words of a failure, and
    goes nowhere but verification.md."""

    state: State
    stands: PartSolution  # what the key is written from
    first: PartSolution
    second: PartSolution | None  # None when the second solve failed
    compared: str  # AGREE, DIFFER or UNSURE
    checks: tuple[str | None, str | None]
    ruling: str | None
    reason: str | None


async def verify(
    workspace: Workspace,
    gateway: Gateway,
    models: Mapping[str, ModelInfo],
    image_budget: int,
    spec: Spec,
    first: Mapping[str, PartSolution],
    policies: Policies,
    keyed: bool,
) -> dict[str, Verdict]:
    """Each part's verdict, by part id, the problems verified concurrently, from
    the `checker` and `referee` models. With `keyed`, `first` is an existing key's,
    which keeps no computation: only the second solution's arithmetic is checked,
    and a referee who prefers the second leaves the part unverified, since the key
    is not rewritten."""
    images = [page.image for page in render_blank(workspace)]
    again = solver(workspace, gateway, models[CHECKER], images, image_budget, CHECKER)
    system, version = prompts.load("verify", "problem")

    async def referee(
        parts: Sequence[Part], claims: Mapping[str, dict[str, Any]]
    ) -> dict[str, Ruling]:
        pages, shown = page_images(images, parts, image_budget)
        data = {
            "pages": pages,
            "parts": [
                p.model_dump(
                    include={"id", "label", "response_type", "prompt", "options"}
                )
                | claims.get(p.id, {})
                for p in parts
            ],
        }

        async def ask(extra: Sequence[Content]) -> Rulings:
            result = await gateway.call(
                REFEREE,
                models[REFEREE],
                system,
                [],
                [*shown, json.dumps(data), *extra],
                Rulings,
                version,
                params=PARAMS,
                session_id=f"{workspace.root.name}:verify",
            )
            return result.value

        found = await ask_checked(
            ask, lambda r: ruled(list(claims), r), f"the rulings of {problem_of(parts)}"
        )
        return {r.part_id: r for r in found.rulings}

    async def one(parts: Sequence[Part]) -> list[Verdict]:
        mine = {
            p.id: None if keyed else arithmetic(first[p.id], policies) for p in parts
        }
        try:
            second = {s.part_id: s for s in await again(parts)}
        except Exception as error:  # any failure: the parts stay unverified
            why = f"The second solve failed ({type(error).__name__})."
            return [
                Verdict(
                    "unverified",
                    first[p.id],
                    first[p.id],
                    None,
                    UNSURE,
                    (mine[p.id], None),
                    "failed",
                    why,
                )
                for p in parts
            ]
        checks = {p.id: (mine[p.id], arithmetic(second[p.id], policies)) for p in parts}
        compared = {
            p.id: compare(
                p.response_type,
                answer_of(first[p.id]),
                answer_of(second[p.id]),
                policies,
            )
            for p in parts
        }
        disputed = [
            p.id for p in parts if compared[p.id] != AGREE or slipped(checks[p.id])
        ]
        # A is the first solution or the second by the problem's id alone, so that
        # the referee, which is the first solver's own model, cannot tell its own.
        a_first = a_is_first(problem_of(parts))
        claims: dict[str, dict[str, Any]] = {}
        for id in disputed:
            mine_, theirs = (
                claim(first[id], checks[id][0]),
                claim(second[id], checks[id][1]),
            )
            claims[id] = (
                {"A": mine_, "B": theirs} if a_first else {"A": theirs, "B": mine_}
            )
        rulings: dict[str, Ruling] = {}
        failure = None
        if disputed:
            try:
                rulings = await referee(parts, claims)
            except Exception as error:  # any failure: its parts stay unverified
                failure = f"The referee's call failed ({type(error).__name__})."
        verdicts: list[Verdict] = []
        for p in parts:
            a, b = first[p.id], second[p.id]
            state: State
            if p.id not in disputed:
                state, stands, ruling, reason = "verified", a, None, None
            elif failure:
                state, stands, ruling, reason = "unverified", a, "failed", failure
            else:
                r = rulings[p.id]
                ruling = r.ruling
                if ruling in ("A", "B"):
                    ruling = "first" if (ruling == "A") == a_first else "second"
                states: dict[str, State] = {
                    "same": "refereed" if slipped(checks[p.id]) else "verified",
                    "first": "refereed",
                    "second": "unverified" if keyed else "refereed",
                    "neither": "unverified",
                }
                state = states[ruling]
                stands = b if ruling == "second" and not keyed else a
                reason = r.reason
            verdicts.append(
                Verdict(
                    state, stands, a, b, compared[p.id], checks[p.id], ruling, reason
                )
            )
        return verdicts

    found = await finish_all(one(parts) for parts in by_problem(spec))
    return {v.first.part_id: v for verdicts in found for v in verdicts}


def a_is_first(problem: str) -> bool:
    """Whether solution A is the first for `problem`: fixed by a digest of its id."""
    return hashlib.sha256(problem.encode()).digest()[0] % 2 == 0


def claim(found: PartSolution, check: str | None) -> dict[str, Any]:
    """A solution as the referee sees it: the work, the answer, and in fixed words
    a failed arithmetic check. Nothing says whose it is."""
    shown: dict[str, Any] = {
        "solution": found.solution,
        "answer": answer_of(found).model_dump(mode="json"),
    }
    if check not in (None, PASSES):
        shown["check"] = check
    return shown


def slipped(checks: Iterable[str | None]) -> bool:
    """Whether an arithmetic check was made and failed."""
    return any(c not in (None, PASSES) for c in checks)


def ruled(wanted: Sequence[str], found: Rulings) -> list[str]:
    """Every way `found` fails to rule on each part of `wanted` exactly once."""
    seen = Counter(r.part_id for r in found.rulings)
    wrong = [f"{id} has no ruling" for id in wanted if not seen[id]]
    wrong += [f"{id} is ruled {seen[id]} times" for id in wanted if seen[id] > 1]
    if any(id not in wanted for id in seen):  # its id is the model's text: not named
        wrong.append("a ruling names a part that has no A and B")
    return wrong


def arithmetic(found: PartSolution, policies: Policies) -> str | None:
    """The arithmetic check of `found`: None when its answer states no value, else
    `PASSES` when its computation gives the value within the answer's tolerance,
    else `UNREAD` or `WRONG`."""
    value = found.answer.value
    if value is None:
        return None
    got = evaluate(found.computation)
    if got is None:
        return UNREAD
    tol = policies.rel_tol if found.answer.rel_tol is None else found.answer.rel_tol
    return PASSES if abs(got - value) <= tol * (abs(value) or 1) else WRONG


def compare(
    response_type: ResponseType, first: Answer, second: Answer, policies: Policies
) -> str:
    """`AGREE`, `DIFFER` or `UNSURE`. Option sets are compared directly; numbers and
    expressions by the answer checks, the first taken as the key and the second as
    the response. Two numbers also agree when each one's value is among the other's
    accepted values. Any other type, or a check that raises, cannot tell."""
    try:
        if response_type in CHOICE_TYPES:
            return AGREE if set(first.choices) == set(second.choices) else DIFFER
        if response_type not in COMPARED:
            return UNSURE
        only = first.model_copy(update={"accept_also": []})
        found = _check(response_type, only, second, policies)
        if found != MATCH and response_type == "numeric":
            if (
                MATCH
                == _check(response_type, first, second, policies)
                == _check(response_type, second, first, policies)
            ):
                return AGREE
        return {MATCH: AGREE, NO_MATCH: DIFFER}.get(found or "", UNSURE)
    except Exception:  # fail closed: a comparison that raises settles nothing
        return UNSURE


def _check(
    response_type: ResponseType, key: Answer, said: Answer, policies: Policies
) -> str | None:
    """The answer check's verdict on `said`'s stated value or expression, against
    a key part whose answer is `key`."""
    part = KeyPart(
        id="Q1",
        label="",
        home_page=1,
        response_type=response_type,
        depends_on=[],
        state="unverified",
        prompt="",
        answer=key,
        solution="",
        criteria=[],
        common_errors=[],
    )
    stated = FinalAnswer(
        choices=[], value=said.value, unit=said.unit, expression=said.expression
    )
    reading = Reading(
        part_id="Q1",
        status="answered",
        medium=None,
        lines=[],
        final_answer=stated,
        elements=[],
        uncertain=[],
        addressed_to_grader=[],
        legibility=None,
    )
    found = check_answer(part, reading, policies)
    return found.verdict if found else None


def state_counts(states: Iterable[str]) -> str:
    """How many of `states` are each state: "7 verified, 1 refereed, 1 unverified"."""
    found = Counter(states)
    return ", ".join(f"{found[s]} {s}" for s in STATES)


def verification_md(spec: Spec, found: Mapping[str, Verdict], keyed: bool) -> str:
    """For the instructor, never read back: each part's state, both final answers,
    the arithmetic checks and the referee's reason. Where the referee preferred the
    second solution to an existing key, the second solution in full, so that the key
    can be corrected by hand."""
    whose = "the key's own" if keyed else "the first solver's"
    lines = [
        "# Verification",
        "",
        f"Each problem was solved a second time by another model. The first solution"
        f" is {whose}.",
        "",
        f"States: {state_counts(v.state for v in found.values())}",
        "",
    ]
    rulings = {
        None: "not asked",
        "same": "the two agree in substance",
        "first": "the key is right" if keyed else "the first is right",
        "second": "the second is right",
        "neither": "neither is right, or the question is ambiguous",
        "failed": "none",
    }
    for part in spec.parts:
        v = found[part.id]
        lines += [
            f"## {part.id}: {part.label}, page {part.home_page}",
            "",
            f"State: {v.state}",
            "",
            f"Compared in code: {v.compared}",
            "",
        ]
        for name, solution, check in (
            ("First", v.first, v.checks[0]),
            ("Second", v.second, v.checks[1]),
        ):
            given = answer_lines(answer_of(solution)) if solution else ["- none"]
            done = {None: "not made", PASSES: "passes"}.get(check, f"fails. {check}")
            lines += [
                f"{name} answer:",
                "",
                *given,
                "",
                f"Arithmetic check: {done}",
                "",
            ]
        lines += [f"Referee: {rulings[v.ruling]}", ""]
        if v.reason and v.ruling == "failed":
            lines += [v.reason, ""]
        elif v.reason:  # the referee's own words call the two A and B
            a = a_is_first(part.id.split(".")[0])
            whose = ("the key" if keyed else "the first") if a else "the second"
            lines += [f"Reason, where A is {whose}: {v.reason}", ""]
        if keyed and v.ruling == "second" and v.second:
            lines += ["The second solution in full:", "", "```text"]
            lines += [v.second.solution, "```", ""]
    return "\n".join(lines)
