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
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Literal

from redpen import prompts
from redpen.grade.check import MATCH, NO_MATCH, check_answer, evaluate
from redpen.grade.grade import ROLE as GRADER
from redpen.llm.client import Gateway, ask_checked, finish_all
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.prepare.key import answer_lines, number, part_hash
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
from redpen.report.report import one_line
from redpen.schemas.common import Points, ResponseType, Schema
from redpen.schemas.grade import AnswerCheck
from redpen.schemas.key import Answer, Key, KeyPart, Policies
from redpen.schemas.reading import FinalAnswer, Reading
from redpen.schemas.spec import CHOICE_TYPES, Part, Spec
from redpen.workspace import Workspace, log_failure

CHECKER, REFEREE = "checker", "referee"
type Compared = Literal["agree", "differ", "cannot tell", "not compared"]
AGREE: Final = "agree"
DIFFER: Final = "differ"
UNSURE: Final = "cannot tell"
NOT_COMPARED: Final = "not compared"  # the second solve failed
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


# A criterion a response graded against the rubric did not earn in full: its
# requirement as the rubric then wrote it, and the grader's rationale.
class Lost(Schema):
    id: str
    kind: Literal["method", "final_answer"]
    requirement: str
    awarded: Points
    points: Points
    rationale: str


# A response built in code, graded against the key's rubric (design 6.5): "passed"
# when every criterion is at its full points, else "failed" with those below it in
# `lost`; "not graded" when the call failed or its answer failed the grade stage's
# checks twice.
class Trial(Schema):
    outcome: Literal["passed", "failed", "not graded"]
    lost: list[Lost]


# What verification found for one part. `checks` holds the arithmetic check of the
# first solution and the second: None when none was made, "passes", or the words of
# its failure. `ruling` is the referee's, as "same", "first", "second" or
# "neither"; "failed" when a call the part needed failed; None when the referee was
# not asked. `reason` is the referee's own text, or the fixed words of a failure.
# `stands` is what the key is written from. `part_hash` is the key part's when its
# state was set, filled in once the key is written. The self-test's fields keep
# their defaults in a record written before it existed, and for a part scored by
# code: `tried`, the key's own solution graded against the rubric, once, or twice
# when a fresh prepare wrote the problem's rubric again; `key_check`, the answer
# check of the key's own answer (None when it has nothing to check); and
# `second_tried`, the second solution graded the same way where the two solvers
# agreed, which sets no state.
class Verdict(Schema):
    part_id: str
    state: State
    part_hash: str = ""
    compared: Compared
    checks: tuple[str | None, str | None]
    ruling: Literal["same", "first", "second", "neither", "failed"] | None
    reason: str | None
    first: PartSolution
    second: PartSolution | None  # None when the second solve failed
    stands: PartSolution
    tried: list[Trial] = []
    key_check: AnswerCheck | None = None
    second_tried: Trial | None = None


# key/verification.json: the model that made the second solutions, whether the
# first ones were an existing key's, and each part's verdict in key order.
class Verification(Schema):
    checker: str
    keyed: bool
    parts: list[Verdict]


def vendor(slug: str) -> str:
    """The vendor of a model slug: the part before the slash."""
    return slug.split("/")[0]


def role_problems(models: Mapping[str, ModelInfo]) -> list[str]:
    """Why `models` cannot verify a key: a role among solver, checker, referee and
    the grader of the rubric's self-test that has no model, or a checker of the
    solver's own vendor, which would reuse the solver's answers. Nothing when it
    can."""
    roles = ("solver", CHECKER, REFEREE, GRADER)
    if missing := [r for r in roles if r not in models]:
        return [f"no model is pinned for {', '.join(missing)}"]
    solver_, checker = models["solver"].slug, models[CHECKER].slug
    if vendor(solver_) == vendor(checker):
        return [
            f"the checker {checker} is of the solver's vendor, so it cannot check"
            f" the solver {solver_}; pin a model of another vendor"
        ]
    return []


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
    is not rewritten. A call or a comparison that fails is written to the run log
    and leaves its parts unverified. Raises ValueError for `role_problems`."""
    if problems := role_problems(models):
        raise ValueError("; ".join(problems))
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

    def compared(part: Part, second: PartSolution) -> Compared:
        try:
            return compare(
                part.response_type,
                answer_of(first[part.id]),
                answer_of(second),
                policies,
            )
        except Exception as error:  # keep the traceback
            log_failure(workspace, f"verify {part.id}: the comparison", error)
            return DIFFER  # fail closed: no referee's "same" lifts it to verified

    async def one(parts: Sequence[Part]) -> list[Verdict]:
        problem = problem_of(parts)
        mine = {
            p.id: None if keyed else arithmetic(first[p.id], policies) for p in parts
        }
        try:
            second = {s.part_id: s for s in await again(parts)}
        except Exception as error:  # any failure: the parts stay unverified
            log_failure(workspace, f"verify {problem}: the second solve", error)
            why = f"The second solve failed ({type(error).__name__})."
            return [
                Verdict(
                    part_id=p.id,
                    state="unverified",
                    compared=NOT_COMPARED,
                    checks=(mine[p.id], None),
                    ruling="failed",
                    reason=why,
                    first=first[p.id],
                    second=None,
                    stands=first[p.id],
                )
                for p in parts
            ]
        checks = {p.id: (mine[p.id], arithmetic(second[p.id], policies)) for p in parts}
        found: dict[str, Compared] = {p.id: compared(p, second[p.id]) for p in parts}
        disputed = [
            p.id for p in parts if found[p.id] != AGREE or slipped(checks[p.id])
        ]
        # A is the first solution or the second by the problem's id alone, so that
        # the referee, which is the first solver's own model, cannot tell its own.
        a_first = a_is_first(problem)
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
                log_failure(workspace, f"verify {problem}: the referee", error)
                failure = f"The referee's call failed ({type(error).__name__})."
        verdicts: list[Verdict] = []
        for p in parts:
            a, b = first[p.id], second[p.id]
            state: State
            ruling: Literal["same", "first", "second", "neither", "failed"] | None
            if p.id not in disputed:
                state, stands, ruling, reason = "verified", a, None, None
            elif failure or p.id not in rulings:
                why = failure or "The referee gave no ruling on it."
                state, stands, ruling, reason = "unverified", a, "failed", why
            else:
                r = rulings[p.id]
                if r.ruling in ("A", "B"):
                    ruling = "first" if (r.ruling == "A") == a_first else "second"
                else:
                    ruling = r.ruling
                same: State = "refereed" if slipped(checks[p.id]) else "verified"
                if found[p.id] == DIFFER:  # code read the two as different
                    same = "unverified"  # and no model overrules it
                states: dict[str, State] = {
                    "same": same,
                    "first": "refereed",
                    "second": "unverified" if keyed else "refereed",
                    "neither": "unverified",
                }
                state = states[ruling]
                stands = b if ruling == "second" and not keyed else a
                reason = r.reason
            verdicts.append(
                Verdict(
                    part_id=p.id,
                    state=state,
                    compared=found[p.id],
                    checks=checks[p.id],
                    ruling=ruling,
                    reason=reason,
                    first=a,
                    second=b,
                    stands=stands,
                )
            )
        return verdicts

    done = await finish_all(one(parts) for parts in by_problem(spec))
    return {v.part_id: v for verdicts in done for v in verdicts}


def verification(
    found: Mapping[str, Verdict], key: Key, checker: str, keyed: bool
) -> Verification:
    """What `verification.json` holds once `key` is written from `found`: each
    verdict, in key order, with the hash of the key part its state was set on."""
    return Verification(
        checker=checker,
        keyed=keyed,
        parts=[
            found[p.id].model_copy(update={"part_hash": part_hash(p, key.policies)})
            for p in key.parts
        ],
    )


def changed(key: Key, record: Verification) -> list[str]:
    """One line for each part that verification settled and whose grading content
    has changed since, by hand: its hash is not the one its state was set on."""
    settled = {v.part_id: v for v in record.parts if v.state != "unverified"}
    return [
        f"{p.id} was changed after it was {settled[p.id].state}."
        for p in key.parts
        if p.id in settled and settled[p.id].part_hash != part_hash(p, key.policies)
    ]


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
) -> Compared:
    """`AGREE`, `DIFFER` or `UNSURE`. Option sets are compared directly. Numbers and
    expressions go through the answer check both ways, each answer taken once as
    the key and once as the response, with no leniency for rounding: the two agree
    when each matches the other, and differ when either is read as different. A
    missing unit is taken as the other's, since whether one must be written is a
    grading policy, not a disagreement between two solvers. An answer without the
    value or expression its type needs differs. Any other type cannot tell."""
    if response_type in CHOICE_TYPES:
        same = bool(first.choices) and set(first.choices) == set(second.choices)
        return AGREE if same else DIFFER
    if response_type not in COMPARED:
        return UNSURE
    if response_type == "numeric":
        missing = first.value is None or second.value is None
    else:
        missing = not (first.expression or "").strip()
        missing |= not (second.expression or "").strip()
    if missing:
        return DIFFER
    strict = policies.model_copy(
        update={"penalise_sig_figs": True, "require_units": False}
    )
    found = {
        _check(response_type, first, second, strict),
        _check(response_type, second, first, strict),
    }
    if found == {MATCH}:
        return AGREE
    return DIFFER if NO_MATCH in found else UNSURE


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


def summary(parts: Sequence[Verdict]) -> list[str]:
    """The self-test in counts and fixed words, for the terminal and the top of
    verification.md: the rubric's test, the key answers the answer check does not
    read as matching, and the criteria the second solution lost, by kind."""
    last = Counter(v.tried[-1].outcome for v in parts if v.tried)
    again = sum(len(v.tried) > 1 for v in parts)
    unread = sum(bool(v.key_check and v.key_check.verdict != MATCH) for v in parts)
    seconds = [v.second_tried for v in parts if v.second_tried]
    kinds = Counter(c.kind for t in seconds for c in t.lost)
    return [
        f"Rubric self-test: {last['passed']} passed, {last['failed']} failed,"
        f" {last['not graded']} not graded; {again} with the rubric written a second"
        " time.",
        f"Key answers the answer check does not read as matching: {unread}.",
        f"Second solution graded on {len(seconds)} parts: {kinds['method']} method and"
        f" {kinds['final_answer']} final-answer criteria lost.",
    ]


def _lost(found: Lost) -> str:
    """One Markdown item on one line: the model's text cannot break out of it."""
    kind = found.kind.replace("_", " ")
    worth = f"{number(found.awarded)} of {number(found.points)}"
    return "- " + one_line(
        f"{found.id} ({kind}, {worth}): {found.requirement} Grader: {found.rationale}"
    )


def _tried(v: Verdict) -> list[str]:
    """The self-test's lines of one part: each try of the rubric, the answer check
    of the key's own answer, and the second solution graded."""
    twice = len(v.tried) > 1
    lines = ["Its rubric was written a second time.", ""] if twice else []
    if not v.tried:
        lines = ["Self-test: not made, as code scores this part.", ""]
    which = ["first rubric", "rubric written again"] if twice else ["rubric"]
    for name, t in zip(which, v.tried, strict=False):
        lines += [f"Self-test on the {name}: {t.outcome}", ""]
        lines += [*map(_lost, t.lost), ""] if t.lost else []
    if v.key_check and v.key_check.verdict != MATCH:
        lines += [
            f"The answer check of the key's own answer: {v.key_check.verdict}."
            f" {v.key_check.reason} So the check will never decide this part for a"
            " student.",
            "",
        ]
    if second := v.second_tried:
        lines += [
            "The second solution graded the same way, as evidence that changes no"
            f" state: {second.outcome}",
            "",
        ]
        lines += [*map(_lost, second.lost), ""] if second.lost else []
    return lines


def verification_md(record: Verification, key: Key) -> str:
    """For the instructor, never read back, rendered from `record` with the
    headings of `key`: who made the second solutions, the counts by state, the parts
    that need a look and what to do; then each part's state, both final answers,
    the arithmetic checks and the referee's reason. Where the referee preferred the
    second solution to an existing key, the second solution in full, so that the key
    can be corrected by hand. Then the part's self-test. Whatever a model wrote is
    kept to one line, or fenced by more backticks than it holds."""
    keyed = record.keyed
    whose = "the key's own" if keyed else "the first solver's"
    look = [v.part_id for v in record.parts if v.state == "unverified"]
    lines = [
        "# Verification",
        "",
        f"The second solution of each problem was made by {one_line(record.checker)}."
        f" The first solution is {whose}. Then the key's own solution of each part a"
        " model grades was graded against its rubric as a response, by the grade"
        " stage, and has to earn every criterion in full.",
        "",
        *(
            [
                "The key keeps no final computation, so its side has no arithmetic"
                " check.",
                "",
            ]
            if keyed
            else []
        ),
        f"States: {state_counts(v.state for v in record.parts)}",
        "",
        *(line for found in summary(record.parts) for line in (found, "")),
        f"Parts that need a look: {', '.join(look)}. For each, correct key.yaml if"
        " it is wrong, then run `redpen key verify` again."
        if look
        else "Parts that need a look: none.",
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
    parts = {p.id: p for p in key.parts}
    for v in record.parts:
        part = parts[v.part_id]
        lines += [
            f"## {part.id}: {one_line(part.label)}, page {part.home_page}",
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
                *("- " + one_line(item.removeprefix("- ")) for item in given),
                "",
                f"Arithmetic check: {done}",
                "",
            ]
        lines += [f"Referee: {rulings[v.ruling]}", ""]
        if v.reason and v.ruling == "failed":
            lines += [one_line(v.reason), ""]
        elif v.reason:  # the referee's own words call the two A and B
            a = a_is_first(part.id.split(".")[0])
            whose = ("the key" if keyed else "the first") if a else "the second"
            lines += [f"Reason, where A is {whose}: {one_line(v.reason)}", ""]
        if keyed and v.ruling == "second" and v.second:
            text = v.second.solution
            fence = "`" * max(3, max(map(len, re.findall("`+", text)), default=0) + 1)
            lines += ["The second solution in full:", "", f"{fence}text", text, fence]
            lines.append("")
        lines += _tried(v)
    return "\n".join(lines)
