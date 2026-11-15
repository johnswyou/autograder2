"""Grade one read student (without a second grader or referee, a second
read or a re-map). Code scores a part read blank and a choice part; every other part
of a top-level problem goes to the grader in one call, with the answer check's
verdict beside each part that has one, and each award it returns is checked against
the key and held to that verdict. Code adds up points."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from autograder2 import prompts
from autograder2.grade.check import MATCH, UNSURE, Check, check_answer
from autograder2.grade.read import digest, problem_of
from autograder2.llm.client import Gateway, ask_checked, finish_all
from autograder2.llm.client import Part as Content
from autograder2.llm.models import ModelInfo
from autograder2.prepare.key import grading_content, number, part_hash
from autograder2.schemas.common import Provenance, Schema
from autograder2.schemas.grade import (
    AnswerCheck,
    CriterionGrade,
    Grade,
    GradeRecord,
    ProblemGrades,
)
from autograder2.schemas.key import Criterion, Key, KeyPart
from autograder2.schemas.reading import ProblemReadings, Reading, ReadingRecord
from autograder2.schemas.spec import CHOICE_TYPES
from autograder2.workspace import Workspace, read_model, write_model

ROLE = "grader"
PARAMS = {"max_tokens": 8000}


# What the grader returns for one problem.
class Grades(Schema):
    grades: list[Grade]


async def grade_student(
    workspace: Workspace,
    student: str,
    key: Key,
    gateway: Gateway,
    model: ModelInfo,
) -> dict[str, GradeRecord]:
    """Grade `student` from their stored readings and return each graded part's
    record, by part id, with the digest of the readings it was graded from
    (`graded_from`). A part
    read as unreadable gets none, and so does a choice part answered with no option
    read, and a problem with no stored reading is not graded. Each top-level
    problem's file is replaced as soon as it is graded, keeping each stored record
    that nothing it was graded from changed for (`kept`). The
    problems run concurrently; when one still fails its checks after one more call,
    `StageError` is raised once every other problem has finished, and that
    problem's earlier file, if any, is left as it was."""
    done: dict[str, GradeRecord] = {}

    async def grade(problem: str, parts: list[KeyPart]) -> None:
        if (found := request(workspace, student, key, parts)) is None:
            return
        records = found.by_code | await ask_grader(found, key, gateway, model)
        path = workspace.grade_json(student, problem)
        records = kept(path, parts, records)
        file = ProblemGrades(records=[records[p.id] for p in parts if p.id in records])
        write_model(path, file)
        done.update(records)

    await finish_all(grade(problem, parts) for problem, parts in by_problem(key))
    return {p.id: done[p.id] for p in key.parts if p.id in done}


def kept(
    path: Path, parts: list[KeyPart], records: dict[str, GradeRecord]
) -> dict[str, GradeRecord]:
    """`records`, the new records of one problem's `parts`, with the record stored
    in `path` in place of each new one made under the same key part, prompt and
    model, from the same readings and with the same answer check, when every part
    it depends on is of this problem and kept too: a fresh draw of the grader does
    not change an award that nothing changed for."""
    try:
        stored = {r.grade.part_id: r for r in read_model(path, ProblemGrades).records}
    except (OSError, ValueError):  # none, or damaged: nothing to keep
        stored = {}
    found = dict(records)
    same: set[str] = set()
    for part in parts:  # a part depends on earlier parts only (`key.check`)
        old, new = stored.get(part.id), records.get(part.id)
        if (
            old is not None
            and new is not None
            and (old.provenance, old.reading_digest, old.check)
            == (new.provenance, new.reading_digest, new.check)
            and all(d in same for d in part.depends_on)
        ):
            same.add(part.id)
            found[part.id] = old
    return found


def by_problem(key: Key) -> list[tuple[str, list[KeyPart]]]:
    """The key's parts by top-level problem, in key order."""
    found: defaultdict[str, list[KeyPart]] = defaultdict(list)
    for part in key.parts:
        found[problem_of(part.id)].append(part)
    return list(found.items())


@dataclass(frozen=True)
class Request:
    """The grade stage's request for one top-level problem of a student, the same
    however often it is made: the records of the parts code scores, the parts the
    grader is asked about, and what it is sent. `shown` holds every reading the
    parts are graded from: their own, and those of the parts they depend on."""

    problem: str
    student: str
    shown: dict[str, ReadingRecord]
    by_code: dict[str, GradeRecord]
    asked: list[KeyPart]
    wanted: dict[str, Reading]
    checks: dict[str, Check | None]
    system: str
    version: str
    session: str
    shared: list[str]
    items: list[Content]


def request(
    workspace: Workspace, student: str, key: Key, parts: list[KeyPart]
) -> Request | None:
    """The request for `parts`, one top-level problem of `student`, from the stored
    readings; None when the problem has no stored reading. A part read as
    unreadable is neither scored nor asked about, nor is a choice part answered
    with no option read. Nothing is sent."""
    problem = problem_of(parts[0].id)
    if not workspace.reading_json(student, problem).exists():
        return None
    system, version = prompts.load("grade", "problem")
    # As `part_hash` digests them, so that the hash covers what the grader is sent.
    policies = json.dumps(key.policies.model_dump(), default=number)

    def readings(problem: str) -> dict[str, ReadingRecord]:
        file = read_model(workspace.reading_json(student, problem), ProblemReadings)
        return {r.reading.part_id: r for r in file.records}

    mine = readings(problem)
    # Every reading the problem's parts are graded from: their own, and those of
    # the parts they depend on, which may be another problem's.
    shown = dict(mine)
    for d in dict.fromkeys(d for p in parts for d in p.depends_on):
        if d not in shown and workspace.reading_json(student, problem_of(d)).exists():
            shown |= readings(problem_of(d))
    records: dict[str, GradeRecord] = {}
    asked: list[KeyPart] = []
    for part in parts:
        reading = mine[part.id].reading
        choice = part.response_type in CHOICE_TYPES
        # Withheld, never zero: work that cannot be read, and a choice
        # part answered with no option read, which no grader that knows the key
        # may settle.
        if reading.status == "unreadable" or (
            choice and reading.status == "answered" and not reading.final_answer.choices
        ):
            continue
        if reading.status == "blank" or choice:
            records[part.id] = GradeRecord(
                provenance=Provenance(
                    key_part_hash=part_hash(part, key.policies),
                    prompt_version=None,
                    model_slug=None,
                ),
                grade=by_code(part, reading),
                check=None,
                reading_digest=graded_from(part, shown),
            )
        else:
            asked.append(part)
    # The same for every student, so that it is cached as a prefix.
    shared = [
        *(grading_content(p) for p in parts if p.response_type not in CHOICE_TYPES),
        policies,
    ]
    wanted = {p.id: mine[p.id].reading for p in asked}
    checks = {p.id: check_answer(p, wanted[p.id], key.policies) for p in asked}
    items: list[Content] = []
    for d in dict.fromkeys(d for p in asked for d in p.depends_on):
        if d not in wanted:
            record = mine.get(d) or readings(problem_of(d))[d]
            items.append(json.dumps({"context": _json(record.reading)}))
    for part in asked:
        record = mine[part.id]
        item = {"grade": _json(record.reading)}
        if verdict := checks[part.id]:
            item["check"] = verdict._asdict()
        items.append(json.dumps(item))
        # shortcut: a region the map gave to two parts is sent once for each;
        # send it once when real maps share boxes often.
        for n in range(1, len(record.crops) + 1):
            png = workspace.crop_png(student, part.id, n).read_bytes()
            items += [json.dumps({"crop": n, "part": part.id}), png]
    return Request(
        problem=problem,
        student=student,
        shown=shown,
        by_code=records,
        asked=asked,
        wanted=wanted,
        checks=checks,
        system=system,
        version=version,
        session=f"{workspace.root.name}:grade",
        shared=shared,
        items=items,
    )


async def ask_grader(
    found: Request,
    key: Key,
    gateway: Gateway,
    model: ModelInfo,
    role: str = ROLE,
    draw: int = 1,
) -> dict[str, GradeRecord]:
    """The record of each part `found` asks about, from `model`'s answer to its
    request (none when it asks about none), checked, with one
    more call when the answer fails the checks; `StageError` when it still does.
    `role` names the calls in the ledger, and a `draw` after the first asks for
    another answer to the same request (`Gateway.call`)."""
    if not found.asked:
        return {}

    async def ask(extra: Sequence[Content]) -> Grades:
        result = await gateway.call(
            role,
            model,
            found.system,
            found.shared,
            [*found.items, *extra],
            Grades,
            found.version,
            params=PARAMS,
            session_id=found.session,
            draw=draw,
        )
        return result.value

    answer = await ask_checked(
        ask,
        lambda g: problems(key, found.wanted, g),
        f"the grade of {found.problem} of {found.student}",
    )
    by_id = {p.id: p for p in found.asked}
    records: dict[str, GradeRecord] = {}
    for g in answer.grades:
        allowed = {c.id: levels(c) for c in by_id[g.part_id].criteria}
        # Each award as the key writes the level the checks matched it to.
        criteria = [
            c.model_copy(
                update={"awarded": next(x for x in allowed[c.id] if x == c.awarded)}
            )
            for c in g.criteria
        ]
        # A reason that overrules no verdict is not an overrule.
        verdict = found.checks[g.part_id]
        # On a part with several final-answer criteria, a reason is kept,
        # unasked, only when every one of them goes against the check.
        against = _against(by_id[g.part_id], verdict, g)
        overruled = bool(against) and len(against) == finals(by_id[g.part_id])
        kept = g.model_copy(
            update={
                "criteria": criteria,
                "code_overruled": g.code_overruled if overruled else None,
            }
        )
        provenance, graded = stamps(found, by_id[g.part_id], key, model)
        records[g.part_id] = GradeRecord(
            provenance=provenance,
            grade=kept,
            check=AnswerCheck(**verdict._asdict()) if verdict else None,
            reading_digest=graded,
        )
    return records


def stamps(
    found: Request, part: KeyPart, key: Key, model: ModelInfo
) -> tuple[Provenance, str]:
    """The provenance and the reading digest (`graded_from`) of the record of
    `part` made from `model`'s answer to `found`."""
    provenance = Provenance(
        key_part_hash=part_hash(part, key.policies),
        prompt_version=found.version,
        model_slug=model.slug,
    )
    return provenance, graded_from(part, found.shown)


def graded_from(part: KeyPart, readings: Mapping[str, ReadingRecord]) -> str:
    """The digest of the stored reading records `part` is graded from, found by part
    id in `readings`: its own and, shown to the grader as context, those of the
    parts it depends on (one with no reading counts as none). With no dependency it
    is the digest of its own record, as a record made before dependencies counted
    has it."""
    own = digest(readings[part.id])
    if not part.depends_on:
        return own
    found = [digest(readings[d]) if d in readings else "none" for d in part.depends_on]
    return hashlib.sha256(" ".join([own, *found]).encode()).hexdigest()[:12]


def by_code(part: KeyPart, reading: Reading) -> Grade:
    """The grade code gives a part read blank (zero on every criterion) or a choice
    part with an option read (full points when the options read are the key's,
    compared without regard to case or surrounding space, else zero; two or more
    options where the key has one are zero, as contradictory answers)."""
    read = reading.final_answer.choices
    if reading.status == "blank":
        earned, why = False, "No response."
    elif len(options(part.answer.choices)) == 1 and len(options(read)) > 1:
        earned, why = False, "Two or more options are marked; the question has one."
    else:
        earned = options(read) == options(part.answer.choices)
        chosen, answer = ", ".join(read), ", ".join(part.answer.choices)
        why = (
            f"Chooses {chosen}, the answer."
            if earned
            else f"Chooses {chosen}; the answer is {answer}."
        )
    return Grade(
        part_id=part.id,
        criteria=[
            CriterionGrade(
                id=c.id,
                awarded=c.points if earned else Decimal(0),
                evidence_line=None,
                evidence_element=None,
                rationale=why,
                feedback="",
            )
            for c in part.criteria
        ],
        transcript_dispute=None,
        belongs_elsewhere=None,
        code_overruled=None,
    )


def problems(key: Key, asked: Mapping[str, Reading], found: Grades) -> list[str]:
    """Every way `found` fails the checks for the parts `asked`, by
    id with their readings, in a fixed order (it is sent back to the model). Only
    the key's ids, levels and line and element numbers are named: nothing else the
    model wrote is repeated."""
    parts = {p.id: p for p in key.parts}
    seen = Counter(g.part_id for g in found.grades)
    wrong = [f"{id} has no grade" for id in asked if not seen[id]]
    wrong += [
        f"{id} is not to be graded in this request"
        if id in parts
        else "a grade is for a part that is not in the key"
        for id in seen
        if id not in asked
    ]
    wrong += [
        f"{id} is graded {n} times" for id, n in seen.items() if n > 1 and id in asked
    ]
    for g in found.grades:
        if (reading := asked.get(g.part_id)) is None:
            continue
        criteria = {c.id: c for c in parts[g.part_id].criteria}
        if Counter(c.id for c in g.criteria) != Counter(list(criteria)):
            wrong.append(
                f"{g.part_id}: its criteria are {', '.join(criteria)}, each once"
            )
        for c in g.criteria:
            if (want := criteria.get(c.id)) is None:
                continue
            allowed = levels(want)
            if c.awarded not in allowed:
                wrong.append(
                    f"{c.id}: the award is one of {', '.join(map(number, allowed))}"
                )
            for what, n, count in (
                ("line", c.evidence_line, len(reading.lines)),
                ("element", c.evidence_element, len(reading.elements)),
            ):
                if n is not None and n > count:
                    wrong.append(
                        f"{c.id}: it cites {what} {n}, and the reading of"
                        f" {g.part_id} has no {what} {n}"
                    )
            if c.awarded > 0 and c.evidence_line is None and c.evidence_element is None:
                wrong.append(
                    f"{c.id}: an award above zero cites a line or an element"
                    " of the reading"
                )
            # A student who loses points is told why. This also catches an award
            # written against its own reasoning.
            if c.awarded < want.points and not c.feedback.strip():
                wrong.append(
                    f"{c.id}: an award below its full points gives feedback to the"
                    " student"
                )
        # Only a part with exactly one final-answer criterion is held: with two,
        # one is often about something the check did not compare.
        part = parts[g.part_id]
        if not (g.code_overruled or "").strip() and finals(part) == 1:
            wrong += [
                f"{id}: the award goes against the answer check, so code_overruled"
                " gives the reason"
                for id in _against(part, check_answer(part, reading, key.policies), g)
            ]
    return list(dict.fromkeys(wrong))


def part_points(record: GradeRecord, key: Key) -> tuple[Decimal, Decimal]:
    """The points `record` awards and the points its part is worth in `key`. A
    record graded under another version of the part raises `ValueError`."""
    [part] = [p for p in key.parts if p.id == record.grade.part_id]
    if record.provenance.key_part_hash != part_hash(part, key.policies):
        raise ValueError(
            f"{part.id} was graded under another version of its key part:"
            " grade it again"
        )
    awarded = sum((c.awarded for c in record.grade.criteria), Decimal(0))
    return awarded, sum((c.points for c in part.criteria), Decimal(0))


def _against(part: KeyPart, found: Check | None, grade: Grade) -> list[str]:
    """The final-answer criteria whose awards in `grade` go against the answer
    check `found`: below full points on a match, full points on a mismatch (a
    partial level the key lists is a wrong answer's, so it agrees with a mismatch)."""
    full = {c.id: c.points for c in part.criteria if c.kind == "final_answer"}
    if found is None or found.verdict == UNSURE:
        return []
    return [
        c.id
        for c in grade.criteria
        if c.id in full
        and (
            c.awarded < full[c.id]
            if found.verdict == MATCH
            else c.awarded >= full[c.id] > 0
        )
    ]


def finals(part: KeyPart) -> int:
    return sum(c.kind == "final_answer" for c in part.criteria)


def levels(criterion: Criterion) -> list[Decimal]:
    """The awards `criterion` allows, as the key writes them: zero, its full points
    and its partial levels."""
    return sorted(
        {Decimal(0), criterion.points, *(x.points for x in criterion.partial)}
    )


def options(labels: Sequence[str]) -> set[str]:
    return {label.strip().casefold() for label in labels}


def _json(reading: Reading) -> object:
    return reading.model_dump(mode="json")
