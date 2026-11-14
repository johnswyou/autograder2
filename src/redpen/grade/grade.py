"""Grade one read student (design 7.5, without a second grader or referee, a second
read or a re-map). Code scores a part read blank and a choice part; every other part
of a top-level problem goes to the grader in one call, with the answer check's
verdict beside each part that has one, and each award it returns is checked against
the key and held to that verdict. Code adds up points."""

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal

from redpen import prompts
from redpen.grade.check import MATCH, UNSURE, Check, check_answer
from redpen.grade.read import digest, problem_of
from redpen.llm.client import Gateway, ask_checked, finish_all
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.prepare.key import grading_content, number, part_hash
from redpen.schemas.common import Provenance, Schema
from redpen.schemas.grade import (
    AnswerCheck,
    CriterionGrade,
    Grade,
    GradeRecord,
    ProblemGrades,
)
from redpen.schemas.key import Criterion, Key, KeyPart
from redpen.schemas.reading import ProblemReadings, Reading, ReadingRecord
from redpen.schemas.spec import CHOICE_TYPES
from redpen.workspace import Workspace, read_model, write_model

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
    record, by part id, with the digest of the reading it was graded from. A part
    read as unreadable gets none, and so does a choice part answered with no option
    read. Each top-level problem's file is replaced as soon as it is graded. The
    problems run concurrently; when one still fails its checks after one more call,
    `StageError` is raised once every other problem has finished, and that
    problem's earlier file, if any, is left as it was."""
    by_problem: defaultdict[str, list[KeyPart]] = defaultdict(list)
    for part in key.parts:
        by_problem[problem_of(part.id)].append(part)
    system, version = prompts.load("grade", "problem")
    # As `part_hash` digests them, so that the hash covers what the grader is sent.
    policies = json.dumps(key.policies.model_dump(), default=number)
    session = f"{workspace.root.name}:grade"
    done: dict[str, GradeRecord] = {}

    def readings(problem: str) -> dict[str, ReadingRecord]:
        file = read_model(workspace.reading_json(student, problem), ProblemReadings)
        return {r.reading.part_id: r for r in file.records}

    def provenance(part: KeyPart, by_model: bool) -> Provenance:
        return Provenance(
            key_part_hash=part_hash(part, key.policies),
            prompt_version=version if by_model else None,
            model_slug=model.slug if by_model else None,
        )

    async def grade(problem: str, parts: list[KeyPart]) -> None:
        mine = readings(problem)
        records: dict[str, GradeRecord] = {}
        asked: list[KeyPart] = []
        for part in parts:
            reading = mine[part.id].reading
            choice = part.response_type in CHOICE_TYPES
            # Withheld, never zero (design 9): work that cannot be read, and a choice
            # part answered with no option read, which no grader that knows the key
            # may settle (design 7.3).
            if reading.status == "unreadable" or (
                choice
                and reading.status == "answered"
                and not reading.final_answer.choices
            ):
                continue
            if reading.status == "blank" or choice:
                records[part.id] = GradeRecord(
                    provenance=provenance(part, False),
                    grade=by_code(part, reading),
                    check=None,
                    reading_digest=digest(mine[part.id]),
                )
            else:
                asked.append(part)
        if asked:
            # The same for every student, so that it is cached as a prefix.
            shared = [
                *(
                    grading_content(p)
                    for p in parts
                    if p.response_type not in CHOICE_TYPES
                ),
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

            async def ask(extra: Sequence[Content]) -> Grades:
                result = await gateway.call(
                    ROLE,
                    model,
                    system,
                    shared,
                    [*items, *extra],
                    Grades,
                    version,
                    params=PARAMS,
                    session_id=session,
                )
                return result.value

            found = await ask_checked(
                ask,
                lambda g: problems(key, wanted, g),
                f"the grade of {problem} of {student}",
            )
            by_id = {p.id: p for p in asked}
            for g in found.grades:
                allowed = {c.id: levels(c) for c in by_id[g.part_id].criteria}
                # Each award as the key writes the level the checks matched it to.
                criteria = [
                    c.model_copy(
                        update={
                            "awarded": next(x for x in allowed[c.id] if x == c.awarded)
                        }
                    )
                    for c in g.criteria
                ]
                # A reason that overrules no verdict is not an overrule.
                verdict = checks[g.part_id]
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
                records[g.part_id] = GradeRecord(
                    provenance=provenance(by_id[g.part_id], True),
                    grade=kept,
                    check=AnswerCheck(**verdict._asdict()) if verdict else None,
                    reading_digest=digest(mine[g.part_id]),
                )
        file = ProblemGrades(records=[records[p.id] for p in parts if p.id in records])
        write_model(workspace.grade_json(student, problem), file)
        done.update(records)

    await finish_all(grade(problem, parts) for problem, parts in by_problem.items())
    return {p.id: done[p.id] for p in key.parts if p.id in done}


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
    """Every way `found` fails the checks of design 7.5 for the parts `asked`, by
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
