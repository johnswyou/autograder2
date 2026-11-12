"""Grade one read student (design 7.5, first version: no computed answer check, no
second grader or referee, no second read or re-map). Code scores a part read blank
and a choice part; every other part of a top-level problem goes to the grader in one
call, and each award it returns is checked against the key. Code adds up points."""

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal

from redpen import prompts
from redpen.grade.read import problem_of
from redpen.llm.client import Gateway, ask_checked, finish_all
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.prepare.key import grading_content, number, part_hash
from redpen.schemas.common import Provenance, Schema
from redpen.schemas.grade import CriterionGrade, Grade, GradeRecord, ProblemGrades
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
    record, by part id. A part read as unreadable gets none, and so does a choice
    part answered with no option read. Each top-level problem's file is written as
    soon as it is graded. The problems run concurrently; when one still fails its
    checks after one more call, `StageError` is raised once every other problem has
    finished, and that problem is left with no file, not even an earlier one."""
    by_problem: defaultdict[str, list[KeyPart]] = defaultdict(list)
    for part in key.parts:
        by_problem[problem_of(part.id)].append(part)
    for problem in by_problem:
        workspace.grade_json(student, problem).unlink(missing_ok=True)
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
                    provenance=provenance(part, False), grade=by_code(part, reading)
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
            items: list[Content] = []
            for d in dict.fromkeys(d for p in asked for d in p.depends_on):
                if d not in wanted:
                    record = mine.get(d) or readings(problem_of(d))[d]
                    items.append(json.dumps({"context": _json(record.reading)}))
            for part in asked:
                record = mine[part.id]
                items.append(json.dumps({"grade": _json(record.reading)}))
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
                levels = {c.id: _levels(c) for c in by_id[g.part_id].criteria}
                # Each award as the key writes the level the checks matched it to.
                criteria = [
                    c.model_copy(
                        update={
                            "awarded": next(x for x in levels[c.id] if x == c.awarded)
                        }
                    )
                    for c in g.criteria
                ]
                # No computed answer check is given yet (task 4.4): nothing to overrule.
                kept = g.model_copy(
                    update={"criteria": criteria, "code_overruled": None}
                )
                records[g.part_id] = GradeRecord(
                    provenance=provenance(by_id[g.part_id], True), grade=kept
                )
        file = ProblemGrades(records=[records[p.id] for p in parts if p.id in records])
        write_model(workspace.grade_json(student, problem), file)
        done.update(records)

    await finish_all(grade(problem, parts) for problem, parts in by_problem.items())
    return {p.id: done[p.id] for p in key.parts if p.id in done}


def by_code(part: KeyPart, reading: Reading) -> Grade:
    """The grade code gives a part read blank (zero on every criterion) or a choice
    part (full points when the options read are the key's, compared without regard
    to case or surrounding space, else zero)."""
    if reading.status == "blank":
        earned, why = False, "No response."
    else:
        read = reading.final_answer.choices
        earned = _options(read) == _options(part.answer.choices)
        chosen, answer = ", ".join(read) or "nothing", ", ".join(part.answer.choices)
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
            levels = _levels(want)
            if c.awarded not in levels:
                wrong.append(
                    f"{c.id}: the award is one of {', '.join(map(number, levels))}"
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


def _levels(criterion: Criterion) -> list[Decimal]:
    """The awards `criterion` allows, as the key writes them: zero, its full points
    and its partial levels."""
    return sorted(
        {Decimal(0), criterion.points, *(x.points for x in criterion.partial)}
    )


def _options(labels: Sequence[str]) -> set[str]:
    return {label.strip().casefold() for label in labels}


def _json(reading: Reading) -> object:
    return reading.model_dump(mode="json")
