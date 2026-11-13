"""The class audit (design 7.6), run once every student of the class is graded. Two
of the design's three checks; the outlier pass is not built.

Same evidence, same award: for each criterion a model decided, the cases are
grouped by the folded text of the line or element the grader cited and, for a
carry-forward criterion, the student's final answers to the parts it depends on. A
group whose awards differ is decided again in one text-only call to the grader's
model, its cases numbered in the order of a digest of the student key. Left out:
parts scored by code, withheld parts, cases that cite nothing, and a final-answer
criterion the grade stage held to a decided answer check.

The key against the class, in code: the students who answered are grouped by final
answer with `verify.compare`. When the most common answer is not the key's and is
shared by at least 40% of them and at least five, the part is `key_suspect`.

The result is `audit/<part>.json` for each key part. No grade file is touched: the
report stage applies a change only while the stored grade still has the part hash
and the award it was decided on."""

import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal

from pydantic import PositiveInt

from redpen import prompts
from redpen.grade.check import UNSURE
from redpen.grade.grade import PARAMS, ROLE, finals, levels, options
from redpen.grade.read import problem_of
from redpen.llm.client import Gateway, ask_checked
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.prepare.key import grading_content, number, part_hash
from redpen.prepare.verify import AGREE, DIFFER, compare
from redpen.schemas.audit import Change, PartAudit, Suspect
from redpen.schemas.common import Points, ResponseType, Schema
from redpen.schemas.grade import CriterionGrade, GradeRecord, ProblemGrades
from redpen.schemas.key import Answer, Criterion, Key, KeyPart, Policies
from redpen.schemas.reading import ProblemReadings, Reading
from redpen.schemas.spec import CHOICE_TYPES
from redpen.workspace import Workspace, log_failure, read_model, write_model

SHARE = Decimal("0.4")  # of the students who answered
AT_LEAST = 5  # students

type Case = tuple[str, CriterionGrade]  # a student key and their grade of a criterion


# What the model returns for one group: a reason and then an award for each case.
class Decision(Schema):
    case: PositiveInt
    reason: str
    awarded: Points


class Decisions(Schema):
    decisions: list[Decision]


async def audit_class(
    workspace: Workspace,
    key: Key,
    students: Sequence[str],
    gateway: Gateway,
    model: ModelInfo,
) -> list[PartAudit]:
    """Audit the stored grades of `students` (keys of ingested students), write each
    key part's `audit/<part>.json` and return them in key order. A group whose call
    fails, or whose answer still fails its checks after one more call, keeps its
    awards: a fixed line and the traceback go to the run log, and it is counted as
    undecided."""
    # By a digest of the key, so that a case's number says nothing of whose it is
    # and the request is the same on every run.
    order = sorted(students, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    readings, grades = _stored(workspace, key, order)
    system, version = prompts.load("audit", "criterion")
    policies = json.dumps(key.policies.model_dump(), default=number)

    async def decide(part: KeyPart, hashed: str, cases: list[Case]) -> list[Change]:
        [criterion] = [c for c in part.criteria if c.id == cases[0][1].id]
        asked: dict[str, object] = {
            "criterion": criterion.id,
            "cases": [
                {
                    "case": n,
                    "reading": readings[s][part.id].model_dump(mode="json"),
                    "awarded": g.awarded,
                    "rationale": g.rationale,
                }
                for n, (s, g) in enumerate(cases, 1)
            ],
        }
        if criterion.carry_forward:  # the same for every case: it is in the group key
            mine = readings[cases[0][0]]
            asked["earlier"] = {d: _final(mine.get(d)) for d in part.depends_on}
        text = json.dumps(asked, default=number)

        async def ask(extra: Sequence[Content]) -> Decisions:
            result = await gateway.call(
                ROLE,
                model,
                system,
                [grading_content(part), policies],
                [text, *extra],
                Decisions,
                version,
                params=PARAMS,
                session_id=f"{workspace.root.name}:audit",
            )
            return result.value

        found = await ask_checked(
            ask,
            lambda d: problems(criterion, len(cases), d),
            f"the decision of {criterion.id}",
        )
        after = {d.case: d for d in found.decisions}
        return [
            Change(
                student_key=s,
                criterion=criterion.id,
                part_hash=hashed,
                before=g.awarded,
                # As the key writes the level the checks matched it to.
                after=next(x for x in levels(criterion) if x == after[n].awarded),
                reason=after[n].reason,
            )
            for n, (s, g) in enumerate(cases, 1)
            if after[n].awarded != g.awarded
        ]

    async def settle(
        part: KeyPart, hashed: str, cases: list[Case]
    ) -> list[Change] | None:
        try:
            return await decide(part, hashed, cases)
        except Exception as error:  # any failure: the awards stand
            what = f"audit {cases[0][1].id}: a group was not decided again"
            log_failure(workspace, what, error)
            return None

    audits: list[PartAudit] = []
    for part in key.parts:
        hashed = part_hash(part, key.policies)
        found = groups(part, hashed, readings, grades)
        decided = await asyncio.gather(*(settle(part, hashed, c) for c in found))
        done = [d for d in decided if d is not None]
        audit = PartAudit(
            part_id=part.id,
            part_hash=hashed,
            groups=len(found),
            changed=sum(1 for d in done if d),
            undecided=len(decided) - len(done),
            changes=[c for d in done for c in d],
            key_suspect=suspect(
                part, key.policies, [readings[s].get(part.id) for s in order]
            ),
        )
        write_model(workspace.audit_json(part.id), audit)
        audits.append(audit)
    return audits


def groups(
    part: KeyPart,
    hashed: str,
    readings: Mapping[str, Mapping[str, Reading]],
    grades: Mapping[str, Mapping[str, GradeRecord]],
) -> list[list[Case]]:
    """The groups of cases of `part` whose awards differ though they are on the same
    criterion, cite the same evidence (the text of the line and of the element
    cited, white space and letter case folded) and, for a carry-forward criterion,
    come with the same final answers to the parts `part` depends on. A group keeps
    the order of `readings`. Left out: a student with no reading or grade of the
    part, or a grade under another version of it (`hashed` is the current one); a
    part scored by code; a case that cites nothing; and a final-answer criterion
    held to a decided answer check, as the grade stage holds a part with exactly one
    (code decided those alike)."""
    criteria = {c.id: c for c in part.criteria}
    found: defaultdict[tuple[object, ...], list[Case]] = defaultdict(list)
    for student, mine in readings.items():
        record, reading = grades[student].get(part.id), mine.get(part.id)
        if (
            record is None
            or reading is None
            or record.provenance.key_part_hash != hashed
            or record.provenance.model_slug is None  # scored by code
        ):
            continue
        held = (
            finals(part) == 1
            and record.check is not None
            and record.check.verdict != UNSURE
        )
        for grade in record.grade.criteria:
            criterion = criteria.get(grade.id)
            cited = _cited(reading, grade)
            if criterion is None or cited == (None, None):
                continue
            if held and criterion.kind == "final_answer":
                continue
            earlier = (
                json.dumps([_final(mine.get(d)) for d in part.depends_on])
                if criterion.carry_forward
                else None
            )
            found[criterion.id, cited, earlier].append((student, grade))
    return [cases for cases in found.values() if len({g.awarded for _, g in cases}) > 1]


def problems(criterion: Criterion, cases: int, found: Decisions) -> list[str]:
    """Every way `found` fails to decide each of the `cases` cases once with a
    listed level of `criterion`, in a fixed order (it is sent back to the model).
    Only case numbers and the key's levels are named."""
    seen = Counter(d.case for d in found.decisions)
    wrong = [f"case {n} has no decision" for n in range(1, cases + 1) if not seen[n]]
    wrong += [
        f"case {n} is decided {seen[n]} times"
        for n in range(1, cases + 1)
        if seen[n] > 1
    ]
    if any(n > cases for n in seen):
        wrong.append("a decision is for a case that is not in the request")
    allowed = levels(criterion)
    wrong += [
        f"case {d.case}: the award is one of {', '.join(map(number, allowed))}"
        for d in found.decisions
        if d.awarded not in allowed
    ]
    return list(dict.fromkeys(wrong))


def suspect(
    part: KeyPart, policies: Policies, readings: Sequence[Reading | None]
) -> Suspect | None:
    """The key check of `part` against the class's `readings` (None where a student
    has none), for a part whose final answer code can compare. The students who
    answered (not blank, not unreadable) are grouped by final answer, two answers
    being the same when `compare` finds each matches the other; an answer it cannot
    tell from the key's counts for neither side. A `Suspect` when the largest group
    is larger than the key's and holds at least `SHARE` of those who answered and
    at least `AT_LEAST` students, else None."""
    kind = _kind(part)
    if kind is None:
        return None
    # A missing unit is taken as the key's: the question is whether the class's
    # numbers agree with the key's, not whether units were written.
    lenient = policies.model_copy(update={"require_units": False})
    mine = _as(kind, part.answer)
    answered = [r for r in readings if r is not None and r.status == "answered"]
    ours, others = 0, list[tuple[Answer, int]]()
    for reading in answered:
        said = _as(
            kind,
            Answer(
                **reading.final_answer.model_dump(),
                rel_tol=None,
                accept_also=[],
                elements=[],
                must_include=[],
            ),
        )
        verdict = compare(kind, mine, said, lenient)
        if verdict == AGREE:
            ours += 1
        elif verdict == DIFFER:
            same = [
                i
                for i, (first, _) in enumerate(others)
                if compare(kind, first, said, lenient) == AGREE
            ]
            if same:
                first, n = others[same[0]]
                others[same[0]] = (first, n + 1)
            else:
                others.append((said, 1))
    shared = max((n for _, n in others), default=0)
    if shared > ours and shared >= AT_LEAST and shared >= SHARE * len(answered):
        return Suspect(shared=shared, answered=len(answered))
    return None


def _stored(
    workspace: Workspace, key: Key, students: Sequence[str]
) -> tuple[dict[str, dict[str, Reading]], dict[str, dict[str, GradeRecord]]]:
    """Each student's stored readings and grade records, by part id, in the order of
    `students`."""
    readings: dict[str, dict[str, Reading]] = {}
    grades: dict[str, dict[str, GradeRecord]] = {}
    for s in students:
        readings[s], grades[s] = {}, {}
        for problem in dict.fromkeys(problem_of(p.id) for p in key.parts):
            if (path := workspace.reading_json(s, problem)).exists():
                found = read_model(path, ProblemReadings).records
                readings[s] |= {r.reading.part_id: r.reading for r in found}
            if (path := workspace.grade_json(s, problem)).exists():
                graded = read_model(path, ProblemGrades).records
                grades[s] |= {r.grade.part_id: r for r in graded}
    return readings, grades


def _cited(
    reading: Reading, grade: CriterionGrade
) -> tuple[str | None, tuple[str, ...] | None]:
    """The folded text of the line and of the element `grade` cites, each None when
    it cites none the reading has."""
    line = element = None
    if (n := grade.evidence_line) and n <= len(reading.lines):
        line = _fold(reading.lines[n - 1])
    if (n := grade.evidence_element) and n <= len(reading.elements):
        e = reading.elements[n - 1]
        element = tuple(
            _fold(x or "") for x in (e.label, e.kind, e.direction, e.position)
        )
    return line, element


def _fold(text: str) -> str:
    return " ".join(text.split()).casefold()


def _final(reading: Reading | None) -> object:
    return reading.final_answer.model_dump(mode="json") if reading else None


def _kind(part: KeyPart) -> ResponseType | None:
    """The response type `compare` takes `part`'s answers as, as the answer check
    reads them: a choice part's own; numeric for a value; symbolic for an
    expression; None when code cannot compare them."""
    if part.response_type in CHOICE_TYPES:
        return part.response_type
    if part.answer.value is not None:
        return "numeric"
    if part.answer.expression is not None:
        return "symbolic"
    return None


def _as(kind: ResponseType, answer: Answer) -> Answer:
    """`answer` with only the fields an answer of `kind` fills (`compare` makes a key
    part of it, which refuses any other), and its options folded as the grade stage
    folds them."""
    numeric, symbolic = kind == "numeric", kind == "symbolic"
    return Answer(
        choices=[] if numeric or symbolic else sorted(options(answer.choices)),
        value=answer.value if numeric else None,
        unit=answer.unit if numeric else None,
        rel_tol=answer.rel_tol if numeric else None,
        accept_also=answer.accept_also if numeric else [],
        expression=answer.expression if symbolic else None,
        elements=[],
        must_include=[],
    )
