"""The class audit (design 7.6), run once every student of the class is graded. Two
of the design's three checks; the outlier pass is not built.

Same evidence, same award: for each criterion a model decided, the cases are
grouped by the folded text of the line or element the grader cited. A group whose
awards differ is decided again in one text-only call to the grader's model, its
cases numbered in the order of a digest of the student key; a case of a
carry-forward criterion carries the student's own final answers to the parts it
depends on. Left out: parts scored by code, withheld parts, grades that dispute the
reading, cases that cite nothing, and a final-answer criterion the grade stage held
to a decided answer check.

The key against the class, in code: of the students who state a final answer, the
answer check puts each on the key's side or the other; the other side is grouped
with `verify.compare`. When its largest group is larger than the key's side and
holds at least 40% of those who answered and at least five, the part is
`key_suspect`.

The result is `audit/<part>.json` for each key part. No grade file is touched: the
report stage applies a change only while the stored grade still has the part hash
and the award it was decided on."""

import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal
from typing import Literal

from pydantic import PositiveInt

from redpen import prompts
from redpen.grade.check import MATCH, NO_MATCH, OTHER_UNIT, check_answer, converts
from redpen.grade.grade import PARAMS, ROLE, finals, graded_from, levels, options
from redpen.llm.client import Gateway, ask_checked
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.prepare.key import grading_content, number, part_hash
from redpen.prepare.verify import AGREE, compare
from redpen.report.report import Stored, load
from redpen.schemas.audit import Change, PartAudit, Suspect
from redpen.schemas.common import Points, ResponseType, Schema
from redpen.schemas.grade import CriterionGrade
from redpen.schemas.key import Answer, Criterion, Key, KeyPart, Policies
from redpen.schemas.reading import FinalAnswer, Reading
from redpen.schemas.spec import CHOICE_TYPES
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, log_failure, write_model

SHARE = Decimal("0.4")  # of the students who answered
AT_LEAST = 5  # students
# Output tokens allowed per case of a group, above the grade stage's own limit:
# the longest answers measured took about 150 a case (task 4.7 report).
PER_CASE = 160
# The most cases one request decides: at PER_CASE tokens a case the answer stays in
# the grade stage's own token limit (8000), and no request grows with the class. A
# larger group is decided in parts of about the same size.
MAX_CASES = 50

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
    students: Sequence[IngestRecord],
    gateway: Gateway,
    model: ModelInfo,
) -> list[PartAudit]:
    """Audit the stored grades of `students` (the ingest's), write each key part's
    `audit/<part>.json` and return them in key order. A group whose call fails, or
    whose answer still fails its checks after one more call, keeps its awards: a
    fixed line and the traceback go to the run log, and it is counted as
    undecided."""
    # By a digest of the key, so that a case's number says nothing of whose it is
    # and the request is the same on every run.
    order = sorted(
        students, key=lambda s: hashlib.sha256(s.student_key.encode()).hexdigest()
    )
    stored = {s.student_key: load(workspace, key, s) for s in order}
    system, version = prompts.load("audit", "criterion")
    policies = json.dumps(key.policies.model_dump(), default=number)

    async def decide(part: KeyPart, hashed: str, cases: list[Case]) -> list[Change]:
        [criterion] = [c for c in part.criteria if c.id == cases[0][1].id]
        asked: list[dict[str, object]] = []
        for n, (s, g) in enumerate(cases, 1):
            mine = stored[s].readings
            case: dict[str, object] = {
                "case": n,
                "reading": mine[part.id].reading.model_dump(mode="json"),
            }
            if criterion.carry_forward:  # the student's own earlier results
                case["earlier"] = {
                    d: mine[d].reading.final_answer.model_dump(mode="json")
                    if d in mine
                    else None
                    for d in part.depends_on
                }
            asked.append(case | {"awarded": g.awarded, "rationale": g.rationale})
        text = json.dumps({"criterion": criterion.id, "cases": asked}, default=number)
        limit = max(PARAMS["max_tokens"], PER_CASE * len(cases))

        async def ask(extra: Sequence[Content]) -> Decisions:
            result = await gateway.call(
                ROLE,
                model,
                system,
                [grading_content(part), policies],
                [text, *extra],
                Decisions,
                version,
                params={**PARAMS, "max_tokens": limit},
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
        found = groups(part, hashed, list(stored.values()))
        # Each group in parts of at most MAX_CASES, in its order: (group, cases).
        asked = [(n, c) for n, cases in enumerate(found) for c in _parts(cases)]
        decided = await asyncio.gather(*(settle(part, hashed, c) for _, c in asked))
        readings = [s.readings.get(part.id) for s in stored.values()]
        audit = PartAudit(
            part_id=part.id,
            part_hash=hashed,
            groups=len(found),
            changed=len({n for (n, _), d in zip(asked, decided, strict=True) if d}),
            undecided=len(
                {n for (n, _), d in zip(asked, decided, strict=True) if d is None}
            ),
            split=sum(len(cases) > MAX_CASES for cases in found),
            changes=[c for d in decided if d for c in d],
            key_suspect=suspect(
                part, key.policies, [r.reading if r else None for r in readings]
            ),
        )
        write_model(workspace.audit_json(part.id), audit)
        audits.append(audit)
    return audits


def groups(part: KeyPart, hashed: str, students: Sequence[Stored]) -> list[list[Case]]:
    """The groups of cases of `part` whose awards differ though they are on the same
    criterion and cite the same evidence: the text of the line and of the element
    cited, white space and letter case folded. A group keeps the order of
    `students`. Left out: a student with no reading or grade of the part, a grade
    under another version of it (`hashed` is the current one) or from another
    reading than the one stored; a part scored
    by code; a grade that disputes the reading, since the audit sees only the
    reading; a case that cites nothing; and a final-answer criterion held to a
    decided answer check, as the grade stage holds a part with exactly one (code
    decided those alike)."""
    criteria = {c.id: c for c in part.criteria}
    found: defaultdict[tuple[object, ...], list[Case]] = defaultdict(list)
    for student in students:
        record, read = student.grades.get(part.id), student.readings.get(part.id)
        if (
            record is None
            or read is None
            or record.provenance.key_part_hash != hashed
            or record.reading_digest not in (None, graded_from(part, student.readings))
            or record.provenance.model_slug is None  # scored by code
            or (record.grade.transcript_dispute or "").strip()
        ):
            continue
        held = (
            finals(part) == 1
            and record.check is not None
            and record.check.verdict in (MATCH, NO_MATCH)
        )
        for grade in record.grade.criteria:
            criterion = criteria.get(grade.id)
            cited = _cited(read.reading, grade)
            if criterion is None or cited == (None, None):
                continue
            if held and criterion.kind == "final_answer":
                continue
            found[criterion.id, cited].append((student.student.student_key, grade))
    return [cases for cases in found.values() if len({g.awarded for _, g in cases}) > 1]


def _parts(cases: list[Case]) -> list[list[Case]]:
    """`cases` in as few runs of at most MAX_CASES as can be, of about one size."""
    size = math.ceil(len(cases) / math.ceil(len(cases) / MAX_CASES))
    return [cases[i : i + size] for i in range(0, len(cases), size)]


def problems(criterion: Criterion, cases: int, found: Decisions) -> list[str]:
    """Every way `found` fails to decide each of the `cases` cases once, with a
    reason and a listed level of `criterion`, in a fixed order (it is sent back to
    the model). Only case numbers and the key's levels are named."""
    seen = Counter(d.case for d in found.decisions)
    wrong = [f"case {n} has no decision" for n in range(1, cases + 1) if not seen[n]]
    wrong += [
        f"case {n} is decided {seen[n]} times"
        for n in range(1, cases + 1)
        if seen[n] > 1
    ]
    if any(n > cases for n in seen):
        wrong.append("a decision is for a case that is not in the request")
    wrong += [
        f"case {d.case}: the reason is empty"
        for d in found.decisions
        if not d.reason.strip()
    ]
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
    has none), for a part whose final answer code can compare. Those who answered
    are the students whose answered reading states a final answer of the part's
    kind, a missing unit taken as the key's. The answer check, under the
    assignment's policies, puts each on the key's side or the other (see `_side`);
    the other side is grouped by `compare`, two answers being the same when each
    matches the other. A `Suspect` when its largest group is larger than the key's
    side and holds at least `SHARE` of those who answered and at least `AT_LEAST`
    students, else None."""
    kind = kind_of(part)
    if kind is None:
        return None
    unit = part.answer.unit if kind == "numeric" else None
    answered = [
        r.model_copy(update={"final_answer": _with_unit(r.final_answer, unit)})
        for r in readings
        if r is not None and r.status == "answered" and gives(kind, r.final_answer)
    ]
    ours, others = 0, list[tuple[Answer, int]]()
    for reading in answered:
        side = _side(part, kind, reading, policies)
        if side == "key":
            ours += 1
        elif side == "other":
            said = as_answer(kind, reading.final_answer)
            same = [
                i
                for i, (first, _) in enumerate(others)
                if compare(kind, first, said, policies) == AGREE
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


def _side(
    part: KeyPart, kind: ResponseType, reading: Reading, policies: Policies
) -> Literal["key", "other"] | None:
    """Which side of the key check a stated answer is on, as the grade stage would
    judge it: options as `by_code` compares them; else the answer check under the
    assignment's policies, units not required. "Cannot tell" is on neither side,
    except a number in a unit read and convertible to the key's that does not match
    after conversion."""
    stated = reading.final_answer
    if kind in CHOICE_TYPES:
        same = options(stated.choices) == options(part.answer.choices)
        return "key" if same else "other"
    found = check_answer(
        part, reading, policies.model_copy(update={"require_units": False})
    )
    if found is None:
        return None
    if found.verdict == MATCH:
        return "key"
    if found.verdict == NO_MATCH:
        return "other"
    convertible = converts(stated.unit or "", part.answer.unit or "")
    return "other" if found.reason == OTHER_UNIT and convertible else None


def gives(kind: ResponseType, stated: FinalAnswer) -> bool:
    """Whether `stated` gives a final answer of `kind`."""
    if kind == "numeric":
        return stated.value is not None
    if kind == "symbolic":
        return bool((stated.expression or "").strip())
    return bool(stated.choices)


def _with_unit(stated: FinalAnswer, unit: str | None) -> FinalAnswer:
    """`stated` with the key's `unit` where it states none."""
    if (stated.unit or "").strip() or not unit:
        return stated
    return stated.model_copy(update={"unit": unit})


def _cited(
    reading: Reading, grade: CriterionGrade
) -> tuple[str | None, tuple[str, ...] | None]:
    """The folded text of the line and of the element `grade` cites, each None when
    it cites none the reading has."""
    line = element = None
    if (n := grade.evidence_line) and n <= len(reading.lines):
        line = fold(reading.lines[n - 1])
    if (n := grade.evidence_element) and n <= len(reading.elements):
        e = reading.elements[n - 1]
        element = tuple(
            fold(x or "") for x in (e.label, e.kind, e.direction, e.position)
        )
    return line, element


def fold(text: str) -> str:
    return " ".join(text.split()).casefold()


def kind_of(part: KeyPart) -> ResponseType | None:
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


def as_answer(kind: ResponseType, stated: FinalAnswer) -> Answer:
    """A student's `stated` answer as `compare` takes it: an answer with only the
    fields a `kind` answer fills (it makes a key part of it, which refuses any
    other), its options folded as the grade stage folds them."""
    numeric, symbolic = kind == "numeric", kind == "symbolic"
    return Answer(
        choices=[] if numeric or symbolic else sorted(options(stated.choices)),
        value=stated.value if numeric else None,
        unit=stated.unit if numeric else None,
        rel_tol=None,
        accept_also=[],
        expression=stated.expression if symbolic else None,
        elements=[],
        must_include=[],
    )
