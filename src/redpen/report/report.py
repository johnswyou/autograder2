"""Report, first version (design 8 and 9): the three score files, one feedback file
per student, the flags and the run report, computed by code from the stored
readings and grades and the instructor's corrections (design 11). No model call;
every point value is a Decimal.

The caller supplies the students. The `students/` directory is never listed: a
respelled name leaves an old folder behind."""

import csv
import hashlib
import re
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ValidationError

from redpen.config import Profile, read_lock
from redpen.grade.coverage import (
    set_aside_section,
    unclaimed_flags,
    unclaimed_lines,
    unclaimed_shown,
)
from redpen.grade.grade import graded_from, levels, part_points
from redpen.grade.map import name_counts
from redpen.grade.read import digest, problem_of, unsure
from redpen.grade.second_grade import PROFILES, differing
from redpen.imaging import render
from redpen.llm.ledger import Ledger, Usage
from redpen.prepare.key import (
    allocation,
    number,
    page_points,
    part_hash,
    scaled,
    scoring,
)
from redpen.report.overrides import load_overrides, load_reviews
from redpen.schemas.audit import PartAudit
from redpen.schemas.common import Provenance
from redpen.schemas.flags import SEVERITY, Flag, FlagCode, Severity
from redpen.schemas.grade import (
    CriterionGrade,
    Grade,
    GradeRecord,
    Override,
    ProblemGrades,
    Review,
    SecondGrade,
    SecondGrades,
)
from redpen.schemas.key import Key, KeyPart
from redpen.schemas.reading import (
    FinalAnswer,
    ProblemReadings,
    ReadingRecord,
    SecondReading,
    SecondReadings,
)
from redpen.schemas.spec import CHOICE_TYPES
from redpen.schemas.submission import IngestRecord, MapRecord
from redpen.workspace import Workspace, read_model

type Status = Literal["incomplete", "provisional", "final"]

# The five reasons a part is withheld, never scored zero (design 9), in fixed words.
UNREADABLE = "The work could not be read."
NO_OPTION = "No chosen option could be read."
OTHER_KEY = "It was graded under another version of the key and is to be graded again."
OTHER_READING = (
    "It was graded from another reading of the work and is to be graded again."
)
UNGRADED = "No reading or no grade was recorded for it."
# Before the requirement of a criterion the class audit lowered from its full
# points, in place of whatever the grader wrote then (often praise): the student's
# file never shows the audit's own reason.
LOWERED = "This point was marked again for consistency across the class. What it needs:"
# The instructor's corrections, in fixed words: the mark of a corrected award in
# scores_detail.csv; the student's line in its place, before the note, with the
# criterion's own points only when no constraint scales them (the part's line gives
# the award); what a part still withheld lacks, for the instructor's files only;
# and why a correction made before its part was edited is not applied.
CORRECTED = "Corrected by the instructor."
MARKED = "Marked by your instructor."
MARKED_POINTS = "Marked by your instructor: {points} / {full}."
OPEN = "Criteria with no correction from the instructor: {open} of {all}."
STALE = (
    "The key's part of each criterion below was edited after its correction was"
    " recorded, so the correction is not applied. Record it again with `redpen"
    " override` if it still holds."
)
# The instructor's reviews, in fixed words: why one is listed as out of date, and
# what the review list's last section holds.
OUTDATED = (
    "What each entry below shows has changed since it was reviewed, so its review"
    " closes none of its flags. `redpen review` asks about it again."
)
CLOSED = (
    "Each entry below was reviewed, and its flags are closed while it shows the same."
)
# Why a file is in the run report's list of files not ingested, in fixed words.
NOT_INGESTED = "It could not be ingested, so nothing in it is graded."
# Under the pinned models, once a grade run of the workspace had strict privacy off.
STRICT_OFF_RUN = (
    "Strict privacy was off for a grade run of this workspace: the endpoints were"
    " not checked, and student pages may have gone to providers that keep them."
    " This stays recorded; only a fresh workspace clears it."
)

# The other flags' details, in fixed words: never text a student or a model wrote.
DETAIL = {
    FlagCode.KEY_UNVERIFIED: "The key's answer or rubric for this part is not"
    " verified, so every student's score on it is provisional.",
    FlagCode.HARD_TO_READ: "The reader was unsure of some of the writing.",
    FlagCode.READERS_DISAGREE: "A second reader, another vendor's model, read this"
    " part differently. The score comes from the first reading.",
    FlagCode.GRADERS_DISAGREE: "Graded a second time, by the same grader with the"
    " same request, this part got another award. The score comes from the first"
    " grade.",
    FlagCode.LOW_RESOLUTION: "The work was read at the full resolution of the"
    " source and is still hard to read.",
    FlagCode.ADDRESSED_TO_GRADER: "The response holds text addressed to the grader.",
    FlagCode.TRANSCRIPT_DISPUTE: "The grader believes the reading is wrong.",
    FlagCode.CODE_OVERRULED: "The grader overruled the computed answer check.",
    FlagCode.KEY_SUSPECT: "The class's most common answer to this part is not the"
    " key's: check the key.",
    FlagCode.AUDIT_CHANGED: "The class audit changed an award on this part.",
}
# Why a part's work may be another part's response (`weak_attribution`): the map's
# doubt, the grader's, or both, in this order.
WEAK_MAP = (
    "The map placed this work on its content alone or with low confidence, so it"
    " may answer another part."
)
WEAK_GRADER = "The grader reported that this work answers another part."

# The whole review list when no flag is open, after its title.
NOTHING_TO_REVIEW = "Nothing needs review: no flag is open."
SEVERITIES: tuple[Severity, ...] = get_args(Severity)

# A spreadsheet runs a cell that begins with one of these as a formula.
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")

NO_USE = Usage(0, 0, Decimal(0))


@dataclass(frozen=True)
class Stored:
    """One student's ingest record, and each part's reading and grade record that
    are stored, by part id; the class audit of each part, the same for every
    student; each part's second reading compared with the stored first, and its
    second grade compared with the stored first; the instructor's corrections
    made under the key's current version of their part, by criterion id; and the
    stored map."""

    student: IngestRecord
    readings: dict[str, ReadingRecord]
    grades: dict[str, GradeRecord]
    audits: Mapping[str, PartAudit] = field(default_factory=dict[str, PartAudit])
    seconds: Mapping[str, SecondReading] = field(
        default_factory=dict[str, SecondReading]
    )
    second_grades: Mapping[str, SecondGrade] = field(
        default_factory=dict[str, SecondGrade]
    )
    overrides: Mapping[str, Override] = field(default_factory=dict[str, Override])
    mapped: MapRecord | None = None


@dataclass(frozen=True)
class Score:
    """One part of one student: the grade record that counts and the points it
    awards on the key's scale, or, when the part is withheld, why; and the
    instructor's corrections of its criteria, by criterion id."""

    part: KeyPart
    record: GradeRecord | None
    awarded: Decimal | None
    withheld: str | None
    overrides: Mapping[str, Override]


# An entry of the review list: a student key and a part id; no student for a key
# part, and no part for a whole submission.
type Place = tuple[str | None, str | None]


@dataclass(frozen=True)
class Derived:
    """What the report files are made from: each student's stored files and the
    score of each part; every flag raised; the corrections not applied; the review
    that would close each flagged entry as it stands, by place; the places a stored
    review closes; and the stored reviews of a flagged entry that no longer hold."""

    scored: list[tuple[Stored, list[Score]]]
    raised: list[Flag]
    stale: list[Override]
    current: dict[Place, Review]
    reviewed: set[Place]
    outdated: list[Review]

    @property
    def open(self) -> list[Flag]:
        """The flags no review closes."""
        return [
            f for f in self.raised if (f.student_key, f.part_id) not in self.reviewed
        ]

    @property
    def closed(self) -> list[Flag]:
        """The flags a review closes."""
        return [f for f in self.raised if (f.student_key, f.part_id) in self.reviewed]


def derive(workspace: Workspace, key: Key, students: Sequence[IngestRecord]) -> Derived:
    """What is stored for `students`, scored and flagged. The corrections in
    `overrides.yaml` are applied, or listed as stale; a review in `reviews.yaml`
    closes the flags of its entry while the entry shows what it was made for, or is
    listed as out of date. A file of them that cannot be loaded raises
    `OverrideError`."""
    keys = [s.student_key for s in students]
    corrections = load_overrides(workspace.overrides_yaml, key, keys)
    reviews = load_reviews(workspace.reviews_yaml, key, keys)
    hashes = {c.id: part_hash(p, key.policies) for p in key.parts for c in p.criteria}
    stale = [o for o in corrections if o.part_hash != hashes[o.criterion]]
    stored = [
        replace(
            load(workspace, key, s),
            overrides={
                o.criterion: o
                for o in corrections
                if o.student == s.student_key and o not in stale
            },
        )
        for s in students
    ]
    raised = flags(key, stored) + unclaimed_flags(workspace, students)
    scored = [(s, score(key, s)) for s in stored]
    current = {
        place: _shown(workspace, key, scored, place, found)
        for place, found in _grouped(raised).items()
    }
    reviewed = {
        (r.student, r.part) for r in reviews if current.get((r.student, r.part)) == r
    }
    outdated = [
        r
        for r in reviews
        if (r.student, r.part) in current and (r.student, r.part) not in reviewed
    ]
    return Derived(scored, raised, stale, current, reviewed, outdated)


def _shown(
    workspace: Workspace,
    key: Key,
    scored: Sequence[tuple[Stored, list[Score]]],
    place: Place,
    found: Sequence[Flag],
) -> Review:
    """The review that closes the entry at `place`, with flags `found`, as it
    stands: its key part's hash, and the digest of what it shows that a later run
    could change. That is its flag codes; for a key part, the class audit's finding
    that the class disputes the key's answer; for a student's part, its stored
    reading, the bytes of each crop the entry links, its second reading, its second
    grade, its grade after the class audit, why it is withheld, and the
    instructor's corrections of it (with the grade, they give the grade after the
    corrections); for a whole submission, each flag's cause, the bytes of each
    image the entry or the list of blocks set aside links, and the map."""
    who, part_id = place
    part = next((p for p in key.parts if p.id == part_id), None)
    shown: list[BaseModel | str | None] = [*sorted({f.code for f in found})]
    if who is None and part_id is not None:
        audit = scored[0][0].audits.get(part_id) if scored else None
        shown.append(audit.key_suspect if audit else None)
    elif who is not None and part is not None and (s := _score_at(scored, place)):
        [stored] = [st for st, _ in scored if st.student.student_key == who]
        base = None if withheld(key, part, stored) else audited(stored, part)
        first = stored.readings.get(part.id)
        crops = _crops(workspace, who, first) if first else []
        shown += [
            first,
            # The images themselves: a new PDF can be read and graded the same.
            *(_bytes(p) for p in crops),
            stored.seconds.get(part.id),
            stored.second_grades.get(part.id),
            base,
            s.withheld,
            *(s.overrides[c] for c in sorted(s.overrides)),
        ]
    elif who is not None and part_id is None:  # a whole submission
        [student] = [st.student for st, _ in scored if st.student.student_key == who]
        mapped, images = unclaimed_shown(workspace, student)
        shown += [*(f.detail for f in found), *(_bytes(p) for _, p in images), mapped]
    material = " ".join(
        digest(x) if isinstance(x, BaseModel) else str(x) for x in shown
    )
    return Review(
        student=who,
        part=part_id,
        part_hash=None if part is None else part_hash(part, key.policies),
        digest=hashlib.sha256(material.encode()).hexdigest()[:12],
    )


def _bytes(path: Path) -> str | None:
    """The sha256 of the file at `path`, or None when there is none."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_report(
    workspace: Workspace,
    key: Key,
    students: Sequence[IngestRecord],
    not_ingested: Sequence[str] = (),
) -> list[Flag]:
    """Write `out/` for `students`, in the order given: the three score files, one
    feedback file per student (any other is removed), the review list and the run
    report, which also lists the files of `not_ingested` (names inside the
    submissions, never opened here). What is written is `derive`'s: a file of
    corrections or reviews that cannot be loaded raises `OverrideError`. Returns
    the open flags."""
    problems, pages = columns(workspace, key)
    found = derive(workspace, key, students)
    scored, raised = found.scored, found.open
    maximum = points_of(key, key.parts)
    title = workspace.root.name.removesuffix(".redpen")
    by_problem: list[list[Cell]] = []
    by_page: list[list[Cell]] = []
    detail: list[list[Cell]] = []
    feedback: list[tuple[Path, str]] = []
    for s, scores in scored:
        name, state = s.student.student_name, status(key, s, raised)
        mine = [f for f in raised if f.student_key in (None, s.student.student_key)]
        total = _total(scores)
        by_problem.append(
            [
                name,
                *(
                    sum_of(x for x in scores if problem_of(x.part.id) == q)
                    for q in problems
                ),
                total,
                maximum,
                state,
                len(mine),
            ]
        )
        by_page.append(
            [name, *(sum_of(x for x in scores if x.part.home_page == n) for n in pages)]
        )
        for x in scores:
            got = {c.id: c for c in x.record.grade.criteria} if x.record else {}
            for c in x.part.criteria:
                row: list[Cell] = [name, x.part.id, c.id, None, c.points, x.withheld]
                if x.record:
                    row[3], row[5] = got[c.id].awarded, got[c.id].rationale
                detail.append(row)
        text = _feedback(key, title, problems, scores, total, maximum)
        feedback.append((workspace.feedback_md(s.student.student_key), text))
    run_report = _run_report(workspace, key, pages, found, not_ingested)
    review_md = review(
        workspace, key, scored, raised, found.stale, found.closed, found.outdated
    )
    review_md += set_aside_section(workspace, students)
    # Everything is rendered before any old file is removed.
    workspace.feedback_dir.mkdir(parents=True, exist_ok=True)
    for old in workspace.feedback_dir.glob("*.md"):
        old.unlink()
    for path, text in feedback:
        path.write_text(text, encoding="utf-8")
    _write_csv(
        workspace.scores_by_problem_csv,
        ["student", *problems, "total", "maximum", "status", "open_flags"],
        by_problem,
    )
    _write_csv(
        workspace.scores_by_page_csv,
        ["student", *(f"page {n}" for n in pages)],
        by_page,
    )
    _write_csv(
        workspace.scores_detail_csv,
        ["student", "part", "criterion", "awarded", "points", "rationale"],
        detail,
    )
    workspace.review_md.write_text(review_md, encoding="utf-8")
    workspace.run_report_md.write_text(run_report, encoding="utf-8")
    return raised


def columns(workspace: Workspace, key: Key) -> tuple[list[str], list[int]]:
    """The score files' columns: the key's top-level problems in key order, and the
    pages from 1 to the blank's last. Opens the blank (pypdfium2)."""
    with render.open_pdf(workspace.blank_pdf) as pdf:
        last = len(pdf)
    # A part moved by hand past the blank's last page keeps its points on a column.
    pages = list(range(1, max([last, *(p.home_page for p in key.parts)]) + 1))
    return list(dict.fromkeys(problem_of(p.id) for p in key.parts)), pages


def load(workspace: Workspace, key: Key, student: IngestRecord) -> Stored:
    """What is stored for `student`: the readings and grade records of each of the
    key's top-level problems whose file exists, each second reading made for the
    first reading stored now and the key's version of its part, and each second
    grade made for the first grade stored now, the readings it was graded from and
    the key's version of its part, that grades each of the first's criteria; and
    its map, if one is stored."""
    readings: dict[str, ReadingRecord] = {}
    grades: dict[str, GradeRecord] = {}
    who = student.student_key
    for problem in dict.fromkeys(problem_of(p.id) for p in key.parts):
        if found := _read(workspace.reading_json(who, problem), ProblemReadings):
            readings |= {r.reading.part_id: r for r in found.records}
        if graded := _read(workspace.grade_json(who, problem), ProblemGrades):
            grades |= {r.grade.part_id: r for r in graded.records}
    try:
        again = _read(workspace.second_json(who), SecondReadings)
    except ValidationError:  # an earlier shape: made for nothing stored now
        again = None
    try:
        regraded = _read(workspace.second_grade_json(who), SecondGrades)
    except ValidationError:  # not a file of this shape: made for nothing stored now
        regraded = None
    hashes = {p.id: part_hash(p, key.policies) for p in key.parts}
    parts = {p.id: p for p in key.parts}
    seconds = {
        s.part_id: s
        for s in (again.records if again else [])
        if s.part_id in readings
        and s.first == digest(readings[s.part_id])
        and s.part_hash == hashes.get(s.part_id)
    }
    second_grades = {
        s.part_id: s
        for s in (regraded.records if regraded else [])
        if s.part_id in grades
        and s.part_id in readings
        and s.part_hash == hashes.get(s.part_id)
        and s.first == digest(grades[s.part_id])
        and s.reading_digest == graded_from(parts[s.part_id], readings)
        # Only by hand can a second grade lack one of the part's criteria.
        and (
            s.second is None
            or {c.id for c in s.second.grade.criteria}
            == {c.id for c in grades[s.part_id].grade.criteria}
        )
    }
    return Stored(
        student,
        readings,
        grades,
        audits(workspace, key),
        seconds,
        second_grades,
        mapped=_read(workspace.map_json(who), MapRecord),
    )


def audits(workspace: Workspace, key: Key) -> dict[str, PartAudit]:
    """The class audit of each key part, by part id: each stored audit file made
    under the part's current version."""
    found = {p.id: _read(workspace.audit_json(p.id), PartAudit) for p in key.parts}
    return {
        p.id: audit
        for p in key.parts
        if (audit := found[p.id]) and audit.part_hash == part_hash(p, key.policies)
    }


def withheld(key: Key, part: KeyPart, stored: Stored) -> str | None:
    """Why `part` is not scored for the stored student, or None when it is: its
    work cannot be read; it is a choice part answered with no option read; it has
    no reading or no grade record; its record was graded under another version of
    the part, which `part_points` would refuse; or its record was graded from
    other readings than those stored now, its own or its dependencies'
    (`graded_from`; a record from before the digest was kept is taken as graded
    from them)."""
    found = stored.readings.get(part.id)
    record = stored.grades.get(part.id)
    if found is not None and found.reading.status == "unreadable":
        return UNREADABLE
    if (
        found is not None
        and part.response_type in CHOICE_TYPES
        and found.reading.status == "answered"
        and not found.reading.final_answer.choices
    ):
        return NO_OPTION
    if found is None or record is None:
        return UNGRADED
    if record.provenance.key_part_hash != part_hash(part, key.policies):
        return OTHER_KEY
    if record.reading_digest not in (None, graded_from(part, stored.readings)):
        return OTHER_READING
    return None


def score(key: Key, stored: Stored) -> list[Score]:
    """Each key part's score for the stored student, in key order: what its
    criteria award, after the class audit's changes and then the instructor's
    corrections, scaled to the part's points when a constraint is set. A part
    withheld is scored once every criterion of it is corrected."""
    points, step = allocation(key), key.policies.point_step
    scores: list[Score] = []
    for part in key.parts:
        why = withheld(key, part, stored)
        record = None if why else audited(stored, part)
        mine = {c.id: o for c in part.criteria if (o := stored.overrides.get(c.id))}
        if mine and (not why or len(mine) == len(part.criteria)):
            why, record = None, _corrected(key, part, record, mine)
        awarded = None
        if record:
            earned, weight = part_points(record, key)
            awarded = earned
            if key.points is not None:
                awarded = scaled(points[part.id], earned, weight, step)
        scores.append(Score(part, record, awarded, why, mine))
    return scores


def _corrected(
    key: Key, part: KeyPart, record: GradeRecord | None, mine: Mapping[str, Override]
) -> GradeRecord:
    """`record` with the instructor's award in place of the grader's on each
    criterion corrected in `mine`, marked, with fixed words and the note for the
    student; for a part the tool did not score, a record of the corrections alone."""
    if record is None:  # every criterion is corrected
        record = GradeRecord(
            provenance=Provenance(
                key_part_hash=part_hash(part, key.policies),
                prompt_version=None,
                model_slug=None,
            ),
            grade=Grade(
                part_id=part.id,
                criteria=[],
                transcript_dispute=None,
                belongs_elsewhere=None,
                code_overruled=None,
            ),
            check=None,
            reading_digest=None,
        )
    given = {c.id: c for c in record.grade.criteria}
    criteria = [
        CriterionGrade(
            id=c.id,
            rationale=CORRECTED,
            evidence_line=None,
            evidence_element=None,
            awarded=o.points,
            feedback=" ".join(
                [
                    MARKED
                    if key.points is not None
                    else MARKED_POINTS.format(
                        points=number(o.points), full=number(c.points)
                    )
                ]
                + ([o.note] if o.note else [])
            ),
        )
        if (o := mine.get(c.id))
        else given[c.id]
        for c in part.criteria
    ]
    grade = record.grade.model_copy(update={"criteria": criteria})
    return record.model_copy(update={"grade": grade})


def audited(stored: Stored, part: KeyPart) -> GradeRecord:
    """The stored grade record of `part` with each change of the class audit to the
    student's awards that still holds: one decided on the record's own part hash
    and award, to one of the criterion's levels, which the change's reason then
    explains to the instructor. Any other is ignored. An award lowered from full
    points shows the student fixed words and the criterion's requirement in place
    of the grader's feedback, never the reason; one lowered from a partial level
    keeps the grader's feedback."""
    record = stored.grades[part.id]
    audit = stored.audits.get(part.id)
    whose = stored.student.student_key
    changes = (
        {c.criterion: c for c in audit.changes if c.student_key == whose}
        if audit
        else {}
    )
    criteria = {c.id: c for c in part.criteria}
    found: list[CriterionGrade] = []
    for c in record.grade.criteria:
        change = changes.get(c.id)
        if (
            change is None
            or c.id not in criteria
            or change.part_hash != record.provenance.key_part_hash
            or change.before != c.awarded
            or change.after not in levels(criteria[c.id])
        ):
            found.append(c)
            continue
        full = criteria[c.id].points
        feedback = c.feedback
        if change.before == full > change.after:
            feedback = f"{LOWERED} {criteria[c.id].requirement}"
        why = (
            f"Changed by the class audit from {number(change.before)}: {change.reason}"
        )
        found.append(
            c.model_copy(
                update={"awarded": change.after, "rationale": why, "feedback": feedback}
            )
        )
    grade = record.grade.model_copy(update={"criteria": found})
    return record.model_copy(update={"grade": grade})


def flags(key: Key, students: Sequence[Stored]) -> list[Flag]:
    """Every flag the stored files give (this version's set): a `key_unverified`
    for each unverified key part and a `key_suspect` for each part the class audit
    found suspect, on no student, then each student's, part by part in key order.
    `derive` says which a review closes."""
    # The class audit, the same for every student.
    found: Mapping[str, PartAudit] = students[0].audits if students else {}
    on_key = [
        (FlagCode.KEY_UNVERIFIED, p) for p in key.parts if p.state == "unverified"
    ]
    on_key += [
        (FlagCode.KEY_SUSPECT, p)
        for p in key.parts
        if (audit := found.get(p.id)) and audit.key_suspect
    ]
    raised = [
        Flag(code=code, part_id=p.id, student_key=None, detail=DETAIL[code])
        for code, p in on_key
    ]
    for stored in students:
        for part in key.parts:
            raised += [
                Flag(
                    code=code,
                    part_id=part.id,
                    student_key=stored.student.student_key,
                    detail=detail,
                )
                for code, detail in _part_flags(key, part, stored)
            ]
    return raised


def status(key: Key, stored: Stored, flags: Sequence[Flag]) -> Status:
    """`incomplete` when a part is withheld, and not corrected in full; else
    `provisional` when one of the student's open `flags`, or one on the key (which
    counts for every student), is of high or medium severity; else `final`."""
    if any(s.withheld for s in score(key, stored)):
        return "incomplete"
    whose = (None, stored.student.student_key)
    if any(f.severity != "low" and f.student_key in whose for f in flags):
        return "provisional"
    return "final"


def _part_flags(key: Key, part: KeyPart, stored: Stored) -> list[tuple[FlagCode, str]]:
    why = withheld(key, part, stored)
    raised: list[tuple[FlagCode, str]] = []
    entries = stored.mapped.map.entries if stored.mapped else []
    # Placed on its content alone below high confidence, or on anything at low.
    doubted = any(
        e.part_id == part.id
        and (e.confidence == "low" or (e.basis == "content" and e.confidence != "high"))
        for e in entries
    )
    weak = [WEAK_MAP] if doubted else []
    if found := stored.readings.get(part.id):
        reading = found.reading
        second = stored.seconds.get(part.id)
        # A second reader that read the same settles the doubt it was asked about.
        settled = second and second.doubted and second.result == "same"
        hard = unsure(reading) and not settled
        if reading.status == "unreadable":
            raised.append((FlagCode.UNREADABLE, UNREADABLE))
        if hard:
            raised.append((FlagCode.HARD_TO_READ, DETAIL[FlagCode.HARD_TO_READ]))
        if second and second.result == "differ":
            code = FlagCode.READERS_DISAGREE
            raised.append((code, DETAIL[code]))
        # Zooming cannot help when every crop already shows the source at full size.
        # Never on a blank reading: nothing there was read.
        native = found.crops and all(c.scale and c.scale >= 1 for c in found.crops)
        if (reading.status == "unreadable" or hard) and native:
            raised.append((FlagCode.LOW_RESOLUTION, DETAIL[FlagCode.LOW_RESOLUTION]))
        if reading.addressed_to_grader:
            code = FlagCode.ADDRESSED_TO_GRADER
            raised.append((code, DETAIL[code]))
    if why is None:
        grade = stored.grades[part.id].grade
        if (again := stored.second_grades.get(part.id)) and again.result == "differ":
            code = FlagCode.GRADERS_DISAGREE
            raised.append((code, DETAIL[code]))
        # A reason of only white space is no reason.
        if (grade.transcript_dispute or "").strip():
            code = FlagCode.TRANSCRIPT_DISPUTE
            raised.append((code, DETAIL[code]))
        if (grade.code_overruled or "").strip():
            raised.append((FlagCode.CODE_OVERRULED, DETAIL[FlagCode.CODE_OVERRULED]))
        if audited(stored, part) != stored.grades[part.id]:
            raised.append((FlagCode.AUDIT_CHANGED, DETAIL[FlagCode.AUDIT_CHANGED]))
        if (grade.belongs_elsewhere or "").strip():
            weak.append(WEAK_GRADER)
    elif why != UNREADABLE:
        raised.append((FlagCode.INCOMPLETE, why))
    if weak:
        raised.append((FlagCode.WEAK_ATTRIBUTION, " ".join(weak)))
    return raised


def _read[M: BaseModel](path: Path, model: type[M]) -> M | None:
    try:
        return read_model(path, model)
    except FileNotFoundError:
        return None


def points_of(key: Key, parts: Iterable[KeyPart]) -> Decimal:
    """The parts' maximum: the sum of their points in the allocation, under a
    constraint in its shortest form (1.5 + 2.5 is 4, not 4.0)."""
    points = allocation(key)
    total = sum((points[p.id] for p in parts), Decimal(0))
    return total if key.points is None else Decimal(number(total))


def sum_of(scores: Iterable[Score]) -> Decimal | None:
    """The points `scores` award, or None (an empty cell) when one is withheld."""
    total = Decimal(0)
    for s in scores:
        if s.awarded is None:
            return None
        total += s.awarded
    return total


def _total(scores: Iterable[Score]) -> Decimal:
    """The sum of what was scored."""
    return sum((s.awarded for s in scores if s.awarded is not None), Decimal(0))


# --- writing -------------------------------------------------------------------------

type Cell = str | Decimal | int | None


def _write_csv(path: Path, header: list[str], rows: Iterable[list[Cell]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # With a byte order mark, so that a spreadsheet reads the names as UTF-8.
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows([_cell(value) for value in row] for row in rows)


def _cell(value: Cell) -> str:
    """A number as `number` prints it, None as an empty cell, and text that a
    spreadsheet would run as a formula behind an apostrophe."""
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return number(value)
    if isinstance(value, int):
        return str(value)
    # Leading white space too: an importer that trims it would run the rest.
    risky = value.startswith(FORMULA_STARTS) or value.lstrip().startswith(
        FORMULA_STARTS
    )
    return "'" + value if risky else value


def _out_of(awarded: Decimal | None, worth: Decimal, missing: str) -> str:
    if awarded is None:
        return f"{missing}, out of {number(worth)}"
    return f"{number(awarded)} / {number(worth)}"


def _feedback(
    key: Key,
    title: str,
    problems: Sequence[str],
    scores: Sequence[Score],
    total: Decimal,
    maximum: Decimal,
) -> str:
    """Design 8.3's layout: the total, each problem and each part with its points,
    and under a part the feedback of each criterion that lost points, or why the
    part is not scored. A problem of one part has no line of its own for it."""
    lines = [f"# {title}", "", f"Total: {number(total)} / {number(maximum)}"]
    if any(s.withheld for s in scores):
        lines += ["", "Some parts are not scored yet, so this total is not complete."]
    for problem in problems:
        mine = [s for s in scores if problem_of(s.part.id) == problem]
        worth = _out_of(
            sum_of(mine), points_of(key, (s.part for s in mine)), "not fully scored"
        )
        # Every problem id is "Q" and its number (the PartId pattern).
        lines += ["", f"## Problem {problem[1:]}: {worth}"]
        for s in mine:
            if len(mine) > 1 or s.part.id != problem:
                points = _out_of(s.awarded, points_of(key, [s.part]), "not scored")
                lines += ["", f"**{s.part.label}: {points}**"]
            lines += ["- " + one_line(note) for note in _notes(s)]
    return "\n".join(lines) + "\n"


def one_line(note: str) -> str:
    """`note` as one line of plain text: each run of white space as one space, every
    other control or format character (a NUL, a bidi override, a zero-width space)
    left out, so that nothing can reorder or hide the text, and every backslash,
    `<` and `#` escaped, so text a model or the instructor wrote cannot start a
    heading, open an HTML comment or forge a score line. (The backslash is escaped
    too, or a note holding `\\<!--` would undo the escape of its `<`.)"""
    plain = "".join(c for c in note if unicodedata.category(c)[0] != "C" or c.isspace())
    text = " ".join(plain.split())
    return text.replace("\\", "\\\\").replace("<", "\\<").replace("#", "\\#")


def code_span(text: str) -> str:
    """Untrusted `text`, such as a file name, as a Markdown code span on one line:
    each control or line-separating character (a newline, a bidi override) as `?`,
    and fenced by more backticks than any run of them inside."""
    plain = "".join(
        "?" if unicodedata.category(c)[0] == "C" or c in "  " else c for c in text
    )
    fence = "`" * (max(map(len, re.findall("`+", plain)), default=0) + 1)
    pad = " " if plain[:1] in ("`", " ") or plain[-1:] in ("`", " ") else ""
    return f"{fence}{pad}{plain}{pad}{fence}"


def _notes(s: Score) -> list[str]:
    if s.record is None:
        return [s.withheld or UNGRADED]
    got = {c.id: c for c in s.record.grade.criteria}
    return [
        got[c.id].feedback
        for c in s.part.criteria
        if c.id in s.overrides
        or (got[c.id].awarded < c.points and got[c.id].feedback.strip())
    ]


def _audit_lines(found: Mapping[str, PartAudit]) -> list[str]:
    """The class audit's counts, and each part whose key answer the class disputes
    with how many share another answer and how many answered (never the answer)."""
    if not found:
        return ["No class audit is stored for the key as it is."]
    groups = sum(a.groups for a in found.values())
    undecided = sum(a.undecided for a in found.values())
    changed = sum(a.changed for a in found.values())
    split = sum(a.split for a in found.values())
    lines = [
        f"Groups of the same evidence with differing awards: {groups}. Decided"
        f" again: {groups - undecided}. Changed: {changed}. Could not be decided:"
        f" {undecided}."
        + (
            f" Groups too large for one request, decided in parts that may differ:"
            f" {split}."
            if split
            else ""
        ),
        "",
    ]
    suspects = [(p, a.key_suspect) for p, a in found.items() if a.key_suspect]
    if not suspects:
        return [*lines, "No part's key answer is disputed by the class."]
    lines += [
        "Parts whose key answer the class disputes (`key_suspect`):",
        "",
        "| Part | Students sharing another answer | Students who answered |",
        "|---|---|---|",
    ]
    return lines + [f"| {p} | {s.shared} | {s.answered} |" for p, s in suspects]


def _run_report(
    workspace: Workspace,
    key: Key,
    pages: Sequence[int],
    found: Derived,
    not_ingested: Sequence[str],
) -> str:
    """The state of every key part, the class audit, the second read's and the
    second grade's results, each page's maximum and the total, the students with
    parts not scored and why, the files not ingested, the corrections not applied
    and the reviews out of date, the flags raised by code and how many a review
    closes, the pinned models, and the cost by role from the usage ledger."""
    scored, raised = found.scored, found.raised
    maxima = page_points(key)
    pinned = workspace.config_lock_toml.exists()
    lock = read_lock(workspace.config_lock_toml) if pinned else None
    lines = ["# Run report", "", "## Key", ""]
    if key.points is not None:
        lines += [
            f"{scoring(key.points, key.policies.point_step)} scores_detail.csv stays"
            " in the criteria's own points, the grader's units.",
            "",
        ]
    lines += ["| Part | Page | Points | State |", "|---|---|---|---|"]
    lines += [
        f"| {p.id} | {p.home_page} | {number(points_of(key, [p]))} | {p.state} |"
        for p in key.parts
    ]
    lines += ["", "## Class audit", "", *_audit_lines(audits(workspace, key))]
    seconds = [r for s, _ in scored for r in s.seconds.values()]
    doubted = Counter(r.result for r in seconds if r.doubted)
    others = Counter(r.result for r in seconds if not r.doubted)
    lines += [
        "",
        "## Second read",
        "",
        "Parts the reader was unsure of, read a second time by another vendor's"
        f" model: {doubted.total()}. The same: {doubted['same']}. Different:"
        f" {doubted['differ']}. Not decided: {doubted['not decided']}. Failed:"
        f" {doubted['failed']}.",
        "",
        f"The other parts of those units: {others.total()}. The same:"
        f" {others['same']}. Different: {others['differ']}. Not decided:"
        f" {others['not decided']}.",
        "",
        "## Second grade",
        "",
        *_second_grade_lines(scored, lock.profile if lock else None),
    ]
    lines += ["", "## Pages", "", "| Page | Maximum |", "|---|---|"]
    lines += [f"| {n} | {number(maxima.get(n, Decimal(0)))} |" for n in pages]
    lines += [f"| Total | {number(points_of(key, key.parts))} |", "", "## Students", ""]
    short = [(s, sc) for s, sc in scored if any(x.withheld for x in sc)]
    lines.append(
        f"Students: {len(scored)}. With every part scored: {len(scored) - len(short)}."
        f" With parts not scored: {len(short)}."
    )
    if short:
        lines.append("")
    for s, scores in short:
        withheld = [(x, f"{x.withheld}{_still_open(x)}") for x in scores if x.withheld]
        for why in dict.fromkeys(w for _, w in withheld):
            parts = ", ".join(x.part.id for x, w in withheld if w == why)
            lines.append(f"- `{s.student.student_key}`: {parts}. {why}")
    lines += unclaimed_lines(workspace, [s.student for s, _ in scored])
    if not_ingested:
        lines += ["", "## Files not ingested", ""]
        lines.append("The reason ingest gave for each is in `ingest_summary.json`.")
        lines.append("")
        lines += [f"- {code_span(file)}: {NOT_INGESTED}" for file in not_ingested]
    lines += [*_stale(found.stale), *_outdated(found.outdated)]
    counts = Counter(f.code for f in raised)
    lines += ["", "## Flags", ""]
    if counts:
        lines += ["| Flag | Severity | Count |", "|---|---|---|"]
        lines += [
            f"| {c} | {SEVERITY[c]} | {counts[c]} |" for c in FlagCode if c in counts
        ]
        lines += [
            "",
            f"Flags raised: {len(raised)}. Closed by review: {len(found.closed)}."
            f" Open: {len(found.open)}.",
        ]
    else:
        lines.append("No flags.")
    lines += ["", review_line(workspace, found.open), "", "## Models", ""]
    if lock:
        lines += ["| Role | Model |", "|---|---|"]
        lines += [f"| {role} | {m.slug} |" for role, m in lock.models.items()]
        if lock.strict_privacy is False:
            lines += ["", STRICT_OFF_RUN]
        if lock.mask_names:
            who = [s.student.student_key for s, _ in scored]
            blanked, named, left = name_counts(workspace, who)
            lines += [
                "",
                f"Names masked (`mask_names`). Name regions blanked: {blanked}, found"
                f" on {named} pages, each in every image of its page that is sent after"
                f" the map. Left whole because they overlap a response: {left}. The"
                " mapper sees each page whole, before it is blanked.",
            ]
    else:
        lines.append("No model is pinned in this workspace.")
    usage: dict[str, Usage] = {}
    if workspace.cache_sqlite.exists():
        with closing(sqlite3.connect(workspace.cache_sqlite)) as db:
            usage = Ledger(db).totals_by_role()
    lines += ["", "## Cost", "", "| Role | Prompt tokens | Completion tokens | USD |"]
    lines.append("|---|---|---|---|")
    for role, u in [*sorted(usage.items()), ("Total", sum(usage.values(), NO_USE))]:
        cost = number(u.cost)
        lines.append(f"| {role} | {u.prompt_tokens} | {u.completion_tokens} | {cost} |")
    return "\n".join(lines) + "\n"


def _second_grade_lines(
    scored: Sequence[tuple[Stored, list[Score]]], profile: Profile | None
) -> list[str]:
    """The second grade's counts, or why there is none: the lock's profile, when
    no result is stored (a lock edited since keeps the flags of those that are)."""
    found = [(s, r) for s, _ in scored for r in s.second_grades.values()]
    if not found and profile is not None and profile not in PROFILES:
        return [f"The second grade is not done under the profile {profile}."]
    pairs = [(s.grades[r.part_id].grade, r.second.grade) for s, r in found if r.second]
    awards = sum(len(first.criteria) for first, _ in pairs)
    differ = sum(len(differing(first, second)) for first, second in pairs)
    failed = {
        (s.student.student_key, problem_of(r.part_id))
        for s, r in found
        if r.result == "failed"
    }
    rate = f" ({(awards - differ) / awards:.1%})" if awards else ""
    return [
        "Awards graded a second time, by the same grader with the same request, and"
        f" compared: {awards}. Different: {differ}. Parts flagged"
        f" (`graders_disagree`): {sum(r.result == 'differ' for _, r in found)}."
        f" Problems whose second grade failed: {len(failed)}.",
        "",
        f"Agreement: {awards - differ} of {awards} awards the same{rate}.",
    ]


# --- the review list -----------------------------------------------------------------


def review_line(workspace: Workspace, raised: Sequence[Flag]) -> str:
    """The run report's and the terminal's line on the review list: its path in the
    workspace and its number of entries, one for each key part, student and part,
    and whole submission flagged."""
    path = workspace.review_md.relative_to(workspace.root).as_posix()
    entries = len({(f.student_key, f.part_id) for f in raised})
    return f"Entries in the review list, {path}: {entries}."


def review(
    workspace: Workspace,
    key: Key,
    scored: Sequence[tuple[Stored, list[Score]]],
    raised: Sequence[Flag],
    stale: Sequence[Override] = (),
    closed: Sequence[Flag] = (),
    outdated: Sequence[Review] = (),
) -> str:
    """`out/review.md` (design 8.4): an entry for each key part with an open flag
    in `raised`, with its points, the number of students scored on it and what is
    stored behind its flags; then one for each student and part, or whole
    submission, flagged, with every flag on it and its stored cause, the proposed
    score, the instructor's corrections, the transcript and links to the crops and
    the stored files; then the `stale` corrections, not applied, the `outdated`
    reviews, and last each entry whose flags a review closed (`closed`), in one
    line. Entries come in `_order`'s order. Text a student, a model or the
    instructor wrote is only ever in a code span."""
    lines = ["# Review list", ""]
    after = [*_stale(stale), *_outdated(outdated), *_closed(key, scored, closed)]
    if not raised:
        return "\n".join([*lines, NOTHING_TO_REVIEW, *after]) + "\n"
    out = workspace.review_md.parent
    entries = _grouped(raised)
    for place in _order(key, entries):
        if (
            section := "## The key" if place[0] is None else "## Students"
        ) not in lines:
            lines += [section, ""]
        shown, files = _entry(workspace, key, scored, place, entries[place])
        lines += [*shown, *([] if files is None else [_files(out, files)]), ""]
    return "\n".join([*lines[:-1], *after, ""])


def _grouped(raised: Iterable[Flag]) -> dict[Place, list[Flag]]:
    """The flags of each entry, by place, in the order raised."""
    entries: dict[Place, list[Flag]] = {}
    for f in raised:
        entries.setdefault((f.student_key, f.part_id), []).append(f)
    return entries


def _order(key: Key, entries: Mapping[Place, Sequence[Flag]]) -> list[Place]:
    """The review list's order of `entries`: the key's parts in key order, then the
    students' by their highest severity, the points at stake, the student key and
    the key's part order. A whole submission is at stake for the key's total,
    before any part."""
    order = [p.id for p in key.parts]

    def at_stake(place: Place) -> tuple[int, Decimal, str, int]:
        who, part_id = place
        worst = min(SEVERITIES.index(f.severity) for f in entries[place])
        parts = [p for p in key.parts if p.id == part_id] if part_id else key.parts
        at = order.index(part_id) if part_id else -1
        return worst, -points_of(key, parts), who or "", at

    on_key: list[Place] = [(None, p.id) for p in key.parts if (None, p.id) in entries]
    return [*on_key, *sorted((p for p in entries if p[0] is not None), key=at_stake)]


def _score_at(
    scored: Sequence[tuple[Stored, list[Score]]], place: Place
) -> Score | None:
    """The score of a student's part at `place`; None for any other entry."""
    who, part_id = place
    for s, xs in scored:
        if s.student.student_key == who:
            return next((x for x in xs if x.part.id == part_id), None)
    return None


def _entry(
    workspace: Workspace,
    key: Key,
    scored: Sequence[tuple[Stored, list[Score]]],
    place: Place,
    found: Sequence[Flag],
) -> tuple[list[str], list[tuple[str, Path]] | None]:
    """The review list's entry at `place`, with flags `found`: its lines up to its
    files, and the files it links (None for a whole submission, which links
    none)."""
    who, part_id = place
    files: list[tuple[str, Path]] = []
    if who is None:
        [part] = [p for p in key.parts if p.id == part_id]
        # The class audit, the same for every student.
        class_audit: Mapping[str, PartAudit] = scored[0][0].audits if scored else {}
        count = sum(
            x.part.id == part.id and x.awarded is not None
            for _, xs in scored
            for x in xs
        )
        causes: dict[FlagCode, list[str]] = {}
        unverified = FlagCode.KEY_UNVERIFIED in {f.code for f in found}
        if unverified and workspace.verification_md.exists():
            files.append(("verification", workspace.verification_md))
        if (audit := class_audit.get(part.id)) and (suspect := audit.key_suspect):
            causes[FlagCode.KEY_SUSPECT] = [
                f"  - Students who answered: {suspect.answered}. Sharing one answer"
                f" that is not the key's: {suspect.shared}."
            ]
            files.append(("audit", workspace.audit_json(part.id)))
        return [
            f"### {part.id}, worth {number(points_of(key, [part]))}",
            "",
            *_flagged(found, causes),
            f"- Students scored on it: {count}.",
        ], files
    [(stored, scores)] = [(s, xs) for s, xs in scored if s.student.student_key == who]
    name = f"{code_span(stored.student.student_name)} (`{who}`)"
    if part_id is None:
        return [f"### {name}, the whole submission", "", *_flagged(found, {})], None
    [s] = [x for x in scores if x.part.id == part_id]
    record = stored.grades.get(part_id)
    # The tool's own score: the grade record after the class audit.
    base = None if withheld(key, s.part, stored) else audited(stored, s.part)
    worth = _out_of(s.awarded, points_of(key, [s.part]), "not scored")
    causes = _causes(record, base, stored.second_grades.get(part_id))
    lines = [f"### {name}, {part_id}", "", *_flagged(found, causes)]
    lines.append(
        f"- Proposed score: {worth}."
        + (f" {s.withheld}{_still_open(s)}" if s.withheld else "")
    )
    lines += _corrections(s, base)
    codes = {f.code for f in found}
    if first := stored.readings.get(part_id):
        lines += _transcript("First reading", first)
        files += [
            (f"crop {n}", p) for n, p in enumerate(_crops(workspace, who, first), 1)
        ]
        files.append(("reading", workspace.reading_json(who, problem_of(part_id))))
    second = stored.seconds.get(part_id)
    if second and second.second and FlagCode.READERS_DISAGREE in codes:
        lines += _transcript("Second reading, by another vendor's model", second.second)
    if base:  # a grade withheld is not linked: the score shown is not from it
        files.append(("grade", workspace.grade_json(who, problem_of(part_id))))
    # The grade file holds the awards before the audit; its file, the change.
    if FlagCode.AUDIT_CHANGED in codes:
        files.append(("audit", workspace.audit_json(part_id)))
    if FlagCode.GRADERS_DISAGREE in codes:
        files.append(("second grade", workspace.second_grade_json(who)))
    return lines, files


def _crops(workspace: Workspace, who: str, first: ReadingRecord) -> list[Path]:
    """The crop image of each region of `who`'s reading `first`, which its entry
    links and its review covers."""
    part = first.reading.part_id
    return [workspace.crop_png(who, part, n) for n in range(1, len(first.crops) + 1)]


def _whose(place: Place) -> str:
    """Whose entry `place` is, and which part's."""
    who, part_id = place
    if who is None:
        return f"The key, {part_id}"
    return f"`{who}`, {part_id or 'the whole submission'}"


def _line(
    key: Key,
    scored: Sequence[tuple[Stored, list[Score]]],
    place: Place,
    found: Sequence[Flag],
) -> str:
    """The entry at `place` in one line: whose and which part, the codes of its
    flags `found`, and the proposed score, or a key part's points."""
    codes = ", ".join(f"`{f.code}`" for f in _ranked(found))
    line = f"{_whose(place)}: {codes}."
    if s := _score_at(scored, place):
        worth = _out_of(s.awarded, points_of(key, [s.part]), "not scored")
        return f"{line} Proposed score: {worth}."
    if place[0] is None:
        [part] = [p for p in key.parts if p.id == place[1]]
        return f"{line} Worth {number(points_of(key, [part]))}."
    return line


def _outdated(outdated: Sequence[Review]) -> list[str]:
    """The section of the reviews out of date, with a blank line before it, or
    nothing when there are none."""
    if not outdated:
        return []
    found = [f"- {_whose((r.student, r.part))}." for r in outdated]
    return ["", "## Reviews out of date", "", OUTDATED, "", *found]


def _closed(
    key: Key, scored: Sequence[tuple[Stored, list[Score]]], closed: Sequence[Flag]
) -> list[str]:
    """The review list's last section: each entry whose flags `closed` a review
    closed, in one line, in the list's order; nothing when there is none."""
    if not closed:
        return []
    entries = _grouped(closed)
    found = [f"- {_line(key, scored, p, entries[p])}" for p in _order(key, entries)]
    return ["", "## Closed by review", "", CLOSED, "", *found]


@dataclass(frozen=True)
class Entry:
    """An entry of the review list as `redpen review` takes it: the review that
    would close it as it stands, whether a stored review closes it, its line for
    `--list`, what the terminal shows of it, and the score of a student's part."""

    review: Review
    closed: bool
    line: str
    shown: list[str]
    score: Score | None


def review_entries(
    workspace: Workspace, key: Key, students: Sequence[IngestRecord]
) -> list[Entry]:
    """Every entry of the review list for `students`, in its order: what
    `out/review.md` shows of it, then for a student's part each criterion with its
    award and why, and the absolute path of each file it links. Raises as `derive`
    does."""
    found = derive(workspace, key, students)
    entries = _grouped(found.raised)
    listed: list[Entry] = []
    for place in _order(key, entries):
        shown, files = _entry(workspace, key, found.scored, place, entries[place])
        s = _score_at(found.scored, place)
        shown += _criteria(s) if s else []
        if files is None:  # a whole submission: the images review.md links
            [who] = [
                st.student
                for st, _ in found.scored
                if st.student.student_key == place[0]
            ]
            files = unclaimed_shown(workspace, who)[1]
        if files:
            shown += ["- Files:", *(f"  - {t}: {p.absolute()}" for t, p in files)]
        line = _line(key, found.scored, place, entries[place])
        review = found.current[place]
        listed.append(Entry(review, place in found.reviewed, line, shown, s))
    return listed


def _criteria(s: Score) -> list[str]:
    """Each criterion of a student's part with the award shown, in its own points,
    and why; or, on a part withheld, the instructor's award, or none."""
    given = {c.id: c for c in s.record.grade.criteria} if s.record else {}
    lines = ["- Criteria:"]
    for c in s.part.criteria:
        worth = number(c.points)
        if found := given.get(c.id):
            awarded = number(found.awarded)
            lines.append(
                f"  - {c.id}: {awarded} / {worth}. {code_span(found.rationale)}"
            )
        elif o := s.overrides.get(c.id):
            lines.append(f"  - {c.id}: {number(o.points)} / {worth}, the instructor's.")
        else:
            lines.append(f"  - {c.id}: none, out of {worth}.")
    return lines


def _still_open(s: Score) -> str:
    """After the reason a part is withheld, how many of its criteria have no
    correction yet, when some have; for the instructor's files only."""
    if not (s.withheld and s.overrides):
        return ""
    count = len(s.part.criteria)
    return " " + OPEN.format(all=count, open=count - len(s.overrides))


def _corrections(s: Score, base: GradeRecord | None) -> list[str]:
    """Each criterion of the part the instructor corrected: the grader's award in
    `base`, or none when the tool did not score the part, the instructor's, and the
    note."""
    given = {c.id: number(c.awarded) for c in base.grade.criteria} if base else {}
    return [
        f"- Corrected: {c.id}, the grader's award {given.get(c.id, 'none')}, the"
        f" instructor's {number(o.points)}."
        + (f" Note: {code_span(o.note)}" if o.note else "")
        for c in s.part.criteria
        if (o := s.overrides.get(c.id))
    ]


def _stale(stale: Sequence[Override]) -> list[str]:
    """The section of the corrections not applied, with a blank line before it, or
    nothing when there are none."""
    if not stale:
        return []
    found = [f"- `{o.student}`, {o.criterion}: {number(o.points)}." for o in stale]
    return ["", "## Corrections not applied", "", STALE, "", *found]


def _flagged(found: Sequence[Flag], causes: Mapping[FlagCode, list[str]]) -> list[str]:
    """Each flag, in `_ranked` order, each with the lines of what is stored behind
    it."""
    return [
        line
        for f in _ranked(found)
        for line in [f"- `{f.code}`, {f.severity}: {f.detail}", *causes.get(f.code, [])]
    ]


def _ranked(found: Sequence[Flag]) -> list[Flag]:
    """High severity first, and then as raised."""
    return sorted(found, key=lambda f: SEVERITIES.index(f.severity))


def _causes(
    record: GradeRecord | None,
    after: GradeRecord | None,
    again: SecondGrade | None = None,
) -> dict[FlagCode, list[str]]:
    """What a part's stored grade holds behind the flags it raises: the grader's
    own sentences (its dispute, its overruling, the other part it says the work
    answers), the answer check it overruled, each award the class audit
    changed, and each award its second grade `again` gave otherwise, with what
    each grader wrote. The changes are the awards `after` the audit (None when the
    part is withheld) against the stored record's, so a change that no longer holds
    is never shown."""
    if record is None or after is None:
        return {}
    grade, check = record.grade, record.check
    second = again.second.grade if again and again.second else None
    pairs = [
        (c, next(x for x in second.criteria if x.id == c.id))
        for c in grade.criteria
        if second and c.id in differing(grade, second)
    ]
    before = {c.id: c.awarded for c in grade.criteria}
    # Null for a part graded before the checks existed.
    checked = (
        [f"  - The answer check: {check.verdict}, {code_span(check.reason)}"]
        if check
        else []
    )
    return {
        FlagCode.TRANSCRIPT_DISPUTE: _wrote(grade.transcript_dispute),
        FlagCode.CODE_OVERRULED: checked + _wrote(grade.code_overruled),
        FlagCode.WEAK_ATTRIBUTION: _wrote(grade.belongs_elsewhere),
        FlagCode.AUDIT_CHANGED: [
            f"  - {c.id}: {number(before[c.id])} before the audit,"
            f" {number(c.awarded)} after."
            for c in after.grade.criteria
            if c.awarded != before[c.id]
        ],
        FlagCode.GRADERS_DISAGREE: [
            f"  - {c.id}, the {which} grade: {number(c.awarded)}. The grader wrote:"
            f" {code_span(c.rationale)}"
            for pair in pairs
            for which, c in zip(("first", "second"), pair, strict=True)
        ],
    }


def _wrote(sentence: str | None) -> list[str]:
    if sentence is None or not sentence.strip():  # only white space raises nothing
        return []
    return [f"  - The grader wrote: {code_span(sentence)}"]


def _files(out: Path, files: Sequence[tuple[str, Path]]) -> str:
    """Each file as a link from `out/`, labelled."""
    links = [f"[{t}]({p.relative_to(out, walk_up=True).as_posix()})" for t, p in files]
    return f"- Files: {', '.join(links) or 'none'}."


def _transcript(title: str, record: ReadingRecord) -> list[str]:
    """`record`'s reading as stored: poor legibility, each line with the reader's
    marks, each element of a drawing, and the final answer. Each piece the reader
    transcribed is in a code span; the marks are fixed words."""
    reading = record.reading
    lines = [f"- {title}:"]
    if reading.legibility == "poor":
        lines.append("  - The reader marked the legibility as poor.")
    for n, text in enumerate(reading.lines, 1):
        marks = [
            mark
            for mark, on in [
                ("the reader was unsure of this line", n in reading.uncertain),
                ("addressed to the grader", n in reading.addressed_to_grader),
            ]
            if on
        ]
        lines.append(
            f"  - Line {n}: {code_span(text)}{''.join(f' ({m})' for m in marks)}"
        )
    for n, e in enumerate(reading.elements, 1):
        fields = [("label", e.label), ("kind", e.kind)]
        fields += [("direction", e.direction), ("position", e.position)]
        said = [f"{name} {code_span(value)}" for name, value in fields if value]
        lines.append(f"  - Element {n}: {', '.join(said) or 'nothing stated'}.")
    lines.append(f"  - Final answer: {_answer(reading.final_answer)}.")
    return lines


def _answer(answer: FinalAnswer) -> str:
    """What the final answer states: the chosen options, the value with its unit,
    the expression; each in a code span."""
    said: list[str] = []
    if answer.choices:
        said.append(f"options {', '.join(map(code_span, answer.choices))}")
    stated = None if answer.value is None else str(answer.value)
    if value := " ".join(text for text in (stated, answer.unit) if text):
        said.append(f"value {code_span(value)}")
    if answer.expression:
        said.append(f"expression {code_span(answer.expression)}")
    return "; ".join(said) or "none stated"
