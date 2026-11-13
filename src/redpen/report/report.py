"""Report, first version (design 8 and 9): the three score files, one feedback file
per student, the flags and the run report, computed by code from the stored
readings and grades. No model call; every point value is a Decimal.

The caller supplies the students. The `students/` directory is never listed: a
respelled name leaves an old folder behind."""

import csv
import re
import sqlite3
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from redpen.config import read_lock
from redpen.grade.grade import levels, part_points
from redpen.grade.read import digest, problem_of, unsure
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
from redpen.schemas.audit import PartAudit
from redpen.schemas.flags import SEVERITY, Flag, FlagCode
from redpen.schemas.grade import CriterionGrade, GradeRecord, ProblemGrades
from redpen.schemas.key import Key, KeyPart
from redpen.schemas.reading import (
    ProblemReadings,
    ReadingRecord,
    SecondReading,
    SecondReadings,
)
from redpen.schemas.spec import CHOICE_TYPES
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, read_model

type Status = Literal["incomplete", "provisional", "final"]

# The four reasons a part is withheld, never scored zero (design 9), in fixed words.
UNREADABLE = "The work could not be read."
NO_OPTION = "No chosen option could be read."
OTHER_KEY = "It was graded under another version of the key and is to be graded again."
UNGRADED = "No reading or no grade was recorded for it."
# Before the requirement of a criterion the class audit lowered from its full
# points, in place of whatever the grader wrote then (often praise): the student's
# file never shows the audit's own reason.
LOWERED = "This point was marked again for consistency across the class. What it needs:"
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
    FlagCode.LOW_RESOLUTION: "The work was read at the full resolution of the"
    " source and is still hard to read.",
    FlagCode.ADDRESSED_TO_GRADER: "The response holds text addressed to the grader.",
    FlagCode.TRANSCRIPT_DISPUTE: "The grader believes the reading is wrong.",
    FlagCode.CODE_OVERRULED: "The grader overruled the computed answer check.",
    FlagCode.KEY_SUSPECT: "The class's most common answer to this part is not the"
    " key's: check the key.",
    FlagCode.AUDIT_CHANGED: "The class audit changed an award on this part.",
}

# The whole review list when no flag is open, after its title.
NOTHING_TO_REVIEW = "Nothing needs review: no flag is open."
SEVERITIES = ("high", "medium", "low")

# A spreadsheet runs a cell that begins with one of these as a formula.
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")

NO_USE = Usage(0, 0, Decimal(0))


@dataclass(frozen=True)
class Stored:
    """One student's ingest record, and each part's reading and grade record that
    are stored, by part id; the class audit of each part, the same for every
    student; and each part's second reading compared with the stored first."""

    student: IngestRecord
    readings: dict[str, ReadingRecord]
    grades: dict[str, GradeRecord]
    audits: Mapping[str, PartAudit] = field(default_factory=dict[str, PartAudit])
    seconds: Mapping[str, SecondReading] = field(
        default_factory=dict[str, SecondReading]
    )


@dataclass(frozen=True)
class Score:
    """One part of one student: the grade record that counts and the points it
    awards on the key's scale, or, when the part is withheld, why."""

    part: KeyPart
    record: GradeRecord | None
    awarded: Decimal | None
    withheld: str | None


def write_report(
    workspace: Workspace,
    key: Key,
    students: Sequence[IngestRecord],
    not_ingested: Sequence[str] = (),
) -> list[Flag]:
    """Write `out/` for `students`, in the order given: the three score files, one
    feedback file per student (any other is removed), the review list and the run
    report, which also lists the files of `not_ingested` (names inside the
    submissions, never opened here). Returns every flag derived."""
    problems, pages = columns(workspace, key)
    stored = [load(workspace, key, s) for s in students]
    raised = flags(key, stored)
    scored = [(s, score(key, s)) for s in stored]
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
    run_report = _run_report(workspace, key, pages, scored, raised, not_ingested)
    review_md = review(workspace, key, scored, raised)
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
    key's top-level problems whose file exists, and each second reading made for
    the first reading stored now and the key's version of its part."""
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
    hashes = {p.id: part_hash(p, key.policies) for p in key.parts}
    seconds = {
        s.part_id: s
        for s in (again.records if again else [])
        if s.part_id in readings
        and s.first == digest(readings[s.part_id])
        and s.part_hash == hashes.get(s.part_id)
    }
    return Stored(student, readings, grades, audits(workspace, key), seconds)


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
    no reading or no grade record; or its record was graded under another version
    of the part, which `part_points` would refuse."""
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
    return None


def score(key: Key, stored: Stored) -> list[Score]:
    """Each key part's score for the stored student, in key order: what its
    criteria award, scaled to the part's points when a constraint is set."""
    points, step = allocation(key), key.policies.point_step
    scores: list[Score] = []
    for part in key.parts:
        why = withheld(key, part, stored)
        record = None if why else audited(stored, part)
        awarded = None
        if record:
            earned, weight = part_points(record, key)
            awarded = earned
            if key.points is not None:
                awarded = scaled(points[part.id], earned, weight, step)
        scores.append(Score(part, record, awarded, why))
    return scores


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
    All are open: nothing closes a flag yet."""
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
    """`incomplete` when a part is withheld; else `provisional` when one of the
    student's flags, or a flag on the key (which counts for every student), is of
    high or medium severity; else `final`."""
    if any(withheld(key, part, stored) for part in key.parts):
        return "incomplete"
    whose = (None, stored.student.student_key)
    if any(f.severity != "low" and f.student_key in whose for f in flags):
        return "provisional"
    return "final"


def _part_flags(key: Key, part: KeyPart, stored: Stored) -> list[tuple[FlagCode, str]]:
    why = withheld(key, part, stored)
    raised: list[tuple[FlagCode, str]] = []
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
        # A reason of only white space is no reason.
        if (grade.transcript_dispute or "").strip():
            code = FlagCode.TRANSCRIPT_DISPUTE
            raised.append((code, DETAIL[code]))
        if (grade.code_overruled or "").strip():
            raised.append((FlagCode.CODE_OVERRULED, DETAIL[FlagCode.CODE_OVERRULED]))
        if audited(stored, part) != stored.grades[part.id]:
            raised.append((FlagCode.AUDIT_CHANGED, DETAIL[FlagCode.AUDIT_CHANGED]))
    elif why != UNREADABLE:
        raised.append((FlagCode.INCOMPLETE, why))
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
    """`note` as one line of plain text: each run of white space as one space, and
    every backslash, `<` and `#` escaped, so text a model wrote cannot start a
    heading, open an HTML comment or forge a score line. (The backslash is escaped
    too, or a note holding `\\<!--` would undo the escape of its `<`.)"""
    text = " ".join(note.split())
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
        if got[c.id].awarded < c.points and got[c.id].feedback.strip()
    ]


def _audit_lines(found: Mapping[str, PartAudit]) -> list[str]:
    """The class audit's counts, and each part whose key answer the class disputes
    with how many share another answer and how many answered (never the answer)."""
    if not found:
        return ["No class audit is stored for the key as it is."]
    groups = sum(a.groups for a in found.values())
    undecided = sum(a.undecided for a in found.values())
    changed = sum(a.changed for a in found.values())
    lines = [
        f"Groups of the same evidence with differing awards: {groups}. Decided"
        f" again: {groups - undecided}. Changed: {changed}. Could not be decided:"
        f" {undecided}.",
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
    scored: Sequence[tuple[Stored, list[Score]]],
    raised: Sequence[Flag],
    not_ingested: Sequence[str],
) -> str:
    """The state of every key part, the class audit, the second read's results,
    each page's maximum and the total, the students with parts not scored and why,
    the files not ingested, the flags by code, the pinned models, and the cost by
    role from the usage ledger."""
    maxima = page_points(key)
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
        for why in dict.fromkeys(x.withheld for x in scores if x.withheld):
            parts = ", ".join(x.part.id for x in scores if x.withheld == why)
            lines.append(f"- `{s.student.student_key}`: {parts}. {why}")
    if not_ingested:
        lines += ["", "## Files not ingested", ""]
        lines.append("The reason ingest gave for each is in `ingest_summary.json`.")
        lines.append("")
        lines += [f"- {code_span(file)}: {NOT_INGESTED}" for file in not_ingested]
    counts = Counter(f.code for f in raised)
    lines += ["", "## Flags", ""]
    if counts:
        lines += ["| Flag | Severity | Count |", "|---|---|---|"]
        lines += [
            f"| {c} | {SEVERITY[c]} | {counts[c]} |" for c in FlagCode if c in counts
        ]
    else:
        lines.append("No flags.")
    lines += ["", review_line(workspace, raised), "", "## Models", ""]
    if workspace.config_lock_toml.exists():
        lines += ["| Role | Model |", "|---|---|"]
        lock = read_lock(workspace.config_lock_toml)
        lines += [f"| {role} | {m.slug} |" for role, m in lock.models.items()]
        if lock.strict_privacy is False:
            lines += ["", STRICT_OFF_RUN]
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
) -> str:
    """`out/review.md` (design 8.4): an entry for each key part flagged, with its
    points and the number of students scored on it; then one for each student and
    part, or whole submission, flagged, with every flag on it, the proposed score,
    the transcript and links to the crops and the stored files. Students' entries
    come by their highest severity, the points at stake, the student key and the
    key's part order. Text a student wrote is only ever in a code span."""
    lines = ["# Review list", ""]
    if not raised:
        return "\n".join([*lines, NOTHING_TO_REVIEW]) + "\n"
    entries: dict[tuple[str | None, str | None], list[Flag]] = {}
    for f in raised:
        entries.setdefault((f.student_key, f.part_id), []).append(f)
    on_key = [p for p in key.parts if (None, p.id) in entries]
    if on_key:
        lines += ["## The key", ""]
    for part in on_key:
        count = sum(
            x.part.id == part.id and x.awarded is not None
            for _, xs in scored
            for x in xs
        )
        lines += [
            f"### {part.id}, worth {number(points_of(key, [part]))}",
            "",
            *_flagged(entries[None, part.id]),
            f"- Students scored on it: {count}.",
            "",
        ]
    order = [p.id for p in key.parts]

    def at_stake(entry: tuple[str, str | None]) -> tuple[int, Decimal, str, int]:
        who, part_id = entry
        worst = min(SEVERITIES.index(f.severity) for f in entries[entry])
        parts = [p for p in key.parts if p.id == part_id] if part_id else key.parts
        # A whole submission is at stake for the key's total, before any part.
        at = order.index(part_id) if part_id else -1
        return worst, -points_of(key, parts), who, at

    theirs = sorted(((w, p) for w, p in entries if w is not None), key=at_stake)
    if theirs:
        lines += ["## Students", ""]
    by_key = {s.student.student_key: (s, xs) for s, xs in scored}
    for who, part_id in theirs:
        stored, scores = by_key[who]
        found = entries[who, part_id]
        name = f"{code_span(stored.student.student_name)} (`{who}`)"
        if part_id is None:
            lines += [f"### {name}, the whole submission", "", *_flagged(found), ""]
            continue
        [s] = [x for x in scores if x.part.id == part_id]
        worth = _out_of(s.awarded, points_of(key, [s.part]), "not scored")
        lines += [f"### {name}, {part_id}", "", *_flagged(found)]
        lines.append(
            f"- Proposed score: {worth}." + (f" {s.withheld}" if s.withheld else "")
        )
        files: list[tuple[str, Path]] = []
        if first := stored.readings.get(part_id):
            lines += _transcript("First reading", first)
            files += [
                (f"crop {n}", workspace.crop_png(who, part_id, n))
                for n in range(1, len(first.crops) + 1)
            ]
            files.append(("reading", workspace.reading_json(who, problem_of(part_id))))
        second = stored.seconds.get(part_id)
        if (
            second
            and second.second
            and FlagCode.READERS_DISAGREE in {f.code for f in found}
        ):
            lines += _transcript(
                "Second reading, by another vendor's model", second.second
            )
        if part_id in stored.grades:
            files.append(("grade", workspace.grade_json(who, problem_of(part_id))))
        out = workspace.review_md.parent
        links = [
            f"[{t}]({p.relative_to(out, walk_up=True).as_posix()})" for t, p in files
        ]
        lines += [f"- Files: {', '.join(links) or 'none'}.", ""]
    return "\n".join(lines)


def _flagged(found: Sequence[Flag]) -> list[str]:
    return [f"- `{f.code}`, {f.severity}: {f.detail}" for f in found]


def _transcript(title: str, record: ReadingRecord) -> list[str]:
    """`record`'s lines and final answer, as stored, each in a code span."""
    reading = record.reading
    return [
        f"- {title}:",
        *(
            f"  - Line {n}: {code_span(text)}"
            for n, text in enumerate(reading.lines, 1)
        ),
        f"  - Final answer: {code_span(reading.final_answer.model_dump_json())}",
    ]
