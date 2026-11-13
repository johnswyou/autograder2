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
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from redpen.config import read_lock
from redpen.grade.grade import part_points
from redpen.grade.read import problem_of
from redpen.imaging import render
from redpen.llm.ledger import Ledger, Usage
from redpen.prepare.key import number, page_points, part_hash
from redpen.schemas.flags import SEVERITY, Flag, FlagCode
from redpen.schemas.grade import GradeRecord, ProblemGrades
from redpen.schemas.key import Key, KeyPart
from redpen.schemas.reading import ProblemReadings, ReadingRecord
from redpen.schemas.spec import CHOICE_TYPES
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, read_model

type Status = Literal["incomplete", "provisional", "final"]

# The four reasons a part is withheld, never scored zero (design 9), in fixed words.
UNREADABLE = "The work could not be read."
NO_OPTION = "No chosen option could be read."
OTHER_KEY = "It was graded under another version of the key and is to be graded again."
UNGRADED = "No reading or no grade was recorded for it."
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
    FlagCode.LOW_RESOLUTION: "The work was read at the full resolution of the"
    " source and is still hard to read.",
    FlagCode.ADDRESSED_TO_GRADER: "The response holds text addressed to the grader.",
    FlagCode.TRANSCRIPT_DISPUTE: "The grader believes the reading is wrong.",
    FlagCode.CODE_OVERRULED: "The grader overruled the computed answer check.",
}

# A spreadsheet runs a cell that begins with one of these as a formula.
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")

NO_USE = Usage(0, 0, Decimal(0))


@dataclass(frozen=True)
class Stored:
    """One student's ingest record, and each part's reading and grade record that
    are stored, by part id."""

    student: IngestRecord
    readings: dict[str, ReadingRecord]
    grades: dict[str, GradeRecord]


@dataclass(frozen=True)
class Score:
    """One part of one student: the grade record that counts and the points it
    awards, or, when the part is withheld, why."""

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
    feedback file per student (any other is removed) and the run report, which
    also lists the files of `not_ingested` (names inside the submissions, never
    opened here). Returns every flag derived."""
    problems, pages = columns(workspace, key)
    stored = [load(workspace, key, s) for s in students]
    raised = flags(key, stored)
    scored = [(s, score(key, s)) for s in stored]
    maximum = points_of(key.parts)
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
        text = _feedback(title, problems, scores, total, maximum)
        feedback.append((workspace.feedback_md(s.student.student_key), text))
    run_report = _run_report(workspace, key, pages, scored, raised, not_ingested)
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
    key's top-level problems whose file exists."""
    readings: dict[str, ReadingRecord] = {}
    grades: dict[str, GradeRecord] = {}
    who = student.student_key
    for problem in dict.fromkeys(problem_of(p.id) for p in key.parts):
        if found := _read(workspace.reading_json(who, problem), ProblemReadings):
            readings |= {r.reading.part_id: r for r in found.records}
        if graded := _read(workspace.grade_json(who, problem), ProblemGrades):
            grades |= {r.grade.part_id: r for r in graded.records}
    return Stored(student, readings, grades)


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
    """Each key part's score for the stored student, in key order."""
    scores: list[Score] = []
    for part in key.parts:
        why = withheld(key, part, stored)
        record = None if why else stored.grades[part.id]
        awarded = part_points(record, key)[0] if record else None
        scores.append(Score(part, record, awarded, why))
    return scores


def flags(key: Key, students: Sequence[Stored]) -> list[Flag]:
    """Every flag the stored files give (this version's set): a `key_unverified`
    for each unverified key part, on no student, then each student's, part by part
    in key order. All are open: nothing closes a flag yet."""
    raised = [
        Flag(
            code=FlagCode.KEY_UNVERIFIED,
            part_id=p.id,
            student_key=None,
            detail=DETAIL[FlagCode.KEY_UNVERIFIED],
        )
        for p in key.parts
        if p.state == "unverified"
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
        hard = reading.status == "answered" and (
            bool(reading.uncertain) or reading.legibility == "poor"
        )
        if reading.status == "unreadable":
            raised.append((FlagCode.UNREADABLE, UNREADABLE))
        if hard:
            raised.append((FlagCode.HARD_TO_READ, DETAIL[FlagCode.HARD_TO_READ]))
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
    elif why != UNREADABLE:
        raised.append((FlagCode.INCOMPLETE, why))
    return raised


def _read[M: BaseModel](path: Path, model: type[M]) -> M | None:
    try:
        return read_model(path, model)
    except FileNotFoundError:
        return None


def points_of(parts: Iterable[KeyPart]) -> Decimal:
    return sum((c.points for p in parts for c in p.criteria), Decimal(0))


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
            sum_of(mine), points_of(s.part for s in mine), "not fully scored"
        )
        # Every problem id is "Q" and its number (the PartId pattern).
        lines += ["", f"## Problem {problem[1:]}: {worth}"]
        for s in mine:
            if len(mine) > 1 or s.part.id != problem:
                points = _out_of(s.awarded, points_of([s.part]), "not scored")
                lines += ["", f"**{s.part.label}: {points}**"]
            lines += ["- " + _one_line(note) for note in _notes(s)]
    return "\n".join(lines) + "\n"


def _one_line(note: str) -> str:
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


def _run_report(
    workspace: Workspace,
    key: Key,
    pages: Sequence[int],
    scored: Sequence[tuple[Stored, list[Score]]],
    raised: Sequence[Flag],
    not_ingested: Sequence[str],
) -> str:
    """The state of every key part, each page's maximum and the total, the students
    with parts not scored and why, the files not ingested, the flags by code, the
    pinned models, and the cost by role from the usage ledger."""
    maxima = page_points(key)
    lines = ["# Run report", "", "## Key", "", "| Part | Page | Points | State |"]
    lines.append("|---|---|---|---|")
    lines += [
        f"| {p.id} | {p.home_page} | {number(points_of([p]))} | {p.state} |"
        for p in key.parts
    ]
    lines += ["", "## Pages", "", "| Page | Maximum |", "|---|---|"]
    lines += [f"| {n} | {number(maxima.get(n, Decimal(0)))} |" for n in pages]
    lines += [f"| Total | {number(points_of(key.parts))} |", "", "## Students", ""]
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
    lines += ["", "## Models", ""]
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
