"""`redpen eval` (design 15): the tool's scores against a person's for the same
class, from stored files only. No model call. Every score, difference, mean and
share is a Decimal until it is printed.

The tool's side is the report stage's own: its columns, scores and flags. The
students are the ingest summary's; `students/` is never listed. A name is written
only to `out/eval.md` (as a code span) and `out/eval.json`, and never used as a path.
Nothing a student or a model wrote is copied: the listing gives paths."""

import csv
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Literal

from redpen.grade.ingest import fold
from redpen.grade.read import problem_of
from redpen.prepare.key import number
from redpen.report.report import (
    Stored,
    code_span,
    columns,
    flags,
    load,
    points_of,
    score,
    sum_of,
)
from redpen.schemas.common import Schema
from redpen.schemas.flags import Flag
from redpen.schemas.key import Key, KeyPart
from redpen.schemas.spec import CHOICE_TYPES
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace

type Form = Literal["page", "problem"]

# Columns of the tool's score files that hold no score: ignored in a gold file.
IGNORED = ("total", "maximum", "status", "open_flags")
# A total this close to the person's, as a share of the assignment's maximum.
TOTAL_WITHIN = Decimal("0.05")
# A cell further apart than this share of its maximum is a large disagreement.
LARGE = Decimal("0.25")
# A score as a person types one: ASCII digits, and up to six after a point, so that
# every sum and difference stays exact in Decimal's 28 digits.
PLAIN = re.compile(r"[0-9]+(?:\.[0-9]{1,6})?")
# Printed to the hundredth of a point and the tenth of a percent, halves up.
CENT, TENTH = Decimal("0.01"), Decimal("0.1")


class GoldError(Exception):
    """The gold file is refused, in one line. It quotes no cell of the file, and any
    header name of it only as a code span."""


class Share(Schema):
    count: int
    of: int
    share: Decimal | None  # count / of; None when of is 0


class Column(Schema):
    """One page or problem: the mean absolute difference over its compared cells,
    in points and as a share of its maximum."""

    column: str
    maximum: Decimal
    cells: int
    mean: Decimal | None
    share: Decimal | None


class Total(Schema):
    """One matched student's totals over the cells both sides scored."""

    student: str
    student_key: str
    cells: int
    tool: Decimal
    person: Decimal
    difference: Decimal  # tool minus person


class Disagreement(Schema):
    student: str
    student_key: str
    column: str
    tool: Decimal
    person: Decimal
    difference: Decimal
    flags: list[str]  # "<code> on <part>", the key's flags included
    readings: list[str]  # paths inside the workspace
    grades: list[str]
    crops: list[str]


class Evaluation(Schema):
    by: Form
    point_step: Decimal
    maximum: Decimal  # the assignment's
    unknown: list[str]  # gold names matching no student
    no_gold: list[str]  # students with no gold row
    doubled: list[str]  # students with more than one gold row
    columns: list[Column]  # those worth points
    # Each cell of the matched students is in exactly one of these five counts.
    cells: int  # compared: both sides scored, in a column worth points
    withheld: int  # withheld by the tool, scored by the person
    tool_only: int  # scored by the tool, empty in the gold
    neither: int  # empty on both sides
    no_points: int  # in a column worth no points
    mean: Decimal | None
    share: Decimal | None  # sum of |difference| over the sum of the maximums
    within_step: Share
    bias: Decimal | None
    bias_share: Decimal | None
    total_bias: Decimal | None
    total_bias_share: Decimal | None  # of the assignment's maximum
    totals_within: Share
    incomplete_totals: int  # matched students left out of the two totals measures
    choice: Share  # cells of only choice parts that agree exactly
    unflagged: Share  # large and no flag on the cell's parts, the key's counted
    unflagged_named: Share  # large and no flag naming the student there
    totals: list[Total]
    largest: list[Disagreement]


@dataclass(frozen=True)
class _Pair:
    """One cell both sides scored."""

    stored: Stored
    column: str
    parts: list[KeyPart]
    tool: Decimal
    person: Decimal
    maximum: Decimal
    flags: list[Flag]  # the open flags on its parts: the key's and the student's

    @property
    def difference(self) -> Decimal:
        return self.tool - self.person


type Gold = list[tuple[str, dict[str, Decimal | None]]]  # name, and score by column


def evaluate(
    workspace: Workspace,
    key: Key,
    students: Sequence[IngestRecord],
    gold: Path,
    top: int = 20,
) -> Evaluation:
    """Compare the stored scores of `students` with the gold file's. A gold row
    is matched to a student by name with case and spacing ignored, as ingest
    matches a roster. Raises GoldError for a refused gold file or no match."""
    problems, pages = columns(workspace, key)
    forms: dict[Form, dict[str, list[KeyPart]]] = {
        "page": {
            f"page {n}": [p for p in key.parts if p.home_page == n] for n in pages
        },
        "problem": {
            q: [p for p in key.parts if problem_of(p.id) == q] for q in problems
        },
    }
    maxima: dict[Form, dict[str, Decimal]] = {
        f: {c: points_of(key, parts) for c, parts in form.items()}
        for f, form in forms.items()
    }
    by, rows = _read_gold(gold, maxima)
    found: dict[str, list[dict[str, Decimal | None]]] = {}
    for name, cells in rows:
        found.setdefault(_fold(name), []).append(cells)
    names = {_fold(s.student_name) for s in students}
    matched = [
        (s, found[_fold(s.student_name)][0])
        for s in students
        if len(found.get(_fold(s.student_name), [])) == 1
    ]
    doubled = [
        s.student_name
        for s in students
        if len(found.get(_fold(s.student_name), [])) > 1
    ]
    if not matched and doubled:
        raise GoldError(
            f"No row of {gold} matches exactly one student: each student it names"
            " has more than one row."
        )
    if not matched:
        raise GoldError(f"No row of {gold} names a student of the ingest summary.")

    stored = [load(workspace, key, s) for s, _ in matched]
    raised = flags(key, stored)
    pairs: list[_Pair] = []
    left = Counter[str]()  # the cells not compared, by why
    for st, (_, cells) in zip(stored, matched, strict=True):
        scores = score(key, st)
        who = (None, st.student.student_key)
        for column, parts in forms[by].items():
            ids = {p.id for p in parts}
            tool = sum_of(x for x in scores if x.part.id in ids)
            person, maximum = cells[column], maxima[by][column]
            if not maximum:
                left["no_points"] += 1
            elif person is None:
                left["neither" if tool is None else "tool_only"] += 1
            elif tool is None:
                left["withheld"] += 1
            else:
                on = [f for f in raised if f.part_id in ids and f.student_key in who]
                pairs.append(_Pair(st, column, parts, tool, person, maximum, on))

    totals = [_total(st, [p for p in pairs if p.stored is st]) for st in stored]
    # A total is measured against the assignment's maximum, so it counts only when
    # every column that can be scored was compared.
    scorable = sum(1 for m in maxima[by].values() if m)
    complete = [t for t in totals if t.cells == scorable]
    maximum = points_of(key, key.parts)
    signed = _sum(t.difference for t in complete)
    large = [p for p in pairs if abs(p.difference) > LARGE * p.maximum]
    choice = [p for p in pairs if all(x.response_type in CHOICE_TYPES for x in p.parts)]
    ranked = sorted(
        (p for p in pairs if p.difference), key=lambda p: -abs(p.difference)
    )
    return Evaluation(
        by=by,
        point_step=key.policies.point_step,
        maximum=maximum,
        unknown=[name for name, _ in rows if _fold(name) not in names],
        no_gold=[
            s.student_name for s in students if _fold(s.student_name) not in found
        ],
        doubled=doubled,
        columns=[
            _column(
                column, maxima[by][column], [p for p in pairs if p.column == column]
            )
            for column in forms[by]
            if maxima[by][column]
        ],
        cells=len(pairs),
        withheld=left["withheld"],
        tool_only=left["tool_only"],
        neither=left["neither"],
        no_points=left["no_points"],
        mean=_ratio(_sum(abs(p.difference) for p in pairs), len(pairs)),
        share=_ratio(
            _sum(abs(p.difference) for p in pairs), _sum(p.maximum for p in pairs)
        ),
        within_step=_share(
            sum(abs(p.difference) <= key.policies.point_step for p in pairs),
            len(pairs),
        ),
        bias=_ratio(_sum(p.difference for p in pairs), len(pairs)),
        bias_share=_ratio(
            _sum(p.difference for p in pairs), _sum(p.maximum for p in pairs)
        ),
        total_bias=_ratio(signed, len(complete)),
        total_bias_share=_ratio(signed, len(complete) * maximum),
        totals_within=_share(
            sum(abs(t.difference) <= TOTAL_WITHIN * maximum for t in complete),
            len(complete),
        ),
        incomplete_totals=len(totals) - len(complete),
        choice=_share(sum(not p.difference for p in choice), len(choice)),
        unflagged=_share(sum(not p.flags for p in large), len(pairs)),
        unflagged_named=_share(
            sum(not any(f.student_key for f in p.flags) for p in large), len(pairs)
        ),
        totals=totals,
        largest=[_disagreement(workspace, p) for p in ranked[:top]],
    )


def _read_gold(
    path: Path, maxima: Mapping[Form, Mapping[str, Decimal]]
) -> tuple[Form, Gold]:
    """The gold file's form, told from its header, and each row's student and
    scores: a Decimal from 0 to the column's maximum, or None for an empty cell.
    Rows count from 1, the header's, as a spreadsheet numbers them."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.reader(file)
            header = next(reader, list[str]())
            given = sorted(c for c in header if c not in IGNORED)
            by: Form | None = next(
                (f for f, m in maxima.items() if given == sorted(["student", *m])),
                None,
            )
            if by is None:
                raise GoldError(_misfit(path, header, maxima))
            wanted = maxima[by]
            rows: Gold = []
            for n, row in enumerate(reader, 2):
                if not "".join(row).strip():
                    continue
                if len(row) != len(header):
                    raise GoldError(
                        f"Row {n} of {path} has {len(row)} cells and its header"
                        f" {len(header)}."
                    )
                cells = dict(zip(header, row, strict=True))
                scores = {
                    c: _gold_points(cells[c], m, f"Row {n}, column {c} of {path}")
                    for c, m in wanted.items()
                }
                rows.append((cells["student"], scores))
    except (OSError, UnicodeError, csv.Error) as error:
        raise GoldError(
            f"{path} cannot be read as a UTF-8 CSV file ({type(error).__name__})."
        ) from None
    return by, rows


def _misfit(
    path: Path, header: list[str], maxima: Mapping[Form, Mapping[str, Decimal]]
) -> str:
    """Why `header` fits neither form: the nearest form's columns, and the header's
    that are missing, not expected or repeated."""
    nearest = max(maxima, key=lambda f: len(set(header) & {"student", *maxima[f]}))
    wanted = ["student", *maxima[nearest]]
    missing = [c for c in wanted if c not in header]
    extra = [c for c in dict.fromkeys(header) if c not in wanted and c not in IGNORED]
    repeated = [c for c in wanted if header.count(c) > 1]

    def listed(found: list[str]) -> str:
        return ", ".join(map(code_span, found)) or "none"

    text = (
        f"{path} has the columns of neither score file. Nearest is by {nearest}:"
        f" {', '.join(wanted)}, and any of {', '.join(IGNORED)}, which are ignored."
        f" Missing: {listed(missing)}. Not expected: {listed(extra)}."
    )
    return text + (f" Repeated: {listed(repeated)}." if repeated else "")


def _gold_points(text: str, maximum: Decimal, where: str) -> Decimal | None:
    plain = text.strip()
    if not plain:
        return None
    if not PLAIN.fullmatch(plain):
        raise GoldError(f"{where} is not a number of points.")
    value = Decimal(plain)
    if value > maximum:
        raise GoldError(
            f"{where} is above the key's maximum there, {number(maximum)}."
            " Is the gold on another point scale?"
        )
    return value


def _fold(name: str) -> str:
    """`name` as ingest compares names: one Unicode form, case and spacing ignored."""
    return fold(unicodedata.normalize("NFC", name))


def _sum(values: Iterable[Decimal]) -> Decimal:
    return sum(values, Decimal(0))


def _ratio(part: Decimal | int, whole: Decimal | int) -> Decimal | None:
    return Decimal(part) / Decimal(whole) if whole else None


def _share(count: int, of: int) -> Share:
    return Share(count=count, of=of, share=_ratio(count, of))


def _column(column: str, maximum: Decimal, pairs: Sequence[_Pair]) -> Column:
    absolute = _sum(abs(p.difference) for p in pairs)
    return Column(
        column=column,
        maximum=maximum,
        cells=len(pairs),
        mean=_ratio(absolute, len(pairs)),
        share=_ratio(absolute, len(pairs) * maximum),
    )


def _total(stored: Stored, pairs: Sequence[_Pair]) -> Total:
    return Total(
        student=stored.student.student_name,
        student_key=stored.student.student_key,
        cells=len(pairs),
        tool=_sum(p.tool for p in pairs),
        person=_sum(p.person for p in pairs),
        difference=_sum(p.difference for p in pairs),
    )


def _disagreement(workspace: Workspace, pair: _Pair) -> Disagreement:
    who = pair.stored.student.student_key
    ids = [p.id for p in pair.parts]
    problems = list(dict.fromkeys(problem_of(i) for i in ids))

    def inside(path: Path) -> str:
        return path.relative_to(workspace.root).as_posix()

    return Disagreement(
        student=pair.stored.student.student_name,
        student_key=who,
        column=pair.column,
        tool=pair.tool,
        person=pair.person,
        difference=pair.difference,
        flags=[f"{f.code} on {f.part_id}" for f in pair.flags],
        readings=[inside(workspace.reading_json(who, q)) for q in problems],
        grades=[inside(workspace.grade_json(who, q)) for q in problems],
        crops=[
            inside(workspace.crop_png(who, i, n))
            for i in ids
            for n in range(1, len(pair.stored.readings[i].crops) + 1)
        ],
    )


# --- printing ------------------------------------------------------------------------


def _points(value: Decimal | None) -> str:
    return "-" if value is None else number(value.quantize(CENT, ROUND_HALF_UP))


def _percent(value: Decimal | None) -> str:
    if value is None:
        return "-"
    return f"{number((100 * value).quantize(TENTH, ROUND_HALF_UP))}%"


def _count(share: Share, noun: str) -> str:
    return f"{share.count} of {share.of} {noun} ({_percent(share.share)})"


def table(e: Evaluation) -> list[str]:
    """The measures and the counts, for the terminal: no name and no student key."""
    grid = e.cells + e.withheld + e.tool_only + e.neither + e.no_points
    sums = " It compares page sums, so two errors on one page can cancel."
    return [
        f"Compared by {e.by}: {len(e.totals)} students matched to one gold row each.",
        f"Gold rows naming no student: {len(e.unknown)}. Students with no gold row:"
        f" {len(e.no_gold)}. Students with more than one gold row: {len(e.doubled)}.",
        f"Their {grid} cells:",
        f"  compared, scored on both sides: {e.cells}",
        f"  withheld by the tool, scored by the person: {e.withheld}",
        f"  scored by the tool, empty in the gold: {e.tool_only}",
        f"  empty on both sides: {e.neither}",
        f"  in columns worth no points: {e.no_points}",
        "",
        f"  {e.by:<10}{'maximum':>9}{'cells':>7}{'mean |difference|':>19}{'share':>9}",
        *(
            f"  {c.column:<10}{number(c.maximum):>9}{c.cells:>7}"
            f"{_points(c.mean):>19}{_percent(c.share):>9}"
            for c in e.columns
        ),
        f"  {'overall':<10}{'':>9}{e.cells:>7}{_points(e.mean):>19}"
        f"{_percent(e.share):>9}",
        "",
        f"Within one point step ({number(e.point_step)}):"
        f" {_count(e.within_step, 'cells')}.",
        f"Signed bias, tool minus person, on cells: {_points(e.bias)} points,"
        f" {_percent(e.bias_share)} of the compared cells' maximum.",
        f"Signed bias on totals: {_points(e.total_bias)} points,"
        f" {_percent(e.total_bias_share)} of the assignment's maximum"
        f" ({number(e.maximum)}).",
        "Totals within 5% of the assignment's maximum:"
        f" {_count(e.totals_within, 'students')}.",
        "Students left out of the two totals measures, their totals resting on"
        f" fewer than all {len(e.columns)} scorable cells: {e.incomplete_totals}.",
        f"Multiple-choice agreement: {_count(e.choice, 'cells of only choice parts')}."
        + (sums if e.by == "page" else ""),
        "Large disagreements (over 25% of the cell's maximum) not flagged:",
        f"  counting every flag: {_count(e.unflagged, 'cells')}.",
        "  counting only flags that name the student:"
        f" {_count(e.unflagged_named, 'cells')}.",
    ]


def markdown(e: Evaluation) -> str:
    """`out/eval.md`: the measures, each student's totals, the rows and students
    left out, and the largest disagreements with the paths of their files."""
    lines = ["# Evaluation", "", "## Measures", "", "```", *table(e), "```"]
    lines += ["", "## Totals over the cells both sides scored, tool minus person", ""]
    lines += [
        f"- {code_span(t.student)}: tool {number(t.tool)}, person {number(t.person)},"
        f" difference {number(t.difference)}, over {t.cells} cells."
        for t in e.totals
    ]
    for title, names in [
        ("Gold rows naming no student", e.unknown),
        ("Students with no gold row", e.no_gold),
        ("Students with more than one gold row, left out", e.doubled),
    ]:
        lines += ["", f"## {title}", ""]
        lines += [f"- {code_span(name)}" for name in names] or ["None."]
    lines += ["", "## Largest disagreements", ""]
    for n, d in enumerate(e.largest, 1):
        lines += [
            f"{n}. {code_span(d.student)}, {d.column}: tool {number(d.tool)}, person"
            f" {number(d.person)}, difference {number(d.difference)}.",
            f"   - Flags: {', '.join(d.flags) or 'none'}.",
            *(
                f"   - {what}: {', '.join(f'`{p}`' for p in paths) or 'none'}."
                for what, paths in [
                    ("Readings", d.readings),
                    ("Grades", d.grades),
                    ("Crops", d.crops),
                ]
            ),
        ]
    if not e.largest:
        lines.append("None: every compared cell agrees.")
    return "\n".join(lines) + "\n"
