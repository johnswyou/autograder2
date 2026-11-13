"""Running a class (design 5, 12 and 14.1).

`grade_class` maps, reads and grades the students of an ingest, one stage over the
whole class before the next, the students concurrent inside a stage as tasks on
one thread: pdfium, which the stages open PDFs with, is not safe across threads. A
student's failure in a stage, or in one unit of a stage, is recorded with its
traceback in the run log and the class goes on; the later stages run on whatever
files that student has. Resuming is the call cache's work: nothing is skipped
because a file exists.

`privacy_problems` is the gate before the first student page is sent, and
`estimate` is the dry run's cost estimate."""

import asyncio
import math
import sqlite3
import traceback
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress

from redpen.config import Config, Lock
from redpen.grade.grade import grade_student
from redpen.grade.map import map_student
from redpen.grade.read import problem_of, read_student
from redpen.imaging import render
from redpen.llm.client import Gateway
from redpen.llm.ledger import Ledger, Usage
from redpen.llm.models import ModelInfo, recheck_providers
from redpen.schemas.key import Key
from redpen.schemas.spec import CHOICE_TYPES, Part, Spec
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace

# Each grade stage and the role whose model it sends student pages to.
STAGES = {"map": "mapper", "read": "reader", "grade": "grader"}
NO_USE = Usage(0, 0, Decimal(0))


async def privacy_problems(lock: Lock, api_key: str, strict: bool) -> list[str]:
    """Why student pages may not go to the grade roles' pinned endpoints now, or
    `[]` when they may. A check that cannot be completed is a problem: the gate
    fails closed (design 14.1)."""
    missing = [role for role in STAGES.values() if role not in lock.models]
    if missing:
        return [f"{role}: no model is pinned in the lock" for role in missing]
    models = {role: lock.models[role] for role in STAGES.values()}
    try:
        found = await recheck_providers(models, api_key, strict_privacy=strict)
    except Exception as error:  # an answer the check could not take apart
        return [f"the pinned endpoints could not be verified ({type(error).__name__})"]
    return [
        problem
        for role in models
        for problem in found.get(role, [f"{role}: its endpoints were not verified"])
    ]


async def grade_class(
    workspace: Workspace,
    spec: Spec,
    key: Key,
    models: Mapping[str, ModelInfo],
    config: Config,
    api_key: str,
    students: Sequence[str],
) -> list[tuple[str, str]]:
    """Map, read and grade `students` (keys of ingested students), showing each
    stage's progress and running cost. Returns each (stage, student) that failed."""
    budget = config.image_budget
    runs: dict[str, Callable[[Gateway, str], Awaitable[object]]] = {
        "map": lambda g, s: map_student(
            workspace, s, spec, g, models["mapper"], budget
        ),
        "read": lambda g, s: read_student(
            workspace, s, spec, g, models["reader"], budget
        ),
        "grade": lambda g, s: grade_student(workspace, s, key, g, models["grader"]),
    }
    failed: list[tuple[str, str]] = []
    progress = Progress(
        "{task.description}",
        MofNCompleteColumn(),
        "{task.fields[cost]}",
        console=Console(stderr=True),
    )
    tasks = {
        stage: progress.add_task(stage, total=len(students), cost="$0")
        for stage in runs
    }
    strict = config.strict_privacy
    async with Gateway(workspace.cache_sqlite, api_key, strict_privacy=strict) as gw:
        with closing(sqlite3.connect(workspace.cache_sqlite)) as db, progress:
            ledger = Ledger(db)
            before = ledger.totals_by_role()
            # A student's stage loads every one of their pages before its first
            # request waits on the gateway, so no more start than it can serve.
            at_once = asyncio.Semaphore(gw.max_concurrency)

            async def one(stage: str, student: str) -> None:
                try:
                    async with at_once:
                        await runs[stage](gw, student)
                except Exception as error:  # one student never stops the class
                    failed.append((stage, student))
                    log_failure(workspace, f"{stage} {student}", error)
                role = STAGES[stage]
                now = ledger.totals_by_role().get(role, NO_USE).cost
                spent = now - before.get(role, NO_USE).cost
                progress.update(tasks[stage], advance=1, cost=f"${spent:.4f}")

            for stage in runs:
                await asyncio.gather(*(one(stage, s) for s in students))
    return failed


def log_failure(workspace: Workspace, what: str, error: BaseException) -> None:
    """Append `error`'s traceback, under `what` and the time, to the run log."""
    with workspace.run_log.open("a", encoding="utf-8") as log:
        log.write(f"--- {datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ} {what}\n")
        log.write("".join(traceback.format_exception(error)))


# --- the dry run's estimate ----------------------------------------------------------

# Per call, the tokens sent besides images (instructions, outline and JSON parts)
# and the tokens returned, reasoning included: the means over the calls recorded for
# the fixtures (task 2.7 report).
TOKENS: dict[str, tuple[int, int]] = {
    "parse page": (3116, 349),
    "stitch": (3806, 1591),
    "solve": (3644, 579),
    "rubric": (3686, 1318),
    "map page": (3353, 637),
    "resolve": (3916, 748),
    "read": (4684, 461),
    "grade": (5703, 596),
}
# A mapped region as sent, in tokens: the fixture means for the reader (choice
# marks among them) and for the grader.
READ_REGION = 260
GRADE_CROP = 383
# An image costs one token per 28 x 28 px patch (Anthropic's vision docs; the
# fixture calls fit it exactly). shortcut: Claude's rule; measure again when a role
# is pinned to another vendor.
PATCH = 28


@dataclass(frozen=True)
class Shape:
    """The counts an assignment's calls follow from, every part answered: per
    assignment for solve and rubric, per student for read and grade."""

    problems: float  # solve calls
    solve_pages: float  # page images over all solve calls
    rubrics: float  # rubric calls: problems with a part that is not a choice
    reads: float  # read calls: pages with choice parts, problems with other parts
    read_pages: float  # whole pages shown over all read calls
    regions: float  # mapped regions read
    grades: float  # grade calls
    crops: float  # crops sent to the grader


# The fixture assignment's shape (4 pages) per blank page, for a blank not parsed.
PER_PAGE = Shape(1.25, 1.75, 0.5, 0.75, 1.25, 2.25, 0.5, 1.5)


@dataclass
class Cost:
    calls: float = 0
    tokens_in: float = 0
    tokens_out: float = 0
    usd: Decimal = Decimal(0)


def shape(spec: Spec) -> Shape:
    """The counts the calls for `spec` follow from, as the stages group parts."""
    problems: defaultdict[str, list[Part]] = defaultdict(list)
    for part in spec.parts:
        problems[problem_of(part.id)].append(part)
    free = [
        others
        for parts in problems.values()
        if (others := [p for p in parts if p.response_type not in CHOICE_TYPES])
    ]
    choice_pages = {p.home_page for p in spec.parts if p.response_type in CHOICE_TYPES}
    return Shape(
        problems=len(problems),
        solve_pages=sum(
            len(
                {p.home_page for p in parts}
                | {f.page for p in parts for f in p.figures}
            )
            for parts in problems.values()
        ),
        rubrics=len(free),
        reads=len(choice_pages) + len(free),
        read_pages=len(choice_pages) + sum(len({p.home_page for p in f}) for f in free),
        regions=len(spec.parts),
        grades=len(free),
        crops=sum(map(len, free)),
    )


def blank_sizes(blank: Path) -> list[tuple[int, int]]:
    """The size of each page image of `blank`, rendered as `parse` renders it."""
    with render.open_pdf(blank) as pdf:
        kind = "classkick" if render.classkick_sidebars(pdf) else "other"
        return [render.render_page(page, kind).image.size for page in pdf]


def image_tokens(size: tuple[int, int], budget: int) -> int:
    """What an image of `size` costs once reduced to fit the image budget."""
    width, height = size
    scale = min(1.0, budget / max(width, height))
    return math.ceil(round(width * scale) / PATCH) * math.ceil(
        round(height * scale) / PATCH
    )


def estimate(
    models: Mapping[str, ModelInfo],
    blank: Sequence[tuple[int, int]],
    spec: Spec | None,
    keyed: bool,
    students: Sequence[IngestRecord],
    budget: int,
) -> dict[str, Cost]:
    """What prepare (nothing when `keyed`), map, read and grade will cost, by stage,
    from the blank's page sizes, the spec (or `PER_PAGE` without one), the students'
    pages and the models' prices. Nothing is sent."""
    found = (
        shape(spec)
        if spec
        else Shape(
            **{f.name: getattr(PER_PAGE, f.name) * len(blank) for f in fields(Shape)}
        )
    )
    costs = {stage: Cost() for stage in ("prepare", *STAGES)}

    def add(stage: str, role: str, kind: str, calls: float, images: float = 0) -> None:
        text, out = TOKENS[kind]
        sent, got = calls * text + images, calls * out
        cost = costs[stage]
        cost.calls += calls
        cost.tokens_in += sent
        cost.tokens_out += got
        cost.usd += Decimal(round(sent)) * models[role].prompt_price
        cost.usd += Decimal(round(got)) * models[role].completion_price

    pages = [image_tokens(size, budget) for size in blank]
    page = sum(pages) / len(pages)
    if not keyed:
        if spec is None:
            add("prepare", "parser", "parse page", len(pages), sum(pages))
            add("prepare", "parser", "stitch", 1)
        add("prepare", "solver", "solve", found.problems, found.solve_pages * page)
        add("prepare", "solver", "rubric", found.rubrics)
    for student in students:
        mine = [image_tokens((p.width_px, p.height_px), budget) for p in student.pages]
        # A slide export's page is shown beside the same page of the blank.
        shown = [
            pages[p.number - 1]
            for p in student.pages
            if student.source_kind == "classkick" and p.number <= len(pages)
        ]
        add("map", "mapper", "map page", len(mine), sum(mine) + sum(shown))
        add("map", "mapper", "resolve", 1)
        overview = sum(mine) / len(mine)
        regions = found.read_pages * overview + found.regions * READ_REGION
        add("read", "reader", "read", found.reads, regions)
        add("grade", "grader", "grade", found.grades, found.crops * GRADE_CROP)
    return costs
