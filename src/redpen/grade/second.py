"""The second read (design 7.4), after the read stage and before the grade stage. A
part the reader was unsure of (an answered reading, not typed, with a line marked
uncertain or poor legibility) is read again by the lock's `second` model, another
vendor's: the read stage's own call for the unit that holds it, with the same
images, prompt, schema and checks, one call per such unit. Code compares the two
readings part by part. The first reading is always the one graded; the result
changes only the part's flags, in the report stage, read from
`Workspace.second_json`."""

import asyncio
from collections.abc import Mapping
from typing import Literal

from redpen.grade.audit import as_answer, fold, gives, kind_of
from redpen.grade.read import (
    Unit,
    digest,
    problem_of,
    read_unit,
    unit_images,
    units,
    unsure,
)
from redpen.llm.client import Gateway
from redpen.llm.models import ModelInfo
from redpen.prepare.verify import AGREE, DIFFER, compare, vendor
from redpen.schemas.key import Key, KeyPart, Policies
from redpen.schemas.reading import (
    ProblemReadings,
    Reading,
    ReadingRecord,
    SecondReading,
    SecondReadings,
)
from redpen.schemas.spec import Spec
from redpen.schemas.submission import MapRecord
from redpen.workspace import Workspace, log_failure, read_model, write_model

ROLE = "second"


def vendor_problem(models: Mapping[str, ModelInfo]) -> str | None:
    """Why the pinned `second` model cannot read again, or None: it is of the
    reader's vendor, and with the reader's own model the call cache would answer,
    so the two would always agree. A role with no model is the privacy gate's."""
    reader, second = models.get("reader"), models.get(ROLE)
    if reader is None or second is None or vendor(reader.slug) != vendor(second.slug):
        return None
    return (
        f"the second reader {second.slug} is of the reader's vendor, so it cannot"
        f" check the reader {reader.slug}; pin a model of another vendor"
    )


async def read_again(
    workspace: Workspace,
    student: str,
    spec: Spec,
    key: Key,
    gateway: Gateway,
    model: ModelInfo,
    image_budget: int,
) -> list[SecondReading]:
    """Read again, with `model`, each unit of `student` that holds a part the reader
    was unsure of, compare each such part's two readings, and write the result to
    `Workspace.second_json` (none when no part is in doubt). A unit whose call fails,
    or whose answer still fails its checks, gets "failed": a fixed line and the
    traceback go to the run log, and the other units go on."""
    path = workspace.second_json(student)
    path.unlink(missing_ok=True)
    first: dict[str, ReadingRecord] = {}
    for problem in dict.fromkeys(problem_of(p.id) for p in spec.parts):
        if (file := workspace.reading_json(student, problem)).exists():
            first |= {
                r.reading.part_id: r for r in read_model(file, ProblemReadings).records
            }
    doubted = {
        i for i, r in first.items() if unsure(r.reading) and r.reading.medium != "typed"
    }
    if not doubted:
        return []
    mapped = read_model(workspace.map_json(student), MapRecord)
    parts = {p.id: p for p in key.parts}

    async def again(unit: Unit) -> list[SecondReading]:
        found: dict[str, ReadingRecord] = {}
        try:
            images = unit_images(workspace, student, unit, mapped.pages, image_budget)
            records = await read_unit(
                workspace, student, unit, images, gateway, ROLE, model
            )
            found = {r.reading.part_id: r for r in records}
        except Exception as error:  # any failure: the flags stay as they are
            what = f"second {student}: {unit.name} was not read a second time"
            log_failure(workspace, what, error)
        return [
            SecondReading(
                part_id=p.id,
                first=digest(first[p.id]),
                result=compared(
                    parts[p.id], key.policies, first[p.id].reading, found[p.id].reading
                )
                if p.id in found
                else "failed",
                second=found.get(p.id),
            )
            for p in unit.parts
            if p.id in doubted
        ]

    todo = [u for u in units(spec, mapped.map) if any(p.id in doubted for p in u.parts)]
    done = await asyncio.gather(*(again(u) for u in todo))
    records = [r for unit in done for r in unit]
    write_model(path, SecondReadings(records=records))
    return records


def compared(
    part: KeyPart, policies: Policies, first: Reading, second: Reading
) -> Literal["same", "differ", "not decided"]:
    """ "same", "differ" or "not decided", for two readings of `part`. They differ
    when one is blank or unreadable and the other is not. Where code can compare
    the part's final answer and a reading states one, `verify.compare` decides
    (options, numbers, expressions; a missing unit as it takes it). Otherwise they
    are the same when their lines and elements are equal, white space and letter
    case folded, and not decided when they are not."""
    if first.status != second.status:
        return "differ"
    kind = kind_of(part)
    if kind and (gives(kind, first.final_answer) or gives(kind, second.final_answer)):
        a, b = (as_answer(kind, r.final_answer) for r in (first, second))
        found = compare(kind, a, b, policies)
        return (
            "same" if found == AGREE else "differ" if found == DIFFER else "not decided"
        )
    return "same" if _written(first) == _written(second) else "not decided"


def _written(reading: Reading) -> tuple[list[str], list[list[str]]]:
    elements = [
        [fold(x or "") for x in (e.label, e.kind, e.direction, e.position)]
        for e in reading.elements
    ]
    return [fold(line) for line in reading.lines], elements
