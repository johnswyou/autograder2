"""The second read (design 7.4), after the read stage and before the grade stage. A
unit holding a part the reader was unsure of (an answered reading, not typed, with a
line marked uncertain or poor legibility) is read again by the lock's `second`
model, another vendor's: the read stage's own call for the unit, with the same
images, prompt, schema and checks, one call per such unit, and no file of the read
stage written. Code compares the two readings of every part of the unit. The first
reading is always the one graded; the result changes only the part's flags, in the
report stage, read from `Workspace.second_json`."""

from collections.abc import Mapping
from typing import Literal

from autograder2.grade.audit import as_answer, fold, gives, kind_of
from autograder2.grade.read import (
    Unit,
    digest,
    problem_of,
    read_unit,
    unit_images,
    units,
    unsure,
)
from autograder2.llm.client import Gateway, StopError, finish_all
from autograder2.llm.models import ModelInfo
from autograder2.prepare.key import part_hash
from autograder2.prepare.verify import AGREE, DIFFER, compare, vendor
from autograder2.schemas.key import Key, KeyPart, Policies
from autograder2.schemas.reading import (
    ProblemReadings,
    Reading,
    ReadingRecord,
    SecondReading,
    SecondReadings,
)
from autograder2.schemas.spec import Spec
from autograder2.schemas.submission import MapRecord
from autograder2.workspace import Workspace, log_failure, read_model, write_model

ROLE = "second"
type Result = Literal["same", "differ", "not decided", "failed"]


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
    was unsure of, compare the two readings of every part of it, and write the
    results to `Workspace.second_json` (removed first; none when no part is in
    doubt). A unit whose call fails, or whose answer still fails its checks, gets
    "failed" on its parts in doubt: a fixed line and the traceback go to the run
    log, and the other units go on."""
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

    def stored(
        part: str, result: Result, second: ReadingRecord | None
    ) -> SecondReading:
        return SecondReading(
            part_id=part,
            part_hash=part_hash(parts[part], key.policies),
            first=digest(first[part]),
            doubted=part in doubted,
            result=result,
            second=second,
        )

    async def again(unit: Unit) -> list[SecondReading]:
        try:
            images = unit_images(
                workspace, student, unit, mapped.pages, image_budget, save=False
            )
            records = await read_unit(
                workspace, student, unit, images, gateway, ROLE, model
            )
        except Exception as error:  # any failure: the flags stay as they are
            what = f"second {student}: {unit.name} was not read a second time"
            if not isinstance(error, StopError):  # logged once, by the command
                log_failure(workspace, what, error)
            return [stored(p.id, "failed", None) for p in unit.parts if p.id in doubted]
        found = {r.reading.part_id: r for r in records}
        results: list[SecondReading] = []
        for p in unit.parts:
            if p.id in first:  # every part the first reader read
                one, other = first[p.id].reading, found[p.id].reading
                result = compared(parts[p.id], key.policies, one, other)
                results.append(stored(p.id, result, found[p.id]))
        return results

    todo = [u for u in units(spec, mapped.map) if any(p.id in doubted for p in u.parts)]
    done = await finish_all(again(u) for u in todo)
    records = [r for unit in done for r in unit]
    write_model(path, SecondReadings(records=records))
    return records


def compared(
    part: KeyPart, policies: Policies, first: Reading, second: Reading
) -> Literal["same", "differ", "not decided"]:
    """Whether two readings of `part` are the same, differ or are not decided.

    - A second reading that is answered but itself in doubt (`unsure`, or a `?` in a
      line, a label or the final answer) decides nothing.
    - They differ when one is blank or unreadable and the other is not.
    - Where code can compare the part's final answer and a reading states one,
      `verify.compare` decides (options, numbers, expressions; a missing unit as it
      takes it), with numbers the same only when equal: the key's tolerance plays no
      part. Under a policy that penalises significant figures equal numbers are not
      decided, since a reading keeps no trailing zeros.
    - Otherwise they are the same when their lines and elements are equal, white
      space and letter case folded, and something is written."""
    if second.status == "answered" and (unsure(second) or _unread(second)):
        return "not decided"
    if first.status != second.status:
        return "differ"
    kind = kind_of(part)
    if kind and (gives(kind, first.final_answer) or gives(kind, second.final_answer)):
        a, b = (as_answer(kind, r.final_answer) for r in (first, second))
        found = compare(kind, a, b, policies.model_copy(update={"rel_tol": 0.0}))
        if found == AGREE and kind == "numeric" and policies.penalise_sig_figs:
            return "not decided"
        return (
            "same" if found == AGREE else "differ" if found == DIFFER else "not decided"
        )
    written = _written(first)
    return "same" if written == _written(second) != ([], []) else "not decided"


def _unread(reading: Reading) -> bool:
    """Whether `reading` marks a character it could not read: a `?` in a line, an
    element's label or the final answer."""
    stated = reading.final_answer
    texts = [*reading.lines, *(e.label for e in reading.elements), *stated.choices]
    texts += [stated.unit or "", stated.expression or ""]
    return any("?" in text for text in texts)


def _written(reading: Reading) -> tuple[list[str], list[list[str]]]:
    elements = [
        [fold(x or "") for x in (e.label, e.kind, e.direction, e.position)]
        for e in reading.elements
    ]
    return [fold(line) for line in reading.lines], elements
