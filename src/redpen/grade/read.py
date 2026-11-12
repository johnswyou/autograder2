"""Read one mapped student (design 7.4, with the single-region render and the
ceiling of 7.7: no tiling, no second look). Multiple-choice and multiple-select
parts are read one submission page at a time, every other part one top-level
problem at a time. A call is sent the parts' text, each page of the unit whole as
an overview, and each mapped region rendered from the PDF. Nothing from the key
reaches a call."""

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor

import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
from PIL import Image

from redpen import prompts
from redpen.imaging import prep, render
from redpen.llm.client import Gateway, ask_checked, finish_all
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.schemas.common import Provenance, Schema
from redpen.schemas.reading import (
    Crop,
    FinalAnswer,
    ProblemReadings,
    Reading,
    ReadingRecord,
)
from redpen.schemas.spec import CHOICE_TYPES, Part, Spec
from redpen.schemas.submission import IngestRecord, Map, MapEntry, MapPage, MapRecord
from redpen.workspace import Workspace, read_model, write_model

ROLE = "reader"
PARAMS = {"max_tokens": 8000}
# A mapped box is grown by this share of the page on each side before it is
# rendered. The map's replay test checks its boxes cover the work grown by it.
MARGIN = 0.02
# The map's photo flag is trusted only on a page that is not almost all blank: on
# the fixtures, pages that are not photos are 98.4 to 99.5 % pure white, photo
# slides 46 to 70 % and scans 0 %, and the correction harms a page that is no photo.
WHITE_LEVEL = 250  # a grey level counted as blank paper or canvas
MOSTLY_WHITE = 0.9  # a share of such pixels at which a page is taken for no photo
NO_ANSWER = FinalAnswer(choices=[], value=None, unit=None, expression=None)


# What the reader returns for one unit.
class Readings(Schema):
    readings: list[Reading]


@dataclass(frozen=True)
class Unit:
    name: str  # `page <n>` for choices read by page, else the problem id
    parts: list[Part]
    entries: list[MapEntry]  # the map's entries for `parts`, in the map's order


@dataclass(frozen=True)
class UnitImages:
    items: list[Content]  # per image a line of JSON saying what it is, then the PNG
    crops: dict[str, list[Crop]]  # by part id, the regions its reading is made from


async def read_student(
    workspace: Workspace,
    student: str,
    spec: Spec,
    gateway: Gateway,
    model: ModelInfo,
    image_budget: int,
) -> dict[str, ReadingRecord]:
    """Read the mapped `student` and return every part's record, by part id. Each
    top-level problem's file is written as soon as all its parts are read. The
    units run concurrently; when one still fails its checks after one more call,
    `StageError` is raised once every other unit has finished, and that unit's
    problems are left with no file, not even an earlier one."""
    mapped = read_model(workspace.map_json(student), MapRecord)
    by_problem: defaultdict[str, list[str]] = defaultdict(list)
    for part in spec.parts:
        by_problem[problem_of(part.id)].append(part.id)
        crops = workspace.crop_png(student, part.id, 1).parent
        for earlier in crops.glob(f"{part.id}-*.png"):
            earlier.unlink()
    for problem in by_problem:
        workspace.reading_json(student, problem).unlink(missing_ok=True)
    system, version = prompts.load("read", "unit")
    session = f"{workspace.root.name}:read"
    done: dict[str, ReadingRecord] = {}

    def keep(records: Mapping[str, ReadingRecord]) -> None:
        done.update(records)
        for problem, ids in by_problem.items():
            if any(i in records for i in ids) and all(i in done for i in ids):
                file = ProblemReadings(records=[done[i] for i in ids])
                write_model(workspace.reading_json(student, problem), file)

    async def read(unit: Unit) -> None:
        images = unit_images(workspace, student, unit, mapped.pages, image_budget)
        fields = {"id", "label", "prompt", "response_type", "options"}
        outline = json.dumps(
            {"parts": [p.model_dump(include=fields) for p in unit.parts]}
        )

        async def ask(extra: Sequence[Content]) -> Readings:
            result = await gateway.call(
                ROLE,
                model,
                system,
                [outline],
                [*images.items, *extra],
                Readings,
                version,
                params=PARAMS,
                session_id=session,
            )
            return result.value

        found = await ask_checked(
            ask,
            lambda r: problems(unit.parts, r),
            f"the reading of {unit.name} of {student}",
        )
        provenance = Provenance(
            key_part_hash=None, prompt_version=version, model_slug=model.slug
        )
        keep(
            {
                r.part_id: ReadingRecord(
                    provenance=provenance, reading=r, crops=images.crops[r.part_id]
                )
                for r in found.readings
            }
        )

    todo = units(spec, mapped.map)
    sent = {p.id for unit in todo for p in unit.parts}
    keep({p.id: _blank(p.id) for p in spec.parts if p.id not in sent})
    await finish_all(read(unit) for unit in todo)
    return {p.id: done[p.id] for p in spec.parts}


def units(spec: Spec, found: Map) -> list[Unit]:
    """The calls that read a student mapped as `found`. A choice part goes with the
    page of its first `work` entry, any other part with its top-level problem. A
    part with no entry is in no unit."""
    entries: defaultdict[str, list[MapEntry]] = defaultdict(list)
    for e in found.entries:
        entries[e.part_id].append(e)
    groups: defaultdict[str, list[Part]] = defaultdict(list)
    for part in spec.parts:
        if not (mine := entries[part.id]):
            continue
        if part.response_type in CHOICE_TYPES:
            page = next((e.page for e in mine if e.role == "work"), mine[0].page)
            groups[f"page {page}"].append(part)
        else:
            groups[problem_of(part.id)].append(part)
    return [
        Unit(
            name,
            parts,
            [e for e in found.entries if e.part_id in {p.id for p in parts}],
        )
        for name, parts in groups.items()
    ]


def unit_images(
    workspace: Workspace,
    student: str,
    unit: Unit,
    pages: Sequence[MapPage],
    image_budget: int,
) -> UnitImages:
    """What a call for `unit` is sent besides the parts' text, made with no model
    call: each page that holds an entry, whole, then each distinct mapped box grown
    by `MARGIN` and rendered from `source.pdf` to fill the image budget, all turned
    upright as `pages` (the map record's) say, and contrast-corrected on a page they
    call a photo unless it is mostly white. Each region image is also saved with
    `Workspace.crop_png` under each part it holds."""
    kind = read_model(workspace.ingest_json(student), IngestRecord).source_kind
    facts = {p.number: p for p in pages}
    corrected: dict[int, bool] = {}

    def upright(image: Image.Image, number: int) -> bytes:
        turned = prep.rotate(image, facts[number].rotation // 90)
        shown = prep.correct_contrast(turned) if corrected[number] else turned
        return prep.reduced_png(shown, image_budget)

    items: list[Content] = []
    for number in sorted({e.page for e in unit.entries}):
        with Image.open(workspace.student_page(student, number)) as image:
            white = prep.white_share(image, WHITE_LEVEL)
            corrected[number] = facts[number].photo and white < MOSTLY_WHITE
            items += [
                json.dumps({"image": "overview", "page": number}),
                upright(image, number),
            ]

    boxes: dict[tuple[int, tuple[float, ...]], list[MapEntry]] = {}
    for e in unit.entries:
        boxes.setdefault((e.page, tuple(e.box)), []).append(e)
    crops: defaultdict[str, list[Crop]] = defaultdict(list)
    latest: dict[str, int] = {}  # by part id, the region of its latest entry so far
    with render.open_pdf(workspace.source_pdf(student)) as pdf:
        for n, ((number, box), here) in enumerate(boxes.items(), 1):
            continues = [
                latest[e.part_id]
                for e in here
                if e.role == "continuation" and e.part_id in latest
            ]
            parts = list(dict.fromkeys(e.part_id for e in here))
            region = _region(pdf[number - 1], kind, box, image_budget)
            png = upright(region.image, number)
            for part in parts:
                latest[part] = n
                crops[part].append(
                    Crop(page=number, box=region.box, scale=region.scale)
                )
                path = workspace.crop_png(student, part, len(crops[part]))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png)
            about = {
                "image": "region",
                "region": n,
                "page": number,
                "parts": parts,
                "continues": sorted(set(continues)),
            }
            items += [json.dumps(about), png]
    return UnitImages(items, dict(crops))


def problems(parts: Sequence[Part], found: Readings) -> list[str]:
    """Every way `found` fails the read stage's checks, in a fixed order (it is sent
    back to the model). Parts are named by id and lines by number: nothing the
    model transcribed is repeated."""
    wanted = {part.id: part for part in parts}
    seen = Counter(r.part_id for r in found.readings)
    wrong = [f"{id} has no reading" for id in wanted if not seen[id]]
    wrong += [f"{id} is not a part of this request" for id in seen if id not in wanted]
    wrong += [
        f"{id} is read {n} times" for id, n in seen.items() if n > 1 and id in wanted
    ]
    for r in found.readings:
        if (part := wanted.get(r.part_id)) is None:
            continue
        for marked, numbers in (
            ("uncertain", r.uncertain),
            ("addressed to a grader", r.addressed_to_grader),
        ):
            wrong += [
                f"{r.part_id}: line {n} is marked {marked}, and there is no line {n}"
                for n in numbers
                if n > len(r.lines)
            ]
        choices = r.final_answer.choices
        if choices and not part.options:
            wrong.append(
                f"{r.part_id} has no printed options, so its choices are empty"
            )
        elif not set(choices) <= set(part.options):
            wrong.append(f"{r.part_id}: choices must be among {part.options}")
        if len(set(choices)) < len(choices):
            wrong.append(f"{r.part_id}: an option is in choices more than once")
        if r.status == "blank" and (
            r.lines or r.elements or r.final_answer != NO_ANSWER
        ):
            wrong.append(
                f"{r.part_id}: a blank reading has no lines, no elements"
                " and an empty final answer"
            )
        if r.status == "unreadable" and (r.lines or r.final_answer != NO_ANSWER):
            wrong.append(
                f"{r.part_id}: an unreadable reading has no lines"
                " and an empty final answer"
            )
        # Writing that marks a character it cannot read (`?`) and reads no letter or
        # digit is unreadable. Clear writing that is only a sign, such as `+` or an
        # arrow, is not, and neither is a drawing with no writing at all.
        written = [*r.lines, *(e.label for e in r.elements if e.label)]
        legible = any(c.isalnum() for text in written for c in text)
        unread = any("?" in text for text in written)
        if unread and not legible and r.status != "unreadable":
            wrong.append(
                f"{r.part_id}: no letter or digit in it can be read, so its status"
                " is unreadable, with no lines and an empty final answer"
            )
    return list(dict.fromkeys(wrong))


def problem_of(part_id: str) -> str:
    return part_id.split(".")[0]


def _region(
    page: pdfium.PdfPage, kind: render.SourceKind, box: Sequence[float], budget: int
) -> render.Region:
    """`box` grown by `MARGIN` and clamped to the page, rendered to fill `budget`
    but never past the ceiling of design 7.7: twice the page's native resolution.
    `render_region` keeps to it on a slide export; on another page with a raster
    image it is rendered once more at the size that keeps to it."""
    x0, y0, x1, y1 = box
    grown = [
        max(x0 - MARGIN, 0),
        max(y0 - MARGIN, 0),
        min(x1 + MARGIN, 1),
        min(y1 + MARGIN, 1),
    ]
    region = render.render_region(page, kind, grown, budget)
    if region.scale and region.scale > render.MAX_ENLARGEMENT:
        edge = floor(budget * render.MAX_ENLARGEMENT / region.scale)
        region = render.render_region(page, kind, grown, edge)
    return region


def _blank(part_id: str) -> ReadingRecord:
    """The reading of a part the map found no work for, written by code."""
    return ReadingRecord(
        provenance=Provenance(key_part_hash=None, prompt_version=None, model_slug=None),
        reading=Reading(
            part_id=part_id,
            status="blank",
            medium=None,
            lines=[],
            final_answer=NO_ANSWER,
            elements=[],
            uncertain=[],
            addressed_to_grader=[],
            legibility=None,
        ),
        crops=[],
    )
