"""Map one student's submission (design 7.3, before the fingerprint of 7.2 exists):
one call per page lists the blocks of student work on it, one text-only call
attributes the blocks to parts by id, and code checks the result and looks up each
block's page and box. No model copies a coordinate."""

import hashlib
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, PositiveInt, TypeAdapter, ValidationError

from redpen import prompts
from redpen.imaging import prep
from redpen.llm.client import Gateway, ask_checked, finish_all
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.schemas.common import Box, PartId, Provenance, Schema
from redpen.schemas.spec import Spec
from redpen.schemas.submission import (
    Basis,
    Confidence,
    IngestRecord,
    Map,
    MapEntry,
    MapPage,
    MapRecord,
    Role,
    SetAside,
)
from redpen.workspace import Workspace, read_model, write_model

ROLE = "mapper"
PARAMS = {"max_tokens": 8000}
PART_ID = TypeAdapter(PartId)
BLOCK_ID = re.compile(r"p[1-9][0-9]*\.b[1-9][0-9]*")


# One distinct piece of student work, as the inventory call reports it. The box is
# in fractions of the image the call was sent. `candidates` are part ids.
class Block(Schema):
    box: Box
    label: str
    description: str
    candidates: list[str]


# What the inventory call returns for one page image.
class Inventory(Schema):
    rotation: Literal[0, 90, 180, 270]
    photo: bool
    blocks: list[Block]


# What it returns with names masked: also the box of each name on the page.
class NamedInventory(Inventory):
    names: list[Box]


# A block as code keeps it: its id `p<page>.b<n>` and its page added, and its box in
# fractions of the page image as ingest wrote it.
class PageBlock(Block):
    id: str
    page: PositiveInt


# One page's inventory as stored (`Workspace.student_page_json`). Its `names` are
# every name found on the page: which overlap a response is the map's to say.
class PageInventory(MapPage):
    blocks: list[PageBlock]


# What the resolve call returns: parts and set-aside blocks named by block id.
class Entry(Schema):
    part_id: str
    block_id: str
    basis: Basis
    role: Role
    confidence: Confidence


class Aside(Schema):
    block_id: str
    reason: str


class Resolution(Schema):
    entries: list[Entry]
    unanswered: list[str]
    set_aside: list[Aside]


async def map_student(
    workspace: Workspace,
    student: str,
    spec: Spec,
    gateway: Gateway,
    model: ModelInfo,
    image_budget: int,
    mask_names: bool = False,
) -> MapRecord:
    """Map the ingested `student`: write each page's inventory and `map.json`, and
    return what was written. With `mask_names` each inventory call also asks where
    a student's name is on the page. Raises `StageError` if the map still fails its
    checks after one more resolve call; no map is left then, not even an earlier
    one."""
    record = read_model(workspace.ingest_json(student), IngestRecord)
    workspace.map_json(student).unlink(missing_ok=True)
    inventory_prompt = prompts.load("map", "inventory")
    asked = NamedInventory if mask_names else Inventory
    if mask_names:
        text, version = prompts.load("map", "names")
        inventory_prompt = (
            f"{inventory_prompt[0]}\n{text}",
            inventory_prompt[1] + version,
        )
    resolve_prompt = prompts.load("map", "resolve")
    session = f"{workspace.root.name}:map"
    fields = {"id", "label", "response_type", "home_page", "prompt"}
    outline = json.dumps({"parts": [p.model_dump(include=fields) for p in spec.parts]})

    async def ask[T: BaseModel](
        prompt: tuple[str, str], items: Sequence[Content], response_model: type[T]
    ) -> T:
        system, version = prompt
        result = await gateway.call(
            ROLE,
            model,
            system,
            [outline],
            items,
            response_model,
            version,
            params=PARAMS,
            session_id=session,
        )
        return result.value

    async def take(number: int) -> PageInventory:
        image = _open(workspace.student_page(student, number))
        blank = workspace.key_page(number)
        # Stands in for the fingerprint (7.2): on a slide export page k is blank
        # page k, so the model is shown that page before the student touched it.
        before = (
            [prep.reduced_png(_open(blank), image_budget)]
            if record.source_kind == "classkick" and blank.exists()
            else []
        )

        async def look(turns: int) -> Inventory:
            shown = prep.reduced_png(prep.rotate(image, turns), image_budget)
            with Image.open(io.BytesIO(shown)) as sent:
                # Without it the model was seen to measure a tall page as if it
                # had a paper sheet's shape (task 2.3 report).
                size = json.dumps({"width": sent.width, "height": sent.height})
            return await ask(inventory_prompt, [shown, *before, size], asked)

        found = await look(0)
        blocks, rotation, names = found.blocks, found.rotation, _names(found)
        if turns := found.rotation // 90:
            # Boxes found on a sideways page are not trusted: look again upright.
            # A turn reported then means the first look had the direction wrong.
            upright = await look(turns)
            rotation = (found.rotation + upright.rotation) % 360
            blocks = [
                b.model_copy(update={"box": prep.box_from_rotated(b.box, turns)})
                for b in upright.blocks
            ]
            names = [prep.box_from_rotated(b, turns) for b in _names(upright)]
        page = PageInventory(
            number=number,
            rotation=rotation,  # pyright: ignore[reportArgumentType]
            photo=found.photo,
            names=names,
            blocks=[
                PageBlock(id=f"p{number}.b{n}", page=number, **b.model_dump())
                for n, b in enumerate(blocks, 1)
            ],
        )
        write_model(workspace.student_page_json(student, number), page)
        return page

    pages = await finish_all(take(p.number) for p in record.pages)
    blocks = {b.id: b for page in pages for b in page.blocks}
    data = json.dumps({"blocks": [b.model_dump(mode="json") for b in blocks.values()]})

    async def resolve(extra: Sequence[Content]) -> Resolution:
        return await ask(resolve_prompt, [data, *extra], Resolution)

    found = (
        await ask_checked(
            resolve, lambda r: problems(spec, blocks, r), f"the map of {student}"
        )
        if blocks
        else Resolution(entries=[], unanswered=[p.id for p in spec.parts], set_aside=[])
    )
    both = (inventory_prompt[1] + resolve_prompt[1]).encode()
    found_map = to_map(found, blocks)
    mapped = MapRecord(
        provenance=Provenance(
            key_part_hash=None,
            prompt_version=hashlib.sha256(both).hexdigest()[:12],
            model_slug=model.slug,
        ),
        pages=[_masked(p, found_map) for p in pages],
        map=found_map,
    )
    write_model(workspace.map_json(student), mapped)
    return mapped


def problems(
    spec: Spec, blocks: Mapping[str, PageBlock], found: Resolution
) -> list[str]:
    """Every way `found` fails the checks of design 7.3 that need no fingerprint, in
    a fixed order (it is sent back to the model). An id the model wrote is named
    only when it has the shape of a part id or a block id: it is free text."""
    parts = [p.id for p in spec.parts]
    mapped = [e.part_id for e in found.entries]
    working = {e.part_id for e in found.entries if e.role == "work"}
    named = [*mapped, *found.unanswered]
    wrong = [f"{p} is in no entry and not unanswered" for p in parts if p not in named]
    wrong += [
        f"{p} is in an entry and also unanswered"
        for p in parts
        if p in mapped and p in found.unanswered
    ]
    wrong += [
        f"{p} has no entry with role work"
        for p in parts
        if p in mapped and p not in working
    ]
    wrong += [
        f"{p} is not a part of the assignment"
        if _is_part_id(p)
        else "a part id is not well formed"
        for p in dict.fromkeys(named)
        if p not in parts
    ]
    claimed = [e.block_id for e in found.entries]
    aside = [a.block_id for a in found.set_aside]
    wrong += [
        f"{b} is not a block"
        if BLOCK_ID.fullmatch(b)
        else "a block id is not well formed"
        for b in dict.fromkeys([*claimed, *aside])
        if b not in blocks
    ]
    wrong += [
        f"{b} is both in an entry and set aside"
        for b in blocks
        if b in claimed and b in aside
    ]
    wrong += [
        f"{b} is neither in an entry nor set aside"
        for b in blocks
        if b not in claimed and b not in aside
    ]
    return list(dict.fromkeys(wrong))


def _is_part_id(text: str) -> bool:
    try:
        PART_ID.validate_python(text)
    except ValidationError:
        return False
    return True


def to_map(found: Resolution, blocks: Mapping[str, PageBlock]) -> Map:
    """The map `found` describes, with each block's page and box looked up, and a
    part and block named twice, a part unanswered twice or a block set aside twice
    kept once: the first, or for a part and block the first with role `work`, which
    is what the checks counted."""
    entries: dict[tuple[str, str], Entry] = {}
    for e in found.entries:
        kept = entries.setdefault((e.part_id, e.block_id), e)
        if e.role == "work" and kept.role != "work":
            entries[(e.part_id, e.block_id)] = e
    aside: dict[str, Aside] = {}
    for a in found.set_aside:
        aside.setdefault(a.block_id, a)
    return Map(
        entries=[
            MapEntry(
                part_id=e.part_id,
                page=blocks[e.block_id].page,
                box=blocks[e.block_id].box,
                basis=e.basis,
                role=e.role,
                confidence=e.confidence,
            )
            for e in entries.values()
        ],
        unanswered=list(dict.fromkeys(found.unanswered)),
        set_aside=[
            SetAside(
                page=blocks[a.block_id].page,
                box=blocks[a.block_id].box,
                reason=a.reason,
            )
            for a in aside.values()
        ],
    )


def name_counts(workspace: Workspace, students: Iterable[str]) -> tuple[int, int, int]:
    """Over the maps stored for `students`: the names blanked, the pages they are
    on, and the names left whole because they overlap a response. A map that is
    missing or does not load counts nothing."""
    pages: list[MapPage] = []
    for student in students:
        try:
            pages += read_model(workspace.map_json(student), MapRecord).pages
        except (OSError, ValidationError):
            continue
    blanked = sum(len(p.names) for p in pages)
    left = sum(len(p.names_on_responses) for p in pages)
    return blanked, sum(1 for p in pages if p.names), left


def _names(found: Inventory) -> list[list[float]]:
    return found.names if isinstance(found, NamedInventory) else []


def _masked(page: PageInventory, found: Map) -> MapPage:
    """What the map keeps of `page`, its names split: a name that overlaps a box
    of `found` on the page is left whole, since a mask never covers a response."""
    responses = [e.box for e in found.entries if e.page == page.number]
    kept = [
        n
        for n in page.names
        if any(prep.box_within(n, r) is not None for r in responses)
    ]
    return MapPage(
        number=page.number,
        rotation=page.rotation,
        photo=page.photo,
        names=[n for n in page.names if n not in kept],
        names_on_responses=kept,
    )


def _open(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
    return image
