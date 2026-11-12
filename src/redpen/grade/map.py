"""Map one student's submission (design 7.3, before the fingerprint of 7.2 exists):
one call per page lists the blocks of student work on it, one text-only call
attributes the blocks to parts by id, and code checks the result and looks up each
block's page and box. No model copies a coordinate."""

import asyncio
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, PositiveInt

from redpen import prompts
from redpen.imaging import prep
from redpen.llm.client import Gateway, ask_checked
from redpen.llm.client import Part as Content
from redpen.llm.models import ModelInfo
from redpen.schemas.common import Box, Provenance, Schema
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


# A block as code keeps it: its id `p<page>.b<n>` and its page added, and its box in
# fractions of the page image as ingest wrote it.
class PageBlock(Block):
    id: str
    page: PositiveInt


# One page's inventory as stored (`Workspace.student_page_json`).
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
) -> MapRecord:
    """Map the ingested `student`: write each page's inventory and `map.json`, and
    return what was written. Raises `StageError` if the map still fails its checks
    after one more resolve call; no map is written then."""
    record = read_model(workspace.ingest_json(student), IngestRecord)
    inventory_prompt = prompts.load("map", "inventory")
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
            return await ask(inventory_prompt, [shown, *before, size], Inventory)

        found = await look(0)
        blocks = found.blocks
        if turns := found.rotation // 90:
            # Boxes found on a sideways page are not trusted: look again upright.
            blocks = [
                b.model_copy(update={"box": prep.box_from_rotated(b.box, turns)})
                for b in (await look(turns)).blocks
            ]
        page = PageInventory(
            number=number,
            rotation=found.rotation,
            photo=found.photo,
            blocks=[
                PageBlock(id=f"p{number}.b{n}", page=number, **b.model_dump())
                for n, b in enumerate(blocks, 1)
            ],
        )
        write_model(workspace.student_page_json(student, number), page)
        return page

    pages = await asyncio.gather(*(take(p.number) for p in record.pages))
    blocks = {b.id: b for page in pages for b in page.blocks}
    data = json.dumps({"blocks": [b.model_dump(mode="json") for b in blocks.values()]})

    async def resolve(extra: Sequence[Content]) -> Resolution:
        return await ask(resolve_prompt, [data, *extra], Resolution)

    found = await ask_checked(
        resolve, lambda r: problems(spec, blocks, r), f"the map of {student}"
    )
    both = (inventory_prompt[1] + resolve_prompt[1]).encode()
    mapped = MapRecord(
        provenance=Provenance(
            key_part_hash=None,
            prompt_version=hashlib.sha256(both).hexdigest()[:12],
            model_slug=model.slug,
        ),
        pages=[
            MapPage(number=p.number, rotation=p.rotation, photo=p.photo) for p in pages
        ],
        map=to_map(found, blocks),
    )
    write_model(workspace.map_json(student), mapped)
    return mapped


def problems(
    spec: Spec, blocks: Mapping[str, PageBlock], found: Resolution
) -> list[str]:
    """Every way `found` fails the checks of design 7.3 that need no fingerprint, in
    a fixed order (it is sent back to the model)."""
    parts = [p.id for p in spec.parts]
    mapped = [e.part_id for e in found.entries]
    named = [*mapped, *found.unanswered]
    wrong = [f"{p} is in no entry and not unanswered" for p in parts if p not in named]
    wrong += [
        f"{p} is in an entry and also unanswered"
        for p in parts
        if p in mapped and p in found.unanswered
    ]
    wrong += [
        f"{p} is not a part of the assignment"
        for p in dict.fromkeys(named)
        if p not in parts
    ]
    used = [
        *(e.block_id for e in found.entries),
        *(a.block_id for a in found.set_aside),
    ]
    wrong += [f"{b} is not a block" for b in dict.fromkeys(used) if b not in blocks]
    wrong += [
        f"{b} is neither in an entry nor set aside" for b in blocks if b not in used
    ]
    return wrong


def to_map(found: Resolution, blocks: Mapping[str, PageBlock]) -> Map:
    """The map `found` describes, with each block's page and box looked up."""
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
            for e in found.entries
        ],
        unanswered=found.unanswered,
        set_aside=[
            SetAside(
                page=blocks[a.block_id].page,
                box=blocks[a.block_id].box,
                reason=a.reason,
            )
            for a in found.set_aside
        ],
    )


def _open(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
    return image
