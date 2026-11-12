"""The map stage on hand-built pages, inventories and resolver outputs, with a mock
transport (src/redpen/grade/map.py)."""

import asyncio
import base64
import hashlib
import io
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel

from redpen import prompts
from redpen.grade.map import (
    Aside,
    Block,
    Entry,
    Inventory,
    PageBlock,
    PageInventory,
    Resolution,
    map_student,
    problems,
)
from redpen.llm.client import Gateway, StageError
from redpen.llm.models import ModelInfo
from redpen.schemas.spec import Part, Spec
from redpen.schemas.submission import IngestPage, IngestRecord, MapRecord
from redpen.workspace import Workspace, read_model, write_model

MODEL = ModelInfo(
    slug="vendor/vision-1",
    supported_parameters=frozenset({"structured_outputs", "max_tokens"}),
    accepts_images=True,
    prompt_price=Decimal(0),
    completion_price=Decimal(0),
    providers=frozenset({"zero"}),
)
STUDENT = "Sam_Ng-0a1b2c"


def part(id: str, page: int) -> Part:
    return Part(
        id=id,
        label=id[1:],
        prompt="Answer.",
        response_type="numeric",
        home_page=page,
        band=[0.1, 0.5],
        figures=[],
        depends_on=[],
        options=[],
    )


SPEC = Spec(parts=[part("Q1", 1), part("Q2.a", 2), part("Q2.b", 2)])


def block(id: str, box: list[float] | None = None) -> PageBlock:
    page = int(id[1 : id.index(".")])
    box = box or [0.1, 0.1, 0.2, 0.2]
    return PageBlock(
        id=id, page=page, box=box, label="", description="work", candidates=[]
    )


def entry(part_id: str, block_id: str) -> Entry:
    return Entry(
        part_id=part_id,
        block_id=block_id,
        basis="position",
        role="work",
        confidence="high",
    )


GOOD = Resolution(
    entries=[entry("Q1", "p1.b1"), entry("Q2.a", "p2.b1")],
    unanswered=["Q2.b"],
    set_aside=[Aside(block_id="p1.b2", reason="a name")],
)


def blank_inventory(image: Image.Image) -> Inventory:
    return Inventory(rotation=0, photo=False, blocks=[])


class Mapper:
    """Answers an inventory request with `inventory(first image)`, and each resolve
    request with the next of `resolutions`."""

    def __init__(
        self,
        resolutions: list[Resolution],
        inventory: Callable[[Image.Image], Inventory] = blank_inventory,
    ) -> None:
        self.resolutions, self.inventory = resolutions, inventory
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        images = [
            Image.open(
                io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",")[1]))
            )
            for c in content
            if c["type"] == "image_url"
        ]
        name = body["response_format"]["json_schema"]["name"]
        self.requests.append(
            {
                "schema": name,
                "images": images,
                "texts": [c["text"] for c in content if c["type"] == "text"],
                "session": body["session_id"],
            }
        )
        value: BaseModel = (
            self.inventory(images[0])
            if name == "Inventory"
            else self.resolutions.pop(0)
        )
        return httpx.Response(
            200,
            json={
                "model": MODEL.slug,
                "choices": [{"message": {"content": value.model_dump_json()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )

    def of(self, schema: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["schema"] == schema]


def workspace_with(
    root: Path,
    pages: list[Image.Image],
    blank: list[Image.Image] = [],  # noqa: B006 (never changed)
    kind: Literal["classkick", "other"] = "classkick",
) -> Workspace:
    """A workspace holding the blank's page images and one ingested student."""
    workspace = Workspace(root / "ws")
    for number, image in enumerate(blank, 1):
        workspace.key_page(number).parent.mkdir(parents=True, exist_ok=True)
        image.save(workspace.key_page(number))
    for number, image in enumerate(pages, 1):
        workspace.student_page(STUDENT, number).parent.mkdir(
            parents=True, exist_ok=True
        )
        image.save(workspace.student_page(STUDENT, number))
    record = IngestRecord(
        student_name="Sam Ng",
        student_key=STUDENT,
        source_kind=kind,
        pages=[
            IngestPage(
                number=number,
                width_px=image.width,
                height_px=image.height,
                slide_number=number if kind == "classkick" else None,
                native_ppi=None,
            )
            for number, image in enumerate(pages, 1)
        ],
    )
    write_model(workspace.ingest_json(STUDENT), record)
    return workspace


def run(workspace: Workspace, mapper: Mapper) -> MapRecord:
    async def main() -> MapRecord:
        async with Gateway(
            workspace.root.parent / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(mapper),
        ) as gateway:
            return await map_student(workspace, STUDENT, SPEC, gateway, MODEL, 1536)

    return asyncio.run(main())


def page(width: int, height: int = 200) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


BLOCKS = {b.id: b for b in (block("p1.b1"), block("p1.b2"), block("p2.b1"))}


@pytest.mark.parametrize(
    ("found", "wrong"),
    [
        (GOOD, []),
        (
            GOOD.model_copy(update={"unanswered": []}),
            ["Q2.b is in no entry and not unanswered"],
        ),
        (
            GOOD.model_copy(update={"unanswered": ["Q2.b", "Q2.a"]}),
            ["Q2.a is in an entry and also unanswered"],
        ),
        (
            GOOD.model_copy(update={"entries": [*GOOD.entries, entry("Q9", "p1.b1")]}),
            ["Q9 is not a part of the assignment"],
        ),
        (
            GOOD.model_copy(update={"unanswered": ["Q2.b", "Q2"]}),
            ["Q2 is not a part of the assignment"],
        ),
        (
            GOOD.model_copy(
                update={"entries": [*GOOD.entries, entry("Q2.a", "p3.b1")]}
            ),
            ["p3.b1 is not a block"],
        ),
        (
            GOOD.model_copy(
                update={"set_aside": [*GOOD.set_aside, Aside(block_id="x", reason="")]}
            ),
            ["x is not a block"],
        ),
        (
            GOOD.model_copy(update={"set_aside": []}),
            ["p1.b2 is neither in an entry nor set aside"],
        ),
    ],
    ids=[
        "clean",
        "part nowhere",
        "part in both",
        "unknown part in an entry",
        "unknown part unanswered",
        "unknown block in an entry",
        "unknown block set aside",
        "block unclaimed",
    ],
)
def test_each_check_catches_its_failure(found: Resolution, wrong: list[str]) -> None:
    assert problems(SPEC, BLOCKS, found) == wrong


def two_pages(image: Image.Image) -> Inventory:
    """Page 1 (100 px wide) holds two blocks, page 2 (120 px wide) one."""
    boxes = {
        100: [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        120: [[0.2, 0.3, 0.6, 0.9]],
    }
    blocks = [
        Block(box=box, label="", description="work", candidates=[])
        for box in boxes[image.width]
    ]
    return Inventory(rotation=0, photo=image.width == 120, blocks=blocks)


def test_blocks_get_ids_in_order_and_the_map_carries_their_page_and_box(
    tmp_path: Path,
) -> None:
    workspace = workspace_with(tmp_path, [page(100), page(120)])
    resolution = Resolution(
        entries=[entry("Q1", "p1.b2"), entry("Q2.a", "p2.b1")],
        unanswered=["Q2.b"],
        set_aside=[Aside(block_id="p1.b1", reason="a name")],
    )
    mapper = Mapper([resolution], two_pages)

    mapped = run(workspace, mapper)

    [resolve] = mapper.of("Resolution")
    sent = json.loads(resolve["texts"][1])["blocks"]
    assert [(b["id"], b["page"], b["box"]) for b in sent] == [
        ("p1.b1", 1, [0.1, 0.2, 0.3, 0.4]),
        ("p1.b2", 1, [0.5, 0.6, 0.7, 0.8]),
        ("p2.b1", 2, [0.2, 0.3, 0.6, 0.9]),
    ]
    assert [(e.part_id, e.page, e.box) for e in mapped.map.entries] == [
        ("Q1", 1, [0.5, 0.6, 0.7, 0.8]),
        ("Q2.a", 2, [0.2, 0.3, 0.6, 0.9]),
    ]
    assert [(s.page, s.box, s.reason) for s in mapped.map.set_aside] == [
        (1, [0.1, 0.2, 0.3, 0.4], "a name")
    ]
    assert mapped.map.unanswered == ["Q2.b"]
    assert [(p.number, p.rotation, p.photo) for p in mapped.pages] == [
        (1, 0, False),
        (2, 0, True),
    ]
    assert read_model(workspace.map_json(STUDENT), MapRecord) == mapped
    stored = read_model(workspace.student_page_json(STUDENT, 2), PageInventory)
    assert [b.id for b in stored.blocks] == ["p2.b1"]
    versions = prompts.load("map", "inventory")[1] + prompts.load("map", "resolve")[1]
    assert (
        mapped.provenance.prompt_version
        == (hashlib.sha256(versions.encode()).hexdigest()[:12])
    )
    assert mapped.provenance.model_slug == MODEL.slug
    assert {r["session"] for r in mapper.requests} == {"ws:map"}


def test_a_map_that_fails_its_checks_is_resolved_once_more_with_the_failures(
    tmp_path: Path,
) -> None:
    workspace = workspace_with(tmp_path, [page(100), page(120)])
    missing = GOOD.model_copy(update={"unanswered": []})
    mapper = Mapper([missing, GOOD], two_pages)

    mapped = run(workspace, mapper)

    first, second = mapper.of("Resolution")
    assert first["texts"][2:] == []
    assert second["texts"][2:] == [
        json.dumps({"problems": ["Q2.b is in no entry and not unanswered"]})
    ]
    assert mapped.map.unanswered == ["Q2.b"]
    assert workspace.map_json(STUDENT).exists()


def test_a_map_that_fails_twice_raises_and_writes_no_map(tmp_path: Path) -> None:
    workspace = workspace_with(tmp_path, [page(100), page(120)])
    missing = GOOD.model_copy(update={"unanswered": []})
    mapper = Mapper([missing, missing], two_pages)

    with pytest.raises(StageError, match="Q2.b is in no entry and not unanswered"):
        run(workspace, mapper)

    assert len(mapper.of("Resolution")) == 2
    assert not workspace.map_json(STUDENT).exists()


RECTANGLE = (30, 400, 90, 450)  # left, top, right, bottom on a 300 x 600 page


def sideways(image: Image.Image) -> Inventory:
    """Reports a portrait page as turned by 90 degrees, with a box it cannot be
    trusted on, and finds the dark rectangle on the turned page."""
    if image.height > image.width:
        junk = Block(box=[0.0, 0.0, 0.5, 0.5], label="", description="?", candidates=[])
        return Inventory(rotation=90, photo=True, blocks=[junk])
    x0, y0, x1, y1 = ImageOps.invert(image.convert("L")).getbbox() or (0, 0, 1, 1)
    w, h = image.size
    box = [x0 / w, y0 / h, x1 / w, y1 / h]
    found = Block(box=box, label="", description="work", candidates=["Q1"])
    return Inventory(rotation=0, photo=True, blocks=[found])


def test_a_sideways_page_is_looked_at_again_upright_and_its_boxes_come_back(
    tmp_path: Path,
) -> None:
    drawn = page(300, 600)
    ImageDraw.Draw(drawn).rectangle(RECTANGLE, fill="black")
    workspace = workspace_with(tmp_path, [drawn])
    only = Resolution(
        entries=[entry("Q1", "p1.b1")], unanswered=["Q2.a", "Q2.b"], set_aside=[]
    )
    mapper = Mapper([only], sideways)

    mapped = run(workspace, mapper)

    first, again = mapper.of("Inventory")
    assert first["images"][0].size == (300, 600)
    assert again["images"][0].size == (600, 300)
    # The size of the image sent is given as data (see the inventory prompt).
    assert first["texts"][1:] == [json.dumps({"width": 300, "height": 600})]
    assert again["texts"][1:] == [json.dumps({"width": 600, "height": 300})]
    assert mapped.pages[0].rotation == 90
    [found] = mapped.map.entries
    x0, y0, x1, y1 = found.box
    left, top, right, bottom = RECTANGLE  # drawn inclusive of its right and bottom
    assert [x0 * 300, y0 * 600, x1 * 300, y1 * 600] == pytest.approx(
        [left, top, right + 1, bottom + 1]
    )


@pytest.mark.parametrize(
    ("kind", "blank_pages", "images"),
    [("classkick", 1, {100: 2, 120: 1}), ("other", 2, {100: 1, 120: 1})],
    ids=["slide export, blank of one page", "other PDF"],
)
def test_a_page_gets_the_untouched_blank_page_only_on_a_slide_export(
    tmp_path: Path,
    kind: Literal["classkick", "other"],
    blank_pages: int,
    images: dict[int, int],
) -> None:
    blank = [page(90 + n, 210) for n in range(blank_pages)]
    workspace = workspace_with(tmp_path, [page(100), page(120)], blank, kind)
    nothing = Resolution(
        entries=[], unanswered=[p.id for p in SPEC.parts], set_aside=[]
    )
    mapper = Mapper([nothing])

    run(workspace, mapper)

    sent = {r["images"][0].width: r["images"] for r in mapper.of("Inventory")}
    assert {width: len(shown) for width, shown in sent.items()} == images
    if kind == "classkick":
        assert sent[100][1].size == blank[0].size
