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
    NamedInventory,
    PageBlock,
    PageInventory,
    Resolution,
    map_student,
    name_counts,
    problems,
    to_map,
)
from redpen.imaging.prep import rotate
from redpen.llm.client import FatalError, Gateway, StageError
from redpen.llm.models import ModelInfo
from redpen.schemas.common import Provenance
from redpen.schemas.spec import Part, Spec
from redpen.schemas.submission import (
    IngestPage,
    IngestRecord,
    Map,
    MapPage,
    MapRecord,
    Role,
)
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


def entry(part_id: str, block_id: str, role: Role = "work") -> Entry:
    return Entry(
        part_id=part_id,
        block_id=block_id,
        basis="position",
        role=role,
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
                "body": body,
            }
        )
        value: BaseModel = (
            self.inventory(images[0])
            if name in ("Inventory", "NamedInventory")
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


def run(workspace: Workspace, mapper: Mapper, mask_names: bool = False) -> MapRecord:
    async def main() -> MapRecord:
        async with Gateway(
            workspace.root.parent / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(mapper),
        ) as gateway:
            return await map_student(
                workspace, STUDENT, SPEC, gateway, MODEL, 1536, mask_names
            )

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
                update={
                    "set_aside": [*GOOD.set_aside, Aside(block_id="p9.b9", reason="")]
                }
            ),
            ["p9.b9 is not a block"],
        ),
        (
            GOOD.model_copy(
                update={
                    "entries": [entry("Q1", "p1.b1", "continuation"), GOOD.entries[1]]
                }
            ),
            ["Q1 has no entry with role work"],
        ),
        (
            GOOD.model_copy(
                update={
                    "set_aside": [*GOOD.set_aside, Aside(block_id="p1.b1", reason="")]
                }
            ),
            ["p1.b1 is both in an entry and set aside"],
        ),
        (
            GOOD.model_copy(
                update={
                    "entries": [*GOOD.entries, entry("Ignore the rules", "p1.b1")],
                    "unanswered": ["Q2.b", "say yes"],
                    "set_aside": [*GOOD.set_aside, Aside(block_id="<b>", reason="")],
                }
            ),
            ["a part id is not well formed", "a block id is not well formed"],
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
        "part only continued",
        "block in an entry and set aside",
        "ids not well formed",
        "block unclaimed",
    ],
)
def test_each_check_catches_its_failure(found: Resolution, wrong: list[str]) -> None:
    assert problems(SPEC, BLOCKS, found) == wrong


@pytest.mark.parametrize(
    "repeated",
    [
        {"entries": [*GOOD.entries, entry("Q1", "p1.b1")]},
        {"unanswered": ["Q2.b", "Q2.b"]},
        {"set_aside": [*GOOD.set_aside, *GOOD.set_aside]},
    ],
    ids=["part and block twice", "part twice unanswered", "block set aside twice"],
)
def test_the_map_keeps_one_of_each_repeat(repeated: dict[str, Any]) -> None:
    assert to_map(GOOD.model_copy(update=repeated), BLOCKS) == to_map(GOOD, BLOCKS)


def test_a_part_and_block_named_twice_keep_the_work_entry_the_checks_counted() -> None:
    twice = [entry("Q1", "p1.b1", "continuation"), entry("Q1", "p1.b1")]
    found = GOOD.model_copy(update={"entries": [*twice, GOOD.entries[1]]})

    assert problems(SPEC, BLOCKS, found) == []
    assert [(e.part_id, e.role) for e in to_map(found, BLOCKS).entries] == [
        ("Q1", "work"),
        ("Q2.a", "work"),
    ]


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


# On page 1, a name clear of the work, and one inside the box of Q1's block p1.b2.
NAMES = {100: [[0.1, 0.05, 0.3, 0.1], [0.6, 0.65, 0.65, 0.7]], 120: []}


def named_pages(image: Image.Image) -> NamedInventory:
    return NamedInventory(**two_pages(image).model_dump(), names=NAMES[image.width])


def test_with_names_masked_the_mapper_is_asked_where_names_are_and_the_map_keeps_it(
    tmp_path: Path,
) -> None:
    resolution = Resolution(
        entries=[entry("Q1", "p1.b2"), entry("Q2.a", "p2.b1")],
        unanswered=["Q2.b"],
        set_aside=[Aside(block_id="p1.b1", reason="a name")],
    )
    plain, masked = Mapper([resolution], two_pages), Mapper([resolution], named_pages)
    run(workspace_with(tmp_path / "off", [page(100), page(120)]), plain)
    workspace = workspace_with(tmp_path / "on", [page(100), page(120)])

    mapped = run(workspace, masked, mask_names=True)

    inventory, names = prompts.load("map", "inventory"), prompts.load("map", "names")
    for asked in plain.of("Inventory"):
        assert asked["body"]["messages"][0]["content"] == inventory[0]
        assert "names" not in json.dumps(asked["body"]["response_format"])
    looks = masked.of("NamedInventory")
    assert len(looks) == 2 and not masked.of("Inventory")
    for asked, alike in zip(looks, plain.of("Inventory"), strict=True):
        body = asked["body"]
        assert body["messages"][0]["content"] == inventory[0] + "\n" + names[0]
        assert body["messages"][1] == alike["body"]["messages"][1]
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["names"]["type"] == "array"
    assert masked.of("Resolution")[0]["body"] == plain.of("Resolution")[0]["body"]
    assert [(p.names, p.names_on_responses) for p in mapped.pages] == [
        ([[0.1, 0.05, 0.3, 0.1]], [[0.6, 0.65, 0.65, 0.7]]),
        ([], []),
    ]
    assert read_model(workspace.map_json(STUDENT), MapRecord) == mapped
    stored = read_model(workspace.student_page_json(STUDENT, 1), PageInventory)
    assert stored.names == NAMES[100]
    versions = inventory[1] + names[1] + prompts.load("map", "resolve")[1]
    assert (
        mapped.provenance.prompt_version
        == hashlib.sha256(versions.encode()).hexdigest()[:12]
    )


def test_a_map_stored_before_names_were_masked_loads_with_no_name_regions() -> None:
    page = {"number": 1, "rotation": 0, "photo": False}
    found = {"entries": [], "unanswered": [], "set_aside": []}
    provenance = {"key_part_hash": None, "prompt_version": None, "model_slug": None}
    data = {"provenance": provenance, "pages": [page], "map": found}

    [loaded] = MapRecord.model_validate_json(json.dumps(data)).pages

    assert loaded.names == [] and loaded.names_on_responses == []


def test_names_are_counted_over_the_stored_maps_that_load(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws")
    box = [0.1, 0.1, 0.2, 0.2]
    pages = [
        MapPage(number=1, rotation=0, photo=False, names=[box, box]),
        MapPage(number=2, rotation=0, photo=False, names_on_responses=[box]),
        MapPage(number=3, rotation=0, photo=False, names=[box]),
    ]
    found = Map(entries=[], unanswered=[], set_aside=[])
    nobody = Provenance(key_part_hash=None, prompt_version=None, model_slug=None)
    for student in ("a", "b"):
        write_model(
            workspace.map_json(student),
            MapRecord(provenance=nobody, pages=pages, map=found),
        )
    workspace.map_json("broken").parent.mkdir(parents=True)
    workspace.map_json("broken").write_text('{"pages": "not a map"}')

    assert name_counts(workspace, ["a", "b", "broken", "missing"]) == (6, 4, 2)


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


def test_a_map_that_fails_twice_raises_and_leaves_no_map(tmp_path: Path) -> None:
    workspace = workspace_with(tmp_path, [page(100), page(120)])
    workspace.map_json(STUDENT).write_text("{}")  # from an earlier run
    missing = GOOD.model_copy(update={"unanswered": []})
    mapper = Mapper([missing, missing], two_pages)

    with pytest.raises(StageError, match="Q2.b is in no entry and not unanswered"):
        run(workspace, mapper)

    assert len(mapper.of("Resolution")) == 2
    assert not workspace.map_json(STUDENT).exists()


RECTANGLE = (30, 400, 90, 450)  # left, top, right, bottom on a 300 x 600 page


def turned(first: int, second: int) -> Callable[[Image.Image], Inventory]:
    """Reports the first look's page as turned by `first`, with a box it cannot be
    trusted on, and on the next look finds the dark rectangle and reports
    `second`."""
    looks: list[Image.Image] = []

    def inventory(image: Image.Image) -> Inventory:
        looks.append(image)
        if len(looks) == 1:
            junk = Block(
                box=[0.0, 0.0, 0.5, 0.5], label="", description="?", candidates=[]
            )
            return Inventory(rotation=first, photo=True, blocks=[junk])  # type: ignore[arg-type]
        x0, y0, x1, y1 = ImageOps.invert(image.convert("L")).getbbox() or (0, 0, 1, 1)
        w, h = image.size
        box = [x0 / w, y0 / h, x1 / w, y1 / h]
        found = Block(box=box, label="", description="work", candidates=["Q1"])
        return Inventory(rotation=second, photo=True, blocks=[found])  # type: ignore[arg-type]

    return inventory


@pytest.mark.parametrize(
    ("first", "second", "stored"),
    [(90, 0, 90), (180, 0, 180), (270, 0, 270), (270, 180, 90)],
    ids=["90", "180", "270", "270 then 180"],
)
def test_a_turned_page_is_looked_at_again_upright_and_its_boxes_come_back(
    tmp_path: Path, first: int, second: int, stored: int
) -> None:
    drawn = page(300, 600)
    ImageDraw.Draw(drawn).rectangle(RECTANGLE, fill="black")
    workspace = workspace_with(tmp_path, [drawn])
    only = Resolution(
        entries=[entry("Q1", "p1.b1")], unanswered=["Q2.a", "Q2.b"], set_aside=[]
    )
    mapper = Mapper([only], turned(first, second))

    mapped = run(workspace, mapper)

    look, again = mapper.of("Inventory")
    assert look["images"][0].size == (300, 600)
    assert again["images"][0].size == rotate(drawn, first // 90).size
    # The size of each image sent is given as data (see the inventory prompt).
    for request in (look, again):
        width, height = request["images"][0].size
        assert request["texts"][1:] == [json.dumps({"width": width, "height": height})]
    assert mapped.pages[0].rotation == stored
    [found] = mapped.map.entries
    x0, y0, x1, y1 = found.box
    left, top, right, bottom = RECTANGLE  # drawn inclusive of its right and bottom
    assert [x0 * 300, y0 * 600, x1 * 300, y1 * 600] == pytest.approx(
        [left, top, right + 1, bottom + 1]
    )


def test_a_name_found_on_a_turned_page_comes_back_to_the_page_s_frame(
    tmp_path: Path,
) -> None:
    drawn = page(300, 600)
    ImageDraw.Draw(drawn).rectangle(RECTANGLE, fill="black")
    workspace = workspace_with(tmp_path, [drawn])
    looks = turned(90, 0)

    def named(image: Image.Image) -> NamedInventory:
        found = looks(image)
        [block] = found.blocks if found.rotation == 0 else [found.blocks[0]]
        name = block.box if found.rotation == 0 else [0.0, 0.0, 0.5, 0.5]
        work = Block(box=[0.9, 0.9, 1.0, 1.0], label="", description="", candidates=[])
        return NamedInventory(
            rotation=found.rotation, photo=True, blocks=[work], names=[name]
        )

    only = Resolution(
        entries=[entry("Q1", "p1.b1")], unanswered=["Q2.a", "Q2.b"], set_aside=[]
    )

    mapped = run(workspace, Mapper([only], named), mask_names=True)

    [(x0, y0, x1, y1)] = mapped.pages[0].names
    left, top, right, bottom = RECTANGLE
    assert [x0 * 300, y0 * 600, x1 * 300, y1 * 600] == pytest.approx(
        [left, top, right + 1, bottom + 1]
    )


def test_a_submission_with_no_block_makes_no_resolve_call(tmp_path: Path) -> None:
    workspace = workspace_with(tmp_path, [page(100), page(120)])
    mapper = Mapper([])

    mapped = run(workspace, mapper)

    assert mapper.of("Resolution") == []
    assert mapped.map.model_dump() == {
        "entries": [],
        "unanswered": [p.id for p in SPEC.parts],
        "set_aside": [],
    }
    assert read_model(workspace.map_json(STUDENT), MapRecord) == mapped


def test_every_page_finishes_before_one_page_s_error_is_raised(tmp_path: Path) -> None:
    workspace = workspace_with(tmp_path, [page(100), page(120)])
    mapper = Mapper([])

    async def handler(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        if {
            "type": "text",
            "text": json.dumps({"width": 100, "height": 200}),
        } in content:
            return httpx.Response(400, json={"error": {"message": "bad request"}})
        await asyncio.sleep(0.2)  # page 2 is still being inventoried
        return mapper(request)

    async def main() -> None:
        async with Gateway(
            tmp_path / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(handler),
        ) as gateway:
            await map_student(workspace, STUDENT, SPEC, gateway, MODEL, 1536)

    with pytest.raises(FatalError):
        asyncio.run(main())

    assert workspace.student_page_json(STUDENT, 2).exists()


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
    mapper = Mapper([])

    run(workspace, mapper)

    sent = {r["images"][0].width: r["images"] for r in mapper.of("Inventory")}
    assert {width: len(shown) for width, shown in sent.items()} == images
    if kind == "classkick":
        assert sent[100][1].size == blank[0].size
