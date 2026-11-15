import asyncio
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel
from reportlab.pdfgen import canvas

from autograder2.imaging import render
from autograder2.llm.client import Gateway
from autograder2.llm.models import ModelInfo
from autograder2.prepare.parse import (
    PageItem,
    PageParse,
    SpecError,
    check,
    missing_numbers,
    parse,
    with_text_bands,
)
from autograder2.schemas.common import ResponseType
from autograder2.schemas.spec import Part, Spec
from autograder2.workspace import Workspace
from tests.fixture_truth import FIXTURES

MODEL = ModelInfo(
    slug="vendor/vision-1",
    supported_parameters=frozenset({"structured_outputs", "max_tokens"}),
    accepts_images=True,
    prompt_price=Decimal(0),
    completion_price=Decimal(0),
    providers=frozenset({"zero"}),
)


def part(id: str, label: str, home_page: int = 1, **fields: Any) -> Part:
    return Part.model_validate(
        {
            "id": id,
            "label": label,
            "prompt": "Find it.",
            "response_type": "numeric",
            "home_page": home_page,
            "band": [0.1, 0.2],
            "figures": [],
            "depends_on": [],
            "options": [],
        }
        | fields
    )


@pytest.mark.parametrize(
    ("parts", "failures"),
    [
        ([part("Q1", "1"), part("Q2", "2", 2)], []),
        ([], ["no part was found"]),
        ([part("Q1", "1"), part("Q1", "1")], ["Q1 is used 2 times"]),
        (
            [part("Q1", "1", response_type="multiple_choice", options=["A"])],
            ["Q1 is multiple_choice with fewer than two options"],
        ),
        (
            [part("Q1", "1", response_type="multiple_select")],
            ["Q1 is multiple_select with fewer than two options"],
        ),
        ([part("Q1", "1", 3)], ["Q1 has home page 3 of 2"]),
        (
            [part("Q1", "1", figures=[{"page": 3, "box": [0, 0, 1, 1]}])],
            ["Q1 has a figure on page 3 of 2"],
        ),
        ([part("Q1", "1", depends_on=["Q1"])], ["Q1 depends on itself"]),
        (
            [part("Q1", "1", depends_on=["Q2"])],
            ["Q1 depends on Q2, which does not exist"],
        ),
        # A gap is a warning, not a failure: an assignment may skip a number.
        ([part("Q1.a", "1(a)"), part("Q3", "3", 2)], []),
    ],
)
def test_each_check_catches_its_failure(parts: list[Part], failures: list[str]) -> None:
    assert check(Spec(parts=parts), 2) == failures


def numbered(*labels: str) -> list[Part]:
    """Parts with these labels, numbered by their ids in order of appearance."""
    return [part(f"Q{n}", label) for n, label in enumerate(labels, 1)]


@pytest.mark.parametrize(
    ("parts", "missing"),
    [
        ([part("Q1.a", "1(a)"), part("Q1.b", "1(b)"), part("Q3", "3")], [2]),
        ([part(f"Q{n}", str(n)) for n in (4, 5, 7, 9)], [6, 8]),  # starts past 1
        (numbered("1", "2", "3"), []),
        # Numbering that restarts (two sections from 1) is numbered in order.
        (numbered("1", "2", "3", "1", "2"), []),
        (numbered("A", "C"), []),  # no printed numbers
    ],
)
def test_a_gap_is_found_only_in_numbers_as_printed(
    parts: list[Part], missing: list[int]
) -> None:
    spec = Spec(parts=parts)

    assert check(spec, 2) == []
    assert missing_numbers(spec) == missing


def blank_page(spans: list[tuple[str, float, float]]) -> render.Page:
    """A page with no ink and these (text, top, left edge) spans."""
    return render.Page(
        image=Image.new("RGB", (100, 1000), "white"),
        dpi=200,
        native_ppi=None,
        sidebar=None,
        text=[
            render.TextSpan(text, [left, top, left + 0.1, top + 0.01])
            for text, top, left in spans
        ],
    )


def test_a_band_runs_from_its_printed_label_to_the_next() -> None:
    page = blank_page(
        [
            ("12 apples", 0.05, 0.1),
            ("1.5 m", 0.07, 0.1),
            ("1.", 0.10, 0.1),
            ("2.", 0.30, 0.1),
            ("(a)", 0.35, 0.12),
            ("(b)", 0.45, 0.12),
            ("3.", 0.55, 0.1),
            ("(a)", 0.60, 0.12),
            ("5.", 0.85, 0.1),
            ("A crate rests on the floor.", 0.87, 0.15),
            ("A.", 0.90, 0.12),
        ]
    )
    spec = Spec(
        parts=[
            part("Q1", "1", band=[0.0, 0.2]),
            part("Q2.a", "2(a)", band=[0.36, 0.44]),
            part("Q2.b", "2(b)", band=[0.46, 0.54]),
            part("Q3.a", "3(a)", band=[0.58, 0.7]),
            part("Q4", "4", band=[0.8, 0.9]),  # not in the text layer
            part("Q5.a", "5A", band=[0.87, 0.95]),
        ]
    )
    intros = [
        item("2.", None, [0.28, 0.3]),
        item("3.", None, [0.5, 0.55]),
        item("5.", None, [0.85, 0.86]),
    ]

    banded = with_text_bands(spec, [page], [PageParse(items=intros, shared_stems=[])])

    assert [p.band for p in banded.parts] == [
        [0.10, 0.30],
        [0.35, 0.45],
        [0.45, 0.55],
        [0.60, 0.8],
        [0.8, 0.9],
        [0.90, 1.0],  # to the bottom of a page with no ink
    ]


def test_only_a_label_in_the_same_column_ends_a_band() -> None:
    page = blank_page(
        [("1.", 0.10, 0.05), ("2.", 0.50, 0.05), ("3.", 0.12, 0.55), ("4.", 0.52, 0.55)]
    )
    spec = Spec(
        parts=[
            part("Q1", "1", band=[0.1, 0.45]),
            part("Q2", "2", band=[0.5, 0.9]),
            part("Q3", "3", band=[0.12, 0.45]),
            part("Q4", "4", band=[0.52, 0.9]),
        ]
    )

    banded = with_text_bands(spec, [page], [PageParse(items=[], shared_stems=[])])

    assert [p.band for p in banded.parts] == [
        [0.10, 0.50],
        [0.50, 1.0],
        [0.12, 0.52],
        [0.52, 1.0],
    ]


def test_a_label_outside_the_estimated_band_is_not_taken() -> None:
    # The question's own label is not in the text layer, the page number is.
    page = blank_page([("A cart rolls down a ramp.", 0.10, 0.15), ("1", 0.97, 0.5)])
    spec = Spec(parts=[part("Q1", "1", band=[0.08, 0.3])])

    banded = with_text_bands(spec, [page], [PageParse(items=[], shared_stems=[])])

    assert banded.parts[0].band == [0.08, 0.3]


# The labels printed on each page of the fixture, top to bottom, and the label of
# each part as printed on its home page.
LABELS = {
    1: ["1.", "2.", "3."],
    2: ["4.", "(a)", "(b)"],
    3: ["(c)", "5.", "(a)"],
    4: ["(b)", "(c)"],
}
PRINTED = {
    "Q1": "1.",
    "Q2": "2.",
    "Q3": "3.",
    "Q4.a": "(a)",
    "Q4.b": "(b)",
    "Q4.c": "(c)",
    "Q5.a": "(a)",
    "Q5.b": "(b)",
    "Q5.c": "(c)",
}


def test_each_band_of_the_vector_blank_starts_at_its_printed_label() -> None:
    truth = json.loads((FIXTURES / "key.json").read_text())["parts"]
    pdf = render.open_pdf(FIXTURES / "blank.pdf")
    pages = [render.render_page(page, "other") for page in pdf]
    spec = Spec(
        parts=[part(t["id"], t["label"], t["home_page"], band=[0, 1]) for t in truth]
    )
    intros = {2: [item("4.", None, [0.08, 0.3])], 3: [item("5.", None, [0.7, 0.73])]}
    read = [PageParse(items=intros.get(n, []), shared_stems=[]) for n in LABELS]

    banded = with_text_bands(spec, pages, read)

    assert [p.id for p in banded.parts] == list(PRINTED)
    for p in banded.parts:
        text = pages[p.home_page - 1].text
        labels = LABELS[p.home_page]
        [top] = [s.box[1] for s in text if s.text == PRINTED[p.id]]
        after = labels[labels.index(PRINTED[p.id]) + 1 :]
        assert p.band[0] == top
        if after:
            [bottom] = [s.box[1] for s in text if s.text == after[0]]
            assert p.band[1] == bottom
        else:  # the last part on a page runs to its lowest ink, the page number
            assert p.band[1] >= max(s.box[3] for s in text)


class Server:
    """OpenRouter for a three-page blank whose page n prints `PAGE n` and the
    problems `first[n]` on a first reading and `again[n]` on a re-read (the same
    as the first unless given). The stitch makes one part per item, with its home
    page moved down by `shift`."""

    def __init__(
        self,
        first: dict[int, list[str]],
        again: dict[int, list[str]] | None = None,
        shift: int = 0,
    ) -> None:
        self.first, self.again, self.shift = first, again or {}, shift
        self.page_reads: list[tuple[int, list[str]]] = []
        self.stitches = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = [c["text"] for c in body["messages"][1]["content"] if "text" in c]
        if body["response_format"]["json_schema"]["name"] == "PageParse":
            page = int(re.findall(r"PAGE (\d)", texts[0])[0])
            self.page_reads.append((page, texts[1:]))
            printed = self.again.get(page, self.first[page])
            if len(texts) == 1:
                printed = self.first[page]
            return self.reply(
                PageParse(items=[item(x) for x in printed], shared_stems=[])
            )
        self.stitches += 1
        read = json.loads(texts[0])["results"]
        return self.reply(
            Spec(
                parts=[
                    part(f"Q{i['label'][0]}", i["label"][0], r["page"] + self.shift)
                    for r in read
                    for i in r["items"]
                ]
            )
        )

    def reply(self, value: BaseModel) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": MODEL.slug,
                "choices": [{"message": {"content": value.model_dump_json()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )


def item(
    label: str,
    response_type: ResponseType | None = "numeric",
    band: list[float] | None = None,
) -> PageItem:
    return PageItem(
        label=label,
        parent_label="",
        text="Find it.",
        response_type=response_type,
        options=[],
        band=band or [0.1, 0.2],
        figures=[],
        continues_from_previous_page=False,
    )


def three_page_blank(workspace: Workspace) -> None:
    workspace.blank_pdf.parent.mkdir(parents=True)
    pdf = canvas.Canvas(str(workspace.blank_pdf))
    for number in (1, 2, 3):
        pdf.drawString(72, 720, f"PAGE {number}")
        pdf.showPage()
    pdf.save()


async def no_sleep(seconds: float) -> None:
    pass


def run_parse(tmp_path: Path, server: Server) -> Spec:
    workspace = Workspace(tmp_path / "ws")
    three_page_blank(workspace)

    async def main() -> Spec:
        async with Gateway(
            tmp_path / "cache.sqlite",
            "test-key",
            transport=httpx.MockTransport(server),
            sleep=no_sleep,
        ) as gateway:
            return await parse(workspace, gateway, MODEL, 1536)

    return asyncio.run(main())


MISSING_2 = '{"missing_problem_numbers": [2]}'


def test_a_gap_is_read_again_on_its_page_once_and_stitched_again(
    tmp_path: Path,
) -> None:
    server = Server({1: ["1."], 2: [], 3: ["3."]}, again={2: ["2."]})

    spec = run_parse(tmp_path, server)

    assert sorted(server.page_reads) == [(1, []), (2, []), (2, [MISSING_2]), (3, [])]
    assert server.stitches == 2
    assert [(p.id, p.home_page) for p in spec.parts] == [
        ("Q1", 1),
        ("Q2", 2),
        ("Q3", 3),
    ]
    workspace = Workspace(tmp_path / "ws")
    assert all(workspace.key_page(n).exists() for n in (1, 2, 3))
    reread = PageParse.model_validate_json(workspace.key_page_json(2).read_bytes())
    assert [i.label for i in reread.items] == ["2."]
    assert Spec.model_validate_json(workspace.spec_json.read_bytes()) == spec


def test_a_gap_that_remains_after_the_re_read_is_left_for_a_warning(
    tmp_path: Path,
) -> None:
    server = Server({1: ["1."], 2: [], 3: ["3."]})

    spec = run_parse(tmp_path, server)

    assert [n for n, _ in server.page_reads].count(2) == 2
    # The second stitch has the same input as the first, so the cache answers it.
    assert server.stitches == 1
    assert [p.id for p in spec.parts] == ["Q1", "Q3"]
    assert missing_numbers(spec) == [2]
    assert Workspace(tmp_path / "ws").spec_json.exists()


@pytest.mark.parametrize(
    ("reread", "ids", "kept"),
    [
        (["2.", "3."], ["Q1", "Q2", "Q3", "Q4"], ["2.", "3."]),
        # It drops 3., so the first reading of the page stands.
        (["2."], ["Q1", "Q3", "Q4"], ["3."]),
    ],
    ids=["whole page", "drops an item"],
)
def test_a_re_read_that_drops_an_item_is_not_used(
    tmp_path: Path, reread: list[str], ids: list[str], kept: list[str]
) -> None:
    server = Server({1: ["1."], 2: ["3."], 3: ["4."]}, again={2: reread})

    spec = run_parse(tmp_path, server)

    assert (2, [MISSING_2]) in server.page_reads
    assert [p.id for p in spec.parts] == ids
    page_2 = PageParse.model_validate_json(
        Workspace(tmp_path / "ws").key_page_json(2).read_bytes()
    )
    assert [i.label for i in page_2.items] == kept


def test_a_spec_that_fails_is_saved_for_correction_and_raises(tmp_path: Path) -> None:
    # Home pages past the last page, with a gap: the re-read stays on the blank.
    server = Server({1: ["1."], 2: [], 3: ["3."]}, shift=6)

    with pytest.raises(SpecError, match="Q1 has home page 7 of 3") as raised:
        run_parse(tmp_path, server)

    workspace = Workspace(tmp_path / "ws")
    assert (3, [MISSING_2]) in server.page_reads
    assert "correct it by hand" in str(raised.value)
    assert not workspace.spec_json.exists()
    rejected = Spec.model_validate_json(workspace.spec_rejected_json.read_bytes())
    assert [(p.id, p.home_page) for p in rejected.parts] == [("Q1", 7), ("Q3", 9)]
