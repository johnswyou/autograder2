"""Parse the blank assignment into the spec (design 6.1): one call per page, one
call that stitches the pages together, then the checks and the bands in code."""

import asyncio
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from PIL import Image, ImageOps
from pydantic import BaseModel

from redpen import prompts
from redpen.imaging import render
from redpen.llm.client import Gateway, Part
from redpen.llm.models import ModelInfo
from redpen.schemas.common import Box, ResponseType, Schema
from redpen.schemas.spec import Band, Spec
from redpen.workspace import Workspace, write_model

ROLE = "parser"
PARAMS = {"max_tokens": 32000}
CHOICE_TYPES = ("multiple_choice", "multiple_select")


class SpecError(Exception):
    """The spec fails the checks of design 6.1. The message lists every failure."""


# What the per-page call returns. Bands and boxes are fractions of the page image.
# `response_type` is null for an introduction that asks for nothing itself.
class PageItem(Schema):
    label: str
    parent_label: str
    text: str
    response_type: ResponseType | None
    options: list[str]
    band: Band
    figures: list[Box]
    continues_from_previous_page: bool


# A passage printed for several questions, and the labels of those it serves.
class SharedStem(Schema):
    text: str
    governs: list[str]
    figures: list[Box]


class PageParse(Schema):
    items: list[PageItem]
    shared_stems: list[SharedStem]


async def parse(
    workspace: Workspace, gateway: Gateway, model: ModelInfo, image_budget: int
) -> Spec:
    """Parse `workspace.blank_pdf`, write the page images, what was read on each
    page and `spec.json`, and return the spec. Raises `SpecError` if the spec still
    fails its checks after the one re-parse a gap in the numbering allows."""
    pdf = render.open_pdf(workspace.blank_pdf)
    kind = "classkick" if render.classkick_sidebars(pdf) else "other"
    pages = [render.render_page(page, kind) for page in pdf]
    for number, page in enumerate(pages, 1):
        workspace.key_page(number).parent.mkdir(parents=True, exist_ok=True)
        page.image.save(workspace.key_page(number), compress_level=1)
    page_prompt = prompts.load("parse", "page")
    stitch_prompt = prompts.load("parse", "stitch")
    session = f"{workspace.root.name}:parse"

    async def ask[T: BaseModel](
        prompt: tuple[str, str], items: Sequence[Part], response_model: type[T]
    ) -> T:
        system, version = prompt
        result = await gateway.call(
            ROLE,
            model,
            system,
            [],
            items,
            response_model,
            version,
            params=PARAMS,
            session_id=session,
        )
        return result.value

    async def read(number: int, *extra: str) -> PageParse:
        page = pages[number - 1]
        text = ["\n".join(span.text for span in page.text)] if page.text else []
        items = [_png(page.image, image_budget), *text, *extra]
        found = await ask(page_prompt, items, PageParse)
        write_model(workspace.key_page_json(number), found)
        return found

    async def stitch(results: list[PageParse]) -> Spec:
        read_pages = [
            {"page": number, **found.model_dump(mode="json")}
            for number, found in enumerate(results, 1)
        ]
        data = json.dumps({"pages": len(results), "results": read_pages})
        return await ask(stitch_prompt, [data], Spec)

    results = list(await asyncio.gather(*(read(n) for n in range(1, len(pages) + 1))))
    spec = await stitch(results)
    if missing := missing_numbers(spec):
        by_page: defaultdict[int, list[int]] = defaultdict(list)
        for number in missing:
            by_page[page_for_missing(spec, number)].append(number)
        again = await asyncio.gather(
            *(
                read(page, json.dumps({"missing_problem_numbers": numbers}))
                for page, numbers in by_page.items()
            )
        )
        for page, found in zip(by_page, again, strict=True):
            results[page - 1] = found
        spec = await stitch(results)
    if failures := check(spec, len(pages)):
        raise SpecError(
            f"the spec of {workspace.blank_pdf} fails its checks: {'; '.join(failures)}"
        )
    spec = with_text_bands(spec, pages, results)
    write_model(workspace.spec_json, spec)
    return spec


def check(spec: Spec, page_count: int) -> list[str]:
    """Every way `spec` fails the checks of design 6.1, `[]` if it passes."""
    ids = Counter(part.id for part in spec.parts)
    failures = [] if spec.parts else ["no part was found"]
    failures += [f"{id} is used {n} times" for id, n in ids.items() if n > 1]
    for part in spec.parts:
        if part.response_type in CHOICE_TYPES and len(part.options) < 2:
            failures.append(
                f"{part.id} is {part.response_type} with fewer than two options"
            )
        if part.home_page > page_count:
            failures.append(f"{part.id} has home page {part.home_page} of {page_count}")
        for figure in part.figures:
            if figure.page > page_count:
                failures.append(
                    f"{part.id} has a figure on page {figure.page} of {page_count}"
                )
        for other in part.depends_on:
            if other == part.id:
                failures.append(f"{part.id} depends on itself")
            elif other not in ids:
                failures.append(f"{part.id} depends on {other}, which does not exist")
    failures += [f"problem {n} is missing" for n in missing_numbers(spec)]
    return failures


def missing_numbers(spec: Spec) -> list[int]:
    """The problem numbers missing between the lowest and the highest, when every
    part's label starts with its problem's number. `[]` when the labels are not
    numbered that way, as when the ids number unnumbered problems in order."""
    numbers: set[int] = set()
    for part in spec.parts:
        printed = re.match(r"\D*(\d+)", part.label)
        if printed is None or int(printed[1]) != _problem(part.id):
            return []
        numbers.add(_problem(part.id))
    if not numbers:
        return []
    return sorted(set(range(min(numbers), max(numbers) + 1)) - numbers)


def page_for_missing(spec: Spec, number: int) -> int:
    """The page to read again for missing problem `number`: the page after the last
    one of the problem before it, but no later than the first one of the problem
    after it."""
    below = max(_problem(p.id) for p in spec.parts if _problem(p.id) < number)
    above = min(_problem(p.id) for p in spec.parts if _problem(p.id) > number)
    last = max(p.home_page for p in spec.parts if _problem(p.id) == below)
    first = min(p.home_page for p in spec.parts if _problem(p.id) == above)
    return min(last + 1, first)


def with_text_bands(
    spec: Spec, pages: Sequence[render.Page], results: Sequence[PageParse]
) -> Spec:
    """`spec` with the band of each part whose label is found in its home page's
    text layer running from that label to the next label on the page, or to the
    bottom of the printed content. The next label can be a part's or that of an
    introduction read on the page. Other parts keep the model's estimate."""
    parts = list(spec.parts)
    for number, (page, found) in enumerate(zip(pages, results, strict=True), 1):
        if not page.text:
            continue
        here = [i for i, part in enumerate(parts) if part.home_page == number]
        tops = {
            i: _label_top(parts[i].label, parts[i].band[0], page.text) for i in here
        }
        starts = [
            top if top is not None else parts[i].band[0] for i, top in tops.items()
        ]
        for item in found.items:
            if item.response_type is None:
                top = _label_top(item.label, item.band[0], page.text)
                starts += [] if top is None else [top]
        end = _content_bottom(page.image)
        for i, top in tops.items():
            if top is None:
                continue
            bottom = min((s for s in starts if s > top), default=end)
            if top < bottom:  # as a Band must be
                parts[i] = parts[i].model_copy(update={"band": [top, bottom]})
    return Spec(parts=parts)


def _problem(part_id: str) -> int:
    return int(part_id.split(".")[0].removeprefix("Q"))


def _label_top(
    label: str, estimate: float, spans: Sequence[render.TextSpan]
) -> float | None:
    """The top of the span where `label` is printed: a span that starts with the
    label, or with a whole trailing segment of it ("(b)" for "7(b)"). The longest
    match wins, then the one nearest the model's `estimate`."""
    label = " ".join(label.split())
    best: tuple[tuple[int, float], float] | None = None
    for span in spans:
        text = " ".join(span.text.split())
        for start in range(len(label)):
            marker = label[start:]
            if _whole(label, start) and _starts(text, marker):
                rank = (-len(marker), abs(span.box[1] - estimate))
                if best is None or rank < best[0]:
                    best = (rank, span.box[1])
                break
    return best[1] if best else None


def _whole(label: str, start: int) -> bool:
    """Whether `label[start:]` is a whole segment: it holds a letter or digit and
    does not start inside a run of digits or of letters."""
    return any(c.isalnum() for c in label[start:]) and (
        start == 0 or not _same_kind(label[start - 1], label[start])
    )


def _starts(text: str, marker: str) -> bool:
    """Whether `text` starts with `marker` as a whole token, so "1" starts "1." and
    "1 The" but not "12" or "1.5"."""
    rest = text.removeprefix(marker)
    if not text.startswith(marker) or not rest:
        return text == marker
    return not _same_kind(marker[-1], rest[0]) and not (
        marker[-1].isdigit() and re.match(r"[.,]\d", rest)
    )


def _same_kind(a: str, b: str) -> bool:
    return (a.isdigit() and b.isdigit()) or (a.isalpha() and b.isalpha())


def _content_bottom(image: Image.Image) -> float:
    """The bottom of the lowest ink on the page, as a fraction of its height."""
    box = ImageOps.invert(image.convert("L")).getbbox()
    return box[3] / image.height if box else 1.0


def _png(image: Image.Image, budget: int) -> bytes:
    """`image` as PNG, reduced to fit `budget` pixels on its longest edge."""
    small = image.copy()
    small.thumbnail((budget, budget))
    buffer = io.BytesIO()
    small.save(buffer, "PNG")
    return buffer.getvalue()
