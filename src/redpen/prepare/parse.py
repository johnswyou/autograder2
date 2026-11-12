"""Parse the blank assignment into the spec (design 6.1): one call per page, one
call that stitches the pages together, then the checks and the bands in code."""

import asyncio
import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from PIL import Image, ImageOps
from pydantic import BaseModel

from redpen import prompts
from redpen.imaging import render
from redpen.imaging.prep import reduced_png
from redpen.llm.client import Gateway, Part
from redpen.llm.models import ModelInfo
from redpen.schemas.common import Box, ResponseType, Schema
from redpen.schemas.spec import CHOICE_TYPES, Band, Spec
from redpen.workspace import Workspace, write_model

ROLE = "parser"
PARAMS = {"max_tokens": 32000}
MARGIN = 0.03  # of the page height, around the model's band, where a label may be
COLUMN = 0.25  # of the page width: a label further left or right is another column


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
    page and `spec.json`, and return the spec. A gap in the numbering gets one
    re-read of its page and one more stitch; a gap that remains is left for the
    caller to warn about. A spec that fails any other check is written to
    `spec.rejected.json` and raises `SpecError`."""
    pages = render_blank(workspace)
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
        items = [reduced_png(page.image, image_budget), *text, *extra]
        return await ask(page_prompt, items, PageParse)

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
            by_page[page_for_missing(spec, number, len(pages))].append(number)
        again = await asyncio.gather(
            *(
                read(page, json.dumps({"missing_problem_numbers": numbers}))
                for page, numbers in by_page.items()
            )
        )
        for page, found in zip(by_page, again, strict=True):
            if keeps_everything(results[page - 1], found):
                results[page - 1] = found
        spec = await stitch(results)
    for number, found in enumerate(results, 1):
        write_model(workspace.key_page_json(number), found)
    if failures := check(spec, len(pages)):
        write_model(workspace.spec_rejected_json, spec)
        raise SpecError(
            f"the spec fails its checks: {'; '.join(failures)}. It is saved as"
            f" {workspace.spec_rejected_json}; correct it by hand and save it as"
            f" {workspace.spec_json} to go on."
        )
    spec = with_text_bands(spec, pages, results)
    write_model(workspace.spec_json, spec)
    return spec


def render_blank(workspace: Workspace) -> list[render.Page]:
    """Every page of `workspace.blank_pdf`, each image also saved as its
    `key_page`."""
    with render.open_pdf(workspace.blank_pdf) as pdf:
        kind = "classkick" if render.classkick_sidebars(pdf) else "other"
        pages = [render.render_page(page, kind) for page in pdf]
    for number, page in enumerate(pages, 1):
        workspace.key_page(number).parent.mkdir(parents=True, exist_ok=True)
        page.image.save(workspace.key_page(number), compress_level=1)
    return pages


def check(spec: Spec, page_count: int) -> list[str]:
    """Every way `spec` fails the checks of design 6.1, `[]` if it passes. A gap in
    the numbering is not a failure here: an assignment may skip a number, so it is
    a warning (`missing_numbers`)."""
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
    return failures


def missing_numbers(spec: Spec) -> list[int]:
    """The problem numbers missing between the lowest and the highest, when every
    part's label starts with its problem's number. `[]` when the labels are not
    numbered that way, as when the ids number problems in order because they carry
    no usable numbers or their numbers repeat."""
    numbers: set[int] = set()
    for part in spec.parts:
        printed = re.match(r"\D*(\d+)", part.label)
        if printed is None or int(printed[1]) != _problem(part.id):
            return []
        numbers.add(_problem(part.id))
    if not numbers:
        return []
    return sorted(set(range(min(numbers), max(numbers) + 1)) - numbers)


def page_for_missing(spec: Spec, number: int, page_count: int) -> int:
    """The page to read again for missing problem `number`: the page after the last
    one of the problem before it, but no later than the first one of the problem
    after it, nor than the last page."""
    below = max(_problem(p.id) for p in spec.parts if _problem(p.id) < number)
    above = min(_problem(p.id) for p in spec.parts if _problem(p.id) > number)
    last = max(p.home_page for p in spec.parts if _problem(p.id) == below)
    first = min(p.home_page for p in spec.parts if _problem(p.id) == above)
    return min(last + 1, first, page_count)


def keeps_everything(first: PageParse, again: PageParse) -> bool:
    """Whether a re-read of a page holds every item (by label and parent label) and
    as many shared stems as the first reading. A re-read that drops anything is
    not used, so that it cannot lose what the first reading found."""

    def labels(found: PageParse) -> Counter[tuple[str, str]]:
        return Counter((item.label, item.parent_label) for item in found.items)

    return not labels(first) - labels(again) and len(again.shared_stems) >= len(
        first.shared_stems
    )


def with_text_bands(
    spec: Spec, pages: Sequence[render.Page], results: Sequence[PageParse]
) -> Spec:
    """`spec` with the band of each part whose label is found in its home page's
    text layer running from that label to the next label below it in the same
    column, or to the bottom of the printed content. The next label can be a
    part's or that of an introduction read on the page. A label is looked for only
    inside the model's estimated band, widened by `MARGIN`. Other parts keep the
    model's estimate."""
    parts = list(spec.parts)
    for number, (page, found) in enumerate(zip(pages, results, strict=True), 1):
        if not page.text:
            continue
        here = [i for i, part in enumerate(parts) if part.home_page == number]
        spans = {i: _label_span(parts[i].label, parts[i].band, page.text) for i in here}
        # (top, left edge or None where only the model's estimate is known)
        starts = [
            (span.box[1], span.box[0]) if span else (parts[i].band[0], None)
            for i, span in spans.items()
        ]
        for item in found.items:
            if item.response_type is None:
                if span := _label_span(item.label, item.band, page.text):
                    starts.append((span.box[1], span.box[0]))
        end = _content_bottom(page.image)
        for i, span in spans.items():
            if span is None:
                continue
            top, left = span.box[1], span.box[0]
            below = [
                t
                for t, x in starts
                if t > top and (x is None or abs(x - left) <= COLUMN)
            ]
            bottom = min(below, default=end)
            if top < bottom:  # as a Band must be
                parts[i] = parts[i].model_copy(update={"band": [top, bottom]})
    return Spec(parts=parts)


def _problem(part_id: str) -> int:
    return int(part_id.split(".")[0].removeprefix("Q"))


def _label_span(
    label: str, estimate: Sequence[float], spans: Sequence[render.TextSpan]
) -> render.TextSpan | None:
    """The span where `label` is printed, looked for only inside the `estimate`
    band widened by `MARGIN`: a span that starts with the label, or with a whole
    trailing segment of it ("(b)" for "7(b)", "A" for "12A"). The longest match
    wins, then one set off by punctuation ("A." over the word "A" of "A ball"),
    then the one nearest the top of `estimate`."""
    label = " ".join(label.split())
    low, high = estimate[0] - MARGIN, estimate[1] + MARGIN
    best: tuple[tuple[int, bool, float], render.TextSpan] | None = None
    for span in spans:
        if not low <= span.box[1] <= high:
            continue
        text = " ".join(span.text.split())
        for start in range(len(label)):
            marker = label[start:]
            if _whole(label, start) and _starts(text, marker):
                after = text[len(marker) : len(marker) + 1]
                bare = marker[-1].isalnum() and after in ("", " ")
                rank = (-len(marker), bare, abs(span.box[1] - estimate[0]))
                if best is None or rank < best[0]:
                    best = (rank, span)
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
