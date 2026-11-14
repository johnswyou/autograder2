"""Work that no part claims (design 7.3 and 13), found in code with no fingerprint
and no model call. On a slide export every student's page k is the same slide, so
the class itself gives each page's background, and a student's ink is where their
page is darker than it. Ink that no region of the student's map claims (an entry, a
set-aside block or a name, each grown by the read stage's margin) raises
`unclaimed_work` on the student, naming the page, where it is dense; so does a
block the mapper set aside while a part has no response. Nothing here changes a
score, a map or a reading, or sends a request.

`check_class` runs in `grade` and `run` once the class is mapped. Beside each
student's map it stores `coverage.json`, tied to the map by its digest, and the
images the review list links: each patch of leftover ink and each set-aside block,
cut from the page. `unclaimed_flags`, `unclaimed_lines` and `set_aside_section`
give the report what those files hold, in fixed words."""

import contextlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageChops
from pydantic import ValidationError

from redpen.grade.read import MARGIN, WHITE_LEVEL, digest
from redpen.imaging import prep
from redpen.schemas.flags import Flag, FlagCode
from redpen.schemas.submission import (
    Coverage,
    CoveragePage,
    IngestRecord,
    MapRecord,
    SetAside,
)
from redpen.workspace import Workspace, log_failure, read_model, write_model

# The thresholds, measured on the fixture classes (task 4.3 report). A channel
# darker than the background's by more than INK is ink. At 20 the check finds 88 of
# 100 responses left out of the fixtures' maps, against 84 at 40, with no false
# flag; but the fixtures are drawn and lossless, and 40 leaves twice the room for
# the antialiasing and compression of real exports.
INK = 40
# A page with ink on more than NEARLY of its background's paper (WHITE_LEVEL or more
# in every channel) differs from the class's over much of it, and is not compared:
# another rendering, or a large shaded area. A photo pasted on a fixture slide
# covers 0.29 to 0.54 of it, the most ink a student added 0.008.
NEARLY = 0.1
# A page lighter than the background by more than INK on more than LIGHTER of the
# print, where every other comparable page is dark (a channel under PRINT), is
# another rendering too: shifted by a pixel or resampled, its print edges would show
# as ink. A fixture page reaches 0.059 of it, a page among others who wrote at the
# same place 0.079, a slide shifted by one pixel 0.163 and more (task 4.3 fix 1).
LIGHTER = 0.12
PRINT = 128
# Leftover ink is judged in square cells of CELL pixels; a cell with DENSE of it ink
# raises the flag. A stroke w pixels wide fills at most 1.41 w / CELL of a cell, so
# no lone stroke up to 3 px wide does.
CELL = 40
DENSE = 0.125
FEWEST = 3  # comparable students a page needs
# The background is a pixel's k-th lightest value, k = min(LIGHTEST, n - 2): two
# students whose pages are lighter there (white over the print) cannot move it.
LIGHTEST = 3
# Dense cells no more than REACH cells apart are one patch: a gap of one cell, as
# between the lines and words of a response.
REACH = 2
NAMED = 0.5  # a block set aside at least this much inside a name region is that name

type Reason = Literal[
    "few comparable", "not classkick", "other rendering", "unreadable", "check failed"
]

# The flags' details and the report's lines, in fixed words.
LEFTOVER = (
    "Ink on page {page} lies outside every region the map gives a part, a set-aside"
    " block or a name: {patches}."
)
ASIDE = (
    "The mapper set a block on page {page} aside while a part has no response, so the"
    " block may be that response. It is listed under Blocks set aside."
)
ASIDE_INTRO = (
    "Each block of work the mapper set aside as no part's response, with the"
    " mapper's reason."
)
NO_IMAGE = "(no image: the check made none for this map)"
COMPARED = (
    "Student pages compared with their class's own background: {compared} of"
    " {pages}. Students with dense ink that no region of their map claims"
    " (`unclaimed_work`): {flagged}."
)
NOT_COMPARED = (
    "Student pages not compared: {count}. On a page with fewer than three comparable"
    " students, {few}; not a Classkick page, {other}; of another size than the"
    " class's page, or differing from it over much of its paper or its print,"
    " {rendering}; an image that cannot be read at the size ingest recorded,"
    " {unreadable}; the check failed, its traceback in {log}, {failed};"
    " not checked, with no map or a map changed since the check, {unchecked}."
)
BLOCKS = (
    "Blocks the mapper set aside, each listed in {review}: {blocks}. Students with"
    " one while a part has no response (`unclaimed_work`): {asking}."
)


def check_class(
    workspace: Workspace,
    students: Sequence[IngestRecord],
    done: Callable[[], object] = lambda: None,
) -> None:
    """Compare each page of the class with its background, one page image held at a
    time, calling `done` as each page number is done, and store each mapped
    student's coverage file and images, the earlier ones removed first. A student
    with no map gets none but counts for the background. A page whose image cannot
    be read is not compared, that page alone. When the check itself fails, the
    failure goes to the run log and each student's file says so of every page, with
    nothing else of the check kept."""
    maps: dict[str, MapRecord] = {}
    for student in students:
        _clear(workspace, student.student_key)
        with contextlib.suppress(OSError, ValidationError):  # the map stage failed
            maps[student.student_key] = read_model(
                workspace.map_json(student.student_key), MapRecord
            )
    try:
        found = _compare(workspace, students, maps, done)
        for key, mapped in maps.items():
            for n, block in _blocks(mapped):
                page = _open(workspace.student_page(key, block.page))
                if page is not None:  # else no image, and the list says so
                    _cut(
                        page,
                        block.box,
                        _turns(mapped, block.page),
                        workspace.aside_png(key, n),
                    )
    except Exception as error:  # a fault of the check: nothing of it is kept
        log_failure(workspace, "coverage", error)
        for student in students:
            _clear(workspace, student.student_key)
        found = {
            s.student_key: [
                CoveragePage(number=p.number, not_compared="check failed", patches=[])
                for p in s.pages
            ]
            for s in students
        }
    for key, mapped in maps.items():
        record = Coverage(map_digest=digest(mapped), pages=found[key])
        write_model(workspace.coverage_json(key), record)


def _clear(workspace: Workspace, student: str) -> None:
    workspace.coverage_json(student).unlink(missing_ok=True)
    for old in workspace.aside_png(student, 1).parent.glob("*.png"):
        old.unlink()


def _compare(
    workspace: Workspace,
    students: Sequence[IngestRecord],
    maps: Mapping[str, MapRecord],
    done: Callable[[], object],
) -> dict[str, list[CoveragePage]]:
    """Each mapped student's pages, page number by page number: why each was not
    compared, or the patches of ink its map leaves, each patch's image saved."""
    found: dict[str, list[CoveragePage]] = {key: [] for key in maps}
    for number in sorted({p.number for s in students for p in s.pages}):
        here = [(s, p) for s in students for p in s.pages if p.number == number]
        slides = [
            (s.student_key, (p.width_px, p.height_px))
            for s, p in here
            if s.source_kind == "classkick"
        ]
        # The size most of the class's slides have; another is another rendering.
        common = (
            Counter(size for _, size in slides).most_common(1)[0][0]
            if slides
            else (0, 0)
        )
        reasons: dict[str, Reason] = {
            s.student_key: "not classkick"
            for s, _ in here
            if s.source_kind != "classkick"
        }
        reasons |= {k: "other rendering" for k, size in slides if size != common}
        paths = {
            k: workspace.student_page(k, number) for k, _ in slides if k not in reasons
        }
        background, why = _background(paths, common)
        reasons |= why
        for student, _ in here:
            key = student.student_key
            if (mapped := maps.get(key)) is None:
                continue
            patches: list[list[float]] = []
            if background is not None and key not in reasons:
                with Image.open(paths[key]) as page:
                    patches = _leftover(page, background, mapped, number)
                    for n, patch in enumerate(patches, 1):
                        _cut(
                            page,
                            patch,
                            _turns(mapped, number),
                            workspace.patch_png(key, number, n),
                        )
            found[key].append(
                CoveragePage(
                    number=number, not_compared=reasons.get(key), patches=patches
                )
            )
        done()
    return found


def _background(
    paths: Mapping[str, Path], size: tuple[int, int]
) -> tuple[Image.Image | None, dict[str, Reason]]:
    """The background of one page from the students' images of it at `paths`, all
    recorded as `size`, and why each student not compared was not. Each image is
    read three times, one at a time: for a first background of those that can be
    read at that size; to keep those with little ink on its paper, whose background
    is the one returned; and to leave out those lighter than it on much of the
    print (the rest read once more when one is). Another rendering (a photo, a
    scan), like a large shaded area, is darker than much of the slide's paper; one
    shifted by a pixel is lighter along every printed line. None when fewer than
    FEWEST are comparable."""
    few: Reason = "few comparable"
    why: dict[str, Reason] = {}

    def readable() -> Iterator[Image.Image]:
        for key, path in paths.items():
            if (page := _open(path)) is None or page.size != size:
                why[key] = "unreadable"
            else:
                yield page

    top = _lightest(readable())
    read = [k for k in paths if k not in why]
    if len(read) < FEWEST:
        return None, why | dict.fromkeys(read, few)
    first = _kth(top, len(read))
    paper = _paper(first)
    kept: list[str] = []

    def comparable() -> Iterator[Image.Image]:
        for key in read:
            page = _load(paths[key])
            if _share(_ink(page, first), paper) <= NEARLY:
                kept.append(key)
                yield page

    top = _lightest(comparable())
    why |= {k: "other rendering" for k in read if k not in kept}
    # The second lightest is dark where every page but one is: there, the print.
    if len(kept) >= FEWEST and (printed := _print(top[1])).getbbox():
        background = _kth(top, len(kept))
        lighter = {  # where the background is darker than the page
            k
            for k in kept
            if _share(_ink(background, _load(paths[k])), printed) > LIGHTER
        }
        if lighter:
            why |= dict.fromkeys(lighter, "other rendering")
            kept = [k for k in kept if k not in lighter]
            top = _lightest(_load(paths[k]) for k in kept)
    if len(kept) < FEWEST:
        return None, why | dict.fromkeys(kept, few)
    return _kth(top, len(kept)), why


def _lightest(pages: Iterable[Image.Image]) -> list[Image.Image]:
    """Per pixel and channel, the LIGHTEST lightest values of `pages`, the lightest
    first: that many images kept besides the page read."""
    top: list[Image.Image] = []
    for page in pages:
        for i in range(len(top)):  # no name keeps a page alive past its turn
            top[i], page = (
                ImageChops.lighter(top[i], page),
                ImageChops.darker(top[i], page),
            )
        if len(top) < LIGHTEST:
            top.append(page)
    return top


def _kth(top: Sequence[Image.Image], count: int) -> Image.Image:
    """Of the lightest values `top` of `count` pages (FEWEST or more), the k-th, k =
    min(LIGHTEST, count - 2): ink darkens a page, so any number of students writing
    at a pixel leave it while two do not, and from five students on, two whose
    pages are lighter there (white over the print) cannot move it either."""
    return top[min(LIGHTEST, count - 2) - 1]


def _ink(page: Image.Image, background: Image.Image) -> Image.Image:
    """Where a channel of `page` is darker than `background`'s by more than INK."""
    r, g, b = ImageChops.subtract(background, page).split()
    most = ImageChops.lighter(ImageChops.lighter(r, g), b)
    return _over(most, INK)


def _paper(background: Image.Image) -> Image.Image:
    """Where `background` is blank paper or canvas."""
    return _over(_least(background), WHITE_LEVEL - 1)


def _print(image: Image.Image) -> Image.Image:
    """Where a channel of `image` is under PRINT."""
    return _over(ImageChops.invert(_least(image)), 255 - PRINT)


def _least(image: Image.Image) -> Image.Image:
    """Each pixel's darkest channel."""
    r, g, b = image.split()
    return ImageChops.darker(ImageChops.darker(r, g), b)


def _over(band: Image.Image, level: int) -> Image.Image:
    """255 where `band` is over `level`, else 0."""
    table = [255 if v > level else 0 for v in range(256)]
    return band.point(table)  # pyright: ignore[reportUnknownMemberType] (numpy's half)


def _share(ink: Image.Image, paper: Image.Image) -> float:
    """The share of `paper` under `ink`; all of it when there is no paper."""
    total = paper.histogram()[255]
    return ImageChops.multiply(ink, paper).histogram()[255] / total if total else 1.0


def _leftover(
    page: Image.Image, background: Image.Image, mapped: MapRecord, number: int
) -> list[list[float]]:
    """The patches of `page`'s ink, page `number`, outside every region `mapped`
    gives it, each region grown by the read stage's margin as the reader's crop is."""
    ink = _ink(page, background)
    regions = [e.box for e in mapped.map.entries if e.page == number]
    regions += [a.box for a in mapped.map.set_aside if a.page == number]
    for facts in mapped.pages:
        if facts.number == number:
            regions += [*facts.names, *facts.names_on_responses]
    for box in regions:
        ink.paste(0, prep.pixels(ink.size, box, MARGIN))
    return _dense(ink)


def _dense(ink: Image.Image) -> list[list[float]]:
    """The box of each group of dense cells of `ink`, the topmost first: cells a
    gap of one cell apart are one group, as the lines and words of a response are,
    and the cells touching it that hold any ink, where its sparse edges fall, are
    in its box."""
    cells = ink.reduce(CELL)
    columns = cells.width
    values = {(i % columns, i // columns): v for i, v in enumerate(cells.tobytes())}
    dense = {cell for cell, value in values.items() if value >= DENSE * 255}
    # The same on cells half a cell across and down, so that a response across the
    # corner of four cells is found: each dense one there makes the four it overlaps
    # that hold ink dense.
    half = ink.crop((CELL // 2, CELL // 2, *ink.size)).reduce(CELL)
    for i, value in enumerate(half.tobytes()):
        if value >= DENSE * 255:
            x, y = i % half.width, i // half.width
            four = {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}
            dense |= {cell for cell in four if values.get(cell)}
    width, height = ink.size
    boxes: list[list[float]] = []
    while dense:
        group = [min(dense, key=lambda cell: (cell[1], cell[0]))]
        dense.discard(group[0])
        for cell in group:  # grows as it goes
            near = _around(cell, REACH) & dense
            dense -= near
            group += sorted(near)
        inky = {c for cell in group for c in _around(cell, 1) if values.get(c)}
        xs, ys = [x for x, _ in inky], [y for _, y in inky]
        boxes.append(
            [
                min(xs) * CELL / width,
                min(ys) * CELL / height,
                min((max(xs) + 1) * CELL, width) / width,
                min((max(ys) + 1) * CELL, height) / height,
            ]
        )
    return boxes


def _around(cell: tuple[int, int], reach: int) -> set[tuple[int, int]]:
    """The cells no more than `reach` cells from `cell` across or down."""
    x, y = cell
    steps = range(-reach, reach + 1)
    return {(x + dx, y + dy) for dx in steps for dy in steps}


def _cut(page: Image.Image, box: Sequence[float], turns: int, path: Path) -> None:
    """Save the part of `page` in `box`, grown by the read margin, turned upright."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prep.rotate(prep.crop(page, box, MARGIN)[0], turns).save(path)


def _turns(mapped: MapRecord, number: int) -> int:
    return next((p.rotation // 90 for p in mapped.pages if p.number == number), 0)


def _load(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
    return image


def _open(path: Path) -> Image.Image | None:
    """The image at `path`, or None when it cannot be read."""
    try:
        return _load(path)
    except (OSError, SyntaxError):  # what Pillow raises for a file it cannot read
        return None


def _blocks(mapped: MapRecord) -> list[tuple[int, SetAside]]:
    """Each block the map set aside, numbered from 1, but one that lies mostly in a
    name region of its page (names masked): that is not listed again."""
    names = {p.number: [*p.names, *p.names_on_responses] for p in mapped.pages}
    return [
        (n, block)
        for n, block in enumerate(mapped.map.set_aside, 1)
        if not any(
            _inside(block.box, name) >= NAMED for name in names.get(block.page, [])
        )
    ]


def _inside(box: Sequence[float], frame: Sequence[float]) -> float:
    """The share of `box`'s area inside `frame`."""
    width = min(box[2], frame[2]) - max(box[0], frame[0])
    height = min(box[3], frame[3]) - max(box[1], frame[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max(width, 0) * max(height, 0) / area


# --- what the report shows ------------------------------------------------------------


@dataclass(frozen=True)
class _Stored:
    student: IngestRecord
    mapped: MapRecord | None
    coverage: Coverage | None  # only one made from `mapped`


def _stored(workspace: Workspace, students: Sequence[IngestRecord]) -> list[_Stored]:
    found: list[_Stored] = []
    for student in students:
        mapped = coverage = None
        with contextlib.suppress(OSError, ValidationError):
            mapped = read_model(workspace.map_json(student.student_key), MapRecord)
            kept = read_model(workspace.coverage_json(student.student_key), Coverage)
            coverage = kept if kept.map_digest == digest(mapped) else None
        found.append(_Stored(student, mapped, coverage))
    return found


def _link(workspace: Workspace, path: Path) -> str:
    return path.relative_to(workspace.review_md.parent, walk_up=True).as_posix()


def _images(workspace: Workspace, found: _Stored) -> list[tuple[str, Path]]:
    """Each image the review list links for `found`'s student, labelled: each patch
    of leftover ink, page by page, then each block set aside; none while the
    check's file is not of the map."""
    if not (found.coverage and found.mapped):
        return []
    key = found.student.student_key
    patches = [
        (f"patch {n}, page {page.number}", workspace.patch_png(key, page.number, n))
        for page in found.coverage.pages
        for n in range(1, len(page.patches) + 1)
    ]
    blocks = [
        (f"block {n}, page {block.page}", image)
        for n, block in _blocks(found.mapped)
        if (image := _aside(workspace, found, n))
    ]
    return patches + blocks


def _aside(workspace: Workspace, found: _Stored, n: int) -> Path | None:
    """The image of `found`'s block `n` set aside, which the list links: none while
    the check's file is not of the map, or when the check could not cut it."""
    image = workspace.aside_png(found.student.student_key, n)
    return image if found.coverage and image.is_file() else None


def unclaimed_shown(
    workspace: Workspace, student: IngestRecord
) -> tuple[MapRecord | None, list[tuple[str, Path]]]:
    """What `student`'s whole-submission entry shows besides its flags: the map the
    check ran on, and each image the entry or the list of blocks set aside links,
    labelled."""
    [found] = _stored(workspace, [student])
    return found.mapped, _images(workspace, found)


def unclaimed_flags(
    workspace: Workspace, students: Sequence[IngestRecord]
) -> list[Flag]:
    """`unclaimed_work` on each student, for each page whose stored check found
    leftover ink, each patch linked from `out/`; then, while a part has no
    response, for each page of a block set aside."""
    raised: list[Flag] = []
    for found in _stored(workspace, students):
        key = found.student.student_key
        details: list[str] = []
        for page in found.coverage.pages if found.coverage else []:
            images = [
                workspace.patch_png(key, page.number, n)
                for n in range(1, len(page.patches) + 1)
            ]
            links = [
                f"[patch {n}]({_link(workspace, p)})" for n, p in enumerate(images, 1)
            ]
            if links:
                details.append(
                    LEFTOVER.format(page=page.number, patches=", ".join(links))
                )
        if found.mapped and found.mapped.map.unanswered:
            pages = sorted({b.page for _, b in _blocks(found.mapped)})
            details += [ASIDE.format(page=n) for n in pages]
        raised += [
            Flag(code=FlagCode.UNCLAIMED_WORK, part_id=None, student_key=key, detail=d)
            for d in details
        ]
    return raised


def unclaimed_lines(
    workspace: Workspace, students: Sequence[IngestRecord]
) -> list[str]:
    """The run report's section: the pages compared and the students with leftover
    ink, the pages not compared and why, and the blocks set aside."""
    found = _stored(workspace, students)
    pages = sum(len(s.pages) for s in students)
    why = Counter(p.not_compared for f in found if f.coverage for p in f.coverage.pages)
    flagged = [
        f for f in found if f.coverage and any(p.patches for p in f.coverage.pages)
    ]
    blocks = [(f, len(_blocks(f.mapped))) for f in found if f.mapped]
    asking = [f for f, n in blocks if n and f.mapped and f.mapped.map.unanswered]
    review = workspace.review_md.relative_to(workspace.root).as_posix()
    log = workspace.run_log.relative_to(workspace.root).as_posix()
    return [
        "",
        "## Work no part claims",
        "",
        COMPARED.format(compared=why[None], pages=pages, flagged=len(flagged)),
        "",
        NOT_COMPARED.format(
            count=pages - why[None],
            few=why["few comparable"],
            other=why["not classkick"],
            rendering=why["other rendering"],
            unreadable=why["unreadable"],
            log=log,
            failed=why["check failed"],
            unchecked=pages - why.total(),
        ),
        "",
        BLOCKS.format(
            review=review, blocks=sum(n for _, n in blocks), asking=len(asking)
        ),
    ]


def set_aside_section(workspace: Workspace, students: Sequence[IngestRecord]) -> str:
    """The review list's last section: each block set aside, but a name region, one
    line each with its student, page, the mapper's reason and its image; or nothing
    when there is none. The name and the reason are untrusted, in code spans."""
    from redpen.report.report import code_span  # report.report imports this module

    lines: list[str] = []
    for found in _stored(workspace, students):
        if found.mapped is None:
            continue
        key = found.student.student_key
        name = f"{code_span(found.student.student_name)} (`{key}`)"
        for n, block in _blocks(found.mapped):
            image = _aside(workspace, found, n)
            shown = f"[image]({_link(workspace, image)})" if image else NO_IMAGE
            lines.append(
                f"- {name}, page {block.page}: {code_span(block.reason)} {shown}."
            )
    if not lines:
        return ""
    return "\n".join(["", "## Blocks set aside", "", ASIDE_INTRO, "", *lines]) + "\n"
